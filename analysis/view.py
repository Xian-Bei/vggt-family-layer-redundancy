"""
Interactive browser results panel.

This app reuses Stage1 caches and the pair-attention reconstruction helpers from
``visualize_pair_attention.py``. Users can choose a cached scene, a global
layer, two selected frames, score space, and head view; the panel updates
in-browser with bidirectional overlay summaries and Stage2 saved heatmaps.

- frame i as q, frame j as k
- frame j as q, frame i as k
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import threading

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard for optional demo dependency
    raise SystemExit(
        "Missing optional dependency 'gradio'. Install demo dependencies first, "
        "for example: `pip install -r requirements_demo.txt` or `pip install .[demo]`."
    ) from exc

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from .redundancy.metrics import upper_triangle_pack, upper_triangle_unpack_for_heatmap
from analysis.prune_plan.build_prune_plan_common import load_similarity
from analysis.redundancy.similarity_v3 import (
    REDUNDANCY_GRAPH_REGIONS,
)
from analysis.triangular_heatmap import configure_triangle_axes, imshow_upper_triangle, triangle_xy
from analysis.eval.scannet_eval import SceneData, load_scene_images
from .redundancy.cache_layout import count_scene_replay_layers_disk, load_scene_metadata, scene_cache_dir
from .redundancy.visualize_pair_attention import (
    _build_global_positions,
    _load_cache_row,
    _overlay_patch_map,
    _reconstruct_qk,
)
from .vis import build_cka_heatmaps_tab, build_eval_tab, build_heatmaps_tab, build_recovery_tab, build_summary_tab
# from .vis import build_pair_tab  # Pair Attention tab disabled

OUTPUT_ROOT_DEFAULT = Path(os.environ.get("OUTPUT_ROOT", "analysis_outputs_v3"))


def _repo_root() -> Path:
    """Project root (parent of the ``analysis`` package). ``view.py`` lives in ``analysis/``."""
    return Path(__file__).resolve().parent.parent


def _resolve_path_cwd_then_repo(path: Path) -> Path:
    """Resolve a CLI path that may be relative to cwd or to the repo root."""
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (_repo_root() / path).resolve()


def _gradio_allowed_paths(controller: PairAttentionController) -> List[str]:
    """Directories Gradio may serve when ``Image`` uses ``type='filepath'`` (Gradio 5)."""
    roots: List[str] = [str(_repo_root())]
    for raw in (
        controller.lora_root,
        controller.eval_root,
        controller.cache_root,
        controller.analysis_similarity_root,
        controller.analysis_skip_root,
    ):
        if raw is None:
            continue
        try:
            roots.append(str(Path(raw).resolve()))
        except OSError:
            continue
    return sorted(set(roots))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive browser panel for v3 results.")
    # v3 datasets live under analysis_outputs_v3/<backbone>/<dataset_tag>/...
    # Keep this free-form so new tags like scannet_dev50/scannet_test50 work.
    p.add_argument("--dataset", type=str, default="scannet_dev50")
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="v3 output root, usually analysis_outputs_v3 or $OUTPUT_ROOT.",
    )
    p.add_argument("--scannet-root", type=Path, default=Path("data/scans_dev50/"))
    p.add_argument("--scannet-test-root", type=Path, default=Path("data/scans_test50/"))
    p.add_argument("--root-7scenes", type=Path, default=Path("data/7scenes"))
    p.add_argument("--root-nrgbd", type=Path, default=Path("data/neural_rgbd_data"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--server-name", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7866)
    p.add_argument("--viser-host", type=str, default="127.0.0.1", help="Host for embedded Viser server.")
    p.add_argument("--viser-port", type=int, default=8090, help="Port for embedded Viser server.")
    p.add_argument(
        "--v3-backbone",
        type=str,
        default="",
        help="Initial backbone under analysis_outputs_v3/ (default: first discovered).",
    )
    return p


@dataclass
class SceneCache:
    item: Dict[str, Any]
    row: Dict[str, Any]
    scene_eff: SceneData
    images: torch.Tensor
    frame_ids: List[str]
    image_hw: Tuple[int, int]
    patch_start_idx: int
    replay_layers: List[torch.Tensor]
    patch_hw: Tuple[int, int]
    tokens_per_frame: int
    frame_labels: List[str]
    frame_label_to_index: Dict[str, int]


@dataclass
class Stage2SceneReport:
    scene_id: str
    report: Dict[str, Any]


SIMILARITY_METRICS: List[Tuple[str, str]] = [
    ("cka", "CKA"),
    ("cosine", "cosine"),
    ("pwcca", "PWCCA"),
    ("neg_mse", "-MSE"),
]

SKIP_MAIN_METRICS: List[Tuple[str, str]] = [
    ("are", "ARE"),
    ("ape", "APE"),
    ("rpe_rot", "RPE_rot"),
    ("rpe_trans", "RPE_trans"),
]

SKIP_DEPTH_METRICS: List[Tuple[str, str]] = [
    ("depth_abs_rel", "Depth abs rel"),
    ("depth_delta1", "Depth delta1"),
]

SKIP_POINT_METRICS: List[Tuple[str, str]] = [
    ("acc", "Accuracy"),
    ("comp", "Completeness"),
    ("chamfer_distance", "Chamfer"),
    ("nc", "Normal consistency"),
]

SKIP_HIGHER_IS_BETTER = {"depth_delta1", "nc"}


class PairAttentionController:
    def __init__(
        self,
        output_root: Path,
        dataset: str,
        data_roots_by_dataset: Dict[str, Path],
        device_arg: str,
        *,
        viser_host: str = "127.0.0.1",
        viser_port: int = 8090,
        v3_backbone: str = "",
    ) -> None:
        from analysis.vis.view_v3_loader import discover_v3_backbones

        self.v3_output_root = _resolve_path_cwd_then_repo(Path(output_root))
        self.v3_mode = True
        self.v3_backbones = discover_v3_backbones(self.v3_output_root)
        preferred = str(v3_backbone).strip()
        if preferred and preferred in self.v3_backbones:
            self.v3_backbone = preferred
        elif self.v3_backbones:
            self.v3_backbone = self.v3_backbones[0]
        else:
            self.v3_backbone = preferred or "vggt"
        self.lora_root = self.v3_output_root
        self.eval_root: Optional[Path] = None
        self.viser_host = str(viser_host)
        self.viser_port = int(viser_port)
        self.cache_root = self.v3_backbone_root()
        self.analysis_similarity_root = self.v3_backbone_root()
        self.analysis_skip_root = self.v3_backbone_root()
        self.data_root = Path(data_roots_by_dataset.get(str(dataset).strip().lower(), Path(".")))
        self.data_roots_by_dataset = dict(data_roots_by_dataset)
        self.device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
        self.dataset = str(dataset).strip().lower()
        self._apply_v3_backbone_paths()
        self.cache_root_exists = self.cache_root.exists() and self.cache_root.is_dir()

        self.cache_dir = self._resolve_dataset_dir(self.cache_root, self.dataset)
        self.analysis_similarity_dir = self._resolve_dataset_dir(self.analysis_similarity_root, self.dataset)
        self.analysis_skip_dir = self._resolve_dataset_dir(self.analysis_skip_root, self.dataset)
        self.data_root = self.data_roots_by_dataset.get(self.dataset, self.data_root)

        self.manifest, self.scene_items = self._load_manifest_and_scenes()
        self.cache_available = self._cache_available_on_disk()

        self.hf_model = str(self.manifest.get("hf_model", "")).strip()
        # Model is required only for Pair Attention reconstruction; Stage2/LoRA/Eval tabs can work without it.
        self.pair_available = self.cache_available and bool(self.scene_items) and bool(self.hf_model)

        self.model: Optional[Any] = None
        self.scene_cache: Dict[str, SceneCache] = {}
        self.qk_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._v3_similarity_report_cache: Dict[Tuple[str, str], Dict[str, Stage2SceneReport]] = {}
        self._v3_skip_report_cache: Dict[Tuple[str, str], Dict[str, Stage2SceneReport]] = {}
        self._stage2_modes_cache: Dict[Tuple[str, str], List[str]] = {}
        self.stage1_eval_by_scene = self._load_stage1_eval_by_scene()
        self.similarity_reports = self._discover_similarity_reports()
        self.skip_reports = self._discover_skip_reports()
        self._cache_current_v3_stage2_reports()
        self.stage2_render_cache: Dict[Tuple[str, str, int, int, int], Tuple[Dict[str, List[np.ndarray]], str]] = {}
        self.stage2_request_seq = 0

    def _v3_cache_key(self, dataset: str, backbone: Optional[str] = None) -> Tuple[str, str]:
        bb = str(backbone if backbone is not None else self.v3_backbone).strip()
        ds = str(dataset).strip().lower()
        return (bb, ds)

    def _cache_current_v3_stage2_reports(self) -> None:
        if _is_stage2_virtual_dataset(self.dataset):
            return
        key = self._v3_cache_key(self.dataset, self.v3_backbone)
        self._v3_similarity_report_cache[key] = self.similarity_reports
        self._v3_skip_report_cache[key] = self.skip_reports
        self._stage2_modes_cache[key] = _discover_stage2_modes(
            self.similarity_reports,
            self.skip_reports,
        )

    def _v3_similarity_reports_for_dataset(
        self,
        dataset: str,
        *,
        backbone: Optional[str] = None,
    ) -> Dict[str, Stage2SceneReport]:
        key = self._v3_cache_key(dataset, backbone)
        cached = self._v3_similarity_report_cache.get(key)
        if cached is not None:
            return cached
        root = self.v3_output_root / key[0]
        reports = self._discover_v3_similarity_reports(root / key[1] / "redundancy")
        self._v3_similarity_report_cache[key] = reports
        return reports

    def _v3_skip_reports_for_dataset(
        self,
        dataset: str,
        *,
        backbone: Optional[str] = None,
    ) -> Dict[str, Stage2SceneReport]:
        key = self._v3_cache_key(dataset, backbone)
        cached = self._v3_skip_report_cache.get(key)
        if cached is not None:
            return cached
        root = self.v3_output_root / key[0]
        reports = self._discover_v3_skip_reports(root / key[1] / "skip_eval")
        if not reports:
            reports = self._discover_skip_reports_in_dir(root / key[1] / "prune")
        self._v3_skip_report_cache[key] = reports
        return reports

    def v3_backbone_root(self) -> Path:
        return self.v3_output_root / self.v3_backbone

    def _apply_v3_backbone_paths(self) -> None:
        root = self.v3_backbone_root()
        self.cache_root = root
        self.eval_root = root / self.dataset / "eval"

    def _reload_v3_artifacts(self) -> None:
        self.manifest, self.scene_items = self._load_manifest_and_scenes()
        self.cache_available = self._cache_available_on_disk()
        self.hf_model = str(self.manifest.get("hf_model", "")).strip() or self.hf_model
        self.pair_available = self.cache_available and bool(self.scene_items) and bool(self.hf_model)
        self.scene_cache = {}
        self.qk_cache = {}
        self.stage2_render_cache = {}
        self.stage1_eval_by_scene = self._load_stage1_eval_by_scene()
        self.similarity_reports = self._discover_similarity_reports()
        self.skip_reports = self._discover_skip_reports()
        self._cache_current_v3_stage2_reports()
        self.stage2_request_seq += 1

    def switch_backbone(self, backbone: str) -> None:
        bb = str(backbone).strip()
        if not bb or bb == self.v3_backbone:
            return
        self.v3_backbone = bb
        self._apply_v3_backbone_paths()
        self.cache_root_exists = self.cache_root.exists() and self.cache_root.is_dir()
        self.cache_dir = self._resolve_dataset_dir(self.cache_root, self.dataset)
        self._reload_v3_artifacts()

    def _v3_cache_manifest_path(self) -> Path:
        return self.v3_backbone_root() / self.dataset / "cache" / "cache_manifest_v3.json"

    def _cache_available_on_disk(self) -> bool:
        return self._v3_cache_manifest_path().is_file()

    def _resolve_dataset_dir(self, root: Path, dataset: str) -> Path:
        p = Path(root)
        from analysis.vis.view_v3_loader import redundancy_dir_has_graphs

        cand = p / dataset / "cache"
        if (cand / "cache_manifest_v3.json").exists():
            return cand
        ds_root = p / dataset
        if redundancy_dir_has_graphs(ds_root / "redundancy"):
            return ds_root
        return p

    def _load_manifest_and_scenes(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        from analysis.vis.view_v3_loader import load_v3_bundle

        exp_root = self.v3_backbone_root()
        bundle = load_v3_bundle(exp_root, self.dataset)
        manifest_obj = bundle.get("manifest")
        from analysis.vis.view_v3_loader import scene_item_from_v3_row

        if manifest_obj is None:
            self.analysis_similarity_dir = exp_root / self.dataset / "redundancy"
            self.analysis_skip_dir = exp_root / self.dataset / "prune"
            return {}, {}

        manifest: Dict[str, Any] = {
            "schema_version": "v3",
            "hf_model": manifest_obj.hf_model,
            "backbone": manifest_obj.backbone,
            "scenes": manifest_obj.scenes,
            "layer_units": [asdict(u) for u in manifest_obj.layer_units],
            "num_heads": int(manifest_obj.num_heads),
        }
        scene_items: Dict[str, Dict[str, Any]] = {}
        for item in manifest_obj.scenes:
            sid = str(item.get("scene_id", "")).strip()
            if not sid:
                continue
            scene_items[sid] = scene_item_from_v3_row(sid, item, self.cache_dir)
        self.eval_root = exp_root / self.dataset / "eval"
        self.analysis_similarity_dir = exp_root / self.dataset / "redundancy"
        self.analysis_skip_dir = exp_root / self.dataset / "prune"
        return manifest, scene_items

    def switch_dataset(self, dataset: str) -> None:
        new_ds = str(dataset).strip().lower()
        if _is_stage2_virtual_dataset(new_ds):
            self.dataset = new_ds
            self.stage2_render_cache = {}
            self.similarity_reports = {}
            self.skip_reports = {}
            self.stage2_request_seq += 1
            return
        self.dataset = new_ds
        self._apply_v3_backbone_paths()
        self.cache_dir = self._resolve_dataset_dir(self.cache_root, self.dataset)
        self.data_root = self.data_roots_by_dataset.get(self.dataset, self.data_root)

        self._reload_v3_artifacts()

    def _get_model(self) -> Any:
        if not self.hf_model:
            raise SystemExit(
                "Pair Attention unavailable: no HF model provided and cache manifest lacks 'hf_model'."
            )
        if self.model is None:
            apply_pythonpath("vggt")
            self.model = get_backbone("vggt").load_pretrained(
                hf_model=self.hf_model,
                device=str(self.device),
                eval_mode=True,
            )
        return self.model

    def scene_ids(self) -> List[str]:
        return sorted(self.scene_items.keys())

    def stage2_scene_ids(self) -> List[str]:
        if _is_stage2_virtual_dataset(self.dataset):
            return ["mean"]
        all_ids = set(self.similarity_reports.keys()) | set(self.skip_reports.keys())
        scene_ids: List[str] = []
        if "mean" in all_ids and self.resolve_stage2_num_layers("mean") > 0:
            scene_ids.append("mean")
        scene_ids.extend(sorted(k for k in all_ids if k != "mean"))
        return scene_ids

    def get_similarity_report_for_dataset(
        self, dataset: str, scene_id: str, *, backbone: Optional[str] = None
    ) -> Optional[Stage2SceneReport]:
        ds = str(dataset).strip().lower()
        sid = str(scene_id).strip()
        reports = self._v3_similarity_reports_for_dataset(ds, backbone=backbone)
        return reports.get(sid)

    def has_cross_dataset_similarity(self) -> bool:
        return any(
            self.get_similarity_report_for_dataset(ds, "mean") is not None
            for ds in getattr(self, "stage2_real_datasets", [])
        )

    def num_heads(self) -> int:
        model = self._get_model()
        return int(model.aggregator.global_blocks[0].attn.num_heads)

    def _ensure_scene_viz_meta(self, scene_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Fill ``frame_ids`` / ``image_paths`` for v3 caches (manifest, metadata, or data-root fallback)."""
        out = dict(item)
        if out.get("frame_ids") and out.get("image_paths"):
            return out
        meta = out.get("metadata_v3")
        if isinstance(meta, dict):
            if meta.get("frame_ids") and meta.get("image_paths"):
                out["frame_ids"] = list(meta["frame_ids"])
                out["image_paths"] = list(meta["image_paths"])
                return out
        if self.dataset.startswith("scannet") and self.data_root.exists():
            from analysis.eval.scannet_eval import discover_scannet_scenes, select_scene_image_paths_fastvggt

            scenes = {s.scene_id: s for s in discover_scannet_scenes(self.data_root)}
            if scene_id in scenes:
                n = int(out.get("num_frames") or (meta or {}).get("num_frames") or 0)
                max_frames = n if n > 0 else 50
                scene = select_scene_image_paths_fastvggt(scenes[scene_id], max_frames)
                out["frame_ids"] = [p.stem for p in scene.image_paths]
                out["image_paths"] = [str(p.resolve()) for p in scene.image_paths]
        return out

    def _resolve_scene_eff(self, scene_id: str, item: Dict[str, Any]) -> SceneData:
        item = self._ensure_scene_viz_meta(scene_id, item)
        image_paths_raw = item.get("image_paths")
        if not isinstance(image_paths_raw, list) or not image_paths_raw:
            raise SystemExit(
                f"Scene {scene_id!r} manifest entry lacks image_paths; please rerun cache with cache_dataset."
            )
        image_paths = [Path(str(p)) for p in image_paths_raw]
        scene_dir = image_paths[0].parent if image_paths else self.data_root
        return SceneData(scene_id=scene_id, scene_dir=scene_dir, image_paths=image_paths)

    def get_scene_cache(self, scene_id: str) -> SceneCache:
        cached = self.scene_cache.get(scene_id)
        if cached is not None:
            return cached

        item = self._ensure_scene_viz_meta(scene_id, self.scene_items[scene_id])
        self.scene_items[scene_id] = item
        frame_ids = [str(x) for x in item.get("frame_ids", [])]
        if not frame_ids:
            raise SystemExit(
                f"Manifest entry for {scene_id!r} lacks frame_ids. "
                "Re-run cache+redundancy (run_cache_redundancy.sh) so manifest includes frame_ids/image_paths."
            )
        row = _load_cache_row(self.cache_dir, item)
        replay_layers = row.get("aggregated_concat_per_layer")
        if not isinstance(replay_layers, list) or not replay_layers:
            raise SystemExit(f"Scene {scene_id!r} cache lacks aggregated_concat_per_layer. Please rerun Stage1.")

        scene_eff = self._resolve_scene_eff(scene_id, item)
        images = load_scene_images(scene_eff, mode="crop")
        image_hw = tuple(int(x) for x in row["image_hw"])
        patch_start_idx = int(row["patch_start_idx"])
        sample = replay_layers[0]
        if not isinstance(sample, torch.Tensor) or sample.dim() != 4:
            raise SystemExit(f"Scene {scene_id!r} has invalid replay layer 0.")
        tokens_per_frame = int(sample.shape[2])
        patch_tokens = tokens_per_frame - patch_start_idx
        model = self._get_model()
        patch_h = image_hw[0] // model.aggregator.patch_size
        patch_w = image_hw[1] // model.aggregator.patch_size
        if patch_h * patch_w != patch_tokens:
            raise SystemExit(
                f"Scene {scene_id!r} patch grid mismatch: image_hw={image_hw}, "
                f"patch_size={model.aggregator.patch_size}, patch_tokens={patch_tokens}."
            )

        scene_cache = SceneCache(
            item=item,
            row=row,
            scene_eff=scene_eff,
            images=images,
            frame_ids=frame_ids,
            image_hw=image_hw,
            patch_start_idx=patch_start_idx,
            replay_layers=replay_layers,
            patch_hw=(patch_h, patch_w),
            tokens_per_frame=tokens_per_frame,
            frame_labels=[],
            frame_label_to_index={},
        )
        scene_cache.frame_labels = [
            f"index={idx} | frame_id={scene_cache.frame_ids[idx]} | {scene_cache.scene_eff.image_paths[idx].name}"
            for idx in range(len(scene_cache.frame_ids))
        ]
        scene_cache.frame_label_to_index = {label: idx for idx, label in enumerate(scene_cache.frame_labels)}
        self.scene_cache[scene_id] = scene_cache
        return scene_cache

    def get_qk(self, scene_id: str, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (scene_id, int(layer_idx))
        cached = self.qk_cache.get(key)
        if cached is not None:
            return cached

        scene_cache = self.get_scene_cache(scene_id)
        replay_layers = scene_cache.replay_layers
        if not (0 <= layer_idx < len(replay_layers)):
            raise SystemExit(f"Layer index {layer_idx} out of range for scene {scene_id!r}.")

        model = self._get_model()
        pos_global = _build_global_positions(
            model,
            num_frames=len(scene_cache.frame_ids),
            image_hw=scene_cache.image_hw,
            device=self.device,
        )
        concat = replay_layers[layer_idx]
        if not isinstance(concat, torch.Tensor) or concat.dim() != 4:
            raise SystemExit(f"Layer {layer_idx} replay cache is invalid for scene {scene_id!r}.")
        two_c = int(concat.shape[-1])
        if two_c % 2 != 0:
            raise SystemExit(f"Layer {layer_idx} replay cache has odd channel dimension: {two_c}")
        frame_layer = concat[..., : two_c // 2]
        qk = _reconstruct_qk(
            model,
            frame_layer,
            layer_idx=layer_idx,
            pos_global=pos_global,
            device=self.device,
        )
        self.qk_cache[key] = qk
        return qk

    def scene_metadata(self, scene_id: str) -> Tuple[int, int, List[str]]:
        scene_cache = self.get_scene_cache(scene_id)
        num_layers = len(scene_cache.replay_layers)
        num_frames = len(scene_cache.frame_ids)
        return num_layers, num_frames, list(scene_cache.frame_labels)

    def scene_metadata_light(self, scene_id: str) -> Tuple[int, int, List[str]]:
        """
        Scene shape + frame labels without loading VGGT or replay tensors (per-scene directory layout).

        Falls back to :meth:`scene_metadata` for legacy single-file caches (may load a large row once).
        """
        if scene_id not in self.scene_items:
            raise KeyError(scene_id)
        item = self._ensure_scene_viz_meta(scene_id, self.scene_items[scene_id])
        self.scene_items[scene_id] = item
        if scene_cache_dir(self.cache_dir, item) is None:
            return self.scene_metadata(scene_id)

        n_disk = count_scene_replay_layers_disk(self.cache_dir, item)
        meta = load_scene_metadata(self.cache_dir, item)
        if n_disk and n_disk > 0:
            num_layers = int(n_disk)
        else:
            ac = meta.get("aggregated_concat_per_layer")
            if isinstance(ac, list) and ac:
                num_layers = len(ac)
            else:
                return self.scene_metadata(scene_id)

        frame_ids = [str(x) for x in item.get("frame_ids", [])]
        if not frame_ids:
            raise SystemExit(f"Manifest entry for {scene_id!r} lacks frame_ids.")
        image_paths_raw = item.get("image_paths")
        if not isinstance(image_paths_raw, list) or not image_paths_raw:
            raise SystemExit(f"Manifest entry for {scene_id!r} lacks image_paths.")
        image_paths = [Path(str(p)) for p in image_paths_raw]
        num_frames = int(meta.get("num_frames", len(frame_ids)))
        frame_labels = [
            f"index={idx} | frame_id={frame_ids[idx]} | {image_paths[idx].name}"
            for idx in range(len(frame_ids))
        ]
        return num_layers, num_frames, frame_labels

    def pair_ui_num_heads(self) -> int:
        """Attention head count for Pair UI sliders without loading checkpoint weights."""
        raw = self.manifest.get("num_heads")
        if raw is not None:
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass
        return 16

    def frame_index_from_label(self, scene_id: str, label: str) -> int:
        scene_cache = self.get_scene_cache(scene_id)
        if label not in scene_cache.frame_label_to_index:
            raise gr.Error(f"Unknown frame selection for scene {scene_id!r}: {label!r}")
        return int(scene_cache.frame_label_to_index[label])

    def _load_json_if_exists(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _load_stage1_eval_by_scene(self) -> Dict[str, Dict[str, float]]:
        eval_path = self.cache_dir / "eval.json"
        payload = self._load_json_if_exists(eval_path)
        out: Dict[str, Dict[str, float]] = {}
        for row in payload.get("scenes", []):
            if not isinstance(row, dict):
                continue
            scene_id = str(row.get("scene_id", "")).strip()
            if not scene_id:
                continue
            out[scene_id] = row
        return out

    def _discover_similarity_reports_in_dir(self, base_dir: Path) -> Dict[str, Stage2SceneReport]:
        result: Dict[str, Stage2SceneReport] = {}
        if not base_dir.exists():
            return result

        mean_report = self._load_json_if_exists(base_dir / "full_report_similarity.json")
        if mean_report:
            result["mean"] = Stage2SceneReport(scene_id="mean", report=mean_report)

        per_scene_dir = base_dir / "per_scene"
        if per_scene_dir.exists():
            for activation_json in sorted(per_scene_dir.glob("*_similarity.json")):
                payload = self._load_json_if_exists(activation_json)
                scene_id = str(payload.get("scene_id", "")).strip()
                if scene_id:
                    result[scene_id] = Stage2SceneReport(scene_id=scene_id, report=payload)
        return result

    def _discover_skip_reports_in_dir(self, base_dir: Path) -> Dict[str, Stage2SceneReport]:
        result: Dict[str, Stage2SceneReport] = {}
        if not base_dir.exists():
            return result

        mean_report = self._load_json_if_exists(base_dir / "full_report_skip.json")
        if mean_report:
            result["mean"] = Stage2SceneReport(scene_id="mean", report=mean_report)

        per_scene_dir = base_dir / "per_scene"
        if per_scene_dir.exists():
            for activation_json in sorted(per_scene_dir.glob("*_skip.json")):
                payload = self._load_json_if_exists(activation_json)
                scene_id = str(payload.get("scene_id", "")).strip()
                if scene_id:
                    result[scene_id] = Stage2SceneReport(scene_id=scene_id, report=payload)
        return result

    def _discover_v3_similarity_reports(self, redundancy_dir: Path) -> Dict[str, Stage2SceneReport]:
        from analysis.vis.view_v3_loader import discover_v3_similarity_reports

        raw = discover_v3_similarity_reports(redundancy_dir)
        return {
            sid: Stage2SceneReport(scene_id=sid, report=rep)
            for sid, rep in raw.items()
        }

    def _discover_v3_skip_reports(self, skip_eval_dir: Path) -> Dict[str, Stage2SceneReport]:
        from analysis.vis.view_v3_loader import discover_v3_skip_reports

        raw = discover_v3_skip_reports(skip_eval_dir)
        return {
            sid: Stage2SceneReport(scene_id=sid, report=rep)
            for sid, rep in raw.items()
        }

    def _discover_similarity_reports(self) -> Dict[str, Stage2SceneReport]:
        return self._discover_v3_similarity_reports(Path(self.analysis_similarity_dir))

    def _discover_skip_reports(self) -> Dict[str, Stage2SceneReport]:
        skip_v3_dir = self.v3_backbone_root() / self.dataset / "skip_eval"
        reports = self._discover_v3_skip_reports(skip_v3_dir)
        if reports:
            return reports
        return self._discover_skip_reports_in_dir(self.analysis_skip_dir)

    def stage2_modes_for_dataset(self, dataset: str, *, backbone: Optional[str] = None) -> List[str]:
        ds = str(dataset).strip().lower()
        key = self._v3_cache_key(ds, backbone)
        cached_modes = self._stage2_modes_cache.get(key)
        if cached_modes is not None:
            return list(cached_modes)
        similarity_reports = self._v3_similarity_reports_for_dataset(ds, backbone=backbone)
        skip_reports = self._v3_skip_reports_for_dataset(ds, backbone=backbone)
        modes = _discover_stage2_modes(similarity_reports, skip_reports)
        self._stage2_modes_cache[key] = list(modes)
        return modes

    def union_stage2_modes(self, dataset_choices: Sequence[str]) -> List[str]:
        modes: List[str] = []
        backbones = self.v3_backbones or [self.v3_backbone]
        for bb in backbones:
            for ds in dataset_choices:
                for mode in self.stage2_modes_for_dataset(ds, backbone=bb):
                    if mode not in modes:
                        modes.append(mode)
        if not modes:
            modes = _discover_stage2_modes(self.similarity_reports, self.skip_reports)
        return modes

    def get_similarity_report(self, scene_id: str) -> Optional[Stage2SceneReport]:
        return self.similarity_reports.get(scene_id)

    def get_skip_report(self, scene_id: str) -> Optional[Stage2SceneReport]:
        return self.skip_reports.get(scene_id)

    def resolve_stage2_num_layers(self, scene_id: str) -> int:
        def _num_layers_from_report(report: Dict[str, Any]) -> int:
            try:
                return int(report.get("num_layers", 0))
            except Exception:
                return 0

        if _is_stage2_virtual_dataset(self.dataset):
            for ds in getattr(self, "stage2_real_datasets", []):
                rep = self.get_similarity_report_for_dataset(ds, "mean")
                if rep is not None:
                    n = _num_layers_from_report(rep.report)
                    if n > 0:
                        return n
            return 0

        sim = self.get_similarity_report(scene_id)
        if sim is not None:
            n = _num_layers_from_report(sim.report)
            if n > 0:
                return n
        skip = self.get_skip_report(scene_id)
        if skip is not None:
            n = _num_layers_from_report(skip.report)
            if n > 0:
                return n

        # "mean" can exist even when full_report_* lacks num_layers (e.g. interrupted run).
        # Fall back to any per-scene report.
        if scene_id == "mean":
            for sid, entry in self.similarity_reports.items():
                if sid == "mean":
                    continue
                n = _num_layers_from_report(entry.report)
                if n > 0:
                    return n
            for sid, entry in self.skip_reports.items():
                if sid == "mean":
                    continue
                n = _num_layers_from_report(entry.report)
                if n > 0:
                    return n

        return 0


class EvalViserController:
    """Serve per-scene pointcloud/trajectory assets via a single Viser server."""

    def __init__(self, *, eval_root: Path, host: str, port: int) -> None:
        self.eval_root = Path(eval_root).resolve()
        self.host = str(host)
        self.port = int(port)
        self._server = None
        self._server_lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._loaded_scene: str | None = None
        self._callbacks_bound = False
        self._request_lock = threading.Lock()
        self._latest_request_id = 0
        self._latest_scene_id: str | None = None
        self._default_point_size = 0.003

        # Mutable scene state read by GUI callbacks.
        self._pred_pts = np.zeros((0, 3), dtype=np.float32)
        self._pred_cols = np.zeros((0, 3), dtype=np.uint8)
        self._pred_order = np.zeros((0,), dtype=np.int64)

    def set_eval_root(self, eval_root: Path) -> None:
        """Point the loader at a different on-disk eval run (e.g. another model or dataset)."""
        self.eval_root = Path(eval_root).resolve()

    def _ensure_server(self):
        if self._server is not None:
            return self._server
        with self._server_lock:
            if self._server is not None:
                return self._server
            import viser

            server = viser.ViserServer(host=self.host, port=self.port)
            server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
            self._server = server
            return server

    def _next_request_id(self, scene_id: str) -> int:
        with self._request_lock:
            self._latest_request_id += 1
            self._latest_scene_id = str(scene_id)
            return int(self._latest_request_id)

    def _is_request_current(self, request_id: int) -> bool:
        with self._request_lock:
            return int(request_id) == int(self._latest_request_id)

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        data = np.load(path)
        return {k: np.asarray(data[k]) for k in data.files}

    def _add_or_update_point_cloud(self, server, name: str, points: np.ndarray, colors_u8: np.ndarray):
        # Viser point cloud handle supports in-place update of points/colors.
        handle = getattr(self, f"_pc_{name}", None)
        if handle is None:
            handle = server.scene.add_point_cloud(
                name=name,
                points=np.asarray(points, dtype=np.float32),
                colors=np.asarray(colors_u8, dtype=np.uint8),
                point_size=float(getattr(self, "_gui_point_size", None).value if hasattr(self, "_gui_point_size") else self._default_point_size),
                point_shape="circle",
            )
            setattr(self, f"_pc_{name}", handle)
        else:
            handle.points = np.asarray(points, dtype=np.float32)
            handle.colors = np.asarray(colors_u8, dtype=np.uint8)

    def _add_or_update_traj(self, server, name: str, centers: np.ndarray, color_rgb=(1.0, 0.4, 0.1)):
        handle = getattr(self, f"_traj_{name}", None)
        pts = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < 2:
            if handle is not None:
                handle.remove()
                setattr(self, f"_traj_{name}", None)
            return
        # viser==0.2.23 add_line_segments expects points shaped (N,2,3).
        seg_points = np.stack([pts[:-1], pts[1:]], axis=1)  # [N,2,3]
        rgb = np.asarray(color_rgb, dtype=np.float32).reshape(-1)[:3]
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        seg_colors = np.tile(rgb_u8.reshape(1, 1, 3), (int(seg_points.shape[0]), 2, 1))  # [N,2,3] uint8
        if handle is None:
            handle = server.scene.add_line_segments(
                name=name,
                points=seg_points,
                colors=seg_colors,
                line_width=2.0,
            )
            setattr(self, f"_traj_{name}", handle)
        else:
            handle.points = seg_points
            handle.colors = seg_colors

    def _bind_gui_callbacks_once(self, server) -> None:
        if self._callbacks_bound:
            return
        self._gui_point_size = server.gui.add_slider(
            "Point Size", min=0.0005, max=0.02, step=0.0005, initial_value=float(self._default_point_size)
        )
        self._gui_conf = server.gui.add_slider("Confidence Percent", min=0, max=100, step=0.1, initial_value=50.0)
        self._gui_show_gt = server.gui.add_checkbox("Show GT", initial_value=False)
        self._gui_show_pred = server.gui.add_checkbox("Show Pred", initial_value=True)

        @self._gui_point_size.on_update
        def _(_evt):
            size = float(self._gui_point_size.value)
            for name in ("_pc_pred_points", "_pc_gt_points"):
                pc = getattr(self, name, None)
                if pc is not None:
                    pc.point_size = size

        @self._gui_conf.on_update
        def _(_evt):
            with self._apply_lock:
                self._apply_conf_locked(server)

        @self._gui_show_pred.on_update
        def _(_evt):
            pc = getattr(self, "_pc_pred_points", None)
            if pc is not None:
                pc.visible = bool(self._gui_show_pred.value)

        @self._gui_show_gt.on_update
        def _(_evt):
            pc = getattr(self, "_pc_gt_points", None)
            if pc is not None:
                pc.visible = bool(self._gui_show_gt.value)
            traj = getattr(self, "_traj_gt_traj", None)
            if traj is not None:
                traj.visible = bool(self._gui_show_gt.value)

        self._callbacks_bound = True

    def _apply_conf_locked(self, server) -> None:
        # Keep top (100-pct)% points: pct=0 -> keep all; pct=100 -> keep none.
        pct = float(self._gui_conf.value)
        n = int(self._pred_pts.shape[0])
        if n <= 0:
            return
        k = int(round(n * (pct / 100.0)))
        keep = self._pred_order[k:]
        if keep.size == 0:
            keep = self._pred_order[:1]
        self._add_or_update_point_cloud(server, "pred_points", self._pred_pts[keep], self._pred_cols[keep])
        pc = getattr(self, "_pc_pred_points", None)
        if pc is not None:
            pc.visible = bool(self._gui_show_pred.value)

    def load_scene(self, scene_id: str, *, show_gt: bool) -> str:
        """Load one scene assets and return a status string.

        Latest-request-wins: if a newer scene request arrives while this one is loading,
        this request exits early and does not overwrite scene state.
        """
        request_id = self._next_request_id(str(scene_id))
        scene_dir = self.eval_root / str(scene_id)
        pred_dir = scene_dir / "vis" / "pred"
        gt_dir = scene_dir / "vis" / "gt"
        pred_npz = pred_dir / "viser_points.npz"
        pred_cam = pred_dir / "cameras.json"
        gt_npz = gt_dir / "viser_points_gt.npz"
        gt_cam = gt_dir / "cameras_gt.json"

        if not pred_npz.exists():
            return f"Missing {pred_npz}"

        # Load files outside apply lock so UI callbacks can continue.
        pred = self._load_npz(pred_npz)
        pts = pred["points"]
        cols = pred["colors"]
        conf = pred.get("confidence", np.ones((pts.shape[0],), dtype=np.float32))
        order = np.argsort(conf.astype(np.float32), kind="mergesort")
        pred_centers = None
        if pred_cam.exists():
            payload = json.loads(pred_cam.read_text(encoding="utf-8"))
            pred_centers = np.asarray([f["camera_center_world"] for f in payload.get("frames", [])], dtype=np.float32)
        gt = None
        gt_centers = None
        if show_gt and gt_npz.exists():
            gt = self._load_npz(gt_npz)
        if show_gt and gt_cam.exists():
            payload = json.loads(gt_cam.read_text(encoding="utf-8"))
            gt_centers = np.asarray(payload.get("camera_centers_world", []), dtype=np.float32)

        if not self._is_request_current(request_id):
            return f"Skipped stale request for scene={scene_id}"

        server = self._ensure_server()
        with self._apply_lock:
            if not self._is_request_current(request_id):
                return f"Skipped stale request for scene={scene_id}"

            self._bind_gui_callbacks_once(server)
            self._pred_pts = pts.astype(np.float32)
            self._pred_cols = cols.astype(np.uint8)
            self._pred_order = order.astype(np.int64)
            self._apply_conf_locked(server)

            # Pred cameras / trajectory
            if pred_centers is not None:
                self._add_or_update_traj(server, "pred_traj", pred_centers, color_rgb=(1.0, 0.4, 0.1))

            # GT optional
            if show_gt and gt is not None:
                self._add_or_update_point_cloud(server, "gt_points", gt["points"], gt["colors"])
                pc = getattr(self, "_pc_gt_points", None)
                if pc is not None:
                    pc.visible = bool(self._gui_show_gt.value)
            if show_gt and gt_centers is not None:
                self._add_or_update_traj(server, "gt_traj", gt_centers, color_rgb=(0.2, 0.8, 0.2))
                traj = getattr(self, "_traj_gt_traj", None)
                if traj is not None:
                    traj.visible = bool(self._gui_show_gt.value)

            self._loaded_scene = str(scene_id)
            return f"Loaded scene={scene_id} | pred_points={int(pts.shape[0])} | gt={'on' if show_gt else 'off'}"


def _gallery_update(items: List[np.ndarray]):
    return gr.update(value=items, columns=max(1, len(items)))


STAGE2_METRIC_ROW_POSE = "pose"
STAGE2_METRIC_ROW_DEPTH = "depth"
STAGE2_METRIC_ROW_POINT = "point cloud"
STAGE2_TOKEN_ROW_SPECIAL = "special"
STAGE2_TOKEN_ROW_CAMERA = "camera"
STAGE2_TOKEN_ROW_PATCH = "patch"
STAGE2_TOKEN_ROW_ALL = "all"
STAGE2_TOKEN_ROW_MEAN = "mean"
STAGE2_TOKEN_ROW_MIN = "min"
STAGE2_TOKEN_ROW_ORDER = [
    STAGE2_TOKEN_ROW_SPECIAL,
    STAGE2_TOKEN_ROW_CAMERA,
    STAGE2_TOKEN_ROW_PATCH,
    STAGE2_TOKEN_ROW_ALL,
    STAGE2_TOKEN_ROW_MEAN,
    STAGE2_TOKEN_ROW_MIN,
]
STAGE2_MIN_DATASET = "min"
STAGE2_MEAN_DATASET = "mean"
STAGE2_VIRTUAL_DATASETS = {STAGE2_MIN_DATASET, STAGE2_MEAN_DATASET}


def _is_stage2_virtual_dataset(dataset: str) -> bool:
    return str(dataset or "").strip().lower() in STAGE2_VIRTUAL_DATASETS


def _discover_v3_dataset_tags(backbone_root: Path) -> List[str]:
    """Discover dataset tags under analysis_outputs_v3/<backbone>/.

    For Heatmap Stage2 we only care about datasets that actually have v3 redundancy
    outputs (similarity/skip reports). Datasets with only v3 cache but no redundancy
    should be omitted from the dropdown and excluded from "min" aggregation.
    """
    root = Path(backbone_root)
    if not root.is_dir():
        return ["scannet_dev50", "scannet_test50", "7scenes", "nrgbd"]
    tags: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        name = str(p.name).strip()
        if not name or name.startswith(".") or _is_stage2_virtual_dataset(name):
            continue
        # Only include datasets with actual v3 redundancy outputs.
        from analysis.vis.view_v3_loader import redundancy_dir_has_graphs

        if redundancy_dir_has_graphs(p / "redundancy"):
            tags.append(name)
    # Prefer ScanNet splits first for UX.
    prefer = ["scannet_dev50", "scannet_test50", "scannet", "7scenes", "nrgbd"]
    out: List[str] = []
    for t in prefer:
        if t in tags and t not in out:
            out.append(t)
    for t in tags:
        if t not in out:
            out.append(t)
    return out


def _discover_v3_dataset_tags_across_backbones(output_root: Path, backbones: Sequence[str]) -> List[str]:
    tags: List[str] = []
    for backbone in backbones:
        for tag in _discover_v3_dataset_tags(Path(output_root) / str(backbone)):
            if tag not in tags:
                tags.append(tag)
    return tags


def _heatmaps_resolve_filters(
    controller: PairAttentionController,
    scene_id: Optional[str],
    value_mode: str,
    metric_rows: Optional[List[str]],
    token_rows: Optional[List[str]],
) -> Tuple[str, List[str], List[str], List[str], List[str], bool, bool, bool]:
    """Coerce value mode and row filters from UI + current reports (per-dataset Stage1 eval for pct)."""
    scene_key = str(scene_id or "").strip()
    if scene_key == "":
        pct_ok = False
    elif scene_key == "mean":
        pct_ok = bool(controller.stage1_eval_by_scene)
    else:
        pct_ok = scene_key in controller.stage1_eval_by_scene
    choices_vm = ["absolute", "pct"]
    vm = str(value_mode or "absolute")
    if vm == "pct" and not pct_ok:
        vm = "absolute"
    if vm not in choices_vm:
        vm = "absolute"
    if _is_stage2_virtual_dataset(controller.dataset):
        has_skip = False
        has_sim = controller.has_cross_dataset_similarity()
    else:
        has_skip = bool(scene_id) and controller.get_skip_report(scene_id) is not None
        has_sim = bool(scene_id) and controller.get_similarity_report(scene_id) is not None

    token_choices: List[str] = []
    if has_sim:
        if _is_stage2_virtual_dataset(controller.dataset):
            token_choices = [STAGE2_TOKEN_ROW_ALL]
        else:
            sim_report = controller.get_similarity_report(scene_key)
            by_mode = (
                _similarity_maps_with_token_aggregations(
                    _extract_similarity_metric_maps_by_mode(sim_report.report)
                )
                if sim_report is not None
                else {}
            )
            present = {
                region
                for mode_entry in by_mode.values()
                for region in STAGE2_TOKEN_ROW_ORDER
                if _similarity_region_has_data(mode_entry, region)
            }
            token_choices = [x for x in STAGE2_TOKEN_ROW_ORDER if x in present]
        has_sim = bool(token_choices)

    if has_skip:
        allowed_m = {
            STAGE2_METRIC_ROW_POSE,
            STAGE2_METRIC_ROW_DEPTH,
            STAGE2_METRIC_ROW_POINT,
        }
        if metric_rows is None:
            m_val = [
                STAGE2_METRIC_ROW_POSE,
                STAGE2_METRIC_ROW_DEPTH,
                STAGE2_METRIC_ROW_POINT,
            ]
        else:
            m_val = [x for x in metric_rows if x in allowed_m]
    else:
        m_val = []

    if has_sim:
        allowed_t = set(token_choices)
        if token_rows is None:
            t_val = list(token_choices)
        else:
            t_val = [x for x in token_rows if x in allowed_t]
            if not t_val:
                t_val = list(token_choices)
    else:
        t_val = []

    return vm, m_val, t_val, choices_vm, token_choices, has_skip, has_sim, pct_ok


def _heatmaps_gr_controls_from_resolved(
    vm: str,
    m_val: List[str],
    t_val: List[str],
    sim_gamma: float,
    sim_cutoff: float,
    _choices_vm: List[str],
    token_choices: List[str],
    has_skip: bool,
    has_sim: bool,
    pct_ok: bool,
    *,
    write_values: bool,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    cls_vm = ["hm-value-mode"]
    if not pct_ok:
        cls_vm.append("hm-value-mode-pct-locked")
    metric_update: Dict[str, Any] = {"interactive": has_skip}
    token_update: Dict[str, Any] = {
        "interactive": has_sim,
        "choices": list(token_choices),
    }
    sim_gamma_update: Dict[str, Any] = {"interactive": has_sim}
    sim_cutoff_update: Dict[str, Any] = {"interactive": has_sim or has_skip}
    if write_values:
        metric_update["value"] = m_val
        token_update["value"] = list(t_val) if has_sim else []
        sim_gamma_update["value"] = float(sim_gamma)
        sim_cutoff_update["value"] = float(sim_cutoff)
    return (
        gr.update(choices=["absolute", "pct"], value=vm, interactive=True, elem_classes=cls_vm),
        gr.update(interactive=pct_ok and vm == "pct"),
        gr.update(**metric_update),
        gr.update(**token_update),
        gr.update(**sim_gamma_update),
        gr.update(**sim_cutoff_update),
    )


_LORA_METRIC_ORDER = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "depth_abs_rel",
    "depth_delta1",
    "acc",
    "comp",
    "cd",
    "nc",
    "chamfer_distance",
)


_LORA_DIRECT_TRAINING_PLAN = "direct"
_RECOVERY_PLOT_GROUP = "all_prune_spans"
_RECOVERY_VIEW = "strategy_comparison"
_RECOVERY_SPAN_RE = re.compile(r"^prune_span(\d+)$")
_RECOVERY_METRIC_GRID = (
    ("ape", "are", "rpe_trans", "rpe_rot"),
    ("acc", "comp", "cd", "nc"),
    ("depth_abs_rel", "depth_delta1", "inference_time_ms", "infer_peak_vram_gb"),
)
_RECOVERY_METRIC_HEADERS = {
    "ape": "APE ↓",
    "are": "ARE (°) ↓",
    "rpe_trans": "RPE trans ↓",
    "rpe_rot": "RPE rot (°) ↓",
    "depth_abs_rel": "AbsRel ↓",
    "depth_delta1": "δ1 ↑",
    "acc": "Acc ↓",
    "comp": "Comp ↓",
    "cd": "CD ↓",
    "nc": "NC ↑",
    "inference_time_ms": "Time (ms) ↓",
    "infer_peak_vram_gb": "VRAM (GB) ↓",
}


def _has_lora_run_outputs(run_dir: Path) -> bool:
    return (
        any(run_dir.glob("eval_plot_*.png"))
        or (run_dir / "train_curves.png").is_file()
        or (run_dir / "train_loss_weighted_components.png").is_file()
    )


def _discover_lora_runs(root: Path) -> Dict[str, Dict[str, List[str]]]:
    """Return backbone -> prune plan -> training plans for v3 LoRA train runs."""
    root = Path(root)
    runs: Dict[str, Dict[str, List[str]]] = {}
    if not root.is_dir():
        return runs
    for model_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        train_roots = [model_dir / "train", model_dir / "scannet_dev50" / "train"]
        for train_root in train_roots:
            if not train_root.is_dir():
                continue
            for prune_plan_dir in sorted((p for p in train_root.iterdir() if p.is_dir()), key=lambda p: p.name):
                if _has_lora_run_outputs(prune_plan_dir):
                    runs.setdefault(model_dir.name, {}).setdefault(
                        prune_plan_dir.name, []
                    ).append(_LORA_DIRECT_TRAINING_PLAN)
                for training_plan_dir in sorted(
                    (p for p in prune_plan_dir.iterdir() if p.is_dir()), key=lambda p: p.name
                ):
                    if _has_lora_run_outputs(training_plan_dir):
                        runs.setdefault(model_dir.name, {}).setdefault(
                            prune_plan_dir.name, []
                        ).append(training_plan_dir.name)
    return {
        backbone: {
            prune_plan: sorted(set(training_plans))
            for prune_plan, training_plans in sorted(prune_runs.items())
        }
        for backbone, prune_runs in sorted(runs.items())
    }


def _lora_metric_sort_key(metric: str) -> Tuple[int, str]:
    try:
        return (_LORA_METRIC_ORDER.index(metric), metric)
    except ValueError:
        return (len(_LORA_METRIC_ORDER), metric)


def _lora_dataset_gallery_items(run_dir: Path, dataset_tag: str) -> List[Tuple[str, str]]:
    prefix = f"eval_plot_{dataset_tag}_"
    items: List[Tuple[str, str]] = []
    for path in sorted(run_dir.glob(f"{prefix}*.png")):
        metric = path.stem[len(prefix):]
        items.append((str(path.resolve()), metric))
    return sorted(items, key=lambda item: _lora_metric_sort_key(item[1]))


def _lora_run_dir(root: Path, model_name: str, prune_plan_name: str, training_plan_name: str) -> Path:
    training_plan_name = str(training_plan_name or "").strip()
    candidates = [
        root / str(model_name) / "train" / str(prune_plan_name),
        root / str(model_name) / "scannet_dev50" / "train" / str(prune_plan_name),
    ]
    for prune_plan_dir in candidates:
        run_dir = (
            prune_plan_dir
            if not training_plan_name or training_plan_name == _LORA_DIRECT_TRAINING_PLAN
            else prune_plan_dir / training_plan_name
        )
        if run_dir.is_dir():
            return run_dir
    prune_plan_dir = candidates[0]
    if not training_plan_name or training_plan_name == _LORA_DIRECT_TRAINING_PLAN:
        return prune_plan_dir
    return prune_plan_dir / training_plan_name


def _lora_plot_rows(
    root: Path,
    prune_plan_name: str,
    training_plan_name: str,
    model_name: str,
) -> Dict[str, List[Tuple[str, str]]]:
    root = Path(root)
    run_dir = _lora_run_dir(root, model_name, prune_plan_name, training_plan_name)
    training: List[Tuple[str, str]] = []
    for filename, label in (
        ("train_curves.png", "train_curves"),
        ("train_loss_weighted_components.png", "train_loss_weighted_components"),
    ):
        path = run_dir / filename
        if path.is_file():
            training.append((str(path.resolve()), label))
    if not run_dir.is_dir():
        return {
            "scans_dev50": [],
            "scans_test50": [],
            "7scenes_test": [],
            "nrgbd": [],
            "nuscenes_val": [],
            "waymo_validation": [],
            "training": training,
        }
    return {
        "scans_dev50": _lora_dataset_gallery_items(run_dir, "scannet_dev50"),
        "scans_test50": _lora_dataset_gallery_items(run_dir, "scannet_test50"),
        "7scenes_test": _lora_dataset_gallery_items(run_dir, "7scenes_test"),
        "nrgbd": _lora_dataset_gallery_items(run_dir, "nrgbd"),
        "nuscenes_val": _lora_dataset_gallery_items(run_dir, "nuscenes_val"),
        "waymo_validation": _lora_dataset_gallery_items(run_dir, "waymo_validation"),
        "training": training,
    }


def _has_recovery_plot_outputs(backbone_dir: Path) -> bool:
    return any(backbone_dir.glob("*/eval/recover/*_recover_strategies_by_span.png"))


def _discover_recovery_runs(root: Path) -> Dict[str, Dict[str, List[str]]]:
    """Return backbone -> plot group -> recovery views for summary-generated recovery plots."""
    root = Path(root)
    runs: Dict[str, Dict[str, List[str]]] = {}
    if not root.is_dir():
        return runs
    for model_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if _has_recovery_plot_outputs(model_dir):
            runs.setdefault(model_dir.name, {}).setdefault(_RECOVERY_PLOT_GROUP, []).append(
                _RECOVERY_VIEW
            )
    return {
        backbone: {
            plot_group: sorted(set(recovery_views))
            for plot_group, recovery_views in sorted(recovery_runs.items())
        }
        for backbone, recovery_runs in sorted(runs.items())
    }


def _recovery_dataset_gallery_item(
    root: Path,
    model_name: str,
    dataset_tag: str,
    label: str,
) -> List[Tuple[str, str]]:
    if str(dataset_tag) == "mean":
        path = Path(root) / str(model_name) / "eval" / "recover" / "mean_recover_strategies_by_span.png"
    else:
        path = (
            Path(root)
            / str(model_name)
            / str(dataset_tag)
            / "eval"
            / "recover"
            / f"{dataset_tag}_recover_strategies_by_span.png"
        )
    if not path.is_file():
        return []
    return [(str(path.resolve()), label)]


def _recovery_dataset_tags(root: Path, model_name: str) -> List[str]:
    model_dir = Path(root) / str(model_name)
    tags = set()
    if not model_dir.is_dir():
        return []
    for path in model_dir.glob("*/eval/recover/*_recover_strategies_by_span.png"):
        dataset_tag = path.parent.parent.parent.name
        if path.name == f"{dataset_tag}_recover_strategies_by_span.png":
            tags.add(dataset_tag)
    order = (
        "scannet_dev50",
        "scannet_test50",
        "7scenes_test",
        "7scenes_train",
        "nrgbd",
        "nuscenes_val",
        "nuscenes_train",
        "waymo_validation",
        "waymo_training",
    )
    return sorted(tags, key=lambda tag: (order.index(tag) if tag in order else len(order), tag))


def _read_eval_average(path: Path) -> Dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = payload.get("average", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            out[str(key)] = v
    return out


def _append_average_values(bucket: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for metric, value in values.items():
        if math.isfinite(float(value)):
            bucket.setdefault(metric, []).append(float(value))


def _mean_metric_values(bucket: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        metric: float(sum(values) / len(values))
        for metric, values in bucket.items()
        if values
    }


def _generate_mean_recovery_plot(root: Path, model_name: str) -> Optional[Path]:
    root = Path(root)
    model_dir = root / str(model_name)
    dataset_tags = _recovery_dataset_tags(root, model_name)
    if not dataset_tags:
        return None

    upper_bucket: Dict[str, List[float]] = {}
    lower_buckets: Dict[int, Dict[str, List[float]]] = {}
    recover_buckets: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    all_spans = set()

    for dataset_tag in dataset_tags:
        eval_root = model_dir / dataset_tag / "eval"
        _append_average_values(
            upper_bucket,
            _read_eval_average(eval_root / "upper" / "eval_record_v3.json"),
        )

        lower_root = eval_root / "lower"
        if lower_root.is_dir():
            for rec_path in sorted(lower_root.glob("prune_span*/eval_record_v3.json")):
                m = _RECOVERY_SPAN_RE.match(rec_path.parent.name)
                if not m:
                    continue
                span = int(m.group(1))
                all_spans.add(span)
                _append_average_values(
                    lower_buckets.setdefault(span, {}),
                    _read_eval_average(rec_path),
                )

        recover_root = eval_root / "recover"
        if recover_root.is_dir():
            for plan_dir in sorted((p for p in recover_root.iterdir() if p.is_dir()), key=lambda p: p.name):
                for rec_path in sorted(plan_dir.glob("prune_span*/eval_record_v3.json")):
                    m = _RECOVERY_SPAN_RE.match(rec_path.parent.name)
                    if not m:
                        continue
                    span = int(m.group(1))
                    all_spans.add(span)
                    _append_average_values(
                        recover_buckets.setdefault(plan_dir.name, {}).setdefault(span, {}),
                        _read_eval_average(rec_path),
                    )

    if not all_spans:
        return None

    upper_mean = _mean_metric_values(upper_bucket)
    lower_mean = {span: _mean_metric_values(bucket) for span, bucket in lower_buckets.items()}
    recover_mean = {
        plan: {span: _mean_metric_values(bucket) for span, bucket in by_span.items()}
        for plan, by_span in recover_buckets.items()
    }
    spans = sorted(all_spans)

    fig, axes = plt.subplots(3, 4, figsize=(20, 9.75), squeeze=False)
    for row_idx, row in enumerate(_RECOVERY_METRIC_GRID):
        for col_idx, metric in enumerate(row):
            ax = axes[row_idx][col_idx]
            upper_value = float(upper_mean.get(metric, float("nan")))
            has_data = math.isfinite(upper_value)
            if math.isfinite(upper_value):
                ax.axhline(
                    upper_value,
                    linestyle="--",
                    color="#333333",
                    linewidth=1.4,
                    alpha=0.75,
                    label="upper",
                )
                ax.scatter([0], [upper_value], color="#333333", marker="*", s=70, zorder=4)

            lower_xs: List[int] = []
            lower_ys: List[float] = []
            if math.isfinite(upper_value):
                lower_xs.append(0)
                lower_ys.append(upper_value)
            for span in spans:
                value = float(lower_mean.get(span, {}).get(metric, float("nan")))
                if not math.isfinite(value):
                    continue
                lower_xs.append(span)
                lower_ys.append(value)
            if len(lower_ys) > (1 if math.isfinite(upper_value) else 0):
                has_data = True
                ax.plot(
                    lower_xs,
                    lower_ys,
                    marker="s",
                    linestyle=":",
                    color="#666666",
                    linewidth=1.9,
                    markersize=4,
                    label="lower",
                )

            for plan in sorted(recover_mean.keys()):
                xs: List[int] = []
                ys: List[float] = []
                if math.isfinite(upper_value):
                    xs.append(0)
                    ys.append(upper_value)
                for span in spans:
                    value = float(recover_mean.get(plan, {}).get(span, {}).get(metric, float("nan")))
                    if not math.isfinite(value):
                        continue
                    xs.append(span)
                    ys.append(value)
                if len(ys) <= (1 if math.isfinite(upper_value) else 0):
                    continue
                has_data = True
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    label=plan.removeprefix("recovery_"),
                )

            if not has_data:
                ax.axis("off")
                continue
            ax.set_title(_RECOVERY_METRIC_HEADERS.get(metric, metric), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units / prune span")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            ax.set_xticks([0] + spans)
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{model_name} / mean: lower and recover strategies vs upper",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = model_dir / "eval" / "recover"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean_recover_strategies_by_span.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _recovery_plot_rows(
    root: Path,
    plot_group_name: str,
    recovery_view_name: str,
    model_name: str,
) -> Dict[str, List[Tuple[str, str]]]:
    _ = (plot_group_name, recovery_view_name)
    rows: Dict[str, List[Tuple[str, str]]] = {}
    mean_path = _generate_mean_recovery_plot(root, model_name)
    if mean_path is not None and mean_path.is_file():
        rows["mean"] = [(str(mean_path.resolve()), "mean lower + recovery")]
    for dataset_tag in _recovery_dataset_tags(root, model_name):
        rows[dataset_tag] = _recovery_dataset_gallery_item(
            root,
            model_name,
            dataset_tag,
            f"{dataset_tag.replace('_', ' ')} lower + recovery",
        )
    return rows


def _load_image_if_exists(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = plt.imread(path)
    except Exception:
        return None
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if isinstance(arr, np.ndarray) and arr.dtype != np.uint8:
        arr = (np.asarray(arr, dtype=np.float32) * 255.0).clip(0, 255).round().astype(np.uint8)
    return np.asarray(arr, dtype=np.uint8)


def _stage2_row_visibility_flags(
    mode_order: List[str],
    selected_modes: List[str],
    metric_rows: List[str],
    token_rows: List[str],
    skip_modes: set[str],
    similarity_regions_by_mode: Dict[str, set[str]],
) -> List[bool]:
    active_modes = {str(m) for m in selected_modes}
    metrics = {str(m) for m in metric_rows}
    tokens = {str(t) for t in token_rows}
    out: List[bool] = []
    for mode in mode_order:
        on = mode in active_modes
        has_skip = mode in skip_modes
        sim_regions = similarity_regions_by_mode.get(mode, set())
        out.append(on and has_skip and STAGE2_METRIC_ROW_POSE in metrics)
        out.append(on and has_skip and STAGE2_METRIC_ROW_DEPTH in metrics)
        out.append(on and has_skip and STAGE2_METRIC_ROW_POINT in metrics)
        out.append(on and STAGE2_TOKEN_ROW_SPECIAL in tokens and "special" in sim_regions)
        out.append(on and STAGE2_TOKEN_ROW_CAMERA in tokens and "camera" in sim_regions)
        out.append(on and STAGE2_TOKEN_ROW_PATCH in tokens and "patch" in sim_regions)
        out.append(on and STAGE2_TOKEN_ROW_ALL in tokens and "all" in sim_regions)
        out.append(on and STAGE2_TOKEN_ROW_MEAN in tokens and "mean" in sim_regions)
        out.append(on and STAGE2_TOKEN_ROW_MIN in tokens and "min" in sim_regions)
    return out


def _stage2_block_keys_for_mode(mode: str) -> Tuple[str, str, str, str, str, str, str, str, str]:
    return (
        f"skip_{mode}_main",
        f"skip_{mode}_depth",
        f"skip_{mode}_point",
        f"sim_{mode}_special",
        f"sim_{mode}_camera",
        f"sim_{mode}_patch",
        f"sim_{mode}_all",
        f"sim_{mode}_mean",
        f"sim_{mode}_min",
    )


def _stage2_requested_block_keys(
    mode_order: List[str],
    selected_modes: List[str],
    metric_rows: List[str],
    token_rows: List[str],
    skip_modes: set[str],
    similarity_regions_by_mode: Dict[str, set[str]],
) -> set[str]:
    active_modes = {str(m) for m in selected_modes}
    metrics = {str(m) for m in metric_rows}
    tokens = {str(t) for t in token_rows}
    requested: set[str] = set()
    for mode in mode_order:
        if mode not in active_modes:
            continue
        skip_main, skip_depth, skip_point, sim_special, sim_camera, sim_patch, sim_all, sim_mean, sim_min = _stage2_block_keys_for_mode(mode)
        has_skip = mode in skip_modes
        sim_regions = similarity_regions_by_mode.get(mode, set())
        if has_skip and STAGE2_METRIC_ROW_POSE in metrics:
            requested.add(skip_main)
        if has_skip and STAGE2_METRIC_ROW_DEPTH in metrics:
            requested.add(skip_depth)
        if has_skip and STAGE2_METRIC_ROW_POINT in metrics:
            requested.add(skip_point)
        if STAGE2_TOKEN_ROW_SPECIAL in tokens and "special" in sim_regions:
            requested.add(sim_special)
        if STAGE2_TOKEN_ROW_CAMERA in tokens and "camera" in sim_regions:
            requested.add(sim_camera)
        if STAGE2_TOKEN_ROW_PATCH in tokens and "patch" in sim_regions:
            requested.add(sim_patch)
        if STAGE2_TOKEN_ROW_ALL in tokens and "all" in sim_regions:
            requested.add(sim_all)
        if STAGE2_TOKEN_ROW_MEAN in tokens and "mean" in sim_regions:
            requested.add(sim_mean)
        if STAGE2_TOKEN_ROW_MIN in tokens and "min" in sim_regions:
            requested.add(sim_min)
    return requested


def _coerce_packed(values: Any) -> Optional[List[float]]:
    if not isinstance(values, list):
        return None
    out: List[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            out.append(float("nan"))
    return out


def _canonical_skip_mode(mode: str) -> str:
    key = str(mode).strip()
    if key == "patch_special_bipartite":
        return "patch_special"
    return key


def _extract_skip_metric_maps_by_mode(report: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    prefer_pct = str(report.get("skip_value_mode", "")).strip().lower() == "pct"

    fusion_by_mode = report.get("fusion_by_mode")
    if isinstance(fusion_by_mode, dict):
        for mode, mode_payload in fusion_by_mode.items():
            if not isinstance(mode_payload, dict):
                continue
            metric_map = None
            if prefer_pct:
                metric_map = mode_payload.get("gt_degrade_pct_upper_by_metric")
                if not isinstance(metric_map, dict):
                    metric_map = mode_payload.get("gt_delta_upper_by_metric")
            else:
                metric_map = mode_payload.get("gt_delta_upper_by_metric")
                if not isinstance(metric_map, dict):
                    metric_map = mode_payload.get("gt_degrade_pct_upper_by_metric")
            if not isinstance(metric_map, dict):
                continue
            clean = {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items())
                if packed is not None
            }
            if clean:
                out[_canonical_skip_mode(str(mode))] = clean

    by_mode = report.get("global_skip_range_loss_by_mode")
    if isinstance(by_mode, dict):
        for mode, payload in by_mode.items():
            if not isinstance(payload, dict):
                continue
            metric_map = payload.get("skip_range_loss_upper_by_metric")
            if not isinstance(metric_map, dict):
                # Some legacy reports used this alternative key name.
                metric_map = payload.get("global_skip_range_loss_upper_by_metric")
            if not isinstance(metric_map, dict):
                continue
            clean = {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items())
                if packed is not None
            }
            if clean:
                out[_canonical_skip_mode(str(mode))] = clean

    by_mode = report.get("global_skip_range_loss_upper_by_metric_by_mode")
    if isinstance(by_mode, dict):
        for mode, metric_map in by_mode.items():
            if not isinstance(metric_map, dict):
                continue
            clean = {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items())
                if packed is not None
            }
            if clean:
                out[_canonical_skip_mode(str(mode))] = clean

    single = report.get("global_skip_range_loss")
    if isinstance(single, dict):
        mode = _canonical_skip_mode(str(single.get("skipped_global_mode", "")).strip())
        metric_map = single.get("skip_range_loss_upper_by_metric")
        if not isinstance(metric_map, dict):
            metric_map = single.get("global_skip_range_loss_upper_by_metric")
        if mode and isinstance(metric_map, dict):
            clean = {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items())
                if packed is not None
            }
            if clean:
                out[mode] = clean

    single_mode = _canonical_skip_mode(str(report.get("skipped_global_mode", "")).strip())
    single_metric_map = report.get("global_skip_range_loss_upper_by_metric")
    if single_mode and isinstance(single_metric_map, dict):
        clean = {
            str(k): packed
            for k, packed in ((k, _coerce_packed(v)) for k, v in single_metric_map.items())
            if packed is not None
        }
        if clean:
            out[single_mode] = clean

    return out


def _extract_similarity_metric_maps_by_mode(report: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    fusion_by_mode = report.get("fusion_by_mode")
    if isinstance(fusion_by_mode, dict):
        for mode, mode_payload in fusion_by_mode.items():
            if not isinstance(mode_payload, dict):
                continue
            patch_map = mode_payload.get("global_patch_similarity_upper_by_metric")
            special_map = mode_payload.get("global_special_similarity_upper_by_metric")
            camera_map = mode_payload.get("global_camera_similarity_upper_by_metric")
            all_map = mode_payload.get("global_all_similarity_upper_by_metric")
            patch_clean = (
                {
                    str(k): packed
                    for k, packed in ((k, _coerce_packed(v)) for k, v in patch_map.items())
                    if packed is not None
                }
                if isinstance(patch_map, dict)
                else {}
            )
            special_clean = (
                {
                    str(k): packed
                    for k, packed in ((k, _coerce_packed(v)) for k, v in special_map.items())
                    if packed is not None
                }
                if isinstance(special_map, dict)
                else {}
            )
            camera_clean = (
                {
                    str(k): packed
                    for k, packed in ((k, _coerce_packed(v)) for k, v in camera_map.items())
                    if packed is not None
                }
                if isinstance(camera_map, dict)
                else {}
            )
            all_clean = (
                {
                    str(k): packed
                    for k, packed in ((k, _coerce_packed(v)) for k, v in all_map.items())
                    if packed is not None
                }
                if isinstance(all_map, dict)
                else {}
            )
            if patch_clean or special_clean or camera_clean or all_clean:
                out[str(mode)] = {
                    "patch": patch_clean,
                    "special": special_clean,
                    "camera": camera_clean,
                    "all": all_clean,
                }

    # Legacy fallback: single-mode top-level similarity maps.
    legacy_patch = report.get("global_patch_similarity_upper")
    legacy_special = report.get("global_special_similarity_upper")
    if isinstance(legacy_patch, dict) or isinstance(legacy_special, dict):
        patch_clean = (
            {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in legacy_patch.items())
                if packed is not None
            }
            if isinstance(legacy_patch, dict)
            else {}
        )
        special_clean = (
            {
                str(k): packed
                for k, packed in ((k, _coerce_packed(v)) for k, v in legacy_special.items())
                if packed is not None
            }
            if isinstance(legacy_special, dict)
            else {}
        )
        if patch_clean or special_clean:
            out.setdefault("default", {"patch": patch_clean, "special": special_clean, "all": {}})

    return out


def _discover_skip_modes(skip_reports: Dict[str, Stage2SceneReport]) -> List[str]:
    ordered: List[str] = []

    def _add(mode: str) -> None:
        key = str(mode).strip()
        if key and not key.endswith("_nomlp") and key not in ordered:
            ordered.append(key)

    mean_report = skip_reports.get("mean")
    if mean_report is not None:
        for mode in mean_report.report.get("fusion_modes", []):
            _add(str(mode))
        for mode in mean_report.report.get("skipped_global_modes", []):
            _add(str(mode))

    report_items = [("mean", mean_report)] if mean_report is not None else []
    report_items.extend((sid, rep) for sid, rep in skip_reports.items() if sid != "mean")
    for _sid, scene_report in report_items:
        if scene_report is None:
            continue
        for mode in _extract_skip_metric_maps_by_mode(scene_report.report).keys():
            _add(mode)
    return ordered


def _discover_similarity_modes(similarity_reports: Dict[str, Stage2SceneReport]) -> List[str]:
    ordered: List[str] = []

    def _add(mode: str) -> None:
        key = str(mode).strip()
        if key and key not in ordered:
            ordered.append(key)

    report_items = [("mean", similarity_reports.get("mean"))]
    report_items.extend((sid, rep) for sid, rep in similarity_reports.items() if sid != "mean")
    for _sid, scene_report in report_items:
        if scene_report is None:
            continue
        by_mode = _extract_similarity_metric_maps_by_mode(scene_report.report)
        for mode in by_mode.keys():
            _add(mode)
    return ordered


def _discover_stage2_modes(
    similarity_reports: Dict[str, Stage2SceneReport],
    skip_reports: Dict[str, Stage2SceneReport],
) -> List[str]:
    """Union of skip/similarity modes, preserving stable discovery order."""
    out: List[str] = []
    for mode in _discover_skip_modes(skip_reports):
        if mode not in out:
            out.append(mode)
    for mode in _discover_similarity_modes(similarity_reports):
        if mode not in out:
            out.append(mode)
    return out


def _mean_stage1_baseline_for_metric(
    stage1_by_scene: Dict[str, Dict[str, Any]],
    metric_key: str,
    scene_ids: Optional[Sequence[str]],
) -> Optional[float]:
    if scene_ids:
        ids = [str(s).strip() for s in scene_ids if str(s).strip()]
    else:
        ids = list(stage1_by_scene.keys())
    vals: List[float] = []
    for sid in ids:
        row = stage1_by_scene.get(sid)
        if row is None or metric_key not in row:
            continue
        try:
            v = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _similarity_maps_for_skip_mode(
    similarity_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]],
    skip_mode: str,
) -> Dict[str, Dict[str, List[float]]]:
    if skip_mode in similarity_by_mode:
        return similarity_by_mode[skip_mode]
    if "default" in similarity_by_mode:
        return similarity_by_mode["default"]
    return {"patch": {}, "special": {}, "camera": {}, "all": {}}


def _similarity_region_has_data(
    sim_entry: Dict[str, Dict[str, List[float]]],
    region: str,
) -> bool:
    region_map = sim_entry.get(region)
    if not isinstance(region_map, dict):
        return False
    return any(isinstance(packed, list) and len(packed) > 0 for packed in region_map.values())


def _missing_similarity_region_note(
    controller: PairAttentionController,
    *,
    scene_id: str,
    mode: str,
    region: str,
) -> str:
    label = "all-token" if region == "all" else f"{region}-token"
    note = f"missing {label} similarity for mode={mode}"
    from analysis.redundancy.similarity_v3 import redundancy_graph_filename

    note += f" ({controller.analysis_similarity_dir / redundancy_graph_filename(region)})"
    return f"{scene_id}: {note}"


def _packed_delta_to_pct(
    packed: List[float],
    baseline_value: float,
    *,
    metric_key: str,
    clip_percent: float,
    eps: float = 1e-12,
) -> List[float]:
    denom = max(abs(float(baseline_value)), eps)
    clip = float(np.clip(float(clip_percent), 0.0, 100.0))
    lower = -clip
    out: List[float] = []
    for value in packed:
        scalar = float(value)
        if math.isnan(scalar):
            out.append(float("nan"))
            continue
        degradation = -scalar if metric_key in SKIP_HIGHER_IS_BETTER else scalar
        pct = 100.0 * degradation / denom
        out.append(float(max(pct, lower)))
    return out


def _heatmap_rgb_from_packed(
    packed: List[float],
    num_layers: int,
    title: str,
    *,
    fill_value: float = float("nan"),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    tick_suffix: str = "",
    cmap: str = "viridis",
    markers: Optional[List[Tuple[int, int, str]]] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
) -> np.ndarray:
    grid = upper_triangle_unpack_for_heatmap(packed, num_layers, fill_value)
    arr = np.asarray(grid, dtype=np.float64)
    finite_vals = arr[np.isfinite(arr)]
    if np.any(np.isnan(arr)):
        arr = np.ma.masked_invalid(arr)

    title_fs, label_fs, tick_fs = 18, 16, 14
    if label_fontsize is not None:
        label_fs = float(label_fontsize)
    if tick_fontsize is not None:
        tick_fs = float(tick_fontsize)
    fig, ax = plt.subplots(figsize=(9, 8))
    imshow_kw: Dict[str, Any] = {"cmap": cmap}
    if vmin is not None:
        imshow_kw["vmin"] = vmin
    if vmax is not None:
        imshow_kw["vmax"] = vmax
    im = imshow_upper_triangle(ax, arr, num_layers, **imshow_kw)
    if str(title or "").strip():
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=12)
    configure_triangle_axes(
        ax,
        num_layers,
        index_name="layer",
        label_fontsize=label_fs,
        label_fontweight="bold",
        tick_fontsize=tick_fs,
        tick_width=2.0,
        tick_length=6,
        bold_tick_labels=True,
        use_major_locator=True,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fs, width=2.0, length=6)
    for lb in cbar.ax.get_yticklabels():
        lb.set_fontweight("bold")
    if markers:
        style_map = {
            "0": dict(marker="o", s=140, c="red", edgecolors="white", linewidths=2.0),
            "1": dict(marker="^", s=160, c="orange", edgecolors="white", linewidths=2.0),
            "2": dict(marker="s", s=160, c="magenta", edgecolors="white", linewidths=2.0),
            "3": dict(marker="D", s=160, c="cyan", edgecolors="white", linewidths=2.0),
        }
        for (mi, mj, tag) in markers:
            kw = style_map.get(str(tag), style_map["0"])
            x, y = triangle_xy(mi, mj)
            ax.scatter([x], [y], **kw)
    data_min = float(vmin) if vmin is not None else (float(finite_vals.min()) if finite_vals.size else None)
    data_max = float(vmax) if vmax is not None else (float(finite_vals.max()) if finite_vals.size else None)
    if data_min is not None and data_max is not None:
        if abs(data_max - data_min) < 1e-12:
            cbar.set_ticks([data_min])
        else:
            cbar.set_ticks(np.linspace(data_min, data_max, num=5).tolist())
        if tick_suffix:
            ticks = list(cbar.get_ticks())
            labels: List[str] = []
            for tick in ticks:
                tick_val = float(tick)
                if abs(tick_val - round(tick_val)) < 1e-9:
                    labels.append(f"{int(round(tick_val))}{tick_suffix}")
                else:
                    labels.append(f"{tick_val:.1f}{tick_suffix}")
            cbar.set_ticklabels(labels)
            for lb in cbar.ax.get_yticklabels():
                lb.set_fontweight("bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    plt.close(fig)
    buf.seek(0)
    arr = plt.imread(buf)
    buf.close()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return (np.asarray(arr, dtype=np.float32) * 255.0).round().astype(np.uint8)


def _sim_gamma_key(sim_gamma: float) -> int:
    return int(round(max(float(sim_gamma), 1e-3) * 1000.0))


def _sim_cutoff_key(sim_cutoff: float) -> int:
    return int(round(float(np.clip(sim_cutoff, 0.0, 1.0)) * 1000.0))


def _similarity_packed_gamma_01(
    packed: List[float],
    sim_gamma: float,
    *,
    eps: float = 1e-6,
) -> List[float]:
    gamma = max(float(sim_gamma), eps)
    if abs(gamma - 1.0) <= 1e-12:
        return packed
    out: List[float] = []
    for value in packed:
        scalar = float(value)
        if math.isnan(scalar):
            out.append(float("nan"))
            continue
        x = float(np.clip(scalar, 0.0, 1.0))
        y = math.pow(x, gamma)
        out.append(float(y))
    return out


def _packed_elementwise_min(packs: Sequence[List[float]]) -> List[float]:
    if not packs:
        return []
    length = max(len(p) for p in packs)
    out: List[float] = []
    for idx in range(length):
        vals: List[float] = []
        for packed in packs:
            if idx >= len(packed):
                continue
            scalar = float(packed[idx])
            if math.isnan(scalar):
                continue
            vals.append(scalar)
        out.append(float(min(vals)) if vals else float("nan"))
    return out


def _packed_elementwise_mean(packs: Sequence[List[float]]) -> List[float]:
    if not packs:
        return []
    length = max(len(p) for p in packs)
    out: List[float] = []
    for idx in range(length):
        vals: List[float] = []
        for packed in packs:
            if idx >= len(packed):
                continue
            scalar = float(packed[idx])
            if math.isnan(scalar):
                continue
            vals.append(scalar)
        out.append(float(sum(vals) / len(vals)) if vals else float("nan"))
    return out


def _similarity_maps_with_token_aggregations(
    similarity_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Add token-region mean/min maps over every available token region."""
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for mode, raw_entry in similarity_by_mode.items():
        entry = {str(region): dict(metric_map) for region, metric_map in raw_entry.items()}
        available_regions = [
            region
            for region in REDUNDANCY_GRAPH_REGIONS
            if _similarity_region_has_data(entry, region)
        ]
        metric_keys = {
            metric_key
            for region in available_regions
            for metric_key in entry.get(region, {}).keys()
        }
        mean_map: Dict[str, List[float]] = {}
        min_map: Dict[str, List[float]] = {}
        for metric_key in metric_keys:
            packs = [
                entry[region][metric_key]
                for region in available_regions
                if metric_key in entry.get(region, {})
            ]
            if packs:
                mean_map[metric_key] = _packed_elementwise_mean(packs)
                min_map[metric_key] = _packed_elementwise_min(packs)
        if mean_map:
            entry[STAGE2_TOKEN_ROW_MEAN] = mean_map
            entry[STAGE2_TOKEN_ROW_MIN] = min_map
        out[str(mode)] = entry
    return out


def _similarity_packed_cutoff_01(
    packed: List[float],
    sim_cutoff: float,
) -> List[float]:
    cutoff = float(np.clip(sim_cutoff, 0.0, 1.0))
    if cutoff <= 0.0:
        return packed
    out: List[float] = []
    for value in packed:
        scalar = float(value)
        if math.isnan(scalar):
            out.append(float("nan"))
            continue
        out.append(0.0 if scalar < cutoff else scalar)
    return out


def _signed_packed_cutoff(
    packed: List[float],
    cutoff_value: float,
) -> List[float]:
    """Zero values close to zero while preserving significant values of either sign."""
    cutoff = float(np.clip(cutoff_value, 0.0, 1.0))
    if cutoff <= 0.0:
        return packed
    out: List[float] = []
    for value in packed:
        scalar = float(value)
        if math.isnan(scalar):
            out.append(float("nan"))
            continue
        out.append(0.0 if abs(scalar) < cutoff else scalar)
    return out


def _similarity_metric_entries(similarity_upper: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Metrics present in this report (subset of :data:`SIMILARITY_METRICS`)."""
    available = {str(k) for k in similarity_upper.keys()}
    return [(k, label) for k, label in SIMILARITY_METRICS if k in available]


def _build_cross_dataset_similarity_all(
    controller: PairAttentionController,
    mode_order: List[str],
    *,
    sim_gamma: float,
    sim_cutoff: float,
    aggregate: str,
) -> Dict[str, Dict[str, List[float]]]:
    """Per skip mode: all-token similarity packed vectors aggregated over real datasets (scene mean)."""
    per_dataset_maps: List[Dict[str, Dict[str, Dict[str, List[float]]]]] = []
    for ds in getattr(controller, "stage2_real_datasets", []):
        report = controller.get_similarity_report_for_dataset(ds, "mean")
        if report is None:
            continue
        per_dataset_maps.append(_extract_similarity_metric_maps_by_mode(report.report))

    if not per_dataset_maps:
        return {}
    agg_key = str(aggregate or STAGE2_MIN_DATASET).strip().lower()
    reducer = _packed_elementwise_mean if agg_key == STAGE2_MEAN_DATASET else _packed_elementwise_min

    out: Dict[str, Dict[str, List[float]]] = {}
    for mode in mode_order:
        metric_packs: Dict[str, List[List[float]]] = {}
        for by_mode in per_dataset_maps:
            all_map = by_mode.get(mode, {}).get("all", {})
            if not isinstance(all_map, dict):
                continue
            for metric_key, packed in all_map.items():
                if not packed:
                    continue
                processed = list(packed)
                if metric_key in {"cka", "cosine"}:
                    if abs(sim_gamma - 1.0) > 1e-12:
                        processed = _similarity_packed_gamma_01(processed, sim_gamma)
                    if sim_cutoff > 0.0:
                        processed = _similarity_packed_cutoff_01(processed, sim_cutoff)
                metric_packs.setdefault(str(metric_key), []).append(processed)
        if metric_packs:
            out[mode] = {k: reducer(v) for k, v in metric_packs.items()}
    return out


def _build_cross_dataset_min_similarity_all(
    controller: PairAttentionController,
    mode_order: List[str],
    *,
    sim_gamma: float,
    sim_cutoff: float,
) -> Dict[str, Dict[str, List[float]]]:
    """Backward-compatible wrapper for the virtual ``min`` dataset."""
    return _build_cross_dataset_similarity_all(
        controller,
        mode_order,
        sim_gamma=sim_gamma,
        sim_cutoff=sim_cutoff,
        aggregate=STAGE2_MIN_DATASET,
    )


def _metric_entries_for_similarity_maps(
    similarity_upper: Dict[str, Any],
    *,
    report_metrics: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    if report_metrics:
        keys = {str(m) for m in report_metrics}
        return [(k, lbl) for k, lbl in SIMILARITY_METRICS if k in keys]
    keys = {str(k) for k in similarity_upper.keys()}
    return [(k, lbl) for k, lbl in SIMILARITY_METRICS if k in keys]


def _render_similarity_gallery(
    similarity_upper: Dict[str, Any],
    num_layers: int,
    *,
    role_short: str,
    sim_gamma: float,
    sim_cutoff: float,
    metric_entries: Optional[List[Tuple[str, str]]] = None,
    prune_overlay: bool = False,
    prune_metric_key: str = "cka",
    prune_n: int = 2,
) -> List[np.ndarray]:
    """role_short: one-letter tag, e.g. S (special) / P (patch) / A (all tokens)."""
    images: List[np.ndarray] = []
    tag = (role_short or "?").strip()[:1].upper()
    entries = metric_entries if metric_entries is not None else _metric_entries_for_similarity_maps(similarity_upper)
    if not entries:
        entries = list(SIMILARITY_METRICS)

    overlay_markers: Optional[List[Tuple[int, int, str]]] = None
    prune_key = str(prune_metric_key or "").strip().lower()
    if prune_overlay and prune_key in {"cka", "cosine"}:
        packed_for_prune = _coerce_packed(similarity_upper.get(prune_key))
        if packed_for_prune is not None:
            processed = list(packed_for_prune)
            if abs(sim_gamma - 1.0) > 1e-12:
                processed = _similarity_packed_gamma_01(processed, sim_gamma)
            triangle_thresh = float(sim_cutoff) if float(sim_cutoff) > 1e-9 else 0.9
            for_triangle = (
                _similarity_packed_cutoff_01(processed, sim_cutoff)
                if float(sim_cutoff) > 1e-9
                else processed
            )
            grid = upper_triangle_unpack_for_heatmap(for_triangle, num_layers, float("nan"))
            sim_mat = np.zeros((num_layers, num_layers), dtype=np.float64)
            for i in range(num_layers):
                sim_mat[i, i] = 1.0
                for j in range(i + 1, num_layers):
                    v = grid[i][j]
                    if v == v:
                        sim_mat[i, j] = float(v)
            from analysis.prune_plan.triangle_prune import greedy_dense_triangles

            picks, _ = greedy_dense_triangles(
                sim_mat,
                unit_ids=[f"u{i}" for i in range(num_layers)],
                threshold=triangle_thresh,
                max_triangles=int(max(0, prune_n)),
            )
            overlay_markers = [(p.i, p.j, str(k)) for k, p in enumerate(picks)]
    for metric_key, metric_label in entries:
        packed = _coerce_packed(similarity_upper.get(metric_key))
        if packed is None:
            continue
        if metric_key in {"cka", "cosine"}:
            if abs(sim_gamma - 1.0) > 1e-12:
                packed = _similarity_packed_gamma_01(packed, sim_gamma)
            if sim_cutoff > 0.0:
                packed = _similarity_packed_cutoff_01(packed, sim_cutoff)
        vmin = 0.0 if metric_key != "neg_mse" else None
        vmax = 1.0 if metric_key != "neg_mse" else None
        markers = overlay_markers if overlay_markers and str(metric_key).lower() == prune_key else None
        images.append(
            _heatmap_rgb_from_packed(
                packed,
                num_layers,
                f"{tag}·{metric_label}",
                vmin=vmin,
                vmax=vmax,
                markers=markers,
            )
        )
    return images


_CKA_TOKEN_REGION_ORDER: Tuple[str, ...] = ("camera", "special", "all", "patch", "mean", "min")


def _discover_cka_token_regions(
    controller: PairAttentionController,
    datasets: Sequence[str],
    models: Sequence[str],
) -> List[str]:
    seen: set[str] = set()
    discovered: List[str] = []

    for dataset in datasets:
        for backbone in models:
            report_entry = controller.get_similarity_report_for_dataset(
                str(dataset),
                "mean",
                backbone=str(backbone),
            )
            if report_entry is None:
                continue
            similarity_by_mode = _extract_similarity_metric_maps_by_mode(report_entry.report)
            for mode_maps in similarity_by_mode.values():
                if not isinstance(mode_maps, dict):
                    continue
                for region_name, region_map in mode_maps.items():
                    key = str(region_name or "").strip().lower()
                    if not key or key in seen:
                        continue
                    if not isinstance(region_map, dict):
                        continue
                    if _coerce_packed(region_map.get("cka")) is None:
                        continue
                    seen.add(key)
                    discovered.append(key)

    def _sort_key(name: str) -> Tuple[int, str]:
        try:
            return (_CKA_TOKEN_REGION_ORDER.index(name), name)
        except ValueError:
            return (len(_CKA_TOKEN_REGION_ORDER), name)

    discovered = sorted(discovered, key=_sort_key)
    if discovered:
        for agg in ("mean", "min"):
            if agg not in discovered:
                discovered.append(agg)
        discovered = sorted(discovered, key=_sort_key)
    return discovered


def _cka_detail_from_provenance(provenance: Dict[str, object]) -> str:
    dataset_sel = str(provenance.get("dataset_selector", "") or "")
    token_sel = str(provenance.get("token_region_selector", "") or "")
    datasets = provenance.get("datasets", [])
    regions_by_dataset = provenance.get("token_regions_by_dataset", {})
    used_regions = sorted(
        {
            str(region)
            for region_list in regions_by_dataset.values()
            if isinstance(region_list, list)
            for region in region_list
        }
    )
    parts = [f"dataset={dataset_sel}", f"token={token_sel}"]
    if used_regions:
        parts.append(f"regions={','.join(used_regions)}")
    if isinstance(datasets, list) and datasets:
        parts.append(f"datasets={','.join(str(dataset) for dataset in datasets)}")
    return " | ".join(parts)


def _render_cka_heatmap_from_packed(
    packed: List[float],
    *,
    num_layers: int,
    sim_cutoff: float,
) -> np.ndarray:
    cutoff = float(np.clip(float(sim_cutoff), 0.0, 1.0))
    if cutoff > 0.0:
        packed = _similarity_packed_cutoff_01(packed, cutoff)
    return _heatmap_rgb_from_packed(
        packed,
        num_layers,
        "",
        vmin=0.0,
        vmax=1.0,
        label_fontsize=22,
        tick_fontsize=20,
    )


def _render_cka_heatmap_for_backbone(
    *,
    backbone: str,
    dataset: str,
    token_region: str,
    sim_cutoff: float,
) -> Tuple[Optional[np.ndarray], str]:
    region_key = str(token_region or "").strip().lower()
    try:
        unit_ids, sim, provenance = load_similarity(
            experiment=str(backbone),
            dataset=str(dataset),
            token_region=region_key,
            metric="cka",
        )
    except FileNotFoundError as exc:
        return None, str(exc)
    except (KeyError, ValueError) as exc:
        return None, str(exc)

    packed = upper_triangle_pack(torch.from_numpy(sim))
    image = _render_cka_heatmap_from_packed(
        packed,
        num_layers=len(unit_ids),
        sim_cutoff=sim_cutoff,
    )
    return image, _cka_detail_from_provenance(provenance)


def _render_skip_gallery(
    skip_metric_map: Dict[str, List[float]],
    num_layers: int,
    metrics: List[Tuple[str, str]],
    *,
    value_mode: str,
    sim_cutoff: float = 0.0,
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for metric_key, metric_label in metrics:
        packed = skip_metric_map.get(metric_key)
        if packed is None:
            continue
        degradation_packed = (
            [-float(value) for value in packed]
            if value_mode == "absolute" and metric_key in SKIP_HIGHER_IS_BETTER
            else packed
        )
        if sim_cutoff > 0.0:
            degradation_packed = _signed_packed_cutoff(
                degradation_packed,
                sim_cutoff,
            )
        if value_mode == "pct":
            title = f"{metric_label} degradation %"
        else:
            title = f"{metric_label} degradation Δ"
        images.append(
            _heatmap_rgb_from_packed(
                degradation_packed,
                num_layers,
                title,
                tick_suffix="%" if value_mode == "pct" else "",
                cmap="viridis_r",
            )
        )
    return images


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb


def _plot_overlay(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _frame_patch_slice(frame_index: int, *, tokens_per_frame: int, patch_start_idx: int) -> slice:
    start = frame_index * tokens_per_frame + patch_start_idx
    stop = (frame_index + 1) * tokens_per_frame
    return slice(start, stop)


def _compute_directional_overlay_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_q_idx: int,
    frame_k_idx: int,
    tokens_per_frame: int,
    patch_start_idx: int,
    score_space: str,
    head_mode: str,
    head_index: int,
) -> Dict[str, np.ndarray]:
    q_slice = _frame_patch_slice(frame_q_idx, tokens_per_frame=tokens_per_frame, patch_start_idx=patch_start_idx)
    k_slice = _frame_patch_slice(frame_k_idx, tokens_per_frame=tokens_per_frame, patch_start_idx=patch_start_idx)
    q_sel = q[0, :, q_slice, :]
    k_sel = k[0, :, k_slice, :]
    score_per_head = torch.matmul(q_sel, k_sel.transpose(-2, -1))
    if score_space == "after_softmax":
        score_per_head = torch.softmax(score_per_head, dim=-1)
    elif score_space != "before_softmax":
        raise gr.Error(f"Unknown score space: {score_space}")

    if head_mode == "mean":
        score = score_per_head.mean(dim=0)
    elif head_mode == "single":
        head_index = max(0, min(int(head_index), int(score_per_head.shape[0]) - 1))
        score = score_per_head[head_index]
    else:
        raise gr.Error(f"Unknown head mode: {head_mode}")

    overlays: Dict[str, np.ndarray] = {
        "src_max": score.max(dim=1).values.detach().cpu().float().numpy(),
        "tgt_max": score.max(dim=0).values.detach().cpu().float().numpy(),
    }
    if score_space == "before_softmax":
        overlays["src_mean"] = score.mean(dim=1).detach().cpu().float().numpy()
        overlays["tgt_mean"] = score.mean(dim=0).detach().cpu().float().numpy()
    return overlays


def _render_panel_array(
    *,
    scene_id: str,
    layer_idx: int,
    frame_a_index: int,
    frame_b_index: int,
    frame_a_id: str,
    frame_b_id: str,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    maps_ab: Dict[str, np.ndarray],
    maps_ba: Dict[str, np.ndarray],
    patch_hw: Tuple[int, int],
    score_space: str,
    head_mode: str,
    head_index: int | None,
) -> np.ndarray:
    if score_space == "before_softmax":
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    score_title = "after softmax" if score_space == "after_softmax" else "before softmax"
    head_title = "head mean" if head_mode == "mean" else f"head {head_index}"
    fig.suptitle(
        f"{scene_id} | layer {layer_idx} | {score_title} | {head_title} | "
        f"A=index {frame_a_index} (id={frame_a_id}) | B=index {frame_b_index} (id={frame_b_id})"
    )

    _plot_overlay(axes[0, 0], _overlay_patch_map(image_a, maps_ab["src_max"], patch_hw=patch_hw), "A as q: max over B patches")
    _plot_overlay(axes[0, 1], _overlay_patch_map(image_b, maps_ab["tgt_max"], patch_hw=patch_hw), "B as k: max over A queries")
    _plot_overlay(axes[1, 0], _overlay_patch_map(image_b, maps_ba["src_max"], patch_hw=patch_hw), "B as q: max over A patches")
    _plot_overlay(axes[1, 1], _overlay_patch_map(image_a, maps_ba["tgt_max"], patch_hw=patch_hw), "A as k: max over B queries")

    if score_space == "before_softmax":
        _plot_overlay(axes[0, 2], _overlay_patch_map(image_a, maps_ab["src_mean"], patch_hw=patch_hw), "A as q: mean over B patches")
        _plot_overlay(axes[0, 3], _overlay_patch_map(image_b, maps_ab["tgt_mean"], patch_hw=patch_hw), "B as k: mean over A queries")
        _plot_overlay(axes[1, 2], _overlay_patch_map(image_b, maps_ba["src_mean"], patch_hw=patch_hw), "B as q: mean over A patches")
        _plot_overlay(axes[1, 3], _overlay_patch_map(image_a, maps_ba["tgt_mean"], patch_hw=patch_hw), "A as k: mean over B queries")
    fig.tight_layout()
    return _figure_to_rgb(fig)


def _build_stage2_skip_maps(
    controller: PairAttentionController,
    scene_id: str,
    value_mode: str,
    pct_clip_percent: float,
) -> Tuple[Dict[str, Dict[str, List[float]]], str]:
    skip_report_entry = controller.get_skip_report(scene_id)
    if skip_report_entry is None:
        return {}, "skip report missing"
    report = dict(skip_report_entry.report)
    report["skip_value_mode"] = "pct" if value_mode != "absolute" else "absolute"
    skip_by_mode = _extract_skip_metric_maps_by_mode(report)
    if value_mode == "absolute":
        return skip_by_mode, "skip rows use saved absolute delta"

    fusion_by_mode = report.get("fusion_by_mode")
    if isinstance(fusion_by_mode, dict):
        has_pct = any(
            isinstance(payload, dict) and isinstance(payload.get("gt_degrade_pct_upper_by_metric"), dict)
            for payload in fusion_by_mode.values()
        )
        if has_pct:
            return skip_by_mode, "skip rows use saved pct degrade vs upper"

    if scene_id == "mean":
        if not skip_by_mode:
            raise gr.Error(
                "Mean skip view needs aggregated skip data in full_report_skip.json "
                "(e.g. fusion_by_mode / gt_delta_upper_by_metric)."
            )
        raw_scene_ids = report.get("scene_ids")
        scene_ids_list: Optional[List[str]] = None
        if isinstance(raw_scene_ids, list) and raw_scene_ids:
            scene_ids_list = [str(s).strip() for s in raw_scene_ids if str(s).strip()]
        try:
            n_fused = int(report.get("num_scenes", 0))
        except (TypeError, ValueError):
            n_fused = 0
        if n_fused <= 0 and scene_ids_list:
            n_fused = len(scene_ids_list)
        pct_by_mode: Dict[str, Dict[str, List[float]]] = {}
        for mode, metric_map in skip_by_mode.items():
            pct_metric_map: Dict[str, List[float]] = {}
            for metric_key, packed in metric_map.items():
                baseline = _mean_stage1_baseline_for_metric(
                    controller.stage1_eval_by_scene,
                    metric_key,
                    scene_ids_list,
                )
                if baseline is None:
                    continue
                pct_metric_map[metric_key] = _packed_delta_to_pct(
                    packed,
                    baseline,
                    metric_key=metric_key,
                    clip_percent=pct_clip_percent,
                )
            if pct_metric_map:
                pct_by_mode[mode] = pct_metric_map
        if not pct_by_mode:
            raise gr.Error(
                "Mean pct view: no skip heatmaps built. "
                f"Check {controller.cache_dir / 'eval.json'} has Stage1 metrics for "
                f"scene_ids in full_report_skip.json."
            )
        detail = f"{n_fused} scenes" if n_fused > 0 else "full_report"
        return pct_by_mode, f"mean Δ from full_report, pct+clip vs mean Stage1 ({detail})"

    baseline_row = controller.stage1_eval_by_scene.get(scene_id)
    if baseline_row is None:
        raise gr.Error(f"Missing Stage1 baseline metrics for {scene_id!r} in {controller.cache_dir / 'eval.json'}.")

    pct_by_mode: Dict[str, Dict[str, List[float]]] = {}
    for mode, metric_map in skip_by_mode.items():
        pct_metric_map: Dict[str, List[float]] = {}
        for metric_key, packed in metric_map.items():
            if metric_key not in baseline_row:
                continue
            pct_metric_map[metric_key] = _packed_delta_to_pct(
                packed,
                float(baseline_row[metric_key]),
                metric_key=metric_key,
                clip_percent=pct_clip_percent,
            )
        if pct_metric_map:
            pct_by_mode[mode] = pct_metric_map
    return pct_by_mode, "skip rows use on-the-fly pct vs Stage1 baseline"


def _normalize_prune_n(value: Any, *, default: int = 2) -> int:
    try:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("non-finite prune n")
        n = int(round(v))
    except (TypeError, ValueError):
        n = int(default)
    return max(0, min(n, 20))


def _stage2_prune_render_sig(
    *,
    prune_overlay: bool,
    prune_metric_key: str,
    prune_n: int,
    sim_cutoff: float,
) -> Tuple[bool, str, int, int]:
    return (
        bool(prune_overlay),
        str(prune_metric_key or "cka").strip().lower(),
        int(prune_n),
        _sim_cutoff_key(sim_cutoff),
    )


def _stage2_prune_status_text(
    similarity_upper: Dict[str, Any],
    *,
    num_layers: int,
    sim_gamma: float,
    sim_cutoff: float,
    prune_metric_key: str,
    prune_n: int,
    prune_overlay: bool,
) -> str:
    if not bool(prune_overlay):
        return ""
    metric_key = str(prune_metric_key or "").strip().lower()
    if metric_key not in {"cka", "cosine"}:
        return f"Prune status unavailable: metric={metric_key} not supported."
    packed = _coerce_packed(similarity_upper.get(metric_key))
    if packed is None:
        return f"Prune status unavailable: missing {metric_key} similarity."
    processed = list(packed)
    if abs(float(sim_gamma) - 1.0) > 1e-12:
        processed = _similarity_packed_gamma_01(processed, float(sim_gamma))
    threshold = float(sim_cutoff) if float(sim_cutoff) > 1e-9 else 0.9
    for_triangle = (
        _similarity_packed_cutoff_01(processed, float(sim_cutoff))
        if float(sim_cutoff) > 1e-9
        else processed
    )
    grid = upper_triangle_unpack_for_heatmap(for_triangle, int(num_layers), float("nan"))
    sim_mat = np.zeros((int(num_layers), int(num_layers)), dtype=np.float64)
    for i in range(int(num_layers)):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, int(num_layers)):
            v = grid[i][j]
            if v == v:
                sim_mat[i, j] = float(v)
    from analysis.prune_plan.triangle_prune import greedy_dense_triangles

    picks, _ = greedy_dense_triangles(
        sim_mat,
        unit_ids=[f"u{i}" for i in range(int(num_layers))],
        threshold=float(threshold),
        max_triangles=int(max(0, prune_n)),
    )
    if not picks:
        return f"No dense triangles found (metric={metric_key}, threshold={threshold:.3f})."
    lines = [f"metric={metric_key} threshold={threshold:.3f} n={int(max(0, prune_n))}"]
    for idx, p in enumerate(picks, start=1):
        start = int(p.i) + 1
        length = int(p.j) - int(p.i)
        corner = float(sim_mat[int(p.i), int(p.j)])
        lines.append(
            f"[{idx}] start={start} len={length} corner_sim={corner:.4f} (i={int(p.i)}, j={int(p.j)})"
        )
    return "\n".join(lines)


def _prepare_stage2_render(
    controller: PairAttentionController,
    scene_id: str,
    value_mode: str,
    pct_clip_percent: float,
    sim_gamma: float,
    sim_cutoff: float,
    mode_order: List[str],
    *,
    requested_blocks: Optional[set[str]] = None,
    prune_overlay: bool = False,
    prune_metric_key: str = "cka",
    prune_n: int = 2,
) -> Tuple[Dict[str, List[np.ndarray]], str, str]:
    virtual_dataset = str(controller.dataset).strip().lower() if _is_stage2_virtual_dataset(controller.dataset) else ""
    virtual_view = bool(virtual_dataset)
    if virtual_view:
        scene_id = "mean"

    clip_key = int(round(float(np.clip(pct_clip_percent, 0.0, 100.0))))
    gamma_key = _sim_gamma_key(sim_gamma)
    cutoff_key = _sim_cutoff_key(sim_cutoff)
    prune_n_i = _normalize_prune_n(prune_n)
    prune_metric = str(prune_metric_key or "cka").strip().lower()
    prune_sig = _stage2_prune_render_sig(
        prune_overlay=bool(prune_overlay),
        prune_metric_key=prune_metric,
        prune_n=prune_n_i,
        sim_cutoff=sim_cutoff,
    )
    cache_key = (scene_id, value_mode, clip_key, gamma_key, cutoff_key, virtual_dataset, *prune_sig)
    cached = controller.stage2_render_cache.get(cache_key)
    if cached is not None:
        galleries = dict(cached[0])
        info = cached[1]
        prune_status = cached[2] if len(cached) > 2 else ""
    else:
        galleries = {}
        info = ""
        prune_status = ""

    all_blocks: set[str] = set()
    for mode in mode_order:
        all_blocks.update(_stage2_block_keys_for_mode(mode))
    target_blocks = set(requested_blocks) if requested_blocks is not None else all_blocks
    if all(key in galleries for key in target_blocks) and info and not bool(prune_overlay):
        return galleries, info, prune_status

    num_layers = int(controller.resolve_stage2_num_layers(scene_id))
    if num_layers <= 0:
        label = f"dataset={virtual_dataset} (mean)" if virtual_view else scene_id
        raise gr.Error(f"No similarity report with num_layers found for {label!r}.")

    virtual_all_by_mode: Dict[str, Dict[str, List[float]]] = {}
    if virtual_view:
        if not controller.has_cross_dataset_similarity():
            ds_list = list(getattr(controller, "stage2_real_datasets", []))
            raise gr.Error(
                f"Dataset {virtual_dataset} needs full_report_similarity.json (mean) under each of "
                f"{', '.join(ds_list) if ds_list else '<no datasets discovered>'}."
            )
        virtual_all_by_mode = _build_cross_dataset_similarity_all(
            controller,
            mode_order,
            sim_gamma=sim_gamma,
            sim_cutoff=sim_cutoff,
            aggregate=virtual_dataset,
        )
        similarity_report = None
        similarity_by_mode = {}
        skip_by_mode: Dict[str, Dict[str, List[float]]] = {}
        op_label = "element-wise mean" if virtual_dataset == STAGE2_MEAN_DATASET else "element-wise min"
        skip_source = (
            f"{virtual_dataset} view: {op_label} of all-token similarity (mean) over "
            + ", ".join(list(getattr(controller, "stage2_real_datasets", [])))
        )
    else:
        similarity_report = controller.get_similarity_report(scene_id)
        similarity_by_mode = {}
        if similarity_report is not None:
            similarity_by_mode = _similarity_maps_with_token_aggregations(
                _extract_similarity_metric_maps_by_mode(similarity_report.report)
            )
        skip_by_mode, skip_source = _build_stage2_skip_maps(controller, scene_id, value_mode, float(clip_key))

    missing_similarity_notes: List[str] = []
    for mode in mode_order:
        skip_main, skip_depth, skip_point, sim_special, sim_camera, sim_patch, sim_all, sim_mean, sim_min = _stage2_block_keys_for_mode(mode)
        if not virtual_view and skip_main in target_blocks and skip_main not in galleries:
            galleries[skip_main] = _render_skip_gallery(
                skip_by_mode.get(mode, {}),
                num_layers,
                SKIP_MAIN_METRICS,
                value_mode=value_mode,
                sim_cutoff=sim_cutoff,
            )
        if virtual_view:
            all_maps = virtual_all_by_mode.get(mode, {})
            if not all_maps:
                continue
            sim_entry = {"all": all_maps, "special": {}, "patch": {}}
            report_metrics = list(all_maps.keys())
        else:
            sim_entry = _similarity_maps_for_skip_mode(similarity_by_mode, mode)
            report_metrics = list(similarity_report.report.get("metrics", [])) if similarity_report else None
        metric_entries = _metric_entries_for_similarity_maps(
            sim_entry.get("all", {}),
            report_metrics=report_metrics,
        )
        if not virtual_view and sim_special in target_blocks and sim_special not in galleries:
            if not _similarity_region_has_data(sim_entry, "special"):
                missing_similarity_notes.append(
                    _missing_similarity_region_note(
                        controller,
                        scene_id=scene_id,
                        mode=mode,
                        region="special",
                    )
                )
            galleries[sim_special] = _render_similarity_gallery(
                sim_entry.get("special", {}),
                num_layers,
                role_short="S",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if not virtual_view and sim_camera in target_blocks and sim_camera not in galleries:
            if not _similarity_region_has_data(sim_entry, "camera"):
                missing_similarity_notes.append(
                    _missing_similarity_region_note(
                        controller,
                        scene_id=scene_id,
                        mode=mode,
                        region="camera",
                    )
                )
            galleries[sim_camera] = _render_similarity_gallery(
                sim_entry.get("camera", {}),
                num_layers,
                role_short="C",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if not virtual_view and skip_depth in target_blocks and skip_depth not in galleries:
            galleries[skip_depth] = _render_skip_gallery(
                skip_by_mode.get(mode, {}),
                num_layers,
                SKIP_DEPTH_METRICS,
                value_mode=value_mode,
                sim_cutoff=sim_cutoff,
            )
        if not virtual_view and skip_point in target_blocks and skip_point not in galleries:
            galleries[skip_point] = _render_skip_gallery(
                skip_by_mode.get(mode, {}),
                num_layers,
                SKIP_POINT_METRICS,
                value_mode=value_mode,
                sim_cutoff=sim_cutoff,
            )
        if not virtual_view and sim_patch in target_blocks and sim_patch not in galleries:
            if not _similarity_region_has_data(sim_entry, "patch"):
                missing_similarity_notes.append(
                    _missing_similarity_region_note(
                        controller,
                        scene_id=scene_id,
                        mode=mode,
                        region="patch",
                    )
                )
            galleries[sim_patch] = _render_similarity_gallery(
                sim_entry.get("patch", {}),
                num_layers,
                role_short="P",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if sim_all in target_blocks and (sim_all not in galleries or bool(prune_overlay)):
            sim_all_maps = sim_entry.get("all", {})
            if not _similarity_region_has_data(sim_entry, "all"):
                missing_similarity_notes.append(
                    _missing_similarity_region_note(
                        controller,
                        scene_id=scene_id,
                        mode=mode,
                        region="all",
                    )
                )
            galleries[sim_all] = _render_similarity_gallery(
                sim_all_maps,
                num_layers,
                role_short="A",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
                metric_entries=metric_entries,
                prune_overlay=bool(prune_overlay),
                prune_metric_key=prune_metric,
                prune_n=prune_n_i,
            )
            if bool(prune_overlay):
                prune_status = _stage2_prune_status_text(
                    sim_all_maps,
                    num_layers=num_layers,
                    sim_gamma=sim_gamma,
                    sim_cutoff=sim_cutoff,
                    prune_metric_key=prune_metric,
                    prune_n=prune_n_i,
                    prune_overlay=bool(prune_overlay),
                )
        if not virtual_view and sim_mean in target_blocks and sim_mean not in galleries:
            galleries[sim_mean] = _render_similarity_gallery(
                sim_entry.get("mean", {}),
                num_layers,
                role_short="μ",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if not virtual_view and sim_min in target_blocks and sim_min not in galleries:
            galleries[sim_min] = _render_similarity_gallery(
                sim_entry.get("min", {}),
                num_layers,
                role_short="m",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
    rendered = len(galleries)
    total = len(all_blocks)
    if virtual_view:
        info = (
            f"dataset={virtual_dataset} | scene=mean | rendered={rendered}/{total} blocks | {skip_source} | "
            f"pct_clip={clip_key}%"
        )
    else:
        info = f"scene={scene_id} | view={value_mode} | rendered={rendered}/{total} blocks | {skip_source} | pct_clip={clip_key}%"
    if abs(sim_gamma - 1.0) > 1e-12:
        info += f" | sim_gamma={gamma_key / 1000.0:.3f}"
    if cutoff_key > 0:
        info += f" | sim_cutoff={cutoff_key / 1000.0:.3f}"
    if bool(prune_overlay):
        info += f" | prune_metric={prune_metric} n={prune_n_i}"
    if missing_similarity_notes:
        unique_notes = list(dict.fromkeys(missing_similarity_notes))
        info += " | " + " ; ".join(unique_notes[:4])
        if len(unique_notes) > 4:
            info += f" ; +{len(unique_notes) - 4} more missing similarity rows"
    controller.stage2_render_cache[cache_key] = (galleries, info, prune_status)
    return galleries, info, prune_status


def _stage2_modes_for_controller(controller: PairAttentionController) -> List[str]:
    if _is_stage2_virtual_dataset(controller.dataset):
        modes = controller.union_stage2_modes(getattr(controller, "stage2_real_datasets", []))
        if modes:
            return modes
    return _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)


def _preferred_stage2_scene(
    controller: PairAttentionController,
    scene_ids: Sequence[str],
) -> Optional[str]:
    """Prefer a concrete scene with skip data so the Δ controls are useful immediately."""
    for scene_id in scene_ids:
        if scene_id != "mean" and controller.get_skip_report(scene_id) is not None:
            return str(scene_id)
    return str(scene_ids[0]) if scene_ids else None


def _build_ui(controller: PairAttentionController) -> gr.Blocks:
    # Stage2 dataset tags come from v3 outputs on disk.
    stage2_real = _discover_v3_dataset_tags(controller.v3_backbone_root())
    # If no dataset tag has v3 redundancy for this backbone, keep empty.
    # This prevents "only-cache" datasets (e.g. scannet_test50) from showing up
    # and from participating in "min" computations.
    controller.stage2_real_datasets = list(stage2_real)  # type: ignore[attr-defined]
    dataset_choices = [STAGE2_MIN_DATASET, STAGE2_MEAN_DATASET, *stage2_real]
    if controller.dataset not in dataset_choices:
        if stage2_real:
            controller.switch_dataset(stage2_real[0])
        else:
            controller.switch_dataset(STAGE2_MIN_DATASET)
    # scene_ids = controller.scene_ids()  # Pair Attention tab disabled
    # Pair Attention tab disabled — skip pair-tab metadata.
    # pair_enabled = bool(scene_ids)
    # initial_scene = scene_ids[0] if scene_ids else None
    # if pair_enabled and initial_scene is not None:
    #     num_layers, num_frames, frame_labels = controller.scene_metadata_light(initial_scene)
    #     num_heads = controller.pair_ui_num_heads()
    # else:
    #     num_layers, num_frames, frame_labels = 0, 0, []
    #     num_heads = 1
    # initial_frame_a = 0
    # initial_frame_b = min(1, len(frame_labels) - 1) if frame_labels else 0
    stage2_modes_all = _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)
    for fallback_mode in ("mixed_units", "v3_units", "default"):
        if fallback_mode not in stage2_modes_all:
            stage2_modes_all.append(fallback_mode)
    n_stage2_galleries = 9 * len(stage2_modes_all)
    cka_models = list(controller.v3_backbones)
    cka_datasets = (
        _discover_v3_dataset_tags_across_backbones(controller.v3_output_root, cka_models)
        if cka_models
        else list(stage2_real)
    )
    cka_token_regions = [
        STAGE2_TOKEN_ROW_CAMERA,
        STAGE2_TOKEN_ROW_SPECIAL,
        STAGE2_TOKEN_ROW_ALL,
        STAGE2_TOKEN_ROW_PATCH,
        STAGE2_TOKEN_ROW_MEAN,
        STAGE2_TOKEN_ROW_MIN,
    ] if cka_models and cka_datasets else []
    lora_root = controller.lora_root
    stage2_last_render_key: Dict[str, Any] = {"cache_key": None}
    dataset_switch_output_len: Dict[str, int] = {"count": 0}
    cka_render_cache: Dict[Tuple[Any, ...], Optional[np.ndarray]] = {}

    def _empty_stage2_visual_updates() -> Tuple[Any, ...]:
        gal = tuple(gr.update(value=[]) for _ in range(n_stage2_galleries))
        col = tuple(gr.update(visible=False) for _ in range(n_stage2_galleries))
        # outputs: galleries, cols, prune_status, stage2_info, then 6 controls
        return (*gal, *col, gr.update(value=""), "No Stage2 scenes for this dataset.")

    def _stage2_noop_updates() -> Tuple[Any, ...]:
        # outputs: 2*n_galleries + prune_status + stage2_info + 6 controls
        return tuple(gr.update() for _ in range(2 * n_stage2_galleries + 8))

    def _dataset_noop_updates() -> Tuple[Any, ...]:
        count = max(0, int(dataset_switch_output_len["count"]))
        return tuple(gr.update() for _ in range(count))

    def _stage2_visual_bundle(
        scene_id: Optional[str],
        value_mode: str,
        pct_clip_percent: float,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma_value: float,
        sim_cutoff_value: float,
        prune_overlay: bool,
        prune_metric_key: str,
        prune_n: float,
        *,
        request_seq: Optional[int],
        force_gallery_refresh: bool,
    ) -> Tuple[Any, ...]:
        try:
            sim_gamma = float(sim_gamma_value)
        except (TypeError, ValueError):
            sim_gamma = 1.0
        if not math.isfinite(sim_gamma) or sim_gamma <= 0.0:
            sim_gamma = 1.0
        try:
            sim_cutoff = float(sim_cutoff_value)
        except (TypeError, ValueError):
            sim_cutoff = 0.0
        if not math.isfinite(sim_cutoff):
            sim_cutoff = 0.0
        sim_cutoff = float(np.clip(sim_cutoff, 0.0, 1.0))
        prune_on = bool(prune_overlay)
        prune_metric = str(prune_metric_key or "cka").strip().lower()
        prune_n_i = _normalize_prune_n(prune_n)
        prune_sig = _stage2_prune_render_sig(
            prune_overlay=prune_on,
            prune_metric_key=prune_metric,
            prune_n=prune_n_i,
            sim_cutoff=sim_cutoff,
        )
        if _is_stage2_virtual_dataset(controller.dataset):
            scene_id = "mean"
        if not scene_id:
            stage2_last_render_key["cache_key"] = None
            vm, m_val, t_val, choices_vm, token_choices, has_skip, has_sim, pct_ok = _heatmaps_resolve_filters(
                controller,
                None,
                str(value_mode or "absolute"),
                filter_metric_rows,
                filter_token_rows,
            )
            ctrl = _heatmaps_gr_controls_from_resolved(
                vm,
                m_val,
                t_val,
                sim_gamma,
                sim_cutoff,
                choices_vm,
                token_choices,
                has_skip,
                has_sim,
                pct_ok,
                write_values=True,
            )
            if request_seq is not None and request_seq != controller.stage2_request_seq:
                return _stage2_noop_updates()
            return (*_empty_stage2_visual_updates(), *ctrl)

        vm, m_val, t_val, choices_vm, token_choices, has_skip, has_sim, pct_ok = _heatmaps_resolve_filters(
            controller,
            scene_id,
            str(value_mode or "absolute"),
            list(filter_metric_rows) if filter_metric_rows is not None else None,
            list(filter_token_rows) if filter_token_rows is not None else None,
        )
        current_modes = _stage2_modes_for_controller(controller)
        selected_modes = list(current_modes)
        if _is_stage2_virtual_dataset(controller.dataset):
            skip_modes: set[str] = set()
            similarity_regions_by_mode = {mode: {"all"} for mode in current_modes}
        else:
            skip_report = controller.get_skip_report(str(scene_id))
            skip_modes = set(
                _extract_skip_metric_maps_by_mode(skip_report.report).keys()
                if skip_report is not None
                else []
            )
            similarity_report = controller.get_similarity_report(str(scene_id))
            similarity_by_mode = (
                _similarity_maps_with_token_aggregations(
                    _extract_similarity_metric_maps_by_mode(similarity_report.report)
                )
                if similarity_report is not None
                else {}
            )
            similarity_regions_by_mode: Dict[str, set[str]] = {}
            default_regions = {
                region
                for region in STAGE2_TOKEN_ROW_ORDER
                if _similarity_region_has_data(similarity_by_mode.get("default", {}), region)
            }
            for mode in current_modes:
                entry = similarity_by_mode.get(mode, {})
                regions = {
                    region
                    for region in STAGE2_TOKEN_ROW_ORDER
                    if _similarity_region_has_data(entry, region)
                }
                similarity_regions_by_mode[mode] = regions or set(default_regions)

        clip_key = int(round(float(np.clip(float(pct_clip_percent), 0.0, 100.0))))
        gamma_key = _sim_gamma_key(sim_gamma)
        cutoff_key = _sim_cutoff_key(sim_cutoff)
        virtual_dataset = str(controller.dataset).strip().lower() if _is_stage2_virtual_dataset(controller.dataset) else ""
        cache_key = (str(scene_id), vm, clip_key, gamma_key, cutoff_key, virtual_dataset, *prune_sig)
        prune_sig_changed = stage2_last_render_key.get("prune_sig") != prune_sig
        force_updates = (
            force_gallery_refresh
            or stage2_last_render_key["cache_key"] != cache_key
            or prune_sig_changed
        )
        cached_before = controller.stage2_render_cache.get(cache_key)
        before_keys = set(cached_before[0].keys()) if cached_before is not None else set()
        requested_blocks = _stage2_requested_block_keys(
            stage2_modes_all,
            selected_modes,
            m_val,
            t_val,
            skip_modes,
            similarity_regions_by_mode,
        )
        galleries, details, prune_status_text = _prepare_stage2_render(
            controller,
            scene_id,
            vm,
            pct_clip_percent,
            sim_gamma,
            sim_cutoff,
            stage2_modes_all,
            requested_blocks=requested_blocks,
            prune_overlay=prune_on,
            prune_metric_key=prune_metric,
            prune_n=prune_n_i,
        )
        after_keys = set(galleries.keys())
        keys_to_update = (requested_blocks & after_keys) if force_updates else (requested_blocks - before_keys)
        if prune_sig_changed:
            for mode in selected_modes:
                keys_to_update.add(f"sim_{mode}_all")

        if request_seq is not None and request_seq != controller.stage2_request_seq:
            return _stage2_noop_updates()

        block_updates: List[Any] = []
        for mode in stage2_modes_all:
            for block_key in _stage2_block_keys_for_mode(mode):
                if block_key in keys_to_update:
                    block_updates.append(_gallery_update(galleries.get(block_key, [])))
                else:
                    block_updates.append(gr.update())
        vis_flags = _stage2_row_visibility_flags(
            stage2_modes_all,
            selected_modes,
            m_val,
            t_val,
            skip_modes,
            similarity_regions_by_mode,
        )
        vis_updates = [gr.update(visible=bool(v)) for v in vis_flags]
        vm_changed = vm != str(value_mode or "absolute")
        metric_input = list(filter_metric_rows) if filter_metric_rows is not None else None
        token_input = list(filter_token_rows) if filter_token_rows is not None else None
        write_values = bool(
            force_updates
            or vm_changed
            or (metric_input is not None and metric_input != m_val)
            or (token_input is not None and token_input != t_val)
        )
        ctrl = _heatmaps_gr_controls_from_resolved(
            vm,
            m_val,
            t_val,
            sim_gamma,
                sim_cutoff,
            choices_vm,
            token_choices,
            has_skip,
            has_sim,
            pct_ok,
            write_values=write_values,
        )
        stage2_last_render_key["cache_key"] = cache_key
        stage2_last_render_key["prune_sig"] = prune_sig
        return tuple([*block_updates, *vis_updates, prune_status_text, details, *ctrl])

    def _refresh_outputs(
        scene_id: str,
        layer_idx: int,
        frame_a_index: int,
        frame_b_index: int,
        score_space: str,
        head_mode: str,
        head_index: int,
    ):
        if not scene_id or scene_id not in controller.scene_items:
            return (
                None,
                "No cached scene selected (Stage1 manifest missing for this dataset).",
                "",
                "",
            )
        scene_cache = controller.get_scene_cache(scene_id)
        frame_a_index = int(frame_a_index)
        frame_b_index = int(frame_b_index)
        if not (0 <= frame_a_index < len(scene_cache.frame_labels)):
            raise gr.Error(f"Frame A index out of range: {frame_a_index}")
        if not (0 <= frame_b_index < len(scene_cache.frame_labels)):
            raise gr.Error(f"Frame B index out of range: {frame_b_index}")
        if frame_a_index == frame_b_index:
            raise gr.Error("Please choose two different frames.")
        q, k = controller.get_qk(scene_id, int(layer_idx))
        maps_ab = _compute_directional_overlay_maps(
            q,
            k,
            frame_q_idx=frame_a_index,
            frame_k_idx=frame_b_index,
            tokens_per_frame=scene_cache.tokens_per_frame,
            patch_start_idx=scene_cache.patch_start_idx,
            score_space=score_space,
            head_mode=head_mode,
            head_index=int(head_index),
        )
        maps_ba = _compute_directional_overlay_maps(
            q,
            k,
            frame_q_idx=frame_b_index,
            frame_k_idx=frame_a_index,
            tokens_per_frame=scene_cache.tokens_per_frame,
            patch_start_idx=scene_cache.patch_start_idx,
            score_space=score_space,
            head_mode=head_mode,
            head_index=int(head_index),
        )
        panel = _render_panel_array(
            scene_id=scene_id,
            layer_idx=int(layer_idx),
            frame_a_index=frame_a_index,
            frame_b_index=frame_b_index,
            frame_a_id=scene_cache.frame_ids[frame_a_index],
            frame_b_id=scene_cache.frame_ids[frame_b_index],
            image_a=scene_cache.images[frame_a_index],
            image_b=scene_cache.images[frame_b_index],
            maps_ab=maps_ab,
            maps_ba=maps_ba,
            patch_hw=scene_cache.patch_hw,
            score_space=score_space,
            head_mode=head_mode,
            head_index=None if head_mode == "mean" else int(head_index),
        )
        summary = (
            f"scene={scene_id} | layer={int(layer_idx)} | score={score_space} | "
            f"head={'mean' if head_mode == 'mean' else int(head_index)} | "
            f"A idx={frame_a_index} id={scene_cache.frame_ids[frame_a_index]} | "
            f"B idx={frame_b_index} id={scene_cache.frame_ids[frame_b_index]}"
        )
        return panel, summary, scene_cache.frame_labels[frame_a_index], scene_cache.frame_labels[frame_b_index]

    def _scene_changed(scene_id: str, score_space: str, head_mode: str, head_index: int):
        if not scene_id or scene_id not in controller.scene_items:
            return (
                gr.update(maximum=0, value=0),
                gr.update(maximum=0, value=0),
                gr.update(maximum=0, value=0),
                "layers: 0, frames: 0",
                None,
                "No cached scene selected (Stage1 manifest missing for this dataset).",
                "",
                "",
            )
        num_layers_, num_frames_, labels = controller.scene_metadata_light(scene_id)
        max_layer = max(0, num_layers_ - 1)
        max_frame = max(0, num_frames_ - 1)
        info = f"layers: {num_layers_}, frames: {num_frames_}"
        frame_a = 0
        frame_b = min(1, max_frame)
        panel, summary, frame_a_info, frame_b_info = _refresh_outputs(
            scene_id=scene_id,
            layer_idx=min(0, max_layer),
            frame_a_index=frame_a,
            frame_b_index=frame_b,
            score_space=score_space,
            head_mode=head_mode,
            head_index=head_index,
        )
        return (
            gr.update(maximum=max_layer, value=min(0, max_layer)),
            gr.update(maximum=max_frame, value=frame_a),
            gr.update(maximum=max_frame, value=frame_b),
            info,
            panel,
            summary,
            frame_a_info,
            frame_b_info,
        )

    def _heatmaps_scope_changed(
        dataset: str,
        backbone: str | None,
        cur_value_mode: str,
        pct_clip: float,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma: float,
        sim_cutoff: float,
        prune_overlay: bool,
        prune_metric_key: str,
        prune_n: float,
    ):
        controller.stage2_request_seq += 1
        if backbone:
            controller.switch_backbone(backbone)
        ds_key = str(dataset).strip().lower()
        if ds_key != controller.dataset:
            controller.switch_dataset(dataset)
        virtual_view = _is_stage2_virtual_dataset(controller.dataset)
        if virtual_view:
            stage2_scene_ids_new = ["mean"]
            initial_stage2_scene_new = "mean"
        else:
            stage2_scene_ids_new = controller.stage2_scene_ids()
            initial_stage2_scene_new = _preferred_stage2_scene(controller, stage2_scene_ids_new)
        tup_hm = _stage2_visual_bundle(
            initial_stage2_scene_new,
            str(cur_value_mode or "absolute"),
            float(pct_clip),
            list(filter_metric_rows) if filter_metric_rows is not None else None,
            list(filter_token_rows) if filter_token_rows is not None else None,
            float(sim_gamma),
            float(sim_cutoff),
            bool(prune_overlay),
            str(prune_metric_key or "cka"),
            float(prune_n),
            request_seq=None,
            force_gallery_refresh=True,
        )
        # tup_hm tail: prune_status, info, then 6 filter controls
        hm_main = tup_hm[:-8]
        hm_prune_status = tup_hm[-8]
        hm_info = tup_hm[-7]
        hm_ctrl = tup_hm[-6:]

        updates: List[Any] = [
            gr.update(value=controller.dataset),
            gr.update(
                choices=stage2_scene_ids_new,
                value=initial_stage2_scene_new,
                interactive=not virtual_view,
            ),
            *hm_main,
            hm_prune_status,
            hm_info,
            *hm_ctrl,
        ]
        updates.insert(1, gr.update(value=controller.v3_backbone))
        return tuple(updates)

    def _refresh_stage2_full(
        scene_id: str,
        value_mode: str,
        pct_clip_percent: float,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma: float,
        sim_cutoff: float,
        prune_overlay: bool,
        prune_metric_key: str,
        prune_n: float,
    ):
        controller.stage2_request_seq += 1
        request_seq = int(controller.stage2_request_seq)
        return _stage2_visual_bundle(
            scene_id,
            str(value_mode or "absolute"),
            float(pct_clip_percent),
            filter_metric_rows,
            filter_token_rows,
            float(sim_gamma),
            float(sim_cutoff),
            bool(prune_overlay),
            str(prune_metric_key or "cka"),
            float(prune_n),
            request_seq=request_seq,
            force_gallery_refresh=False,
        )

    def _refresh_cka_heatmaps(
        token_region: str,
        sim_cutoff_value: float,
    ) -> Tuple[Any, ...]:
        region = str(token_region or "").strip().lower()
        if region not in set(cka_token_regions):
            region = cka_token_regions[0] if cka_token_regions else "special"
        try:
            sim_cutoff = float(sim_cutoff_value)
        except (TypeError, ValueError):
            sim_cutoff = 0.0
        if not math.isfinite(sim_cutoff):
            sim_cutoff = 0.0
        sim_cutoff = float(np.clip(sim_cutoff, 0.0, 1.0))
        cutoff_key = _sim_cutoff_key(sim_cutoff)

        images: List[Any] = []
        notes: List[str] = []
        rendered = 0
        dataset_total = len(cka_datasets) * len(cka_models)
        row_mean_total = len(cka_models)
        display_total = dataset_total + row_mean_total
        for backbone in cka_models:
            for dataset in cka_datasets:
                cache_key = (str(dataset), str(backbone), region, cutoff_key)
                if cache_key in cka_render_cache:
                    image = cka_render_cache[cache_key]
                    detail = "cached" if image is not None else "missing"
                else:
                    image, detail = _render_cka_heatmap_for_backbone(
                        dataset=str(dataset),
                        backbone=str(backbone),
                        token_region=region,
                        sim_cutoff=sim_cutoff,
                    )
                    cka_render_cache[cache_key] = image
                if image is not None:
                    rendered += 1
                elif len(notes) < 6:
                    notes.append(f"{dataset}/{backbone}: {detail}")
                images.append(image)

            row_mean_cache_key = ("mean", str(backbone), region, cutoff_key)
            if row_mean_cache_key in cka_render_cache:
                row_mean_image = cka_render_cache[row_mean_cache_key]
                row_mean_detail = "cached" if row_mean_image is not None else "missing"
            else:
                row_mean_image, row_mean_detail = _render_cka_heatmap_for_backbone(
                    dataset="mean",
                    backbone=str(backbone),
                    token_region=region,
                    sim_cutoff=sim_cutoff,
                )
                cka_render_cache[row_mean_cache_key] = row_mean_image
            if row_mean_image is not None:
                rendered += 1
            elif len(notes) < 6:
                notes.append(f"mean/{backbone}: {row_mean_detail}")
            images.append(row_mean_image)

        region_label = region if region != "all" else "all-token"
        info = (
            f"scene=mean | token={region_label} | metric=CKA | "
            f"rendered={rendered}/{display_total} | cutoff={cutoff_key / 1000.0:.3f}"
        )
        if notes:
            info += " | " + " ; ".join(notes)
            missing_more = display_total - rendered - len(notes)
            if missing_more > 0:
                info += f" ; +{missing_more} more missing cells"
        return tuple([*images, info])

    with gr.Blocks(
        title="Results Panel",
        css="""
        .results-media, .results-gallery {
            width: 100%;
        }
        .results-media img {
            width: 100%;
            object-fit: contain;
        }
        .results-gallery .grid-wrap {
            width: 100%;
        }
        .results-gallery figure,
        .results-gallery .thumbnail-item,
        .results-gallery .gallery-item,
        .results-gallery [class*="thumbnail"],
        .results-gallery [class*="gallery-item"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .results-gallery button {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: center !important;
        }
        .results-gallery button > div,
        .results-gallery button > span,
        .results-gallery button p,
        .results-gallery button label,
        .results-gallery button small {
            width: 100% !important;
            text-align: center !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .results-gallery button * {
            text-align: center !important;
        }
        .results-gallery [class*="caption-container"],
        .results-gallery [class*="caption-wrap"],
        .results-gallery [class*="caption"],
        .results-gallery figcaption {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        .results-gallery [class*="caption"],
        .results-gallery figcaption,
        .results-gallery .caption-label {
            text-align: center !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
            line-height: 1.2 !important;
            font-size: 12px !important;
            min-height: 2.6em;
        }
        .results-gallery button,
        .results-gallery .thumbnail-item,
        .results-gallery .gallery-item {
            width: 100% !important;
        }
        .recovery-eval-gallery.wrap,
        .lora-eval-gallery.wrap,
        .lora-training-gallery.wrap {
            min-height: 0 !important;
            height: auto !important;
            overflow: visible !important;
        }
        .recovery-eval-gallery .grid-wrap,
        .recovery-eval-gallery [class*="grid"],
        .lora-eval-gallery .grid-wrap,
        .lora-eval-gallery [class*="grid"],
        .lora-training-gallery .grid-wrap,
        .lora-training-gallery [class*="grid"] {
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            align-content: start !important;
        }
        .recovery-eval-gallery .thumbnail-item,
        .recovery-eval-gallery .thumbnail-small,
        .recovery-eval-gallery button.thumbnail-item,
        .recovery-eval-gallery button,
        .lora-eval-gallery .thumbnail-item,
        .lora-eval-gallery .thumbnail-small,
        .lora-eval-gallery button.thumbnail-item,
        .lora-eval-gallery button,
        .lora-training-gallery .thumbnail-item,
        .lora-training-gallery .thumbnail-small,
        .lora-training-gallery button.thumbnail-item,
        .lora-training-gallery button {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            aspect-ratio: auto !important;
        }
        .recovery-eval-gallery .thumbnail-item > div,
        .recovery-eval-gallery .thumbnail-small > div,
        .recovery-eval-gallery button > div,
        .recovery-eval-gallery [class*="image"],
        .recovery-eval-gallery [class*="preview"],
        .lora-eval-gallery .thumbnail-item > div,
        .lora-eval-gallery .thumbnail-small > div,
        .lora-eval-gallery button > div,
        .lora-eval-gallery [class*="image"],
        .lora-eval-gallery [class*="preview"],
        .lora-training-gallery .thumbnail-item > div,
        .lora-training-gallery .thumbnail-small > div,
        .lora-training-gallery button > div,
        .lora-training-gallery [class*="image"],
        .lora-training-gallery [class*="preview"] {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            aspect-ratio: auto !important;
        }
        .recovery-eval-gallery .thumbnail-item img,
        .recovery-eval-gallery img,
        .lora-eval-gallery .thumbnail-item img,
        .lora-eval-gallery img,
        .lora-training-gallery .thumbnail-item img,
        .lora-training-gallery img {
            display: block !important;
            height: auto !important;
            max-height: none !important;
            vertical-align: bottom !important;
        }
        .recovery-eval-gallery [class*="caption"],
        .recovery-eval-gallery figcaption,
        .recovery-eval-gallery .caption-label,
        .lora-eval-gallery [class*="caption"],
        .lora-eval-gallery figcaption,
        .lora-eval-gallery .caption-label,
        .lora-training-gallery [class*="caption"],
        .lora-training-gallery figcaption,
        .lora-training-gallery .caption-label {
            min-height: 0 !important;
            margin-top: 2px !important;
            padding-top: 0 !important;
        }
        .stage2-right {
            gap: 8px !important;
        }
        .stage2-gallery-row {
            align-items: flex-start;
        }
        .stage2-heatmap-gallery.wrap {
            min-height: 0 !important;
            height: auto !important;
        }
        .stage2-heatmap-gallery .grid-wrap,
        .stage2-heatmap-gallery [class*="grid"] {
            min-height: 0 !important;
            height: auto !important;
            align-content: start !important;
        }
        .stage2-heatmap-gallery .thumbnail-item,
        .stage2-heatmap-gallery .thumbnail-small,
        .stage2-heatmap-gallery button.thumbnail-item {
            height: auto !important;
            min-height: 0 !important;
        }
        .stage2-heatmap-gallery .thumbnail-item img,
        .stage2-heatmap-gallery img {
            display: block !important;
            height: auto !important;
            max-height: none !important;
            vertical-align: bottom !important;
        }
        .cka-heatmap-matrix {
            gap: 8px !important;
            min-width: 0 !important;
            overflow-x: hidden !important;
        }
        .cka-grid-row,
        .cka-grid-header {
            align-items: center !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        .cka-grid-row > *,
        .cka-grid-header > * {
            min-width: 0 !important;
        }
        .cka-grid-cell,
        .cka-grid-dataset-label {
            flex: 1 1 0 !important;
        }
        .cka-grid-corner,
        .cka-grid-model-label {
            flex: 0 0 96px !important;
            width: 96px !important;
        }
        .cka-grid-model-label {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }
        .cka-grid-model-label .prose,
        .cka-grid-model-label p {
            margin: 0 !important;
            width: 100% !important;
            text-align: center !important;
            font-size: 1.35rem !important;
            line-height: 1.25 !important;
        }
        .cka-grid-dataset-label {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }
        .cka-grid-dataset-label .prose,
        .cka-grid-dataset-label p {
            margin: 0 !important;
            width: 100% !important;
            text-align: center !important;
            font-size: clamp(0.8rem, 1.1vw, 1.1rem) !important;
            line-height: 1.25 !important;
            overflow-wrap: anywhere !important;
        }
        .cka-heatmap-image {
            width: 100% !important;
            min-width: 0 !important;
        }
        .cka-heatmap-image img,
        .cka-heatmap-image [class*="image"] img {
            display: block !important;
            height: auto !important;
            max-height: none !important;
        }
        .cka-heatmap-image .label-wrap,
        .cka-heatmap-image > label,
        .cka-heatmap-image .image-label {
            display: none !important;
        }
        /* Pct radio option: keep visible but non-clickable when Stage1 eval is missing (structure varies by Gradio version). */
        .hm-value-mode-pct-locked .wrap label:last-of-type,
        .hm-value-mode-pct-locked label:last-of-type {
            opacity: 0.45 !important;
            pointer-events: none !important;
            cursor: not-allowed !important;
        }
        .summary-panel {
            width: 100%;
        }
        .summary-table-wrap {
            width: 100%;
            overflow: auto;
            max-height: 78vh;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .summary-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14.5px;
            line-height: 1.45;
        }
        .summary-table thead th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            color: #0f172a;
            font-weight: 600;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #cbd5e1;
            white-space: nowrap;
        }
        .summary-table tbody td {
            padding: 8px 12px;
            border-bottom: 1px solid #eef2f7;
            color: #1e293b;
            vertical-align: middle;
        }
        .summary-table tbody tr:nth-child(even) td {
            background: #fafbfd;
        }
        .summary-table tbody tr:hover td {
            background: #f1f5f9;
        }
        .summary-col-dataset {
            font-weight: 600;
            color: #0f172a;
            white-space: nowrap;
            text-align: center;
            vertical-align: middle !important;
        }
        .summary-col-backbone {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 14px;
            white-space: nowrap;
            text-align: center;
            vertical-align: middle !important;
        }
        .summary-col-mode {
            white-space: nowrap;
            font-weight: 600;
        }
        .summary-mode-upper {
            color: #0369a1;
        }
        .summary-mode-plan {
            color: #7c3aed;
        }
        .summary-na {
            color: #94a3b8;
        }
        .summary-pct {
            display: inline-block;
            margin-left: 6px;
            padding: 1px 6px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
            background: #f1f5f9;
            color: #475569;
        }
        .summary-pct-up {
            background: #fef2f2;
            color: #b91c1c;
        }
        .summary-pct-down {
            background: #ecfdf5;
            color: #047857;
        }
        .summary-empty {
            padding: 24px;
            color: #64748b;
            font-size: 15.5px;
        }
        """,
    ) as demo:
        with gr.Tabs(selected="recovery_results"):
            build_cka_heatmaps_tab(
                datasets=cka_datasets,
                models=cka_models,
                token_regions=cka_token_regions,
                refresh_cka_heatmaps_cb=_refresh_cka_heatmaps,
            )

            # pair_ui = build_pair_tab(
            #     controller=controller,
            #     dataset_choices=dataset_choices,
            #     scene_ids=scene_ids,
            #     initial_scene=initial_scene,
            #     num_layers=num_layers,
            #     num_frames=num_frames,
            #     frame_labels=frame_labels,
            #     initial_frame_a=initial_frame_a,
            #     initial_frame_b=initial_frame_b,
            #     num_heads=num_heads,
            #     scene_changed_cb=_scene_changed,
            #     refresh_outputs_cb=_refresh_outputs,
            # )

            stage2_scene_ids = controller.stage2_scene_ids()
            initial_stage2_scene = _preferred_stage2_scene(controller, stage2_scene_ids)
            heatmaps_ui = build_heatmaps_tab(
                controller=controller,
                dataset_choices=dataset_choices,
                backbone_choices=controller.v3_backbones,
                initial_backbone=controller.v3_backbone,
                stage2_modes=stage2_modes_all,
                stage2_scene_ids=stage2_scene_ids,
                initial_stage2_scene=initial_stage2_scene,
                stage2_metric_row_pose=STAGE2_METRIC_ROW_POSE,
                stage2_metric_row_depth=STAGE2_METRIC_ROW_DEPTH,
                stage2_metric_row_point=STAGE2_METRIC_ROW_POINT,
                stage2_token_row_special=STAGE2_TOKEN_ROW_SPECIAL,
                stage2_token_row_camera=STAGE2_TOKEN_ROW_CAMERA,
                stage2_token_row_patch=STAGE2_TOKEN_ROW_PATCH,
                stage2_token_row_all=STAGE2_TOKEN_ROW_ALL,
                stage2_token_row_mean=STAGE2_TOKEN_ROW_MEAN,
                stage2_token_row_min=STAGE2_TOKEN_ROW_MIN,
                heatmaps_resolve_filters_cb=_heatmaps_resolve_filters,
                refresh_stage2_full_cb=_refresh_stage2_full,
            )

            build_summary_tab(blocks=demo, output_root=OUTPUT_ROOT_DEFAULT, tab_id="eval_summary")

            build_recovery_tab(
                blocks=demo,
                recovery_root=Path(lora_root),
                discover_recovery_runs=_discover_recovery_runs,
                recovery_plot_rows=_recovery_plot_rows,
                tab_id="recovery_results",
            )

            build_eval_tab(
                blocks=demo,
                controller=controller,
                eval_viser_controller_cls=EvalViserController,
                load_image_if_exists=_load_image_if_exists,
            )

        dataset_switch_outputs: List[Any] = [
            heatmaps_ui["stage2_dataset_dd"],
        ]
        if heatmaps_ui.get("stage2_backbone_dd") is not None:
            dataset_switch_outputs.append(heatmaps_ui["stage2_backbone_dd"])
        dataset_switch_outputs.extend(
            [
            heatmaps_ui["stage2_scene_dd"],
            *heatmaps_ui["stage2_mode_galleries"],
            *heatmaps_ui["stage2_row_body_cols"],
            heatmaps_ui["stage2_prune_status"],
            heatmaps_ui["stage2_info"],
            heatmaps_ui["stage2_value_mode"],
            heatmaps_ui["stage2_pct_clip_slider"],
            heatmaps_ui["stage2_metric_rows_filter"],
            heatmaps_ui["stage2_token_rows_filter"],
            heatmaps_ui["stage2_sim_gamma_slider"],
            heatmaps_ui["stage2_sim_cutoff_slider"],
            ]
        )
        dataset_switch_output_len["count"] = len(dataset_switch_outputs)
        scope_switch_inputs: List[Any] = [
            heatmaps_ui["stage2_dataset_dd"],
            heatmaps_ui["stage2_value_mode"],
            heatmaps_ui["stage2_pct_clip_slider"],
            heatmaps_ui["stage2_metric_rows_filter"],
            heatmaps_ui["stage2_token_rows_filter"],
            heatmaps_ui["stage2_sim_gamma_slider"],
            heatmaps_ui["stage2_sim_cutoff_slider"],
            heatmaps_ui["stage2_prune_show"],
            heatmaps_ui["stage2_prune_metric"],
            heatmaps_ui["stage2_prune_n"],
        ]
        if heatmaps_ui.get("stage2_backbone_dd") is not None:
            scope_switch_inputs = [
                heatmaps_ui["stage2_dataset_dd"],
                heatmaps_ui["stage2_backbone_dd"],
                *scope_switch_inputs[1:],
            ]
        else:
            scope_switch_inputs = [heatmaps_ui["stage2_dataset_dd"], gr.State(None), *scope_switch_inputs[1:]]

        # pair_ui["dataset_dd"].change(...)  # Pair Attention tab disabled
        heatmaps_ui["stage2_dataset_dd"].change(
            _heatmaps_scope_changed,
            inputs=scope_switch_inputs,
            outputs=dataset_switch_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        if heatmaps_ui.get("stage2_backbone_dd") is not None:
            heatmaps_ui["stage2_backbone_dd"].change(
                _heatmaps_scope_changed,
                inputs=scope_switch_inputs,
                outputs=dataset_switch_outputs,
                show_progress="minimal",
                trigger_mode="always_last",
            )

        def _save_prune_plan(
            scene_id: str,
            value_mode: str,
            pct_clip_percent: float,
            filter_metric_rows: List[str] | None,
            filter_token_rows: List[str] | None,
            sim_gamma: float,
            sim_cutoff: float,
            prune_overlay: bool,
            prune_metric_key: str,
            prune_n: float,
        ) -> str:
            # Save uses the *same* settings as current visualization.
            if not prune_overlay:
                raise gr.Error("Enable 'Show dense-triangle markers' before saving.")
            metric = str(prune_metric_key or "cka").strip().lower()
            if metric not in {"cka", "cosine"}:
                raise gr.Error(f"Unsupported prune metric: {metric}")
            n_tri = _normalize_prune_n(prune_n)
            if n_tri <= 0:
                raise gr.Error("n must be > 0 to save a prune plan.")
            cutoff = float(np.clip(float(sim_cutoff), 0.0, 1.0))
            if cutoff <= 0.0:
                raise gr.Error("Cutoff must be > 0 to generate a prune plan.")

            allowed_modes = _stage2_modes_for_controller(controller)
            mode = allowed_modes[0] if allowed_modes else ""
            if not mode:
                raise gr.Error("No Stage2 mode available to save prune plan.")

            rep = controller.get_similarity_report(str(scene_id))
            if _is_stage2_virtual_dataset(controller.dataset):
                virtual_dataset = str(controller.dataset).strip().lower()
                # Build virtual-dataset similarity maps as displayed (already includes gamma/cutoff in builder).
                maps_by_mode = _build_cross_dataset_similarity_all(
                    controller,
                    [mode],
                    sim_gamma=float(sim_gamma),
                    sim_cutoff=cutoff,
                    aggregate=virtual_dataset,
                )
                all_map = maps_by_mode.get(mode, {})
                packed = _coerce_packed(all_map.get(metric))
                unit_ids = [f"u{i}" for i in range(int(controller.resolve_stage2_num_layers('mean')))]
            else:
                if rep is None:
                    raise gr.Error("Missing similarity report for this scene.")
                unit_ids = [str(u) for u in list(rep.report.get("unit_ids", []))]
                by_mode = _extract_similarity_metric_maps_by_mode(rep.report)
                sim_entry = _similarity_maps_for_skip_mode(by_mode, mode)
                packed = _coerce_packed(sim_entry.get("all", {}).get(metric))
            if packed is None:
                raise gr.Error(f"Missing packed similarity for metric={metric}")
            num_layers = len(unit_ids) if unit_ids else int(rep.report.get("num_layers", 0)) if rep else 0
            if num_layers <= 0:
                raise gr.Error("Cannot resolve num_layers for prune plan.")

            # Apply gamma/cutoff exactly as in rendering.
            processed = list(packed)
            if metric in {"cka", "cosine"}:
                if abs(float(sim_gamma) - 1.0) > 1e-12:
                    processed = _similarity_packed_gamma_01(processed, float(sim_gamma))
                processed = _similarity_packed_cutoff_01(processed, cutoff)

            grid = upper_triangle_unpack_for_heatmap(processed, num_layers, float("nan"))
            sim_mat = np.zeros((num_layers, num_layers), dtype=np.float64)
            for i in range(num_layers):
                sim_mat[i, i] = 1.0
                for j in range(i + 1, num_layers):
                    v = grid[i][j]
                    if v == v:
                        sim_mat[i, j] = float(v)

            from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan, default_prune_plan_root
            from analysis.prune_plan.triangle_prune import build_triangle_prune_actions

            actions, extra, picks = build_triangle_prune_actions(
                sim_mat,
                unit_ids,
                threshold=cutoff,
                max_triangles=n_tri,
                lora_rank=32,
                lora_alpha=64.0,
            )
            extra = dict(extra)
            extra.setdefault("view_save", {})
            extra["view_save"].update(
                {
                    "scene_id": str(scene_id),
                    "mode": str(mode),
                    "metric": metric,
                    "sim_cutoff": cutoff,
                    "sim_gamma": float(sim_gamma),
                    "n": int(n_tri),
                }
            )

            drop_indices: set[int] = set()
            lora_specs: dict[str, UnitLoraSpec] = {}
            for p in picks:
                for idx in range(int(p.i) + 1, int(p.j) + 1):
                    drop_indices.add(int(idx))
                lora_specs[str(int(p.i))] = UnitLoraSpec(
                    rank=32,
                    alpha=64.0,
                    dropout=0.0,
                    target_submodules=("qkv", "proj"),
                    alignment_teacher_unit_index=int(p.j),
                )
            drop_sorted = sorted(drop_indices)
            plan = UnitPrunePlan(
                backbone=str(controller.v3_backbone or controller.backbone or ""),
                dataset=str(controller.dataset),
                num_units=int(num_layers),
                drop=drop_sorted,
                drop_length=len(drop_sorted),
                lora=lora_specs,
                source_cache=str(controller.v3_output_root / str(controller.v3_backbone) / str(controller.dataset) / "cache"),
                extra=extra,
            )
            out_root = default_prune_plan_root()
            json_path = out_root / f"{str(controller.v3_backbone).strip().lower()}.json"
            plan.save(json_path)

            # Render and save the exact image shown (with markers).
            markers = [(p.i, p.j, str(k)) for k, p in enumerate(picks)]
            img = _heatmap_rgb_from_packed(
                processed,
                num_layers,
                f"A·{metric.upper()} (saved)",
                vmin=0.0,
                vmax=1.0,
                markers=markers,
            )
            png_path = out_root / f"{str(controller.v3_backbone).strip().lower()}_{controller.dataset}_{metric}_triangles.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(png_path, img)

            return (
                f"Saved plan={json_path.name} (drop_length={plan.drop_length}) "
                f"and viz={png_path.name} (scene={scene_id}, mode={mode})."
            )

        heatmaps_ui["stage2_prune_save_btn"].click(
            _save_prune_plan,
            inputs=heatmaps_ui["stage2_inputs"],
            outputs=[heatmaps_ui["stage2_prune_status"]],
            show_progress="minimal",
        )

    return demo


def main() -> None:
    args = _build_parser().parse_args()
    v3_backbone = str(args.v3_backbone).strip()
    controller = PairAttentionController(
        output_root=args.output_root,
        dataset=args.dataset,
        data_roots_by_dataset={
            "scannet": args.scannet_root,
            "scannet_dev50": args.scannet_root,
            "scannet_test50": args.scannet_test_root,
            "7scenes": args.root_7scenes,
            "nrgbd": args.root_nrgbd,
        },
        device_arg=args.device,
        viser_host=args.viser_host,
        viser_port=args.viser_port,
        v3_backbone=v3_backbone,
    )
    demo = _build_ui(controller)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        share=True,
        allowed_paths=_gradio_allowed_paths(controller),
    )


if __name__ == "__main__":
    main()

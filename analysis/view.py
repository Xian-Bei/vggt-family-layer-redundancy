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
from dataclasses import dataclass
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

from vggt.models.vggt import VGGT

from .redundancy.metrics import upper_triangle_unpack_for_heatmap
from analysis.eval.scannet_eval import SceneData, load_scene_images
from .redundancy.cache_layout import count_scene_replay_layers_disk, load_scene_metadata, scene_cache_dir
from .redundancy.visualize_pair_attention import (
    _build_global_positions,
    _load_cache_row,
    _overlay_patch_map,
    _reconstruct_qk,
)
from .vis import build_eval_tab, build_heatmaps_tab, build_lora_tab, build_pair_tab

DEFAULT_LORA_ROOT = Path("analysis_outputs/lora_skip")


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
    p = argparse.ArgumentParser(description="Interactive browser panel for pair attention.")
    p.add_argument("--dataset", type=str, default="scannet", choices=["scannet", "7scenes", "nrgbd"])
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/cache"))
    p.add_argument(
        "--lora-dir",
        type=Path,
        default=DEFAULT_LORA_ROOT,
        help="Directory containing LoRA experiment subfolders (each subfolder is one run).",
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Legacy single analysis dir. If set, used for both similarity and skip reports.",
    )
    p.add_argument("--analysis-sim-dir", type=Path, default=Path("analysis_outputs/similarity"))
    p.add_argument("--analysis-skip-dir", type=Path, default=Path("analysis_outputs/skip"))
    p.add_argument("--data-root", type=Path, default=Path("data/scans_dev50/"))
    p.add_argument("--scannet-root", type=Path, default=Path("data/scans_dev50/"))
    p.add_argument("--root-7scenes", type=Path, default=Path("data/7scenes"))
    p.add_argument("--root-nrgbd", type=Path, default=Path("data/neural_rgbd_data"))
    p.add_argument("--hf-model", type=str, default="", help="Optional HF model override; default reads manifest.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--server-name", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7861)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path("analysis_outputs/evaluation"),
        help=(
            "Eval outputs directory. Either the shared bundle root ``analysis_outputs/evaluation`` "
            "(suites: 7andn, scans_*) so the UI can pick model + dataset, or a single run folder "
            "like ``.../scans_test50_input_frame_50/Ours_vggt`` with per-scene subdirs."
        ),
    )
    p.add_argument("--viser-host", type=str, default="127.0.0.1", help="Host for embedded Viser server.")
    p.add_argument("--viser-port", type=int, default=8090, help="Port for embedded Viser server.")
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
    ("ate", "ATE"),
    ("rpe_rot", "RPE_rot"),
    ("rpe_trans", "RPE_trans"),
]

SKIP_DEPTH_METRICS: List[Tuple[str, str]] = [
    ("depth_abs_rel", "Depth abs rel"),
    ("depth_delta1", "Depth delta1"),
    ("chamfer_distance", "Chamfer"),
]


class PairAttentionController:
    def __init__(
        self,
        cache_dir: Path,
        analysis_sim_dir: Path,
        analysis_skip_dir: Path,
        data_root: Path,
        dataset: str,
        data_roots_by_dataset: Dict[str, Path],
        hf_model: str,
        device_arg: str,
        *,
        lora_root: Path = DEFAULT_LORA_ROOT,
        eval_root: Optional[Path] = None,
        viser_host: str = "127.0.0.1",
        viser_port: int = 8090,
    ) -> None:
        self.lora_root = _resolve_path_cwd_then_repo(Path(lora_root))
        self.eval_root = (
            _resolve_path_cwd_then_repo(Path(eval_root)) if eval_root is not None else None
        )
        self.viser_host = str(viser_host)
        self.viser_port = int(viser_port)
        self.cache_root = cache_dir
        self.analysis_similarity_root = analysis_sim_dir
        self.analysis_skip_root = analysis_skip_dir
        self.data_root = data_root
        self.data_roots_by_dataset = dict(data_roots_by_dataset)
        self.device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
        self.dataset = str(dataset).strip().lower()
        self.cache_root_exists = self.cache_root.exists() and self.cache_root.is_dir()

        self.cache_dir = self._resolve_dataset_dir(self.cache_root, self.dataset)
        self.analysis_similarity_dir = self._resolve_dataset_dir(self.analysis_similarity_root, self.dataset)
        self.analysis_skip_dir = self._resolve_dataset_dir(self.analysis_skip_root, self.dataset)
        self.data_root = self.data_roots_by_dataset.get(self.dataset, self.data_root)

        manifest_path = self.cache_dir / "manifest.json"
        self.cache_available = manifest_path.exists()
        self.manifest: Dict[str, Any] = {}
        if self.cache_available:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.scene_items: Dict[str, Dict[str, Any]] = {}
        for item in self.manifest.get("scenes", []):
            scene_id = str(item.get("scene_id", "")).strip()
            has_dir = str(item.get("dir", "")).strip() != ""
            has_file = str(item.get("file", "")).strip() != ""
            if scene_id and (has_dir or has_file):
                self.scene_items[scene_id] = item

        self.hf_model = hf_model.strip() or str(self.manifest.get("hf_model", "")).strip()
        # Model is required only for Pair Attention reconstruction; Stage2/LoRA/Eval tabs can work without it.
        self.pair_available = self.cache_available and bool(self.scene_items) and bool(self.hf_model)

        self.model: Optional[VGGT] = None
        self.scene_cache: Dict[str, SceneCache] = {}
        self.qk_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.stage1_eval_by_scene = self._load_stage1_eval_by_scene()
        self.similarity_reports = self._discover_similarity_reports()
        self.skip_reports = self._discover_skip_reports()
        self.stage2_render_cache: Dict[Tuple[str, str, int, int, int], Tuple[Dict[str, List[np.ndarray]], str]] = {}
        self.stage2_request_seq = 0

    def _resolve_dataset_dir(self, root: Path, dataset: str) -> Path:
        p = Path(root)
        cand = p / dataset
        if (cand / "manifest.json").exists() or (cand / "per_scene").exists() or cand.exists():
            return cand
        return p

    def switch_dataset(self, dataset: str) -> None:
        new_ds = str(dataset).strip().lower()
        self.dataset = new_ds
        self.cache_dir = self._resolve_dataset_dir(self.cache_root, self.dataset)
        self.analysis_similarity_dir = self._resolve_dataset_dir(self.analysis_similarity_root, self.dataset)
        self.analysis_skip_dir = self._resolve_dataset_dir(self.analysis_skip_root, self.dataset)
        self.data_root = self.data_roots_by_dataset.get(self.dataset, self.data_root)

        manifest_path = self.cache_dir / "manifest.json"
        self.cache_available = manifest_path.exists()
        self.manifest = {}
        if self.cache_available:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.scene_items = {}
        for item in self.manifest.get("scenes", []):
            scene_id = str(item.get("scene_id", "")).strip()
            has_dir = str(item.get("dir", "")).strip() != ""
            has_file = str(item.get("file", "")).strip() != ""
            if scene_id and (has_dir or has_file):
                self.scene_items[scene_id] = item

        self.hf_model = str(self.manifest.get("hf_model", "")).strip() or self.hf_model
        self.pair_available = self.cache_available and bool(self.scene_items) and bool(self.hf_model)
        self.scene_cache = {}
        self.qk_cache = {}
        self.stage2_render_cache = {}
        self.stage1_eval_by_scene = self._load_stage1_eval_by_scene()
        self.similarity_reports = self._discover_similarity_reports()
        self.skip_reports = self._discover_skip_reports()
        self.stage2_request_seq = 0

    def _get_model(self) -> VGGT:
        if not self.hf_model:
            raise SystemExit(
                "Pair Attention unavailable: no HF model provided and cache manifest lacks 'hf_model'."
            )
        if self.model is None:
            self.model = VGGT.from_pretrained(self.hf_model).to(self.device).eval()
        return self.model

    def scene_ids(self) -> List[str]:
        return sorted(self.scene_items.keys())

    def stage2_scene_ids(self) -> List[str]:
        all_ids = set(self.similarity_reports.keys()) | set(self.skip_reports.keys())
        scene_ids: List[str] = []
        if "mean" in all_ids and self.resolve_stage2_num_layers("mean") > 0:
            scene_ids.append("mean")
        scene_ids.extend(sorted(k for k in all_ids if k != "mean"))
        return scene_ids

    def num_heads(self) -> int:
        model = self._get_model()
        return int(model.aggregator.global_blocks[0].attn.num_heads)

    def _resolve_scene_eff(self, scene_id: str, item: Dict[str, Any]) -> SceneData:
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

        item = self.scene_items[scene_id]
        frame_ids = [str(x) for x in item.get("frame_ids", [])]
        if not frame_ids:
            raise SystemExit(f"Manifest entry for {scene_id!r} lacks frame_ids.")
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
        item = self.scene_items[scene_id]
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

    def _discover_similarity_reports(self) -> Dict[str, Stage2SceneReport]:
        return self._discover_similarity_reports_in_dir(self.analysis_similarity_dir)

    def _discover_skip_reports(self) -> Dict[str, Stage2SceneReport]:
        return self._discover_skip_reports_in_dir(self.analysis_skip_dir)

    def stage2_modes_for_dataset(self, dataset: str) -> List[str]:
        ds = str(dataset).strip().lower()
        sim_dir = self._resolve_dataset_dir(self.analysis_similarity_root, ds)
        skip_dir = self._resolve_dataset_dir(self.analysis_skip_root, ds)
        similarity_reports = self._discover_similarity_reports_in_dir(sim_dir)
        skip_reports = self._discover_skip_reports_in_dir(skip_dir)
        return _discover_stage2_modes(similarity_reports, skip_reports)

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
STAGE2_METRIC_ROW_POINT = "point"
STAGE2_TOKEN_ROW_SPECIAL = "special"
STAGE2_TOKEN_ROW_PATCH = "patch"
STAGE2_TOKEN_ROW_ALL = "all"


def _heatmaps_resolve_filters(
    controller: PairAttentionController,
    scene_id: Optional[str],
    value_mode: str,
    metric_rows: Optional[List[str]],
    token_rows: Optional[List[str]],
) -> Tuple[str, List[str], List[str], List[str], bool, bool, bool]:
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
    has_skip = bool(scene_id) and controller.get_skip_report(scene_id) is not None
    has_sim = bool(scene_id) and controller.get_similarity_report(scene_id) is not None

    if has_skip:
        allowed_m = {STAGE2_METRIC_ROW_POSE, STAGE2_METRIC_ROW_POINT}
        if metric_rows is None:
            m_val = [STAGE2_METRIC_ROW_POSE, STAGE2_METRIC_ROW_POINT]
        else:
            m_val = [x for x in metric_rows if x in allowed_m]
    else:
        m_val = []

    if has_sim:
        allowed_t = {STAGE2_TOKEN_ROW_SPECIAL, STAGE2_TOKEN_ROW_PATCH, STAGE2_TOKEN_ROW_ALL}
        if token_rows is None:
            t_val = [STAGE2_TOKEN_ROW_SPECIAL, STAGE2_TOKEN_ROW_PATCH, STAGE2_TOKEN_ROW_ALL]
        else:
            t_val = [x for x in token_rows if x in allowed_t]
    else:
        t_val = []

    return vm, m_val, t_val, choices_vm, has_skip, has_sim, pct_ok


def _heatmaps_gr_controls_from_resolved(
    vm: str,
    m_val: List[str],
    t_val: List[str],
    sim_gamma: float,
    sim_cutoff: float,
    _choices_vm: List[str],
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
    token_update: Dict[str, Any] = {"interactive": has_sim}
    sim_gamma_update: Dict[str, Any] = {"interactive": has_sim}
    sim_cutoff_update: Dict[str, Any] = {"interactive": has_sim}
    if write_values:
        metric_update["value"] = m_val
        token_update["value"] = t_val
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


def _discover_lora_experiments(root: Path) -> List[str]:
    if not root.exists():
        return []
    exps: List[str] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            exps.append(entry.name)
    return exps


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


def _first_existing(paths: Sequence[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _lora_expected_plot_paths(exp_dir: Path) -> Dict[str, Path]:
    """Paths for LoRA tab images.

    Training now saves ``eval_plot_{group}_{metric}.png`` (see ``_plot_metric_group`` in
    ``train_lora_skip_baseline.py``): ``scannet`` bundles dev/test ScanNet curves; ``7andn``
    bundles 7-scenes / RGB-D style metrics. Legacy unprefixed filenames are still tried.
    """
    return {
        # ScanNet group (row 1–2 in the LoRA tab)
        "eval_plot_are": _first_existing(
            [exp_dir / "eval_plot_scannet_are.png", exp_dir / "eval_plot_are.png"]
        ),
        "eval_plot_ate": _first_existing(
            [exp_dir / "eval_plot_scannet_ate.png", exp_dir / "eval_plot_ate.png"]
        ),
        "eval_plot_rpe_rot": _first_existing(
            [exp_dir / "eval_plot_scannet_rpe_rot.png", exp_dir / "eval_plot_rpe_rot.png"]
        ),
        "eval_plot_rpe_trans": _first_existing(
            [exp_dir / "eval_plot_scannet_rpe_trans.png", exp_dir / "eval_plot_rpe_trans.png"]
        ),
        "eval_plot_depth_abs_abs_rel": _first_existing(
            [
                exp_dir / "eval_plot_scannet_depth_abs_rel.png",
                exp_dir / "eval_plot_depth_abs_abs_rel.png",
                exp_dir / "eval_plot_depth_abs_rel.png",
            ]
        ),
        "eval_plot_depth_abs_delta1": _first_existing(
            [
                exp_dir / "eval_plot_scannet_depth_delta1.png",
                exp_dir / "eval_plot_depth_abs_delta1.png",
                exp_dir / "eval_plot_depth_delta1.png",
            ]
        ),
        "eval_plot_chamfer_distance": _first_existing(
            [exp_dir / "eval_plot_scannet_chamfer_distance.png", exp_dir / "eval_plot_chamfer_distance.png"]
        ),
        # 7-scenes / NRGBD group (row 3)
        "eval_plot_7andn_acc_med": exp_dir / "eval_plot_7andn_acc_med.png",
        "eval_plot_7andn_comp_med": exp_dir / "eval_plot_7andn_comp_med.png",
        "eval_plot_7andn_nc_med": exp_dir / "eval_plot_7andn_nc_med.png",
        # Training curves (row 4)
        "train_curves": exp_dir / "train_curves.png",
        "train_loss_weighted_components": exp_dir / "train_loss_weighted_components.png",
    }


def _stage2_row_visibility_flags(
    mode_order: List[str],
    selected_modes: List[str],
    metric_rows: List[str],
    token_rows: List[str],
) -> List[bool]:
    active_modes = {str(m) for m in selected_modes}
    metrics = {str(m) for m in metric_rows}
    tokens = {str(t) for t in token_rows}
    out: List[bool] = []
    for mode in mode_order:
        on = mode in active_modes
        out.append(on and STAGE2_METRIC_ROW_POSE in metrics)
        out.append(on and STAGE2_TOKEN_ROW_SPECIAL in tokens)
        out.append(on and STAGE2_METRIC_ROW_POINT in metrics)
        out.append(on and STAGE2_TOKEN_ROW_PATCH in tokens)
        out.append(on and STAGE2_TOKEN_ROW_ALL in tokens)
    return out


def _stage2_block_keys_for_mode(mode: str) -> Tuple[str, str, str, str, str]:
    return (
        f"skip_{mode}_main",
        f"sim_{mode}_special",
        f"skip_{mode}_depth",
        f"sim_{mode}_patch",
        f"sim_{mode}_all",
    )


def _stage2_requested_block_keys(
    mode_order: List[str],
    selected_modes: List[str],
    metric_rows: List[str],
    token_rows: List[str],
) -> set[str]:
    active_modes = {str(m) for m in selected_modes}
    metrics = {str(m) for m in metric_rows}
    tokens = {str(t) for t in token_rows}
    requested: set[str] = set()
    for mode in mode_order:
        if mode not in active_modes:
            continue
        skip_main, sim_special, skip_depth, sim_patch, sim_all = _stage2_block_keys_for_mode(mode)
        if STAGE2_METRIC_ROW_POSE in metrics:
            requested.add(skip_main)
        if STAGE2_TOKEN_ROW_SPECIAL in tokens:
            requested.add(sim_special)
        if STAGE2_METRIC_ROW_POINT in metrics:
            requested.add(skip_depth)
        if STAGE2_TOKEN_ROW_PATCH in tokens:
            requested.add(sim_patch)
        if STAGE2_TOKEN_ROW_ALL in tokens:
            requested.add(sim_all)
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

    fusion_by_mode = report.get("fusion_by_mode")
    if isinstance(fusion_by_mode, dict):
        for mode, mode_payload in fusion_by_mode.items():
            if not isinstance(mode_payload, dict):
                continue
            metric_map = mode_payload.get("gt_delta_upper_by_metric")
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
            all_clean = (
                {
                    str(k): packed
                    for k, packed in ((k, _coerce_packed(v)) for k, v in all_map.items())
                    if packed is not None
                }
                if isinstance(all_map, dict)
                else {}
            )
            if patch_clean or special_clean or all_clean:
                out[str(mode)] = {"patch": patch_clean, "special": special_clean, "all": all_clean}

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
    return {"patch": {}, "special": {}, "all": {}}


def _packed_delta_to_pct(
    packed: List[float],
    baseline_value: float,
    *,
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
        pct = 100.0 * scalar / denom
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
) -> np.ndarray:
    grid = upper_triangle_unpack_for_heatmap(packed, num_layers, fill_value)
    arr = np.asarray(grid, dtype=np.float64)
    finite_vals = arr[np.isfinite(arr)]
    if np.any(np.isnan(arr)):
        arr = np.ma.masked_invalid(arr)

    title_fs, label_fs, tick_fs = 18, 16, 14
    fig, ax = plt.subplots(figsize=(9, 8))
    imshow_kw: Dict[str, Any] = {"cmap": "viridis"}
    if vmin is not None:
        imshow_kw["vmin"] = vmin
    if vmax is not None:
        imshow_kw["vmax"] = vmax
    im = ax.imshow(arr, **imshow_kw)
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=12)
    ax.set_xlabel("layer j", fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("layer i", fontsize=label_fs, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_fs, width=2.0, length=6)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fs, width=2.0, length=6)
    for lb in cbar.ax.get_yticklabels():
        lb.set_fontweight("bold")
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


def _render_similarity_gallery(
    similarity_upper: Dict[str, Any],
    num_layers: int,
    *,
    role_short: str,
    sim_gamma: float,
    sim_cutoff: float,
) -> List[np.ndarray]:
    """role_short: one-letter tag, e.g. S (special) / P (patch) / A (all tokens)."""
    images: List[np.ndarray] = []
    tag = (role_short or "?").strip()[:1].upper()
    for metric_key, metric_label in SIMILARITY_METRICS:
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
        images.append(
            _heatmap_rgb_from_packed(
                packed,
                num_layers,
                f"{tag}·{metric_label}",
                vmin=vmin,
                vmax=vmax,
            )
        )
    return images


def _render_skip_gallery(
    skip_metric_map: Dict[str, List[float]],
    num_layers: int,
    metrics: List[Tuple[str, str]],
    *,
    value_mode: str,
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for metric_key, metric_label in metrics:
        packed = skip_metric_map.get(metric_key)
        if packed is None:
            continue
        if value_mode == "pct":
            title = f"{metric_label} %"
        else:
            title = metric_label
        images.append(
            _heatmap_rgb_from_packed(
                packed,
                num_layers,
                title,
                tick_suffix="%" if value_mode == "pct" else "",
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
    report = skip_report_entry.report
    skip_by_mode = _extract_skip_metric_maps_by_mode(report)
    if value_mode == "absolute":
        return skip_by_mode, "skip rows use saved absolute delta"

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
                clip_percent=pct_clip_percent,
            )
        if pct_metric_map:
            pct_by_mode[mode] = pct_metric_map
    return pct_by_mode, "skip rows use on-the-fly pct vs Stage1 baseline"


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
) -> Tuple[Dict[str, List[np.ndarray]], str]:
    clip_key = int(round(float(np.clip(pct_clip_percent, 0.0, 100.0))))
    gamma_key = _sim_gamma_key(sim_gamma)
    cutoff_key = _sim_cutoff_key(sim_cutoff)
    cache_key = (scene_id, value_mode, clip_key, gamma_key, cutoff_key)
    cached = controller.stage2_render_cache.get(cache_key)
    if cached is not None:
        galleries, info = cached
    else:
        galleries = {}
        info = ""

    all_blocks: set[str] = set()
    for mode in mode_order:
        all_blocks.update(_stage2_block_keys_for_mode(mode))
    target_blocks = set(requested_blocks) if requested_blocks is not None else all_blocks
    if all(key in galleries for key in target_blocks) and info:
        return galleries, info

    similarity_report = controller.get_similarity_report(scene_id)
    num_layers = int(controller.resolve_stage2_num_layers(scene_id))
    if num_layers <= 0:
        raise gr.Error(f"No similarity/skip report with num_layers found for {scene_id!r}.")

    similarity_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    if similarity_report is not None:
        similarity_by_mode = _extract_similarity_metric_maps_by_mode(similarity_report.report)

    skip_by_mode, skip_source = _build_stage2_skip_maps(controller, scene_id, value_mode, float(clip_key))

    for mode in mode_order:
        skip_main, sim_special, skip_depth, sim_patch, sim_all = _stage2_block_keys_for_mode(mode)
        if skip_main in target_blocks and skip_main not in galleries:
            galleries[skip_main] = _render_skip_gallery(
                skip_by_mode.get(mode, {}),
                num_layers,
                SKIP_MAIN_METRICS,
                value_mode=value_mode,
            )
        sim_entry = _similarity_maps_for_skip_mode(similarity_by_mode, mode)
        if sim_special in target_blocks and sim_special not in galleries:
            galleries[sim_special] = _render_similarity_gallery(
                sim_entry.get("special", {}),
                num_layers,
                role_short="S",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if skip_depth in target_blocks and skip_depth not in galleries:
            galleries[skip_depth] = _render_skip_gallery(
                skip_by_mode.get(mode, {}),
                num_layers,
                SKIP_DEPTH_METRICS,
                value_mode=value_mode,
            )
        if sim_patch in target_blocks and sim_patch not in galleries:
            galleries[sim_patch] = _render_similarity_gallery(
                sim_entry.get("patch", {}),
                num_layers,
                role_short="P",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )
        if sim_all in target_blocks and sim_all not in galleries:
            galleries[sim_all] = _render_similarity_gallery(
                sim_entry.get("all", {}),
                num_layers,
                role_short="A",
                sim_gamma=sim_gamma,
                sim_cutoff=sim_cutoff,
            )

    rendered = len(galleries)
    total = len(all_blocks)
    info = f"scene={scene_id} | view={value_mode} | rendered={rendered}/{total} blocks | {skip_source} | pct_clip={clip_key}%"
    if abs(sim_gamma - 1.0) > 1e-12:
        info += f" | sim_gamma={gamma_key / 1000.0:.3f}"
    if cutoff_key > 0:
        info += f" | sim_cutoff={cutoff_key / 1000.0:.3f}"
    controller.stage2_render_cache[cache_key] = (galleries, info)
    return galleries, info


def _build_ui(controller: PairAttentionController) -> gr.Blocks:
    dataset_choices = ["scannet", "7scenes", "nrgbd"]
    scene_ids = controller.scene_ids()
    pair_enabled = bool(scene_ids)
    initial_scene = scene_ids[0] if scene_ids else None
    if pair_enabled and initial_scene is not None:
        num_layers, num_frames, frame_labels = controller.scene_metadata_light(initial_scene)
        num_heads = controller.pair_ui_num_heads()
    else:
        num_layers, num_frames, frame_labels = 0, 0, []
        num_heads = 1
    initial_frame_a = 0
    initial_frame_b = min(1, len(frame_labels) - 1) if frame_labels else 0
    stage2_modes_all: List[str] = []
    for ds in dataset_choices:
        for mode in controller.stage2_modes_for_dataset(ds):
            if mode not in stage2_modes_all:
                stage2_modes_all.append(mode)
    if not stage2_modes_all:
        stage2_modes_all = _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)
    n_stage2_galleries = 5 * len(stage2_modes_all)
    lora_root = controller.lora_root
    stage2_last_render_key: Dict[str, Optional[Tuple[str, str, int, int, int]]] = {"cache_key": None}
    dataset_switch_output_len: Dict[str, int] = {"count": 0}

    def _empty_stage2_visual_updates() -> Tuple[Any, ...]:
        gal = tuple(gr.update(value=[]) for _ in range(n_stage2_galleries))
        col = tuple(gr.update(visible=False) for _ in range(n_stage2_galleries))
        return (*gal, *col, "No Stage2 scenes for this dataset.")

    def _stage2_noop_updates() -> Tuple[Any, ...]:
        return tuple(gr.update() for _ in range(2 * n_stage2_galleries + 7))

    def _dataset_noop_updates() -> Tuple[Any, ...]:
        count = max(0, int(dataset_switch_output_len["count"]))
        return tuple(gr.update() for _ in range(count))

    def _stage2_visual_bundle(
        scene_id: Optional[str],
        value_mode: str,
        pct_clip_percent: float,
        filter_modes: List[str] | None,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma_value: float,
        sim_cutoff_value: float,
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
        if not scene_id:
            stage2_last_render_key["cache_key"] = None
            vm, m_val, t_val, choices_vm, has_skip, has_sim, pct_ok = _heatmaps_resolve_filters(
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
                has_skip,
                has_sim,
                pct_ok,
                write_values=True,
            )
            if request_seq is not None and request_seq != controller.stage2_request_seq:
                return _stage2_noop_updates()
            return (*_empty_stage2_visual_updates(), *ctrl)

        vm, m_val, t_val, choices_vm, has_skip, has_sim, pct_ok = _heatmaps_resolve_filters(
            controller,
            scene_id,
            str(value_mode or "absolute"),
            list(filter_metric_rows) if filter_metric_rows is not None else None,
            list(filter_token_rows) if filter_token_rows is not None else None,
        )
        current_modes = _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)
        allowed_modes = set(current_modes)
        if filter_modes is None:
            selected_modes = list(current_modes)
        else:
            selected_modes = [str(m) for m in filter_modes if str(m) in allowed_modes]
            if filter_modes and not selected_modes and current_modes:
                selected_modes = list(current_modes)

        clip_key = int(round(float(np.clip(float(pct_clip_percent), 0.0, 100.0))))
        gamma_key = _sim_gamma_key(sim_gamma)
        cutoff_key = _sim_cutoff_key(sim_cutoff)
        cache_key = (str(scene_id), vm, clip_key, gamma_key, cutoff_key)
        force_updates = force_gallery_refresh or stage2_last_render_key["cache_key"] != cache_key
        cached_before = controller.stage2_render_cache.get(cache_key)
        before_keys = set(cached_before[0].keys()) if cached_before is not None else set()
        requested_blocks = _stage2_requested_block_keys(stage2_modes_all, selected_modes, m_val, t_val)
        galleries, details = _prepare_stage2_render(
            controller,
            scene_id,
            vm,
            pct_clip_percent,
            sim_gamma,
            sim_cutoff,
            stage2_modes_all,
            requested_blocks=requested_blocks,
        )
        after_keys = set(galleries.keys())
        keys_to_update = (requested_blocks & after_keys) if force_updates else (requested_blocks - before_keys)

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
            has_skip,
            has_sim,
            pct_ok,
            write_values=write_values,
        )
        stage2_last_render_key["cache_key"] = cache_key
        return tuple([*block_updates, *vis_updates, details, *ctrl])

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

    def _dataset_changed(
        dataset: str,
        cur_value_mode: str,
        pct_clip: float,
        filter_modes: List[str] | None,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma: float,
        sim_cutoff: float,
    ):
        # Invalidate outstanding Stage2 refresh requests from previous state.
        controller.stage2_request_seq += 1
        controller.switch_dataset(dataset)
        current_modes = _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)
        current_modes_set = set(current_modes)
        if filter_modes is None:
            selected_modes = list(current_modes)
        else:
            selected_modes = [str(m) for m in filter_modes if str(m) in current_modes_set]
            if filter_modes and not selected_modes and current_modes:
                selected_modes = list(current_modes)
        scene_ids_new = controller.scene_ids()
        stage2_scene_ids_new = controller.stage2_scene_ids()
        initial_scene_new = scene_ids_new[0] if scene_ids_new else None
        initial_stage2_scene_new = stage2_scene_ids_new[0] if stage2_scene_ids_new else None
        tup_hm = _stage2_visual_bundle(
            initial_stage2_scene_new,
            str(cur_value_mode or "absolute"),
            float(pct_clip),
            selected_modes,
            list(filter_metric_rows) if filter_metric_rows is not None else None,
            list(filter_token_rows) if filter_token_rows is not None else None,
            float(sim_gamma),
            float(sim_cutoff),
            request_seq=None,
            force_gallery_refresh=True,
        )
        hm_main = tup_hm[:-7]
        hm_info = tup_hm[-7]
        hm_ctrl = tup_hm[-6:]

        if initial_scene_new is None:
            return (
                gr.update(value=controller.dataset),
                gr.update(value=controller.dataset),
                gr.update(choices=scene_ids_new, value=None),
                "layers: 0, frames: 0",
                gr.update(maximum=0, value=0),
                gr.update(maximum=0, value=0),
                gr.update(maximum=0, value=0),
                None,
                f"No scenes found under {controller.cache_dir}",
                "",
                "",
                gr.update(choices=stage2_scene_ids_new, value=initial_stage2_scene_new),
                gr.update(choices=current_modes, value=selected_modes),
                *hm_main,
                hm_info,
                *hm_ctrl,
            )

        max_layer_u, fa_u, fb_u, info_u, panel_u, summary_u, fa_info_u, fb_info_u = _scene_changed(
            initial_scene_new,
            score_space="before_softmax",
            head_mode="mean",
            head_index=0,
        )
        return (
            gr.update(value=controller.dataset),
            gr.update(value=controller.dataset),
            gr.update(choices=scene_ids_new, value=initial_scene_new),
            info_u,
            max_layer_u,
            fa_u,
            fb_u,
            panel_u,
            summary_u,
            fa_info_u,
            fb_info_u,
            gr.update(choices=stage2_scene_ids_new, value=initial_stage2_scene_new),
            gr.update(choices=current_modes, value=selected_modes),
            *hm_main,
            hm_info,
            *hm_ctrl,
        )

    def _refresh_stage2_full(
        scene_id: str,
        value_mode: str,
        pct_clip_percent: float,
        filter_modes: List[str] | None,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
        sim_gamma: float,
        sim_cutoff: float,
    ):
        controller.stage2_request_seq += 1
        request_seq = int(controller.stage2_request_seq)
        return _stage2_visual_bundle(
            scene_id,
            str(value_mode or "absolute"),
            float(pct_clip_percent),
            filter_modes,
            filter_metric_rows,
            filter_token_rows,
            float(sim_gamma),
            float(sim_cutoff),
            request_seq=request_seq,
            force_gallery_refresh=False,
        )

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
        /* Pct radio option: keep visible but non-clickable when Stage1 eval is missing (structure varies by Gradio version). */
        .hm-value-mode-pct-locked .wrap label:last-of-type,
        .hm-value-mode-pct-locked label:last-of-type {
            opacity: 0.45 !important;
            pointer-events: none !important;
            cursor: not-allowed !important;
        }
        """,
    ) as demo:
        build_lora_tab(
            blocks=demo,
            lora_root=Path(lora_root),
            discover_lora_experiments=_discover_lora_experiments,
            lora_expected_plot_paths=_lora_expected_plot_paths,
        )

        pair_ui = build_pair_tab(
            controller=controller,
            dataset_choices=dataset_choices,
            scene_ids=scene_ids,
            initial_scene=initial_scene,
            num_layers=num_layers,
            num_frames=num_frames,
            frame_labels=frame_labels,
            initial_frame_a=initial_frame_a,
            initial_frame_b=initial_frame_b,
            num_heads=num_heads,
            scene_changed_cb=_scene_changed,
            refresh_outputs_cb=_refresh_outputs,
        )

        stage2_scene_ids = controller.stage2_scene_ids()
        initial_stage2_scene = stage2_scene_ids[0] if stage2_scene_ids else None
        initial_stage2_modes = _discover_stage2_modes(controller.similarity_reports, controller.skip_reports)
        heatmaps_ui = build_heatmaps_tab(
            controller=controller,
            dataset_choices=dataset_choices,
            stage2_modes=stage2_modes_all,
            initial_mode_values=initial_stage2_modes,
            stage2_scene_ids=stage2_scene_ids,
            initial_stage2_scene=initial_stage2_scene,
            stage2_metric_row_pose=STAGE2_METRIC_ROW_POSE,
            stage2_metric_row_point=STAGE2_METRIC_ROW_POINT,
            stage2_token_row_special=STAGE2_TOKEN_ROW_SPECIAL,
            stage2_token_row_patch=STAGE2_TOKEN_ROW_PATCH,
            stage2_token_row_all=STAGE2_TOKEN_ROW_ALL,
            heatmaps_resolve_filters_cb=_heatmaps_resolve_filters,
            refresh_stage2_full_cb=_refresh_stage2_full,
        )

        dataset_switch_outputs = [
            pair_ui["dataset_dd"],
            heatmaps_ui["stage2_dataset_dd"],
            pair_ui["scene_dd"],
            pair_ui["scene_info"],
            pair_ui["layer_slider"],
            pair_ui["frame_a_slider"],
            pair_ui["frame_b_slider"],
            pair_ui["panel"],
            pair_ui["summary"],
            pair_ui["frame_a_info"],
            pair_ui["frame_b_info"],
            heatmaps_ui["stage2_scene_dd"],
            heatmaps_ui["stage2_mode_filter"],
            *heatmaps_ui["stage2_mode_galleries"],
            *heatmaps_ui["stage2_row_body_cols"],
            heatmaps_ui["stage2_info"],
            heatmaps_ui["stage2_value_mode"],
            heatmaps_ui["stage2_pct_clip_slider"],
            heatmaps_ui["stage2_metric_rows_filter"],
            heatmaps_ui["stage2_token_rows_filter"],
            heatmaps_ui["stage2_sim_gamma_slider"],
            heatmaps_ui["stage2_sim_cutoff_slider"],
        ]
        dataset_switch_output_len["count"] = len(dataset_switch_outputs)
        pair_ui["dataset_dd"].change(
            _dataset_changed,
            inputs=[
                pair_ui["dataset_dd"],
                heatmaps_ui["stage2_value_mode"],
                heatmaps_ui["stage2_pct_clip_slider"],
                heatmaps_ui["stage2_mode_filter"],
                heatmaps_ui["stage2_metric_rows_filter"],
                heatmaps_ui["stage2_token_rows_filter"],
                heatmaps_ui["stage2_sim_gamma_slider"],
                heatmaps_ui["stage2_sim_cutoff_slider"],
            ],
            outputs=dataset_switch_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        heatmaps_ui["stage2_dataset_dd"].change(
            _dataset_changed,
            inputs=[
                heatmaps_ui["stage2_dataset_dd"],
                heatmaps_ui["stage2_value_mode"],
                heatmaps_ui["stage2_pct_clip_slider"],
                heatmaps_ui["stage2_mode_filter"],
                heatmaps_ui["stage2_metric_rows_filter"],
                heatmaps_ui["stage2_token_rows_filter"],
                heatmaps_ui["stage2_sim_gamma_slider"],
                heatmaps_ui["stage2_sim_cutoff_slider"],
            ],
            outputs=dataset_switch_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )

        build_eval_tab(
            blocks=demo,
            controller=controller,
            eval_viser_controller_cls=EvalViserController,
            load_image_if_exists=_load_image_if_exists,
        )

    return demo


def main() -> None:
    args = _build_parser().parse_args()
    if args.analysis_dir is not None:
        analysis_sim_dir = args.analysis_dir
        analysis_skip_dir = args.analysis_dir
    else:
        analysis_sim_dir = args.analysis_sim_dir
        analysis_skip_dir = args.analysis_skip_dir
    controller = PairAttentionController(
        cache_dir=args.cache_dir,
        analysis_sim_dir=analysis_sim_dir,
        analysis_skip_dir=analysis_skip_dir,
        data_root=args.data_root,
        dataset=args.dataset,
        data_roots_by_dataset={
            "scannet": args.scannet_root,
            "7scenes": args.root_7scenes,
            "nrgbd": args.root_nrgbd,
        },
        hf_model=args.hf_model,
        device_arg=args.device,
        lora_root=args.lora_dir,
        eval_root=args.eval_root,
        viser_host=args.viser_host,
        viser_port=args.viser_port,
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


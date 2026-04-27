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

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard for optional demo dependency
    raise SystemExit(
        "Missing optional dependency 'gradio'. Install demo dependencies first, "
        "for example: `pip install -r requirements_demo.txt` or `pip install .[demo]`."
    ) from exc

from vggt.models.vggt import VGGT

from .metrics import upper_triangle_unpack_for_heatmap
from .scannet_eval import SceneData, discover_scannet_scenes, load_scene_images
from .visualize_pair_attention import _build_global_positions, _load_cache_row, _overlay_patch_map, _reconstruct_qk


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive browser panel for pair attention.")
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/scannet_cache"))
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Legacy single analysis dir. If set, used for both similarity and skip reports.",
    )
    p.add_argument("--analysis-sim-dir", type=Path, default=Path("analysis_outputs/scannet_redundancy_similarity"))
    p.add_argument("--analysis-skip-dir", type=Path, default=Path("analysis_outputs/scannet_redundancy_skip"))
    p.add_argument("--data-root", type=Path, default=Path("data/scans_dev50/"))
    p.add_argument("--hf-model", type=str, default="", help="Optional HF model override; default reads manifest.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--server-name", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7861)
    return p


@dataclass
class SceneCache:
    item: Dict[str, Any]
    row: Dict[str, Any]
    scene_eff: SceneData
    images: torch.Tensor
    frame_ids: List[int]
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
        hf_model: str,
        device_arg: str,
    ) -> None:
        self.cache_dir = cache_dir
        self.analysis_similarity_dir = analysis_sim_dir
        self.analysis_skip_dir = analysis_skip_dir
        self.data_root = data_root
        self.device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")

        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing {manifest_path}; run Stage1 first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.scene_items: Dict[str, Dict[str, Any]] = {}
        for item in self.manifest.get("scenes", []):
            scene_id = str(item.get("scene_id", "")).strip()
            has_dir = str(item.get("dir", "")).strip() != ""
            has_file = str(item.get("file", "")).strip() != ""
            if scene_id and (has_dir or has_file):
                self.scene_items[scene_id] = item
        if not self.scene_items:
            raise SystemExit(f"No cached scenes found in {cache_dir}.")

        self.discovered = {s.scene_id: s for s in discover_scannet_scenes(data_root)}
        self.hf_model = hf_model.strip() or str(self.manifest.get("hf_model", "")).strip()
        if not self.hf_model:
            raise SystemExit("No HF model provided and manifest.json lacks 'hf_model'.")

        self.model: Optional[VGGT] = None
        self.scene_cache: Dict[str, SceneCache] = {}
        self.qk_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.stage1_eval_by_scene = self._load_stage1_eval_by_scene()
        self.similarity_reports = self._discover_similarity_reports()
        self.skip_reports = self._discover_skip_reports()
        self.stage2_render_cache: Dict[Tuple[str, str, int], Tuple[Dict[str, List[np.ndarray]], str]] = {}

    def _get_model(self) -> VGGT:
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

    def _resolve_scene_eff(self, scene_id: str, frame_ids: List[int]) -> SceneData:
        if scene_id not in self.discovered:
            raise SystemExit(f"Scene {scene_id!r} not found under {self.data_root}.")
        scene = self.discovered[scene_id]
        frame_to_path = {int(p.stem): p for p in scene.image_paths}
        selected_paths: List[Path] = []
        missing: List[int] = []
        for fid in frame_ids:
            path = frame_to_path.get(int(fid))
            if path is None:
                missing.append(int(fid))
            else:
                selected_paths.append(path)
        if missing:
            raise SystemExit(f"Scene {scene_id!r} is missing frame files for ids: {missing}")
        return SceneData(scene_id=scene.scene_id, scene_dir=scene.scene_dir, image_paths=selected_paths)

    def get_scene_cache(self, scene_id: str) -> SceneCache:
        cached = self.scene_cache.get(scene_id)
        if cached is not None:
            return cached

        item = self.scene_items[scene_id]
        frame_ids = [int(x) for x in item.get("frame_ids", [])]
        if not frame_ids:
            raise SystemExit(f"Manifest entry for {scene_id!r} lacks frame_ids.")
        row = _load_cache_row(self.cache_dir, item)
        replay_layers = row.get("aggregated_concat_per_layer")
        if not isinstance(replay_layers, list) or not replay_layers:
            raise SystemExit(f"Scene {scene_id!r} cache lacks aggregated_concat_per_layer. Please rerun Stage1.")

        scene_eff = self._resolve_scene_eff(scene_id, frame_ids)
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
        eval_path = self.cache_dir / "eval_scannet.json"
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

    def _discover_similarity_reports(self) -> Dict[str, Stage2SceneReport]:
        result: Dict[str, Stage2SceneReport] = {}
        if not self.analysis_similarity_dir.exists():
            return result

        mean_report = self._load_json_if_exists(self.analysis_similarity_dir / "full_report_similarity.json")
        if mean_report:
            result["mean"] = Stage2SceneReport(scene_id="mean", report=mean_report)

        per_scene_dir = self.analysis_similarity_dir / "per_scene"
        if per_scene_dir.exists():
            for activation_json in sorted(per_scene_dir.glob("*_similarity.json")):
                payload = self._load_json_if_exists(activation_json)
                scene_id = str(payload.get("scene_id", "")).strip()
                if scene_id:
                    result[scene_id] = Stage2SceneReport(scene_id=scene_id, report=payload)
        return result

    def _discover_skip_reports(self) -> Dict[str, Stage2SceneReport]:
        result: Dict[str, Stage2SceneReport] = {}
        if not self.analysis_skip_dir.exists():
            return result

        mean_report = self._load_json_if_exists(self.analysis_skip_dir / "full_report_skip.json")
        if mean_report:
            result["mean"] = Stage2SceneReport(scene_id="mean", report=mean_report)

        per_scene_dir = self.analysis_skip_dir / "per_scene"
        if per_scene_dir.exists():
            for activation_json in sorted(per_scene_dir.glob("*_skip.json")):
                payload = self._load_json_if_exists(activation_json)
                scene_id = str(payload.get("scene_id", "")).strip()
                if scene_id:
                    result[scene_id] = Stage2SceneReport(scene_id=scene_id, report=payload)
        return result

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


def _gallery_update(items: List[np.ndarray]):
    return gr.update(value=items, columns=max(1, len(items)))


STAGE2_METRIC_ROW_POSE = "pose"
STAGE2_METRIC_ROW_POINT = "point"
STAGE2_TOKEN_ROW_SPECIAL = "special"
STAGE2_TOKEN_ROW_PATCH = "patch"


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
    return out


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
            clean = {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items()) if packed is not None}
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
            clean = {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items()) if packed is not None}
            if clean:
                out[_canonical_skip_mode(str(mode))] = clean

    by_mode = report.get("global_skip_range_loss_upper_by_metric_by_mode")
    if isinstance(by_mode, dict):
        for mode, metric_map in by_mode.items():
            if not isinstance(metric_map, dict):
                continue
            clean = {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items()) if packed is not None}
            if clean:
                out[_canonical_skip_mode(str(mode))] = clean

    single = report.get("global_skip_range_loss")
    if isinstance(single, dict):
        mode = _canonical_skip_mode(str(single.get("skipped_global_mode", "")).strip())
        metric_map = single.get("skip_range_loss_upper_by_metric")
        if not isinstance(metric_map, dict):
            metric_map = single.get("global_skip_range_loss_upper_by_metric")
        if mode and isinstance(metric_map, dict):
            clean = {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in metric_map.items()) if packed is not None}
            if clean:
                out[mode] = clean

    single_mode = _canonical_skip_mode(str(report.get("skipped_global_mode", "")).strip())
    single_metric_map = report.get("global_skip_range_loss_upper_by_metric")
    if single_mode and isinstance(single_metric_map, dict):
        clean = {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in single_metric_map.items()) if packed is not None}
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
            patch_clean = (
                {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in patch_map.items()) if packed is not None}
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
            if patch_clean or special_clean:
                out[str(mode)] = {"patch": patch_clean, "special": special_clean}

    # Legacy fallback: single-mode top-level similarity maps.
    legacy_patch = report.get("global_patch_similarity_upper")
    legacy_special = report.get("global_special_similarity_upper")
    if isinstance(legacy_patch, dict) or isinstance(legacy_special, dict):
        patch_clean = (
            {str(k): packed for k, packed in ((k, _coerce_packed(v)) for k, v in legacy_patch.items()) if packed is not None}
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
            out.setdefault("default", {"patch": patch_clean, "special": special_clean})

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
    return {"patch": {}, "special": {}}


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


def _render_similarity_gallery(
    similarity_upper: Dict[str, Any],
    num_layers: int,
    *,
    role_short: str,
) -> List[np.ndarray]:
    """role_short: one-letter tag, e.g. S (special) / P (patch)."""
    images: List[np.ndarray] = []
    tag = (role_short or "?").strip()[:1].upper()
    for metric_key, metric_label in SIMILARITY_METRICS:
        packed = _coerce_packed(similarity_upper.get(metric_key))
        if packed is None:
            continue
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
    frame_a_id: int,
    frame_b_id: int,
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
                f"Check {controller.cache_dir / 'eval_scannet.json'} has Stage1 metrics for "
                f"scene_ids in full_report_skip.json."
            )
        detail = f"{n_fused} scenes" if n_fused > 0 else "full_report"
        return pct_by_mode, f"mean Δ from full_report, pct+clip vs mean Stage1 ({detail})"

    baseline_row = controller.stage1_eval_by_scene.get(scene_id)
    if baseline_row is None:
        raise gr.Error(f"Missing Stage1 baseline metrics for {scene_id!r} in {controller.cache_dir / 'eval_scannet.json'}.")

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
    mode_order: List[str],
) -> Tuple[Dict[str, List[np.ndarray]], str]:
    clip_key = int(round(float(np.clip(pct_clip_percent, 0.0, 100.0))))
    cache_key = (scene_id, value_mode, clip_key)
    cached = controller.stage2_render_cache.get(cache_key)
    if cached is not None:
        return cached

    similarity_report = controller.get_similarity_report(scene_id)
    skip_report = controller.get_skip_report(scene_id)
    num_layers = int(controller.resolve_stage2_num_layers(scene_id))
    if num_layers <= 0:
        raise gr.Error(f"No similarity/skip report with num_layers found for {scene_id!r}.")

    similarity_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    if similarity_report is not None:
        similarity_by_mode = _extract_similarity_metric_maps_by_mode(similarity_report.report)

    skip_by_mode, skip_source = _build_stage2_skip_maps(controller, scene_id, value_mode, float(clip_key))

    galleries: Dict[str, List[np.ndarray]] = {}
    for mode in mode_order:
        galleries[f"skip_{mode}_main"] = _render_skip_gallery(
            skip_by_mode.get(mode, {}),
            num_layers,
            SKIP_MAIN_METRICS,
            value_mode=value_mode,
        )
        sim_entry = _similarity_maps_for_skip_mode(similarity_by_mode, mode)
        galleries[f"sim_{mode}_special"] = _render_similarity_gallery(
            sim_entry.get("special", {}),
            num_layers,
            role_short="S",
        )
        galleries[f"skip_{mode}_depth"] = _render_skip_gallery(
            skip_by_mode.get(mode, {}),
            num_layers,
            SKIP_DEPTH_METRICS,
            value_mode=value_mode,
        )
        galleries[f"sim_{mode}_patch"] = _render_similarity_gallery(
            sim_entry.get("patch", {}),
            num_layers,
            role_short="P",
        )

    mode_summaries = " | ".join(
        f"{mode}: skip {len(galleries[f'skip_{mode}_main'])+len(galleries[f'skip_{mode}_depth'])}, "
        f"sim {len(galleries[f'sim_{mode}_special'])+len(galleries[f'sim_{mode}_patch'])}"
        for mode in mode_order
    )
    info = (
        f"scene={scene_id} | view={value_mode} | {mode_summaries} | {skip_source} | pct_clip={clip_key}%"
    )
    controller.stage2_render_cache[cache_key] = (galleries, info)
    return galleries, info


def _build_ui(controller: PairAttentionController) -> gr.Blocks:
    scene_ids = controller.scene_ids()
    initial_scene = scene_ids[0]
    num_layers, num_frames, frame_labels = controller.scene_metadata(initial_scene)
    initial_frame_a = 0
    initial_frame_b = min(1, len(frame_labels) - 1) if frame_labels else 0
    num_heads = controller.num_heads()
    stage2_skip_modes = _discover_skip_modes(controller.skip_reports)

    def _refresh_outputs(
        scene_id: str,
        layer_idx: int,
        frame_a_index: int,
        frame_b_index: int,
        score_space: str,
        head_mode: str,
        head_index: int,
    ):
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
        num_layers_, num_frames_, labels = controller.scene_metadata(scene_id)
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

    def _refresh_stage2_full(
        scene_id: str,
        value_mode: str,
        pct_clip_percent: float,
        filter_modes: List[str] | None,
        filter_metric_rows: List[str] | None,
        filter_token_rows: List[str] | None,
    ):
        galleries, details = _prepare_stage2_render(
            controller, scene_id, value_mode, pct_clip_percent, stage2_skip_modes
        )
        block_updates: List[Any] = []
        for mode in stage2_skip_modes:
            block_updates.append(_gallery_update(galleries[f"skip_{mode}_main"]))
            block_updates.append(_gallery_update(galleries[f"sim_{mode}_special"]))
            block_updates.append(_gallery_update(galleries[f"skip_{mode}_depth"]))
            block_updates.append(_gallery_update(galleries[f"sim_{mode}_patch"]))
        vis_flags = _stage2_row_visibility_flags(
            stage2_skip_modes,
            list(filter_modes or []),
            list(filter_metric_rows or []),
            list(filter_token_rows or []),
        )
        vis_updates = [gr.update(visible=bool(v)) for v in vis_flags]
        return tuple([*block_updates, *vis_updates, details])

    stage2_scene_ids = controller.stage2_scene_ids()
    initial_stage2_scene = stage2_scene_ids[0] if stage2_scene_ids else None

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
        """,
    ) as demo:
        with gr.Tab("Pair Attention"):
            gr.Markdown("## Pair Attention Panel")
            gr.Markdown(
                "Choose a Stage1-cached scene, global layer, two frames, score space, and head view. "
                "The panel refreshes automatically and shows overlay summaries for both directions: `i -> j` and `j -> i`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    scene_dd = gr.Dropdown(label="Scene", choices=scene_ids, value=initial_scene)
                    scene_info = gr.Textbox(
                        label="Scene metadata",
                        value=f"layers: {num_layers}, frames: {num_frames}",
                        interactive=False,
                    )
                    layer_slider = gr.Slider(label="Global layer", minimum=0, maximum=max(0, num_layers - 1), value=0, step=1)
                    frame_a_slider = gr.Slider(label="Frame A", minimum=0, maximum=max(0, num_frames - 1), value=initial_frame_a, step=1)
                    frame_a_info = gr.Textbox(
                        label="Frame A selection",
                        value=frame_labels[initial_frame_a] if frame_labels else "",
                        interactive=False,
                    )
                    frame_b_slider = gr.Slider(label="Frame B", minimum=0, maximum=max(0, num_frames - 1), value=initial_frame_b, step=1)
                    frame_b_info = gr.Textbox(
                        label="Frame B selection",
                        value=frame_labels[initial_frame_b] if frame_labels else "",
                        interactive=False,
                    )
                    score_space = gr.Radio(
                        label="Score space",
                        choices=["before_softmax", "after_softmax"],
                        value="before_softmax",
                    )
                    head_mode = gr.Radio(label="Head view", choices=["mean", "single"], value="mean")
                    head_index = gr.Slider(label="Head index", minimum=0, maximum=max(0, num_heads - 1), value=0, step=1)
                    summary = gr.Textbox(label="Selection", interactive=False)
                with gr.Column(scale=5):
                    panel = gr.Image(label="Bidirectional pair attention panel", type="numpy", elem_classes=["results-media"])

            refresh_inputs = [scene_dd, layer_slider, frame_a_slider, frame_b_slider, score_space, head_mode, head_index]
            refresh_outputs = [panel, summary, frame_a_info, frame_b_info]

            scene_dd.change(
                _scene_changed,
                inputs=[scene_dd, score_space, head_mode, head_index],
                outputs=[layer_slider, frame_a_slider, frame_b_slider, scene_info, panel, summary, frame_a_info, frame_b_info],
            )
            layer_slider.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)
            frame_a_slider.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)
            frame_b_slider.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)
            score_space.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)
            head_mode.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)
            head_index.change(_refresh_outputs, inputs=refresh_inputs, outputs=refresh_outputs)

            demo.load(
                _refresh_outputs,
                inputs=refresh_inputs,
                outputs=refresh_outputs,
            )

        with gr.Tab("Stage2 Heatmaps"):
            gr.Markdown("## Stage2 Heatmaps")
            gr.Markdown(
                "Choose `mean` or a scene. Per skip mode: pose → special sim → depth → patch sim. "
                "Use the **left filters** to show or hide fusion modes and row groups. "
                "`pct` applies to skip rows only (mean uses full_report Δ then pct+clip vs mean Stage1)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stage2_scene_dd = gr.Dropdown(label="Scene", choices=stage2_scene_ids, value=initial_stage2_scene)
                    stage2_value_mode = gr.Radio(
                        label="Skip heatmap value",
                        choices=["absolute", "pct"],
                        value="absolute",
                    )
                    stage2_pct_clip_slider = gr.Slider(
                        label="Pct clip threshold (%)",
                        minimum=0,
                        maximum=100,
                        value=100,
                        step=1,
                    )
                    stage2_mode_filter = gr.CheckboxGroup(
                        label="Fusion / skip modes",
                        choices=list(stage2_skip_modes),
                        value=list(stage2_skip_modes),
                    )
                    stage2_metric_rows_filter = gr.CheckboxGroup(
                        label="Skip Δ rows",
                        choices=[STAGE2_METRIC_ROW_POSE, STAGE2_METRIC_ROW_POINT],
                        value=[STAGE2_METRIC_ROW_POSE, STAGE2_METRIC_ROW_POINT],
                        info="pose: ARE/ATE/RPE. point: depth & Chamfer.",
                    )
                    stage2_token_rows_filter = gr.CheckboxGroup(
                        label="Token similarity rows",
                        choices=[STAGE2_TOKEN_ROW_SPECIAL, STAGE2_TOKEN_ROW_PATCH],
                        value=[STAGE2_TOKEN_ROW_SPECIAL, STAGE2_TOKEN_ROW_PATCH],
                        info="special- vs patch-token heatmaps.",
                    )
                    stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
                with gr.Column(scale=5, elem_classes=["stage2-right"]):
                    stage2_mode_galleries: List[gr.Gallery] = []
                    stage2_row_body_cols: List[gr.Column] = []
                    for mode in stage2_skip_modes:
                        for label, n_columns in (
                            (f"{mode}: pose", 4),
                            (f"{mode}: sim · special", 4),
                            (f"{mode}: depth", 3),
                            (f"{mode}: sim · patch", 4),
                        ):
                            with gr.Row(elem_classes=["stage2-gallery-row"], equal_height=False):
                                with gr.Column(scale=1, min_width=280) as row_body:
                                    gal = gr.Gallery(
                                        label=label,
                                        columns=n_columns,
                                        height="auto",
                                        object_fit="contain",
                                        elem_classes=["results-gallery", "stage2-heatmap-gallery"],
                                    )
                                stage2_row_body_cols.append(row_body)
                                stage2_mode_galleries.append(gal)

            stage2_inputs = [
                stage2_scene_dd,
                stage2_value_mode,
                stage2_pct_clip_slider,
                stage2_mode_filter,
                stage2_metric_rows_filter,
                stage2_token_rows_filter,
            ]
            stage2_outputs = [*stage2_mode_galleries, *stage2_row_body_cols, stage2_info]

            if stage2_scene_ids:
                stage2_scene_dd.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                stage2_value_mode.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                stage2_pct_clip_slider.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                stage2_mode_filter.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                stage2_metric_rows_filter.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                stage2_token_rows_filter.change(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
                demo.load(_refresh_stage2_full, inputs=stage2_inputs, outputs=stage2_outputs)
            else:
                stage2_info.value = (
                    f"No Stage2/Stage3 reports found under "
                    f"sim={controller.analysis_similarity_dir} and skip={controller.analysis_skip_dir}"
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
        hf_model=args.hf_model,
        device_arg=args.device,
    )
    demo = _build_ui(controller)
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, show_error=True, share=True)


if __name__ == "__main__":
    main()

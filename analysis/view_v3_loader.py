"""Helpers for loading v3 pipeline artifacts in the view panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from analysis.prune_plan import UnitPrunePlan
from analysis.redundancy.metrics import upper_triangle_pack
from analysis.schema.v3 import (
    SCHEMA_VERSION,
    CacheManifestV3,
    EvalRecordV3,
    TrainRecordV3,
)

_V3_SIM_METRIC_CANDIDATES = ("cka", "cosine", "pwcca", "neg_mse")


def _backbone_dir_has_v3_artifacts(backbone_dir: Path) -> bool:
    if not backbone_dir.is_dir():
        return False
    for ds_dir in backbone_dir.iterdir():
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        if (ds_dir / "cache" / "cache_manifest_v3.json").is_file():
            return True
        if (ds_dir / "redundancy" / "redundancy_graph_v3.npz").is_file():
            return True
    return False


def discover_v3_backbones(output_root: Path) -> List[str]:
    """List backbone folder names under ``analysis_outputs_v3/``."""
    base = Path(output_root)
    if not base.is_dir():
        return []
    names: List[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if _backbone_dir_has_v3_artifacts(child):
            names.append(child.name)
    return names


def detect_v3_backbone_root(path: Path, *, output_root: Optional[Path] = None) -> Optional[Path]:
    """Return ``analysis_outputs_v3/<backbone>`` if *path* lies under a v3 backbone tree."""
    from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT

    root = Path(output_root or OUTPUT_ROOT_DEFAULT).resolve()
    p = path.resolve()
    if not root.is_dir():
        return None
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    candidate = root / rel.parts[0]
    if _backbone_dir_has_v3_artifacts(candidate):
        return candidate
    return None


# Backward-compatible alias
detect_v3_experiment_root = detect_v3_backbone_root


def _manifest_from_redundancy_artifacts(
    dataset: str,
    summary: Optional[Dict[str, Any]],
    unit_ids_from_npz: List[str],
    scene_ids_from_npz: List[str],
) -> CacheManifestV3:
    """Synthetic manifest when disk cache was skipped but redundancy was written."""
    backbone = ""
    unit_ids: List[str] = []
    scene_ids: List[str] = []
    if isinstance(summary, dict):
        backbone = str(summary.get("backbone", ""))
        unit_ids = [str(u) for u in summary.get("unit_ids", []) if str(u)]
        scene_ids = [str(s) for s in summary.get("scene_ids", []) if str(s)]
    if not unit_ids:
        unit_ids = list(unit_ids_from_npz)
    if not scene_ids:
        scene_ids = list(scene_ids_from_npz)
    scenes = [{"scene_id": sid, "layer_files": {}} for sid in scene_ids]
    return CacheManifestV3(
        backbone=backbone,
        hf_model="",
        dataset=dataset,
        layer_units=[],
        scenes=scenes,
        num_heads=0,
        extra={"source": "redundancy_only"},
    )


def load_v3_bundle(experiment_root: Path, dataset: str) -> Dict[str, Any]:
    base = experiment_root / dataset
    cache_dir = base / "cache"
    manifest_path = cache_dir / "cache_manifest_v3.json"
    manifest: Optional[CacheManifestV3] = None
    if manifest_path.is_file():
        manifest = CacheManifestV3.load(manifest_path)

    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "scenes": [],
    }

    red_npz = base / "redundancy" / "redundancy_graph_v3.npz"
    summary_path = base / "redundancy" / "redundancy_summary_v3.json"
    summary: Optional[Dict[str, Any]] = None
    if summary_path.is_file():
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                summary = raw
        except Exception:
            summary = None

    unit_ids_npz: List[str] = []
    scene_ids_npz: List[str] = []
    if red_npz.is_file():
        z = np.load(red_npz, allow_pickle=True)
        try:
            bundle["redundancy"] = dict(z)
            unit_ids_npz = [str(u) for u in list(z.get("unit_ids", [])) if str(u)]
            if "scene_ids" in z.files:
                scene_ids_npz = [str(s) for s in list(z["scene_ids"]) if str(s)]
        finally:
            z.close()

    if manifest is None and red_npz.is_file():
        manifest = _manifest_from_redundancy_artifacts(
            dataset,
            summary,
            unit_ids_npz,
            scene_ids_npz,
        )
        bundle["manifest"] = manifest

    if manifest is not None:
        bundle["scenes"] = [str(s.get("scene_id", "")) for s in manifest.scenes if s.get("scene_id")]
    backbone_name = ""
    if manifest is not None:
        backbone_name = str(manifest.backbone)
    elif summary is not None:
        backbone_name = str(summary.get("backbone", ""))
    prune_path = base / "prune" / "prune_plan_v3.json"
    if prune_path.is_file():
        bundle["prune_plan"] = UnitPrunePlan.load(prune_path)
    elif backbone_name:
        try:
            from analysis.prune_plan import default_prune_plan_root, resolve_prune_plan_path

            bundle["prune_plan"] = UnitPrunePlan.load(
                resolve_prune_plan_path(default_prune_plan_root(), backbone_name)
            )
        except FileNotFoundError:
            pass
    train_path = base / "train" / "train_record_v3.json"
    if train_path.is_file():
        bundle["train_record"] = TrainRecordV3.load(train_path)
    eval_root = base / "eval"
    eval_records: Dict[str, EvalRecordV3] = {}
    if eval_root.is_dir():
        for run in eval_root.iterdir():
            er = run / "eval_record_v3.json"
            if er.is_file():
                eval_records[run.name] = EvalRecordV3.load(er)
    bundle["eval_records"] = eval_records
    return bundle


def scene_item_from_v3_row(scene_id: str, row: Dict[str, Any], cache_dir: Path) -> Dict[str, Any]:
    """Flatten manifest / metadata fields for :class:`~analysis.view.PairAttentionController`."""
    merged = dict(row)
    meta_path = cache_dir / scene_id / "metadata_v3.json"
    if meta_path.is_file():
        on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(on_disk, dict):
            for key in ("frame_ids", "image_paths", "num_frames", "image_hw", "patch_start_idx", "scene_dir"):
                if key not in merged or not merged.get(key):
                    merged[key] = on_disk.get(key)
    return {
        "scene_id": scene_id,
        "dir": scene_id,
        "metadata_v3": merged,
        "frame_ids": list(merged.get("frame_ids") or []),
        "image_paths": list(merged.get("image_paths") or []),
        "num_frames": int(merged.get("num_frames", 0)),
        "image_hw": merged.get("image_hw"),
        "patch_start_idx": merged.get("patch_start_idx"),
    }


def _metrics_present_in_npz(z: np.lib.npyio.NpzFile) -> List[str]:
    out: List[str] = []
    for m in _V3_SIM_METRIC_CANDIDATES:
        if f"{m}_mean" in z.files or f"{m}_scenes" in z.files:
            out.append(m)
    return out


def _build_v3_similarity_report(
    *,
    scene_id: str,
    unit_ids: List[str],
    metric_packed: Dict[str, List[float]],
    metrics: List[str],
) -> Dict[str, Any]:
    if not metric_packed:
        return {}
    n = len(unit_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "num_layers": n,
        "unit_ids": [str(u) for u in unit_ids],
        "metrics": list(metrics),
        "fusion_modes": ["v3_units"],
        "fusion_by_mode": {
            "v3_units": {
                "global_patch_similarity_upper_by_metric": dict(metric_packed),
                "global_special_similarity_upper_by_metric": {},
                "global_all_similarity_upper_by_metric": dict(metric_packed),
            }
        },
    }


def discover_v3_similarity_reports(redundancy_dir: Path) -> Dict[str, Any]:
    """Build Stage2-style reports from ``redundancy_graph_v3.npz`` (mean + per-scene)."""
    npz_path = redundancy_dir / "redundancy_graph_v3.npz"
    if not npz_path.is_file():
        return {}

    z = np.load(npz_path, allow_pickle=True)
    unit_ids = [str(u) for u in list(z.get("unit_ids", []))]
    n = len(unit_ids)
    if n <= 0:
        return {}

    metrics = _metrics_present_in_npz(z)
    summary_path = redundancy_dir / "redundancy_summary_v3.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                from_summary = [str(m) for m in summary.get("metrics", []) if str(m) in metrics]
                if from_summary:
                    metrics = from_summary
        except Exception:
            pass
    if not metrics:
        return {}

    def _packed_for_metric(metric: str, mat: np.ndarray) -> Optional[List[float]]:
        t = torch.from_numpy(np.asarray(mat, dtype=np.float64))
        if t.shape != (n, n):
            return None
        return upper_triangle_pack(t)

    reports: Dict[str, Any] = {}

    scene_ids_arr = z.get("scene_ids")
    if scene_ids_arr is not None:
        scene_ids_list = [str(s) for s in list(scene_ids_arr)]
        for si, sid in enumerate(scene_ids_list):
            packed: Dict[str, List[float]] = {}
            for metric in metrics:
                stack_key = f"{metric}_scenes"
                if stack_key not in z.files:
                    continue
                stack = np.asarray(z[stack_key])
                if stack.ndim != 3 or si >= stack.shape[0]:
                    continue
                row = _packed_for_metric(metric, stack[si])
                if row is not None:
                    packed[metric] = row
            report = _build_v3_similarity_report(
                scene_id=sid,
                unit_ids=unit_ids,
                metric_packed=packed,
                metrics=metrics,
            )
            if report:
                reports[sid] = report

    mean_packed: Dict[str, List[float]] = {}
    for metric in metrics:
        key = f"{metric}_mean"
        if key not in z.files:
            continue
        row = _packed_for_metric(metric, np.asarray(z[key]))
        if row is not None:
            mean_packed[metric] = row
    mean_report = _build_v3_similarity_report(
        scene_id="mean",
        unit_ids=unit_ids,
        metric_packed=mean_packed,
        metrics=metrics,
    )
    if mean_report:
        reports["mean"] = mean_report

    return reports


def discover_v3_skip_reports(skip_eval_dir: Path) -> Dict[str, Any]:
    """Build Stage2-style skip reports from ``skip_eval_report_v3.json`` (+ optional scenes file)."""
    reports: Dict[str, Any] = {}
    mean_path = skip_eval_dir / "skip_eval_report_v3.json"
    if not mean_path.is_file():
        return reports
    try:
        mean_raw = json.loads(mean_path.read_text(encoding="utf-8"))
    except Exception:
        return reports
    def _normalize_skip_report(row: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
        out = dict(row)
        out["scene_id"] = scene_id
        fm = out.get("fusion_modes")
        if not isinstance(fm, list) or not fm:
            mode_keys = list((out.get("fusion_by_mode") or {}).keys())
            out["fusion_modes"] = mode_keys if mode_keys else ["mixed_units"]
        return out

    if isinstance(mean_raw, dict):
        reports["mean"] = _normalize_skip_report(mean_raw, "mean")

    scenes_path = skip_eval_dir / "skip_eval_scenes_v3.json"
    if not scenes_path.is_file():
        return reports
    try:
        scenes_raw = json.loads(scenes_path.read_text(encoding="utf-8"))
    except Exception:
        return reports
    if not isinstance(scenes_raw, dict):
        return reports
    scene_rows = scenes_raw.get("scenes")
    if not isinstance(scene_rows, list):
        return reports
    for row in scene_rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scene_id", "")).strip()
        if not sid:
            continue
        reports[sid] = _normalize_skip_report(row, sid)
    return reports


def unified_metric_rows(record: EvalRecordV3) -> List[str]:
    from analysis.schema.v3 import UNIFIED_METRIC_KEYS

    lines = []
    for k in UNIFIED_METRIC_KEYS:
        v = record.average.get(k)
        if v is None:
            continue
        lines.append(f"{k}: {v:.6f}")
    return lines

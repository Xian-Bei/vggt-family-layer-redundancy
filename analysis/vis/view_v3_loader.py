"""Helpers for loading v3 pipeline artifacts in the view panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from analysis.prune_plan import UnitPrunePlan
from analysis.redundancy.similarity_v3 import (
    list_redundancy_graph_regions,
    redundancy_dir_has_graphs,
    resolve_redundancy_graph_path,
)
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
        if redundancy_dir_has_graphs(ds_dir / "redundancy"):
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
    if output_root is None:
        return None
    root = Path(output_root).resolve()
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
    unit_ids_from_npz: List[str],
    scene_ids_from_npz: List[str],
) -> CacheManifestV3:
    """Synthetic manifest when disk cache was skipped but redundancy npz exists."""
    unit_ids = list(unit_ids_from_npz)
    scene_ids = list(scene_ids_from_npz)
    scenes = [{"scene_id": sid, "layer_files": {}} for sid in scene_ids]
    return CacheManifestV3(
        backbone="",
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

    redundancy_dir = base / "redundancy"

    unit_ids_npz: List[str] = []
    scene_ids_npz: List[str] = []
    for region in list_redundancy_graph_regions(redundancy_dir):
        npz_path = resolve_redundancy_graph_path(redundancy_dir, region)
        if npz_path is None:
            continue
        z = np.load(npz_path, allow_pickle=True)
        try:
            bundle.setdefault("redundancy_by_region", {})[region] = dict(z)
            if not unit_ids_npz:
                unit_ids_npz = [str(u) for u in list(z.get("unit_ids", [])) if str(u)]
            if "scene_ids" in z.files and not scene_ids_npz:
                scene_ids_npz = [str(s) for s in list(z["scene_ids"]) if str(s)]
        finally:
            z.close()

    if manifest is None and unit_ids_npz:
        manifest = _manifest_from_redundancy_artifacts(
            dataset,
            unit_ids_npz,
            scene_ids_npz,
        )
        bundle["manifest"] = manifest

    if manifest is not None:
        bundle["scenes"] = [str(s.get("scene_id", "")) for s in manifest.scenes if s.get("scene_id")]
    backbone_name = str(manifest.backbone) if manifest is not None else ""
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
    special_packed: Dict[str, List[float]],
    camera_packed: Dict[str, List[float]],
    patch_packed: Dict[str, List[float]],
    all_packed: Dict[str, List[float]],
    metrics: List[str],
) -> Dict[str, Any]:
    if not special_packed and not camera_packed and not patch_packed and not all_packed:
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
                "global_patch_similarity_upper_by_metric": dict(patch_packed),
                "global_special_similarity_upper_by_metric": dict(special_packed),
                "global_camera_similarity_upper_by_metric": dict(camera_packed),
                "global_all_similarity_upper_by_metric": dict(all_packed),
            }
        },
    }


def _packed_for_metric_matrix(mat: np.ndarray, n: int) -> Optional[List[float]]:
    t = torch.from_numpy(np.asarray(mat, dtype=np.float64))
    if t.shape != (n, n):
        return None
    return upper_triangle_pack(t)


def _load_redundancy_npz_metric_packs(
    npz_path: Path,
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    """Return unit ids, metrics, scene ids, per-scene packs, and mean packs for one region npz."""
    z = np.load(npz_path, allow_pickle=True)
    try:
        unit_ids = [str(u) for u in list(z.get("unit_ids", []))]
        n = len(unit_ids)
        if n <= 0:
            return [], [], [], {}, {}

        metrics = _metrics_present_in_npz(z)
        if not metrics:
            return unit_ids, [], [], {}, {}

        scene_ids_list: List[str] = []
        scene_ids_arr = z.get("scene_ids")
        if scene_ids_arr is not None:
            scene_ids_list = [str(s) for s in list(scene_ids_arr)]

        per_scene: Dict[str, Dict[str, List[float]]] = {}
        if scene_ids_list:
            for si, sid in enumerate(scene_ids_list):
                packed: Dict[str, List[float]] = {}
                for metric in metrics:
                    stack_key = f"{metric}_scenes"
                    if stack_key not in z.files:
                        continue
                    stack = np.asarray(z[stack_key])
                    if stack.ndim != 3 or si >= stack.shape[0]:
                        continue
                    row = _packed_for_metric_matrix(stack[si], n)
                    if row is not None:
                        packed[metric] = row
                if packed:
                    per_scene[sid] = packed

        mean_packed: Dict[str, List[float]] = {}
        for metric in metrics:
            key = f"{metric}_mean"
            if key not in z.files:
                continue
            row = _packed_for_metric_matrix(np.asarray(z[key]), n)
            if row is not None:
                mean_packed[metric] = row

        return unit_ids, metrics, scene_ids_list, per_scene, mean_packed
    finally:
        z.close()


def discover_v3_similarity_reports(redundancy_dir: Path) -> Dict[str, Any]:
    """Build Stage2-style reports from per-region ``redundancy_graph_<region>.npz`` files."""
    red_dir = Path(redundancy_dir)
    regions = list_redundancy_graph_regions(red_dir)
    if not regions:
        return {}

    region_scene_packs: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    region_mean_packs: Dict[str, Dict[str, List[float]]] = {}
    unit_ids: List[str] = []
    metrics: List[str] = []
    scene_ids: List[str] = []

    for region in regions:
        npz_path = resolve_redundancy_graph_path(red_dir, region)
        if npz_path is None:
            continue
        uids, region_metrics, region_scene_ids, per_scene, mean_packed = _load_redundancy_npz_metric_packs(
            npz_path
        )
        if not uids:
            continue
        if not unit_ids:
            unit_ids = uids
        elif uids != unit_ids:
            raise ValueError(
                f"Region {region!r} unit_ids {uids} != prior {unit_ids} in {red_dir}"
            )
        if region_metrics and not metrics:
            metrics = region_metrics
        if region_scene_ids and not scene_ids:
            scene_ids = region_scene_ids
        region_scene_packs[region] = per_scene
        region_mean_packs[region] = mean_packed

    if not metrics or not unit_ids:
        return {}

    reports: Dict[str, Any] = {}
    all_scene_keys = {s for s in scene_ids if s != "mean"}
    for packs in region_scene_packs.values():
        all_scene_keys.update(s for s in packs.keys() if s != "mean")

    ordered_scene_ids = [s for s in scene_ids if s in all_scene_keys]
    for sid in sorted(all_scene_keys):
        if sid not in ordered_scene_ids:
            ordered_scene_ids.append(sid)

    for sid in ordered_scene_ids:
        report = _build_v3_similarity_report(
            scene_id=sid,
            unit_ids=unit_ids,
            special_packed=region_scene_packs.get("special", {}).get(sid, {}),
            camera_packed=region_scene_packs.get("camera", {}).get(sid, {}),
            patch_packed=region_scene_packs.get("patch", {}).get(sid, {}),
            all_packed=region_scene_packs.get("all", {}).get(sid, {}),
            metrics=metrics,
        )
        if report:
            reports[sid] = report

    mean_report = _build_v3_similarity_report(
        scene_id="mean",
        unit_ids=unit_ids,
        special_packed=region_mean_packs.get("special", {}),
        camera_packed=region_mean_packs.get("camera", {}),
        patch_packed=region_mean_packs.get("patch", {}),
        all_packed=region_mean_packs.get("all", {}),
        metrics=metrics,
    )
    if mean_report:
        reports["mean"] = mean_report

    return reports


def discover_v3_skip_reports(skip_eval_dir: Path) -> Dict[str, Any]:
    """Build Stage2-style skip reports from finalized files or live JSONL checkpoints."""
    reports: Dict[str, Any] = {}
    mean_path = skip_eval_dir / "skip_eval_report_v3.json"

    def _normalize_skip_report(row: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
        out = dict(row)
        out["scene_id"] = scene_id
        fm = out.get("fusion_modes")
        if not isinstance(fm, list) or not fm:
            mode_keys = list((out.get("fusion_by_mode") or {}).keys())
            out["fusion_modes"] = mode_keys if mode_keys else ["mixed_units"]
        return out

    if mean_path.is_file():
        try:
            mean_raw = json.loads(mean_path.read_text(encoding="utf-8"))
        except Exception:
            mean_raw = None
        if isinstance(mean_raw, dict):
            reports["mean"] = _normalize_skip_report(mean_raw, "mean")

    scenes_path = skip_eval_dir / "skip_eval_scenes_v3.json"
    if scenes_path.is_file():
        try:
            scenes_raw = json.loads(scenes_path.read_text(encoding="utf-8"))
        except Exception:
            scenes_raw = None
        scene_rows = scenes_raw.get("scenes") if isinstance(scenes_raw, dict) else None
        if isinstance(scene_rows, list):
            for row in scene_rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("scene_id", "")).strip()
                if sid:
                    reports[sid] = _normalize_skip_report(row, sid)

    # Incremental skip evaluation writes one JSONL file per scene long before the
    # final aggregate files appear. Expose those completed/partial scenes in the UI.
    manifest_path = skip_eval_dir / "skip_eval_manifest_v3.json"
    checkpoint_dir = skip_eval_dir / "checkpoints"
    if not manifest_path.is_file() or not checkpoint_dir.is_dir():
        return reports
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return reports
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        return reports
    unit_ids = [str(x) for x in config.get("unit_ids", [])]
    metrics = [str(x) for x in config.get("metrics", [])]
    fusion_mode = str(config.get("mode", "mixed_units")).strip() or "mixed_units"
    n = len(unit_ids)
    if n <= 0:
        return reports

    upper_by_scene: Dict[str, Dict[str, float]] = {}
    upper_path = skip_eval_dir.parent / "eval" / "upper" / "eval_record_v3.json"
    if upper_path.is_file():
        try:
            upper_raw = json.loads(upper_path.read_text(encoding="utf-8"))
        except Exception:
            upper_raw = None
        scene_values = upper_raw.get("scenes") if isinstance(upper_raw, dict) else None
        rows = list(scene_values.values()) if isinstance(scene_values, dict) else scene_values
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("scene_id", "")).strip()
                if not sid:
                    continue
                clean: Dict[str, float] = {}
                for key, value in row.items():
                    metric_key = "chamfer_distance" if str(key) == "cd" else str(key)
                    try:
                        clean[metric_key] = float(value)
                    except (TypeError, ValueError):
                        continue
                upper_by_scene[sid] = clean

    higher_is_better = {"depth_delta1", "nc"}
    # ``upper_triangle_unpack_for_heatmap`` uses the inclusive upper triangle,
    # so checkpoint spans must include (i, i) as well as i < j.
    packed_len = n * (n + 1) // 2
    for checkpoint_path in sorted(checkpoint_dir.glob("*.jsonl")):
        span_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
        scene_id = ""
        try:
            lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            sid = str(row.get("scene_id", "")).strip()
            values = row.get("metrics")
            try:
                span = (int(row["span_i"]), int(row["span_j"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not sid or not isinstance(values, dict):
                continue
            scene_id = sid
            clean_values: Dict[str, float] = {}
            for key, value in values.items():
                metric_key = "chamfer_distance" if str(key) == "cd" else str(key)
                try:
                    clean_values[metric_key] = float(value)
                except (TypeError, ValueError):
                    continue
            span_metrics[span] = clean_values
        upper = upper_by_scene.get(scene_id)
        if not scene_id or not upper:
            continue
        delta_by_metric: Dict[str, List[float]] = {}
        pct_by_metric: Dict[str, List[float]] = {}
        for metric in metrics:
            if metric not in upper:
                continue
            baseline = float(upper[metric])
            denom = max(abs(baseline), 1e-8)
            delta: List[float] = []
            pct: List[float] = []
            for i in range(n):
                for j in range(i, n):
                    current = span_metrics.get((i, j), {}).get(metric)
                    if current is None:
                        delta.append(float("nan"))
                        pct.append(float("nan"))
                        continue
                    d = float(current) - baseline
                    delta.append(d)
                    pct.append(100.0 * (-d if metric in higher_is_better else d) / denom)
            if len(delta) == packed_len:
                delta_by_metric[metric] = delta
                pct_by_metric[metric] = pct
        if delta_by_metric:
            reports[scene_id] = {
                "schema_version": SCHEMA_VERSION,
                "kind": "skip_gt_eval_v3_checkpoint",
                "scene_id": scene_id,
                "num_layers": n,
                "unit_ids": unit_ids,
                "metrics": metrics,
                "fusion_modes": [fusion_mode],
                "upper_average": upper,
                "fusion_by_mode": {
                    fusion_mode: {
                        "gt_delta_upper_by_metric": delta_by_metric,
                        "gt_degrade_pct_upper_by_metric": pct_by_metric,
                    }
                },
                "meta": {
                    "source": str(checkpoint_path),
                    "completed_spans": len(span_metrics),
                    "expected_spans": int(config.get("spans") and len(config["spans"]) or 0),
                },
            }
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

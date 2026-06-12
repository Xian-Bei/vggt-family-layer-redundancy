"""Skip-range GT eval (v3): measure metric degrade vs upper when dropping unit spans."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.registry import get_dataset
from analysis.eval.eval_v3 import evaluate_model_v3
from analysis.redundancy.metrics import upper_triangle_pack
from analysis.schema.v3 import SCHEMA_VERSION, dataset_output_tag, stage_dir
from analysis.schema.v3 import PruneAction

SKIP_EVAL_METRICS: Tuple[str, ...] = (
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "depth_abs_rel",
    "depth_delta1",
    "chamfer_distance",
)

# Maps eval_v3 / metrics_core keys -> heatmap keys in analysis.view.
_EVAL_TO_VIEW_KEY: Dict[str, str] = {
    "ape": "ate",
    "cd": "chamfer_distance",
}

_LOWER_IS_WORSE = frozenset(
    {
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "depth_abs_rel",
        "comp",
        "chamfer_distance",
        "ape",
        "cd",
    }
)
_HIGHER_IS_WORSE = frozenset({"depth_delta1", "acc", "nc"})


def _timing_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    v = os.environ.get("SKIP_GT_EVAL_TIMING", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _timing_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skip-range GT eval vs upper (v3).")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True, choices=["scannet", "7scenes", "nrgbd"])
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="7scenes split (required when --dataset=7scenes).",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument(
        "--output-dataset",
        type=str,
        default="",
        help="Optional output dataset tag override (e.g. scannet_dev50/scannet_test50).",
    )
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument(
        "--mode",
        type=str,
        default="mixed_units",
        choices=["mixed_units"],
        help="Skip fusion mode. Currently only mixed_units is supported.",
    )
    p.add_argument(
        "--max-span",
        type=int,
        default=0,
        help="Max unit span (j-i) inclusive; 0 means full upper triangle.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--timing",
        action="store_true",
        help="Log per-step timings (also set SKIP_GT_EVAL_TIMING=1).",
    )
    return p


def unit_ids_from_catalog(adapter, model: torch.nn.Module) -> List[str]:
    return [u.unit_id for u in adapter.layer_unit_catalog(model)]


def iter_unit_spans(n: int, *, max_span: int) -> Iterable[Tuple[int, int]]:
    if n <= 0:
        return
    cap = int(max_span)
    for i in range(n):
        j_max = n - 1 if cap <= 0 else min(n - 1, i + cap)
        for j in range(i, j_max + 1):
            yield i, j


def _span_drop_label(unit_ids: Sequence[str], i: int, j: int) -> str:
    if int(i) == int(j):
        return str(unit_ids[i])
    return f"{unit_ids[i]}..{unit_ids[j]}"


def drop_actions_for_span(unit_ids: Sequence[str], i: int, j: int) -> List[PruneAction]:
    actions: List[PruneAction] = []
    for k in range(int(i), int(j) + 1):
        actions.append(PruneAction(action="drop_unit", unit_id=str(unit_ids[k])))
    return actions


def normalize_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        fv = float(v)
        if not np.isfinite(fv):
            continue
        view_k = _EVAL_TO_VIEW_KEY.get(str(k), str(k))
        if view_k in SKIP_EVAL_METRICS:
            out[view_k] = fv
    return out


def metric_worse_direction(metric: str) -> Literal["lower", "higher"]:
    m = str(metric)
    if m in _HIGHER_IS_WORSE:
        return "higher"
    return "lower"


def degrade_pct_vs_upper(upper: float, current: float, metric: str, *, eps: float = 1e-8) -> float:
    """Positive percent means performance got worse vs upper."""
    u = float(upper)
    c = float(current)
    denom = max(abs(u), eps)
    if metric_worse_direction(metric) == "lower":
        return 100.0 * (c - u) / denom
    return 100.0 * (u - c) / denom


def delta_vs_upper(upper: float, current: float) -> float:
    return float(current) - float(upper)


def _upper_eval_path(experiment: str, tagged_dataset: str, backbone: str) -> Path:
    return stage_dir(experiment, tagged_dataset, "eval") / f"upper_{backbone.strip().lower()}" / "eval_record_v3.json"


def load_upper_eval_or_fail(
    *,
    experiment: str,
    tagged_dataset: str,
    backbone: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    upper_path = _upper_eval_path(experiment, tagged_dataset, backbone)
    if not upper_path.is_file():
        raise FileNotFoundError(
            "Missing upper eval record: "
            f"{upper_path}. Please run run_eval.sh in upper mode before run_skip_eval.sh."
        )
    payload = json.loads(upper_path.read_text(encoding="utf-8"))
    avg = dict(payload.get("average", {}))
    scenes_payload = payload.get("scenes", {})
    scene_metrics: Dict[str, Dict[str, float]] = {}
    if isinstance(scenes_payload, dict):
        for row in scenes_payload.values():
            if not isinstance(row, dict):
                continue
            sid = str(row.get("scene_id", "")).strip()
            if not sid:
                continue
            scene_metrics[sid] = normalize_metrics(row)
    elif isinstance(scenes_payload, list):
        for row in scenes_payload:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("scene_id", "")).strip()
            if not sid:
                continue
            scene_metrics[sid] = normalize_metrics(row)
    return normalize_metrics(avg), scene_metrics


def evaluate_span(
    *,
    adapter,
    hf_model: str,
    device: torch.device,
    ds_plugin,
    experiment: str,
    dataset: str,
    tagged_dataset: str,
    data_root: Path,
    backbone: str,
    unit_ids: Sequence[str],
    span_i: int,
    span_j: int,
    max_scenes: int,
    max_frames: int,
    split: str | None,
    progress: bool,
    timing: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    span_t0 = time.perf_counter()
    t_load = time.perf_counter()
    model = adapter.load_pretrained(hf_model=hf_model, device=str(device), eval_mode=True)
    load_s = time.perf_counter() - t_load
    try:
        t_plan = time.perf_counter()
        actions = drop_actions_for_span(unit_ids, span_i, span_j)
        adapter.apply_prune_plan(model, actions)
        model.eval()
        plan_s = time.perf_counter() - t_plan

        out_dir = (
            stage_dir(experiment, tagged_dataset, "skip_eval")
            / "runs"
            / f"span_{span_i}_{span_j}"
        )
        t_eval = time.perf_counter()
        evaluate_model_v3(
            adapter=adapter,
            model=model,
            ds_plugin=ds_plugin,
            experiment=experiment,
            dataset=dataset,
            data_root=data_root,
            backbone=backbone,
            model_name=f"skip_{span_i}_{span_j}",
            device=device,
            output_dir=out_dir,
            skip_depth=adapter.periodic_eval_skip_depth(),
            max_scenes=int(max_scenes),
            max_frames=int(max_frames),
            split=split,
            output_dataset=tagged_dataset,
            progress=progress,
        )
        eval_s = time.perf_counter() - t_eval
        t_read = time.perf_counter()
        rec_path = out_dir / "eval_record_v3.json"
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
        read_s = time.perf_counter() - t_read
        if timing:
            _timing_log(
                f"[timing] span=({span_i},{span_j}) fallback "
                f"load={load_s:.3f}s plan={plan_s:.3f}s evaluate_model_v3={eval_s:.3f}s "
                f"read_record={read_s:.3f}s total={time.perf_counter() - span_t0:.3f}s"
            )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    avg = normalize_metrics(dict(payload.get("average", {})))
    scenes_payload = payload.get("scenes", {})
    by_scene: Dict[str, Dict[str, float]] = {}
    if isinstance(scenes_payload, dict):
        rows = list(scenes_payload.values())
    elif isinstance(scenes_payload, list):
        rows = list(scenes_payload)
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scene_id", "")).strip()
        if not sid:
            continue
        by_scene[sid] = normalize_metrics(row)
    return avg, by_scene


def build_packed_matrices(
    *,
    n_units: int,
    spans: Sequence[Tuple[int, int]],
    upper_metrics: Dict[str, float],
    span_metrics: Dict[Tuple[int, int], Dict[str, float]],
    metrics: Sequence[str],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    delta_packed: Dict[str, List[float]] = {}
    pct_packed: Dict[str, List[float]] = {}
    for metric in metrics:
        if metric not in upper_metrics:
            continue
        mat = np.zeros((n_units, n_units), dtype=np.float64)
        u = float(upper_metrics[metric])
        for i, j in spans:
            cur = span_metrics.get((i, j), {}).get(metric)
            if cur is None:
                mat[i, j] = float("nan")
            else:
                mat[i, j] = degrade_pct_vs_upper(u, float(cur), metric)
        pct_packed[metric] = upper_triangle_pack(torch.from_numpy(mat))

        mat_d = np.zeros((n_units, n_units), dtype=np.float64)
        for i, j in spans:
            cur = span_metrics.get((i, j), {}).get(metric)
            if cur is None:
                mat_d[i, j] = float("nan")
            else:
                mat_d[i, j] = delta_vs_upper(u, float(cur))
        delta_packed[metric] = upper_triangle_pack(torch.from_numpy(mat_d))
    return delta_packed, pct_packed


def build_skip_report(
    *,
    scene_id: str,
    unit_ids: Sequence[str],
    metrics: Sequence[str],
    delta_packed: Dict[str, List[float]],
    pct_packed: Dict[str, List[float]],
    upper_metrics: Dict[str, float],
    backbone: str,
    dataset: str,
    max_span: int,
    fusion_mode: str,
    scene_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    n = len(unit_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3",
        "backbone": backbone,
        "dataset": dataset,
        "scene_id": scene_id,
        "num_layers": n,
        "unit_ids": list(unit_ids),
        "metrics": list(metrics),
        "fusion_modes": [str(fusion_mode)],
        "max_span": int(max_span),
        "scene_ids": [str(s) for s in (scene_ids or []) if str(s).strip()],
        "num_scenes": int(len(scene_ids or [])),
        "upper_average": dict(upper_metrics),
        "fusion_by_mode": {
            str(fusion_mode): {
                "gt_delta_upper_by_metric": dict(delta_packed),
                "gt_degrade_pct_upper_by_metric": dict(pct_packed),
            }
        },
        "meta": {
            "metric_direction": {m: metric_worse_direction(m) for m in metrics},
            "span_semantics": "closed_interval_unit_indices",
            "pct_semantics": "positive_means_worse_than_upper",
        },
    }


def _span_ij_key(i: int, j: int) -> str:
    return f"{i},{j}"


def _parse_span_ij_key(key: str) -> Tuple[int, int]:
    parts = str(key).split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid span key: {key!r}")
    return int(parts[0]), int(parts[1])


def _serialize_span_raw(span_raw: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {_span_ij_key(i, j): dict(v) for (i, j), v in span_raw.items()}


def _deserialize_span_raw(obj: Any) -> Dict[Tuple[int, int], Dict[str, float]]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, val in obj.items():
        if not isinstance(val, dict):
            continue
        out[_parse_span_ij_key(str(key))] = dict(val)
    return out


def _safe_scene_filename(scene_id: str) -> str:
    return str(scene_id).replace("/", "__").replace("\\", "__")


def _scene_staging_dir(out_root: Path) -> Path:
    return out_root / "scene_staging"


def _scene_staging_path(out_root: Path, scene_id: str) -> Path:
    return _scene_staging_dir(out_root) / f"{_safe_scene_filename(scene_id)}.json"


def _scenes_json_path(out_root: Path) -> Path:
    return out_root / "skip_eval_scenes_v3.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _init_scenes_payload(*, backbone: str, dataset: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3_scenes",
        "backbone": backbone,
        "dataset": dataset,
        "num_scenes": 0,
        "scenes": [],
    }


def _load_scenes_payload(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return _init_scenes_payload(backbone="", dataset="")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid scenes payload: {path}")
    return payload


def _upsert_scene_report(
    out_root: Path,
    scene_report: Dict[str, Any],
    *,
    backbone: str,
    dataset: str,
) -> None:
    path = _scenes_json_path(out_root)
    payload = _load_scenes_payload(path) if path.is_file() else _init_scenes_payload(backbone=backbone, dataset=dataset)
    payload["schema_version"] = SCHEMA_VERSION
    payload["kind"] = "skip_gt_eval_v3_scenes"
    payload["backbone"] = backbone
    payload["dataset"] = dataset
    sid = str(scene_report.get("scene_id", "")).strip()
    scenes = [row for row in payload.get("scenes", []) if str(row.get("scene_id", "")).strip() != sid]
    scenes.append(scene_report)
    scenes.sort(key=lambda row: str(row.get("scene_id", "")))
    payload["scenes"] = scenes
    payload["num_scenes"] = len(scenes)
    _atomic_write_json(path, payload)


def _span_raw_is_complete(span_raw: Dict[Tuple[int, int], Dict[str, float]], spans: Sequence[Tuple[int, int]]) -> bool:
    return all(span in span_raw for span in spans)


def _staging_row_is_complete(row: Dict[str, Any], spans: Sequence[Tuple[int, int]]) -> bool:
    sid = str(row.get("scene_id", "")).strip()
    if not sid:
        return False
    span_raw = _deserialize_span_raw(row.get("span_raw_metrics", {}))
    return _span_raw_is_complete(span_raw, spans)


def _save_scene_staging(
    out_root: Path,
    *,
    scene_id: str,
    upper_average: Dict[str, float],
    span_raw: Dict[Tuple[int, int], Dict[str, float]],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3_scene_staging",
        "scene_id": str(scene_id),
        "upper_average": dict(upper_average),
        "span_raw_metrics": _serialize_span_raw(span_raw),
    }
    _atomic_write_json(_scene_staging_path(out_root, scene_id), payload)


def _load_scene_staging(out_root: Path, scene_id: str) -> Optional[Dict[str, Any]]:
    path = _scene_staging_path(out_root, scene_id)
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    return row if isinstance(row, dict) else None


def _merge_span_into_scene_staging(
    out_root: Path,
    *,
    scene_id: str,
    upper_average: Dict[str, float],
    span: Tuple[int, int],
    metric_row: Dict[str, float],
) -> None:
    path = _scene_staging_path(out_root, scene_id)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "skip_gt_eval_v3_scene_staging",
            "scene_id": str(scene_id),
            "upper_average": dict(upper_average),
            "span_raw_metrics": {},
        }
    payload["upper_average"] = dict(upper_average)
    raw_obj = payload.setdefault("span_raw_metrics", {})
    if not isinstance(raw_obj, dict):
        raw_obj = {}
        payload["span_raw_metrics"] = raw_obj
    raw_obj[_span_ij_key(*span)] = dict(metric_row)
    _atomic_write_json(path, payload)


def _load_all_scene_staging(out_root: Path) -> List[Dict[str, Any]]:
    staging_dir = _scene_staging_dir(out_root)
    if not staging_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(staging_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _build_scene_report_from_staging_row(
    staging_row: Dict[str, Any],
    *,
    spans: Sequence[Tuple[int, int]],
    n_units: int,
    unit_ids: Sequence[str],
    metrics: Sequence[str],
    backbone: str,
    dataset: str,
    max_span: int,
    fusion_mode: str,
) -> Optional[Dict[str, Any]]:
    sid = str(staging_row.get("scene_id", "")).strip()
    if not sid:
        return None
    span_raw = _deserialize_span_raw(staging_row.get("span_raw_metrics", {}))
    if not _span_raw_is_complete(span_raw, spans):
        return None
    scene_upper = normalize_metrics(dict(staging_row.get("upper_average", {})))
    scene_metrics = [m for m in metrics if m in scene_upper]
    if not scene_metrics:
        return None
    scene_span_metrics = {span: span_raw[span] for span in spans}
    s_delta_packed, s_pct_packed = build_packed_matrices(
        n_units=n_units,
        spans=spans,
        upper_metrics=scene_upper,
        span_metrics=scene_span_metrics,
        metrics=scene_metrics,
    )
    return build_skip_report(
        scene_id=sid,
        unit_ids=unit_ids,
        metrics=scene_metrics,
        delta_packed=s_delta_packed,
        pct_packed=s_pct_packed,
        upper_metrics=scene_upper,
        backbone=backbone,
        dataset=dataset,
        max_span=int(max_span),
        fusion_mode=str(fusion_mode),
    )


def _persist_scene_result(
    out_root: Path,
    *,
    scene_id: str,
    upper_average: Dict[str, float],
    span_raw: Dict[Tuple[int, int], Dict[str, float]],
    spans: Sequence[Tuple[int, int]],
    n_units: int,
    unit_ids: Sequence[str],
    metrics: Sequence[str],
    backbone: str,
    dataset: str,
    max_span: int,
    fusion_mode: str,
) -> None:
    _save_scene_staging(
        out_root,
        scene_id=scene_id,
        upper_average=upper_average,
        span_raw=span_raw,
    )
    staging_row = _load_scene_staging(out_root, scene_id)
    if staging_row is None:
        return
    scene_report = _build_scene_report_from_staging_row(
        staging_row,
        spans=spans,
        n_units=n_units,
        unit_ids=unit_ids,
        metrics=metrics,
        backbone=backbone,
        dataset=dataset,
        max_span=max_span,
        fusion_mode=fusion_mode,
    )
    if scene_report is None:
        return
    _upsert_scene_report(out_root, scene_report, backbone=backbone, dataset=dataset)
    print(f"Wrote scene {scene_id} -> {_scenes_json_path(out_root)}", file=sys.stderr)


def _average_span_metrics_from_staging(
    staging_rows: Sequence[Dict[str, Any]],
    spans: Sequence[Tuple[int, int]],
    metrics: Sequence[str],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    span_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    for span in spans:
        avg_row: Dict[str, float] = {}
        for m in metrics:
            vals: List[float] = []
            for row in staging_rows:
                span_raw = _deserialize_span_raw(row.get("span_raw_metrics", {}))
                sm = span_raw.get(span, {})
                if m in sm and np.isfinite(float(sm[m])):
                    vals.append(float(sm[m]))
            if vals:
                avg_row[m] = float(np.mean(np.asarray(vals, dtype=np.float64)))
        if avg_row:
            span_metrics[span] = avg_row
    return span_metrics


def _finalize_skip_outputs_from_disk(
    *,
    out_root: Path,
    spans: Sequence[Tuple[int, int]],
    n_units: int,
    unit_ids: Sequence[str],
    metrics: Sequence[str],
    upper_metrics: Dict[str, float],
    backbone: str,
    dataset: str,
    max_span: int,
    fusion_mode: str,
    elapsed_seconds: float,
    ensure_scene_reports: bool = False,
) -> Path:
    staging_rows = _load_all_scene_staging(out_root)
    complete_rows = [row for row in staging_rows if _staging_row_is_complete(row, spans)]
    if ensure_scene_reports:
        for row in complete_rows:
            scene_report = _build_scene_report_from_staging_row(
                row,
                spans=spans,
                n_units=n_units,
                unit_ids=unit_ids,
                metrics=metrics,
                backbone=backbone,
                dataset=dataset,
                max_span=max_span,
                fusion_mode=fusion_mode,
            )
            if scene_report is not None:
                _upsert_scene_report(out_root, scene_report, backbone=backbone, dataset=dataset)

    scene_ids_for_mean = sorted(str(row.get("scene_id", "")).strip() for row in complete_rows if str(row.get("scene_id", "")).strip())
    span_metrics = _average_span_metrics_from_staging(complete_rows, spans, metrics)

    delta_packed, pct_packed = build_packed_matrices(
        n_units=n_units,
        spans=spans,
        upper_metrics=upper_metrics,
        span_metrics=span_metrics,
        metrics=metrics,
    )
    mean_report = build_skip_report(
        scene_id="mean",
        unit_ids=unit_ids,
        metrics=metrics,
        delta_packed=delta_packed,
        pct_packed=pct_packed,
        upper_metrics=upper_metrics,
        backbone=backbone,
        dataset=dataset,
        max_span=int(max_span),
        fusion_mode=str(fusion_mode),
        scene_ids=scene_ids_for_mean,
    )
    report_path = out_root / "skip_eval_report_v3.json"
    _atomic_write_json(report_path, mean_report)

    npz_path = out_root / "skip_eval_v3.npz"
    npz_payload: Dict[str, Any] = {
        "unit_ids": np.array(unit_ids, dtype=object),
        "metrics": np.array(metrics, dtype=object),
        "schema_version": np.array([SCHEMA_VERSION]),
    }
    for m in metrics:
        if m in pct_packed:
            npz_payload[f"{m}_degrade_pct_mean"] = np.array(
                upper_triangle_unpack_list(pct_packed[m], n_units), dtype=np.float64
            )
    np.savez(npz_path, **npz_payload)

    scenes_path = _scenes_json_path(out_root)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3_summary",
        "backbone": backbone,
        "dataset": dataset,
        "num_units": n_units,
        "num_spans": len(spans),
        "max_span": int(max_span),
        "metrics": list(metrics),
        "upper_average": dict(upper_metrics),
        "num_scenes_completed": len(complete_rows),
        "elapsed_seconds": elapsed_seconds,
        "report_path": str(report_path),
        "scenes_path": str(scenes_path),
        "staging_dir": str(_scene_staging_dir(out_root)),
    }
    summary_path = out_root / "skip_eval_summary_v3.json"
    _atomic_write_json(summary_path, summary)
    print(
        f"Wrote {report_path} ({len(spans)} spans, {n_units} units, {len(complete_rows)} scenes)",
        file=sys.stderr,
    )
    return report_path


def run_skip_gt_eval(
    *,
    experiment: str,
    dataset: str,
    data_root: Path,
    backbone: str,
    hf_model: str = "",
    output_dataset: str = "",
    device: str = "cuda",
    max_scenes: int = 0,
    max_frames: int = 50,
    max_span: int = 0,
    mode: str = "mixed_units",
    output_dir: Optional[Path] = None,
    split: str | None = None,
    progress: bool = True,
    timing: Optional[bool] = None,
) -> Path:
    timing_on = _timing_enabled(timing)
    if timing_on:
        _timing_log(f"[timing] skip_gt_eval timing enabled dataset={dataset} backbone={backbone}")
    if dataset == "7scenes" and not split:
        raise ValueError("7scenes requires split='train' or 'test'")
    tagged_dataset = str(output_dataset).strip() or dataset_output_tag(dataset, split)
    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    ds_plugin = get_dataset(dataset)
    torch_device = torch.device(device)
    hf = str(hf_model or adapter.default_hf_model())

    out_root = output_dir or stage_dir(experiment, tagged_dataset, "skip_eval")
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model_upper = adapter.load_pretrained(hf_model=hf, device=str(torch_device), eval_mode=True)
    unit_ids = unit_ids_from_catalog(adapter, model_upper)
    n_units = len(unit_ids)
    if n_units < 1:
        raise RuntimeError(f"No units in catalog for backbone={backbone}")

    spans = list(iter_unit_spans(n_units, max_span=int(max_span)))
    metrics_available = list(SKIP_EVAL_METRICS)

    upper_metrics, upper_scene_metrics = load_upper_eval_or_fail(
        experiment=experiment,
        tagged_dataset=tagged_dataset,
        backbone=backbone,
    )
    del model_upper
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = [m for m in metrics_available if m in upper_metrics]
    if not metrics:
        raise RuntimeError(f"No overlapping metrics between upper eval and skip_eval list: {upper_metrics}")

    total_spans = len(spans)
    desc = f"skip_eval {adapter.name} ({tagged_dataset})"
    span_iter: Iterable[Tuple[int, Tuple[int, int]]] = (
        tqdm(
            enumerate(spans),
            total=total_spans,
            desc=desc,
            unit="span",
            file=sys.stderr,
            dynamic_ncols=True,
        )
        if progress
        else enumerate(spans)
    )
    for si, (i, j) in span_iter:
        drop_label = _span_drop_label(unit_ids, i, j)
        if progress and hasattr(span_iter, "set_postfix_str"):
            span_iter.set_postfix_str(drop_label, refresh=False)
        try:
            _, by_scene_metrics = evaluate_span(
                adapter=adapter,
                hf_model=hf,
                device=torch_device,
                ds_plugin=ds_plugin,
                experiment=experiment,
                dataset=dataset,
                tagged_dataset=tagged_dataset,
                data_root=data_root,
                backbone=backbone,
                unit_ids=unit_ids,
                span_i=i,
                span_j=j,
                max_scenes=int(max_scenes),
                max_frames=int(max_frames),
                split=split,
                progress=False,
                timing=timing_on,
            )
        except RuntimeError as e:
            msg = f"[warn] span {i},{j} ({drop_label}) failed: {e}"
            if progress:
                tqdm.write(msg, file=sys.stderr)
            else:
                print(msg, file=sys.stderr, flush=True)
            by_scene_metrics = {}
        for sid, metric_row in by_scene_metrics.items():
            scene_upper = normalize_metrics(dict(upper_scene_metrics.get(str(sid), {})))
            _merge_span_into_scene_staging(
                out_root,
                scene_id=str(sid),
                upper_average=scene_upper,
                span=(int(i), int(j)),
                metric_row=dict(metric_row),
            )
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    return _finalize_skip_outputs_from_disk(
        out_root=out_root,
        spans=spans,
        n_units=n_units,
        unit_ids=unit_ids,
        metrics=metrics,
        upper_metrics=upper_metrics,
        backbone=adapter.name,
        dataset=tagged_dataset,
        max_span=int(max_span),
        fusion_mode=str(mode),
        elapsed_seconds=time.perf_counter() - t0,
        ensure_scene_reports=True,
    )


def upper_triangle_unpack_list(packed: List[float], n: int) -> List[List[float]]:
    from analysis.redundancy.metrics import upper_triangle_unpack_for_heatmap

    return upper_triangle_unpack_for_heatmap(packed, n, float("nan"))


def main() -> None:
    args = _build_parser().parse_args()
    run_skip_gt_eval(
        experiment=str(args.experiment),
        dataset=str(args.dataset),
        data_root=Path(args.data_root),
        backbone=str(args.backbone),
        hf_model=str(args.hf_model),
        output_dataset=str(args.output_dataset),
        device=str(args.device),
        max_scenes=int(args.max_scenes),
        max_frames=int(args.max_frames),
        max_span=int(args.max_span),
        mode=str(args.mode),
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        split=args.split,
        progress=not bool(args.no_progress),
        timing=bool(args.timing) if args.timing else None,
    )


if __name__ == "__main__":
    main()

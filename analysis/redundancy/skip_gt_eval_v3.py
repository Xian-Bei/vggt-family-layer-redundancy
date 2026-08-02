"""Skip-range GT eval (v3): measure metric degrade vs upper when dropping unit spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import EvalSceneContext
from analysis.backbones.registry import get_backbone
from analysis.datasets.registry import dataset_names, get_dataset
from analysis.redundancy.metrics import upper_triangle_pack
from analysis.schema.v3 import SCHEMA_VERSION, stage_dir

SKIP_EVAL_METRICS: Tuple[str, ...] = (
    "ape",
    "are",
    "rpe_rot",
    "rpe_trans",
    "depth_abs_rel",
    "depth_delta1",
    "acc",
    "comp",
    "chamfer_distance",
    "nc",
)

# Normalize legacy/evaluator metric keys to the public skip-eval names.
_EVAL_TO_VIEW_KEY: Dict[str, str] = {
    "ate": "ape",
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
_HIGHER_IS_WORSE = frozenset({"depth_delta1", "nc"})


def _timing_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    v = os.environ.get("SKIP_GT_EVAL_TIMING", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _timing_log(msg: str) -> None:
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
        print(msg, file=sys.stderr, flush=True)


def _latency_log(msg: str) -> None:
    """Always-on milestone log for long-running skip_gt_eval phases."""
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
        print(msg, file=sys.stderr, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skip-range GT eval vs upper (v3).")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True, choices=dataset_names())
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split, validated by the dataset adaptor.",
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
        help="Max closed-interval drop count; 0 means full upper triangle.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--timing",
        action="store_true",
        help="Log per-step timings (also set SKIP_GT_EVAL_TIMING=1).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard skip-eval JSONL checkpoints and recompute all scene/span results.",
    )
    return p


def unit_ids_from_catalog(adapter, model: torch.nn.Module) -> List[str]:
    return [u.unit_id for u in adapter.layer_unit_catalog(model)]


def iter_unit_spans(n: int, *, max_span: int) -> Iterable[Tuple[int, int]]:
    if n <= 0:
        return
    cap = int(max_span)
    for i in range(n):
        j_max = n - 1 if cap <= 0 else min(n - 1, i + cap - 1)
        for j in range(i, j_max + 1):
            yield i, j


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


def _upper_eval_path(experiment: str, tagged_dataset: str) -> Path:
    return stage_dir(experiment, tagged_dataset, "eval") / "upper" / "eval_record_v3.json"


def load_upper_eval_or_fail(
    *,
    experiment: str,
    tagged_dataset: str,
    require_depth: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    upper_path = _upper_eval_path(experiment, tagged_dataset)
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
    avg_norm = normalize_metrics(avg)
    if require_depth and not {"depth_abs_rel", "depth_delta1"}.issubset(avg_norm):
        raise RuntimeError(
            f"Upper eval {upper_path} predates sparse driving-depth evaluation. "
            "Re-run run_eval.sh in upper mode before skip evaluation."
        )
    return avg_norm, scene_metrics


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


def _checkpoint_dir(out_root: Path) -> Path:
    return out_root / "checkpoints"


def _checkpoint_path(out_root: Path, scene_id: str) -> Path:
    return _checkpoint_dir(out_root) / f"{_safe_scene_filename(scene_id)}.jsonl"


def _manifest_path(out_root: Path) -> Path:
    return out_root / "skip_eval_manifest_v3.json"


def _manifest_fingerprint(config: Dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prepare_checkpoint_manifest(
    out_root: Path,
    *,
    config: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    manifest_path = _manifest_path(out_root)
    checkpoint_dir = _checkpoint_dir(out_root)
    derived = (
        out_root / "skip_eval_report_v3.json",
        out_root / "skip_eval_scenes_v3.json",
        out_root / "skip_eval_v3.npz",
        out_root / "skip_eval_summary_v3.json",
    )
    if not resume:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        for path in (manifest_path, *derived):
            if path.exists():
                path.unlink()

    fingerprint = _manifest_fingerprint(config)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(existing.get("fingerprint", "")) != fingerprint:
            raise RuntimeError(
                f"Skip-eval checkpoint configuration mismatch at {manifest_path}. "
                "Re-run with --no-resume to start a new checkpoint set."
            )
        return existing
    if resume and checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.jsonl")):
        raise RuntimeError(
            f"Found skip-eval checkpoints without {manifest_path}. "
            "Re-run with --no-resume to start a new checkpoint set."
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3_manifest",
        "fingerprint": fingerprint,
        "config": config,
    }
    _atomic_write_json(manifest_path, payload)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return payload


def _load_scene_checkpoint(out_root: Path, scene_id: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    path = _checkpoint_path(out_root, scene_id)
    if not path.is_file():
        return {}
    rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                _latency_log(
                    f"[skip_gt_eval][resume] ignore malformed checkpoint line {path}:{line_no}"
                )
                continue
            if not isinstance(row, dict) or str(row.get("scene_id", "")) != str(scene_id):
                continue
            try:
                span = (int(row["span_i"]), int(row["span_j"]))
            except (KeyError, TypeError, ValueError):
                continue
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict):
                rows[span] = normalize_metrics(metrics)
    return rows


def _append_scene_checkpoint(
    out_root: Path,
    *,
    scene_id: str,
    span: Tuple[int, int],
    metrics: Dict[str, float],
) -> None:
    path = _checkpoint_path(out_root, scene_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": SCHEMA_VERSION,
        "kind": "skip_gt_eval_v3_span",
        "scene_id": str(scene_id),
        "span_i": int(span[0]),
        "span_j": int(span[1]),
        "metrics": dict(metrics),
    }
    with path.open("a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _load_all_checkpoint_rows(out_root: Path) -> List[Dict[str, Any]]:
    checkpoint_dir = _checkpoint_dir(out_root)
    if not checkpoint_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.jsonl")):
        scene_id = ""
        span_raw: Dict[Tuple[int, int], Dict[str, float]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("scene_id", "")).strip()
                if not sid:
                    continue
                scene_id = sid
                try:
                    span = (int(item["span_i"]), int(item["span_j"]))
                except (KeyError, TypeError, ValueError):
                    continue
                metric_row = item.get("metrics", {})
                if isinstance(metric_row, dict):
                    span_raw[span] = normalize_metrics(metric_row)
        if scene_id:
            rows.append(
                {
                    "scene_id": scene_id,
                    "upper_average": {},
                    "span_raw_metrics": _serialize_span_raw(span_raw),
                }
            )
    return rows


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


def _span_raw_is_complete(span_raw: Dict[Tuple[int, int], Dict[str, float]], spans: Sequence[Tuple[int, int]]) -> bool:
    return all(span in span_raw for span in spans)


def _staging_row_is_complete(row: Dict[str, Any], spans: Sequence[Tuple[int, int]]) -> bool:
    sid = str(row.get("scene_id", "")).strip()
    if not sid:
        return False
    span_raw = _deserialize_span_raw(row.get("span_raw_metrics", {}))
    return _span_raw_is_complete(span_raw, spans)


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


def _average_span_metrics_from_staging(
    staging_rows: Sequence[Dict[str, Any]],
    spans: Sequence[Tuple[int, int]],
    metrics: Sequence[str],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    span_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    decoded_rows = [
        _deserialize_span_raw(row.get("span_raw_metrics", {})) for row in staging_rows
    ]
    for span in spans:
        avg_row: Dict[str, float] = {}
        for m in metrics:
            vals: List[float] = []
            for span_raw in decoded_rows:
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
    upper_scene_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    backbone: str,
    dataset: str,
    max_span: int,
    fusion_mode: str,
    elapsed_seconds: float,
    ensure_scene_reports: bool = False,
    supplemental_span_metrics_by_scene: Optional[
        Dict[str, Dict[Tuple[int, int], Dict[str, float]]]
    ] = None,
    materialize_intervals: bool = True,
) -> Path:
    staging_rows = _load_all_checkpoint_rows(out_root)
    if supplemental_span_metrics_by_scene:
        rows_by_scene = {
            str(row.get("scene_id", "")): row
            for row in staging_rows
            if str(row.get("scene_id", "")).strip()
        }
        for scene_id, supplemental in supplemental_span_metrics_by_scene.items():
            row = rows_by_scene.get(str(scene_id))
            if row is None:
                row = {
                    "scene_id": str(scene_id),
                    "upper_average": {},
                    "span_raw_metrics": {},
                }
                staging_rows.append(row)
                rows_by_scene[str(scene_id)] = row
            span_raw = _deserialize_span_raw(row.get("span_raw_metrics", {}))
            for span, metrics_row in supplemental.items():
                logical_span = (int(span[0]), int(span[1]))
                existing = span_raw.get(logical_span)
                normalized = normalize_metrics(metrics_row)
                if existing is not None and existing != normalized:
                    raise ValueError(
                        "Supplemental skip metrics conflict with checkpoint data for "
                        f"scene={scene_id}, span={logical_span}"
                    )
                span_raw[logical_span] = normalized
            row["span_raw_metrics"] = _serialize_span_raw(span_raw)
    for row in staging_rows:
        sid = str(row.get("scene_id", ""))
        row["upper_average"] = dict((upper_scene_metrics or {}).get(sid, {}))
    complete_rows = [row for row in staging_rows if _staging_row_is_complete(row, spans)]
    if ensure_scene_reports:
        scenes_payload = _init_scenes_payload(backbone=backbone, dataset=dataset)
        scene_reports: List[Dict[str, Any]] = []
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
                scene_reports.append(scene_report)
        scene_reports.sort(key=lambda item: str(item.get("scene_id", "")))
        scenes_payload["scenes"] = scene_reports
        scenes_payload["num_scenes"] = len(scene_reports)
        _atomic_write_json(_scenes_json_path(out_root), scenes_payload)

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
        "checkpoint_dir": str(_checkpoint_dir(out_root)),
    }
    summary_path = out_root / "skip_eval_summary_v3.json"
    _atomic_write_json(summary_path, summary)
    if materialize_intervals:
        try:
            from analysis.redundancy.convert_skip_eval_to_intervals import (
                materialize_skip_eval_intervals,
            )

            materialize_skip_eval_intervals(out_root, commit=True)
        except Exception as exc:
            print(
                f"[warn] Could not materialize skip_eval interval layout under {out_root}: {exc}",
                file=sys.stderr,
            )
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
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
    resume: bool = True,
) -> Path:
    timing_on = _timing_enabled(timing)
    run_t0 = time.perf_counter()
    if timing_on:
        _timing_log(f"[timing] skip_gt_eval timing enabled dataset={dataset} backbone={backbone}")

    t_phase = time.perf_counter()
    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    if not adapter.supports_incremental_skip_eval():
        raise RuntimeError(
            f"Backbone {adapter.name!r} does not yet implement the incremental skip-eval adaptor API"
        )
    ds_plugin = get_dataset(dataset)
    split = ds_plugin.validate_split(split)
    tagged_dataset = str(output_dataset).strip() or ds_plugin.output_tag(split)
    _latency_log(
        f"[skip_gt_eval] start dataset={tagged_dataset} backbone={backbone} "
        f"max_scenes={max_scenes} max_frames={max_frames} max_span={max_span}"
    )
    torch_device = torch.device(device)
    hf = str(hf_model or adapter.default_hf_model())
    _latency_log(f"[skip_gt_eval] init adapter+dataset {time.perf_counter() - t_phase:.2f}s")

    out_root = output_dir or stage_dir(experiment, tagged_dataset, "eval/pruning")
    out_root.mkdir(parents=True, exist_ok=True)

    t_phase = time.perf_counter()
    _latency_log("[skip_gt_eval] load_pretrained (single dataset model) ...")
    model = adapter.load_pretrained(hf_model=hf, device=str(torch_device), eval_mode=True)
    model.eval()
    _latency_log(f"[skip_gt_eval] load_pretrained done {time.perf_counter() - t_phase:.2f}s")

    t_phase = time.perf_counter()
    unit_ids = unit_ids_from_catalog(adapter, model)
    n_units = len(unit_ids)
    if n_units < 1:
        raise RuntimeError(f"No units in catalog for backbone={backbone}")
    _latency_log(f"[skip_gt_eval] unit catalog n_units={n_units} {time.perf_counter() - t_phase:.3f}s")

    t_phase = time.perf_counter()
    spans = list(iter_unit_spans(n_units, max_span=int(max_span)))
    metrics_available = list(SKIP_EVAL_METRICS)
    _latency_log(f"[skip_gt_eval] span enumeration n_spans={len(spans)} {time.perf_counter() - t_phase:.3f}s")

    t_phase = time.perf_counter()
    upper_metrics, upper_scene_metrics = load_upper_eval_or_fail(
        experiment=experiment,
        tagged_dataset=tagged_dataset,
        require_depth=dataset in {"nuscenes", "waymo"} and not adapter.periodic_eval_skip_depth(),
    )
    _latency_log(
        f"[skip_gt_eval] load upper eval ({len(upper_scene_metrics)} scenes) "
        f"{time.perf_counter() - t_phase:.2f}s"
    )

    metrics = [m for m in metrics_available if m in upper_metrics]
    if not metrics:
        raise RuntimeError(f"No overlapping metrics between upper eval and skip_eval list: {upper_metrics}")
    total_spans = len(spans)
    manifest_config: Dict[str, Any] = {
        "backbone": str(adapter.name),
        "dataset": str(tagged_dataset),
        "source_dataset": str(dataset),
        "data_root": str(Path(data_root).resolve()),
        "split": split,
        "hf_model": hf,
        "max_scenes": int(max_scenes),
        "max_frames": int(max_frames),
        "max_span": int(max_span),
        "mode": str(mode),
        "metrics": list(metrics),
        "unit_ids": list(unit_ids),
        "spans": [[int(i), int(j)] for i, j in spans],
        "skip_depth": bool(adapter.periodic_eval_skip_depth()),
    }
    _prepare_checkpoint_manifest(out_root, config=manifest_config, resume=bool(resume))
    _latency_log(
        f"[skip_gt_eval] entering scene-first loop: {total_spans} spans, "
        f"{len(metrics)} metrics, resume={bool(resume)}, elapsed={time.perf_counter() - run_t0:.2f}s"
    )
    profile = ds_plugin.eval_profile()
    eval_ctx = EvalSceneContext(
        device=torch_device,
        profile=profile,
        skip_depth=bool(adapter.periodic_eval_skip_depth()),
        export_vis_2d=False,
        export_vis_3d=False,
        vis_save_root=None,
        vis_max_points=0,
        backbone_name=str(adapter.name),
        dataset_name=str(ds_plugin.name),
    )
    total_scenes = ds_plugin.count_scenes(
        data_root, max_scenes=int(max_scenes), split=split
    )
    scene_source = ds_plugin.iter_scenes(
        data_root,
        max_scenes=int(max_scenes),
        max_frames=int(max_frames),
        split=split,
    )
    scene_iter = tqdm(
        scene_source,
        total=total_scenes,
        desc=(
            "    scenes"
            if os.environ.get("COMPACT_LOG", "").lower() in {"1", "true", "yes"}
            else f"skip_eval {adapter.name} ({tagged_dataset})"
        ),
        unit="scene",
        file=sys.stderr,
        dynamic_ncols=True,
        position=0,
        disable=not progress,
    )
    span_set = set(spans)
    for scene_index, batch in enumerate(scene_iter):
        sid = str(batch.scene_id)
        completed = _load_scene_checkpoint(out_root, sid) if resume else {}
        missing = [span for span in spans if span not in completed]
        if not missing:
            _latency_log(
                f"[skip_gt_eval][resume] scene={sid} reused all {total_spans} spans"
            )
            continue
        if progress:
            scene_iter.set_postfix_str(f"{sid} missing={len(missing)}", refresh=False)
        scene_t0 = time.perf_counter()
        prepared = None
        span_iter = None
        try:
            prepared = ds_plugin.prepare_skip_eval_scene(adapter, batch, eval_ctx)
            if not isinstance(prepared.images, torch.Tensor):
                raise TypeError(
                    f"Dataset adaptor {ds_plugin.name!r} returned non-tensor skip-eval images"
                )
            prepare_s = time.perf_counter() - scene_t0
            yielded: Set[Tuple[int, int]] = set()
            missing_set = set(missing)
            replay_t0 = time.perf_counter()
            prediction_iter = adapter.iter_skip_span_predictions(
                model, prepared.images, missing
            )
            span_iter = tqdm(
                prediction_iter,
                total=len(missing),
                desc=(
                    f"      spans ({sid})"
                    if os.environ.get("COMPACT_LOG", "").lower() in {"1", "true", "yes"}
                    else f"  spans {sid}"
                ),
                unit="span",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                position=1,
                disable=not progress,
            )
            for i, j, prediction in span_iter:
                span = (int(i), int(j))
                if progress:
                    span_iter.set_postfix_str(f"({span[0]},{span[1]})", refresh=False)
                if span not in span_set or span not in missing_set:
                    raise RuntimeError(f"Adaptor yielded unrequested skip span {span}")
                if span in yielded:
                    raise RuntimeError(f"Adaptor yielded duplicate skip span {span}")
                yielded.add(span)
                try:
                    metric_row = normalize_metrics(
                        ds_plugin.score_skip_eval_prediction(
                            adapter, prediction, prepared, eval_ctx
                        )
                    )
                    _append_scene_checkpoint(
                        out_root,
                        scene_id=sid,
                        span=span,
                        metrics=metric_row,
                    )
                except Exception as e:
                    _latency_log(f"[warn] scene={sid} span={span} scoring failed: {e}")
                finally:
                    del prediction
            span_iter.close()
            if yielded != missing_set:
                absent = sorted(missing_set.difference(yielded))
                extra = sorted(yielded.difference(missing_set))
                raise RuntimeError(
                    f"Incremental adaptor span coverage mismatch: missing={absent[:8]} extra={extra[:8]}"
                )
            scene_s = time.perf_counter() - scene_t0
            replay_s = time.perf_counter() - replay_t0
            _latency_log(
                f"[skip_gt_eval] scene {scene_index + 1}/{total_scenes or '?'} {sid} "
                f"prepare={prepare_s:.2f}s replay+score={replay_s:.2f}s total={scene_s:.2f}s"
            )
            if timing_on:
                _timing_log(
                    f"[timing] scene={sid} prepare={prepare_s:.3f}s "
                    f"replay_score={replay_s:.3f}s total={scene_s:.3f}s"
                )
        except Exception as e:
            _latency_log(f"[warn] scene={sid} incremental skip eval failed: {e}")
        finally:
            if span_iter is not None:
                span_iter.close()
            if prepared is not None:
                del prepared
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    _latency_log(f"[skip_gt_eval] scene loop finished, finalizing ... elapsed={time.perf_counter() - run_t0:.1f}s")
    return _finalize_skip_outputs_from_disk(
        out_root=out_root,
        spans=spans,
        n_units=n_units,
        unit_ids=unit_ids,
        metrics=metrics,
        upper_metrics=upper_metrics,
        upper_scene_metrics=upper_scene_metrics,
        backbone=adapter.name,
        dataset=tagged_dataset,
        max_span=int(max_span),
        fusion_mode=str(mode),
        elapsed_seconds=time.perf_counter() - run_t0,
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
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()

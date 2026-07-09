"""Build span-indexed prune-plan bundles from CKA triangle picks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    interval_selector,
)
from analysis.prune_plan.build_prune_plan_common import (
    default_plan_root as _default_plan_root,
    load_similarity,
)
from analysis.redundancy.similarity_v3 import (
    REDUNDANCY_GRAPH_REGIONS,
)
from analysis.prune_plan.triangle_prune import (
    CORNER_RANKING_TOLERANCE,
    REQUIRE_ENDPOINT_LINKED_SPANS,
    TrianglePick,
    scan_endpoint_linked_closed_intervals_with_candidates,
)
from analysis.prune_plan.visualize_triangle_prune import (
    render_corner_overview_heatmap,
    render_gt_component_heatmaps,
    render_gt_trajectory_score_plot,
    render_prune_span_drop_grid,
    render_budget_similarity_curve as render_span_similarity_curve,
)

GT_RELATIVE_DEGRADATION_CAP_MULTIPLE = 0.0
GT_COMPONENT_AGGREGATION = "mean"  # "mean", "min", or "max"; lower aggregate is selected.
GT_COMPONENT_TOP_RANGE_CLIP_FRACTION = 0.0
CKA_CORNER_COMPARISON_TOKEN_REGIONS = ("special", "patch", "all", "mean")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Build span-indexed prune-plan bundle as {PRUNE_PLAN_BUNDLE_FILENAME}."
    )
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset name, or comma/space separated dataset names to average explicitly "
            "(for example scannet_dev50,nuscenes_train). Dataset values 'mean'/'min' are not supported."
        ),
    )
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--metric", type=str, default="cka")
    p.add_argument(
        "--token-region",
        type=str,
        choices=(*REDUNDANCY_GRAPH_REGIONS, "mean", "min"),
        default="all",
    )
    p.add_argument("--prune-plan-root", type=Path, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Explicit bundle output path (default: prune-plan-root/{PRUNE_PLAN_BUNDLE_FILENAME}).",
    )
    p.add_argument(
        "--viz-out-dir",
        type=Path,
        default=None,
        help="Curve output directory (default: prune-plan-root).",
    )
    p.add_argument(
        "--skip-param-accounting",
        action="store_true",
        help="Do not compute/store linear aggregator block parameter counts in the bundle.",
    )
    p.add_argument(
        "--gt-corner-skip-eval-report",
        type=Path,
        default=None,
        help=(
            "Optional skip_eval_report_v3.json used to build the multi-metric GT aggregate "
            "corner heatmap. Default: prune-plan-root/scannet_dev50/skip_eval/skip_eval_report_v3.json if present."
        ),
    )
    p.add_argument(
        "--gt-corner-metric",
        type=str,
        default="all",
        help=(
            "Metric list from skip_eval_report_v3.json used for the GT aggregate path. "
            "Use 'all' (default), or comma/space separated names."
        ),
    )
    return p


def _format_param_count(n: int) -> str:
    v = float(n)
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return str(int(v))


def _compute_aggregator_total_params(backbone: str) -> int:
    """Count the full prunable aggregator/block parameter scope once."""
    import torch

    from analysis.backbones.bootstrap import apply_pythonpath
    from analysis.backbones.cache_replay import get_cache_replay_engine
    from analysis.backbones.registry import get_backbone

    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    engine = get_cache_replay_engine(backbone)
    device = torch.device("cpu")
    model, _trainable, pre_prune_total_params, _removed = engine.build_train_model(
        hf_model=adapter.default_hf_model(),
        device=device,
        drop_actions=[],
        lora_actions=[],
        frame_skip_intervals=[],
        global_skip_intervals=[],
        skipped_frame_idxs=[],
        skipped_global_idxs=[],
        prune=False,
        inject_lora=False,
    )
    # Prefer the explicit pre-prune count returned by the engine, but verify it
    # against the shared accounting scope for backbones that override it.
    counted = sum(p.numel() for p in engine.parameter_accounting_parameters(model))
    if int(pre_prune_total_params) != int(counted):
        print(
            "[warn] Aggregator param count mismatch: "
            f"engine returned {_format_param_count(int(pre_prune_total_params))}, "
            f"accounting scope counted {_format_param_count(int(counted))}; using counted value.",
            flush=True,
        )
    return int(counted)


def _build_param_accounting(
    *,
    backbone: str,
    num_units: int,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {
            "available": False,
            "method": "linear_by_remaining_units",
            "scope": "prunable_aggregator_blocks",
            "skip_reason": "skipped_by_cli",
        }
    try:
        total_params = _compute_aggregator_total_params(backbone)
    except Exception as exc:
        print(f"[warn] Could not compute aggregator param accounting for {backbone}: {exc}", flush=True)
        return {
            "available": False,
            "method": "linear_by_remaining_units",
            "scope": "prunable_aggregator_blocks",
            "skip_reason": str(exc),
        }
    per_unit = float(total_params) / float(num_units) if int(num_units) > 0 else float("nan")
    print(
        f"[params] {backbone}: aggregator_total={_format_param_count(total_params)} "
        f"units={int(num_units)} per_unit≈{_format_param_count(int(round(per_unit)))}",
        flush=True,
    )
    return {
        "available": True,
        "method": "linear_by_remaining_units",
        "scope": "prunable_aggregator_blocks",
        "total_units": int(num_units),
        "total_params": int(total_params),
        "per_unit_params": float(per_unit),
        "assumption": "equal parameter count per prunable unit",
    }


def _span_param_record(
    *,
    param_accounting: dict[str, Any],
    num_units: int,
    drop_count: int,
) -> dict[str, Any]:
    remaining_units = max(0, int(num_units) - int(drop_count))
    ratio = float(remaining_units) / float(num_units) if int(num_units) > 0 else float("nan")
    out: dict[str, Any] = {
        "drop_count": int(drop_count),
        "remaining_units": int(remaining_units),
        "aggregator_param_ratio": float(ratio) if math.isfinite(ratio) else None,
    }
    if bool(param_accounting.get("available", False)):
        total_params = int(param_accounting.get("total_params", 0))
        out["aggregator_params"] = int(round(float(total_params) * ratio))
    return out


def _unpack_upper_triangle(packed: list, n: int) -> np.ndarray:
    expected = int(n) * (int(n) + 1) // 2
    if len(packed) != expected:
        raise ValueError(f"packed length {len(packed)} != expected {expected} for n={n}")
    mat = np.full((int(n), int(n)), np.nan, dtype=np.float64)
    k = 0
    for i in range(int(n)):
        for j in range(i, int(n)):
            mat[i, j] = float(packed[k])
            k += 1
    return mat


def build_interval_score_matrix(sim: np.ndarray) -> np.ndarray:
    """Build closed-interval scores from unit-to-unit similarities.

    Cell ``[start, end]`` represents deleting the closed interval
    ``[start, end]``.  For ``start > 0`` it uses the raw similarity between
    boundary unit ``start - 1`` and endpoint ``end``.  The input/embedding
    boundary for ``start == 0`` is not stored yet, so this row is a deterministic
    placeholder.
    """
    values = np.asarray(sim, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected square similarity matrix, got shape={values.shape}")
    n = int(values.shape[0])
    out = np.full((n, n), np.nan, dtype=np.float64)
    for start in range(n):
        for end in range(start, n):
            out[start, end] = 0.0 if start == 0 else float(values[start - 1, end])
    return out


def _parse_metric_selector(selector: str, available: Sequence[str]) -> list[str]:
    key = str(selector or "all").strip()
    available_list = [str(x) for x in available]
    if not key or key.lower() == "all":
        return available_list
    requested = [x for x in key.replace(",", " ").split() if x.strip()]
    available_set = set(available_list)
    missing = [m for m in requested if m not in available_set]
    if missing:
        raise ValueError(
            f"requested GT metric(s) not found: {missing}; available={available_list}"
        )
    return requested


_GT_POSE_METRICS = ("ape", "are", "rpe_trans", "rpe_rot")
_GT_POINT_METRICS = ("acc", "comp", "chamfer_distance", "nc")
_GT_AGGREGATE_METRICS = (*_GT_POSE_METRICS, *_GT_POINT_METRICS)
_HIGHER_IS_BETTER_GT_METRICS = frozenset({"depth_delta1", "nc"})


def _clip_top_score_values(
    scores: np.ndarray,
    *,
    fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip the largest fraction of a component's finite score values."""
    arr = np.asarray(scores, dtype=np.float64)
    out = np.array(arr, copy=True, dtype=np.float64)
    frac = max(0.0, min(1.0, float(fraction)))
    meta: dict[str, Any] = {
        "clip_fraction": float(frac),
        "clip_mode": "top_value_fraction",
        "enabled": bool(frac > 0.0),
        "raw_min": None,
        "raw_max": None,
        "clip_threshold": None,
        "num_clipped": 0,
    }
    finite = np.isfinite(arr)
    if not np.any(finite) or frac <= 0.0:
        return out, meta
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    meta["raw_min"] = lo
    meta["raw_max"] = hi
    if hi <= lo:
        meta["clip_threshold"] = hi
        return out, meta
    threshold = float(np.nanquantile(arr[finite], 1.0 - frac))
    clip_mask = finite & (arr > threshold)
    out[clip_mask] = threshold
    meta["clip_threshold"] = float(threshold)
    meta["num_clipped"] = int(np.count_nonzero(clip_mask))
    return out, meta


def _degrade_pct_to_upper_multiple_score(
    values: np.ndarray,
    *,
    metric: str,
    upper_baseline: Optional[float],
    max_multiple: float = GT_RELATIVE_DEGRADATION_CAP_MULTIPLE,
    top_value_clip_fraction: float = GT_COMPONENT_TOP_RANGE_CLIP_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert direction-normalized degrade_pct to a clipped 1..K raw-metric ratio.

    ``values`` are positive-when-worse percentages from skip_eval.  For metrics
    where lower is better, the score is ``current / upper``.  For metrics where
    higher is better, the score is ``upper / current``.  Scores better than the
    upper baseline are floored to 1.0.  Before the absolute ``max_multiple`` cap,
    the largest fraction of this metric's own finite score values is clipped to
    the corresponding quantile threshold.  If
    ``max_multiple >= 1`` catastrophic values are capped at ``max_multiple``; if
    ``max_multiple < 1`` the score is left uncapped.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    meta: dict[str, Any] = {
        "scale_mode": "upper_metric_multiple",
        "upper_baseline": None,
        "max_multiple": float(max_multiple),
        "cap_enabled": bool(float(max_multiple) >= 1.0),
        "raw_bound": None,
        "top_value_clip": None,
        "top_range_clip": None,
        "num_capped_high": 0,
        "num_floored_to_one": 0,
        "enabled": False,
    }
    if not np.any(finite):
        return out, meta
    if upper_baseline is None:
        meta["skip_reason"] = "missing_upper_baseline"
        return out, meta
    upper = float(upper_baseline)
    multiple = float(max_multiple)
    if not math.isfinite(upper) or not math.isfinite(multiple):
        meta["skip_reason"] = "invalid_upper_or_multiple"
        return out, meta
    cap_enabled = bool(multiple >= 1.0)
    denom = max(abs(upper), 1e-8)
    metric_key = str(metric)
    if metric_key in _HIGHER_IS_BETTER_GT_METRICS:
        if upper <= 0:
            meta["skip_reason"] = "nonpositive_upper_for_higher_is_better_metric"
            return out, meta
        raw_floor = upper / multiple if cap_enabled else None
        raw_current = upper - arr[finite] * denom / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = upper / np.maximum(raw_current, 1e-8)
        scores = np.where(np.isfinite(scores), scores, multiple if cap_enabled else np.nan)
        scores, top_clip_meta = _clip_top_score_values(
            scores,
            fraction=top_value_clip_fraction,
        )
        capped_scores = (
            np.clip(scores, 1.0, multiple)
            if cap_enabled
            else np.maximum(scores, 1.0)
        )
        meta.update(
            {
                "enabled": True,
                "cap_enabled": cap_enabled,
                "raw_bound": float(raw_floor) if raw_floor is not None else None,
                "bound_kind": "raw_floor" if cap_enabled else "none",
                "top_value_clip": top_clip_meta,
                "top_range_clip": top_clip_meta,
                "num_capped_high": int(np.count_nonzero(scores > multiple)) if cap_enabled else 0,
                "num_floored_to_one": int(np.count_nonzero(scores < 1.0)),
            }
        )
    else:
        if upper <= 0:
            meta["skip_reason"] = "nonpositive_upper_for_lower_is_better_metric"
            return out, meta
        raw_ceiling = upper * multiple if cap_enabled else None
        raw_current = upper + arr[finite] * denom / 100.0
        scores = raw_current / upper
        scores, top_clip_meta = _clip_top_score_values(
            scores,
            fraction=top_value_clip_fraction,
        )
        capped_scores = (
            np.clip(scores, 1.0, multiple)
            if cap_enabled
            else np.maximum(scores, 1.0)
        )
        meta.update(
            {
                "enabled": True,
                "cap_enabled": cap_enabled,
                "raw_bound": float(raw_ceiling) if raw_ceiling is not None else None,
                "bound_kind": "raw_ceiling" if cap_enabled else "none",
                "top_value_clip": top_clip_meta,
                "top_range_clip": top_clip_meta,
                "num_capped_high": int(np.count_nonzero(scores > multiple)) if cap_enabled else 0,
                "num_floored_to_one": int(np.count_nonzero(scores < 1.0)),
            }
        )
    out[finite] = capped_scores
    meta["upper_baseline"] = float(upper)
    return out, meta


def _load_gt_aggregate_from_skip_eval_report(
    report_path: Path,
    *,
    unit_ids: Sequence[str],
    metric_selector: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[TrianglePick], dict[str, Any]]:
    """Build an aggregate GT heatmap over closed drop intervals [start, end]."""
    report_path = Path(report_path)
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    mode = ""
    modes = raw.get("fusion_modes", [])
    if isinstance(modes, list) and modes:
        mode = str(modes[0])
    fusion = raw.get("fusion_by_mode", {})
    mode_payload = fusion.get(mode) if isinstance(fusion, dict) else None
    if not isinstance(mode_payload, dict):
        raise ValueError(f"skip_eval report has no fusion payload for mode={mode!r}: {report_path}")
    by_metric = mode_payload.get("gt_degrade_pct_upper_by_metric", {})
    if not isinstance(by_metric, dict):
        raise ValueError(f"skip_eval report has no gt_degrade_pct_upper_by_metric: {report_path}")
    selected_metric_names = _parse_metric_selector(
        str(metric_selector), sorted(str(k) for k in by_metric.keys())
    )
    selected_set = set(selected_metric_names)
    metric_names = [m for m in _GT_AGGREGATE_METRICS if m in selected_set]
    if not metric_names:
        raise ValueError(
            "No supported non-depth GT metric selected. "
            f"Supported metrics={list(_GT_AGGREGATE_METRICS)}, selected={selected_metric_names}"
        )
    upper_average = raw.get("upper_average", {})
    if not isinstance(upper_average, dict):
        upper_average = {}

    n = int(raw.get("num_layers", len(unit_ids)))
    if n != len(unit_ids):
        raise ValueError(
            f"skip_eval num_layers={n} does not match current unit count={len(unit_ids)}"
        )
    max_span = int(raw.get("max_span", 0))
    has_cap = max_span > 0
    metric_records: list[dict[str, Any]] = []
    component_heatmaps: dict[str, np.ndarray] = {}
    component_metric_names: list[str] = []

    for metric_key in metric_names:
        packed = by_metric.get(metric_key)
        if not isinstance(packed, list):
            continue
        skip_values = _unpack_upper_triangle(packed, n)
        corner_values = np.full((n, n), np.nan, dtype=np.float64)
        for a in range(0, n):
            b_max = n - 1 if not has_cap else min(n - 1, a + max_span - 1)
            for b in range(a, b_max + 1):
                val = float(skip_values[a, b])
                if not math.isfinite(val):
                    continue
                corner_values[a, b] = val
        multiple_score, scaling_meta = _degrade_pct_to_upper_multiple_score(
            corner_values,
            metric=str(metric_key),
            upper_baseline=(
                float(upper_average[str(metric_key)])
                if str(metric_key) in upper_average
                and isinstance(upper_average[str(metric_key)], (int, float))
                else None
            ),
            max_multiple=GT_RELATIVE_DEGRADATION_CAP_MULTIPLE,
            top_value_clip_fraction=GT_COMPONENT_TOP_RANGE_CLIP_FRACTION,
        )
        component_heatmaps[str(metric_key)] = multiple_score
        component_metric_names.append(str(metric_key))

        finite_raw = np.isfinite(corner_values)
        finite_score = np.isfinite(multiple_score)
        if np.any(finite_raw):
            metric_records.append(
                {
                    "metric": str(metric_key),
                    "degrade_pct_min": float(np.nanmin(corner_values[finite_raw])),
                    "degrade_pct_max": float(np.nanmax(corner_values[finite_raw])),
                    "multiple_score_min": (
                        float(np.nanmin(multiple_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "multiple_score_max": (
                        float(np.nanmax(multiple_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "scaling": scaling_meta,
                    "num_finite": int(np.count_nonzero(finite_raw)),
                }
            )

    aggregate = np.full((n, n), np.nan, dtype=np.float64)
    if component_heatmaps:
        stack = np.stack(
            [np.asarray(component_heatmaps[name], dtype=np.float64) for name in component_metric_names],
            axis=0,
        )
        finite_aggregate = np.any(np.isfinite(stack), axis=0)
        aggregation = str(GT_COMPONENT_AGGREGATION).strip().lower()
        with np.errstate(all="ignore"):
            if aggregation == "mean":
                aggregate[finite_aggregate] = np.nanmean(stack, axis=0)[finite_aggregate]
            elif aggregation == "min":
                aggregate[finite_aggregate] = np.nanmin(stack, axis=0)[finite_aggregate]
            elif aggregation == "max":
                aggregate[finite_aggregate] = np.nanmax(stack, axis=0)[finite_aggregate]
            else:
                raise ValueError(
                    f"Unknown GT_COMPONENT_AGGREGATION={GT_COMPONENT_AGGREGATION!r}; "
                    "expected 'mean', 'min', or 'max'"
                )

    best_by_drop_count: dict[int, tuple[float, int, int]] = {}
    for start in range(0, n):
        for end in range(start, n):
            val = float(aggregate[start, end])
            if not math.isfinite(val):
                continue
            drop_count = int(end - start + 1)
            current = best_by_drop_count.get(drop_count)
            if current is None or val < current[0] - 1e-12:
                best_by_drop_count[drop_count] = (val, start, end)

    picks: list[TrianglePick] = []
    records: list[dict[str, Any]] = []
    for drop_count in sorted(best_by_drop_count):
        value, start, end = best_by_drop_count[drop_count]
        pick = TrianglePick(
            i=start,
            j=end,
            span=int(drop_count),
            unit_i=str(unit_ids[start]),
            unit_j=str(unit_ids[end]),
            dropped_unit_ids=tuple(str(unit_ids[k]) for k in range(start, end + 1)),
        )
        picks.append(pick)
        records.append(
            {
                "span": int(drop_count),
                "i": int(start),
                "j": int(end),
                "start": int(start),
                "end": int(end),
                "skip_span_i": int(start),
                "skip_span_j": int(end),
                "aggregate_score": float(value),
            }
        )
    meta = {
        "source": str(report_path),
        "metric_selector": str(metric_selector),
        "metrics": [str(x) for x in component_metric_names],
        "component_aggregation": str(GT_COMPONENT_AGGREGATION),
        "metric_normalization": (
            "Only pose metrics (ape, are, rpe_trans, rpe_rot) and point-cloud "
            "metrics (acc, comp, chamfer_distance, nc) are used. For each metric, "
            "direction-normalized degrade_pct is converted back to a raw-metric "
            "relative degradation multiple: lower-is-better metrics use "
            "current/upper, higher-is-better metrics use upper/current. Values "
            "better than upper are floored to 1. "
            + (
                f"Catastrophic values are capped at {GT_RELATIVE_DEGRADATION_CAP_MULTIPLE:g}, "
                f"and the resulting 1..{GT_RELATIVE_DEGRADATION_CAP_MULTIPLE:g} component heatmaps are averaged."
                if GT_RELATIVE_DEGRADATION_CAP_MULTIPLE >= 1.0
                else "No maximum degradation cap is applied; uncapped component heatmaps are averaged."
            )
            + (
                f" Before the absolute cap, the top "
                f"{GT_COMPONENT_TOP_RANGE_CLIP_FRACTION:g} fraction of each "
                "component's largest finite score values are clipped to the "
                "corresponding quantile threshold."
            )
            + f" Components are aggregated with {GT_COMPONENT_AGGREGATION!r}."
        ),
        "metric_records": metric_records,
        "fusion_mode": mode,
        "span_mapping": "skip_eval closed [a,b] -> prune closed interval i{a}_j{b}",
        "records": records,
    }
    return aggregate, component_heatmaps, picks, meta


def _default_gt_corner_report_path(plan_root: Path) -> Path:
    return Path(plan_root) / "scannet_dev50" / "skip_eval" / "skip_eval_report_v3.json"


def _pick_drop_indices(pick: TrianglePick) -> list[int]:
    return list(range(int(pick.i), int(pick.j) + 1))


def _record_from_pick(
    *,
    pick: TrianglePick,
    unit_ids: Sequence[str],
    num_units: int,
    param_accounting: dict[str, Any],
    score_name: str,
    score_value: float,
    source_strategy: str,
) -> dict[str, Any]:
    start = int(pick.i)
    end = int(pick.j)
    key = interval_selector(start, end)
    drop = _pick_drop_indices(pick)
    score_source_index = start - 1
    score_placeholder = bool(start == 0 and str(score_name) == "corner_similarity")
    record: dict[str, Any] = {
        "selector": key,
        "i": start,
        "j": end,
        "start": start,
        "end": end,
        "span": int(pick.span),
        "drop_count": len(drop),
        "score_source_index": int(score_source_index),
        "score_source_unit_id": (
            "input_embedding" if start == 0 else str(unit_ids[score_source_index])
        ),
        "score_endpoint_unit_index": end,
        "score_endpoint_unit_id": str(unit_ids[end]) if 0 <= end < len(unit_ids) else "",
        "score_placeholder": score_placeholder,
        "unit_i": str(pick.unit_i),
        "unit_j": str(pick.unit_j),
        "drop": [int(idx) for idx in drop],
        "dropped_unit_ids": [
            str(unit_ids[int(idx)]) for idx in drop if 0 <= int(idx) < len(unit_ids)
        ],
        "source_strategies": [str(source_strategy)],
        **_span_param_record(
            param_accounting=param_accounting,
            num_units=num_units,
            drop_count=len(drop),
        ),
    }
    if math.isfinite(float(score_value)):
        record[str(score_name)] = float(score_value)
    return record


def _merge_interval_record(
    intervals: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    source_strategy: str,
) -> None:
    key = str(record["selector"])
    existing = intervals.get(key)
    if existing is None:
        intervals[key] = record
        return
    strategies = set(str(x) for x in list(existing.get("source_strategies", [])))
    strategies.add(str(source_strategy))
    existing["source_strategies"] = sorted(strategies)
    for k, v in record.items():
        if k not in existing:
            existing[k] = v


def _strategy_from_picks(
    *,
    key: str,
    display_name: str,
    picks: Sequence[TrianglePick],
    unit_ids: Sequence[str],
    num_units: int,
    param_accounting: dict[str, Any],
    score: Optional[np.ndarray],
    score_name: str,
    source: dict[str, Any],
    intervals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    points: dict[str, str] = {}
    scores: dict[str, float] = {}
    score_arr = None if score is None else np.asarray(score, dtype=np.float64)
    for pick in sorted(picks, key=lambda p: int(p.span)):
        span = int(pick.span)
        value = float("nan")
        if score_arr is not None:
            value = float(score_arr[int(pick.i), int(pick.j)])
        record = _record_from_pick(
            pick=pick,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name=score_name,
            score_value=value,
            source_strategy=key,
        )
        interval_key = str(record["selector"])
        _merge_interval_record(intervals, record, source_strategy=key)
        points[str(span)] = interval_key
        if math.isfinite(value):
            scores[str(span)] = float(value)
    return {
        "display_name": display_name,
        "x_axis": "drop_count",
        "points": points,
        "scores": scores,
        "score_name": score_name,
        "source": source,
    }


def run_build_prune_plan(
    *,
    experiment: str,
    dataset: str,
    backbone: str = "vggt",
    metric: str = "cka",
    token_region: str = "all",
    prune_plan_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    viz_out_dir: Optional[Path] = None,
    skip_param_accounting: bool = False,
    gt_corner_skip_eval_report: Optional[Path] = None,
    gt_corner_metric: str = "all",
) -> List[Path]:
    """Build a prune-only span bundle and return the written path."""
    unit_ids, sim, similarity_source = load_similarity(
        experiment=experiment,
        dataset=dataset,
        token_region=token_region,
        metric=metric,
    )
    num_units = int(sim.shape[0])
    interval_score = build_interval_score_matrix(sim)
    scan, eligible_candidates_by_span = (
        scan_endpoint_linked_closed_intervals_with_candidates(
            interval_score,
            unit_ids,
            ranking_tolerance=CORNER_RANKING_TOLERANCE,
            require_endpoint_linked_spans=REQUIRE_ENDPOINT_LINKED_SPANS,
        )
    )
    selection_label = (
        "endpoint-linked"
        if bool(REQUIRE_ENDPOINT_LINKED_SPANS)
        else "independent best-corner"
    )
    max_available_span = max((int(pick.span) for pick in scan), default=0)
    param_accounting = _build_param_accounting(
        backbone=backbone,
        num_units=num_units,
        skip=bool(skip_param_accounting),
    )

    plan_root = prune_plan_root or _default_plan_root(backbone)
    plan_root.mkdir(parents=True, exist_ok=True)
    bundle_path = output_path or (plan_root / PRUNE_PLAN_BUNDLE_FILENAME)
    visualization_root = (
        viz_out_dir or (output_path.parent if output_path is not None else plan_root)
    )
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    for stale_name in [
        f"prune_plan_{safe_backbone}_all_corners.png",
        f"prune_plan_{safe_backbone}_token_corners.png",
        f"prune_plan_{safe_backbone}_gt_corners.png",
        f"prune_plan_{safe_backbone}_gt_ranked_candidates.png",
        f"prune_plan_{safe_backbone}_gt_corners_and_ranked_candidates.png",
    ]:
        stale_path = visualization_root / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_path in visualization_root.glob(f"prune_plan_{safe_backbone}_token_*_all_corners.png"):
        stale_path.unlink()
    gt_corner_score: Optional[np.ndarray] = None
    gt_component_heatmaps: dict[str, np.ndarray] = {}
    gt_corner_picks: list[TrianglePick] = []
    gt_corner_meta: Optional[dict[str, Any]] = None
    gt_score_trajectories: list[tuple[str, Sequence[TrianglePick]]] = []
    gt_report_path = gt_corner_skip_eval_report or _default_gt_corner_report_path(plan_root)
    if Path(gt_report_path).is_file():
        try:
            (
                gt_corner_score,
                gt_component_heatmaps,
                gt_corner_picks,
                gt_corner_meta,
            ) = _load_gt_aggregate_from_skip_eval_report(
                Path(gt_report_path),
                unit_ids=unit_ids,
                metric_selector=str(gt_corner_metric),
            )
            print(
                f"Loaded GT aggregate corner heatmap from {gt_report_path} "
                f"(metrics={len(gt_corner_meta.get('metrics', [])) if gt_corner_meta else 0}, "
                f"spans={len(gt_corner_picks)})",
                flush=True,
            )
        except Exception as exc:
            print(f"[warn] Could not load GT corner path from {gt_report_path}: {exc}", flush=True)

    interval_records: dict[str, dict[str, Any]] = {}
    strategies: dict[str, dict[str, Any]] = {}
    strategies["ours"] = _strategy_from_picks(
        key="ours",
        display_name="Ours",
        picks=scan,
        unit_ids=unit_ids,
        num_units=num_units,
        param_accounting=param_accounting,
        score=interval_score,
        score_name="corner_similarity",
        source={
            "kind": "similarity",
            "metric": str(metric),
            "token_region": str(token_region),
            "dataset": str(dataset),
            "similarity_source": similarity_source,
            "interval_score": "C[start,end]=sim[start-1,end] for start>0; C[0,end]=0 placeholder",
            "triangle_selection": (
                "endpoint_linked_descending_best_interval_score"
                if bool(REQUIRE_ENDPOINT_LINKED_SPANS)
                else "independent_best_interval_score_per_drop_count"
            ),
        },
        intervals=interval_records,
    )
    try:
        cosine_unit_ids, cosine_sim, cosine_similarity_source = load_similarity(
            experiment=experiment,
            dataset=dataset,
            token_region=token_region,
            metric="cosine",
        )
        if list(cosine_unit_ids) != list(unit_ids):
            print(
                "[warn] Skip Cosine strategy: unit_ids do not match Ours source.",
                flush=True,
            )
        else:
            cosine_interval_score = build_interval_score_matrix(cosine_sim)
            cosine_scan, _cosine_candidates = scan_endpoint_linked_closed_intervals_with_candidates(
                cosine_interval_score,
                cosine_unit_ids,
                ranking_tolerance=CORNER_RANKING_TOLERANCE,
                require_endpoint_linked_spans=REQUIRE_ENDPOINT_LINKED_SPANS,
            )
            strategies["cosine"] = _strategy_from_picks(
                key="cosine",
                display_name="Cosine",
                picks=cosine_scan,
                unit_ids=unit_ids,
                num_units=num_units,
                param_accounting=param_accounting,
                score=cosine_interval_score,
                score_name="corner_similarity",
                source={
                    "kind": "similarity",
                    "metric": "cosine",
                    "token_region": str(token_region),
                    "dataset": str(dataset),
                    "similarity_source": cosine_similarity_source,
                    "interval_score": "C[start,end]=sim[start-1,end] for start>0; C[0,end]=0 placeholder",
                    "triangle_selection": (
                        "endpoint_linked_descending_best_interval_score"
                        if bool(REQUIRE_ENDPOINT_LINKED_SPANS)
                        else "independent_best_interval_score_per_drop_count"
                    ),
                },
                intervals=interval_records,
            )
    except Exception as exc:
        print(f"[warn] Could not build Cosine strategy: {exc}", flush=True)
    if gt_corner_score is not None and gt_corner_picks:
        strategies["gt"] = _strategy_from_picks(
            key="gt",
            display_name="GT",
            picks=gt_corner_picks,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score=gt_corner_score,
            score_name="gt_aggregate_score",
            source={
                "kind": "skip_eval_gt",
                "report": str(gt_report_path),
                "metric_selector": str(gt_corner_metric),
                "meta": gt_corner_meta,
            },
            intervals=interval_records,
        )

    if gt_corner_score is not None and gt_corner_picks:
        gt_score_trajectories.append(("GT best path", gt_corner_picks))
        if str(token_region) not in set(CKA_CORNER_COMPARISON_TOKEN_REGIONS):
            gt_score_trajectories.append((f"CKA selected path ({token_region})", scan))

    token_corner_items: list[
        tuple[str, np.ndarray, Sequence[TrianglePick], dict[int, Sequence[TrianglePick]]]
    ] = []
    for comparison_region in CKA_CORNER_COMPARISON_TOKEN_REGIONS:
        try:
            if str(comparison_region) == str(token_region):
                region_unit_ids = list(unit_ids)
                region_sim = interval_score
                region_scan = scan
                region_candidates_by_span = eligible_candidates_by_span
            else:
                region_unit_ids, region_sim, _region_source = load_similarity(
                    experiment=experiment,
                    dataset=dataset,
                    token_region=comparison_region,
                    metric=metric,
                )
                if list(region_unit_ids) != list(unit_ids):
                    print(
                        f"[warn] Skip token-region corner panel for {comparison_region}: "
                        "unit_ids do not match the selected plan source.",
                        flush=True,
                    )
                    continue
                region_sim = build_interval_score_matrix(region_sim)
                region_scan, region_candidates_by_span = (
                    scan_endpoint_linked_closed_intervals_with_candidates(
                        region_sim,
                        region_unit_ids,
                        ranking_tolerance=CORNER_RANKING_TOLERANCE,
                        require_endpoint_linked_spans=REQUIRE_ENDPOINT_LINKED_SPANS,
                    )
                )
            token_corner_items.append(
                (
                    str(comparison_region),
                    np.asarray(region_sim, dtype=np.float64),
                    list(region_scan),
                    dict(region_candidates_by_span),
                )
            )
            if gt_corner_score is not None and gt_corner_picks:
                gt_score_trajectories.append(
                    (f"CKA selected path ({comparison_region})", region_scan)
                )
        except Exception as exc:
            print(
                f"[warn] Could not prepare token-region corner panel for "
                f"{comparison_region}: {exc}",
                flush=True,
            )
    cka_score_trajectories = [
        (name, picks)
        for name, picks in gt_score_trajectories
        if not str(name).lower().startswith("gt")
    ]
    if token_corner_items:
        corner_overview_path = render_corner_overview_heatmap(
            token_corner_items,
            out_dir=visualization_root,
            backbone=backbone,
            selected_token_region=str(token_region),
            gt_score=gt_corner_score,
            gt_picks=gt_corner_picks,
            metric_names=(
                list(gt_corner_meta.get("metrics", []))
                if isinstance(gt_corner_meta, dict)
                else []
            ),
            cka_trajectories=cka_score_trajectories,
            relative_degradation_cap_multiple=GT_RELATIVE_DEGRADATION_CAP_MULTIPLE,
            component_aggregation=GT_COMPONENT_AGGREGATION,
            selection_label=selection_label,
            tag=f"prune_plan_{safe_backbone}_ours_corner_overview",
        )
        print(f"Wrote corner overview heatmap {corner_overview_path}", flush=True)
    if gt_corner_score is not None and gt_score_trajectories:
        if gt_corner_picks:
            component_heatmap_path = render_gt_component_heatmaps(
                gt_component_heatmaps,
                out_dir=visualization_root,
                backbone=backbone,
                relative_degradation_cap_multiple=GT_RELATIVE_DEGRADATION_CAP_MULTIPLE,
            )
            if component_heatmap_path is not None:
                print(f"Wrote GT component heatmaps {component_heatmap_path}", flush=True)
        score_plot_path = render_gt_trajectory_score_plot(
            gt_corner_score,
            gt_score_trajectories,
            out_dir=visualization_root,
            backbone=backbone,
            relative_degradation_cap_multiple=GT_RELATIVE_DEGRADATION_CAP_MULTIPLE,
            tag=f"prune_plan_{safe_backbone}_ours_gt_trajectory_scores",
        )
        if score_plot_path is not None:
            print(f"Wrote GT trajectory score plot {score_plot_path}", flush=True)

    for strategy_key, strategy_record in strategies.items():
        points = strategy_record.get("points", {})
        if not isinstance(points, dict) or not points:
            continue
        span_to_drop_indices: dict[int, list[int]] = {}
        for span_key, interval_key in points.items():
            if not str(span_key).isdigit():
                continue
            interval_record = interval_records.get(str(interval_key))
            if not isinstance(interval_record, dict):
                continue
            span_to_drop_indices[int(span_key)] = [
                int(x) for x in list(interval_record.get("drop", []))
            ]
        if not span_to_drop_indices:
            continue
        display_name = str(strategy_record.get("display_name", strategy_key))
        prune_grid_path = render_prune_span_drop_grid(
            num_units=num_units,
            span_to_drop_indices=span_to_drop_indices,
            out_dir=visualization_root,
            backbone=backbone,
            tag=f"prune_plan_{safe_backbone}_{strategy_key}_drop_grid",
            title=f"{backbone}: {display_name} prune drop grid",
        )
        print(f"Wrote {display_name} prune drop grid {prune_grid_path}", flush=True)

    ours_points = strategies.get("ours", {}).get("points", {})
    span_records: dict[str, dict[str, Any]] = {}
    if isinstance(ours_points, dict):
        for span_key, interval_key in ours_points.items():
            if not str(span_key).isdigit():
                continue
            record = interval_records.get(str(interval_key))
            if isinstance(record, dict):
                span_records[str(int(span_key))] = dict(record)

    bundle = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": str(backbone),
        "dataset": str(dataset),
        "num_units": int(num_units),
        "param_accounting": param_accounting,
        "source_cache": "",
        "intervals": {
            key: interval_records[key]
            for key in sorted(
                interval_records.keys(),
                key=lambda k: (
                    int(interval_records[k].get("drop_count", 0)),
                    int(interval_records[k].get("i", 0)),
                    int(interval_records[k].get("j", 0)),
                    k,
                ),
            )
        },
        "strategies": strategies,
        "spans": {
            str(span): span_records[str(span)]
            for span in sorted((int(k) for k in span_records.keys()))
        },
        "extra": {
            "similarity_source": similarity_source,
            "metric": str(metric),
            "token_region": str(token_region),
            "triangle_legality": "all_diagonal_candidates",
            "interval_semantics": "closed interval iS_jE drops units S through E",
            "interval_score": {
                "matrix": "C[start,end]",
                "start_positive": "C[start,end]=raw_similarity[start-1,end]",
                "start_zero": "C[0,end]=0.0",
                "start_zero_placeholder": True,
            },
            "corner_ranking_tolerance": float(CORNER_RANKING_TOLERANCE),
            "require_endpoint_linked_spans": bool(REQUIRE_ENDPOINT_LINKED_SPANS),
            "triangle_selection": (
                "endpoint_linked_descending_best_interval_score"
                if bool(REQUIRE_ENDPOINT_LINKED_SPANS)
                else "independent_best_interval_score_per_drop_count"
            ),
            "gt_corner_path": gt_corner_meta,
            "max_available_span": int(max_available_span),
            "available_spans": len(scan),
        },
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(
        f"Wrote {bundle_path} (intervals={len(interval_records)}, strategies={len(strategies)}, "
        f"ours_spans={len(span_records)}, max_available_span={max_available_span})",
        flush=True,
    )

    curve_path = render_span_similarity_curve(
        scan,
        interval_score,
        {},
        out_dir=visualization_root,
        backbone=backbone,
        threshold=None,
        selection_label=f"{selection_label.title()} triangle",
        tag=f"prune_plan_{safe_backbone}_ours_curve",
    )
    print(
        f"Wrote prune-span curve {curve_path} "
        f"(available spans={len(scan)}, max available span={max_available_span})",
        flush=True,
    )
    return [bundle_path]


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_build_prune_plan(
        experiment=args.experiment,
        dataset=args.dataset,
        backbone=args.backbone,
        metric=args.metric,
        token_region=args.token_region,
        prune_plan_root=args.prune_plan_root,
        output_path=args.output,
        viz_out_dir=args.viz_out_dir,
        skip_param_accounting=bool(args.skip_param_accounting),
        gt_corner_skip_eval_report=args.gt_corner_skip_eval_report,
        gt_corner_metric=str(args.gt_corner_metric),
    )


if __name__ == "__main__":
    main()

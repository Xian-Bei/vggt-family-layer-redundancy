"""Build total-drop-indexed interval and set pruning-plan bundles."""

from __future__ import annotations

import argparse
import json
import math
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    interval_bounds_from_selector,
    interval_selector,
    physical_pruning_selector_from_drop,
)
from analysis.prune_plan.build_prune_plan_common import (
    default_plan_root as _default_plan_root,
    load_similarity,
    parse_dataset_list,
)
from analysis.redundancy.similarity_v3 import (
    REDUNDANCY_GRAPH_REGIONS,
    unit_similarity_to_interval_score,
)
from analysis.redundancy.skip_metric_scaling import (
    normalize_metric_components,
)
from analysis.prune_plan.triangle_prune import TrianglePick
from analysis.prune_plan.visualize_triangle_prune import (
    render_prune_strategy_drop_grids,
)

# ---------------------------------------------------------------------------
# GT / Ours controls
# Edit this block to tune GT and all Ours variants (CKA, cosine, inverse-MSE, CCA).
# ---------------------------------------------------------------------------

# GT score construction and metric fusion.
GT_COMPONENT_AGGREGATION = "mean"  # "mean", "min", or "max"; lower is selected.
GT_POSE_METRICS = ("ape", "are", "rpe_trans", "rpe_rot")
GT_POINT_METRICS = ("acc", "comp", "chamfer_distance")
GT_AGGREGATE_METRICS = (*GT_POSE_METRICS, *GT_POINT_METRICS)

GT_OURS_COMPARISON_TOLERANCE = GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE = GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE = 5e-3
MIN_RETAINED_UNITS_BETWEEN_INTERVALS = 1
# For Double-to-Double transitions, continuity is one-to-one: each current
# interval overlaps a previous interval and each previous interval survives.
GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP = True
# Independently limit the number of Single-to-Double transitions.
GT_OURS_DOUBLE_ONE_SPLIT_ONLY = False

# Joint scores for adaptive Double variants. GT keeps the additive degradation
# proxy, while Ours is governed by its worse original-model component.
GT_TWO_INTERVAL_SCORE_AGGREGATION = "sum"  # "sum" or "max"
OURS_TWO_INTERVAL_SCORE_AGGREGATION = "sum"  # "sum" or "max"
SUPPORTED_PRUNING_STRATEGIES = (
    "ours_single",
    "ours_double",
    "cosine_single",
    "cosine_double",
    "gt_single",
    "gt_double",
    "replaceme_single",
    "replaceme_double",
    "inv_mse_single",
    "inv_mse_double",
    "cca_single",
    "cca_double",
    "shortgpt",
    "simpler",
)
PRUNING_STRATEGY_FAMILY_ALIASES = {
    "ours": ("ours_single", "ours_double"),
    "cosine": ("cosine_single", "cosine_double"),
    "inv_mse": ("inv_mse_single", "inv_mse_double"),
    "cca": ("cca_single", "cca_double"),
    "gt": ("gt_single", "gt_double"),
    "replaceme": ("replaceme_single", "replaceme_double"),
}


def _normalize_max_drop_budget(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        budget = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            "max_drop_budget must be a positive integer or None"
        ) from exc
    if isinstance(value, bool) or budget <= 0:
        raise ValueError("max_drop_budget must be a positive integer or None")
    return int(budget)


def _parse_pruning_strategies(raw: Optional[str]) -> list[str]:
    values = [
        value.strip().lower()
        for value in str(raw or "").replace(",", " ").split()
        if value.strip()
    ]
    if not values:
        values = list(SUPPORTED_PRUNING_STRATEGIES)
    selected = list(
        dict.fromkeys(
            expanded
            for value in values
            for expanded in PRUNING_STRATEGY_FAMILY_ALIASES.get(value, (value,))
        )
    )
    unknown = sorted(set(selected) - set(SUPPORTED_PRUNING_STRATEGIES))
    if unknown:
        raise ValueError(
            f"unsupported pruning strategies {unknown}; "
            f"expected a subset of {list(SUPPORTED_PRUNING_STRATEGIES)}"
        )
    return selected


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
        "--strategies",
        type=str,
        default=",".join(SUPPORTED_PRUNING_STRATEGIES),
        help="Comma/space separated pruning strategies to keep in the bundle and drop grid.",
    )
    p.add_argument(
        "--max-drop-budget",
        type=int,
        default=None,
        help=(
            "Highest total-drop budget generated for every strategy "
            "(default: no cap)."
        ),
    )
    p.add_argument(
        "--token-region",
        type=str,
        choices=(*REDUNDANCY_GRAPH_REGIONS, "mean", "min"),
        default="mean",
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
            "Optional eval/pruning directory or legacy skip_eval_report_v3.json used to "
            "build the multi-metric GT aggregate corner heatmap. This override is only "
            "valid with one dataset; otherwise every --dataset entry is resolved below "
            "prune-plan-root and all reports are required."
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


def _normalize_and_aggregate_gt_components(
    components: dict[str, np.ndarray],
    *,
    metric_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Normalize GT metrics, then apply the aggregation selected above."""
    normalized, mean_aggregate, meta = normalize_metric_components(
        components,
        metric_names=metric_names,
    )
    mode = str(GT_COMPONENT_AGGREGATION).strip().lower()
    if mode == "mean":
        aggregate = mean_aggregate
    elif mode in {"min", "max"}:
        stack = np.stack(
            [normalized[name] for name in metric_names if name in normalized],
            axis=0,
        )
        masked = np.ma.masked_invalid(stack)
        reduced = masked.min(axis=0) if mode == "min" else masked.max(axis=0)
        aggregate = np.asarray(reduced.filled(np.nan), dtype=np.float64)
    else:
        raise ValueError(
            "GT_COMPONENT_AGGREGATION must be 'mean', 'min', or 'max', got "
            f"{GT_COMPONENT_AGGREGATION!r}"
        )
    return normalized, aggregate, {
        **meta,
        "component_aggregation": mode,
    }


def _raw_metric_to_upper_clamped_score(
    values: np.ndarray,
    *,
    upper_baseline: Optional[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp better-than-upper values without ratio scaling.

    For a positive upper baseline, min-max normalization of
    ``max(current, upper)`` is exactly equivalent to normalizing
    ``max(current / upper, 1)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    meta: dict[str, Any] = {
        "scale_mode": "raw_metric_upper_clamped",
        "upper_baseline": upper_baseline,
        "enabled": False,
    }
    if not np.any(finite) or upper_baseline is None:
        meta["skip_reason"] = "no_finite_values_or_missing_upper_baseline"
        return out, meta
    upper = float(upper_baseline)
    if not math.isfinite(upper) or upper <= 0.0:
        meta["skip_reason"] = "invalid_upper_baseline"
        return out, meta
    out[finite] = np.maximum(arr[finite], upper)
    meta["enabled"] = True
    return out, meta


def _degradation_pct_to_upper_clamped_score(
    values: np.ndarray,
    *,
    upper_baseline: Optional[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp signed degradation percentages at zero before min-max scaling."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    meta: dict[str, Any] = {
        "scale_mode": "degradation_pct_zero_clamped",
        "upper_baseline": upper_baseline,
        "enabled": False,
    }
    if not np.any(finite) or upper_baseline is None:
        meta["skip_reason"] = "no_finite_values_or_missing_upper_baseline"
        return out, meta
    upper = float(upper_baseline)
    if not math.isfinite(upper) or upper <= 0.0:
        meta["skip_reason"] = "invalid_upper_baseline"
        return out, meta
    out[finite] = np.maximum(arr[finite], 0.0)
    meta["enabled"] = True
    return out, meta


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
    """Derive closed-interval pruning scores from unit-output similarity."""
    return unit_similarity_to_interval_score(sim)


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


_PRUNING_METRIC_GRID = (
    ("ape", "are", "rpe_trans", "rpe_rot"),
    ("acc", "comp", "chamfer_distance", None),
)
_PRUNING_METRIC_HEADERS = {
    "ape": "APE ↓",
    "are": "ARE (deg) ↓",
    "rpe_trans": "RPE trans ↓",
    "rpe_rot": "RPE rot (deg) ↓",
    "acc": "Acc ↓",
    "comp": "Comp ↓",
    "chamfer_distance": "CD ↓",
    "cd": "CD ↓",
}


def _metric_aliases(metric: str) -> tuple[str, ...]:
    key = str(metric)
    if key == "chamfer_distance":
        return ("chamfer_distance", "cd")
    if key == "cd":
        return ("cd", "chamfer_distance")
    return (key,)


def _dict_metric_value(values: dict[str, Any], metric: str) -> Optional[float]:
    for key in _metric_aliases(metric):
        value = values.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _load_upper_average_for_pruning_dir(pruning_dir: Path) -> dict[str, float]:
    eval_root = Path(pruning_dir).parent
    candidates = [
        eval_root / "upper" / "eval_record_v3.json",
        eval_root.parent / "eval" / "upper" / "eval_record_v3.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        average = raw.get("average", {})
        if isinstance(average, dict):
            return {
                str(k): float(v)
                for k, v in average.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            }
    return {}


def _load_gt_aggregate_from_pruning_dir(
    pruning_dir: Path,
    *,
    unit_ids: Sequence[str],
    metric_selector: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[TrianglePick], dict[str, Any]]:
    """Build a GT heatmap from eval/pruning/iS_jE/summary.json files."""
    pruning_dir = Path(pruning_dir)
    n = len(unit_ids)
    summaries: dict[str, dict[str, Any]] = {}
    available_metrics: set[str] = set()
    for summary_path in sorted(pruning_dir.glob("i*_j*/summary.json")):
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
            selector = str(raw.get("selector") or summary_path.parent.name)
            start, end = interval_bounds_from_selector(selector)
        except Exception:
            continue
        if start <= 0 or end < start or end >= n:
            continue
        metrics_mean = raw.get("metrics_mean", {})
        if not isinstance(metrics_mean, dict):
            continue
        available_metrics.update(str(k) for k in metrics_mean.keys())
        summaries[selector] = raw
    if not summaries:
        raise FileNotFoundError(f"No eval/pruning interval summaries found under {pruning_dir}")

    upper_average = _load_upper_average_for_pruning_dir(pruning_dir)
    selected_metric_names = _parse_metric_selector(
        str(metric_selector), sorted(str(k) for k in available_metrics)
    )
    selected_set = set(selected_metric_names)
    metric_names = [
        m
        for m in GT_AGGREGATE_METRICS
        if m in selected_set or any(alias in selected_set for alias in _metric_aliases(m))
    ]
    if not metric_names:
        raise ValueError(
            "No supported non-depth GT metric selected. "
            f"Supported metrics={list(GT_AGGREGATE_METRICS)}, selected={selected_metric_names}"
        )

    metric_records: list[dict[str, Any]] = []
    component_heatmaps: dict[str, np.ndarray] = {}
    component_metric_names: list[str] = []
    selector_scores: dict[str, float] = {}
    for metric_key in metric_names:
        raw_values = np.full((n, n), np.nan, dtype=np.float64)
        for selector, summary in summaries.items():
            metrics_mean = summary.get("metrics_mean", {})
            if not isinstance(metrics_mean, dict):
                continue
            value = _dict_metric_value(metrics_mean, metric_key)
            if value is None:
                continue
            start, end = interval_bounds_from_selector(selector)
            raw_values[start, end] = value
        upper_value = _dict_metric_value(upper_average, metric_key)
        clamped_score, scaling_meta = _raw_metric_to_upper_clamped_score(
            raw_values,
            upper_baseline=upper_value,
        )
        component_heatmaps[str(metric_key)] = clamped_score
        component_metric_names.append(str(metric_key))

        finite_raw = np.isfinite(raw_values)
        finite_score = np.isfinite(clamped_score)
        if np.any(finite_raw):
            metric_records.append(
                {
                    "metric": str(metric_key),
                    "raw_min": float(np.nanmin(raw_values[finite_raw])),
                    "raw_max": float(np.nanmax(raw_values[finite_raw])),
                    "clamped_score_min": (
                        float(np.nanmin(clamped_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "clamped_score_max": (
                        float(np.nanmax(clamped_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "upper_baseline": upper_value,
                    "scaling": scaling_meta,
                    "num_finite": int(np.count_nonzero(finite_raw)),
                }
            )

    component_heatmaps, aggregate, normalization_meta = (
        _normalize_and_aggregate_gt_components(
            component_heatmaps,
            metric_names=component_metric_names,
        )
    )
    for selector in summaries.keys():
        start, end = interval_bounds_from_selector(selector)
        value = float(aggregate[start, end])
        if math.isfinite(value):
            selector_scores[selector] = value

    best_by_drop_count: dict[int, tuple[float, int, int]] = {}
    for start in range(1, n):
        for end in range(start, n):
            val = float(aggregate[start, end])
            if not math.isfinite(val):
                continue
            drop_count = int(end - start + 1)
            current = best_by_drop_count.get(drop_count)
            if (
                current is None
                or val < current[0] - GT_OURS_COMPARISON_TOLERANCE
            ):
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
                "aggregate_score": float(value),
            }
        )

    meta = {
        "source": str(pruning_dir),
        "source_kind": "eval_pruning_interval_summaries",
        "metric_selector": str(metric_selector),
        "metrics": [str(x) for x in component_metric_names],
        "component_aggregation": str(GT_COMPONENT_AGGREGATION),
        "component_normalization": normalization_meta,
        "metric_normalization": (
            "Raw eval/pruning interval metrics are clamped at the upper baseline: "
            "lower-is-better values that outperform upper are replaced by upper. "
            "This removes ratio scaling while preserving the exact subsequent "
            "min-max result. Each metric heatmap is then independently "
            "min-max normalized to [0,1] before the seven metrics are averaged."
        ),
        "metric_records": metric_records,
        "upper_average": upper_average,
        "span_mapping": "eval/pruning/iS_jE summary -> prune closed interval iS_jE",
        "records": records,
        "selector_scores": selector_scores,
    }
    return aggregate, component_heatmaps, picks, meta


def _load_pruning_summary_records(pruning_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(Path(pruning_dir).glob("*/summary.json")):
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        selector = str(raw.get("selector") or summary_path.parent.name).strip()
        if not selector:
            continue
        metrics_mean = raw.get("metrics_mean", {})
        if not isinstance(metrics_mean, dict):
            continue
        records[selector] = {
            "selector": selector,
            "summary_path": str(summary_path),
            "metrics_mean": {
                str(k): float(v)
                for k, v in metrics_mean.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            },
            "num_scenes": (
                float(raw["num_scenes"])
                if isinstance(raw.get("num_scenes"), (int, float))
                and math.isfinite(float(raw.get("num_scenes")))
                else None
            ),
            "drop_count": (
                int(raw["drop_count"])
                if isinstance(raw.get("drop_count"), (int, float))
                else None
            ),
            "start": int(raw["start"]) if isinstance(raw.get("start"), (int, float)) else None,
            "end": int(raw["end"]) if isinstance(raw.get("end"), (int, float)) else None,
        }
    return records


def _metrics_relative_to_upper(
    metrics_mean: dict[str, Any],
    upper_average: dict[str, Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric_key, value in metrics_mean.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        upper = _dict_metric_value(upper_average, str(metric_key))
        if upper is None or upper <= 0.0:
            continue
        ratio = float(value) / upper
        if math.isfinite(ratio):
            out[str(metric_key)] = float(max(1.0, ratio))
    return out


def _strategy_points_sorted(strategy_record: dict[str, Any]) -> list[tuple[int, str]]:
    points = strategy_record.get("points", {})
    if not isinstance(points, dict):
        return []
    out: list[tuple[int, str]] = []
    for drop_key, selector in sorted(
        points.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if str(drop_key).isdigit():
            out.append((int(drop_key), str(selector)))
    return out


def _collect_strategy_pruning_eval(
    *,
    plan_root: Path,
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    plan_root = Path(plan_root)
    datasets: dict[str, Any] = {}
    for pruning_dir in sorted(plan_root.glob("*/eval/pruning")):
        if not pruning_dir.is_dir():
            continue
        summaries = _load_pruning_summary_records(pruning_dir)
        if not summaries:
            continue
        dataset = pruning_dir.parent.parent.name
        upper_average = _load_upper_average_for_pruning_dir(pruning_dir)
        strategy_payload: dict[str, Any] = {}
        for strategy_key, strategy_record in strategies.items():
            if not isinstance(strategy_record, dict):
                continue
            display_name = str(strategy_record.get("display_name", strategy_key))
            points_payload: dict[str, Any] = {}
            for drop_count, selector in _strategy_points_sorted(strategy_record):
                summary = summaries.get(selector)
                if summary is None:
                    points_payload[str(drop_count)] = {
                        "available": False,
                        "selector": selector,
                        "drop_count": int(drop_count),
                        "skip_reason": "missing_exact_pruning_summary",
                    }
                    continue
                metrics_mean = dict(summary.get("metrics_mean", {}))
                points_payload[str(drop_count)] = {
                    "available": True,
                    "selector": selector,
                    "drop_count": int(drop_count),
                    "summary_path": str(summary.get("summary_path", "")),
                    "num_scenes": summary.get("num_scenes"),
                    "metrics_mean": metrics_mean,
                    "relative_to_upper": _metrics_relative_to_upper(
                        metrics_mean,
                        upper_average,
                    ),
                }
            strategy_payload[strategy_key] = {
                "display_name": display_name,
                "points": points_payload,
            }
        datasets[dataset] = {
            "pruning_dir": str(pruning_dir),
            "upper_average": upper_average,
            "strategies": strategy_payload,
        }

    metric_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    relative_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    upper_values: dict[str, list[float]] = {}
    for dataset_payload in datasets.values():
        upper_average = dataset_payload.get("upper_average", {})
        if isinstance(upper_average, dict):
            for metric_key, value in upper_average.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    upper_values.setdefault(str(metric_key), []).append(float(value))
        strategies_payload = dataset_payload.get("strategies", {})
        if not isinstance(strategies_payload, dict):
            continue
        for strategy_key, strategy_payload in strategies_payload.items():
            points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
            if not isinstance(points, dict):
                continue
            for drop_key, point in points.items():
                if not isinstance(point, dict) or not bool(point.get("available", False)):
                    continue
                metrics_mean = point.get("metrics_mean", {})
                if isinstance(metrics_mean, dict):
                    for metric_key, value in metrics_mean.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            metric_values.setdefault(str(strategy_key), {}).setdefault(
                                str(drop_key), {}
                            ).setdefault(str(metric_key), []).append(float(value))
                relative = point.get("relative_to_upper", {})
                if isinstance(relative, dict):
                    for metric_key, value in relative.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            relative_values.setdefault(str(strategy_key), {}).setdefault(
                                str(drop_key), {}
                            ).setdefault(str(metric_key), []).append(float(value))

    dataset_average_strategies: dict[str, Any] = {}
    for strategy_key, strategy_record in strategies.items():
        display_name = str(strategy_record.get("display_name", strategy_key))
        points_payload: dict[str, Any] = {}
        for drop_count, selector in _strategy_points_sorted(strategy_record):
            metrics_for_point = metric_values.get(strategy_key, {}).get(str(drop_count), {})
            rel_for_point = relative_values.get(strategy_key, {}).get(str(drop_count), {})
            if not metrics_for_point:
                points_payload[str(drop_count)] = {
                    "available": False,
                    "selector": selector,
                    "drop_count": int(drop_count),
                    "skip_reason": "no_dataset_has_exact_pruning_summary",
                }
                continue
            points_payload[str(drop_count)] = {
                "available": True,
                "selector": selector,
                "drop_count": int(drop_count),
                "num_datasets": int(max(len(v) for v in metrics_for_point.values())),
                "metrics_mean": {
                    metric_key: float(np.mean(values))
                    for metric_key, values in sorted(metrics_for_point.items())
                    if values
                },
                "relative_to_upper": {
                    metric_key: float(np.mean(values))
                    for metric_key, values in sorted(rel_for_point.items())
                    if values
                },
            }
        dataset_average_strategies[strategy_key] = {
            "display_name": display_name,
            "points": points_payload,
        }
    return {
        "schema_version": "strategy_pruning_eval_v1",
        "source": "dataset/eval/pruning/<selector>/summary.json",
        "datasets": datasets,
        "dataset_average": {
            "upper_average": {
                metric_key: float(np.mean(values))
                for metric_key, values in sorted(upper_values.items())
                if values
            },
            "strategies": dataset_average_strategies,
            "num_datasets": len(datasets),
        },
    }


def _metric_from_mapping(values: dict[str, Any], metric: str) -> Optional[float]:
    value = _dict_metric_value(values, metric)
    return value if value is not None and math.isfinite(float(value)) else None


def _write_strategy_pruning_metric_plot(
    *,
    payload: dict[str, Any],
    out_path: Path,
    title: str,
) -> Optional[Path]:
    strategies = payload.get("strategies", {})
    if not isinstance(strategies, dict) or not strategies:
        return None
    upper_average = payload.get("upper_average", {})
    if not isinstance(upper_average, dict):
        upper_average = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for pruning metric plot: {exc}", flush=True)
        return None

    rows = len(_PRUNING_METRIC_GRID)
    cols = len(_PRUNING_METRIC_GRID[0])
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.8 * rows), squeeze=False)
    any_data = False
    for row_idx, row in enumerate(_PRUNING_METRIC_GRID):
        for col_idx, metric_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            upper_value = _metric_from_mapping(upper_average, metric_key)
            has_data = upper_value is not None
            if upper_value is not None:
                ax.axhline(
                    float(upper_value),
                    linestyle="--",
                    color="#333333",
                    linewidth=1.2,
                    alpha=0.7,
                    label="upper",
                )
                ax.scatter([0], [float(upper_value)], color="#333333", marker="*", s=65, zorder=4)
            all_xs: list[int] = []
            for strategy_key, strategy_payload in strategies.items():
                if not isinstance(strategy_payload, dict):
                    continue
                display_name = str(strategy_payload.get("display_name", strategy_key))
                points = strategy_payload.get("points", {})
                if not isinstance(points, dict):
                    continue
                xs: list[int] = []
                ys: list[float] = []
                if upper_value is not None:
                    xs.append(0)
                    ys.append(float(upper_value))
                for drop_key, point in sorted(
                    points.items(),
                    key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
                ):
                    if not str(drop_key).isdigit() or not isinstance(point, dict):
                        continue
                    if not bool(point.get("available", False)):
                        continue
                    metrics_mean = point.get("metrics_mean", {})
                    if not isinstance(metrics_mean, dict):
                        continue
                    value = _metric_from_mapping(metrics_mean, metric_key)
                    if value is None:
                        continue
                    xs.append(int(drop_key))
                    ys.append(float(value))
                    all_xs.append(int(drop_key))
                if len(xs) <= (1 if upper_value is not None else 0):
                    continue
                has_data = True
                any_data = True
                ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=display_name)
            if not has_data:
                ax.axis("off")
                continue
            ax.set_title(_PRUNING_METRIC_HEADERS.get(metric_key, metric_key), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            if all_xs:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.legend(loc="best", fontsize=8)
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_pruning_eval_plots(
    *,
    plan_root: Path,
    backbone: str,
    pruning_eval: dict[str, Any],
) -> list[Path]:
    written: list[Path] = []
    datasets = pruning_eval.get("datasets", {})
    if isinstance(datasets, dict):
        for dataset, payload in sorted(datasets.items()):
            if not isinstance(payload, dict):
                continue
            pruning_dir_raw = str(payload.get("pruning_dir", "")).strip()
            if not pruning_dir_raw:
                continue
            pruning_dir = Path(pruning_dir_raw)
            safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
            out_path = pruning_dir.parent / f"{safe_ds}_drop.png"
            written_path = _write_strategy_pruning_metric_plot(
                payload=payload,
                out_path=out_path,
                title=f"{backbone} / {dataset}: pruning strategies vs upper",
            )
            if written_path is not None:
                written.append(written_path)
    average = pruning_eval.get("dataset_average", {})
    if isinstance(average, dict):
        safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
        out_path = Path(plan_root) / f"prune_plan_{safe_backbone}_drop.png"
        written_path = _write_strategy_pruning_metric_plot(
            payload=average,
            out_path=out_path,
            title=f"{backbone}: pruning strategies averaged over datasets",
        )
        if written_path is not None:
            written.append(written_path)
    return written


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
    metric_names = [m for m in GT_AGGREGATE_METRICS if m in selected_set]
    if not metric_names:
        raise ValueError(
            "No supported non-depth GT metric selected. "
            f"Supported metrics={list(GT_AGGREGATE_METRICS)}, selected={selected_metric_names}"
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
        for a in range(1, n):
            b_max = n - 1 if not has_cap else min(n - 1, a + max_span - 1)
            for b in range(a, b_max + 1):
                val = float(skip_values[a, b])
                if not math.isfinite(val):
                    continue
                corner_values[a, b] = val
        clamped_score, scaling_meta = _degradation_pct_to_upper_clamped_score(
            corner_values,
            upper_baseline=(
                float(upper_average[str(metric_key)])
                if str(metric_key) in upper_average
                and isinstance(upper_average[str(metric_key)], (int, float))
                else None
            ),
        )
        component_heatmaps[str(metric_key)] = clamped_score
        component_metric_names.append(str(metric_key))

        finite_raw = np.isfinite(corner_values)
        finite_score = np.isfinite(clamped_score)
        if np.any(finite_raw):
            metric_records.append(
                {
                    "metric": str(metric_key),
                    "degrade_pct_min": float(np.nanmin(corner_values[finite_raw])),
                    "degrade_pct_max": float(np.nanmax(corner_values[finite_raw])),
                    "clamped_score_min": (
                        float(np.nanmin(clamped_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "clamped_score_max": (
                        float(np.nanmax(clamped_score[finite_score]))
                        if np.any(finite_score)
                        else None
                    ),
                    "scaling": scaling_meta,
                    "num_finite": int(np.count_nonzero(finite_raw)),
                }
            )

    component_heatmaps, aggregate, normalization_meta = (
        _normalize_and_aggregate_gt_components(
            component_heatmaps,
            metric_names=component_metric_names,
        )
    )

    best_by_drop_count: dict[int, tuple[float, int, int]] = {}
    for start in range(1, n):
        for end in range(start, n):
            val = float(aggregate[start, end])
            if not math.isfinite(val):
                continue
            drop_count = int(end - start + 1)
            current = best_by_drop_count.get(drop_count)
            if (
                current is None
                or val < current[0] - GT_OURS_COMPARISON_TOLERANCE
            ):
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
        "component_normalization": normalization_meta,
        "metric_normalization": (
            "Only pose metrics (ape, are, rpe_trans, rpe_rot) and point-cloud "
            "metrics (acc, comp, chamfer_distance) are used. NC is excluded. "
            "Saved signed degradation percentages are clamped at zero, which is "
            "min-max equivalent to the previous upper-relative multiple without "
            "performing ratio scaling. Each metric heatmap is "
            "independently min-max normalized to [0,1] "
            + f"before components are aggregated with {GT_COMPONENT_AGGREGATION!r}."
        ),
        "metric_records": metric_records,
        "fusion_mode": mode,
        "span_mapping": "skip_eval closed [a,b] -> prune closed interval i{a}_j{b}",
        "records": records,
    }
    return aggregate, component_heatmaps, picks, meta


def _default_gt_corner_report_path(plan_root: Path, dataset: str) -> Path:
    pruning_dir = Path(plan_root) / str(dataset) / "eval" / "pruning"
    if any(pruning_dir.glob("i*_j*/summary.json")):
        return pruning_dir
    new_path = pruning_dir / "skip_eval_report_v3.json"
    if new_path.is_file():
        return new_path
    return Path(plan_root) / str(dataset) / "skip_eval" / "skip_eval_report_v3.json"


def _gt_picks_from_score(score: np.ndarray, unit_ids: Sequence[str]) -> list[TrianglePick]:
    """Select the lowest-scoring interval for every available drop count."""
    picks: list[TrianglePick] = []
    n = len(unit_ids)
    for drop_count in range(1, n):
        best: Optional[tuple[float, int, int]] = None
        for start in range(1, n - drop_count + 1):
            end = start + drop_count - 1
            value = float(score[start, end])
            if not math.isfinite(value):
                continue
            if (
                best is None
                or value < best[0] - GT_OURS_COMPARISON_TOLERANCE
            ):
                best = (value, start, end)
        if best is None:
            continue
        _value, start, end = best
        picks.append(
            TrianglePick(
                i=start,
                j=end,
                span=drop_count,
                unit_i=str(unit_ids[start]),
                unit_j=str(unit_ids[end]),
                dropped_unit_ids=tuple(str(unit_ids[k]) for k in range(start, end + 1)),
            )
        )
    return picks


def _load_gt_aggregate_for_datasets(
    *,
    plan_root: Path,
    datasets: Sequence[str],
    unit_ids: Sequence[str],
    metric_selector: str,
    explicit_report: Optional[Path] = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[TrianglePick],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    """Strictly load and mean GT heatmaps for exactly the requested datasets."""
    dataset_names = [str(name) for name in datasets]
    if explicit_report is not None and len(dataset_names) != 1:
        raise ValueError(
            "--gt-corner-skip-eval-report can only override a single PRUNE_DATASET; "
            "multiple datasets resolve their GT reports from each dataset directory"
        )
    loaded: list[tuple[str, Path, np.ndarray, dict[str, np.ndarray], dict[str, Any]]] = []
    for dataset_name in dataset_names:
        report_path = (
            Path(explicit_report)
            if explicit_report is not None
            else _default_gt_corner_report_path(plan_root, dataset_name)
        )
        if not report_path.exists():
            raise FileNotFoundError(
                f"Missing GT report for PRUNE_DATASET={dataset_name!r}: {report_path}"
            )
        if report_path.is_dir():
            aggregate, components, _picks, meta = _load_gt_aggregate_from_pruning_dir(
                report_path, unit_ids=unit_ids, metric_selector=metric_selector
            )
        else:
            aggregate, components, _picks, meta = _load_gt_aggregate_from_skip_eval_report(
                report_path, unit_ids=unit_ids, metric_selector=metric_selector
            )
        loaded.append((dataset_name, report_path, aggregate, components, meta))

    metric_names = list(loaded[0][3])
    for dataset_name, _path, _aggregate, components, _meta in loaded[1:]:
        if list(components) != metric_names:
            raise ValueError(
                f"GT metrics for {dataset_name!r} do not match the other PRUNE_DATASET reports: "
                f"{list(components)} != {metric_names}"
            )
    with np.errstate(all="ignore"):
        aggregate = np.mean(np.stack([item[2] for item in loaded], axis=0), axis=0)
        components = {
            metric: np.mean(np.stack([item[3][metric] for item in loaded], axis=0), axis=0)
            for metric in metric_names
        }
    picks = _gt_picks_from_score(aggregate, unit_ids)
    meta = {
        "source_kind": "prune_dataset_gt_mean",
        "datasets": dataset_names,
        "dataset_aggregation": "mean" if len(loaded) > 1 else None,
        "sources": [str(item[1]) for item in loaded],
        "metric_selector": str(metric_selector),
        "metrics": metric_names,
        "per_dataset": {item[0]: item[4] for item in loaded},
    }
    per_dataset_components = {item[0]: item[3] for item in loaded}
    per_dataset_aggregates = {item[0]: item[2] for item in loaded}
    return aggregate, components, picks, meta, per_dataset_components, per_dataset_aggregates


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
    if start <= 0:
        raise ValueError(f"prune interval [{start},{end}] deletes protected unit 0")
    key = interval_selector(start, end)
    drop = _pick_drop_indices(pick)
    score_source_index = start - 1
    record: dict[str, Any] = {
        "selector": key,
        "eval_selector": key,
        "i": start,
        "j": end,
        "start": start,
        "end": end,
        "span": int(pick.span),
        "drop_count": len(drop),
        "score_source_index": int(score_source_index),
        "score_source_unit_id": str(unit_ids[score_source_index]),
        "score_endpoint_unit_index": end,
        "score_endpoint_unit_id": str(unit_ids[end]) if 0 <= end < len(unit_ids) else "",
        "score_placeholder": False,
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


def _set_selector(strategy: str, drop_count: int) -> str:
    key = str(strategy or "").strip().lower()
    if not key:
        raise ValueError("set strategy key must be non-empty")
    count = int(drop_count)
    if count < 0:
        raise ValueError(f"drop_count must be >= 0, got {count}")
    return f"{key}_k{count}"


def _record_from_drop_set(
    *,
    selector: str,
    drop: Sequence[int],
    unit_ids: Sequence[str],
    num_units: int,
    param_accounting: dict[str, Any],
    score_name: str,
    score_value: float,
    source_strategy: str,
    source: dict[str, Any],
    ranking: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    drop_sorted = sorted({int(x) for x in list(drop)})
    bad = [idx for idx in drop_sorted if idx < 0 or idx >= int(num_units)]
    if bad:
        raise ValueError(f"drop set contains indices outside [0,{num_units}): {bad}")
    if 0 in drop_sorted:
        raise ValueError("drop set deletes protected unit 0")
    record: dict[str, Any] = {
        "selector": str(selector),
        "eval_selector": physical_pruning_selector_from_drop(drop_sorted),
        "selector_kind": "set",
        "drop_count": len(drop_sorted),
        "span": len(drop_sorted),
        "drop": drop_sorted,
        "dropped_unit_ids": [str(unit_ids[idx]) for idx in drop_sorted],
        "source_strategy": str(source_strategy),
        "source_strategies": [str(source_strategy)],
        "source": source,
        "score_name": str(score_name),
        **_span_param_record(
            param_accounting=param_accounting,
            num_units=num_units,
            drop_count=len(drop_sorted),
        ),
    }
    if ranking is not None:
        record["ranking"] = [int(x) for x in list(ranking)]
    if math.isfinite(float(score_value)):
        record["score"] = float(score_value)
        record[str(score_name)] = float(score_value)
    return record


def _merge_set_record(
    sets: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    source_strategy: str,
) -> None:
    key = str(record["selector"])
    existing = sets.get(key)
    if existing is None:
        sets[key] = record
        return
    strategies = set(str(x) for x in list(existing.get("source_strategies", [])))
    strategies.add(str(source_strategy))
    existing["source_strategies"] = sorted(strategies)
    for k, v in record.items():
        if k not in existing:
            existing[k] = v


def _strategy_from_drop_sets(
    *,
    key: str,
    display_name: str,
    drops_by_count: dict[int, Sequence[int]],
    scores_by_count: dict[int, float],
    unit_ids: Sequence[str],
    num_units: int,
    param_accounting: dict[str, Any],
    score_name: str,
    source: dict[str, Any],
    sets: dict[str, dict[str, Any]],
    ranking: Optional[Sequence[int]] = None,
    record_extra_by_count: Optional[dict[int, dict[str, Any]]] = None,
    max_drop_budget: Optional[int] = None,
) -> dict[str, Any]:
    max_budget = _normalize_max_drop_budget(max_drop_budget)
    points: dict[str, str] = {}
    scores: dict[str, float] = {}
    drop_counts = sorted(int(k) for k in drops_by_count.keys())
    if max_budget is not None:
        drop_counts = [count for count in drop_counts if count <= max_budget]
    for drop_count in drop_counts:
        selector = _set_selector(key, drop_count)
        value = float(scores_by_count.get(drop_count, float("nan")))
        record = _record_from_drop_set(
            selector=selector,
            drop=drops_by_count[drop_count],
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name=score_name,
            score_value=value,
            source_strategy=key,
            source=source,
            ranking=ranking,
        )
        if record_extra_by_count is not None:
            extra = record_extra_by_count.get(int(drop_count), {})
            if isinstance(extra, dict):
                record.update(extra)
        _merge_set_record(sets, record, source_strategy=key)
        points[str(int(drop_count))] = selector
        if math.isfinite(value):
            scores[str(int(drop_count))] = float(value)
    return {
        "display_name": display_name,
        "x_axis": "drop_count",
        "points": points,
        "scores": scores,
        "score_name": score_name,
        "source": {**source, "max_drop_budget": max_budget},
    }


def _one_or_two_interval_drop_sets_from_score(
    score: np.ndarray,
    *,
    maximize: bool,
    two_interval_score_aggregation: str = GT_TWO_INTERVAL_SCORE_AGGREGATION,
    comparison_tolerance: float = GT_OURS_COMPARISON_TOLERANCE,
    double_gap_comparison_tolerance: float = (
        GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE
    ),
    single_double_comparison_tolerance: float = (
        GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
    ),
    selection_mode: str = "best",
    min_retained_units_between_intervals: int = MIN_RETAINED_UNITS_BETWEEN_INTERVALS,
    max_drop_budget: Optional[int] = None,
    require_cross_budget_interval_overlap: bool = (
        GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
    ),
    limit_double_to_one_split: bool = False,
) -> tuple[
    dict[int, list[int]],
    dict[int, float],
    dict[int, dict[str, Any]],
]:
    """Select one, exactly two, or the better of one/two closed intervals.

    Both component scores come from the unchanged original-model score matrix.
    Components must leave at least ``min_retained_units_between_intervals``
    retained units between them.
    A two-interval candidate combines its original-model component scores using
    ``two_interval_score_aggregation`` (``sum`` or ``max``). In ``best`` mode,
    it is selected only when that joint score is strictly better than the best
    single interval by more than ``single_double_comparison_tolerance``.
    ``comparison_tolerance`` compares Single candidates: differences at or
    below it keep the earlier candidate. Double candidates instead prefer the
    larger component gap when their scores differ by no more than
    ``double_gap_comparison_tolerance``. A smaller-gap Double replaces a
    larger-gap Double only when its score is strictly better by more than that
    tolerance. Equal-gap Doubles use their score directly. Single/Double ties
    prefer the simpler Single interval.
    ``single`` and ``two`` modes restrict the candidate set without comparing
    across interval counts.
    Budgets are scanned from large to small. Every component interval selected
    at count ``n`` must overlap by at least one unit with a component selected
    at count ``n + 1`` when
    ``require_cross_budget_interval_overlap`` is enabled. When both plans have
    two intervals, the overlap must admit a one-to-one correspondence: every
    interval on both sides must have an overlapping counterpart.
    In adaptive ``best`` mode, ``limit_double_to_one_split`` permits at most one
    Single-to-Double transition. Once a split path returns to Single, all lower
    budgets remain Single.
    When ``max_drop_budget`` is set, larger-budget plans are excluded before
    path selection begins.
    """
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected square interval score matrix, got {values.shape}")
    tolerance = float(comparison_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("comparison_tolerance must be finite and non-negative")
    double_gap_tolerance = float(double_gap_comparison_tolerance)
    if not math.isfinite(double_gap_tolerance) or double_gap_tolerance < 0.0:
        raise ValueError(
            "double_gap_comparison_tolerance must be finite and non-negative"
        )
    single_double_tolerance = float(single_double_comparison_tolerance)
    if not math.isfinite(single_double_tolerance) or single_double_tolerance < 0.0:
        raise ValueError(
            "single_double_comparison_tolerance must be finite and non-negative"
        )
    min_gap = int(min_retained_units_between_intervals)
    if min_gap < 0:
        raise ValueError(
            "min_retained_units_between_intervals must be non-negative"
        )
    max_budget = _normalize_max_drop_budget(max_drop_budget)
    mode = str(selection_mode).strip().lower()
    if mode not in {"best", "single", "two"}:
        raise ValueError(
            f"selection_mode must be 'best', 'single', or 'two', got {selection_mode!r}"
        )
    two_interval_aggregation = str(two_interval_score_aggregation).strip().lower()
    if two_interval_aggregation not in {"sum", "max"}:
        raise ValueError(
            "two_interval_score_aggregation must be 'sum' or 'max', got "
            f"{two_interval_score_aggregation!r}"
        )
    n = int(values.shape[0])
    two_candidates_by_count: dict[
        int, list[tuple[float, tuple[int, int], tuple[int, int]]]
    ] = {}
    single_candidates_by_count: dict[int, list[tuple[float, tuple[int, int]]]] = {}
    intervals: list[tuple[int, int, int, float]] = []
    for start in range(1, n):
        for end in range(start, n):
            value = float(values[start, end])
            if math.isfinite(value):
                drop_count = int(end - start + 1)
                intervals.append((start, end, drop_count, value))
                single_candidates_by_count.setdefault(drop_count, []).append(
                    (value, (int(start), int(end)))
                )
    for first_idx, (first_start, first_end, first_len, first_score) in enumerate(
        intervals
    ):
        for second_start, second_end, second_len, second_score in intervals[
            first_idx + 1 :
        ]:
            if second_start <= first_end + min_gap:
                continue
            total = int(first_len + second_len)
            combined_score = (
                float(first_score + second_score)
                if two_interval_aggregation == "sum"
                else float(max(first_score, second_score))
            )
            two_candidates_by_count.setdefault(total, []).append(
                (
                    combined_score,
                    (int(first_start), int(first_end)),
                    (int(second_start), int(second_end)),
                )
            )

    drops_by_count: dict[int, list[int]] = {}
    scores_by_count: dict[int, float] = {}
    extra_by_count: dict[int, dict[str, Any]] = {}
    available_counts = (
        set(single_candidates_by_count)
        if mode == "single"
        else set(two_candidates_by_count)
        if mode == "two"
        else set(single_candidates_by_count) | set(two_candidates_by_count)
    )
    if max_budget is not None:
        available_counts = {
            count for count in available_counts if int(count) <= max_budget
        }
    cross_budget_constraints_enabled = bool(require_cross_budget_interval_overlap)
    path_constraints_enabled = bool(
        cross_budget_constraints_enabled or limit_double_to_one_split
    )
    totals = sorted(available_counts, reverse=True)
    previous_total: Optional[int] = None
    previous_components: Optional[list[tuple[int, int]]] = None
    previous_num_intervals: Optional[int] = None
    split_has_occurred = False
    for total in totals:
        single_candidates = single_candidates_by_count.get(total, [])
        two_candidates = two_candidates_by_count.get(total, [])
        if path_constraints_enabled and previous_components is not None:
            if previous_total != int(total) + 1:
                break

            if cross_budget_constraints_enabled:
                single_candidates = [
                    candidate
                    for candidate in single_candidates
                    if _each_interval_overlaps_previous_plan(
                        [candidate[1]], previous_components
                    )
                ]
                two_candidates = [
                    candidate
                    for candidate in two_candidates
                    if _each_interval_overlaps_previous_plan(
                        [candidate[1], candidate[2]],
                        previous_components,
                    )
                ]

            if (
                cross_budget_constraints_enabled
                and previous_num_intervals == 2
            ):
                # Double-to-Double continuity is one-to-one. The forward check
                # above covers every current branch; this reverse check also
                # requires every previous branch to have a current successor.
                two_candidates = [
                    candidate
                    for candidate in two_candidates
                    if _each_interval_overlaps_previous_plan(
                        previous_components,
                        [candidate[1], candidate[2]],
                    )
                ]

            if (
                limit_double_to_one_split
                and mode == "best"
                and split_has_occurred
                and previous_num_intervals == 1
            ):
                two_candidates = []

            has_eligible_candidate = (
                bool(single_candidates)
                if mode == "single"
                else bool(two_candidates)
                if mode == "two"
                else bool(single_candidates or two_candidates)
            )
            if not has_eligible_candidate:
                break

        best_single = single_candidates[0] if single_candidates else None
        for candidate in single_candidates[1:]:
            assert best_single is not None
            delta = float(candidate[0] - best_single[0])
            if (maximize and delta > tolerance) or (
                not maximize and delta < -tolerance
            ):
                best_single = candidate

        # Start from the widest gap so the result does not inherit interval
        # enumeration order. Narrower gaps must earn their way in by exceeding
        # the dedicated score tolerance; equal gaps compare scores directly.
        ordered_two_candidates = sorted(
            two_candidates,
            key=lambda candidate: candidate[2][0] - candidate[1][1] - 1,
            reverse=True,
        )
        best_two = ordered_two_candidates[0] if ordered_two_candidates else None
        for candidate in ordered_two_candidates[1:]:
            assert best_two is not None
            delta = float(candidate[0] - best_two[0])
            candidate_gap = int(candidate[2][0] - candidate[1][1] - 1)
            best_gap = int(best_two[2][0] - best_two[1][1] - 1)
            candidate_strictly_better = (maximize and delta > 0.0) or (
                not maximize and delta < 0.0
            )
            candidate_better_beyond_gap_tolerance = (
                maximize and delta > double_gap_tolerance
            ) or (not maximize and delta < -double_gap_tolerance)
            if (
                candidate_gap < best_gap
                and candidate_better_beyond_gap_tolerance
            ) or (
                candidate_gap == best_gap and candidate_strictly_better
            ):
                best_two = candidate

        use_two = mode == "two"
        if mode == "best" and best_two is not None:
            if best_single is None:
                use_two = True
            else:
                delta = float(best_two[0] - best_single[0])
                use_two = (maximize and delta > single_double_tolerance) or (
                    not maximize and delta < -single_double_tolerance
                )

        if use_two:
            assert best_two is not None
            combined_score, first, second = best_two
            components = [first, second]
            aggregation = two_interval_aggregation
        else:
            assert best_single is not None
            combined_score, first = best_single
            components = [first]
            aggregation = "single_interval_score"

        num_intervals = int(len(components))
        interval_count_transition: Optional[str] = None
        if limit_double_to_one_split:
            if previous_num_intervals is None:
                # A path whose highest available budget is already Double is
                # treated as having split before the observed budget range.
                split_has_occurred = num_intervals == 2
            elif previous_num_intervals == 1 and num_intervals == 2:
                interval_count_transition = "single_to_double"
                split_has_occurred = True
            elif previous_num_intervals == 2 and num_intervals == 1:
                interval_count_transition = "double_to_single"

        drop = [
            idx
            for component_start, component_end in components
            for idx in range(int(component_start), int(component_end) + 1)
        ]
        drops_by_count[int(total)] = [int(x) for x in drop]
        scores_by_count[int(total)] = float(combined_score)
        extra_by_count[int(total)] = {
            "component_intervals": [
                {
                    "start": int(component_start),
                    "end": int(component_end),
                    "drop_count": int(component_end - component_start + 1),
                    "score": float(values[component_start, component_end]),
                }
                for component_start, component_end in components
            ],
            "num_intervals": num_intervals,
            "joint_score_aggregation": aggregation,
            "scores_recomputed_after_first_drop": False,
            "best_single_interval_score": (
                None if best_single is None else float(best_single[0])
            ),
            "best_single_interval": (
                None
                if best_single is None
                else {
                    "start": int(best_single[1][0]),
                    "end": int(best_single[1][1]),
                    "drop_count": int(
                        best_single[1][1] - best_single[1][0] + 1
                    ),
                    "score": float(best_single[0]),
                }
            ),
            **(
                {
                    "best_two_interval_score_sum": (
                        None if best_two is None else float(best_two[0])
                    )
                }
                if two_interval_aggregation == "sum"
                else {
                    "best_two_interval_score_max": (
                        None if best_two is None else float(best_two[0])
                    )
                }
            ),
            "selection_mode": mode,
            "min_retained_units_between_intervals": min_gap,
            "max_drop_budget": max_budget,
            "tie_policy": (
                "prefer_single_interval"
                if mode == "best"
                else f"force_{mode}_interval"
            ),
            "comparison_tolerance": tolerance,
            "double_gap_comparison_tolerance": double_gap_tolerance,
            "double_candidate_tie_policy": "prefer_larger_interval_gap",
            "single_double_comparison_tolerance": (
                single_double_tolerance if mode == "best" else None
            ),
            "cross_budget_interval_overlap_required": bool(
                require_cross_budget_interval_overlap
            ),
            "cross_budget_scan_direction": "large_to_small",
            "double_one_split_only": bool(limit_double_to_one_split),
            "cross_budget_double_interval_one_to_one_required": bool(
                require_cross_budget_interval_overlap and mode != "single"
            ),
            "single_to_double_split_has_occurred": bool(split_has_occurred),
            "interval_count_transition": interval_count_transition,
            "linked_to_drop_count": (
                None if previous_components is None else int(previous_total)
            ),
        }
        previous_total = int(total)
        previous_components = list(components)
        previous_num_intervals = num_intervals
    return drops_by_count, scores_by_count, extra_by_count


def _official_replaceme_drop_sets_from_distance(
    distance: np.ndarray,
    *,
    num_blocks: int,
    merge_consecutive: bool = False,
) -> tuple[
    dict[int, list[int]],
    dict[int, float],
    dict[int, dict[str, Any]],
]:
    """Reproduce ReplaceMe's equal-length greedy non-overlapping block scan.

    Every selected block has length ``layers_to_skip``. Candidates are stably
    sorted by their individual angular distance and greedily accepted when
    they do not overlap a prior selection. Adjacent blocks remain legal.
    The returned mean is diagnostic only, never a joint selection objective.
    """
    values = np.asarray(distance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected square interval distance matrix, got {values.shape}")
    blocks_required = int(num_blocks)
    if blocks_required < 1:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")

    n = int(values.shape[0])
    drops_by_count: dict[int, list[int]] = {}
    scores_by_count: dict[int, float] = {}
    extra_by_count: dict[int, dict[str, Any]] = {}
    for layers_to_skip in range(1, n):
        candidates: list[tuple[int, int, float]] = []
        for start in range(1, n - layers_to_skip + 1):
            end = int(start + layers_to_skip - 1)
            value = float(values[start, end])
            if math.isfinite(value):
                candidates.append((int(start), end, value))

        # Official code builds candidates in start order and uses a stable sort.
        candidates.sort(key=lambda candidate: candidate[2])
        selected: list[tuple[int, int, float]] = []
        used_units: set[int] = set()
        for start, end, value in candidates:
            block_units = set(range(start, end + 1))
            if block_units & used_units:
                continue
            selected.append((start, end, value))
            used_units.update(block_units)
            if len(selected) >= blocks_required:
                break
        if len(selected) != blocks_required:
            continue

        selected_for_eval = sorted(selected, key=lambda item: (item[0], item[1]))
        if merge_consecutive and selected_for_eval:
            merged: list[tuple[int, int, float]] = [selected_for_eval[0]]
            for start, end, value in selected_for_eval[1:]:
                previous_start, previous_end, previous_value = merged[-1]
                if start == previous_end + 1:
                    merged[-1] = (
                        previous_start,
                        end,
                        float(np.mean([previous_value, value])),
                    )
                else:
                    merged.append((start, end, value))
            selected_for_eval = merged

        total_drop = int(blocks_required * layers_to_skip)
        drops_by_count[total_drop] = sorted(
            idx
            for start, end, _value in selected_for_eval
            for idx in range(start, end + 1)
        )
        selected_distances = [float(value) for _start, _end, value in selected]
        scores_by_count[total_drop] = float(np.mean(selected_distances))
        extra_by_count[total_drop] = {
            "num_A": blocks_required,
            "layers_to_skip": int(layers_to_skip),
            "official_selection_order": [
                {
                    "start": int(start),
                    "end": int(end),
                    "drop_count": int(end - start + 1),
                    "angular_distance": float(value),
                }
                for start, end, value in selected
            ],
            "selected_block_distances": selected_distances,
            "reported_score_aggregation": "mean_for_diagnostics_only",
            "selection_objective": "greedy_individual_angular_distance",
            "merge_consecutive": bool(merge_consecutive),
            "scores_recomputed_after_first_drop": False,
        }
    return drops_by_count, scores_by_count, extra_by_count


def _each_interval_overlaps_previous_plan(
    components: Sequence[tuple[int, int]],
    previous_components: Sequence[tuple[int, int]],
) -> bool:
    """Return whether every current closed interval overlaps the previous plan."""
    return all(
        any(
            current_start <= previous_end and previous_start <= current_end
            for previous_start, previous_end in previous_components
        )
        for current_start, current_end in components
    )


def select_gt_single_interval_path(
    score: np.ndarray,
    *,
    max_drop_budget: Optional[int] = None,
) -> tuple[
    dict[int, list[int]],
    dict[int, float],
    dict[int, dict[str, Any]],
]:
    """Select the canonical continuous GT single-interval path."""
    return _one_or_two_interval_drop_sets_from_score(
        score,
        maximize=False,
        comparison_tolerance=GT_OURS_COMPARISON_TOLERANCE,
        selection_mode="single",
        max_drop_budget=max_drop_budget,
        require_cross_budget_interval_overlap=(
            GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
        ),
    )


def _similarity_to_proxy_drop_score(score: np.ndarray) -> np.ndarray:
    """Convert Ours similarity to a lower-is-better proxy drop."""
    values = np.asarray(score, dtype=np.float64)
    return np.where(np.isfinite(values), 1.0 - values, np.nan)


def _similarity_proxy_evidence(
    score: np.ndarray,
    *,
    display_name: str,
    metric_label: str,
) -> dict[str, Any]:
    """Build consistently scaled evidence for similarity-based strategies."""
    return {
        "values": _similarity_to_proxy_drop_score(score),
        "title": f"{display_name} evidence: 1 − {metric_label}",
        "colorbar_label": f"1 − {metric_label} ↓",
        "vmin": 0.0,
        "vmax": 1.0,
    }


def _shortgpt_drop_sets_from_unit_similarity(
    unit_similarity: np.ndarray,
) -> tuple[
    dict[int, list[int]],
    dict[int, float],
    list[int],
    dict[int, dict[str, Any]],
]:
    """Apply the official one-shot ShortGPT BI ranking at every budget."""
    scores = np.asarray(unit_similarity, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"expected square unit similarity matrix, got {scores.shape}")
    n = int(scores.shape[0])
    redundancy = {
        int(k): float(scores[k - 1, k])
        for k in range(1, n)
        if math.isfinite(float(scores[k - 1, k]))
    }
    # BI[k] = 1 - cosine[k-1,k]; smallest BI is largest cosine.
    ranking = sorted(redundancy, key=lambda k: (-float(redundancy[k]), int(k)))

    drops_by_count: dict[int, list[int]] = {}
    scores_by_count: dict[int, float] = {}
    extra_by_count: dict[int, dict[str, Any]] = {}
    for drop_count in range(1, len(ranking) + 1):
        selected = sorted(int(idx) for idx in ranking[:drop_count])
        components: list[tuple[int, int]] = []
        component_start = component_end = selected[0]
        for idx in selected[1:]:
            if idx == component_end + 1:
                component_end = idx
            else:
                components.append((component_start, component_end))
                component_start = component_end = idx
        components.append((component_start, component_end))

        drops_by_count[drop_count] = selected
        scores_by_count[drop_count] = float(
            sum(float(redundancy[idx]) for idx in selected)
        )
        extra_by_count[drop_count] = {
            "component_intervals": [
                {
                    "start": int(start),
                    "end": int(end),
                    "drop_count": int(end - start + 1),
                }
                for start, end in components
            ],
            "num_intervals": len(components),
            "block_influence": {
                str(int(idx)): float(1.0 - redundancy[idx]) for idx in selected
            },
            "global_pruning_order": [int(idx) for idx in ranking],
            "selection": "prefix_of_single_global_bi_ranking",
            "budget_path_nested": True,
            "interval_count_constraint": None,
            "interval_gap_constraint": None,
            "comparison_tolerance": None,
            "joint_score_aggregation": "sum_selected_unit_redundancy_for_diagnostics",
            "scores_recomputed_after_first_drop": False,
        }
    return drops_by_count, scores_by_count, ranking, extra_by_count


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
    max_drop_budget: Optional[int] = None,
) -> dict[str, Any]:
    max_budget = _normalize_max_drop_budget(max_drop_budget)
    points: dict[str, str] = {}
    scores: dict[str, float] = {}
    score_arr = None if score is None else np.asarray(score, dtype=np.float64)
    for pick in sorted(picks, key=lambda p: int(p.span)):
        span = int(pick.span)
        if max_budget is not None and span > max_budget:
            continue
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
        "source": {**source, "max_drop_budget": max_budget},
    }


def _mean_abs_consecutive_row_delta(block: np.ndarray) -> float:
    values = np.asarray(block, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(values, axis=0))
    finite = np.isfinite(diffs)
    if not np.any(finite):
        return 0.0
    return float(np.nanmean(diffs[finite]))


def _simpler_cutoff_scores(sim: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Score suffix cutoffs with the SIMPLER TL-vs-BR row-delta criterion."""
    values = np.asarray(sim, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected square similarity matrix, got shape={values.shape}")
    n = int(values.shape[0])
    scores = np.full(n + 1, np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for cutoff in range(0, n + 1):
        retained = values[:cutoff, :cutoff]
        pruned = values[cutoff:n, cutoff:n]
        retained_delta = _mean_abs_consecutive_row_delta(retained)
        pruned_delta = _mean_abs_consecutive_row_delta(pruned)
        score = retained_delta - pruned_delta
        scores[cutoff] = float(score)
        records.append(
            {
                "cutoff": int(cutoff),
                "drop_count": int(n - cutoff),
                "retained_units": int(cutoff),
                "pruned_units": int(n - cutoff),
                "retained_delta": float(retained_delta),
                "pruned_delta": float(pruned_delta),
                "score": float(score),
                "paper_valid": bool(2 <= cutoff <= n - 2),
            }
        )
    valid_cutoffs = [c for c in range(0, n + 1) if 2 <= c <= n - 2 and math.isfinite(float(scores[c]))]
    best_cutoff: Optional[int] = None
    if valid_cutoffs:
        best_cutoff = max(valid_cutoffs, key=lambda c: (float(scores[c]), -int(c)))
    return scores, {
        "cutoff_score": "delta(top_left_retained)-delta(bottom_right_pruned)",
        "paper_valid_cutoffs": [int(c) for c in valid_cutoffs],
        "best_cutoff": int(best_cutoff) if best_cutoff is not None else None,
        "best_drop_count": int(n - best_cutoff) if best_cutoff is not None else None,
        "records": records,
    }


def _simpler_suffix_picks(
    *,
    scores_by_cutoff: np.ndarray,
    unit_ids: Sequence[str],
) -> tuple[list[TrianglePick], np.ndarray]:
    """Build the full suffix-pruning curve used by the SIMPLER baseline."""
    n = int(len(unit_ids))
    score_matrix = np.full((n, n), np.nan, dtype=np.float64)
    picks: list[TrianglePick] = []
    for drop_count in range(1, max(1, n - 1)):
        cutoff = n - int(drop_count)
        if cutoff < 1 or cutoff >= n:
            continue
        start = int(cutoff)
        end = n - 1
        value = float(scores_by_cutoff[cutoff])
        if math.isfinite(value):
            score_matrix[start, end] = value
        picks.append(
            TrianglePick(
                i=start,
                j=end,
                span=int(drop_count),
                unit_i=str(unit_ids[start]),
                unit_j=str(unit_ids[end]),
                dropped_unit_ids=tuple(str(unit_ids[k]) for k in range(start, end + 1)),
            )
        )
    return picks, score_matrix


def _strategy_summary_score_series(
    *,
    strategy_record: dict[str, Any],
    gt_meta: Optional[dict[str, Any]],
) -> list[tuple[int, float]]:
    if not isinstance(gt_meta, dict):
        return []
    selector_scores = gt_meta.get("selector_scores", {})
    if not isinstance(selector_scores, dict):
        return []
    points = strategy_record.get("points", {})
    if not isinstance(points, dict):
        return []
    series: list[tuple[int, float]] = []
    for drop_key, selector in sorted(
        points.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if not str(drop_key).isdigit():
            continue
        value = selector_scores.get(str(selector))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            series.append((int(drop_key), float(value)))
    return series


def _build_ours_metric_strategy_pair(
    *,
    family_key: str,
    display_name: str,
    metric: str,
    interval_score: np.ndarray,
    similarity_source: dict[str, Any],
    token_region: str,
    dataset: str,
    unit_ids: Sequence[str],
    num_units: int,
    param_accounting: dict[str, Any],
    sets: dict[str, dict[str, Any]],
    max_drop_budget: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Build Ours Single/Double identically for any similarity metric."""
    proxy_drop_score = _similarity_to_proxy_drop_score(interval_score)
    single_drops, single_scores, single_extra = (
        _one_or_two_interval_drop_sets_from_score(
            proxy_drop_score,
            maximize=False,
            comparison_tolerance=GT_OURS_COMPARISON_TOLERANCE,
            selection_mode="single",
            max_drop_budget=max_drop_budget,
            require_cross_budget_interval_overlap=(
                GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
            ),
        )
    )
    double_drops, double_scores, double_extra = (
        _one_or_two_interval_drop_sets_from_score(
            proxy_drop_score,
            maximize=False,
            two_interval_score_aggregation=OURS_TWO_INTERVAL_SCORE_AGGREGATION,
            comparison_tolerance=GT_OURS_COMPARISON_TOLERANCE,
            single_double_comparison_tolerance=(
                GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
            ),
            max_drop_budget=max_drop_budget,
            require_cross_budget_interval_overlap=(
                GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
            ),
            limit_double_to_one_split=GT_OURS_DOUBLE_ONE_SPLIT_ONLY,
        )
    )
    source = {
        "kind": "similarity",
        "metric": str(metric),
        "token_region": str(token_region),
        "dataset": str(dataset),
        "similarity_source": similarity_source,
        "unit_similarity": "loaded from unit-output pairwise redundancy graph",
        "interval_score": "derived in memory as C[start,end]=S[start-1,end]",
        "proxy_drop": "D[start,end]=1-C[start,end]",
        "optimization_direction": "minimize",
        "cross_budget_path_constraint": (
            (
                "each_component_interval_at_drop_count_n_overlaps_at_least_one_"
                "component_interval_at_drop_count_n_plus_1;"
                "double_to_double_requires_one_to_one_overlap"
            )
            if GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
            else False
        ),
        "cross_budget_scan_direction": "large_to_small",
        "max_drop_budget": max_drop_budget,
        "component_gap_policy": "at_least_one_retained_unit",
        "scores_recomputed_after_first_drop": False,
    }
    single_key = f"{family_key}_single"
    double_key = f"{family_key}_double"
    return {
        single_key: _strategy_from_drop_sets(
            key=single_key,
            display_name=f"{display_name} Single",
            drops_by_count=single_drops,
            scores_by_count=single_scores,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name="single_interval_similarity_proxy_drop",
            source={
                **source,
                "selection": (
                    "descending-drop-count minimum single interval, overlapping "
                    "the selected interval at the next larger drop count"
                ),
                "max_num_intervals": 1,
                "interval_mode": "single",
            },
            sets=sets,
            record_extra_by_count=single_extra,
            max_drop_budget=max_drop_budget,
        ),
        double_key: _strategy_from_drop_sets(
            key=double_key,
            display_name=f"{display_name} Double",
            drops_by_count=double_drops,
            scores_by_count=double_scores,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name="one_or_two_interval_similarity_proxy_drop",
            source={
                **source,
                "two_interval_score_aggregation": (
                    OURS_TWO_INTERVAL_SCORE_AGGREGATION
                ),
                "selection": (
                    "descending-drop-count minimum of one-interval proxy drop or "
                    f"{OURS_TWO_INTERVAL_SCORE_AGGREGATION}-aggregated "
                    "two-separated-interval proxy drops, with every "
                    "component overlapping the selected plan at the next larger "
                    "drop count"
                ),
                "comparison_tolerance": float(
                    GT_OURS_COMPARISON_TOLERANCE
                ),
                "double_gap_comparison_tolerance": float(
                    GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE
                ),
                "single_double_comparison_tolerance": float(
                    GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
                ),
                "max_num_intervals": 2,
                "interval_mode": "best_of_one_or_two",
                "tie_policy": "prefer_single_interval",
                "double_one_split_only": bool(
                    GT_OURS_DOUBLE_ONE_SPLIT_ONLY
                ),
                "cross_budget_double_interval_one_to_one_required": bool(
                    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                ),
            },
            sets=sets,
            record_extra_by_count=double_extra,
            max_drop_budget=max_drop_budget,
        ),
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
    strategy_selector: Optional[str] = None,
    max_drop_budget: Optional[int] = None,
) -> List[Path]:
    """Build a prune-only span bundle and return the written path."""
    max_drop_budget = _normalize_max_drop_budget(max_drop_budget)
    unit_ids, unit_similarity, similarity_source = load_similarity(
        experiment=experiment,
        dataset=dataset,
        token_region=token_region,
        metric=metric,
    )
    interval_score = build_interval_score_matrix(unit_similarity)
    num_units = int(interval_score.shape[0])
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
    gt_corner_score: Optional[np.ndarray] = None
    gt_corner_picks: list[TrianglePick] = []
    gt_corner_meta: Optional[dict[str, Any]] = None
    prune_datasets = parse_dataset_list(dataset)
    (
        gt_corner_score,
        _gt_component_heatmaps,
        gt_corner_picks,
        gt_corner_meta,
        _gt_component_heatmaps_by_dataset,
        _gt_aggregate_heatmaps_by_dataset,
    ) = _load_gt_aggregate_for_datasets(
        plan_root=plan_root,
        datasets=prune_datasets,
        unit_ids=unit_ids,
        metric_selector=str(gt_corner_metric),
        explicit_report=gt_corner_skip_eval_report,
    )
    print(
        f"Loaded GT aggregate corner heatmap for PRUNE_DATASET={','.join(prune_datasets)} "
        f"(metrics={len(gt_corner_meta.get('metrics', []))}, spans={len(gt_corner_picks)})",
        flush=True,
    )

    interval_records: dict[str, dict[str, Any]] = {}
    set_records: dict[str, dict[str, Any]] = {}
    strategies: dict[str, dict[str, Any]] = {}
    cosine_interval_score: Optional[np.ndarray] = None
    inv_mse_interval_score: Optional[np.ndarray] = None
    cca_interval_score: Optional[np.ndarray] = None
    angular_interval_distance: Optional[np.ndarray] = None
    strategies.update(
        _build_ours_metric_strategy_pair(
            family_key="ours",
            display_name="Ours_CKA",
            metric=str(metric),
            interval_score=interval_score,
            similarity_source=similarity_source,
            token_region=token_region,
            dataset=dataset,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            sets=set_records,
            max_drop_budget=max_drop_budget,
        )
    )
    try:
        cosine_unit_ids, cosine_unit_similarity, cosine_similarity_source = load_similarity(
            experiment=experiment,
            dataset=dataset,
            token_region=token_region,
            metric="cosine",
        )
        if list(cosine_unit_ids) != list(unit_ids):
            print(
                "[warn] Skip cosine-based strategies: unit_ids do not match Ours source.",
                flush=True,
            )
        else:
            cosine_interval_score = build_interval_score_matrix(cosine_unit_similarity)
            strategies.update(
                _build_ours_metric_strategy_pair(
                    family_key="cosine",
                    display_name="Ours_Cosine",
                    metric="cosine",
                    interval_score=cosine_interval_score,
                    similarity_source=cosine_similarity_source,
                    token_region=token_region,
                    dataset=dataset,
                    unit_ids=cosine_unit_ids,
                    num_units=num_units,
                    param_accounting=param_accounting,
                    sets=set_records,
                    max_drop_budget=max_drop_budget,
                )
            )
            (
                shortgpt_drops,
                shortgpt_scores,
                shortgpt_ranking,
                shortgpt_extra,
            ) = _shortgpt_drop_sets_from_unit_similarity(
                cosine_unit_similarity
            )
            strategies["shortgpt"] = _strategy_from_drop_sets(
                key="shortgpt",
                display_name="ShortGPT",
                drops_by_count=shortgpt_drops,
                scores_by_count=shortgpt_scores,
                unit_ids=cosine_unit_ids,
                num_units=num_units,
                param_accounting=param_accounting,
                score_name="selected_unit_redundancy_sum",
                source={
                    "kind": "shortgpt",
                    "metric": "cosine",
                    "token_region": str(token_region),
                    "dataset": str(dataset),
                    "similarity_source": cosine_similarity_source,
                    "unit_similarity": (
                        "loaded from unit-output pairwise redundancy graph"
                    ),
                    "selection": (
                        "official ShortGPT global BI ranking prefix; "
                        "BI[k]=1-cosine(unit[k-1],unit[k])"
                    ),
                    "max_num_intervals": None,
                    "interval_mode": "unconstrained_global_ranking_prefix",
                    "scores_recomputed_after_first_drop": False,
                },
                sets=set_records,
                ranking=shortgpt_ranking,
                record_extra_by_count=shortgpt_extra,
                max_drop_budget=max_drop_budget,
            )
    except Exception as exc:
        print(f"[warn] Could not build cosine-based strategies: {exc}", flush=True)

    try:
        inv_mse_unit_ids, inv_mse_unit_similarity, inv_mse_similarity_source = load_similarity(
            experiment=experiment,
            dataset=dataset,
            token_region=token_region,
            metric="inv_mse",
        )
        if list(inv_mse_unit_ids) != list(unit_ids):
            print(
                "[warn] Skip INV_MSE strategies: unit_ids do not match Ours source.",
                flush=True,
            )
        else:
            inv_mse_interval_score = build_interval_score_matrix(inv_mse_unit_similarity)
            strategies.update(
                _build_ours_metric_strategy_pair(
                    family_key="inv_mse",
                    display_name="Ours_INV_MSE",
                    metric="inv_mse",
                    interval_score=inv_mse_interval_score,
                    similarity_source=inv_mse_similarity_source,
                    token_region=token_region,
                    dataset=dataset,
                    unit_ids=inv_mse_unit_ids,
                    num_units=num_units,
                    param_accounting=param_accounting,
                    sets=set_records,
                    max_drop_budget=max_drop_budget,
                )
            )
    except Exception as exc:
        print(f"[warn] Could not build INV_MSE strategies: {exc}", flush=True)

    try:
        cca_unit_ids, cca_unit_similarity, cca_similarity_source = load_similarity(
            experiment=experiment,
            dataset=dataset,
            token_region=token_region,
            metric="cca",
        )
        if list(cca_unit_ids) != list(unit_ids):
            print(
                "[warn] Skip CCA strategies: unit_ids do not match Ours source.",
                flush=True,
            )
        else:
            cca_interval_score = build_interval_score_matrix(cca_unit_similarity)
            strategies.update(
                _build_ours_metric_strategy_pair(
                    family_key="cca",
                    display_name="Ours_CCA",
                    metric="cca",
                    interval_score=cca_interval_score,
                    similarity_source=cca_similarity_source,
                    token_region=token_region,
                    dataset=dataset,
                    unit_ids=cca_unit_ids,
                    num_units=num_units,
                    param_accounting=param_accounting,
                    sets=set_records,
                    max_drop_budget=max_drop_budget,
                )
            )
    except Exception as exc:
        print(f"[warn] Could not build CCA strategies: {exc}", flush=True)

    try:
        angular_unit_ids, angular_unit_distance, angular_distance_source = load_similarity(
            experiment=experiment,
            dataset=dataset,
            token_region=token_region,
            metric="angular",
        )
        if list(angular_unit_ids) != list(unit_ids):
            print(
                "[warn] Skip ReplaceMe strategies: unit_ids do not match Ours source.",
                flush=True,
            )
        else:
            angular_interval_distance = build_interval_score_matrix(angular_unit_distance)
            (
                replace_me_single_drops,
                replace_me_single_scores,
                replace_me_single_extra,
            ) = _official_replaceme_drop_sets_from_distance(
                angular_interval_distance,
                num_blocks=1,
                merge_consecutive=False,
            )
            (
                replace_me_double_drops,
                replace_me_double_scores,
                replace_me_double_extra,
            ) = _official_replaceme_drop_sets_from_distance(
                angular_interval_distance,
                num_blocks=2,
                merge_consecutive=False,
            )
            replace_me_source = {
                "kind": "distance",
                "metric": "angular",
                "token_region": str(token_region),
                "dataset": str(dataset),
                "distance_source": angular_distance_source,
                "unit_distance": (
                    "mean token-wise acos(cosine)/pi loaded from unit-output "
                    "pairwise graph"
                ),
                "interval_distance": "derived in memory as D[start,end]=D[start-1,end]",
                "selection": (
                    "official ReplaceMe stable distance sort followed by greedy "
                    "non-overlapping equal-length block selection"
                ),
                "optimization_direction": "minimize_each_candidate_distance",
                "joint_score_optimization": False,
                "cross_budget_path_constraint": False,
                "component_gap_policy": "non_overlapping_only; adjacency_allowed",
                "merge_consecutive": False,
                "scores_recomputed_after_first_drop": False,
            }
            strategies["replaceme_single"] = _strategy_from_drop_sets(
                key="replaceme_single",
                display_name="ReplaceMe Single",
                drops_by_count=replace_me_single_drops,
                scores_by_count=replace_me_single_scores,
                unit_ids=angular_unit_ids,
                num_units=num_units,
                param_accounting=param_accounting,
                score_name="selected_block_mean_angular_distance",
                source={
                    **replace_me_source,
                    "num_A": 1,
                    "max_num_intervals": 1,
                    "interval_mode": "single",
                },
                sets=set_records,
                record_extra_by_count=replace_me_single_extra,
                max_drop_budget=max_drop_budget,
            )
            strategies["replaceme_double"] = _strategy_from_drop_sets(
                key="replaceme_double",
                display_name="ReplaceMe Double",
                drops_by_count=replace_me_double_drops,
                scores_by_count=replace_me_double_scores,
                unit_ids=angular_unit_ids,
                num_units=num_units,
                param_accounting=param_accounting,
                score_name="selected_block_mean_angular_distance",
                source={
                    **replace_me_source,
                    "num_A": 2,
                    "max_num_intervals": 2,
                    "min_num_intervals_before_optional_merge": 2,
                    "interval_mode": "force_two_equal_length_blocks",
                    "budget_rule": "total_drop=2*layers_to_skip; even budgets only",
                },
                sets=set_records,
                record_extra_by_count=replace_me_double_extra,
                max_drop_budget=max_drop_budget,
            )
    except Exception as exc:
        print(f"[warn] Could not build angular ReplaceMe strategies: {exc}", flush=True)

    simpler_scores_by_cutoff, simpler_meta = _simpler_cutoff_scores(unit_similarity)
    simpler_picks, simpler_score_matrix = _simpler_suffix_picks(
        scores_by_cutoff=simpler_scores_by_cutoff,
        unit_ids=unit_ids,
    )
    if simpler_picks:
        strategies["simpler"] = _strategy_from_picks(
            key="simpler",
            display_name="SIMPLER",
            picks=simpler_picks,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score=simpler_score_matrix,
            score_name="simpler_cutoff_score",
            source={
                "kind": "simpler",
                "metric": str(metric),
                "token_region": str(token_region),
                "dataset": str(dataset),
                "similarity_source": similarity_source,
                "input_similarity": "unit-output pairwise similarity loaded directly from disk",
                "paper": (
                    "SIMPLER: Efficient Foundation Model Adaptation via "
                    "Similarity-Guided Layer Pruning for Earth Observation"
                ),
                "selection": "full suffix-pruning curve; cutoff c drops closed interval [c,n-1]",
                "cross_budget_path_constraint": False,
                "uses_llm_depth_prior": True,
                "depth_prior": "redundancy_is_concentrated_in_later_layers",
                "score": simpler_meta,
            },
            intervals=interval_records,
            max_drop_budget=max_drop_budget,
        )
    if gt_corner_score is not None and gt_corner_picks:
        (
            gt_single_drops,
            gt_single_scores,
            gt_single_extra,
        ) = select_gt_single_interval_path(
            gt_corner_score,
            max_drop_budget=max_drop_budget,
        )
        gt_drops, gt_scores, gt_extra = _one_or_two_interval_drop_sets_from_score(
            gt_corner_score,
            maximize=False,
            two_interval_score_aggregation=GT_TWO_INTERVAL_SCORE_AGGREGATION,
            comparison_tolerance=GT_OURS_COMPARISON_TOLERANCE,
            single_double_comparison_tolerance=(
                GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
            ),
            max_drop_budget=max_drop_budget,
            require_cross_budget_interval_overlap=(
                GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
            ),
            limit_double_to_one_split=GT_OURS_DOUBLE_ONE_SPLIT_ONLY,
        )
        gt_source = {
            "kind": "skip_eval_gt",
            "reports": list(gt_corner_meta.get("sources", [])),
            "metric_selector": str(gt_corner_metric),
            "meta": gt_corner_meta,
            "cross_budget_path_constraint": (
                (
                    "each_component_interval_at_drop_count_n_overlaps_at_least_one_"
                    "component_interval_at_drop_count_n_plus_1;"
                    "double_to_double_requires_one_to_one_overlap"
                )
                if GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                else False
            ),
            "cross_budget_scan_direction": "large_to_small",
            "max_drop_budget": max_drop_budget,
            "double_one_split_only": bool(GT_OURS_DOUBLE_ONE_SPLIT_ONLY),
            "cross_budget_double_interval_one_to_one_required": bool(
                GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
            ),
            "component_gap_policy": "at_least_one_retained_unit",
        }
        strategies["gt_single"] = _strategy_from_drop_sets(
            key="gt_single",
            display_name="GT Single",
            drops_by_count=gt_single_drops,
            scores_by_count=gt_single_scores,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name="single_interval_gt_degradation",
            source={
                **gt_source,
                "selection": (
                    "descending-drop-count minimum single-interval degradation, "
                    "overlapping the selected interval at the next larger drop count"
                ),
                "max_num_intervals": 1,
                "interval_mode": "single",
            },
            sets=set_records,
            record_extra_by_count=gt_single_extra,
            max_drop_budget=max_drop_budget,
        )
        strategies["gt_double"] = _strategy_from_drop_sets(
            key="gt_double",
            display_name="GT Double",
            drops_by_count=gt_drops,
            scores_by_count=gt_scores,
            unit_ids=unit_ids,
            num_units=num_units,
            param_accounting=param_accounting,
            score_name="one_or_two_interval_gt_degradation",
            source={
                **gt_source,
                "interaction_assumption": (
                    "each interval keeps its original-model degradation after the "
                    "other interval is removed; joint degradation uses "
                    f"{GT_TWO_INTERVAL_SCORE_AGGREGATION} aggregation"
                ),
                "two_interval_score_aggregation": (
                    GT_TWO_INTERVAL_SCORE_AGGREGATION
                ),
                "selection": (
                    "descending-drop-count minimum of one-interval degradation or "
                    f"{GT_TWO_INTERVAL_SCORE_AGGREGATION}-aggregated "
                    "two-separated-interval degradation, with every component "
                    "overlapping the selected plan at the next larger drop count"
                ),
                "max_num_intervals": 2,
                "interval_mode": "best_of_one_or_two",
                "tie_policy": "prefer_single_interval",
                "comparison_tolerance": float(
                    GT_OURS_COMPARISON_TOLERANCE
                ),
                "double_gap_comparison_tolerance": float(
                    GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE
                ),
                "single_double_comparison_tolerance": float(
                    GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
                ),
                "cross_budget_path_constraint": (
                    "each_component_interval_at_drop_count_n_overlaps_at_least_one_"
                    "component_interval_at_drop_count_n_plus_1;"
                    "double_to_double_requires_one_to_one_overlap"
                ),
            },
            sets=set_records,
            record_extra_by_count=gt_extra,
            max_drop_budget=max_drop_budget,
        )

    requested_strategies = _parse_pruning_strategies(strategy_selector)
    missing_strategies = [
        key for key in requested_strategies if key not in strategies
    ]
    if missing_strategies:
        raise RuntimeError(
            "requested pruning strategies could not be built: "
            + ", ".join(missing_strategies)
        )
    strategies = {key: strategies[key] for key in requested_strategies}
    referenced_plan_keys = {
        str(plan_key)
        for strategy in strategies.values()
        for plan_key in (
            strategy.get("points", {}).values()
            if isinstance(strategy.get("points", {}), dict)
            else ()
        )
    }
    interval_records = {
        key: record
        for key, record in interval_records.items()
        if key in referenced_plan_keys
    }
    set_records = {
        key: record
        for key, record in set_records.items()
        if key in referenced_plan_keys
    }

    drop_grid_items: list[tuple[str, dict[int, list[int]]]] = []
    plan_records_for_grid = {**interval_records, **set_records}
    for strategy_key, strategy_record in strategies.items():
        points = strategy_record.get("points", {})
        if not isinstance(points, dict) or not points:
            continue
        span_to_drop_indices: dict[int, list[int]] = {}
        for span_key, interval_key in points.items():
            if not str(span_key).isdigit():
                continue
            plan_record = plan_records_for_grid.get(str(interval_key))
            if not isinstance(plan_record, dict):
                continue
            span_to_drop_indices[int(span_key)] = [
                int(x) for x in list(plan_record.get("drop", []))
            ]
        if not span_to_drop_indices:
            continue
        display_name = str(strategy_record.get("display_name", strategy_key))
        drop_grid_items.append((display_name, span_to_drop_indices))
    if drop_grid_items:
        strategy_evidence: dict[str, dict[str, Any]] = {
            "Ours_CKA": _similarity_proxy_evidence(
                interval_score,
                display_name="Ours_CKA",
                metric_label="CKA",
            ),
        }
        if gt_corner_score is not None:
            strategy_evidence["GT"] = {
                "values": gt_corner_score,
                "title": "GT evidence: 7-metric mean",
                "colorbar_label": "normalized degradation ↓",
            }
        if cosine_interval_score is not None:
            strategy_evidence["Ours_Cosine"] = _similarity_proxy_evidence(
                cosine_interval_score,
                display_name="Ours_Cosine",
                metric_label="cosine",
            )
        if inv_mse_interval_score is not None:
            strategy_evidence["Ours_INV_MSE"] = _similarity_proxy_evidence(
                inv_mse_interval_score,
                display_name="Ours_INV_MSE",
                metric_label="INV_MSE",
            )
        if cca_interval_score is not None:
            strategy_evidence["Ours_CCA"] = _similarity_proxy_evidence(
                cca_interval_score,
                display_name="Ours_CCA",
                metric_label="CCA",
            )
        if angular_interval_distance is not None:
            strategy_evidence["ReplaceMe"] = {
                "values": angular_interval_distance,
                "title": "ReplaceMe evidence: 1 − angular similarity",
                "colorbar_label": "angular distance ↓",
                "vmin": 0.0,
                "vmax": 1.0,
            }
        prune_grid_path = render_prune_strategy_drop_grids(
            num_units=num_units,
            strategy_drop_indices=drop_grid_items,
            strategy_evidence=strategy_evidence,
            out_dir=visualization_root,
            backbone=backbone,
            tag=f"prune_plan_{safe_backbone}_drop_grid",
            title=f"{backbone}: prune drop grids",
        )
        print(f"Wrote combined prune drop grid {prune_grid_path}", flush=True)

    ours_points = strategies.get("ours_double", {}).get("points", {})
    ours_drop_counts = sorted(
        int(key)
        for key in (ours_points if isinstance(ours_points, dict) else {})
        if str(key).isdigit()
    )
    span_records: dict[str, dict[str, Any]] = {}
    if isinstance(ours_points, dict):
        for span_key, plan_key in ours_points.items():
            if not str(span_key).isdigit():
                continue
            record = interval_records.get(str(plan_key)) or set_records.get(str(plan_key))
            if isinstance(record, dict):
                span_records[str(int(span_key))] = dict(record)

    bundle = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": str(backbone),
        "dataset": str(dataset),
        "num_units": int(num_units),
        "unit_ids": [str(x) for x in unit_ids],
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
        "sets": {
            key: set_records[key]
            for key in sorted(
                set_records.keys(),
                key=lambda k: (
                    int(set_records[k].get("drop_count", 0)),
                    k,
                ),
            )
        },
        "strategies": strategies,
        "pruning_eval": {},
        "spans": {
            str(span): span_records[str(span)]
            for span in sorted((int(k) for k in span_records.keys()))
        },
        "extra": {
            "similarity_source": similarity_source,
            "metric": str(metric),
            "token_region": str(token_region),
            "max_drop_budget": max_drop_budget,
            "gt_ours_controls": {
                "gt_component_aggregation": str(GT_COMPONENT_AGGREGATION),
                "gt_aggregate_metrics": list(GT_AGGREGATE_METRICS),
                "comparison_tolerance": float(
                    GT_OURS_COMPARISON_TOLERANCE
                ),
                "double_gap_comparison_tolerance": float(
                    GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE
                ),
                "single_double_comparison_tolerance": float(
                    GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE
                ),
                "min_retained_units_between_intervals": int(
                    MIN_RETAINED_UNITS_BETWEEN_INTERVALS
                ),
                "require_cross_budget_interval_overlap": bool(
                    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                ),
                "cross_budget_scan_direction": "large_to_small",
                "double_one_split_only": bool(
                    GT_OURS_DOUBLE_ONE_SPLIT_ONLY
                ),
                "cross_budget_double_interval_one_to_one_required": bool(
                    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                ),
                "gt_two_interval_score_aggregation": str(
                    GT_TWO_INTERVAL_SCORE_AGGREGATION
                ),
                "ours_two_interval_score_aggregation": str(
                    OURS_TWO_INTERVAL_SCORE_AGGREGATION
                ),
            },
            "triangle_legality": "closed_intervals_with_start_at_least_1",
            "multi_interval_legality": (
                "strategy_specific: ReplaceMe requires only non-overlap and allows "
                "adjacency; other multi-interval strategies require at least one "
                "retained unit between components"
            ),
            "interval_semantics": "closed interval iS_jE drops units S through E",
            "interval_score": {
                "matrix": "C[start,end]",
                "source": "derived from unit-output pairwise similarity S",
                "start_positive": "C[start,end] stores boundary/start-1 to endpoint/end similarity",
                "unit_zero_policy": "C[0,:] is NaN because no prior unit output exists",
            },
            "require_endpoint_linked_spans": False,
            "cross_budget_path_constraint": {
                "gt_ours_enabled": bool(
                    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                ),
                "gt_ours_rule": (
                    "component_overlap_with_next_larger_drop_count;"
                    "double_to_double_one_to_one"
                ),
                "other_strategies": False,
            },
            "cross_budget_scan_direction": "large_to_small",
            "max_drop_budget": max_drop_budget,
            "double_one_split_only": {
                "gt_ours_double": bool(GT_OURS_DOUBLE_ONE_SPLIT_ONLY),
                "other_strategies": False,
            },
            "cross_budget_double_interval_one_to_one_required": {
                "gt_ours_double": bool(
                    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                ),
                "other_strategies": False,
            },
            "triangle_selection": "strategy_specific_best_plan_per_total_drop_count",
            "gt_corner_path": gt_corner_meta,
            "max_available_span": int(max(ours_drop_counts, default=0)),
            "available_spans": len(ours_drop_counts),
        },
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(
        f"Wrote {bundle_path} (intervals={len(interval_records)}, sets={len(set_records)}, "
        f"strategies={len(strategies)}, "
        f"ours_spans={len(span_records)}, "
        f"max_available_span={max(ours_drop_counts, default=0)})",
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
        strategy_selector=str(args.strategies),
        max_drop_budget=args.max_drop_budget,
    )


if __name__ == "__main__":
    main()

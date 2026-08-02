"""Shared scaling for skip-eval metrics used by plans and visualizations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SEVEN_METRIC_KEYS = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "acc",
    "comp",
    "chamfer_distance",
)
HIGHER_IS_BETTER = frozenset({"depth_delta1", "nc"})


def degradation_pct_to_multiple(values: np.ndarray, *, metric: str) -> np.ndarray:
    """Convert positive-when-worse percentages directly to degradation multiples."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if str(metric) in HIGHER_IS_BETTER:
        current_ratio = 1.0 - arr[finite] / 100.0
        out[finite] = np.maximum(1.0, 1.0 / np.maximum(current_ratio, 1e-8))
    else:
        out[finite] = np.maximum(1.0, 1.0 + arr[finite] / 100.0)
    return out


def raw_metric_to_upper_multiple_score(
    values: np.ndarray,
    *,
    metric: str,
    upper_baseline: Optional[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    meta: dict[str, Any] = {
        "scale_mode": "raw_metric_upper_multiple",
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
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (
            upper / np.maximum(arr[finite], 1e-8)
            if str(metric) in HIGHER_IS_BETTER
            else arr[finite] / upper
        )
    scores = np.where(np.isfinite(scores), scores, np.nan)
    out[finite] = np.maximum(scores, 1.0)
    meta.update(
        enabled=True,
        num_floored_to_one=int(np.count_nonzero(scores < 1.0)),
    )
    return out, meta


def degradation_pct_to_upper_multiple_score(
    values: np.ndarray,
    *,
    metric: str,
    upper_baseline: Optional[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    if upper_baseline is None:
        return np.full(arr.shape, np.nan), {"enabled": False, "skip_reason": "missing_upper_baseline"}
    upper = float(upper_baseline)
    current = (
        upper - arr * max(abs(upper), 1e-8) / 100.0
        if str(metric) in HIGHER_IS_BETTER
        else upper + arr * max(abs(upper), 1e-8) / 100.0
    )
    return raw_metric_to_upper_multiple_score(
        current,
        metric=metric,
        upper_baseline=upper,
    )


def minmax_normalize_finite(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float64)
    out = np.array(arr, copy=True)
    finite = np.isfinite(arr)
    meta: dict[str, Any] = {"method": "finite_minmax", "min": None, "max": None, "constant": False}
    if not np.any(finite):
        return out, meta
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    out[finite] = (arr[finite] - lo) / (hi - lo) if hi > lo else 0.0
    meta.update(min=lo, max=hi, constant=bool(hi <= lo))
    return out, meta


def normalized_degradation_to_retention(values: np.ndarray) -> np.ndarray:
    """Map normalized degradation to a display score where 1 means no degradation.

    Skip heatmaps historically displayed ``-degradation`` on ``[-1, 0]``.
    Adding one gives the equivalent retention score on ``[0, 1]`` while
    preserving the heatmap's visual ordering.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.array(arr, copy=True)
    finite = np.isfinite(arr)
    out[finite] = np.clip(1.0 - arr[finite], 0.0, 1.0)
    return out


def apply_minmax_bounds(value: float, *, minimum: float, maximum: float) -> float:
    """Apply fitted min-max bounds to one finite value."""
    if not math.isfinite(float(value)):
        return float("nan")
    if float(maximum) <= float(minimum):
        return 0.0
    return float(np.clip((float(value) - float(minimum)) / (float(maximum) - float(minimum)), 0.0, 1.0))


def normalize_metric_components(
    components: Mapping[str, np.ndarray],
    *,
    metric_names: Sequence[str] = SEVEN_METRIC_KEYS,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    normalized: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for metric in metric_names:
        if metric not in components:
            continue
        normalized[metric], records[metric] = minmax_normalize_finite(components[metric])
    if not normalized:
        raise ValueError("No metric components available for normalization")
    stack = np.stack([normalized[name] for name in metric_names if name in normalized], axis=0)
    with np.errstate(all="ignore"):
        aggregate = np.nanmean(stack, axis=0)
    aggregate[~np.any(np.isfinite(stack), axis=0)] = np.nan
    return normalized, aggregate, {"method": "per_metric_finite_minmax_then_nanmean", "metrics": records}

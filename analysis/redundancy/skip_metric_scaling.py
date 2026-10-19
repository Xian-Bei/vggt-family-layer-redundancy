"""Shared scaling for skip-eval metrics used by plans and visualizations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


GT_METRIC_KEYS = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "acc",
    "comp",
    "chamfer_distance",
)
def raw_metric_to_degradation(
    values: np.ndarray,
    *,
    metric: str,
    upper_baseline: Optional[float],
) -> np.ndarray:
    """Convert raw metric values to a non-negative upper-relative delta."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite) or upper_baseline is None:
        return out
    upper = float(upper_baseline)
    if not math.isfinite(upper):
        return out
    delta = arr[finite] - upper
    out[finite] = np.maximum(delta, 0.0)
    return out


def signed_delta_to_degradation(
    values: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Clamp saved ``current - upper`` deltas to direction-aware degradation."""
    arr = np.asarray(values, dtype=np.float64)
    del metric
    out = np.array(arr, copy=True)
    finite = np.isfinite(out)
    out[finite] = np.maximum(out[finite], 0.0)
    return out


def normalize_canonical_degradation(
    values: np.ndarray,
    *,
    maximum: float,
    clamp_better_than_upper: bool = True,
) -> np.ndarray:
    """Normalize degradation against upper=0 and the worst interval=1.

    ``values`` are signed upper-relative deltas.  Heatmaps and Retention pass
    ``clamp_better_than_upper=True`` so improvements cannot offset degradation;
    line plots pass ``False`` so improvements remain visible below zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.array(arr, copy=True)
    finite = np.isfinite(out)
    if clamp_better_than_upper:
        out[finite] = np.maximum(out[finite], 0.0)
    upper = float(maximum)
    if not math.isfinite(upper) or upper <= 0.0:
        out[finite] = 0.0
    else:
        out[finite] = out[finite] / upper
    return out


def normalize_raw_metric_against_upper_worst(
    values: np.ndarray,
    *,
    upper_baseline: float,
    worst_value: float,
    clamp_better_than_upper: bool,
) -> np.ndarray:
    """Apply the shared upper-zero / all-interval-worst normalization."""
    arr = np.asarray(values, dtype=np.float64)
    upper = float(upper_baseline)
    worst = float(worst_value)
    signed_delta = arr - upper
    return normalize_canonical_degradation(
        signed_delta,
        maximum=max(0.0, worst - upper),
        clamp_better_than_upper=clamp_better_than_upper,
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
    metric_names: Sequence[str] = GT_METRIC_KEYS,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    normalized: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for metric in metric_names:
        if metric not in components:
            continue
        values = np.asarray(components[metric], dtype=np.float64)
        finite = np.isfinite(values)
        maximum = (
            float(np.nanmax(np.maximum(values[finite], 0.0)))
            if np.any(finite)
            else float("nan")
        )
        normalized[metric] = normalize_canonical_degradation(
            values,
            maximum=maximum,
            clamp_better_than_upper=True,
        )
        records[metric] = {
            "method": "upper_zero_all_interval_worst",
            "min": 0.0,
            "max": maximum if math.isfinite(maximum) else None,
            "constant": bool(not math.isfinite(maximum) or maximum <= 0.0),
        }
    if not normalized:
        raise ValueError("No metric components available for normalization")
    stack = np.stack([normalized[name] for name in metric_names if name in normalized], axis=0)
    with np.errstate(all="ignore"):
        aggregate = np.nanmean(stack, axis=0)
    aggregate[~np.any(np.isfinite(stack), axis=0)] = np.nan
    return normalized, aggregate, {
        "method": "per_metric_upper_zero_all_interval_worst_then_nanmean",
        "metrics": records,
    }

"""Rebase remaining-skip metrics against the exact base prune-plan result."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from analysis.prune_plan import interval_selector
from analysis.redundancy.metrics import (
    mask_unevaluated_upper_spans,
    upper_packed_to_symmetric_matrix,
    upper_triangle_pack,
)


def resolve_plan_interval(
    plan: Any,
    *,
    full_unit_ids: Sequence[str],
    expected_dropped_unit_ids: Sequence[str] | None = None,
) -> Tuple[int, int, str]:
    """Resolve one contiguous prune plan to its exact brute-force interval."""
    full_ids = [str(unit_id) for unit_id in full_unit_ids]
    dropped = sorted({int(index) for index in plan.drop})
    if not dropped:
        raise ValueError("Base prune plan drops no units")
    if dropped[0] < 0 or dropped[-1] >= len(full_ids):
        raise IndexError(
            f"Base prune plan indices {dropped} exceed the {len(full_ids)}-unit catalog"
        )
    expected_range = list(range(dropped[0], dropped[-1] + 1))
    if dropped != expected_range:
        raise ValueError(
            "Remaining-skip marginal visualization requires a contiguous base "
            f"prune plan, got indices {dropped}"
        )
    resolved_ids = [full_ids[index] for index in dropped]
    if expected_dropped_unit_ids is not None:
        expected_ids = [str(unit_id) for unit_id in expected_dropped_unit_ids]
        if resolved_ids != expected_ids:
            raise ValueError(
                "Base prune plan unit IDs do not match remaining-skip provenance: "
                f"plan={resolved_ids}, provenance={expected_ids}"
            )
    start, end = dropped[0], dropped[-1]
    return start, end, interval_selector(start, end)


def _mode_delta_maps(report: Mapping[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    output: Dict[str, Dict[str, List[float]]] = {}
    try:
        num_layers = int(report.get("num_layers", 0))
        max_span = int(report.get("max_span", 0))
    except (TypeError, ValueError):
        num_layers = 0
        max_span = 0
    fusion = report.get("fusion_by_mode")
    if not isinstance(fusion, Mapping):
        return output
    for mode, payload in fusion.items():
        if not isinstance(payload, Mapping):
            continue
        metric_maps = payload.get("gt_delta_upper_by_metric")
        if not isinstance(metric_maps, Mapping):
            continue
        clean: Dict[str, List[float]] = {}
        for metric, packed in metric_maps.items():
            if not isinstance(packed, list):
                continue
            values = [
                float(value) if value is not None else float("nan")
                for value in packed
            ]
            clean[str(metric)] = mask_unevaluated_upper_spans(
                values,
                num_layers=num_layers,
                max_span=max_span,
            )
        if clean:
            output[str(mode)] = clean
    return output


def build_marginal_skip_maps(
    *,
    original_report: Mapping[str, Any],
    remaining_report: Mapping[str, Any],
    base_interval: Tuple[int, int],
) -> Dict[str, Dict[str, List[float]]]:
    """Return compact remaining-unit maps rebased to the base-pruned model."""
    full_ids = [str(unit_id) for unit_id in original_report.get("unit_ids", [])]
    remaining_ids = [
        str(unit_id) for unit_id in remaining_report.get("unit_ids", [])
    ]
    if not full_ids or not remaining_ids:
        raise ValueError("Original/remaining reports must both contain unit_ids")
    original_delta = _mode_delta_maps(original_report)
    remaining_delta = _mode_delta_maps(remaining_report)
    original_upper = original_report.get("upper_average")
    remaining_upper = remaining_report.get("upper_average")
    if not isinstance(original_upper, Mapping) or not isinstance(
        remaining_upper, Mapping
    ):
        raise ValueError("Original/remaining reports must both contain upper_average")

    start, end = (int(base_interval[0]), int(base_interval[1]))
    output: Dict[str, Dict[str, List[float]]] = {}
    for mode in sorted(set(original_delta).intersection(remaining_delta)):
        metrics: Dict[str, List[float]] = {}
        for metric in sorted(
            set(original_delta[mode]).intersection(remaining_delta[mode])
        ):
            if metric not in original_upper or metric not in remaining_upper:
                continue
            full_upper = float(original_upper[metric])
            remaining_full_upper = float(remaining_upper[metric])
            if not np.isclose(
                full_upper,
                remaining_full_upper,
                rtol=1e-6,
                atol=1e-8,
                equal_nan=False,
            ):
                raise ValueError(
                    f"Full-model baseline mismatch for {metric}: "
                    f"pruning={full_upper}, remaining={remaining_full_upper}"
                )
            original_matrix = np.asarray(
                upper_packed_to_symmetric_matrix(
                    original_delta[mode][metric], len(full_ids)
                ),
                dtype=np.float64,
            )
            base_delta = float(original_matrix[start, end])
            if not np.isfinite(base_delta):
                raise ValueError(
                    f"Base pruning result ({start},{end}) is missing for {metric}"
                )
            compact_delta = np.asarray(
                upper_packed_to_symmetric_matrix(
                    remaining_delta[mode][metric], len(remaining_ids)
                ),
                dtype=np.float64,
            )
            marginal_delta = compact_delta - base_delta
            rebased = marginal_delta
            metrics[metric] = upper_triangle_pack(
                torch.from_numpy(np.asarray(rebased, dtype=np.float64))
            )
        if metrics:
            output[str(mode)] = metrics
    return output

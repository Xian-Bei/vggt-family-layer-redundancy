"""Closed-interval selection for pruning score matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


CORNER_RANKING_TOLERANCE = 0.0
REQUIRE_ENDPOINT_LINKED_SPANS = True


@dataclass(frozen=True)
class TrianglePick:
    """One zero-based closed interval ``[i, j]`` selected for pruning.

    ``span`` is always the number of dropped units, ``j - i + 1``.
    """

    i: int
    j: int
    span: int
    unit_i: str
    unit_j: str
    dropped_unit_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        start = int(self.i)
        end = int(self.j)
        expected_span = end - start + 1
        if start < 0 or end < start:
            raise ValueError(f"invalid closed interval [{start}, {end}]")
        if int(self.span) != expected_span:
            raise ValueError(
                f"closed interval [{start}, {end}] has span={self.span}; "
                f"expected drop_count={expected_span}"
            )
        if len(self.dropped_unit_ids) != expected_span:
            raise ValueError(
                f"closed interval [{start}, {end}] has {len(self.dropped_unit_ids)} "
                f"dropped_unit_ids; expected {expected_span}"
            )


def interval_pick_from_bounds(
    unit_ids: Sequence[str],
    start: int,
    end: int,
) -> TrianglePick:
    """Build a validated pick for the closed interval ``[start, end]``."""
    start_i = int(start)
    end_i = int(end)
    n = len(unit_ids)
    if start_i < 0 or end_i < start_i or end_i >= n:
        raise ValueError(f"invalid closed interval [{start_i}, {end_i}] for {n} units")
    return TrianglePick(
        i=start_i,
        j=end_i,
        span=end_i - start_i + 1,
        unit_i=str(unit_ids[start_i]),
        unit_j=str(unit_ids[end_i]),
        dropped_unit_ids=tuple(str(unit_ids[k]) for k in range(start_i, end_i + 1)),
    )


def scan_endpoint_linked_closed_intervals_with_candidates(
    score: np.ndarray,
    unit_ids: Sequence[str],
    *,
    ranking_tolerance: float = CORNER_RANKING_TOLERANCE,
    require_endpoint_linked_spans: bool = REQUIRE_ENDPOINT_LINKED_SPANS,
) -> Tuple[
    List[TrianglePick],
    Dict[int, Tuple[TrianglePick, ...]],
]:
    """Select one closed interval per available drop count.

    ``score[start, end]`` is the score for deleting every unit in the closed
    interval ``[start, end]``. Candidates are grouped by
    ``drop_count = end - start + 1``. When endpoint linking is enabled, each
    smaller selected interval must have at least one endpoint inside the
    previously selected larger interval.
    """
    n = len(unit_ids)
    if score.shape != (n, n):
        raise ValueError(f"score shape {score.shape} != ({n}, {n}) for {n} unit_ids")
    values = np.asarray(score, dtype=np.float64)
    rank_tol = float(ranking_tolerance)
    candidates_by_drop_count: dict[int, List[TrianglePick]] = {}
    # Legality comes entirely from finite score cells.  Pairwise similarity
    # conversion leaves C[0, :] as NaN because unit zero has no prior unit
    # output, so intervals deleting unit zero are excluded structurally.
    for drop_count in range(1, n + 1):
        candidates: List[TrianglePick] = []
        for start in range(0, n - drop_count + 1):
            end = start + drop_count - 1
            if not np.isfinite(float(values[start, end])):
                continue
            candidates.append(interval_pick_from_bounds(unit_ids, start, end))
        if candidates:
            candidates_by_drop_count[drop_count] = candidates
    if not candidates_by_drop_count:
        return [], {}

    def best_by_score(candidates: Sequence[TrianglePick]) -> TrianglePick:
        best = candidates[0]
        best_score = float(values[int(best.i), int(best.j)])
        for candidate in candidates[1:]:
            score_value = float(values[int(candidate.i), int(candidate.j)])
            if score_value > best_score + rank_tol:
                best = candidate
                best_score = score_value
        return best

    selected_by_drop_count: dict[int, TrianglePick] = {}
    eligible_by_drop_count: dict[int, Tuple[TrianglePick, ...]] = {}
    current: Optional[TrianglePick] = None
    for drop_count in sorted(candidates_by_drop_count, reverse=True):
        candidates = candidates_by_drop_count[drop_count]
        if current is None or not bool(require_endpoint_linked_spans):
            eligible = candidates
        else:
            current_i = int(current.i)
            current_j = int(current.j)
            eligible = [
                candidate
                for candidate in candidates
                if current_i <= int(candidate.i) <= current_j
                or current_i <= int(candidate.j) <= current_j
            ]
            if not eligible:
                continue
        selected = best_by_score(eligible)
        selected_by_drop_count[drop_count] = selected
        eligible_by_drop_count[drop_count] = tuple(eligible)
        current = selected

    picks = [
        selected_by_drop_count[drop_count]
        for drop_count in sorted(selected_by_drop_count)
    ]
    return picks, eligible_by_drop_count

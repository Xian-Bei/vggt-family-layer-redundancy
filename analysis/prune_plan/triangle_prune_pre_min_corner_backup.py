"""Dense upper-triangle greedy selection for prune plans from CKA similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.schema.v3 import PruneAction


def triangle_span(i: int, j: int) -> int:
    """Edge length of T(i,j): number of unit indices from i through j (j - i)."""
    return max(0, int(j) - int(i))


def triangle_area(i: int, j: int) -> int:
    """Number of strict upper-tri cells in T(i,j); monotone in ``triangle_span``."""
    span = triangle_span(i, j)
    if span <= 0:
        return 0
    return span * (span + 1) // 2


def build_strict_upper_mask(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Bool mask for i < j where sim[i,j] > threshold."""
    n = int(sim.shape[0])
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            mask[i, j] = float(sim[i, j]) > float(threshold)
    return mask


def clear_triangle_mask(mask: np.ndarray, i: int, j: int) -> None:
    for a in range(i, j + 1):
        for b in range(a + 1, j + 1):
            mask[a, b] = False


def is_dense_triangle(mask: np.ndarray, i: int, j: int) -> bool:
    if j <= i:
        return False
    for a in range(i, j + 1):
        for b in range(a + 1, j + 1):
            if not mask[a, b]:
                return False
    return True


def find_largest_dense_triangle(
    mask: np.ndarray,
    sim: np.ndarray,
) -> Optional[Tuple[int, int, int]]:
    """Return (i, j, span) for the largest dense triangle, or None.

    Primary key: maximum span ``j - i``. Tie-break: largest ``sim[i, j]`` (corner).
    """
    n = int(mask.shape[0])
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != mask shape {(n, n)}")
    best: Optional[Tuple[int, int, int]] = None
    best_span = 0
    best_corner = -float("inf")

    for i in range(n):
        dense = True
        for j in range(i + 1, n):
            if dense:
                for a in range(i, j):
                    if not mask[a, j]:
                        dense = False
                        break
            if not dense:
                break
            span = j - i
            corner = float(sim[i, j])
            if span > best_span or (span == best_span and corner > best_corner):
                best_span = span
                best_corner = corner
                best = (i, j, span)
    return best


def find_dense_triangle_for_span(
    mask: np.ndarray,
    sim: np.ndarray,
    span: int,
) -> Optional[Tuple[int, int, int]]:
    """Return the dense triangle of exactly ``span`` with the largest corner value."""
    n = int(mask.shape[0])
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != mask shape {(n, n)}")
    target_span = int(span)
    if target_span <= 0 or target_span >= n:
        return None

    best: Optional[Tuple[int, int, int]] = None
    best_corner = -float("inf")
    for i in range(0, n - target_span):
        j = i + target_span
        if not is_dense_triangle(mask, i, j):
            continue
        corner = float(sim[i, j])
        if corner > best_corner:
            best_corner = corner
            best = (i, j, target_span)
    return best


@dataclass(frozen=True)
class TrianglePick:
    i: int
    j: int
    span: int
    unit_i: str
    unit_j: str
    dropped_unit_ids: Tuple[str, ...]

    @property
    def area(self) -> int:
        return triangle_area(self.i, self.j)


def dense_triangle_for_span(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    *,
    threshold: float,
    span: int,
) -> Optional[TrianglePick]:
    """Select one exact-span dense triangle, breaking ties by corner similarity."""
    n = len(unit_ids)
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != ({n}, {n}) for {n} unit_ids")
    mask = build_strict_upper_mask(np.asarray(sim, dtype=np.float64), threshold)
    found = find_dense_triangle_for_span(mask, sim, span)
    if found is None:
        return None
    i, j, actual_span = found
    return TrianglePick(
        i=i,
        j=j,
        span=actual_span,
        unit_i=str(unit_ids[i]),
        unit_j=str(unit_ids[j]),
        dropped_unit_ids=tuple(unit_ids[k] for k in range(i + 1, j + 1)),
    )


def scan_best_dense_triangles(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    *,
    threshold: float,
) -> List[TrianglePick]:
    """Return the best dense triangle for each consecutive valid span.

    The scan reuses the dense state for span ``s - 1`` and precomputes whether
    each new right-boundary column is entirely above ``threshold``.  This keeps
    the full scan quadratic in the number of units.  Once a span has no dense
    candidate, no larger dense triangle can exist, so the scan stops.
    """
    n = len(unit_ids)
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != ({n}, {n}) for {n} unit_ids")
    values = np.asarray(sim, dtype=np.float64)
    mask = build_strict_upper_mask(values, threshold)

    # column_all[i, j] means mask[i:j, j] is entirely true.
    column_all = np.zeros((n, n), dtype=bool)
    for j in range(1, n):
        running = True
        for i in range(j - 1, -1, -1):
            running = running and bool(mask[i, j])
            column_all[i, j] = running

    dense = np.zeros((n, n), dtype=bool)
    picks: List[TrianglePick] = []
    for span in range(1, n):
        best_i: Optional[int] = None
        best_corner = -float("inf")
        for i in range(0, n - span):
            j = i + span
            previous_dense = span == 1 or bool(dense[i, j - 1])
            dense[i, j] = previous_dense and bool(column_all[i, j])
            if not dense[i, j]:
                continue
            corner = float(values[i, j])
            if corner > best_corner:
                best_i = i
                best_corner = corner
        if best_i is None:
            break
        j = best_i + span
        picks.append(
            TrianglePick(
                i=best_i,
                j=j,
                span=span,
                unit_i=str(unit_ids[best_i]),
                unit_j=str(unit_ids[j]),
                dropped_unit_ids=tuple(unit_ids[k] for k in range(best_i + 1, j + 1)),
            )
        )
    return picks


def _triangle_pick_from_indices(
    unit_ids: Sequence[str],
    i: int,
    j: int,
) -> TrianglePick:
    return TrianglePick(
        i=int(i),
        j=int(j),
        span=int(j) - int(i),
        unit_i=str(unit_ids[int(i)]),
        unit_j=str(unit_ids[int(j)]),
        dropped_unit_ids=tuple(unit_ids[k] for k in range(int(i) + 1, int(j) + 1)),
    )


def scan_nested_dense_triangles_from_max(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    *,
    threshold: float,
) -> List[TrianglePick]:
    """Return one nested dense-triangle chain for spans 1..max-valid-span.

    The largest valid span is selected with the same rule as
    :func:`scan_best_dense_triangles`: among all dense triangles of that maximum
    span, use the one with the largest upper-right corner similarity.  Smaller
    spans are then obtained by shrinking that interval from right to left: for
    each span ``s - 1``, compare the two dense sub-triangles ``[i + 1, j]`` and
    ``[i, j - 1]`` and keep the one with the larger corner similarity.

    This guarantees every smaller selected interval is a subset of every larger
    selected interval, so budget choices on the resulting curve are nested.
    """
    n = len(unit_ids)
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != ({n}, {n}) for {n} unit_ids")
    values = np.asarray(sim, dtype=np.float64)
    independent = scan_best_dense_triangles(values, unit_ids, threshold=threshold)
    if not independent:
        return []

    current_i = int(independent[-1].i)
    current_j = int(independent[-1].j)
    by_span: dict[int, TrianglePick] = {
        int(independent[-1].span): _triangle_pick_from_indices(unit_ids, current_i, current_j)
    }

    while current_j - current_i > 1:
        left_i = current_i + 1
        left_j = current_j
        right_i = current_i
        right_j = current_j - 1
        left_corner = float(values[left_i, left_j])
        right_corner = float(values[right_i, right_j])
        if left_corner > right_corner:
            current_i, current_j = left_i, left_j
        else:
            current_i, current_j = right_i, right_j
        by_span[current_j - current_i] = _triangle_pick_from_indices(
            unit_ids,
            current_i,
            current_j,
        )

    return [by_span[span] for span in sorted(by_span)]


def greedy_dense_triangles(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    *,
    threshold: float,
    max_triangles: int,
) -> Tuple[List[TrianglePick], np.ndarray]:
    """Select up to ``max_triangles`` dense triangles on CKA strict upper triangle."""
    n = len(unit_ids)
    if sim.shape != (n, n):
        raise ValueError(f"sim shape {sim.shape} != ({n}, {n}) for {n} unit_ids")

    mask = build_strict_upper_mask(np.asarray(sim, dtype=np.float64), threshold)
    picks: List[TrianglePick] = []

    for _ in range(max(0, int(max_triangles))):
        found = find_largest_dense_triangle(mask, sim)
        if found is None:
            break
        i, j, span = found
        dropped = tuple(unit_ids[k] for k in range(i + 1, j + 1))
        picks.append(
            TrianglePick(
                i=i,
                j=j,
                span=span,
                unit_i=str(unit_ids[i]),
                unit_j=str(unit_ids[j]),
                dropped_unit_ids=dropped,
            )
        )
        clear_triangle_mask(mask, i, j)

    return picks, mask


def picks_to_actions(
    picks: Sequence[TrianglePick],
    *,
    lora_rank: int,
    lora_alpha: float,
) -> List[PruneAction]:
    actions: List[PruneAction] = []
    lora_units: set[str] = set()

    for pick in picks:
        for uid in pick.dropped_unit_ids:
            actions.append(PruneAction(action="drop_unit", unit_id=uid))
        if pick.unit_i not in lora_units:
            lora_units.add(pick.unit_i)
            actions.append(
                PruneAction(
                    action="lora_unit",
                    unit_id=pick.unit_i,
                    rank=int(lora_rank),
                    alpha=float(lora_alpha),
                )
            )
    return actions


def picks_to_extra(picks: Sequence[TrianglePick]) -> Dict[str, Any]:
    return {
        "triangle_prune": {
            "triangles": [
                {
                    "i": p.i,
                    "j": p.j,
                    "span": p.span,
                    "area": p.area,
                    "unit_i": p.unit_i,
                    "unit_j": p.unit_j,
                    "dropped_unit_ids": list(p.dropped_unit_ids),
                }
                for p in picks
            ]
        }
    }


def build_triangle_prune_actions(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    *,
    threshold: float,
    max_triangles: int,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
) -> Tuple[List[PruneAction], Dict[str, Any], List[TrianglePick]]:
    picks, _ = greedy_dense_triangles(
        sim, unit_ids, threshold=threshold, max_triangles=max_triangles
    )
    actions = picks_to_actions(picks, lora_rank=lora_rank, lora_alpha=lora_alpha)
    return actions, picks_to_extra(picks), picks

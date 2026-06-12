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

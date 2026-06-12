"""Shared frame-count capping after dataset-specific sampling strategies."""

from __future__ import annotations

from typing import Any, List, Sequence, TypeVar

import torch

T = TypeVar("T")


def pick_frame_indices(num_frames: int, max_frames: int) -> List[int]:
    """Uniformly subsample ``num_frames`` down to at most ``max_frames`` indices."""
    if num_frames <= 0:
        return []
    if int(max_frames) <= 0 or num_frames <= int(max_frames):
        return list(range(num_frames))
    return torch.linspace(0, num_frames - 1, int(max_frames)).round().long().tolist()


def cap_sequence(items: Sequence[T], max_frames: int) -> List[T]:
    """Cap a sequence to ``max_frames`` using uniform index selection."""
    pick = pick_frame_indices(len(items), max_frames)
    return [items[i] for i in pick]


def cap_7andn_views(views: List[Any], max_frames: int) -> List[Any]:
    """Cap 7scenes/NRGBD view dicts after kf_every subsampling."""
    if not views or not isinstance(views[0], dict):
        return list(views)
    return cap_sequence(views, max_frames)

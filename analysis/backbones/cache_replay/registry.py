"""Registry for cache-first replay training engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import CacheReplayEngine


def get_cache_replay_engine(backbone: str) -> "CacheReplayEngine":
    from analysis.backbones.registry import get_backbone

    return get_backbone(backbone).cache_replay_engine()

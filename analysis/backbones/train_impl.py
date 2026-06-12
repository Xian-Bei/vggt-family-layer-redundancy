"""Backbone train-from-cache launcher (delegates to shared cache-replay trainer)."""

from __future__ import annotations

from analysis.backbones.contracts import TrainFromCacheArgs
from analysis.backbones.train_from_cache import launch_cache_replay_training


def train_vggt_family_cache_replay(adapter_name: str, args: TrainFromCacheArgs) -> None:
    launch_cache_replay_training(adapter_name, args)

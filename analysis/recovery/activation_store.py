"""File-backed activation access for recovery fitting.

The store deliberately keeps scene tensors sharded.  It provides the same
global row indexing as concatenating shards, without ever materialising that
concatenation in anonymous memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch


def rows_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(
            f"expected 4D unit tensor [B,S,P,C], got {tuple(tensor.shape)}"
        )
    return tensor.reshape(-1, int(tensor.shape[-1]))


DesiredBuilder = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class RowShard:
    scene_id: str
    inputs: torch.Tensor
    desired: Optional[torch.Tensor] = None
    desired_builder: Optional[DesiredBuilder] = None

    def __post_init__(self) -> None:
        if self.inputs.dim() != 2:
            raise ValueError(f"RowShard inputs must be 2D, got {self.inputs.shape}")
        if (self.desired is None) == (self.desired_builder is None):
            raise ValueError("RowShard needs exactly one desired source")
        if self.desired is not None and self.desired.shape != self.inputs.shape:
            raise ValueError(
                f"RowShard mismatch: {self.inputs.shape} vs {self.desired.shape}"
            )

    @property
    def num_rows(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def channels(self) -> int:
        return int(self.inputs.shape[1])

    def gather_desired(self, local_indices: torch.Tensor) -> torch.Tensor:
        if self.desired is not None:
            return self.desired[local_indices]
        assert self.desired_builder is not None
        return self.desired_builder(local_indices)


class ShardedRowStore:
    """A logical row concat backed by independent scene tensors."""

    def __init__(self, shards: Sequence[RowShard]) -> None:
        self.shards = list(shards)
        if not self.shards:
            raise RuntimeError("ShardedRowStore requires at least one shard")
        channels = self.shards[0].channels
        if any(shard.channels != channels for shard in self.shards):
            raise ValueError("All row shards must have the same channel count")
        offsets = [0]
        for shard in self.shards:
            offsets.append(offsets[-1] + shard.num_rows)
        self.offsets = offsets
        self.channels = channels
        self.num_rows = offsets[-1]
        self._buffers: Dict[Tuple[int, bool], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._buffer_turn: Dict[Tuple[int, bool], int] = {}
        self._buffer_events: Dict[Tuple[int, bool, int], torch.cuda.Event] = {}
        self._last_slot: Optional[Tuple[int, bool, int]] = None

    def __len__(self) -> int:
        return self.num_rows

    def iter_scenes(self) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Yield one complete scene; derived targets exist only for this yield."""
        for shard in self.shards:
            local = torch.arange(shard.num_rows, dtype=torch.long)
            yield shard.scene_id, shard.inputs, shard.gather_desired(local)

    def _staging_pair(
        self, capacity: int, *, pin_memory: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(capacity), bool(pin_memory))
        pairs = self._buffers.get(key)
        if pairs is None:
            pairs = [
                (
                    torch.empty(
                        (capacity, self.channels),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                    torch.empty(
                        (capacity, self.channels),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                )
                for _ in range(2)
            ]
            self._buffers[key] = pairs
            self._buffer_turn[key] = 0
        turn = self._buffer_turn[key]
        self._buffer_turn[key] = 1 - turn
        event = self._buffer_events.get((key[0], key[1], turn))
        if event is not None and not event.query():
            event.synchronize()
        self._last_slot = (key[0], key[1], turn)
        return pairs[turn]

    def record_transfer(self) -> None:
        """Mark the last pinned slot reusable after current-stream H2D copies."""
        if self._last_slot is None or not self._last_slot[1]:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._buffer_events[self._last_slot] = event

    def gather(
        self,
        indices: torch.Tensor,
        *,
        pin_memory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather global indices while preserving their exact requested order."""
        cpu_indices = indices.detach().to(device="cpu", dtype=torch.long)
        count = int(cpu_indices.numel())
        if count == 0:
            empty = torch.empty((0, self.channels), dtype=torch.float32)
            return empty, empty.clone()
        if int(cpu_indices.min()) < 0 or int(cpu_indices.max()) >= self.num_rows:
            raise IndexError("row index is outside ShardedRowStore")
        xb, yb = self._staging_pair(count, pin_memory=bool(pin_memory))
        # Assign into the positions occupied by each shard.  This matches
        # all_x[indices] even when indices are non-monotonic.
        for shard_index, shard in enumerate(self.shards):
            start, end = self.offsets[shard_index : shard_index + 2]
            mask = (cpu_indices >= start) & (cpu_indices < end)
            if not bool(mask.any()):
                continue
            positions = mask.nonzero(as_tuple=False).flatten()
            local = cpu_indices[positions] - start
            xb[positions] = shard.inputs[local].to(torch.float32)
            yb[positions] = shard.gather_desired(local).to(torch.float32)
        return xb[:count], yb[:count]

    def first(self, count: int, *, pin_memory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gather(
            torch.arange(min(int(count), self.num_rows), dtype=torch.long),
            pin_memory=pin_memory,
        )

    def close(self) -> None:
        self._buffers.clear()
        self._buffer_turn.clear()
        self._buffer_events.clear()
        self._last_slot = None
        self.shards.clear()


class RecoveryActivationStore(ShardedRowStore):
    """Semantic alias used by recovery code and future session registries."""


class RecoverySession:
    """Per-backbone registry shared by multi-request recovery execution."""

    def __init__(self, backbone: str) -> None:
        self.backbone = str(backbone)
        self.manifests: Dict[Path, Any] = {}
        self.mmap_units: Dict[Tuple[Path, str, str], torch.Tensor] = {}
        self.sufficient_statistics: Dict[Tuple[Any, ...], Any] = {}
        self.validated_prefixes: Dict[Tuple[Any, ...], Any] = {}

    def manifest(self, cache_dir: Path) -> Any:
        from analysis.redundancy.cache_layout_v3 import load_v3_manifest

        key = Path(cache_dir).resolve()
        if key not in self.manifests:
            self.manifests[key] = load_v3_manifest(key)
        return self.manifests[key]

    def unit(
        self, cache_dir: Path, item: Dict[str, Any], unit_id: str
    ) -> torch.Tensor:
        from analysis.redundancy.cache_layout_v3 import load_scene_unit_v3

        root = Path(cache_dir).resolve()
        scene = str(item.get("scene_id", item.get("dir", "")))
        key = (root, scene, str(unit_id))
        if key not in self.mmap_units:
            self.mmap_units[key] = load_scene_unit_v3(
                root, item, unit_id, mmap=True, map_location="cpu"
            )
        return self.mmap_units[key]

    @property
    def mapped_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.mmap_units.values()
        )

    def close(self) -> None:
        self.mmap_units.clear()
        self.manifests.clear()
        self.sufficient_statistics.clear()
        self.validated_prefixes.clear()

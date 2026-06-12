"""Dataset plugin contracts for v3 evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from analysis.schema.v3 import DatasetEvalProfile


@dataclass
class EvalSampleBatch:
    scene_id: str
    images: Any
    gt_depth: Optional[np.ndarray] = None
    gt_points: Optional[np.ndarray] = None
    gt_w2c: Optional[np.ndarray] = None
    gt_c2w: Optional[np.ndarray] = None
    frame_ids: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class DatasetPlugin(ABC):
    name: str

    @abstractmethod
    def eval_profile(self) -> DatasetEvalProfile:
        ...

    @abstractmethod
    def iter_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        max_frames: int,
        split: str | None = None,
    ) -> Iterator[EvalSampleBatch]:
        ...

    @abstractmethod
    def load_scene_for_cache(
        self,
        data_root: Path,
        scene_id: str,
        max_frames: int,
        *,
        split: str | None = None,
    ) -> EvalSampleBatch:
        ...

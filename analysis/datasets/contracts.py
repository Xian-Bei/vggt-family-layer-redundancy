"""Dataset adaptor contracts for v3 evaluation."""

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


@dataclass
class PreparedSkipEvalScene:
    """Scene-local inputs and GT state reused across all skip spans."""

    scene_id: str
    images: Any
    state: Dict[str, Any] = field(default_factory=dict)


class DatasetAdaptor(ABC):
    name: str

    def validate_split(self, split: str | None = None) -> str | None:
        """Validate and normalize an optional dataset split."""
        return None if split is None else str(split).strip().lower()

    def output_tag(self, split: str | None = None) -> str:
        """On-disk dataset tag for this dataset/split."""
        self.validate_split(split)
        return str(self.name)

    def count_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        split: str | None = None,
    ) -> int | None:
        """Fast scene count for progress reporting.

        Adaptors should override this when they can count scenes without loading
        image tensors. Returning ``None`` keeps callers generic.
        """
        del data_root, max_scenes, split
        return None

    def evaluate_scene(
        self,
        backbone_adaptor: Any,
        model: Any,
        batch: EvalSampleBatch,
        ctx: Any,
    ) -> Dict[str, float]:
        del backbone_adaptor, model, batch, ctx
        raise NotImplementedError(f"{self.name} does not implement evaluate_scene")

    def prepare_skip_eval_scene(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: Any,
    ) -> PreparedSkipEvalScene:
        del backbone_adaptor, batch, ctx
        raise NotImplementedError(f"{self.name} does not implement skip-eval scene preparation")

    def score_skip_eval_prediction(
        self,
        backbone_adaptor: Any,
        prediction: Any,
        prepared: PreparedSkipEvalScene,
        ctx: Any,
    ) -> Dict[str, float]:
        del backbone_adaptor, prediction, prepared, ctx
        raise NotImplementedError(f"{self.name} does not implement skip-eval prediction scoring")

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

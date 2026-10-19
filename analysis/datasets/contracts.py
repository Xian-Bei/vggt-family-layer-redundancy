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
class PreparedEvalScene:
    """Scene-local model images and CPU GT state reused during evaluation."""

    scene_id: str
    images: Any
    state: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible name used by skip-eval callers and external adaptors.
PreparedSkipEvalScene = PreparedEvalScene


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

    def prepare_eval_scene_cpu(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: Any,
    ) -> PreparedEvalScene:
        del backbone_adaptor, batch, ctx
        raise NotImplementedError(
            f"{self.name} does not implement CPU eval scene preparation"
        )

    def prepare_inference_scene_cpu(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
    ) -> PreparedEvalScene:
        """Load only model inputs for inference benchmarking, without GT state."""
        del backbone_adaptor, batch
        raise NotImplementedError(
            f"{self.name} does not implement CPU inference scene preparation"
        )

    def score_eval_prediction(
        self,
        backbone_adaptor: Any,
        prediction: Any,
        prepared: PreparedEvalScene,
        ctx: Any,
    ) -> Dict[str, float]:
        del backbone_adaptor, prediction, prepared, ctx
        raise NotImplementedError(
            f"{self.name} does not implement eval prediction scoring"
        )

    def prepare_skip_eval_scene(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: Any,
    ) -> PreparedEvalScene:
        """Compatibility wrapper that stages the CPU-prepared images synchronously."""
        prepared = self.prepare_eval_scene_cpu(backbone_adaptor, batch, ctx)
        images = prepared.images
        if not hasattr(images, "to"):
            raise TypeError(f"{self.name} returned non-tensor eval images")
        prepared.images = images.to(ctx.device)
        return prepared

    def score_skip_eval_prediction(
        self,
        backbone_adaptor: Any,
        prediction: Any,
        prepared: PreparedEvalScene,
        ctx: Any,
    ) -> Dict[str, float]:
        """Compatibility wrapper for the shared evaluation scorer."""
        return self.score_eval_prediction(backbone_adaptor, prediction, prepared, ctx)

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

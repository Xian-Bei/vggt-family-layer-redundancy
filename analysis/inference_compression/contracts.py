"""Public contract for deployment-only model compression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet

import torch.nn as nn


@dataclass(frozen=True)
class InferenceCompressionConfig:
    """Normalized configuration passed to an inference compression adapter."""

    name: str = "none"
    fastvggt_merge_start: int = 0
    fastvggt_merge_ratio: float = 0.9
    quantvggt_dtype: str = "w4a4"
    quantvggt_params_dir: Path | None = None

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {"name": self.name}
        if self.name == "fastvggt":
            record.update(
                merge_start=int(self.fastvggt_merge_start),
                merge_ratio=float(self.fastvggt_merge_ratio),
            )
        elif self.name == "quantvggt":
            record.update(
                dtype=str(self.quantvggt_dtype),
                params_dir=(
                    str(self.quantvggt_params_dir.resolve())
                    if self.quantvggt_params_dir is not None
                    else None
                ),
            )
        return record


class InferenceCompressionAdapter(ABC):
    """Transform a restored backbone model for deployment-time inference."""

    name: str
    supported_backbones: FrozenSet[str] = frozenset()

    def validate_backbone(self, backbone_name: str) -> None:
        key = str(backbone_name).strip().lower()
        if key not in self.supported_backbones:
            raise ValueError(
                f"Inference compression {self.name!r} does not support backbone {backbone_name!r}; "
                f"supported={sorted(self.supported_backbones)}"
            )

    def validate_config(self, config: InferenceCompressionConfig) -> None:
        """Validate method-specific options before loading a large model."""
        del config

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        *,
        backbone_name: str,
        config: InferenceCompressionConfig,
    ) -> nn.Module:
        """Apply this deployment transformation in place and return the model."""

    @abstractmethod
    def output_tag(self, config: InferenceCompressionConfig) -> str:
        """Return a stable filesystem-safe tag for this configuration."""

"""Standalone adapter that folds LaRe LoRA weights into ordinary Linear layers."""

from __future__ import annotations

import torch.nn as nn

from analysis.inference_compression.contracts import (
    InferenceCompressionAdapter,
    InferenceCompressionConfig,
)
from analysis.inference_compression.lora import merge_lora_linears


class LoRAMergeInferenceAdapter(InferenceCompressionAdapter):
    name = "lora_merge"
    supported_backbones = frozenset({"vggt"})

    def apply(
        self,
        model: nn.Module,
        *,
        backbone_name: str,
        config: InferenceCompressionConfig,
    ) -> nn.Module:
        self.validate_backbone(backbone_name)
        model.eval()
        merged_count = merge_lora_linears(model)
        record = config.as_record()
        record["merged_lora_linears"] = merged_count
        model._inference_compression = record
        return model

    def output_tag(self, config: InferenceCompressionConfig) -> str:
        del config
        return self.name

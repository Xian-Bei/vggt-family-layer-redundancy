"""Inference-time compression adapters applied after LaRe model restoration."""

from .contracts import InferenceCompressionAdapter, InferenceCompressionConfig
from .lora import merge_lora_linears
from .registry import get_inference_compression, list_inference_compressions

__all__ = [
    "InferenceCompressionAdapter",
    "InferenceCompressionConfig",
    "get_inference_compression",
    "list_inference_compressions",
    "merge_lora_linears",
]

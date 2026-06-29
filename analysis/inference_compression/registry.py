"""Registry for inference-time compression adapters."""

from __future__ import annotations

from typing import Dict

from .contracts import InferenceCompressionAdapter

_REGISTRY: Dict[str, InferenceCompressionAdapter] = {}


def register(adapter: InferenceCompressionAdapter) -> InferenceCompressionAdapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def _load_defaults() -> None:
    if _REGISTRY:
        return
    from .FastVGGT.adapter import FastVGGTInferenceAdapter
    from .LoRAMerge.adapter import LoRAMergeInferenceAdapter
    from .QuantVGGT.adapter import QuantVGGTInferenceAdapter

    register(FastVGGTInferenceAdapter())
    register(LoRAMergeInferenceAdapter())
    register(QuantVGGTInferenceAdapter())


def get_inference_compression(name: str) -> InferenceCompressionAdapter:
    key = str(name).strip().lower()
    _load_defaults()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown inference compression {name!r}. Available: none, {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]


def list_inference_compressions() -> list[str]:
    _load_defaults()
    return sorted(_REGISTRY)

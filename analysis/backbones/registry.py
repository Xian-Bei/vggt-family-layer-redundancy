"""Backbone registry."""

from __future__ import annotations

from typing import Dict

from .contracts import BackboneAdapter

_REGISTRY: Dict[str, BackboneAdapter] = {}


def register(adapter: BackboneAdapter) -> BackboneAdapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_backbone(name: str) -> BackboneAdapter:
    key = name.strip().lower()
    if key not in _REGISTRY:
        _load_defaults()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown backbone {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_backbones() -> list[str]:
    _load_defaults()
    return sorted(_REGISTRY)


def _load_defaults() -> None:
    if _REGISTRY:
        return
    from .adapters.vggt_adapter import VGGTAdapter
    from .adapters.fastvggt_adapter import FastVGGTAdapter
    from .adapters.pi3_adapter import Pi3Adapter
    from .adapters.vggt_omega_adapter import VGGTOmegaAdapter

    register(VGGTAdapter())
    register(FastVGGTAdapter())
    register(Pi3Adapter())
    register(VGGTOmegaAdapter())

from __future__ import annotations

from typing import Dict

from .contracts import DatasetPlugin

_REGISTRY: Dict[str, DatasetPlugin] = {}


def register(plugin: DatasetPlugin) -> DatasetPlugin:
    _REGISTRY[plugin.name] = plugin
    return plugin


def get_dataset(name: str) -> DatasetPlugin:
    key = name.strip().lower()
    if key not in _REGISTRY:
        _load_defaults()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def _load_defaults() -> None:
    if _REGISTRY:
        return
    from .plugins.scannet_plugin import ScanNetPlugin
    from .plugins.sevenscenes_plugin import SevenScenesPlugin
    from .plugins.nrgbd_plugin import NRGBDPlugin

    register(ScanNetPlugin())
    register(SevenScenesPlugin())
    register(NRGBDPlugin())

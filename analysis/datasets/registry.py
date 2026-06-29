from __future__ import annotations

from typing import Dict, List

from .contracts import DatasetAdaptor

_REGISTRY: Dict[str, DatasetAdaptor] = {}
_DEFAULTS_LOADED = False


def register(adaptor: DatasetAdaptor) -> DatasetAdaptor:
    _REGISTRY[adaptor.name] = adaptor
    return adaptor


def get_dataset(name: str) -> DatasetAdaptor:
    key = name.strip().lower()
    _load_defaults()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def dataset_names() -> List[str]:
    _load_defaults()
    return sorted(_REGISTRY)


def _load_defaults() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    from .adaptor.nrgbd import NRGBDAdaptor
    from .adaptor.nuscenes import NuScenesAdaptor
    from .adaptor.scannet import ScanNetAdaptor
    from .adaptor.sevenscenes import SevenScenesAdaptor
    from .adaptor.waymo import WaymoAdaptor

    register(ScanNetAdaptor())
    register(SevenScenesAdaptor())
    register(NRGBDAdaptor())
    register(NuScenesAdaptor())
    register(WaymoAdaptor())
    _DEFAULTS_LOADED = True

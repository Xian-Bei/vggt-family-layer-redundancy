"""Dataset adaptors.

Each module is named after the dataset it adapts. Public pipeline code should
obtain adaptors through ``analysis.datasets.registry.get_dataset``.
"""

from .nrgbd import NRGBDAdaptor
from .nuscenes import NuScenesAdaptor
from .scannet import ScanNetAdaptor
from .sevenscenes import SevenScenesAdaptor
from .waymo import WaymoAdaptor

__all__ = ["NRGBDAdaptor", "NuScenesAdaptor", "ScanNetAdaptor", "SevenScenesAdaptor", "WaymoAdaptor"]

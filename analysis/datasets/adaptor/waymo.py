from __future__ import annotations

from pathlib import Path
from typing import List

from analysis.datasets.adaptor.nuscenes import NuScenesAdaptor

_WAYMO_SPLITS = ("training", "validation")


class WaymoAdaptor(NuScenesAdaptor):
    """Waymo front-camera adaptor for data extracted by ``data/extract_waymo.py``."""

    name = "waymo"
    camera_name = "FRONT"
    valid_splits = _WAYMO_SPLITS
    split_aliases = {
        "train": "training",
        "val": "validation",
    }

    def _scene_names_for_split(self, data_root: Path, split: str) -> List[str]:
        root = self._split_root(Path(data_root), split)
        if not root.is_dir():
            raise FileNotFoundError(f"Waymo split directory not found: {root}")
        return sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "image" / self.camera_name).is_dir()
        )

    def _scene_dir(self, data_root: Path, scene_id: str, split: str) -> Path:
        return self._split_root(Path(data_root), split) / str(scene_id)

    def _split_root(self, data_root: Path, split: str) -> Path:
        split_dir = data_root / split
        if split_dir.is_dir():
            return split_dir
        return data_root

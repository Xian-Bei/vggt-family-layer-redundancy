from __future__ import annotations


_BACKBONE_DISPLAY_NAMES = {
    "vggt": "VGGT",
    "vggt_omega": "VGGT-Ω",
    "vggtomega": "VGGT-Ω",
    "pi3": "π³",
    "mapanything": "MapAnything",
    "mapanything_v1": "MapAnything V1",
    "mean": "Mean",
}

_DATASET_DISPLAY_NAMES = {
    "scannet_dev50": "ScanNet",
    "scans_dev50": "ScanS",
    "scannet_test50": "ScanNet Test50",
    "scans_test50": "ScanS Test50",
    "7scenes_test": "7Scenes Test",
    "7scenes": "7Scenes",
    "7scenes_train": "7Scenes",
    "nrgbd": "NRGBD",
    "nuscenes_val": "nuScenes Val",
    "nuscenes_train": "nuScenes",
    "waymo_validation": "Waymo Validation",
    "waymo_training": "Waymo",
    "mean": "Mean",
}

_BACKBONE_PUBLICATION_ORDER = (
    "mapanything_v1",
    "mapanything",
    "vggt_omega",
    "pi3",
    "vggt",
)


def backbone_display_name(name: str) -> str:
    """Return the publication-facing name for an internal backbone key."""
    raw = str(name or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return _BACKBONE_DISPLAY_NAMES.get(key, raw)


def dataset_display_name(name: str) -> str:
    """Return the publication-facing name for an internal dataset key."""
    raw = str(name or "").strip()
    return _DATASET_DISPLAY_NAMES.get(raw.lower(), raw.replace("_", " ").title())


def backbone_publication_sort_key(name: str) -> tuple[int, str]:
    """Sort backbones consistently across publication-facing figures."""
    raw = str(name or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_PUBLICATION_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_PUBLICATION_ORDER), key)

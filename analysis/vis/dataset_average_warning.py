from __future__ import annotations

from typing import Collection, Dict, Tuple


DATASET_AVERAGE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "training average": ("scannet_dev50", "nuscenes_train"),
    "testing average": (
        "scannet_test50",
        "7scenes_test",
        "nuscenes_val",
        "waymo_validation",
    ),
}

# Publication views distinguish evaluation on held-out splits of the training
# datasets from evaluation on entirely different datasets.  Keep the legacy
# two-way groups above for tabs that still expose a single testing average.
GENERALIZATION_DATASET_AVERAGE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "training average": DATASET_AVERAGE_GROUPS["training average"],
    "same-dataset testing average": (
        "scannet_test50",
        "nuscenes_val",
    ),
    "cross-dataset testing average": (
        "7scenes_test",
        "waymo_validation",
    ),
}

_DATASET_DISPLAY_NAMES = {
    "training average": "Calibration\nDataset",
    "testing average": "Testing\nDataset",
    "same-dataset testing average": "Same-Dataset\nGeneralization",
    "cross-dataset testing average": "Cross-Dataset\nGeneralization",
    "four-dataset average": "Four-Dataset\nAverage",
    "four-dataset testing average": "Four-Dataset\nTest Average",
    "two-dataset testing average": "Two-Dataset\nTest Average",
}


def dataset_display_name(name: str) -> str:
    """Return the user-facing name while preserving legacy result keys."""
    value = str(name)
    return _DATASET_DISPLAY_NAMES.get(value.strip().lower(), value)


def dataset_display_choices(names: Collection[str]) -> list[tuple[str, str]]:
    """Build Gradio label/value pairs for dataset selectors."""
    return [(dataset_display_name(name), str(name)) for name in names]


def format_incomplete_average_warning(
    *,
    owner: str,
    scope: str,
    expected: Collection[str],
    available: Collection[str],
) -> str:
    """Return a prominent status warning when an average omits datasets."""
    expected_names = tuple(dict.fromkeys(str(name) for name in expected))
    available_set = {str(name) for name in available}
    included = tuple(name for name in expected_names if name in available_set)
    missing = tuple(name for name in expected_names if name not in available_set)
    if not missing:
        return ""
    included_text = ", ".join(included) if included else "none"
    return (
        f"⚠️ WARNING: {owner} / {dataset_display_name(scope)} is incomplete "
        f"({len(included)}/{len(expected_names)} datasets). "
        f"Missing: {', '.join(missing)}. Included: {included_text}. "
        "The displayed average uses only the available datasets."
    )

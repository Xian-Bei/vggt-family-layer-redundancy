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
        f"⚠️ WARNING: {owner} / {scope} is incomplete "
        f"({len(included)}/{len(expected_names)} datasets). "
        f"Missing: {', '.join(missing)}. Included: {included_text}. "
        "The displayed average uses only the available datasets."
    )

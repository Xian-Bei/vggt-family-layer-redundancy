"""Final recovery views with Recovery Overview reference curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr

from analysis.vis.recovery_tab import build_recovery_tabs


FINAL_OVERVIEW_DATASETS = (
    "7scenes_test",
    "scannet_test50",
    "nuscenes_val",
    "waymo_validation",
)
FINAL_OVERVIEW_AVERAGE = "four-dataset average"
FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS = {
    FINAL_OVERVIEW_AVERAGE: FINAL_OVERVIEW_DATASETS,
}
FINAL_TRAINING_DATASETS = (
    "scannet_dev50",
    "7scenes_train",
    "nuscenes_train",
    "waymo_training",
)
FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS = {
    "training average": FINAL_TRAINING_DATASETS,
}
FINAL_SAME_DATASET_TESTING_DATASETS = FINAL_OVERVIEW_DATASETS
FINAL_ANALYSIS_SAME_DATASET_TESTING_DATASETS = (
    "scannet_test50",
    "nuscenes_val",
)
FINAL_FOUR_DATASET_TESTING_AVERAGE = "four-dataset testing average"
FINAL_TWO_DATASET_TESTING_AVERAGE = "two-dataset testing average"
FINAL_DATASET_AVERAGE_GROUPS = {
    # Final Detail compares two sources.  Every selector value must resolve to
    # the same concrete datasets for both its solid and dashed curves.
    FINAL_FOUR_DATASET_TESTING_AVERAGE: FINAL_SAME_DATASET_TESTING_DATASETS,
    FINAL_TWO_DATASET_TESTING_AVERAGE: (
        FINAL_ANALYSIS_SAME_DATASET_TESTING_DATASETS
    ),
}
# Keep a separate public constant for callers that distinguish the two roots,
# while deliberately using identical group membership for both curve sources.
FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS = dict(FINAL_DATASET_AVERAGE_GROUPS)
FINAL_TABLE_DATASET_AVERAGE_GROUPS = {
    "training average": FINAL_TRAINING_DATASETS,
    **FINAL_DATASET_AVERAGE_GROUPS,
}
FINAL_DATASETS = (*FINAL_TRAINING_DATASETS, *FINAL_SAME_DATASET_TESTING_DATASETS)


def build_final_results_tabs(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    recovery_output_root: Path,
) -> Dict[str, Any]:
    """Build Final Overview/Detail with source-aligned dataset groups."""
    return build_recovery_tabs(
        blocks=blocks,
        output_root=Path(output_root),
        view_name="Final",
        dataset_average_groups=FINAL_DATASET_AVERAGE_GROUPS,
        comparison_dataset_average_groups=FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS,
        overview_dataset_average_groups=FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS,
        calibration_dataset_average_groups=(
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS
        ),
        comparison_calibration_dataset_average_groups=(
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS
        ),
        allowed_datasets=FINAL_DATASETS,
        overview_allowed_datasets=(
            FINAL_OVERVIEW_AVERAGE,
            *FINAL_OVERVIEW_DATASETS,
        ),
        allowed_prune_strategies=("Retention_min",),
        default_overview_datasets=(FINAL_OVERVIEW_AVERAGE,),
        comparison_output_root=Path(recovery_output_root),
    )

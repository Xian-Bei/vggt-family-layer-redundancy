"""VGGT / ScanNet evaluation: all evaluation code for this repo lives under ``analysis.eval``."""

from .eval_utils import dependencies_available
from .prediction_eval import (
    FastVggtSceneEvalContext,
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
    delta_gt_metrics,
    evaluate_depth_metrics_streamvggt,
    metrics_float_dict,
    prediction_payload_to_fastvggt_reconstruction,
    read_eval_scannet_rows_by_scene,
    resolve_gt_ply_dir,
    write_eval_scannet_json,
)
from .scannet_scene_metrics import (
    FASTVGGT_CORE_METRICS,
    compute_average_metrics_and_save,
    evaluate_scene_and_save,
    evaluate_scene_metrics,
    evaluate_scene_metrics_fastvggt,
    merge_numeric_metrics_into,
)

__all__ = [
    "FASTVGGT_CORE_METRICS",
    "FastVggtSceneEvalContext",
    "build_fastvggt_scene_eval_context",
    "build_scannet_eval_row_from_payload_with_context",
    "compute_average_metrics_and_save",
    "delta_gt_metrics",
    "dependencies_available",
    "evaluate_depth_metrics_streamvggt",
    "evaluate_scene_and_save",
    "evaluate_scene_metrics",
    "evaluate_scene_metrics_fastvggt",
    "merge_numeric_metrics_into",
    "metrics_float_dict",
    "prediction_payload_to_fastvggt_reconstruction",
    "read_eval_scannet_rows_by_scene",
    "resolve_gt_ply_dir",
    "write_eval_scannet_json",
]

"""Evaluation package exports with lazy imports.

Importing a submodule such as ``analysis.eval.scannet_eval`` should not require
heavy optional trajectory/point-cloud dependencies from ``eval_utils``.
"""

from __future__ import annotations

from typing import Any

_EXPORT_MODULES = {
    "dependencies_available": "analysis.eval.eval_utils",
    "FastVggtSceneEvalContext": "analysis.eval.prediction_eval",
    "build_fastvggt_scene_eval_context": "analysis.eval.prediction_eval",
    "build_scannet_eval_row_from_payload_with_context": "analysis.eval.prediction_eval",
    "delta_gt_metrics": "analysis.eval.prediction_eval",
    "evaluate_depth_metrics_streamvggt": "analysis.eval.prediction_eval",
    "metrics_float_dict": "analysis.eval.prediction_eval",
    "prediction_payload_to_fastvggt_reconstruction": "analysis.eval.prediction_eval",
    "read_eval_scannet_rows_by_scene": "analysis.eval.prediction_eval",
    "resolve_gt_ply_dir": "analysis.eval.prediction_eval",
    "write_eval_scannet_json": "analysis.eval.prediction_eval",
    "FASTVGGT_CORE_METRICS": "analysis.eval.metrics",
    "compute_average_metrics_and_save": "analysis.eval.metrics",
    "evaluate_scene_and_save": "analysis.eval.metrics",
    "evaluate_scene_metrics": "analysis.eval.metrics",
    "evaluate_scene_metrics_fastvggt": "analysis.eval.metrics",
    "merge_numeric_metrics_into": "analysis.eval.metrics",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

"""Visualization helpers and tab builders for the results panel."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from .eval_tab import build_eval_tab
from .brute_force_tab import build_brute_force_tab
from .backbone_efficiency_tab import build_backbone_efficiency_tab
from .cka_heatmaps_tab import build_cka_heatmaps_tab
from .lora_tab import build_lora_tab, build_recovery_tab
from .remaining_tab import build_remaining_tab
from .recovery_overview_tab import build_recovery_overview_tab
from .recovery_backbone_comparison_tab import (
    build_recovery_backbone_comparison_tab,
    build_recovery_backbone_remaining_comparison_tab,
)
from .pair_tab import build_pair_tab
from .pruning_plan_tab import build_pruning_plan_tab
from .summary_tab import build_summary_tab


_similarity_metrics_name = f"{__name__}._similarity_metrics"
_similarity_metrics_spec = spec_from_file_location(
    _similarity_metrics_name,
    Path(__file__).with_name("Similarity Metrics.py"),
)
if _similarity_metrics_spec is None or _similarity_metrics_spec.loader is None:
    raise ImportError("Unable to load Similarity Metrics.py")
_similarity_metrics_module = module_from_spec(_similarity_metrics_spec)
sys.modules[_similarity_metrics_name] = _similarity_metrics_module
_similarity_metrics_spec.loader.exec_module(_similarity_metrics_module)
build_heatmaps_tab = _similarity_metrics_module.build_heatmaps_tab
HEATMAP_REAL_DATASETS = _similarity_metrics_module.HEATMAP_REAL_DATASETS

__all__ = [
    "build_cka_heatmaps_tab",
    "build_backbone_efficiency_tab",
    "build_brute_force_tab",
    "build_eval_tab",
    "build_heatmaps_tab",
    "build_lora_tab",
    "build_remaining_tab",
    "build_recovery_tab",
    "build_recovery_overview_tab",
    "build_recovery_backbone_comparison_tab",
    "build_recovery_backbone_remaining_comparison_tab",
    "build_pair_tab",
    "build_pruning_plan_tab",
    "build_summary_tab",
    "HEATMAP_REAL_DATASETS",
]

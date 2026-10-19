"""Visualization helpers and tab builders for the results panel."""

from .eval_tab import build_eval_tab
from .final_results_tab import build_final_results_tabs
from .final_radar_tab import build_final_radar_tab
from .single_interval_tab import (
    BRUTE_FORCE_DETAIL_SUBPLOT_FONT_SIZE,
    BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE,
    build_single_interval_tabs,
    filter_brute_force_detail_datasets,
)
from .backbone_efficiency_tab import build_backbone_efficiency_tab
from .lora_tab import build_lora_tab, build_recovery_tab
from .remaining_tab import (
    REMAINING_ANNOTATION_COLOR,
    REMAINING_GAP_ANNOTATION_FONT_SIZE,
    REMAINING_SUBPLOT_FONT_SIZE,
    build_remaining_tab,
)
from .recovery_tab import build_recovery_tabs
from .recovery_backbone_comparison_tab import (
    build_recovery_ablation_size_tab,
    build_recovery_backbone_remaining_comparison_tab,
)
from .recovery_speed_accuracy_tab import build_recovery_speed_accuracy_tab
from .pair_tab import build_pair_tab
from .pruning_tab import build_pruning_tabs
from .ablation_tab import build_ablation_tab
from .similarity_tab import (
    HEATMAP_REAL_DATASETS,
    SIMILARITY_STATS_FONT_SIZE,
    SIMILARITY_SUBPLOT_FONT_SIZE,
    build_similarity_tab,
)
from .summary_tab import build_summary_tab

__all__ = [
    "build_backbone_efficiency_tab",
    "BRUTE_FORCE_DETAIL_SUBPLOT_FONT_SIZE",
    "BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE",
    "build_single_interval_tabs",
    "filter_brute_force_detail_datasets",
    "build_eval_tab",
    "build_final_results_tabs",
    "build_final_radar_tab",
    "build_similarity_tab",
    "build_lora_tab",
    "build_remaining_tab",
    "REMAINING_ANNOTATION_COLOR",
    "REMAINING_GAP_ANNOTATION_FONT_SIZE",
    "REMAINING_SUBPLOT_FONT_SIZE",
    "build_recovery_tab",
    "build_recovery_tabs",
    "build_recovery_backbone_remaining_comparison_tab",
    "build_recovery_ablation_size_tab",
    "build_recovery_speed_accuracy_tab",
    "build_pair_tab",
    "build_pruning_tabs",
    "build_ablation_tab",
    "build_summary_tab",
    "HEATMAP_REAL_DATASETS",
    "SIMILARITY_STATS_FONT_SIZE",
    "SIMILARITY_SUBPLOT_FONT_SIZE",
]

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

from analysis.metric_labels import SEVEN_METRIC_KEYS, SEVEN_METRIC_LABELS
from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)

from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    OURS_CURVE_ZORDER,
    OURS_MARKER_ZORDER,
    PUBLICATION_COMPARISON_SUBPLOT_HEIGHT,
    PUBLICATION_METRIC_SUBPLOT_WIDTH,
    PUBLICATION_COMPARISON_AXES_BOX_ASPECT,
    PUBLICATION_DETAIL_AXES_BOX_ASPECT,
    PUBLICATION_DETAIL_SUBPLOT_HEIGHT,
    RECOVERY_METHOD_ORDER,
    STRATEGY_PLOT_COLORS,
    _with_dataset_average,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    emphasize_figure,
    metric_value,
    move_y_axis_label_to_top,
    render_strategy_metric_figure,
)
from analysis.redundancy.skip_metric_scaling import (
    normalize_raw_metric_against_upper_worst,
)
from analysis.vis.dataset_average_warning import (
    DATASET_AVERAGE_GROUPS,
    GENERALIZATION_DATASET_AVERAGE_GROUPS,
    dataset_display_choices,
    dataset_display_name,
    format_incomplete_average_warning,
)
from analysis.recovery.shared_results import (
    seven_metric_mean_score,
    single_double_mean_decision,
)
from analysis.vis.pruning_tab import (
    _mean_backbone_plot_payloads,
    _figure_to_rgb as _compact_pruning_figure,
    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP,
    PRUNING_PLAN_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP,
    PRUNING_PLAN_OVERVIEW_YLABEL_PAD,
)
from analysis.vis.composite import (
    Cell,
    Section,
    _font,
    compose_sections,
    crop_horizontal_white_margins,
    gallery_cells,
)
from analysis.vis.shared_legend import (
    SHARED_LEGEND_CAPTION,
    LegendEntry,
    append_shared_legends_right,
    collect_and_remove_legends,
    prepend_shared_legend,
    render_shared_legend_banner,
)

# Recovery Overview typography and outer-label spacing (pixels).
RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE = PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE
RECOVERY_OVERVIEW_OUTER_FONT_SIZE = PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE
RECOVERY_SHARED_LEGEND_FONT_SIZE = PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE
RECOVERY_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP = PRUNING_PLAN_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP
RECOVERY_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP = 0
RECOVERY_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP = PRUNING_PLAN_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP
RECOVERY_OVERVIEW_Y_AXIS_UPPER = 0.40
# Recovery Detail follows the Pruning Plan Detail outer layout.
RECOVERY_DETAIL_SUBPLOT_FONT_SIZE = 24
RECOVERY_DETAIL_OUTER_FONT_SIZE = 32
RECOVERY_DETAIL_LEFT_LABEL_TO_IMAGE_GAP = 0
RECOVERY_DETAIL_TOP_LABEL_TO_IMAGE_GAP = 0
RECOVERY_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP = 0
RECOVERY_DETAIL_SECTION_GAP = 0
RECOVERY_DETAIL_COLUMN_LABELS = tuple(
    SEVEN_METRIC_LABELS[key] for key in SEVEN_METRIC_KEYS
)
_RECOVERY_OVERVIEW_METRIC_GRID = (("gt_mean",),)
_RECOVERY_DETAIL_METRIC_GRID = (
    GT_MEAN_METRICS[:4],
    GT_MEAN_METRICS[4:],
)
_RECOVERY_OVERVIEW_COLUMNS = (
    ("training average", "Calibration\nDataset", "CKA_min"),
    ("training average", "Calibration\nDataset", "Retention_min"),
    (
        "same-dataset testing average",
        "Same-Dataset\nGeneralization",
        "CKA_min",
    ),
    (
        "same-dataset testing average",
        "Same-Dataset\nGeneralization",
        "Retention_min",
    ),
    (
        "cross-dataset testing average",
        "Cross-Dataset\nGeneralization",
        "CKA_min",
    ),
    (
        "cross-dataset testing average",
        "Cross-Dataset\nGeneralization",
        "Retention_min",
    ),
)
_DEFAULT_RECOVERY_OVERVIEW_DATASETS = tuple(
    dict.fromkeys(
        dataset
        for dataset, _, _ in _RECOVERY_OVERVIEW_COLUMNS
        if dataset != "training average"
    )
)
_RECOVERY_OVERVIEW_PRUNE_STRATEGIES = ("CKA_min", "Retention_min")
_MIN_STRATEGY_DISPLAY_NAMES = {
    "CKA_min": "CKA",
    "Retention_min": "Degradation",
}
_RECOVERY_DETAIL_PRUNE_STRATEGY_DISPLAY_NAMES = {
    "CKA_min": "CKA-based",
    "Retention_min": "Degradation-based",
}
_DEFAULT_RECOVERY_PRUNE_STRATEGIES = ("Retention_min",)


_BACKBONE_ORDER = ("vggt_omega", "pi3", "vggt")
_PRUNE_STRATEGIES = (
    "CKA_double", "CKA_single", "retention_double", "retention_single",
)
_BASELINE_STRATEGIES = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
)
_DATASET_AVERAGE_GROUPS = DATASET_AVERAGE_GROUPS
_OVERVIEW_DATASET_AVERAGE_GROUPS = GENERALIZATION_DATASET_AVERAGE_GROUPS
_ALL_DATASET_AVERAGE_GROUPS = {
    **_DATASET_AVERAGE_GROUPS,
    **_OVERVIEW_DATASET_AVERAGE_GROUPS,
}
_MEAN_BACKBONE = "mean"
_MIN_PRUNE_STRATEGY_CANDIDATES = {
    "Retention_min": ("retention_double", "retention_single"),
    "CKA_min": ("CKA_double", "CKA_single"),
}
_RECOVERY_LEGEND_LABELS = {
    "recovery_replaceme_cos": "ReplaceMe Cos",
    "recovery_replaceme_ls": "ReplaceMe LS",
    "recovery_streamline_ffn": "Streamline FFN",
    "recovery_ghost": "Ghost",
    "recovery_ghost_interval": "Ghost",
    "recovery_linearpatch": "LinearPatch",
    "recovery_ghost_interval_sequential": "Current Target",
    "recovery_ghost_interval_sequential_grouped": "Current Target (Special/Patch)",
    "recovery_ghost_interval_sequential_autogroup": "Ours",
}
_RECOVERY_LEGEND_ORDER = (
    "Full Model",
    "No Recovery",
    "LinearPatch",
    "ReplaceMe Cos",
    "ReplaceMe LS",
    "Streamline FFN",
    "Ghost",
    "Current Target",
    "Current Target (Special/Patch)",
    "Ours",
)

_DEFAULT_UNCHECKED_RECOVERY_METHODS = frozenset({
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_interval_sequential_grouped",
})
_AUTOGROUP_METHOD = "recovery_ghost_interval_sequential_autogroup"
_PI3_OURS_FALLBACK_METHOD = "recovery_ghost_interval_sequential"
_COMPARISON_SOURCE_LEGEND_GID = "recovery-comparison-source"
_OURS_LINEWIDTH = 4.4


def _apply_backbone_recovery_method_aliases(
    payload: Dict[str, Any], *, backbone: str
) -> Dict[str, Any]:
    """Expose Pi3 sequential results through the selected adaptive Ours key."""
    result = deepcopy(payload)
    if str(backbone).strip().lower() != "pi3":
        return result

    def has_available_point(strategy: Any) -> bool:
        if not isinstance(strategy, dict):
            return False
        points = strategy.get("points")
        return isinstance(points, dict) and any(
            isinstance(point, dict) and point.get("available", True) is not False
            for point in points.values()
        )

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        strategies = value.get("strategies")
        if (
            isinstance(strategies, dict)
            and not has_available_point(strategies.get(_AUTOGROUP_METHOD))
            and has_available_point(
                strategies.get(_PI3_OURS_FALLBACK_METHOD)
            )
        ):
            strategies[_AUTOGROUP_METHOD] = deepcopy(
                strategies[_PI3_OURS_FALLBACK_METHOD]
            )
            strategies[_AUTOGROUP_METHOD]["display_name"] = "Ours"
            strategies[_AUTOGROUP_METHOD]["source_method"] = (
                _PI3_OURS_FALLBACK_METHOD
            )
        for child in value.values():
            if isinstance(child, dict):
                visit(child)

    visit(result)
    return result


def _effective_dataset_average_groups(
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Tuple[str, ...]]:
    source = (
        _ALL_DATASET_AVERAGE_GROUPS
        if dataset_average_groups is None
        else dataset_average_groups
    )
    return {
        str(group): tuple(
            dict.fromkeys(
                str(dataset)
                for dataset in datasets
                if str(dataset).strip()
            )
        )
        for group, datasets in source.items()
    }


def _dataset_count_average_label(dataset: str, datasets: Sequence[str]) -> str:
    """Describe a source-specific average without assuming shared domains."""
    count = len(tuple(dict.fromkeys(str(item) for item in datasets)))
    count_label = {2: "Two", 4: "Four"}.get(count, str(count))
    suffix = (
        "Test Average"
        if dataset.endswith("testing average")
        else "Calibration Average"
        if dataset == "training average"
        else "Dataset Average"
    )
    return f"{count_label}-Dataset {suffix}"


def _compose_recovery_overview(
    value: Any,
    simplify_dataset_names: bool = False,
) -> np.ndarray:
    all_cells = gallery_cells(value)
    legend_cell = next(
        (cell for cell in all_cells if cell.caption.startswith(SHARED_LEGEND_CAPTION)),
        None,
    )
    legend = legend_cell.image if legend_cell else None
    legend_count = (
        max(1, int(legend_cell.caption.rsplit("|entries=", 1)[1]))
        if legend_cell is not None and "|entries=" in legend_cell.caption
        else 1
    )
    cells = [
        cell for cell in all_cells
        if not cell.caption.startswith(SHARED_LEGEND_CAPTION)
    ]
    records: Dict[Tuple[str, str, str], Cell] = {}
    backbone_labels: List[str] = []
    dataset_keys: List[str] = []
    observed_strategies: List[str] = []
    for cell in cells:
        parts = cell.caption.rsplit(" · ", 2) if cell.caption else []
        if len(parts) != 3:
            continue
        backbone, dataset, prune_strategy = parts
        if backbone not in backbone_labels:
            backbone_labels.append(backbone)
        if dataset not in dataset_keys:
            dataset_keys.append(dataset)
        if prune_strategy not in observed_strategies:
            observed_strategies.append(prune_strategy)
        cell.caption = ""
        records[(backbone, dataset, prune_strategy)] = cell
    strategy_labels = [
        strategy
        for strategy in _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
        if strategy in observed_strategies
    ]
    strategy_labels.extend(
        strategy
        for strategy in observed_strategies
        if strategy not in strategy_labels
    )
    fixed_rows = [
        [
            records.get((backbone, dataset, prune_strategy), Cell(None))
            for backbone in backbone_labels
            for prune_strategy in strategy_labels
        ]
        for dataset in dataset_keys
    ]
    for row in fixed_rows:
        for cell in row:
            if cell.image is not None:
                cell.image = crop_horizontal_white_margins(cell.image)
    row_labels = (
        [f"({chr(ord('a') + index)})" for index in range(len(dataset_keys))]
        if simplify_dataset_names
        else [dataset_display_name(dataset) for dataset in dataset_keys]
    )
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_font = _font(RECOVERY_OVERVIEW_OUTER_FONT_SIZE, bold=True)
    label_width = max(
        measure.textbbox((0, 0), label, font=row_font)[2]
        for label in row_labels
    ) + 12
    column_count = len(strategy_labels) * len(backbone_labels)
    content_width = 2400 - label_width - RECOVERY_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP
    column_gap = 6
    column_weights = [
        max(
            (cell.image.width for row in fixed_rows if (cell := row[column]).image is not None),
            default=1,
        )
        for column in range(column_count)
    ]
    available_width = max(
        column_count,
        content_width - column_gap * (column_count - 1),
    )
    column_widths = np.maximum(
        1,
        np.floor(
            available_width
            * np.asarray(column_weights, dtype=np.float64)
            / sum(column_weights)
        ),
    ).astype(int).tolist()
    row_axis_bounds: List[Tuple[int, int]] = []
    row_start = 0
    for row in fixed_rows:
        scaled_heights = [
            max(1, round(cell.image.height * column_widths[column] / cell.image.width))
            for column, cell in enumerate(row)
            if cell.image is not None
        ]
        row_height = max(scaled_heights, default=1)
        anchor_column = next(
            (column for column, cell in enumerate(row) if cell.image is not None),
            None,
        )
        anchor = row[anchor_column].image if anchor_column is not None else None
        if anchor is None:
            row_axis_bounds.append((row_start, row_start + row_height))
        else:
            scaled_height = max(
                1,
                round(anchor.height * column_widths[anchor_column] / anchor.width),
            )
            image_top = row_start + max(0, (row_height - scaled_height) // 2)
            row_axis_bounds.append((
                image_top + round(float(anchor.info.get("axes_top_y", 0.0)) * scaled_height),
                image_top + round(float(anchor.info.get("axes_bottom_y", 1.0)) * scaled_height),
            ))
        row_start += row_height + 2
    composite = compose_sections(
        [
            Section(
                columns=len(strategy_labels) * len(backbone_labels),
                cells=[],
                column_width_weights=column_weights,
                column_gaps=[column_gap] * max(0, column_count - 1),
                fixed_rows=fixed_rows,
                row_labels=row_labels,
                row_labels_bold=True,
                column_labels=[
                    backbone for backbone in backbone_labels
                    for _ in strategy_labels
                ],
                column_labels_below=True,
                column_label_centers=[
                    next((
                        float(row[column].image.info["axes_center_x"])
                        for row in reversed(fixed_rows)
                        if row[column].image is not None
                        and "axes_center_x" in row[column].image.info
                    ), 0.5)
                    for column in range(len(strategy_labels) * len(backbone_labels))
                ],
                column_labels_bold=True,
                row_label_centers=[
                    next((
                        float(cell.image.info["axes_center_y"])
                        for cell in row
                        if cell.image is not None
                        and "axes_center_y" in cell.image.info
                    ), 0.5)
                    for row in fixed_rows
                ],
            )
        ],
        outer_font_size=RECOVERY_OVERVIEW_OUTER_FONT_SIZE,
        left_label_to_image_gap=RECOVERY_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=RECOVERY_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=RECOVERY_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP,
    )
    if legend is None or not row_axis_bounds:
        return composite
    return append_shared_legends_right(composite, ((
        legend,
        row_axis_bounds[0][0] / composite.shape[0],
        row_axis_bounds[-1][1] / composite.shape[0],
        legend_count,
    ),), frameon=False)


def _compose_recovery_detail(value: Any) -> np.ndarray:
    all_cells = gallery_cells(value)
    legend = next(
        (
            cell.image
            for cell in all_cells
            if cell.caption == SHARED_LEGEND_CAPTION
        ),
        None,
    )
    cells = [
        cell for cell in all_cells if cell.caption != SHARED_LEGEND_CAPTION
    ]
    records: Dict[Tuple[str, str, str], Cell] = {}
    block_keys: List[Tuple[str, str]] = []
    strategy_labels: List[str] = []
    for cell in cells:
        parts = cell.caption.rsplit(" · ", 2) if cell.caption else []
        if len(parts) != 3:
            continue
        backbone, dataset, prune_strategy = parts
        block_key = (backbone, dataset)
        if block_key not in block_keys:
            block_keys.append(block_key)
        if prune_strategy not in strategy_labels:
            strategy_labels.append(prune_strategy)
        cell.caption = ""
        records[(backbone, dataset, prune_strategy)] = cell
    sections = [
        Section(
            columns=1,
            cells=[
                records.get((backbone, dataset, prune_strategy), Cell(None))
            ],
        )
        for backbone, dataset in block_keys
        for prune_strategy in strategy_labels
    ]
    composite = compose_sections(
        sections,
        outer_font_size=RECOVERY_DETAIL_OUTER_FONT_SIZE,
        left_label_to_image_gap=RECOVERY_DETAIL_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=RECOVERY_DETAIL_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=RECOVERY_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP,
        section_gap=RECOVERY_DETAIL_SECTION_GAP,
    )
    return prepend_shared_legend(composite, legend)


def _formalize_recovery_legend_names(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply publication labels without mutating collected evaluation data."""
    result = deepcopy(payload)
    strategies = result.get("strategies", {})
    if not isinstance(strategies, dict):
        return result
    for strategy_key, strategy_payload in strategies.items():
        if not isinstance(strategy_payload, dict):
            continue
        key = str(strategy_key)
        if key.startswith("no_recovery"):
            strategy_payload["display_name"] = "No Recovery"
        elif key in _RECOVERY_LEGEND_LABELS:
            strategy_payload["display_name"] = _RECOVERY_LEGEND_LABELS[key]
    return result


def _overlay_recovery_overview_methods(
    final_payload: Dict[str, Any],
    recovery_payload: Dict[str, Any],
    methods: Sequence[str],
    *,
    primary_source_label: str = "Final",
    comparison_source_label: str = "Recovery Overview",
) -> Tuple[Dict[str, Any], List[str], Dict[str, Dict[str, Any]]]:
    """Overlay matching methods; color encodes method and line style encodes source."""
    result = _formalize_recovery_legend_names(final_payload)
    reference = _formalize_recovery_legend_names(recovery_payload)
    result_strategies = result.get("strategies", {})
    reference_strategies = reference.get("strategies", {})
    if not isinstance(result_strategies, dict):
        result_strategies = {}
        result["strategies"] = result_strategies
    if not isinstance(reference_strategies, dict):
        reference_strategies = {}

    selected: List[str] = []
    styles: Dict[str, Dict[str, Any]] = {}
    baseline_keys = [
        str(key)
        for key in result_strategies
        if str(key).startswith("no_recovery")
    ]
    for baseline_key in baseline_keys:
        final_strategy = result_strategies.get(baseline_key)
        if not isinstance(final_strategy, dict):
            continue
        display_name = str(final_strategy.get("display_name", "No Recovery"))
        final_strategy["display_name"] = display_name
        selected.append(baseline_key)
        color = STRATEGY_PLOT_COLORS.get(baseline_key)
        styles[baseline_key] = {
            "linestyle": "-",
            "dashes": None,
            "color": color,
            "zorder": 2.0,
        }

        recovery_strategy = reference_strategies.get(baseline_key)
        if not isinstance(recovery_strategy, dict):
            continue
        reference_key = f"{baseline_key}__recovery_overview"
        result_strategies[reference_key] = deepcopy(recovery_strategy)
        result_strategies[reference_key]["display_name"] = display_name
        selected.append(reference_key)
        styles[reference_key] = {
            "linestyle": "--",
            "dashes": (5.0, 2.0),
            "color": color,
            "zorder": 3.0,
        }

    for method in methods:
        method = str(method)
        final_strategy = result_strategies.get(method)
        if not isinstance(final_strategy, dict):
            continue
        display_name = str(final_strategy.get("display_name", method))
        final_strategy["display_name"] = display_name
        selected.append(method)
        color = STRATEGY_PLOT_COLORS.get(method)
        styles[method] = {"linestyle": "-", "color": color}
        if method == _AUTOGROUP_METHOD:
            styles[method].update(
                linewidth=_OURS_LINEWIDTH,
                zorder=OURS_CURVE_ZORDER,
                marker_zorder=OURS_MARKER_ZORDER,
            )

        recovery_strategy = reference_strategies.get(method)
        if not isinstance(recovery_strategy, dict):
            continue
        reference_key = f"{method}__recovery_overview"
        result_strategies[reference_key] = deepcopy(recovery_strategy)
        result_strategies[reference_key]["display_name"] = display_name
        selected.append(reference_key)
        styles[reference_key] = {"linestyle": "--", "color": color}
        if method == _AUTOGROUP_METHOD:
            styles[reference_key].update(
                linewidth=_OURS_LINEWIDTH,
                zorder=OURS_CURVE_ZORDER + 1.0,
                marker_zorder=OURS_MARKER_ZORDER + 1.0,
            )
    return result, selected, styles


def _append_comparison_source_legend_entries(
    entries: List[LegendEntry],
    primary_source_label: str,
    comparison_source_label: str,
) -> None:
    """Add source-only legend keys, independently from method/color keys."""
    seen = {str(label) for _handle, label in entries}
    for label, linestyle, dashes in (
        (primary_source_label, "-", None),
        (comparison_source_label, "--", (5.0, 2.0)),
    ):
        if label in seen:
            continue
        handle = Line2D(
            [],
            [],
            color="#222222",
            linestyle=linestyle,
            linewidth=3.2,
            marker="None",
        )
        if dashes is not None:
            handle.set_dashes(dashes)
        handle.set_gid(_COMPARISON_SOURCE_LEGEND_GID)
        entries.append((handle, label))
        seen.add(label)


def _seven_metric_mean(point: Dict[str, Any], payload: Dict[str, Any]) -> float | None:
    """Return the normalized seven-metric mean used for visualization."""
    normalized = point.get("normalized_metric", {})
    if isinstance(normalized, dict):
        values = [metric_value(normalized, metric) for metric in GT_MEAN_METRICS]
        if all(value is not None and np.isfinite(value) for value in values):
            return float(np.mean(values))
    raw = point.get("raw_metric", point.get("metrics_mean", {}))
    if not isinstance(raw, dict):
        return None
    bounds = payload.get("normalization_bounds", {})
    values: List[float] = []
    for metric in GT_MEAN_METRICS:
        value = metric_value(raw, metric)
        if value is None or not np.isfinite(value):
            return None
        metric_bounds = bounds.get(metric, {}) if isinstance(bounds, dict) else {}
        low = metric_bounds.get("min") if isinstance(metric_bounds, dict) else None
        high = metric_bounds.get("max") if isinstance(metric_bounds, dict) else None
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high > low:
            values.append((float(value) - float(low)) / (float(high) - float(low)))
        else:
            values.append(float(value))
    return float(np.mean(values))


def _minimum_strategy_selection(
    candidates: Sequence[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    """Choose complete Single/Double points by the shared seven-metric rule."""
    valid = [(name, payload) for name, payload in candidates if isinstance(payload, dict)]
    if not valid:
        return {}
    strategy_keys = {
        str(key)
        for _, payload in valid
        for key in payload.get("strategies", {})
        if isinstance(payload.get("strategies", {}), dict)
    }
    selected: Dict[str, Dict[str, str]] = {}
    for strategy_key in strategy_keys:
        choices: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = {}
        for source_name, payload in valid:
            strategies = payload.get("strategies", {})
            strategy = strategies.get(strategy_key, {}) if isinstance(strategies, dict) else {}
            if not isinstance(strategy, dict):
                continue
            points = strategy.get("points", {})
            for drop_key, point in points.items() if isinstance(points, dict) else ():
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                raw = point.get("raw_metric", point.get("metrics_mean", {}))
                score = seven_metric_mean_score(raw) if isinstance(raw, dict) else None
                if score is not None:
                    choices.setdefault(str(drop_key), []).append((score, source_name, point))
        for drop_key, point_choices in choices.items():
            by_structure = {
                structure: choice
                for choice in point_choices
                for structure in (
                    "single" if choice[1].endswith("_single") else
                    "double" if choice[1].endswith("_double") else "",
                )
                if structure
            }
            if set(by_structure) == {"single", "double"}:
                decision = single_double_mean_decision(
                    by_structure["single"][0], by_structure["double"][0]
                )
                _, source_name, _ = by_structure[str(decision["winner"])]
            else:
                _, source_name, _ = min(
                    point_choices, key=lambda item: (item[0], item[1])
                )
            selected.setdefault(strategy_key, {})[drop_key] = source_name
    return selected


def _minimum_strategy_plot_payload(
    candidates: Sequence[Tuple[str, Dict[str, Any]]],
    calibration_selection: Mapping[str, Mapping[str, str]] | None = None,
) -> Dict[str, Any]:
    """Apply calibration-selected complete points to one evaluation scope."""
    valid = [(name, payload) for name, payload in candidates if isinstance(payload, dict)]
    if not valid:
        return {}
    selection = (
        {
            str(strategy): {str(drop): str(source) for drop, source in drops.items()}
            for strategy, drops in calibration_selection.items()
        }
        if calibration_selection is not None
        else _minimum_strategy_selection(valid)
    )
    result = deepcopy(valid[0][1])
    result["strategies"] = {}
    payload_by_source = dict(valid)
    for strategy_key, selected_sources in selection.items():
        selected_points: Dict[str, Any] = {}
        display_name = strategy_key
        for drop_key, source_name in selected_sources.items():
            payload = payload_by_source.get(source_name, {})
            strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
            strategy = strategies.get(strategy_key, {}) if isinstance(strategies, dict) else {}
            if not isinstance(strategy, dict):
                continue
            display_name = str(strategy.get("display_name", display_name))
            points = strategy.get("points", {})
            point = points.get(drop_key) if isinstance(points, dict) else None
            if not isinstance(point, dict) or not bool(point.get("available", False)):
                continue
            selected_points[drop_key] = {
                **deepcopy(point),
                "min_prune_strategy": source_name,
                "selection_scope": "calibration",
            }
        if selected_points:
            result["strategies"][strategy_key] = {
                "display_name": display_name, "points": selected_points,
            }
    return result


def _prune_strategy_candidates(prune_strategy: str) -> Tuple[str, ...]:
    """Expand a minimum view into the concrete strategies it compares."""
    return _MIN_PRUNE_STRATEGY_CANDIDATES.get(
        prune_strategy, (prune_strategy,)
    )


def _collect_backbone_normalization_profiles(
    pruning_by_backbone: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Expose the persisted selected-pruning profile for each backbone/dataset.

    Normalization does not depend on the displayed CKA/Retention subset.
    """
    profiles_by_backbone: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for backbone, pruning in pruning_by_backbone.items():
        datasets = pruning.get("datasets", {}) if isinstance(pruning, dict) else {}
        if not isinstance(datasets, dict):
            continue
        dataset_profiles: Dict[str, Dict[str, Any]] = {}
        for dataset, dataset_payload in datasets.items():
            profile = (
                dataset_payload.get("normalization_profile", {})
                if isinstance(dataset_payload, dict)
                else {}
            )
            if not isinstance(profile, dict):
                continue
            upper = profile.get("upper_average", {})
            maxima = profile.get(
                "maximum_brute_force_metric",
                profile.get("maximum_single_interval_metric", {}),
            )
            if not isinstance(upper, dict) or not isinstance(maxima, dict):
                continue
            denominators = {
                metric: max(0.0, float(maximum) - float(baseline))
                for metric in GT_MEAN_METRICS
                for maximum in [metric_value(maxima, metric)]
                for baseline in [metric_value(upper, metric)]
                if maximum is not None and baseline is not None
            }
            if not denominators:
                continue
            dataset_profiles[str(dataset)] = {
                **deepcopy(profile),
                "normalization_denominator": denominators,
            }
        if dataset_profiles:
            profiles_by_backbone[str(backbone)] = dataset_profiles
    return profiles_by_backbone

def _apply_backbone_normalization_profiles(
    recovery_payload: Dict[str, Any],
    profiles_by_backbone: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    backbone: str,
    use_profile_upper: bool = False,
) -> Dict[str, Any]:
    """Re-normalize one backbone with its selected-pruning profile bounds.

    ``use_profile_upper`` also adopts the source profile's lower bound, which
    lets another result tree use exactly the same numeric ranges.
    """
    result = deepcopy(recovery_payload)
    datasets = result.get("datasets", {})
    if not isinstance(datasets, dict):
        return result
    backbone_profiles = profiles_by_backbone.get(str(backbone), {})
    applied_datasets: set[str] = set()
    for dataset, dataset_payload in datasets.items():
        profile = backbone_profiles.get(str(dataset), {})
        if not isinstance(dataset_payload, dict) or not isinstance(profile, dict):
            continue
        denominators = profile.get("normalization_denominator", {})
        upper_average = dataset_payload.get("upper_average", {})
        if use_profile_upper:
            profile_upper = profile.get("upper_average", {})
            if isinstance(profile_upper, dict) and profile_upper:
                normalization_upper_average = profile_upper
            else:
                normalization_upper_average = upper_average
        else:
            normalization_upper_average = upper_average
        if not isinstance(denominators, dict) or not isinstance(upper_average, dict):
            continue
        raw_maxima: Dict[str, float] = {}
        bounds: Dict[str, Dict[str, float]] = {}
        for metric in GT_MEAN_METRICS:
            upper = metric_value(normalization_upper_average, metric)
            denominator = denominators.get(metric)
            if (
                upper is None
                or not isinstance(denominator, (int, float))
                or not np.isfinite(float(denominator))
            ):
                continue
            maximum = float(upper) + float(denominator)
            raw_maxima[metric] = maximum
            bounds[metric] = {"min": float(upper), "max": maximum}
        if not bounds:
            continue

        strategies = dataset_payload.get("strategies", {})
        if isinstance(strategies, dict):
            for strategy in strategies.values():
                points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
                if not isinstance(points, dict):
                    continue
                for point in points.values():
                    if not isinstance(point, dict) or not bool(point.get("available", False)):
                        continue
                    metrics = point.get("metrics_mean", point.get("raw_metric", {}))
                    if not isinstance(metrics, dict):
                        continue
                    raw: Dict[str, float] = {}
                    normalized: Dict[str, float] = {}
                    for metric in GT_MEAN_METRICS:
                        value = metric_value(metrics, metric)
                        upper = metric_value(normalization_upper_average, metric)
                        denominator = denominators.get(metric)
                        if value is None or upper is None:
                            continue
                        raw[metric] = float(value)
                        if isinstance(denominator, (int, float)):
                            normalized[metric] = float(
                                normalize_raw_metric_against_upper_worst(
                                    np.asarray([float(value)], dtype=np.float64),
                                    upper_baseline=float(upper),
                                    worst_value=float(upper) + float(denominator),
                                    clamp_better_than_upper=False,
                                )[0]
                            )
                    point["raw_metric"] = raw
                    point["normalized_metric"] = normalized
                    point["normalized_relative_to_upper"] = dict(normalized)

        dataset_payload["normalized_upper_average"] = {
            metric: (
                (float(value) - float(bounds[metric]["min"]))
                / (float(bounds[metric]["max"]) - float(bounds[metric]["min"]))
                if float(bounds[metric]["max"]) > float(bounds[metric]["min"])
                else 0.0
            )
            for metric, value in upper_average.items()
            if metric in bounds and isinstance(value, (int, float))
        }
        dataset_payload["normalization_bounds"] = bounds
        dataset_payload["normalization_profile"] = {
            **deepcopy(profile),
            "backbone": str(backbone),
            "upper_average": {
                metric: float(value)
                for metric in GT_MEAN_METRICS
                for value in [metric_value(normalization_upper_average, metric)]
                if value is not None
            },
            "evaluated_upper_average": {
                metric: float(value)
                for metric in GT_MEAN_METRICS
                for value in [metric_value(upper_average, metric)]
                if value is not None
            },
            "maximum_brute_force_metric": raw_maxima,
            "maximum_selected_pruning_metric": raw_maxima,
            # Compatibility for shared plotting helpers that consume a
            # generic metric maximum from the normalization profile.
            "maximum_single_interval_metric": raw_maxima,
        }
        applied_datasets.add(str(dataset))

    if not applied_datasets:
        return result
    skipped = result.get("skipped_normalization_datasets", [])
    if isinstance(skipped, list):
        result["skipped_normalization_datasets"] = [
            str(dataset)
            for dataset in skipped
            if str(dataset) not in applied_datasets
        ]
    result["recovery_normalization_method"] = (
        "stage2_brute_force_single_interval_raw_metric_max"
    )
    return _with_dataset_average(result)


def _sort_key(name: str, preferred: Sequence[str]) -> Tuple[int, str]:
    key = str(name)
    try:
        return (preferred.index(key), key)
    except ValueError:
        return (len(preferred), key)


def discover_recovery_overview_options(output_root: Path) -> Tuple[List[str], List[str]]:
    """Discover controls from plans and result folders, including unfinished runs."""
    root = Path(output_root)
    backbones: List[str] = []
    methods = set()
    if root.is_dir():
        for backbone_dir in (path for path in root.iterdir() if path.is_dir()):
            from_results = {
                path.name
                for path in backbone_dir.glob("*/eval/recovery/recovery_*")
                if path.is_dir() and path.name in RECOVERY_METHOD_ORDER
            }
            from_results.update(
                path.name
                for path in backbone_dir.glob("*/eval/recovery/*/recovery_*")
                if path.is_dir() and path.name in RECOVERY_METHOD_ORDER
            )
            from_plans = {
                path.stem
                for path in backbone_dir.glob("recovery_*.json")
                if path.stem in RECOVERY_METHOD_ORDER
            }
            from_bundle = set()
            bundle_path = backbone_dir / "recovery_plan_all.json"
            if bundle_path.is_file():
                try:
                    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    bundle = {}
                bundle_methods = bundle.get("methods", {}) if isinstance(bundle, dict) else {}
                if isinstance(bundle_methods, dict):
                    from_bundle = {
                        str(name)
                        for name in bundle_methods
                        if str(name) in RECOVERY_METHOD_ORDER
                    }
            found = from_results | from_plans | from_bundle
            if (
                backbone_dir.name == "pi3"
                and _PI3_OURS_FALLBACK_METHOD in found
            ):
                found.add(_AUTOGROUP_METHOD)
            if found or (backbone_dir / "prune_plan_all.json").is_file():
                backbones.append(backbone_dir.name)
                methods.update(found)
    return (
        sorted(set(backbones), key=lambda name: _sort_key(name, _BACKBONE_ORDER)),
        sorted(methods, key=lambda name: _sort_key(name, RECOVERY_METHOD_ORDER)),
    )


def discover_recovery_overview_prune_strategies(output_root: Path) -> List[str]:
    """Discover supported strategy views with shared recovery results."""
    root = Path(output_root)
    strategies = set()
    if root.is_dir():
        for backbone_root in (path for path in root.iterdir() if path.is_dir()):
            try:
                bundle = load_prune_plan_bundle(backbone_root)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            available_selectors = {
                selector_root.name
                for recovery_root in backbone_root.glob("*/eval/recovery")
                if recovery_root.is_dir()
                for selector_root in recovery_root.iterdir()
                if selector_root.is_dir()
                and any(
                    method.is_dir() and method.name in RECOVERY_METHOD_ORDER
                    for method in selector_root.iterdir()
                )
            }
            checkpoint_root = backbone_root / "recovery"
            if checkpoint_root.is_dir():
                available_selectors.update(
                    selector_root.name
                    for selector_root in checkpoint_root.iterdir()
                    if selector_root.is_dir()
                    and any(
                        method.is_dir() and method.name in RECOVERY_METHOD_ORDER
                        for method in selector_root.iterdir()
                    )
                )
            plans = bundle_plan_records(bundle)
            for strategy, strategy_record in bundle_strategy_records(bundle).items():
                if strategy not in _PRUNE_STRATEGIES:
                    continue
                points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
                selectors = set()
                for plan_name in points.values() if isinstance(points, dict) else ():
                    record = plans.get(str(plan_name))
                    if isinstance(record, dict):
                        selectors.add(physical_pruning_selector_from_record(record))
                if not available_selectors or selectors.intersection(available_selectors):
                    strategies.add(str(strategy))
    return sorted(strategies, key=lambda name: _sort_key(name, _PRUNE_STRATEGIES))


def discover_recovery_overview_datasets(output_root: Path) -> List[str]:
    """Discover datasets with a recovery evaluation directory."""
    root = Path(output_root)
    datasets = {
        path.parent.parent.name
        for path in root.glob("*/*/eval/recovery")
        if path.is_dir()
    }
    return [*_DATASET_AVERAGE_GROUPS, *sorted(datasets)]


def _discover_recovery_generalization_datasets(
    output_root: Path,
    *,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    allowed_datasets: Sequence[str] | None = None,
) -> List[str]:
    """Discover concrete datasets plus the configured publication averages."""
    groups = _effective_dataset_average_groups(dataset_average_groups)
    datasets = discover_recovery_overview_datasets(output_root)
    known_groups = set(_ALL_DATASET_AVERAGE_GROUPS) | set(groups)
    concrete = [
        dataset for dataset in datasets
        if dataset not in known_groups
    ]
    if allowed_datasets is not None:
        allowed = {str(dataset) for dataset in allowed_datasets}
        concrete = [dataset for dataset in concrete if dataset in allowed]
    return [*groups, *concrete]


def _dataset_group_average(
    payload: Dict[str, Any],
    dataset: str,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    """Average the available datasets belonging to a named overview group."""
    groups = _effective_dataset_average_groups(dataset_average_groups)
    dataset_payloads = payload.get("datasets", {})
    if not isinstance(dataset_payloads, dict):
        return {}
    selected = {
        name: dataset_payloads[name]
        for name in groups.get(dataset, ())
        if name in dataset_payloads
    }
    average = _with_dataset_average({"datasets": selected}).get(
        "dataset_average", {}
    )
    strategies = average.get("strategies", {}) if isinstance(average, dict) else {}
    if isinstance(strategies, dict):
        for strategy_key, strategy_payload in strategies.items():
            if not str(strategy_key).startswith("no_recovery") or not isinstance(
                strategy_payload, dict
            ):
                continue
            points = strategy_payload.get("points", {})
            if isinstance(points, dict):
                strategy_payload["points"] = {
                    str(drop_key): point
                    for drop_key, point in points.items()
                    if str(drop_key).isdigit()
                    and int(drop_key) % 2 == 0
                }
    return average


def _figure_to_rgb(fig: Any) -> np.ndarray:
    buffer = io.BytesIO()
    # Keep every tile on the same fixed canvas. Cropping to visible artists
    # makes a tile with a y-axis title wider, so its actual axes are scaled
    # smaller than otherwise identical tiles when the matrix is composed.
    fig.savefig(buffer, format="png", dpi=150)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _label_recovery_detail_figure(fig: Any, label: str) -> None:
    """Use Ablation Size-style metric y-labels and reserve cell eight for the row label."""
    axes = list(getattr(fig, "axes", ()))
    for axis, metric_label in zip(axes, RECOVERY_DETAIL_COLUMN_LABELS):
        axis.set_title("")
        move_y_axis_label_to_top(
            axis,
            metric_label,
            fontsize=RECOVERY_DETAIL_SUBPLOT_FONT_SIZE,
        )
    if len(axes) < 8:
        return
    label_axis = axes[7]
    label_axis.axis("off")
    label_parts = [
        part.replace("\n", " ")
        for part in str(label).split(" · ", 2)
    ]
    if len(label_parts) == 3:
        for y, part in zip((0.74, 0.50, 0.26), label_parts):
            label_axis.text(
                0.5,
                y,
                part,
                ha="center",
                va="center",
                wrap=False,
                transform=label_axis.transAxes,
                fontsize=RECOVERY_DETAIL_OUTER_FONT_SIZE,
                fontweight="bold",
            )
    else:
        label_axis.text(
            0.5,
            0.5,
            " · ".join(label_parts),
            ha="center",
            va="center",
            wrap=False,
            transform=label_axis.transAxes,
            fontsize=RECOVERY_DETAIL_OUTER_FONT_SIZE,
            fontweight="bold",
        )


def _render_recovery_legend(
    entries: Sequence[LegendEntry],
    font_size: int,
    *,
    vertical: bool = False,
) -> Any:
    method_entries = []
    source_entries = []
    for handle, label in entries:
        source_entry = (
            hasattr(handle, "get_gid")
            and handle.get_gid() == _COMPARISON_SOURCE_LEGEND_GID
        )
        rendered_entry = (
            Line2D(
                [], [],
                color=handle.get_color(),
                linestyle=handle.get_linestyle() if source_entry else "None",
                linewidth=handle.get_linewidth() if source_entry else 0.0,
                marker="None" if source_entry else "o",
                markersize=8,
            ),
            label.split(" (")[0] if label.startswith("Ours (") else label,
        )
        (source_entries if source_entry else method_entries).append(
            rendered_entry
        )
    legend_entries = [*method_entries, *source_entries]
    return render_shared_legend_banner(
        legend_entries, font_size_pixels=font_size,
        max_columns=1 if vertical else max(1, len(legend_entries)),
        handlelength=1.0, columnspacing=0.9,
        fit_single_row=not vertical,
        frameon=False,
    )


def _keep_only_first_column_ylabel(
    fig: Any, metric_grid: Sequence[Sequence[str]]
) -> None:
    """Remove repeated y-axis titles while retaining the first plot column."""
    columns = len(metric_grid[0]) if metric_grid and metric_grid[0] else 0
    if columns <= 1:
        return
    for axis_index, axis in enumerate(getattr(fig, "axes", ())):
        if axis_index % columns:
            axis.set_ylabel("")


def _pruning_only_recovery_like_payload(
    pruning_payload: Dict[str, Any],
    *,
    dataset: str,
    prune_strategy: str = "",
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    """Build a recovery-like plotting payload from pruning-only summaries.

    This keeps the same rendering path as normal recovery plots so that
    missing-recovery backbones still show upper + no-recovery curves.
    """
    groups = _effective_dataset_average_groups(dataset_average_groups)
    if dataset == "mean":
        source = pruning_payload.get("dataset_average", {})
    elif dataset in groups:
        source = _dataset_group_average(
            pruning_payload,
            dataset,
            groups,
        )
    else:
        datasets = pruning_payload.get("datasets", {})
        source = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    if not isinstance(source, dict):
        return {}

    source_strategies = source.get("strategies", {})
    if not isinstance(source_strategies, dict):
        source_strategies = {}

    # Mirror the same baseline family names used by recovery overview.
    mapping: List[Tuple[str, str, str]] = []
    effective_prune_strategy = "" if prune_strategy == "legacy" else prune_strategy
    if effective_prune_strategy and effective_prune_strategy in source_strategies:
        source_strategy = source_strategies.get(effective_prune_strategy, {})
        display_name = (
            str(source_strategy.get("display_name", effective_prune_strategy))
            if isinstance(source_strategy, dict)
            else effective_prune_strategy
        )
        mapping.append((effective_prune_strategy, "no_recovery", f"No recovery · {display_name}"))
    elif "CKA_single" in source_strategies:
        mapping.append(("CKA_single", "no_recovery_single", "No recovery · CKA single"))
    if not effective_prune_strategy and "CKA_double" in source_strategies:
        mapping.append(("CKA_double", "no_recovery_double", "No recovery · CKA double"))
    strategies: Dict[str, Any] = {}
    for src_key, dst_key, display_name in mapping:
        src_payload = source_strategies.get(src_key, {})
        points = src_payload.get("points", {}) if isinstance(src_payload, dict) else {}
        if not isinstance(points, dict) or not points:
            continue
        strategies[dst_key] = {
            "display_name": display_name,
            "points": points,
        }

    return {
        "upper_average": source.get("upper_average", {}),
        "normalized_upper_average": source.get(
            "normalized_upper_average", {}
        ),
        "normalization_bounds": source.get("normalization_bounds", {}),
        "strategies": strategies,
    }


def render_recovery_overview(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    dataset: str = "training average",
    normalize_metrics: bool = True,
    prune_strategy: str = "",
    *,
    detail: bool = False,
    label_only_last_backbone: bool = False,
    show_bottom_xlabel: bool = True,
    show_strategy_title: bool = False,
    show_metric_ylabels: bool = True,
    label_only_first_backbone_yaxis: bool = False,
    shared_legend_entries: List[LegendEntry] | None = None,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_output_root: Path | None = None,
    detail_show_prune_strategy_label: bool = False,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Reload summaries and redraw one recovery figure per backbone."""
    groups = _effective_dataset_average_groups(dataset_average_groups)
    comparison_groups = (
        groups
        if comparison_dataset_average_groups is None
        else _effective_dataset_average_groups(
            comparison_dataset_average_groups
        )
    )
    calibration_groups = (
        groups
        if calibration_dataset_average_groups is None
        else _effective_dataset_average_groups(
            calibration_dataset_average_groups
        )
    )
    comparison_calibration_groups = (
        comparison_groups
        if comparison_calibration_dataset_average_groups is None
        else _effective_dataset_average_groups(
            comparison_calibration_dataset_average_groups
        )
    )
    metric_grid = (
        _RECOVERY_DETAIL_METRIC_GRID
        if detail
        else _RECOVERY_OVERVIEW_METRIC_GRID
    )
    subplot_font_size = (
        RECOVERY_DETAIL_SUBPLOT_FONT_SIZE
        if detail
        else RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE
    )
    backbones = sorted(
        dict.fromkeys(str(name) for name in (selected_backbones or [])),
        key=lambda name: (
            (-1, "")
            if name == _MEAN_BACKBONE
            else backbone_publication_sort_key(name)
        ),
    )
    methods = list(dict.fromkeys(str(name) for name in (selected_methods or [])))
    dataset = str(dataset or "training average").strip()
    prune_strategy = str(prune_strategy or "").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not methods:
        return [], "Select at least one recovery method."

    available_backbones, _ = discover_recovery_overview_options(Path(output_root))
    if _MEAN_BACKBONE in backbones and not available_backbones:
        return [], "No backbones are available to average."

    strategy_candidates = _prune_strategy_candidates(prune_strategy)
    minimum_view = prune_strategy in _MIN_PRUNE_STRATEGY_CANDIDATES
    comparison_contributors: List[str] = []
    for backbone in backbones:
        contributors = (
            available_backbones
            if backbone == _MEAN_BACKBONE
            else [backbone]
        )
        for contributor in contributors:
            if contributor not in comparison_contributors:
                comparison_contributors.append(contributor)

    raw_pruning_by_contributor: Dict[str, Dict[str, Any]] = {}
    collection_errors: Dict[str, str] = {}
    for contributor in comparison_contributors:
        backbone_root = Path(output_root) / contributor
        try:
            raw_pruning_by_contributor[contributor] = (
                collect_pruning_eval_payload(
                    experiment_root=backbone_root,
                    prune_plan_root=backbone_root,
                )
            )
        except Exception as exc:
            collection_errors[contributor] = f"{type(exc).__name__}: {exc}"
    normalization_root = (
        Path(comparison_output_root)
        if comparison_output_root is not None
        else Path(output_root)
    )
    if normalization_root == Path(output_root):
        normalization_pruning_by_contributor = raw_pruning_by_contributor
    else:
        normalization_pruning_by_contributor: Dict[str, Dict[str, Any]] = {}
        for contributor in comparison_contributors:
            try:
                normalization_pruning_by_contributor[contributor] = (
                    collect_pruning_eval_payload(
                        experiment_root=normalization_root / contributor,
                        prune_plan_root=normalization_root / contributor,
                    )
                )
            except Exception as exc:
                collection_errors[contributor] = (
                    "Recovery Overview normalization source: "
                    f"{type(exc).__name__}: {exc}"
                )
    backbone_profiles = _collect_backbone_normalization_profiles(
        normalization_pruning_by_contributor
    )
    profile_drop_counts = sorted(
        {
            int(profile["max_drop_budget"])
            for dataset_profiles in backbone_profiles.values()
            if isinstance(dataset_profiles, dict)
            for profile in dataset_profiles.values()
            if isinstance(profile, dict) and isinstance(
                profile.get("max_drop_budget"), (int, float)
            )
        }
    )
    pruning_by_contributor = {
        contributor: _apply_backbone_normalization_profiles(
            pruning,
            backbone_profiles,
            backbone=contributor,
            use_profile_upper=comparison_output_root is not None,
        )
        for contributor, pruning in raw_pruning_by_contributor.items()
    }
    recovery_by_contributor_and_candidate: Dict[
        Tuple[str, str], Dict[str, Any]
    ] = {}
    recovery_collection_errors: Dict[Tuple[str, str], str] = {}
    for contributor, pruning in pruning_by_contributor.items():
        backbone_root = Path(output_root) / contributor
        for candidate in strategy_candidates:
            try:
                recovery = collect_recovery_eval_payload(
                    experiment_root=backbone_root,
                    prune_strategy=(
                        None if candidate == "legacy" else candidate
                    ),
                )
                recovery = add_recovery_comparison_strategies(
                    recovery,
                    pruning,
                    None if candidate == "legacy" else candidate,
                )
                recovery = _apply_backbone_recovery_method_aliases(
                    recovery, backbone=contributor
                )
                recovery_by_contributor_and_candidate[(
                    contributor,
                    candidate,
                )] = _apply_backbone_normalization_profiles(
                    recovery,
                    backbone_profiles,
                    backbone=contributor,
                    use_profile_upper=comparison_output_root is not None,
                )
            except Exception as exc:
                recovery_collection_errors[(contributor, candidate)] = (
                    f"{type(exc).__name__}: {exc}"
                )

    comparison_recovery_by_contributor_and_candidate: Dict[
        Tuple[str, str], Dict[str, Any]
    ] = {}
    comparison_collection_errors: Dict[Tuple[str, str], str] = {}
    if comparison_output_root is not None:
        for contributor, pruning in normalization_pruning_by_contributor.items():
            backbone_root = Path(comparison_output_root) / contributor
            normalized_pruning = _apply_backbone_normalization_profiles(
                pruning,
                backbone_profiles,
                backbone=contributor,
                use_profile_upper=True,
            )
            for candidate in strategy_candidates:
                try:
                    recovery = collect_recovery_eval_payload(
                        experiment_root=backbone_root,
                        prune_strategy=(None if candidate == "legacy" else candidate),
                    )
                    recovery = add_recovery_comparison_strategies(
                        recovery,
                        normalized_pruning,
                        None if candidate == "legacy" else candidate,
                    )
                    recovery = _apply_backbone_recovery_method_aliases(
                        recovery, backbone=contributor
                    )
                    comparison_recovery_by_contributor_and_candidate[
                        (contributor, candidate)
                    ] = _apply_backbone_normalization_profiles(
                        recovery,
                        backbone_profiles,
                        backbone=contributor,
                        use_profile_upper=True,
                    )
                except Exception as exc:
                    comparison_collection_errors[(contributor, candidate)] = (
                        f"{type(exc).__name__}: {exc}"
                    )

    gallery: List[Tuple[np.ndarray, str]] = []
    warnings: List[str] = []
    messages: List[str] = []
    for backbone_index, backbone in enumerate(backbones):
        try:
            contributors = (
                available_backbones
                if backbone == _MEAN_BACKBONE
                else [backbone]
            )
            contributor_plots: List[Dict[str, Any]] = []
            contributor_fallbacks: List[Dict[str, Any]] = []
            comparison_contributor_plots: List[Dict[str, Any]] = []
            skipped_profiles: List[str] = []
            for contributor in contributors:
                if contributor in collection_errors:
                    raise RuntimeError(collection_errors[contributor])
                pruning = pruning_by_contributor.get(contributor, {})
                candidate_plots: List[Tuple[str, Dict[str, Any]]] = []
                candidate_fallbacks: List[Tuple[str, Dict[str, Any]]] = []
                comparison_candidate_plots: List[Tuple[str, Dict[str, Any]]] = []
                calibration_candidate_plots: List[Tuple[str, Dict[str, Any]]] = []
                calibration_candidate_fallbacks: List[Tuple[str, Dict[str, Any]]] = []
                comparison_calibration_candidate_plots: List[
                    Tuple[str, Dict[str, Any]]
                ] = []
                for candidate in strategy_candidates:
                    error = recovery_collection_errors.get((contributor, candidate))
                    if error:
                        raise RuntimeError(error)
                    recovery = recovery_by_contributor_and_candidate.get(
                        (contributor, candidate), {}
                    )
                    if dataset in groups:
                        recovery_datasets = recovery.get("datasets", {})
                        for method in methods:
                            method_datasets = (
                                [
                                    name for name, payload in recovery_datasets.items()
                                    if isinstance(payload, dict)
                                    and isinstance(payload.get("strategies"), dict)
                                    and method in payload["strategies"]
                                ] if isinstance(recovery_datasets, dict) else []
                            )
                            warning = format_incomplete_average_warning(
                                owner=f"{contributor} / {method} / {candidate}",
                                scope=dataset,
                                expected=groups[dataset],
                                available=method_datasets,
                            )
                            if warning:
                                warnings.append(warning)
                    skipped_profiles.extend(
                        f"{contributor}:{profile}"
                        for profile in recovery.get("skipped_normalization_datasets", [])
                    )
                    if dataset == "mean":
                        candidate_plot = recovery.get("dataset_average", {})
                    elif dataset in groups:
                        candidate_plot = _dataset_group_average(
                            recovery,
                            dataset,
                            groups,
                        )
                    else:
                        dataset_payloads = recovery.get("datasets", {})
                        candidate_plot = (
                            dataset_payloads.get(dataset, {})
                            if isinstance(dataset_payloads, dict) else {}
                        )
                    candidate_plots.append((candidate, candidate_plot))
                    calibration_candidate_plots.append((
                        candidate,
                        _dataset_group_average(
                            recovery,
                            "training average",
                            calibration_groups,
                        ),
                    ))
                    comparison_recovery = (
                        comparison_recovery_by_contributor_and_candidate.get(
                            (contributor, candidate), {}
                        )
                    )
                    if comparison_recovery:
                        if dataset == "mean":
                            comparison_plot = comparison_recovery.get(
                                "dataset_average", {}
                            )
                        elif dataset in comparison_groups:
                            comparison_plot = _dataset_group_average(
                                comparison_recovery,
                                dataset,
                                comparison_groups,
                            )
                        else:
                            comparison_datasets = comparison_recovery.get(
                                "datasets", {}
                            )
                            comparison_plot = (
                                comparison_datasets.get(dataset, {})
                                if isinstance(comparison_datasets, dict)
                                else {}
                            )
                        comparison_candidate_plots.append(
                            (candidate, comparison_plot)
                        )
                        comparison_calibration_candidate_plots.append((
                            candidate,
                            _dataset_group_average(
                                comparison_recovery,
                                "training average",
                                comparison_calibration_groups,
                            ),
                        ))
                    candidate_fallback = _pruning_only_recovery_like_payload(
                            pruning,
                            dataset=dataset,
                            prune_strategy=candidate,
                            dataset_average_groups=groups,
                        )
                    candidate_fallbacks.append((candidate, candidate_fallback))
                    calibration_candidate_fallbacks.append((
                        candidate,
                        _pruning_only_recovery_like_payload(
                            pruning,
                            dataset="training average",
                            prune_strategy=candidate,
                            dataset_average_groups=calibration_groups,
                        ),
                    ))
                title_scope = (
                    "averaged over datasets"
                    if dataset == "mean"
                    else dataset_display_name(dataset)
                )
                calibration_selection = _minimum_strategy_selection(
                    calibration_candidate_plots
                )
                fallback_calibration_selection = _minimum_strategy_selection(
                    calibration_candidate_fallbacks
                )
                comparison_calibration_selection = _minimum_strategy_selection(
                    comparison_calibration_candidate_plots
                )
                contributor_plots.append(
                    _minimum_strategy_plot_payload(
                        candidate_plots, calibration_selection
                    )
                    if minimum_view else candidate_plots[0][1]
                )
                contributor_fallbacks.append(
                    _minimum_strategy_plot_payload(
                        candidate_fallbacks, fallback_calibration_selection
                    )
                    if minimum_view else candidate_fallbacks[0][1]
                )
                comparison_contributor_plots.append(
                    _minimum_strategy_plot_payload(
                        comparison_candidate_plots,
                        comparison_calibration_selection,
                    )
                    if minimum_view and comparison_candidate_plots
                    else (
                        comparison_candidate_plots[0][1]
                        if comparison_candidate_plots
                        else {}
                    )
                )

            plot_payload = (
                _mean_backbone_plot_payloads(contributor_plots)
                if backbone == _MEAN_BACKBONE else contributor_plots[0]
            )
            comparison_plot_payload = (
                _mean_backbone_plot_payloads(comparison_contributor_plots)
                if backbone == _MEAN_BACKBONE
                else (
                    comparison_contributor_plots[0]
                    if comparison_contributor_plots
                    else {}
                )
            )
            available = (
                plot_payload.get("strategies", {})
                if isinstance(plot_payload, dict)
                else {}
            )
            comparison_available = (
                comparison_plot_payload.get("strategies", {})
                if isinstance(comparison_plot_payload, dict)
                else {}
            )
            selected = [
                key for key in available
                if isinstance(available, dict) and str(key).startswith("no_recovery")
            ] + [
                key for key in methods
                if isinstance(available, dict) and key in available
            ]
            strategy_style_overrides: Dict[str, Dict[str, Any]] = (
                {_AUTOGROUP_METHOD: {"linewidth": _OURS_LINEWIDTH}}
                if _AUTOGROUP_METHOD in selected
                else {}
            )
            if comparison_output_root is not None:
                primary_source_label = "Final"
                comparison_source_label = "Recovery Overview"
                if (
                    detail
                    and dataset in groups
                    and dataset in comparison_groups
                    and tuple(groups[dataset])
                    != tuple(comparison_groups[dataset])
                ):
                    primary_source_label = _dataset_count_average_label(
                        dataset, groups[dataset]
                    )
                    comparison_source_label = _dataset_count_average_label(
                        dataset, comparison_groups[dataset]
                    )
                plot_payload, selected, strategy_style_overrides = (
                    _overlay_recovery_overview_methods(
                        plot_payload,
                        comparison_plot_payload,
                        methods,
                        primary_source_label=primary_source_label,
                        comparison_source_label=comparison_source_label,
                    )
                )
            else:
                plot_payload = _formalize_recovery_legend_names(plot_payload)
            fig = render_strategy_metric_figure(
                payload=plot_payload,
                title=f"{backbone_display_name(backbone)}: {prune_strategy or 'legacy'} recovery · {title_scope}",
                metric_grid=metric_grid,
                strategy_keys=selected,
                strategy_style_overrides=strategy_style_overrides,
                draw_solid_before_dashed=True,
                default_linewidth=3.2,
                upper_linewidth=3.2,
                legend_fontsize=subplot_font_size,
                legend_fontweight="bold",
                legend_loc="upper left",
                upper_label="Full Model",
                legend_order=_RECOVERY_LEGEND_ORDER,
                normalize_metrics=bool(normalize_metrics),
                always_normalize_metric_mean=True,
                include_upper_anchor=False,
                subplot_width=PUBLICATION_METRIC_SUBPLOT_WIDTH,
                subplot_box_aspect=(
                    PUBLICATION_DETAIL_AXES_BOX_ASPECT
                    if detail
                    else PUBLICATION_COMPARISON_AXES_BOX_ASPECT
                ),
                figure_height_per_row=(
                    PUBLICATION_DETAIL_SUBPLOT_HEIGHT
                    if detail
                    else PUBLICATION_COMPARISON_SUBPLOT_HEIGHT
                ),
                show_figure_title=False,
                show_metric_titles=False,
                metric_labels_as_ylabels=(
                    show_metric_ylabels
                    and (
                        not label_only_first_backbone_yaxis
                        or backbone_index == 0
                    )
                ),
                metric_ylabel=(
                    "Normalized Degradation" if not detail else None
                ),
                y_tick_percentage=not detail,
                y_tick_max_decimals=2 if detail else None,
                cap_y_at_normalization_max=False,
                minimum_y_axis_upper=None,
                fixed_y_axis_upper=(
                    None if detail else RECOVERY_OVERVIEW_Y_AXIS_UPPER
                ),
                shared_xlabel=None,
                bottom_row_xlabel=(
                    "Dropped Layers" if detail or show_bottom_xlabel else None
                ),
                all_axes_xlabel=detail,
                x_tick_step=4 if detail else 2,
                tick_label_fontsize=subplot_font_size,
                axis_label_fontsize=subplot_font_size,
                legend_markerscale=1.5,
                legend_handlelength=2.8,
                fixed_layout_margins=(
                    None
                    if detail
                    else (0.10, 0.09, 0.99, 0.99)
                ),
                fixed_layout_wspace=None,
            )
            if fig is None:
                fallback_plot_payload = (
                    _mean_backbone_plot_payloads(contributor_fallbacks)
                    if backbone == _MEAN_BACKBONE else contributor_fallbacks[0]
                )
                fallback_available = (
                    fallback_plot_payload.get("strategies", {})
                    if isinstance(fallback_plot_payload, dict)
                    else {}
                )
                fallback_selected = [
                    key
                    for key in (*_BASELINE_STRATEGIES, *methods)
                    if isinstance(fallback_available, dict) and key in fallback_available
                ]
                fallback_plot_payload = _formalize_recovery_legend_names(
                    fallback_plot_payload
                )
                fig = render_strategy_metric_figure(
                    payload=fallback_plot_payload,
                    title=f"{backbone_display_name(backbone)}: {prune_strategy or 'legacy'} recovery · {title_scope}",
                    metric_grid=metric_grid,
                    strategy_keys=fallback_selected if fallback_selected else None,
                    strategy_style_overrides=(
                        {_AUTOGROUP_METHOD: {"linewidth": _OURS_LINEWIDTH}}
                        if _AUTOGROUP_METHOD in fallback_selected
                        else {}
                    ),
                    default_linewidth=3.2,
                    upper_linewidth=3.2,
                    legend_fontsize=subplot_font_size,
                    legend_fontweight="bold",
                    legend_loc="upper left",
                    upper_label="Full Model",
                    legend_order=_RECOVERY_LEGEND_ORDER,
                    normalize_metrics=bool(normalize_metrics),
                    always_normalize_metric_mean=True,
                    include_upper_anchor=False,
                    subplot_width=PUBLICATION_METRIC_SUBPLOT_WIDTH,
                    subplot_box_aspect=(
                        PUBLICATION_DETAIL_AXES_BOX_ASPECT
                        if detail
                        else PUBLICATION_COMPARISON_AXES_BOX_ASPECT
                    ),
                    figure_height_per_row=(
                        PUBLICATION_DETAIL_SUBPLOT_HEIGHT
                        if detail
                        else PUBLICATION_COMPARISON_SUBPLOT_HEIGHT
                    ),
                    show_figure_title=False,
                    show_metric_titles=False,
                    metric_labels_as_ylabels=(
                        show_metric_ylabels
                        and (
                            not label_only_first_backbone_yaxis
                            or backbone_index == 0
                        )
                    ),
                    metric_ylabel=(
                        "Normalized Degradation" if not detail else None
                    ),
                    y_tick_percentage=not detail,
                    y_tick_max_decimals=2 if detail else None,
                    cap_y_at_normalization_max=False,
                    minimum_y_axis_upper=None,
                    fixed_y_axis_upper=(
                        None if detail else RECOVERY_OVERVIEW_Y_AXIS_UPPER
                    ),
                    shared_xlabel=None,
                    bottom_row_xlabel=(
                        "Dropped Layers" if detail or show_bottom_xlabel else None
                    ),
                    all_axes_xlabel=detail,
                    x_tick_step=4 if detail else 2,
                    tick_label_fontsize=subplot_font_size,
                    axis_label_fontsize=subplot_font_size,
                    legend_markerscale=1.5,
                    legend_handlelength=2.8,
                    fixed_layout_margins=(
                        None
                        if detail
                        else (0.10, 0.09, 0.99, 0.99)
                    ),
                    fixed_layout_wspace=None,
                )

                if fig is None:
                    messages.append(
                        f"{backbone}: no matching recovery data (and no pruning baseline fallback)"
                    )
                    continue

                messages.append(
                    f"{backbone}: recovery missing; rendered upper + no-recovery baselines from pruning summaries"
                )
            if hasattr(fig, "findobj"):
                emphasize_figure(fig)
            if shared_legend_entries is not None:
                collect_and_remove_legends(fig, shared_legend_entries)
                if comparison_output_root is not None:
                    _append_comparison_source_legend_entries(
                        shared_legend_entries,
                        primary_source_label,
                        comparison_source_label,
                    )
            _keep_only_first_column_ylabel(fig, metric_grid)
            if detail:
                detail_label_parts = [backbone_display_name(backbone)]
                if detail_show_prune_strategy_label:
                    detail_label_parts.append(
                        _RECOVERY_DETAIL_PRUNE_STRATEGY_DISPLAY_NAMES.get(
                            prune_strategy, prune_strategy
                        )
                    )
                _label_recovery_detail_figure(
                    fig,
                    " · ".join(detail_label_parts),
                )
            compact_overview = not detail and shared_legend_entries is not None
            if compact_overview:
                axis = fig.axes[0]
                axis.set_ylabel(
                    "Normalized Degradation",
                    fontsize=RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
                    fontweight="bold",
                    labelpad=PRUNING_PLAN_OVERVIEW_YLABEL_PAD,
                )
                if not show_metric_ylabels or (
                    label_only_first_backbone_yaxis and backbone_index != 0
                ):
                    axis.yaxis.label.set_alpha(0.0)
                axis.xaxis.label.set_fontweight("bold")
                if show_strategy_title:
                    axis.set_title(
                        _MIN_STRATEGY_DISPLAY_NAMES.get(prune_strategy, prune_strategy),
                        fontsize=RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
                        fontweight="bold",
                    )
                fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99), pad=0.7)
            try:
                gallery.append((
                    _compact_pruning_figure(fig, compact=True)
                    if compact_overview else _figure_to_rgb(fig),
                    backbone_display_name(backbone),
                ))
            finally:
                plt.close(fig)
            missing = [method for method in methods if method not in available]
            suffix = f" (missing: {', '.join(missing)})" if missing else ""
            missing_comparison = [
                method for method in methods
                if method in available
                and (
                    not isinstance(comparison_available, dict)
                    or method not in comparison_available
                )
            ]
            if missing_comparison:
                suffix += (
                    " (missing Recovery Overview comparison: "
                    f"{', '.join(missing_comparison)})"
                )
            if skipped_profiles:
                suffix += f" (missing pruning profile: {', '.join(skipped_profiles)})"
            normalization_source = (
                " | normalization_source=recovery_overview"
                if comparison_output_root is not None
                else ""
            )
            messages.append(
                f"{backbone} / {dataset_display_name(dataset)} / "
                f"{prune_strategy or 'legacy'}: redrawn from current summaries "
                f"| normalized={bool(normalize_metrics)}"
                " | normalization=stage2_brute_force_single_interval_raw_metric_max"
                f"{normalization_source}"
                f" | normalization_drop={','.join(map(str, profile_drop_counts)) or 'unavailable'}"
                f" | backbones={','.join(contributors)}{suffix}"
            )
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    return gallery, "\n".join([*warnings, *messages])


def render_recovery_overview_matrix(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    selected_datasets: Sequence[str] | None = None,
    selected_prune_strategies: Sequence[str] | None = None,
    *,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    default_datasets: Sequence[str] | None = None,
    allowed_prune_strategies: Sequence[str] | None = None,
    default_prune_strategies: Sequence[str] | None = None,
    comparison_output_root: Path | None = None,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Render one dataset row per selected CKA/Retention strategy."""
    available_strategies = tuple(
        dict.fromkeys(
            str(strategy)
            for strategy in (
                _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
                if allowed_prune_strategies is None
                else allowed_prune_strategies
            )
            if str(strategy) in _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
        )
    )
    datasets = (
        list(
            _DEFAULT_RECOVERY_OVERVIEW_DATASETS
            if default_datasets is None
            else default_datasets
        )
        if selected_datasets is None
        else list(
            dict.fromkeys(
                str(dataset).strip()
                for dataset in selected_datasets
                if str(dataset).strip()
            )
        )
    )
    if not datasets:
        return [], "Select at least one dataset."
    prune_strategies = [
        strategy
        for strategy in dict.fromkeys(
            str(item)
            for item in (
                (
                    _DEFAULT_RECOVERY_PRUNE_STRATEGIES
                    if default_prune_strategies is None
                    else default_prune_strategies
                )
                if selected_prune_strategies is None
                else selected_prune_strategies
            )
        )
        if strategy in available_strategies
    ]
    if not prune_strategies:
        return [], "Select at least one prune strategy."
    show_strategy_titles = all(
        strategy in prune_strategies
        for strategy in ("CKA_min", "Retention_min")
    )
    gallery: List[Tuple[np.ndarray, str]] = []
    statuses: List[str] = []
    shared_legend_entries: List[LegendEntry] = []
    for dataset_index, dataset in enumerate(datasets):
        for strategy_index, prune_strategy in enumerate(prune_strategies):
            images, status = render_recovery_overview(
                output_root,
                selected_backbones,
                selected_methods,
                dataset,
                True,
                prune_strategy,
                show_bottom_xlabel=dataset_index == len(datasets) - 1,
                show_strategy_title=(
                    show_strategy_titles and dataset_index == 0
                ),
                show_metric_ylabels=strategy_index == 0,
                label_only_first_backbone_yaxis=True,
                shared_legend_entries=shared_legend_entries,
                dataset_average_groups=dataset_average_groups,
                calibration_dataset_average_groups=(
                    calibration_dataset_average_groups
                ),
                comparison_calibration_dataset_average_groups=(
                    comparison_calibration_dataset_average_groups
                ),
                comparison_output_root=comparison_output_root,
            )
            gallery.extend(
                (
                    image,
                    f"{backbone} · {dataset} · {prune_strategy}",
                )
                for image, backbone in images
            )
            if status:
                statuses.append(status)
    overview_legend_entries = [
        entry for entry in shared_legend_entries if entry[1] != "Full Model"
    ]
    legend = _render_recovery_legend(
        overview_legend_entries, RECOVERY_SHARED_LEGEND_FONT_SIZE,
        vertical=True,
    )
    if legend is not None:
        gallery.append((
            legend,
            f"{SHARED_LEGEND_CAPTION}|entries={len(overview_legend_entries)}",
        ))
    return gallery, "\n".join(statuses)


def render_recovery_detail(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    dataset: str = "training average",
    prune_strategy: str = "",
    *,
    show_bottom_xlabel: bool = True,
    shared_legend_entries: List[LegendEntry] | None = None,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_output_root: Path | None = None,
    show_prune_strategy_label: bool = False,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Render the seven individual metrics in one row per backbone."""
    return render_recovery_overview(
        output_root,
        selected_backbones,
        selected_methods,
        dataset,
        False,
        prune_strategy,
        detail=True,
        label_only_last_backbone=True,
        show_bottom_xlabel=show_bottom_xlabel,
        show_metric_ylabels=True,
        shared_legend_entries=shared_legend_entries,
        dataset_average_groups=dataset_average_groups,
        comparison_dataset_average_groups=comparison_dataset_average_groups,
        calibration_dataset_average_groups=calibration_dataset_average_groups,
        comparison_calibration_dataset_average_groups=(
            comparison_calibration_dataset_average_groups
        ),
        comparison_output_root=comparison_output_root,
        detail_show_prune_strategy_label=show_prune_strategy_label,
    )


def render_recovery_detail_matrix(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    dataset: str,
    prune_strategies: Sequence[str] | None,
    *,
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    allowed_prune_strategies: Sequence[str] | None = None,
    comparison_output_root: Path | None = None,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Render one seven-metric row per backbone and selected prune strategy."""
    available_strategies = tuple(
        _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
        if allowed_prune_strategies is None
        else allowed_prune_strategies
    )
    selected = [
        strategy
        for strategy in dict.fromkeys(
            str(item) for item in (prune_strategies or [])
        )
        if strategy in available_strategies
    ]
    if not selected:
        return [], "Select at least one prune strategy."
    gallery: List[Tuple[np.ndarray, str]] = []
    statuses: List[str] = []
    shared_legend_entries: List[LegendEntry] = []
    for strategy_index, prune_strategy in enumerate(selected):
        images, status = render_recovery_detail(
            output_root,
            selected_backbones,
            selected_methods,
            dataset,
            prune_strategy,
            show_bottom_xlabel=strategy_index == len(selected) - 1,
            shared_legend_entries=shared_legend_entries,
            dataset_average_groups=dataset_average_groups,
            comparison_dataset_average_groups=comparison_dataset_average_groups,
            calibration_dataset_average_groups=(
                calibration_dataset_average_groups
            ),
            comparison_calibration_dataset_average_groups=(
                comparison_calibration_dataset_average_groups
            ),
            comparison_output_root=comparison_output_root,
            show_prune_strategy_label=len(selected) > 1,
        )
        gallery.extend(
            (
                image,
                f"{backbone} · {dataset_display_name(dataset)} · {prune_strategy}",
            )
            for image, backbone in images
        )
        if status:
            statuses.append(status)
    legend = _render_recovery_legend(
        shared_legend_entries, RECOVERY_DETAIL_OUTER_FONT_SIZE,
    )
    if legend is not None:
        gallery.append((legend, SHARED_LEGEND_CAPTION))
    return gallery, "\n".join(statuses)


def build_recovery_tabs(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    view_name: str = "Recovery",
    dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    overview_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    comparison_calibration_dataset_average_groups: Mapping[str, Sequence[str]] | None = None,
    allowed_datasets: Sequence[str] | None = None,
    overview_allowed_datasets: Sequence[str] | None = None,
    allowed_prune_strategies: Sequence[str] | None = None,
    default_overview_datasets: Sequence[str] | None = None,
    comparison_output_root: Path | None = None,
) -> Dict[str, Any]:
    groups = _effective_dataset_average_groups(
        _OVERVIEW_DATASET_AVERAGE_GROUPS
        if dataset_average_groups is None
        else dataset_average_groups
    )
    comparison_groups = (
        groups
        if comparison_dataset_average_groups is None
        else _effective_dataset_average_groups(
            comparison_dataset_average_groups
        )
    )
    overview_groups = _effective_dataset_average_groups(
        groups
        if overview_dataset_average_groups is None
        else overview_dataset_average_groups
    )
    calibration_groups = (
        groups
        if calibration_dataset_average_groups is None
        else _effective_dataset_average_groups(
            calibration_dataset_average_groups
        )
    )
    comparison_calibration_groups = (
        comparison_groups
        if comparison_calibration_dataset_average_groups is None
        else _effective_dataset_average_groups(
            comparison_calibration_dataset_average_groups
        )
    )
    tab_prefix = str(view_name or "Recovery").strip() or "Recovery"
    tab_id_prefix = tab_prefix.lower().replace(" ", "_")
    final_view = tab_prefix.casefold() == "final"
    strategy_options = tuple(
        dict.fromkeys(
            str(strategy)
            for strategy in (
                _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
                if allowed_prune_strategies is None
                else allowed_prune_strategies
            )
            if str(strategy) in _RECOVERY_OVERVIEW_PRUNE_STRATEGIES
        )
    )
    default_prune_strategies = [
        strategy
        for strategy in _DEFAULT_RECOVERY_PRUNE_STRATEGIES
        if strategy in strategy_options
    ] or list(strategy_options[:1])
    overview_dataset_defaults = (
        [str(dataset) for dataset in default_overview_datasets]
        if default_overview_datasets is not None
        else [
            dataset
            for dataset in overview_groups
            if final_view or dataset != "training average"
        ]
    )
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = _discover_recovery_generalization_datasets(
        Path(output_root),
        dataset_average_groups=groups,
        allowed_datasets=allowed_datasets,
    )
    overview_dataset_choices = _discover_recovery_generalization_datasets(
        Path(output_root),
        dataset_average_groups=overview_groups,
        allowed_datasets=allowed_datasets,
    )
    overview_datasets = (
        overview_dataset_choices
        if overview_allowed_datasets is None
        else [
            str(dataset) for dataset in overview_allowed_datasets
            if str(dataset) in set(overview_dataset_choices)
        ]
    )
    default_detail_dataset = (
        next(
            (
                preferred
                for preferred in (
                    "four-dataset testing average",
                    "two-dataset testing average",
                    "same-dataset testing average",
                )
                if preferred in datasets
            ),
            datasets[0] if datasets else "",
        )
    )

    def _detail_dataset_display_choices(
        names: Sequence[str],
    ) -> List[Tuple[str, str]]:
        return dataset_display_choices(names)

    def _default_methods(available: Sequence[str]) -> List[str]:
        return [
            method
            for method in available
            if method not in _DEFAULT_UNCHECKED_RECOVERY_METHODS
        ]

    def _default_overview_datasets(available: Sequence[str]) -> List[str]:
        defaults = [
            dataset
            for dataset in overview_dataset_defaults
            if dataset in available
        ]
        return defaults or list(available[:1])

    def _redraw(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        selected_datasets: Sequence[str],
        selected_prune_strategies: Sequence[str],
        simplify_dataset_names: bool,
    ):
        images, status_value = render_recovery_overview_matrix(
            Path(output_root), selected_backbones, selected_methods,
            selected_datasets, selected_prune_strategies,
            dataset_average_groups=overview_groups,
            calibration_dataset_average_groups=calibration_groups,
            comparison_calibration_dataset_average_groups=(
                comparison_calibration_groups
            ),
            default_datasets=overview_dataset_defaults,
            allowed_prune_strategies=strategy_options,
            default_prune_strategies=default_prune_strategies,
            comparison_output_root=comparison_output_root,
        )
        return (
            _compose_recovery_overview(
                images,
                simplify_dataset_names=bool(simplify_dataset_names),
            ) if images else None,
            status_value,
        )

    def _redraw_detail(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
        prune_strategies: Sequence[str],
    ):
        images, status_value = render_recovery_detail_matrix(
            Path(output_root), selected_backbones, selected_methods, dataset,
            prune_strategies,
            dataset_average_groups=groups,
            comparison_dataset_average_groups=comparison_groups,
            calibration_dataset_average_groups=calibration_groups,
            comparison_calibration_dataset_average_groups=(
                comparison_calibration_groups
            ),
            allowed_prune_strategies=strategy_options,
            comparison_output_root=comparison_output_root,
        )
        return (
            _compose_recovery_detail(images) if images else None,
            status_value,
        )

    def _latest_values(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
        prune_strategies: Sequence[str],
    ) -> Tuple[
        List[str], List[str], List[str], List[str], List[str], str, List[str]
    ]:
        latest_backbones, latest_methods = discover_recovery_overview_options(
            Path(output_root)
        )
        latest_datasets = _discover_recovery_generalization_datasets(
            Path(output_root),
            dataset_average_groups=groups,
            allowed_datasets=allowed_datasets,
        )
        backbone_value = [
            name
            for name in (selected_backbones or [])
            if name == _MEAN_BACKBONE or name in latest_backbones
        ] or latest_backbones
        method_value = [
            name for name in (selected_methods or []) if name in latest_methods
        ] or _default_methods(latest_methods)
        dataset_value = (
            dataset
            if dataset in latest_datasets
            else default_detail_dataset
        )
        prune_strategy_value = [
            strategy
            for strategy in (prune_strategies or [])
            if strategy in strategy_options
        ] or list(default_prune_strategies)
        return (
            latest_backbones,
            latest_methods,
            latest_datasets,
            backbone_value,
            method_value,
            dataset_value,
            prune_strategy_value,
        )

    def _refresh_options(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        selected_datasets: Sequence[str],
        selected_prune_strategies: Sequence[str],
        simplify_dataset_names: bool,
    ):
        latest_backbones, latest_methods = discover_recovery_overview_options(
            Path(output_root)
        )
        latest_datasets = _discover_recovery_generalization_datasets(
            Path(output_root),
            dataset_average_groups=overview_groups,
            allowed_datasets=allowed_datasets,
        )
        latest_overview_datasets = (
            latest_datasets
            if overview_allowed_datasets is None
            else [
                str(item) for item in overview_allowed_datasets
                if str(item) in set(latest_datasets)
            ]
        )
        backbone_value = [
            name
            for name in (selected_backbones or [])
            if name == _MEAN_BACKBONE or name in latest_backbones
        ] or latest_backbones
        method_value = [
            name for name in (selected_methods or []) if name in latest_methods
        ] or _default_methods(latest_methods)
        dataset_value = [
            name for name in (selected_datasets or [])
            if name in latest_overview_datasets
        ] or _default_overview_datasets(latest_overview_datasets)
        prune_strategy_value = [
            strategy
            for strategy in (selected_prune_strategies or [])
            if strategy in strategy_options
        ] or list(default_prune_strategies)
        gallery_value, status_value = render_recovery_overview_matrix(
            Path(output_root), backbone_value, method_value, dataset_value,
            prune_strategy_value,
            dataset_average_groups=overview_groups,
            calibration_dataset_average_groups=calibration_groups,
            comparison_calibration_dataset_average_groups=(
                comparison_calibration_groups
            ),
            default_datasets=overview_dataset_defaults,
            allowed_prune_strategies=strategy_options,
            default_prune_strategies=default_prune_strategies,
            comparison_output_root=comparison_output_root,
        )
        return (
            gr.update(
                choices=[
                    (backbone_display_name(item), item)
                    for item in [_MEAN_BACKBONE, *latest_backbones]
                ],
                value=backbone_value,
            ),
            gr.update(
                choices=[
                    (name.removeprefix("recovery_"), name)
                    for name in latest_methods
                ],
                value=method_value,
            ),
            gr.update(
                choices=dataset_display_choices(latest_overview_datasets),
                value=dataset_value,
            ),
            gr.update(
                choices=[
                    (_MIN_STRATEGY_DISPLAY_NAMES[strategy], strategy)
                    for strategy in strategy_options
                ],
                value=prune_strategy_value,
            ),
            _compose_recovery_overview(
                gallery_value,
                simplify_dataset_names=bool(simplify_dataset_names),
            ) if gallery_value else None,
            status_value,
        )

    def _refresh_detail_options(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
        prune_strategies: Sequence[str],
    ):
        (
            latest_backbones,
            latest_methods,
            latest_datasets,
            backbone_value,
            method_value,
            dataset_value,
            prune_strategy_value,
        ) = _latest_values(
            selected_backbones, selected_methods, dataset, prune_strategies
        )
        gallery_value, status_value = render_recovery_detail_matrix(
            Path(output_root), backbone_value, method_value, dataset_value,
            prune_strategy_value,
            dataset_average_groups=groups,
            comparison_dataset_average_groups=comparison_groups,
            calibration_dataset_average_groups=calibration_groups,
            comparison_calibration_dataset_average_groups=(
                comparison_calibration_groups
            ),
            allowed_prune_strategies=strategy_options,
            comparison_output_root=comparison_output_root,
        )
        return (
            gr.update(
                choices=[
                    (backbone_display_name(item), item)
                    for item in [_MEAN_BACKBONE, *latest_backbones]
                ],
                value=backbone_value,
            ),
            gr.update(
                choices=[
                    (name.removeprefix("recovery_"), name)
                    for name in latest_methods
                ],
                value=method_value,
            ),
            gr.update(
                choices=_detail_dataset_display_choices(latest_datasets),
                value=dataset_value,
            ),
            gr.update(
                choices=[
                    (_MIN_STRATEGY_DISPLAY_NAMES[strategy], strategy)
                    for strategy in strategy_options
                ],
                value=prune_strategy_value,
            ),
            _compose_recovery_detail(gallery_value) if gallery_value else None,
            status_value,
        )

    with gr.Tab(
        tab_prefix, id=f"{tab_id_prefix}_overview"
    ) as tab:
        gr.Markdown(f"## {tab_prefix} Overview")
        gr.Markdown(
            (
                "Compare the four individual test datasets or their four-dataset average. "
                "Solid lines are results "
                "from `main_results`; dashed lines are matching methods and No Recovery "
                "from Recovery Overview. Both use Recovery Overview's per-backbone, per-dataset "
                "normalization ranges. Each source selects the better complete "
                "Retention Single/Double point on the calibration datasets at every "
                "drop count using the raw seven-metric mean with a 0.10 Double-improvement margin, then reuses that "
                "physical pruning choice on every test dataset."
            )
            if final_view
            else (
                "Compare visualization-normalized 7-metric means with one row per selected dataset. "
                "Each backbone occupies one column per selected CKA/Retention strategy. "
                "Every update reloads the latest evaluation summaries; saved PNG files "
                "are not used. Each method, including No Recovery, independently selects "
                "one complete single/double point per drop count on the calibration "
                "datasets using the raw seven-metric mean with a 0.10 Double-improvement margin, then reuses that choice "
                "for all generalization views. Each backbone is normalized independently; for every "
                "dataset and metric, its CKA and Retention columns share the worse of "
                "their max-drop No Recovery degradations."
            )
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[
                        (backbone_display_name(item), item)
                        for item in [_MEAN_BACKBONE, *backbones]
                    ],
                    value=backbones,
                )
                method_cb = gr.CheckboxGroup(
                    label="Recovery methods",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=_default_methods(methods),
                )
                dataset_cb = gr.CheckboxGroup(
                    label="Datasets (multi-select)",
                    choices=dataset_display_choices(overview_datasets),
                    value=_default_overview_datasets(overview_datasets),
                )
                prune_strategy_cb = gr.CheckboxGroup(
                    label="Prune strategies",
                    choices=[
                        (_MIN_STRATEGY_DISPLAY_NAMES[strategy], strategy)
                        for strategy in strategy_options
                    ],
                    value=list(default_prune_strategies),
                )
                simplify_dataset_names = gr.Checkbox(
                    label="Simplify dataset row labels",
                    value=True,
                    info="Show (a), (b), (c), ... instead of full dataset names.",
                )
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=4)
            with gr.Column(scale=5):
                gallery = gr.Image(
                    label=f"{tab_prefix} comparison by dataset and backbone",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=[
                        "results-media",
                        "composite-plot-image",
                    ],
                )

        for event in (
            backbone_cb.change,
            method_cb.change,
            dataset_cb.change,
            prune_strategy_cb.change,
            simplify_dataset_names.change,
        ):
            event(
                _redraw,
                inputs=[
                    backbone_cb,
                    method_cb,
                    dataset_cb,
                    prune_strategy_cb,
                    simplify_dataset_names,
                ],
                outputs=[gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
        refresh_btn.click(
            _refresh_options,
            inputs=[
                backbone_cb,
                method_cb,
                dataset_cb,
                prune_strategy_cb,
                simplify_dataset_names,
            ],
            outputs=[
                backbone_cb,
                method_cb,
                dataset_cb,
                prune_strategy_cb,
                gallery,
                status,
            ],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _redraw,
            inputs=[
                backbone_cb,
                method_cb,
                dataset_cb,
                prune_strategy_cb,
                simplify_dataset_names,
            ],
            outputs=[gallery, status],
            show_progress="minimal",
        )

    with gr.Tab(
        tab_prefix,
        id=f"{tab_id_prefix}_detail",
        elem_id=f"{tab_id_prefix}-detail-tab",
    ) as detail_tab:
        gr.Markdown(f"## {tab_prefix} Detail")
        gr.Markdown(
            (
                "Compare the seven individual recovery metrics from `main_results` "
                "(solid) with the matching Analysis recovery results (dashed) for the "
                "selected training or test split. Each average option uses exactly "
                "the same datasets for both sources; four-dataset and two-dataset "
                "test averages are separate selections. "
                "Each curve selects the better complete Retention Single/Double point "
                "on calibration data using the raw seven-metric mean with a 0.10 Double-improvement margin and reuses "
                "that pruning choice on the selected test split."
            )
            if final_view
            else (
                "Compare the seven individual recovery metrics in CKA and Retention rows "
                "for each backbone; the 7-metric mean is omitted. Each selected strategy "
                "still chooses one complete single/double point per method and drop count "
                "on calibration data using the raw seven-metric mean with a 0.10 Double-improvement margin, then reuses it "
                "on every test split. Each backbone uses its own common "
                "CKA/Retention max-drop No Recovery bound."
            )
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                detail_backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[
                        (backbone_display_name(item), item)
                        for item in [_MEAN_BACKBONE, *backbones]
                    ],
                    value=backbones,
                )
                detail_prune_strategy_dd = gr.CheckboxGroup(
                    label="Prune strategies",
                    choices=[
                        (_MIN_STRATEGY_DISPLAY_NAMES[strategy], strategy)
                        for strategy in strategy_options
                    ],
                    value=list(default_prune_strategies),
                )
                detail_method_cb = gr.CheckboxGroup(
                    label="Recovery methods",
                    choices=[
                        (name.removeprefix("recovery_"), name) for name in methods
                    ],
                    value=_default_methods(methods),
                )
                detail_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=_detail_dataset_display_choices(datasets),
                    value=default_detail_dataset,
                    multiselect=False,
                )
                detail_refresh_btn = gr.Button(
                    "Refresh latest data", variant="primary"
                )
                detail_status = gr.Textbox(
                    label="Status", interactive=False, lines=4
                )
            with gr.Column(scale=5):
                detail_gallery = gr.Image(
                    label=f"{tab_prefix} detail by backbone",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        detail_change_events = [
            detail_backbone_cb.change,
            detail_method_cb.change,
            detail_dataset_dd.change,
            detail_prune_strategy_dd.change,
        ]
        detail_change_inputs = [
            detail_backbone_cb,
            detail_method_cb,
            detail_dataset_dd,
            detail_prune_strategy_dd,
        ]
        for event in detail_change_events:
            event(
                _redraw_detail,
                inputs=detail_change_inputs,
                outputs=[detail_gallery, detail_status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
        detail_refresh_btn.click(
            _refresh_detail_options,
            inputs=detail_change_inputs,
            outputs=[
                detail_backbone_cb,
                detail_method_cb,
                detail_dataset_dd,
                detail_prune_strategy_dd,
                detail_gallery,
                detail_status,
            ],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _redraw_detail,
            inputs=detail_change_inputs,
            outputs=[detail_gallery, detail_status],
            show_progress="minimal",
        )

    result = {
        "tab": tab,
        "detail_tab": detail_tab,
        "backbone_cb": backbone_cb,
        "method_cb": method_cb,
        "dataset_cb": dataset_cb,
        "prune_strategy_cb": prune_strategy_cb,
        "simplify_dataset_names": simplify_dataset_names,
        "gallery": gallery,
        "status": status,
        "detail_backbone_cb": detail_backbone_cb,
        "detail_method_cb": detail_method_cb,
        "detail_prune_strategy_dd": detail_prune_strategy_dd,
        "detail_dataset_dd": detail_dataset_dd,
        "detail_gallery": detail_gallery,
        "detail_status": detail_status,
    }
    return result

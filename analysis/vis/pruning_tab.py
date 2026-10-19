from __future__ import annotations

import copy
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from analysis.metric_labels import SEVEN_METRIC_KEYS, SEVEN_METRIC_LABELS
from analysis.prune_plan import load_prune_plan_bundle
from analysis.prune_plan.eval_plots import (
    PUBLICATION_COMPARISON_SUBPLOT_HEIGHT,
    PUBLICATION_METRIC_SUBPLOT_WIDTH,
    PUBLICATION_COMPARISON_AXES_BOX_ASPECT,
    PUBLICATION_DETAIL_AXES_BOX_ASPECT,
    PUBLICATION_DETAIL_SUBPLOT_HEIGHT,
    PRUNING_MEAN_METRIC_GRID,
    collect_pruning_eval_payload,
    emphasize_figure,
    metric_value,
    move_y_axis_label_to_top,
    pruning_average_strategy_keys,
    pruning_strategy_family,
    pose_selection_score,
    render_strategy_metric_figure,
    strategy_is_single_interval,
)
from analysis.vis.dataset_average_warning import (
    GENERALIZATION_DATASET_AVERAGE_GROUPS,
    dataset_display_name,
    format_incomplete_average_warning,
)
from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
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

# Pruning Plan Overview typography and outer-label spacing (pixels).
PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE = 18
PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE = 42
PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE = 34
PRUNING_PLAN_OVERVIEW_YLABEL_PAD = 4
PRUNING_PLAN_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP = 16
PRUNING_PLAN_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP = 2
PRUNING_PLAN_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP = 12
PRUNING_PLAN_OVERVIEW_EXPORT_PAD_INCHES = 0.02

# Pruning Plan Detail typography and outer-label spacing (pixels).
PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE = 24
PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE = 32
PRUNING_PLAN_DETAIL_LEFT_LABEL_TO_IMAGE_GAP = 0
PRUNING_PLAN_DETAIL_TOP_LABEL_TO_IMAGE_GAP = 2
PRUNING_PLAN_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP = 0
PRUNING_PLAN_DETAIL_COLUMN_LABELS = tuple(
    SEVEN_METRIC_LABELS[key] for key in SEVEN_METRIC_KEYS
)
_PRUNING_DETAIL_METRIC_GRID = (
    SEVEN_METRIC_KEYS[:4],
    SEVEN_METRIC_KEYS[4:],
)


GalleryItems = List[Any]

_BACKBONE_ORDER = ("mapanything_v1", "mapanything", "vggt_omega", "pi3", "vggt")
_AVERAGE_SCOPE_GROUPS = {
    "training": "training average",
    "same_dataset_testing": "same-dataset testing average",
    "cross_dataset_testing": "cross-dataset testing average",
}
_AVERAGE_SCOPES = tuple(_AVERAGE_SCOPE_GROUPS)
_MEAN_SCOPE = "mean"
_MEAN_BACKBONE = "mean"
_DEFAULT_DROP_STEP = 2
_DETAIL_PLOT_GROUPS = (("CKA", "CKA"), ("others", "others"))
_METHOD_CHOICES = (
    ("Similarity", "similarity"),
    ("Baseline", "baseline"),
)
_METHOD_GROUP_INDEX = {
    "similarity": 0,
    "baseline": 1,
}
_OVERVIEW_PLOT_GROUPS = (
    ("double", "CKA"),
    ("others", "others"),
)
# Shared with the Token Ablation tab, where Single keeps distinct geometry.
_OVERVIEW_SINGLE_STYLE = {
    "linewidth": 2.4,
    "alpha": 1.0,
    "zorder": 1.5,
    "markeredgewidth": 1.5,
    "markersize": 49.0,
    "marker_alpha": 1.0,
    "marker_zorder": 1.7,
}
_OTHER_METHOD_LINEWIDTH = 2.4
_EMPHASIZED_METHOD_LINEWIDTH = 3.2
_CKA_TOP_CURVE_ZORDER = 10.0
_CKA_TOP_MARKER_ZORDER = 12.0
_OVERVIEW_LEGEND_LABELS = {
    "retention": "Ours",
    "CKA": "CKA",
    "cosine": "Cosine",
    "inv_mse": "Inv_MSE",
    "cca": "CCA",
    "replaceme": "ReplaceMe",
    "shortgpt": "ShortGPT",
    "gardener": "Gardener",
    "simpler": "SIMPLER",
}
_FIRST_TWO_COLUMN_LEGEND_METHOD_ORDER = (
    "retention",
    "CKA",
    "cosine",
    "cca",
    "inv_mse",
    "replaceme",
    "shortgpt",
    "gardener",
    "simpler",
)
_POINT_METRIC_FIELDS = (
    "metrics_mean",
    "relative_to_upper",
    "normalized_relative_to_upper",
    "raw_metric",
    "normalized_metric",
)
_TRAINING_LEGEND_CAPTION = f"{SHARED_LEGEND_CAPTION}:training"
_TESTING_LEGEND_CAPTION = f"{SHARED_LEGEND_CAPTION}:testing"


def _pruning_overview_figure_kwargs(
    *,
    subplot_font_size: int = PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
    bottom_row_xlabel: str | None = "Dropped Layers",
) -> Dict[str, Any]:
    """Return the shared subplot geometry and typography for overview plots."""
    return {
        "legend_fontsize": subplot_font_size,
        "legend_fontweight": "bold",
        "show_figure_title": False,
        "show_metric_titles": False,
        "shared_ylabel": "Normalized Degradation",
        "y_tick_percentage": True,
        "cap_y_at_normalization_max": False,
        "subplot_width": PUBLICATION_METRIC_SUBPLOT_WIDTH,
        "subplot_box_aspect": PUBLICATION_COMPARISON_AXES_BOX_ASPECT,
        "figure_height_per_row": PUBLICATION_COMPARISON_SUBPLOT_HEIGHT,
        "shared_xlabel": None,
        "bottom_row_xlabel": bottom_row_xlabel,
        "x_tick_step": 2,
        "tick_label_fontsize": subplot_font_size,
        "axis_label_fontsize": subplot_font_size,
        "legend_markerscale": 1.5,
        "legend_handlelength": 2.8,
        "fixed_layout_margins": None,
    }


def _style_pruning_overview_figure(
    fig: Any,
    *,
    backbone_index: int,
    subplot_font_size: int = PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
) -> None:
    """Apply the final shared overview emphasis and axis-label placement."""
    emphasize_figure(fig)
    if getattr(fig, "_supylabel", None) is not None:
        fig._supylabel.remove()
        fig._supylabel = None
    axis = fig.axes[0]
    axis.xaxis.label.set_fontweight("bold")
    axis.set_ylabel(
        "Normalized Degradation",
        fontsize=subplot_font_size,
        fontweight="bold",
        labelpad=PRUNING_PLAN_OVERVIEW_YLABEL_PAD,
    )
    # Keep equal bounds while showing the ylabel only in the first column.
    if backbone_index != 0:
        axis.yaxis.label.set_alpha(0.0)


def _compose_pruning_overview(
    value: Any,
    simplify_dataset_names: bool = False,
    hide_dataset_names: bool = False,
) -> np.ndarray:
    all_cells = gallery_cells(value)
    def _legend_cell(caption: str) -> Cell | None:
        return next(
            (cell for cell in all_cells if cell.caption.startswith(caption)),
            None,
        )

    def _legend_count(cell: Cell | None) -> int:
        if cell is None or "|entries=" not in cell.caption:
            return 1
        return max(1, int(cell.caption.rsplit("|entries=", 1)[1]))

    training_legend_cell = _legend_cell(_TRAINING_LEGEND_CAPTION)
    testing_legend_cell = _legend_cell(_TESTING_LEGEND_CAPTION)
    legends = {
        "training": training_legend_cell.image if training_legend_cell else None,
        "testing": testing_legend_cell.image if testing_legend_cell else None,
    }
    legend_counts = {
        "training": _legend_count(training_legend_cell),
        "testing": _legend_count(testing_legend_cell),
    }
    cells = [
        cell
        for cell in all_cells
        if not cell.caption.startswith(
            (_TRAINING_LEGEND_CAPTION, _TESTING_LEGEND_CAPTION)
        )
    ]
    records: Dict[Tuple[str, str], Cell] = {}
    backbone_labels: List[str] = []
    scope_labels: List[str] = []
    for cell in cells:
        parts = cell.caption.rsplit(" · ", 2) if cell.caption else []
        if len(parts) == 3:
            backbone, scope, _ = parts
        elif len(parts) == 2:
            backbone, scope = parts
        else:
            continue
        if backbone not in backbone_labels:
            backbone_labels.append(backbone)
        if scope not in scope_labels:
            scope_labels.append(scope)
        cell.caption = ""
        records[(backbone, scope)] = cell

    if not records:
        raise ValueError("No pruning-plan overview plots were returned.")

    row_labels = None if hide_dataset_names else (
        [f"({chr(ord('a') + index)})" for index in range(len(scope_labels))]
        if simplify_dataset_names
        else scope_labels
    )

    fixed_rows = [
        [
            records.get((backbone, scope), Cell(None))
            for backbone in backbone_labels
        ]
        for scope in scope_labels
    ]
    for row in fixed_rows:
        for cell in row:
            if cell.image is not None:
                cell.image = crop_horizontal_white_margins(cell.image)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_font = _font(PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE, bold=True)
    label_width = (
        max(
            measure.textbbox((0, 0), label, font=row_font)[2]
            for label in row_labels
        ) + 12
        if row_labels
        else 0
    )
    left_label_gap = (
        PRUNING_PLAN_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP if row_labels else 0
    )
    content_width = (
        2400 - label_width - left_label_gap
    )
    column_gap = 6
    column_weights = [
        max(
            (cell.image.width for row in fixed_rows if (cell := row[column]).image is not None),
            default=1,
        )
        for column in range(len(backbone_labels))
    ]
    available_width = max(
        len(backbone_labels),
        content_width - column_gap * (len(backbone_labels) - 1),
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
                columns=len(backbone_labels),
                cells=[],
                column_width_weights=column_weights,
                column_gaps=[column_gap] * max(0, len(backbone_labels) - 1),
                fixed_rows=fixed_rows,
                row_labels=row_labels,
                row_labels_bold=True,
                column_labels=backbone_labels,
                column_label_centers=[
                    next((
                        float(row[column].image.info["axes_center_x"])
                        for row in reversed(fixed_rows)
                        if row[column].image is not None
                        and "axes_center_x" in row[column].image.info
                    ), 0.5)
                    for column in range(len(backbone_labels))
                ],
                column_labels_below=True,
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
        outer_font_size=PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE,
        left_label_to_image_gap=left_label_gap,
        top_label_to_image_gap=PRUNING_PLAN_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=PRUNING_PLAN_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP,
    )
    positioned_legends: list[tuple[Any, float, float, int]] = []
    if legends.get("training") is not None and row_axis_bounds:
        positioned_legends.append((
            legends["training"],
            row_axis_bounds[0][0] / composite.shape[0],
            row_axis_bounds[0][1] / composite.shape[0],
            legend_counts["training"],
        ))
    if legends.get("testing") is not None and row_axis_bounds:
        testing_start_row = (
            1
            if legends.get("training") is not None
            and len(row_axis_bounds) >= 2
            else 0
        )
        positioned_legends.append((
            legends["testing"],
            row_axis_bounds[testing_start_row][0] / composite.shape[0],
            row_axis_bounds[-1][1] / composite.shape[0],
            legend_counts["testing"],
        ))
    return append_shared_legends_right(
        composite, tuple(positioned_legends), frameon=False
    )


def _compose_pruning_detail(value: Any) -> np.ndarray | None:
    """Stack native 2x4 detail figures without resizing their subplots."""
    all_cells = gallery_cells(value)
    legend_captions = (_TRAINING_LEGEND_CAPTION, _TESTING_LEGEND_CAPTION)
    legends = [
        cell.image
        for caption in legend_captions
        for cell in all_cells
        if cell.caption == caption
    ]
    cells = [cell for cell in all_cells if cell.caption not in legend_captions]
    for cell in cells:
        cell.caption = ""
    sections = [Section(columns=1, cells=[cell]) for cell in cells]
    composite = (
        compose_sections(
            sections,
            outer_font_size=PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE,
            left_label_to_image_gap=PRUNING_PLAN_DETAIL_LEFT_LABEL_TO_IMAGE_GAP,
            top_label_to_image_gap=PRUNING_PLAN_DETAIL_TOP_LABEL_TO_IMAGE_GAP,
            bottom_label_to_image_gap=PRUNING_PLAN_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP,
        )
        if sections
        else None
    )
    if composite is not None:
        for legend in reversed(legends):
            composite = prepend_shared_legend(composite, legend)
    return composite


def _label_pruning_detail_figure(fig: Any, label: str) -> None:
    """Match the Recovery Remaining/Ablation Size 2x4 panel labeling."""
    axes = list(getattr(fig, "axes", ()))
    for axis, metric_label in zip(axes, PRUNING_PLAN_DETAIL_COLUMN_LABELS):
        axis.set_title("")
        move_y_axis_label_to_top(
            axis,
            metric_label,
            fontsize=PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE,
        )
    if len(axes) < 8:
        return
    label_axis = axes[7]
    label_axis.axis("off")
    parts = str(label).split(" · ", 1)
    if len(parts) == 2:
        backbone_label, dataset_label = parts
        label_axis.text(
            0.5,
            0.64,
            backbone_label,
            ha="center",
            va="center",
            wrap=False,
            transform=label_axis.transAxes,
            fontsize=PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE,
            fontweight="bold",
        )
        label_axis.text(
            0.5,
            0.30,
            dataset_label.replace("\n", " "),
            ha="center",
            va="center",
            wrap=False,
            transform=label_axis.transAxes,
            fontsize=PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE,
            fontweight="bold",
        )
    else:
        label_axis.text(
            0.5,
            0.5,
            str(label).replace("\n", " "),
            ha="center",
            va="center",
            wrap=False,
            transform=label_axis.transAxes,
            fontsize=PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE,
            fontweight="bold",
        )


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name)
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _scope_label(scope: str) -> str:
    if scope == _MEAN_SCOPE:
        return "Mean"
    if scope in _AVERAGE_SCOPE_GROUPS:
        return dataset_display_name(_AVERAGE_SCOPE_GROUPS[scope])
    return dataset_display_name(scope)


def _bundle_training_datasets(backbone_root: Path) -> List[str]:
    try:
        bundle = load_prune_plan_bundle(Path(backbone_root))
    except Exception:
        return []
    raw = bundle.get("dataset", "")
    if isinstance(raw, str):
        return [name.strip() for name in raw.split(",") if name.strip()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(name).strip() for name in raw if str(name).strip()]
    return []


def _result_datasets(backbone_root: Path) -> List[str]:
    return sorted(
        {
            path.parent.parent.name
            for path in Path(backbone_root).glob("*/eval/pruning")
            if path.is_dir() and any(path.glob("*/summary.json"))
        }
    )


def discover_pruning_plan_backbones(output_root: Path) -> List[str]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "prune_plan_all.json").is_file()
        and bool(_result_datasets(path))
    ]
    return sorted(set(found), key=_backbone_sort_key)


def discover_pruning_plan_scopes(
    output_root: Path,
    backbone: str,
) -> List[str]:
    datasets = _result_datasets(Path(output_root) / str(backbone))
    return [_MEAN_SCOPE, *datasets]


def resolve_pruning_scope_datasets(
    output_root: Path,
    backbone: str,
    scope: str,
) -> List[str]:
    backbone_root = Path(output_root) / str(backbone)
    available = _result_datasets(backbone_root)
    training = set(_bundle_training_datasets(backbone_root))
    if scope == _MEAN_SCOPE:
        return list(
            dict.fromkeys(
                dataset
                for average_scope in _AVERAGE_SCOPES
                for dataset in resolve_pruning_scope_datasets(
                    output_root, backbone, average_scope
                )
            )
        )
    if scope == "training":
        return [dataset for dataset in available if dataset in training]
    if scope in _AVERAGE_SCOPE_GROUPS:
        expected = _expected_pruning_scope_datasets(backbone_root, scope)
        return [dataset for dataset in expected if dataset in set(available)]
    return [scope] if scope in available else []


def _expected_pruning_scope_datasets(
    backbone_root: Path,
    scope: str,
) -> List[str]:
    """Return the complete dataset set an average is expected to contain."""
    training = _bundle_training_datasets(backbone_root)
    average_key = _AVERAGE_SCOPE_GROUPS.get(scope)
    if scope == "training":
        canonical_training = GENERALIZATION_DATASET_AVERAGE_GROUPS[
            "training average"
        ]
        if set(training).intersection(canonical_training):
            return list(canonical_training)
        return training
    if average_key:
        # Production plans use canonical generalization groups. Keep
        # synthetic/custom plans useful by treating their non-training data as
        # same-dataset tests when no canonical dataset tags are present.
        known_names = {
            dataset
            for datasets in GENERALIZATION_DATASET_AVERAGE_GROUPS.values()
            for dataset in datasets
        }
        available = _result_datasets(backbone_root)
        if set(training).intersection(known_names) or set(available).intersection(known_names):
            return list(GENERALIZATION_DATASET_AVERAGE_GROUPS[average_key])
        if scope == "same_dataset_testing":
            return [dataset for dataset in available if dataset not in set(training)]
        return []
    return [scope]


def _figure_to_rgb(fig: Any, *, compact: bool = False) -> np.ndarray | Image.Image:
    buffer = io.BytesIO()
    # Overview trims unused canvas after removing its per-panel legend.
    # Transparent shared y labels retain equal horizontal bounds across columns.
    # Detail keeps its fixed canvas for consistent seven-metric tile sizes.
    export_options = (
        {"bbox_inches": "tight", "pad_inches": PRUNING_PLAN_OVERVIEW_EXPORT_PAD_INCHES}
        if compact else {}
    )
    if compact:
        fig.set_dpi(150)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.get_tightbbox(renderer).padded(
            PRUNING_PLAN_OVERVIEW_EXPORT_PAD_INCHES
        )
        axes_bounds = fig.axes[0].get_window_extent(renderer)
        axes_center_x = (
            (axes_bounds.x0 + axes_bounds.width / 2) / fig.dpi - bounds.x0
        ) / bounds.width
        axes_center_y = 1.0 - (
            (axes_bounds.y0 + axes_bounds.height / 2) / fig.dpi - bounds.y0
        ) / bounds.height
        axes_top_y = 1.0 - (axes_bounds.y1 / fig.dpi - bounds.y0) / bounds.height
        axes_bottom_y = 1.0 - (axes_bounds.y0 / fig.dpi - bounds.y0) / bounds.height
        export_options["bbox_inches"] = bounds
    fig.savefig(buffer, format="png", dpi=150, **export_options)
    buffer.seek(0)
    with Image.open(buffer) as image:
        if compact:
            result = image.convert("RGB")
            result.info["axes_center_x"] = axes_center_x
            result.info["axes_center_y"] = axes_center_y
            result.info["axes_top_y"] = axes_top_y
            result.info["axes_bottom_y"] = axes_bottom_y
            return result
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _scope_plot_payload(payload: Dict[str, Any], scope: str) -> Dict[str, Any]:
    if scope in _AVERAGE_SCOPES:
        average = payload.get("dataset_average", {})
        return average if isinstance(average, dict) else {}
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    dataset_payload = datasets.get(scope, {})
    return dataset_payload if isinstance(dataset_payload, dict) else {}


def _replaceme_selection(plot_payload: Dict[str, Any]) -> Dict[str, str]:
    """Choose one ReplaceMe structure by the calibration four-pose sum."""
    strategies = plot_payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return {}
    choices: Dict[str, List[Tuple[float, str]]] = {}
    for strategy_key in ("replaceme_single", "replaceme_double"):
        strategy = strategies.get(strategy_key, {})
        points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
        for drop_key, point in points.items() if isinstance(points, dict) else ():
            if not isinstance(point, dict) or not bool(point.get("available", False)):
                continue
            raw = point.get("raw_metric", point.get("metrics_mean", {}))
            score = pose_selection_score(raw) if isinstance(raw, dict) else None
            if score is not None:
                choices.setdefault(str(drop_key), []).append((score, strategy_key))
    return {
        drop_key: min(options, key=lambda item: (item[0], item[1]))[1]
        for drop_key, options in choices.items()
    }


def _merge_replaceme_strategies(
    plot_payload: Dict[str, Any],
    calibration_selection: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Apply one calibration-selected Single/Double ReplaceMe point per budget."""
    strategies = plot_payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return plot_payload
    source_strategies = [
        strategies.get("replaceme_single"),
        strategies.get("replaceme_double"),
    ]
    source_strategies = [item for item in source_strategies if isinstance(item, dict)]
    if not source_strategies:
        return plot_payload

    selection = (
        dict(calibration_selection)
        if calibration_selection is not None
        else _replaceme_selection(plot_payload)
    )
    merged_points: Dict[str, Any] = {}
    for drop_key, strategy_key in sorted(selection.items(), key=lambda item: int(item[0])):
        strategy = strategies.get(strategy_key, {})
        points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
        point = points.get(drop_key) if isinstance(points, dict) else None
        if not isinstance(point, dict) or not bool(point.get("available", False)):
            continue
        merged_point = copy.deepcopy(point)
        merged_point["calibration_selected_prune_strategy"] = strategy_key
        merged_points[drop_key] = merged_point

    merged_payload = dict(plot_payload)
    merged_strategies = dict(strategies)
    merged_strategies.pop("replaceme_single", None)
    merged_strategies.pop("replaceme_double", None)
    merged_strategies["replaceme"] = {
        "display_name": "ReplaceMe (calibration-selected Single/Double)",
        "points": merged_points,
    }
    merged_payload["strategies"] = merged_strategies
    return merged_payload


def _average_metric_maps(maps: Sequence[Any]) -> Dict[str, float]:
    valid = [item for item in maps if isinstance(item, dict)]
    if not valid:
        return {}
    common_keys = set(valid[0])
    for item in valid[1:]:
        common_keys.intersection_update(item)
    averaged: Dict[str, float] = {}
    for key in common_keys:
        values = [item.get(key) for item in valid]
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ):
            averaged[str(key)] = float(np.mean(values))
    return averaged


def _mean_backbone_plot_payloads(
    plot_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average points shared by every backbone payload."""
    if not plot_payloads:
        return {}
    strategy_maps = [payload.get("strategies", {}) for payload in plot_payloads]
    if not all(isinstance(item, dict) for item in strategy_maps):
        return {}
    strategy_keys = set(strategy_maps[0])
    for strategies in strategy_maps[1:]:
        strategy_keys.intersection_update(strategies)

    mean_strategies: Dict[str, Any] = {}
    for strategy_key in sorted(strategy_keys):
        strategy_payloads = [strategies[strategy_key] for strategies in strategy_maps]
        point_maps = [item.get("points", {}) for item in strategy_payloads]
        if not all(isinstance(item, dict) for item in point_maps):
            continue
        drop_keys = set(point_maps[0])
        for points in point_maps[1:]:
            drop_keys.intersection_update(points)
        mean_points: Dict[str, Any] = {}
        for drop_key in sorted(drop_keys, key=lambda value: int(value)):
            points = [point_map[drop_key] for point_map in point_maps]
            if not all(
                isinstance(point, dict) and bool(point.get("available", False))
                for point in points
            ):
                continue
            mean_point: Dict[str, Any] = {
                "available": True,
                "drop_count": int(drop_key),
                "num_backbones": len(points),
            }
            for field in _POINT_METRIC_FIELDS:
                mean_point[field] = _average_metric_maps(
                    [point.get(field, {}) for point in points]
                )
            mean_points[str(drop_key)] = mean_point
        mean_strategies[str(strategy_key)] = {
            "display_name": str(
                strategy_payloads[0].get("display_name", strategy_key)
            ),
            "points": mean_points,
        }
    return {
        "upper_average": _average_metric_maps(
            [payload.get("upper_average", {}) for payload in plot_payloads]
        ),
        "normalized_upper_average": _average_metric_maps(
            [
                payload.get("normalized_upper_average", {})
                for payload in plot_payloads
            ]
        ),
        "strategies": mean_strategies,
        "num_backbones": len(plot_payloads),
    }


def render_pruning_plan_gallery(
    output_root: Path,
    backbone: str | Sequence[str],
    selected_scopes: Sequence[str] | None,
    merge_replaceme: bool = True,
    view: str = "overview",
    detail_method: str = "all",
    overview_method: str = "baseline",
) -> Tuple[GalleryItems, str]:
    """Reload current summaries and render the pruning views in memory."""
    view_key = str(view).strip().lower()
    if view_key not in {"overview", "detail"}:
        raise ValueError(f"Unsupported pruning-plan view: {view!r}")
    detail_method_key = str(detail_method).strip().lower()
    if view_key == "detail" and detail_method_key not in {
        "all",
        *_METHOD_GROUP_INDEX,
    }:
        raise ValueError(f"Unsupported pruning detail method: {detail_method!r}")
    overview_method_key = str(overview_method).strip().lower()
    if view_key == "overview" and overview_method_key not in _METHOD_GROUP_INDEX:
        raise ValueError(f"Unsupported pruning overview method: {overview_method!r}")
    subplot_font_size = (
        PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE
        if view_key == "detail"
        else PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE
    )
    metric_keys = ["gt_mean"]
    requested_scopes = list(
        dict.fromkeys(str(scope) for scope in (selected_scopes or []))
    )
    scopes = list(
        dict.fromkeys(
            expanded_scope
            for scope in requested_scopes
            for expanded_scope in (
                _AVERAGE_SCOPES if scope == _MEAN_SCOPE else (scope,)
            )
        )
    )
    selected_backbones = (
        [str(backbone)]
        if isinstance(backbone, str) and backbone
        else list(dict.fromkeys(str(item) for item in (backbone or []) if str(item)))
    )
    if not selected_backbones:
        return [], "Select at least one backbone."
    selected_backbones = sorted(
        selected_backbones,
        key=lambda name: (
            (-1, "")
            if name == _MEAN_BACKBONE
            else backbone_publication_sort_key(name)
        ),
    )
    if not scopes:
        return [], "Select at least one dataset or average scope."

    available_backbones = discover_pruning_plan_backbones(Path(output_root))
    selected_backbones = [
        item
        for item in selected_backbones
        if item == _MEAN_BACKBONE or item in available_backbones
    ]
    gallery: GalleryItems = []
    shared_legend_entries: Dict[str, List[LegendEntry]] = {
        "training": [],
        "testing": [],
    }
    replaceme_calibration_selections: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    messages: List[str] = []
    for backbone_index, selected_backbone in enumerate(selected_backbones):
        contributors = (
            available_backbones
            if selected_backbone == _MEAN_BACKBONE
            else [selected_backbone]
        )
        for scope_index, scope in enumerate(scopes):
            try:
                contributor_plots: List[Dict[str, Any]] = []
                dataset_labels: List[str] = []
                for contributor in contributors:
                    backbone_root = Path(output_root) / contributor
                    datasets = resolve_pruning_scope_datasets(
                        Path(output_root), contributor, scope
                    )
                    if scope in _AVERAGE_SCOPES:
                        expected = _expected_pruning_scope_datasets(
                            backbone_root, scope
                        )
                        warning = format_incomplete_average_warning(
                            owner=contributor,
                            scope=_scope_label(scope),
                            expected=expected,
                            available=datasets,
                        )
                        if warning:
                            warnings.append(warning)
                    if not datasets:
                        continue
                    payload = collect_pruning_eval_payload(
                        experiment_root=backbone_root,
                        prune_plan_root=backbone_root,
                        datasets=datasets,
                        drop_step=_DEFAULT_DROP_STEP,
                    )
                    contributor_plot = _scope_plot_payload(payload, scope)
                    if merge_replaceme:
                        if contributor not in replaceme_calibration_selections:
                            calibration_datasets = resolve_pruning_scope_datasets(
                                Path(output_root), contributor, "training"
                            )
                            if not calibration_datasets:
                                raise RuntimeError(
                                    f"{contributor}: no calibration datasets are available "
                                    "for ReplaceMe Single/Double selection"
                                )
                            calibration_payload = collect_pruning_eval_payload(
                                experiment_root=backbone_root,
                                prune_plan_root=backbone_root,
                                datasets=calibration_datasets,
                                drop_step=_DEFAULT_DROP_STEP,
                            )
                            calibration_plot = _scope_plot_payload(
                                calibration_payload, "training"
                            )
                            replaceme_calibration_selections[contributor] = (
                                _replaceme_selection(calibration_plot)
                            )
                        contributor_plot = _merge_replaceme_strategies(
                            contributor_plot,
                            replaceme_calibration_selections[contributor],
                        )
                    contributor_plots.append(contributor_plot)
                    dataset_labels.extend(f"{contributor}:{item}" for item in datasets)
                if len(contributor_plots) != len(contributors):
                    messages.append(
                        f"{selected_backbone}/{scope}: missing one or more backbone summaries"
                    )
                    continue
                plot_payload = (
                    _mean_backbone_plot_payloads(contributor_plots)
                    if selected_backbone == _MEAN_BACKBONE
                    else contributor_plots[0]
                )
                strategies = plot_payload.get("strategies", {})
                strategy_keys = tuple(strategies) if isinstance(strategies, dict) else ()
                if view_key == "detail":
                    indexed_plot_groups = list(enumerate(_DETAIL_PLOT_GROUPS))
                    if detail_method_key != "all":
                        selected_group_index = _METHOD_GROUP_INDEX[
                            detail_method_key
                        ]
                        indexed_plot_groups = [
                            indexed_plot_groups[selected_group_index]
                        ]
                else:
                    is_training_scope = scope == "training" or any(
                        scope in _bundle_training_datasets(
                            Path(output_root) / contributor
                        )
                        for contributor in contributors
                    )
                    overview_group_index = _METHOD_GROUP_INDEX[overview_method_key]
                    indexed_plot_groups = [
                        (
                            overview_group_index,
                            _OVERVIEW_PLOT_GROUPS[overview_group_index],
                        )
                    ]
                has_later_rendered_scope = view_key == "overview" and any(
                    all(
                        resolve_pruning_scope_datasets(
                            Path(output_root), contributor, later_scope
                        )
                        for contributor in contributors
                    )
                    for later_scope in scopes[scope_index + 1 :]
                )
                expected_plots = len(indexed_plot_groups)
                rendered = 0
                scope_start = len(gallery)
                for group_index, (plot_group, strategy_group) in indexed_plot_groups:
                    selected = pruning_average_strategy_keys(
                        strategy_keys, strategy_group
                    )
                    style_overrides: Dict[str, Dict[str, Any]] = {}
                    if view_key == "overview" and plot_group in {"single", "double"}:
                        single_keys = tuple(
                            key
                            for key in selected
                            if strategy_is_single_interval(key)
                        )
                        if plot_group == "single":
                            selected = single_keys
                            style_overrides = {
                                key: {"linestyle": "-", "dashes": None}
                                for key in single_keys
                            }
                    for key in selected:
                        key_style = style_overrides.setdefault(key, {})
                        strategy_family = pruning_strategy_family(key)
                        key_style["linewidth"] = (
                            _EMPHASIZED_METHOD_LINEWIDTH
                            if strategy_family in {"retention", "CKA"}
                            else _OTHER_METHOD_LINEWIDTH
                        )
                        if strategy_family == "CKA":
                            key_style["zorder"] = _CKA_TOP_CURVE_ZORDER
                            key_style["marker_zorder"] = _CKA_TOP_MARKER_ZORDER
                    for metric_index, metric_key in enumerate(metric_keys):
                        gallery_column_index = (
                            group_index + 1
                            if view_key == "overview"
                            else 0
                        )
                        metric_grid = (
                            PRUNING_MEAN_METRIC_GRID
                            if view_key == "overview"
                            else _PRUNING_DETAIL_METRIC_GRID
                        )
                        render_payload = plot_payload
                        fig = render_strategy_metric_figure(
                            payload=render_payload,
                            title=(
                                f"{_scope_label(scope)} · "
                                f"{plot_group.title()}"
                            ),
                            metric_grid=metric_grid,
                            strategy_keys=selected,
                            compact_pruning_legend=True,
                            compact_method_labels=_OVERVIEW_LEGEND_LABELS,
                            compact_method_order=(
                                _FIRST_TWO_COLUMN_LEGEND_METHOD_ORDER
                                if gallery_column_index < 2
                                else None
                            ),
                            compact_structure_labels=(
                                ("Single", "Double")
                                if strategy_group == "CKA"
                                else ("Single", "Multi")
                            ),
                            combine_compact_legend=True,
                            normalize_metrics=(
                                metric_key == "gt_mean"
                                if view_key == "overview"
                                else False
                            ),
                            strategy_style_overrides=style_overrides,
                            default_linewidth=_OTHER_METHOD_LINEWIDTH,
                            upper_linewidth=_OTHER_METHOD_LINEWIDTH,
                            draw_solid_before_dashed=True,
                            dodge_structure_markers=(
                                view_key != "overview" or is_training_scope
                            ),
                            metric_labels_as_ylabels=view_key == "detail",
                            y_tick_max_decimals=(
                                2 if view_key == "detail" else None
                            ),
                            fixed_y_axis_upper=(
                                0.4
                                if view_key == "overview"
                                and overview_method_key == "similarity"
                                else None
                            ),
                            all_axes_xlabel=view_key == "detail",
                            **(
                                _pruning_overview_figure_kwargs(
                                    subplot_font_size=subplot_font_size,
                                    bottom_row_xlabel=(
                                        "Dropped Layers"
                                        if not has_later_rendered_scope
                                        else None
                                    ),
                                )
                                if view_key == "overview"
                                else {
                                    "legend_fontsize": subplot_font_size,
                                    "legend_fontweight": "bold",
                                    "show_figure_title": False,
                                    "show_metric_titles": False,
                                    "shared_ylabel": None,
                                    "y_tick_percentage": False,
                                    "cap_y_at_normalization_max": False,
                                    "upper_anchor_bottom_margin_fraction": 0.10,
                                    "subplot_width": PUBLICATION_METRIC_SUBPLOT_WIDTH,
                                    "subplot_box_aspect": PUBLICATION_DETAIL_AXES_BOX_ASPECT,
                                    "figure_height_per_row": PUBLICATION_DETAIL_SUBPLOT_HEIGHT,
                                    "shared_xlabel": None,
                                    "bottom_row_xlabel": "Dropped Layers",
                                    "x_tick_step": 4,
                                    "tick_label_fontsize": subplot_font_size,
                                    "axis_label_fontsize": subplot_font_size,
                                    "legend_markerscale": 1.5,
                                    "legend_handlelength": 2.8,
                                    "fixed_layout_margins": None,
                                }
                            ),
                        )
                        if fig is None:
                            continue
                        if view_key == "detail":
                            detail_label = backbone_display_name(
                                selected_backbone
                            )
                            if len(scopes) > 1:
                                detail_label += f" · {_scope_label(scope)}"
                            _label_pruning_detail_figure(
                                fig,
                                detail_label,
                            )
                        if view_key == "overview":
                            _style_pruning_overview_figure(
                                fig,
                                backbone_index=backbone_index,
                                subplot_font_size=subplot_font_size,
                            )
                        else:
                            emphasize_figure(fig)
                        if view_key in {"overview", "detail"}:
                            collect_and_remove_legends(
                                fig,
                                shared_legend_entries[
                                    "training"
                                    if (
                                        group_index == 0
                                        if view_key == "detail"
                                        else is_training_scope
                                    )
                                    else "testing"
                                ],
                            )
                        try:
                            gallery.append(_figure_to_rgb(fig, compact=view_key == "overview"))
                            rendered += 1
                        finally:
                            plt.close(fig)
                if rendered and view_key == "overview":
                    separator_index = scope_start + (rendered - 1) // 2
                    caption = (
                        f"{backbone_display_name(selected_backbone)} · "
                        f"{_scope_label(scope)}"
                    )
                    if view_key == "overview":
                        caption += (
                            " · training"
                            if is_training_scope
                            else " · testing"
                        )
                    gallery[separator_index] = (
                        gallery[separator_index],
                        caption,
                    )
                messages.append(
                    f"{selected_backbone}/{scope}: datasets={','.join(dataset_labels)} "
                    f"| view={view_key} | plots={rendered}/{expected_plots} "
                    f"| displayed_metrics={'mean' if view_key == 'overview' else 7} "
                    f"| merge_replaceme={bool(merge_replaceme)} "
                    "| normalization_scope=stage2_brute_force_single_interval_raw_metric_max"
                )
            except Exception as exc:
                messages.append(
                    f"{selected_backbone}/{scope}: {type(exc).__name__}: {exc}"
                )
    if view_key in {"overview", "detail"}:
        legend_specs = [
            ("training", _TRAINING_LEGEND_CAPTION),
            ("testing", _TESTING_LEGEND_CAPTION),
        ]
        if view_key == "detail" and detail_method_key != "all":
            legend_specs = [
                legend_specs[_METHOD_GROUP_INDEX[detail_method_key]]
            ]
        for scope_kind, caption in legend_specs:
            legend_method = (
                ("similarity" if scope_kind == "training" else "baseline")
                if view_key == "detail"
                else overview_method_key
            )
            legend = render_shared_legend_banner(
                shared_legend_entries[scope_kind],
                font_size_pixels=(
                    PRUNING_PLAN_DETAIL_OUTER_FONT_SIZE
                    if view_key == "detail"
                    else PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE
                ),
                max_columns=(
                    1
                    if view_key == "overview"
                    else max(1, len(shared_legend_entries[scope_kind]))
                ),
                trailing_labels=(
                    "Single",
                    "Double" if legend_method == "similarity" else "Multi",
                ),
                fit_single_row=view_key == "detail",
                frameon=False,
            )
            if legend is not None:
                resolved_caption = (
                    f"{caption}|entries={len(shared_legend_entries[scope_kind])}"
                    if view_key == "overview"
                    else caption
                )
                gallery.append((legend, resolved_caption))
    return gallery, "\n".join(
        [*warnings, f"backbones={','.join(selected_backbones)}", *messages]
    )


def build_pruning_tabs(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str = "pruning_plan",
) -> Dict[str, Any]:
    root = Path(output_root)
    backbones = discover_pruning_plan_backbones(root)

    def _selected_real_backbones(selected: Sequence[str] | str | None) -> List[str]:
        values = [selected] if isinstance(selected, str) else list(selected or [])
        if _MEAN_BACKBONE in values:
            return list(backbones)
        return [str(value) for value in values if value in backbones]

    def _scope_choices(selected: Sequence[str] | str | None) -> List[Tuple[str, str]]:
        selected_scopes = {
            scope
            for backbone in _selected_real_backbones(selected)
            for scope in discover_pruning_plan_scopes(root, backbone)
        }
        ordered = [
            *(_AVERAGE_SCOPES if _MEAN_SCOPE in selected_scopes else ()),
            *sorted(selected_scopes.difference({_MEAN_SCOPE})),
        ]
        return [
            (_scope_label(scope), scope)
            for scope in ordered
        ]

    def _default_scope_selection(scopes: Sequence[str], *, view: str) -> List[str]:
        if view == "detail" and "cross_dataset_testing" in scopes:
            return ["cross_dataset_testing"]
        averages = [scope for scope in _AVERAGE_SCOPES if scope in scopes]
        if view == "overview":
            averages = [scope for scope in averages if scope != "training"]
        return averages or list(scopes[:1])

    def _render(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        detail_method: str = "all",
        overview_method: str = "baseline",
    ) -> Tuple[Any, str]:
        gallery, status = render_pruning_plan_gallery(
            root,
            backbone,
            selected_scopes,
            merge_replaceme=merge_replaceme,
            view=view,
            detail_method=detail_method,
            overview_method=overview_method,
        )
        columns = (
            7
            if view == "detail" or _MEAN_SCOPE in list(selected_scopes or [])
            else 2
        )
        return gr.update(value=gallery, columns=columns), status

    def _selected_scopes_after_backbone_change(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        *,
        view: str,
    ) -> List[str]:
        scopes = [value for _, value in _scope_choices(backbone)]
        retained = [scope for scope in (selected_scopes or []) if scope in scopes]
        return retained or _default_scope_selection(scopes, view=view)

    def _backbone_changed(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        detail_method: str = "all",
        overview_method: str = "baseline",
    ) -> Tuple[Any, GalleryItems, str]:
        selected = _selected_scopes_after_backbone_change(
            backbone, selected_scopes, view=view
        )
        gallery, status = _render(
            backbone,
            selected,
            merge_replaceme,
            view=view,
            detail_method=detail_method,
            overview_method=overview_method,
        )
        return (
            gr.update(choices=_scope_choices(backbone), value=selected),
            gallery,
            status,
        )

    def _refresh(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        detail_method: str = "all",
        overview_method: str = "baseline",
    ) -> Tuple[Any, Any, GalleryItems, str]:
        latest_backbones = discover_pruning_plan_backbones(root)
        backbone_value = [
            value
            for value in (backbone or [])
            if value == _MEAN_BACKBONE or value in latest_backbones
        ]
        backbone_value = backbone_value or [_MEAN_BACKBONE]
        scopes = [value for _, value in _scope_choices(backbone_value)]
        selected = [scope for scope in (selected_scopes or []) if scope in scopes]
        selected = selected or _default_scope_selection(scopes, view=view)
        gallery, status = _render(
            backbone_value,
            selected,
            merge_replaceme,
            view=view,
            detail_method=detail_method,
            overview_method=overview_method,
        )
        return (
            gr.update(
                choices=[
                    (backbone_display_name(item), item)
                    for item in [_MEAN_BACKBONE, *latest_backbones]
                ],
                value=backbone_value,
            ),
            gr.update(choices=_scope_choices(backbone_value), value=selected),
            gallery,
            status,
        )

    def _controls(*, detail: bool) -> Dict[str, Any]:
        view = "detail" if detail else "overview"
        title = "Pruning Plan Detail" if detail else "Pruning Plan Overview"
        tab_suffix = "detail" if detail else "overview"
        initial_scope_choices = [
            value for _, value in _scope_choices(backbones)
        ]
        initial_selection = _default_scope_selection(
            initial_scope_choices,
            view=view,
        )
        tab_elem_id = "pruning-plan-detail-tab" if detail else None
        with gr.Tab(
            "Pruning",
            id=f"{tab_id}_{tab_suffix}",
            elem_id=tab_elem_id,
        ) as tab:
            gr.Markdown(f"## {title}")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-column"]):
                    backbone_dd = gr.CheckboxGroup(
                        label="Backbones",
                        choices=[
                            (backbone_display_name(item), item)
                            for item in [_MEAN_BACKBONE, *backbones]
                        ],
                        value=backbones,
                    )
                    scope_cb = gr.CheckboxGroup(
                        label="Datasets (multi-select)",
                        choices=_scope_choices(backbones),
                        value=initial_selection,
                        type="value",
                        interactive=True,
                    )
                    merge_replaceme = gr.Checkbox(
                        label=(
                            "Merge ReplaceMe Single/Double "
                            "(calibration 7-metric winner)"
                        ),
                        value=True,
                    )
                    method_selector = gr.Radio(
                        label="Methods",
                        choices=list(_METHOD_CHOICES),
                        value="baseline",
                    )
                    simplify_dataset_names = (
                        gr.Checkbox(
                            label="Simplify dataset row labels",
                            value=True,
                            info="Show (a), (b), (c), ... instead of full dataset names.",
                        )
                        if not detail
                        else None
                    )
                    refresh_btn = gr.Button("Reload summaries", variant="primary")
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=5,
                    )
                with gr.Column(scale=5):
                    gallery = gr.Image(
                        label=f"Live pruning-plan {view} plots",
                        show_label=False,
                        type="numpy",
                        interactive=False,
                        elem_classes=["results-media", "composite-plot-image"],
                    )

            if detail:
                def redraw(backbone, scopes, merge, method):
                    images, status_value = _render(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        detail_method=method,
                    )
                    return _compose_pruning_detail(images), status_value

                def backbone_changed(backbone, scopes, merge, method):
                    scope_update, images, status_value = _backbone_changed(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        detail_method=method,
                    )
                    return (
                        scope_update,
                        _compose_pruning_detail(images),
                        status_value,
                    )

                def refresh(backbone, scopes, merge, method):
                    backbone_update, scope_update, images, status_value = _refresh(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        detail_method=method,
                    )
                    return (
                        backbone_update,
                        scope_update,
                        _compose_pruning_detail(images),
                        status_value,
                    )

                inputs = [
                    backbone_dd,
                    scope_cb,
                    merge_replaceme,
                    method_selector,
                ]
                change_triggers = [
                    scope_cb.change,
                    merge_replaceme.change,
                    method_selector.change,
                ]
            else:
                assert simplify_dataset_names is not None

                def redraw(backbone, scopes, merge, method, simplify_names):
                    images, status_value = _render(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        overview_method=method,
                    )
                    return (
                        _compose_pruning_overview(
                            images,
                            simplify_dataset_names=bool(simplify_names),
                        ),
                        status_value,
                    )

                def backbone_changed(backbone, scopes, merge, method, simplify_names):
                    scope_update, images, status_value = _backbone_changed(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        overview_method=method,
                    )
                    return (
                        scope_update,
                        _compose_pruning_overview(
                            images,
                            simplify_dataset_names=bool(simplify_names),
                        ),
                        status_value,
                    )

                def refresh(backbone, scopes, merge, method, simplify_names):
                    backbone_update, scope_update, images, status_value = _refresh(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        overview_method=method,
                    )
                    return (
                        backbone_update,
                        scope_update,
                        _compose_pruning_overview(
                            images,
                            simplify_dataset_names=bool(simplify_names),
                        ),
                        status_value,
                    )

                inputs = [
                    backbone_dd,
                    scope_cb,
                    merge_replaceme,
                    method_selector,
                    simplify_dataset_names,
                ]
                change_triggers = [
                    scope_cb.change,
                    merge_replaceme.change,
                    method_selector.change,
                    simplify_dataset_names.change,
                ]

            for trigger in change_triggers:
                trigger(
                    redraw,
                    inputs=inputs,
                    outputs=[gallery, status],
                    show_progress="minimal",
                    trigger_mode="always_last",
                )
            backbone_dd.change(
                backbone_changed,
                inputs=inputs,
                outputs=[scope_cb, gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
            refresh_btn.click(
                refresh,
                inputs=inputs,
                outputs=[backbone_dd, scope_cb, gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
            blocks.load(
                redraw,
                inputs=inputs,
                outputs=[gallery, status],
                show_progress="minimal",
            )

        controls = {
            "tab": tab,
            "backbone_dd": backbone_dd,
            "scope_cb": scope_cb,
            "merge_replaceme": merge_replaceme,
            "gallery": gallery,
            "status": status,
            "refresh_btn": refresh_btn,
        }
        if simplify_dataset_names is not None:
            controls["simplify_dataset_names"] = simplify_dataset_names
        if detail:
            controls["detail_method"] = method_selector
        else:
            controls["overview_method"] = method_selector
        return controls

    overview = _controls(detail=False)
    detail = _controls(detail=True)

    return {
        **overview,
        "overview_tab": overview["tab"],
        "overview": overview,
        "detail_tab": detail["tab"],
        "detail": detail,
    }

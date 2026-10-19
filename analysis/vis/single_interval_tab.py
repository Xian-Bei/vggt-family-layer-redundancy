from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr
import numpy as np
from PIL import Image

from analysis.metric_labels import SEVEN_METRIC_KEYS, SEVEN_METRIC_LABELS
from analysis.vis.display_names import backbone_display_name, dataset_display_name
from analysis.vis.composite import (
    Cell,
    Section,
    compose_sections,
    gallery_cells,
)

# Single-interval overview typography and outer-label spacing (pixels).
BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE = 24
BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE = 40
BRUTE_FORCE_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP = 20
BRUTE_FORCE_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP = 1
BRUTE_FORCE_OVERVIEW_COLUMN_GAP = 5
BRUTE_FORCE_OVERVIEW_ROW_GAP = 5

# Brute-Force Detail typography and outer-label spacing (pixels).
BRUTE_FORCE_DETAIL_SUBPLOT_FONT_SIZE = 26
BRUTE_FORCE_DETAIL_OUTER_FONT_SIZE = 30
BRUTE_FORCE_DETAIL_LEFT_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_DETAIL_TOP_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP = 0
# Pull consecutive detail rows slightly together.  Each Matplotlib tile keeps
# a small white bottom margin of its own, so a zero compositor gap still looks
# overly loose once the next row heading is inserted.
BRUTE_FORCE_DETAIL_SECTION_GAP = -20
BRUTE_FORCE_DETAIL_HEADING_Y_OFFSET = 2


BRUTE_FORCE_OVERVIEW_COLUMN_LABELS = (
    "Rotation Metrics",
    "Translation Metrics",
    "Geometry Metrics",
    "Rotation Metrics",
    "Translation Metrics",
    "Geometry Metrics",
)

BRUTE_FORCE_OVERVIEW_LEGACY_COLUMN_LABELS = (
    "Trajectory Metrics",
    "Geometry Metrics",
    "Trajectory Metrics",
    "Geometry Metrics",
)

# These outer labels are rendered by Pillow rather than Matplotlib, so use
# Unicode subscripts for the two RPE labels instead of MathText markup.
BRUTE_FORCE_DETAIL_COLUMN_LABELS = tuple(
    {
        "rpe_trans": "RPEₜ↓ (m)",
        "rpe_rot": "RPEᵣ↓ (°)",
    }.get(key, SEVEN_METRIC_LABELS[key])
    for key in SEVEN_METRIC_KEYS
)

BRUTE_FORCE_DETAIL_EXCLUDED_DATASETS = {
    "7scenes_test",
    "scannet_test50",
    "nuscenes_test",
    "nuscenes_val",
    "waymo_validation",
    "mean",
}
BRUTE_FORCE_DETAIL_DATASET_ORDER = (
    "scannet_dev50",
    "nuscenes_train",
)

BRUTE_FORCE_OVERVIEW_INFO = """\
### Mean metric sources

rotation mean: mean(ARE, RPE_rot)<br>
translation mean: mean(APE, RPE_trans)<br>
point-cloud mean: mean(Accuracy, Completeness, Chamfer)
"""


def _heatmap_axis_geometry(image: Image.Image) -> tuple[float, int]:
    """Return the horizontal center and width of the main chromatic panel."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    chroma = np.ptp(rgb, axis=2)
    colored_per_column = np.count_nonzero(chroma >= 20, axis=0)
    active = colored_per_column >= max(4, round(image.height * 0.08))
    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.flatnonzero(np.diff(padded))
    runs = list(zip(transitions[::2], transitions[1::2]))
    if not runs:
        return (image.width / 2.0, 0)
    start, end = max(runs, key=lambda bounds: bounds[1] - bounds[0])
    return ((start + end) / 2.0, end - start)


def _heatmap_vertical_center(image: Image.Image) -> float:
    """Return the vertical center of the main chromatic heatmap panel."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    chroma = np.ptp(rgb, axis=2)
    colored_per_row = np.count_nonzero(chroma >= 20, axis=1)
    active = colored_per_row >= max(4, round(image.width * 0.08))
    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.flatnonzero(np.diff(padded))
    runs = list(zip(transitions[::2], transitions[1::2]))
    if not runs:
        return image.height / 2.0
    start, end = max(runs, key=lambda bounds: bounds[1] - bounds[0])
    return (start + end) / 2.0


def _compose_brute_force_overview(
    gallery_values: Sequence[Any],
    backbone_labels: Sequence[str],
    column_labels: Sequence[str] = BRUTE_FORCE_OVERVIEW_COLUMN_LABELS,
) -> Any:
    labels = list(column_labels)
    columns = len(labels)
    rows = [gallery_cells(value) for value in gallery_values]
    fixed_rows = [
        row[:columns] + [Cell(None)] * max(0, columns - len(row))
        for row in rows
    ]
    source_widths = [
        max(
            (
                row[column].image.width
                for row in fixed_rows
                if row[column].image is not None
            ),
            default=1,
        )
        for column in range(columns)
    ]
    column_label_centers: List[float] = []
    for column, target_width in enumerate(source_widths):
        candidates: List[tuple[int, float]] = []
        for row in fixed_rows:
            image = row[column].image
            if image is None:
                continue
            center, panel_width = _heatmap_axis_geometry(image)
            padded_center = (target_width - image.width) / 2.0 + center
            candidates.append((panel_width, padded_center))
        widest_panel = max((width for width, _center in candidates), default=0)
        full_panel_centers = [
            center
            for width, center in candidates
            if widest_panel == 0 or width >= 0.95 * widest_panel
        ]
        column_label_centers.append(
            float(np.mean(full_panel_centers)) / target_width
            if full_panel_centers
            else 0.5
        )
    # The heatmap axes are rendered at one fixed Matplotlib size. Colorbars and
    # tick labels only change the surrounding tight-cropped raster. Preserve the
    # source-width proportions so composition applies one scale to every axis.
    for row in fixed_rows:
        row_height = max(
            (cell.image.height for cell in row if cell.image is not None),
            default=1,
        )
        for column, cell in enumerate(row):
            image = cell.image
            if image is None:
                continue
            target_width = source_widths[column]
            if image.width == target_width and image.height == row_height:
                continue
            padded = Image.new("RGB", (target_width, row_height), "white")
            padded.paste(
                image,
                (
                    (target_width - image.width) // 2,
                    row_height - image.height,
                ),
            )
            cell.image = padded
    row_label_centers = [
        (
            _heatmap_vertical_center(
                next(cell.image for cell in row if cell.image is not None)
            )
            / max(
                1,
                next(cell.image for cell in row if cell.image is not None).height,
            )
        )
        if any(cell.image is not None for cell in row)
        else 0.5
        for row in fixed_rows
    ]
    columns_per_domain = max(1, columns // 2)
    group_heading_anchor_columns = [
        (columns_per_domain - 1) / 2.0,
        columns_per_domain + (columns - columns_per_domain - 1) / 2.0,
    ]
    return compose_sections(
        [
            Section(
                columns=columns,
                cells=[],
                fixed_rows=fixed_rows,
                column_width_weights=[float(width) for width in source_widths],
                row_labels=list(backbone_labels),
                row_labels_bold=True,
                row_label_centers=row_label_centers,
                group_headings=[
                    ("Indoor Normalized Degradation", columns_per_domain),
                    ("Outdoor Normalized Degradation", columns - columns_per_domain),
                ],
                group_heading_anchor_columns=group_heading_anchor_columns,
                column_labels=labels,
                column_label_centers=column_label_centers,
                column_labels_below=True,
                column_labels_bold=True,
            )
        ],
        outer_font_size=BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE,
        left_label_to_image_gap=BRUTE_FORCE_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=BRUTE_FORCE_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=BRUTE_FORCE_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP,
        heatmap_column_gap=BRUTE_FORCE_OVERVIEW_COLUMN_GAP,
        heatmap_row_gap=BRUTE_FORCE_OVERVIEW_ROW_GAP,
    )


def _compose_brute_force_detail(
    gallery_values: Sequence[Any],
    backbones: Sequence[str],
    modes: Sequence[str],
    datasets: Sequence[str],
) -> Any | None:
    """Compose one seven-metric row per backbone and dataset."""
    value_index = 0
    sections: List[Section] = []
    for backbone in backbones:
        for _mode in modes:
            for dataset in datasets:
                pose_cells = gallery_cells(gallery_values[value_index])
                value_index += 1
                point_cells = gallery_cells(gallery_values[value_index])
                value_index += 1
                cells = [*pose_cells[:4], *point_cells[:3]]
                if not cells:
                    continue
                cells.extend(Cell(None) for _ in range(7 - len(cells)))
                sections.append(
                    Section(
                        columns=7,
                        cells=cells,
                        heading=(
                            f"{backbone_display_name(backbone).replace('Ω', 'Omega')} "
                            f"{dataset_display_name(dataset)}"
                        ),
                        heading_y_offset=BRUTE_FORCE_DETAIL_HEADING_Y_OFFSET,
                        column_labels_bold=True,
                    )
                )
    if not sections:
        return None
    sections[-1].column_labels = list(BRUTE_FORCE_DETAIL_COLUMN_LABELS)
    sections[-1].column_labels_below = True
    return compose_sections(
        sections,
        outer_font_size=BRUTE_FORCE_DETAIL_OUTER_FONT_SIZE,
        left_label_to_image_gap=BRUTE_FORCE_DETAIL_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=BRUTE_FORCE_DETAIL_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=BRUTE_FORCE_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP,
        section_gap=BRUTE_FORCE_DETAIL_SECTION_GAP,
    )


def filter_brute_force_detail_datasets(
    dataset_choices: Sequence[str],
) -> List[str]:
    """Return the dataset scopes represented by Detail UI/output rows."""
    datasets = [
        str(dataset)
        for dataset in dataset_choices
        if str(dataset).strip().lower()
        not in BRUTE_FORCE_DETAIL_EXCLUDED_DATASETS
    ]
    return sorted(
        datasets,
        key=lambda dataset: (
            BRUTE_FORCE_DETAIL_DATASET_ORDER.index(dataset)
            if dataset in BRUTE_FORCE_DETAIL_DATASET_ORDER
            else len(BRUTE_FORCE_DETAIL_DATASET_ORDER),
            dataset,
        ),
    )


def _bind_refresh(
    *,
    tab: Any,
    inputs: List[Any],
    outputs: List[Any],
    refresh_cb: Any,
    triggers: Sequence[tuple[Any, bool]],
) -> None:
    events: List[Any] = []
    for trigger, release in triggers:
        kwargs: Dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "show_progress": "minimal",
            "cancels": list(events),
        }
        if not release:
            kwargs["trigger_mode"] = "always_last"
        events.append(trigger(refresh_cb, **kwargs))


def build_single_interval_tabs(
    *,
    backbone_choices: Sequence[str],
    dataset_choices: Sequence[str],
    initial_skip_datasets: Sequence[str],
    stage2_modes: List[str],
    refresh_overview_cb: Any,
    refresh_detail_cb: Any,
) -> Dict[str, Any]:
    """Build overview and detailed brute-force pruning tabs."""
    backbones = list(dict.fromkeys(str(x) for x in backbone_choices))
    datasets = list(dict.fromkeys(str(x) for x in dataset_choices))
    detail_datasets = filter_brute_force_detail_datasets(datasets)
    dataset_scopes = list(detail_datasets)
    available_skip_datasets = {
        str(dataset) for dataset in initial_skip_datasets
    }
    with gr.Tab("Single-interval") as overview_tab:
        gr.Markdown("## Brute-Force Overview")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                gr.Markdown("#### Skip results")
                overview_dataset_checks: List[gr.Checkbox] = []
                for dataset in detail_datasets:
                    available = dataset in available_skip_datasets
                    overview_dataset_checks.append(
                        gr.Checkbox(
                            label=dataset,
                            value=available,
                            interactive=available,
                            info=(
                                None
                                if available
                                else "No experiment record; excluded from mean."
                            ),
                            elem_classes=(
                                []
                                if available
                                else ["brute-force-unavailable-dataset"]
                            ),
                        )
                    )
                overview_info = gr.Markdown(
                    BRUTE_FORCE_OVERVIEW_INFO,
                    elem_classes=["brute-force-overview-info"],
                )
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                overview_image = gr.Image(
                    label="Brute-force overview heatmaps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        overview_outputs = [overview_image, overview_info]

        def _refresh_overview_from_checkboxes(*values: Any) -> Any:
            selected_datasets = [
                dataset
                for dataset, selected in zip(detail_datasets, values)
                if bool(selected)
            ]
            raw = refresh_overview_cb(backbones, selected_datasets)
            return (
                _compose_brute_force_overview(
                    raw[: len(backbones)],
                    [backbone_display_name(item) for item in backbones],
                ),
                raw[-1],
            )

        _bind_refresh(
            tab=overview_tab,
            inputs=list(overview_dataset_checks),
            outputs=overview_outputs,
            refresh_cb=_refresh_overview_from_checkboxes,
            triggers=[
                *[(checkbox.change, False) for checkbox in overview_dataset_checks],
                (overview_tab.select, True),
            ],
        )

    with gr.Tab(
        "Single-interval", elem_id="single-interval-detail-tab"
    ) as detail_tab:
        gr.Markdown("## Brute-Force Detail")
        gr.Markdown(
            "Detailed mean-scene pruning results. Each backbone-dataset scope uses "
            "one row containing four pose metrics and three geometry metrics."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                gr.Markdown("#### Skip results")
                skip_dataset_checks: List[gr.Checkbox] = []
                for dataset in detail_datasets:
                    available = dataset in available_skip_datasets
                    skip_dataset_checks.append(
                        gr.Checkbox(
                            label=dataset,
                            value=available,
                            interactive=available,
                            info=(
                                None
                                if available
                                else "No experiment record; excluded from mean."
                            ),
                            elem_classes=(
                                []
                                if available
                                else ["brute-force-unavailable-dataset"]
                            ),
                        )
                    )
                info = gr.Textbox(label="Displayed scopes", interactive=False)

            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                detail_image = gr.Image(
                    label="Brute-force detail heatmaps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        inputs = [
            backbone_filter,
            *skip_dataset_checks,
        ]
        outputs = [detail_image, info]

        def _refresh_detail_from_checkboxes(
            selected_backbones: List[str] | None,
            *values: Any,
        ) -> Any:
            scope_count = len(dataset_scopes)
            scope_flags = values[:scope_count]
            detail_inputs = values[scope_count:]
            selected_skip_datasets = [
                dataset
                for dataset, selected in zip(dataset_scopes, scope_flags)
                if bool(selected)
            ]
            raw = refresh_detail_cb(
                selected_backbones,
                selected_skip_datasets,
                *detail_inputs,
            )
            gallery_count = (
                len(backbones)
                * len(stage2_modes)
                * len(dataset_scopes)
                * 2
            )
            return (
                _compose_brute_force_detail(
                    raw[:gallery_count],
                    backbones,
                    stage2_modes,
                    dataset_scopes,
                ),
                raw[-1],
            )

        _bind_refresh(
            tab=detail_tab,
            inputs=inputs,
            outputs=outputs,
            refresh_cb=_refresh_detail_from_checkboxes,
            triggers=[
                (backbone_filter.change, False),
                *[(checkbox.change, False) for checkbox in skip_dataset_checks],
                (detail_tab.select, True),
            ],
        )

    return {
        "overview_tab": overview_tab,
        "detail_tab": detail_tab,
        "overview_info": overview_info,
        "overview_datasets": detail_datasets,
        "overview_dataset_checks": overview_dataset_checks,
        "detail_datasets": detail_datasets,
        "overview_image": overview_image,
        "detail_image": detail_image,
    }

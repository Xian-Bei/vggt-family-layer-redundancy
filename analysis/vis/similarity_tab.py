"""Similarity results tab."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)
from analysis.vis.composite import (
    DEFAULT_CANVAS_WIDTH,
    Cell,
    Section,
    _font,
    compose_sections,
    gallery_cells,
    update_value,
)

# Similarity Metrics typography and outer-label spacing (pixels).
SIMILARITY_SUBPLOT_FONT_SIZE = 24
SIMILARITY_STATS_FONT_SIZE = 32
SIMILARITY_OUTER_FONT_SIZE = 38
SIMILARITY_LEFT_LABEL_TO_IMAGE_GAP = 0
SIMILARITY_TOP_LABEL_TO_IMAGE_GAP = 0
SIMILARITY_BOTTOM_LABEL_TO_IMAGE_GAP = 0
SIMILARITY_INNER_CROP_PADDING = 2
SIMILARITY_METRIC_COLUMN_GAP = 6
SIMILARITY_ROW_GAP = 8

HEATMAP_REAL_DATASETS: Tuple[str, ...] = (
    "nuscenes_train",
    "scannet_dev50",
)

SIMILARITY_COLUMN_LABELS: Tuple[str, ...] = (
    "Normalized Degradation",
    "1 − CKA",
    "1 − Cosine",
    "1 − CCA",
    "1 − Inv_MSE",
)

# Gallery rows arrive as degradation, Cosine, CKA, CCA, and Inv_MSE.
SIMILARITY_COLUMN_ORDER: Tuple[int, ...] = (0, 2, 1, 3, 4)

_DATASET_ORDER: Tuple[str, ...] = (
    "mean",
    "scannet_dev50",
    "scans_dev50",
    "scannet_test50",
    "scans_test50",
    "7scenes_test",
    "7scenes",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "waymo_validation",
    "nuscenes_train",
    "waymo_training",
)


def _heatmap_axis_geometry(image: Image.Image) -> Tuple[float, int]:
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


def _crop_left_white_margin(
    image: Image.Image,
    *,
    padding: int = SIMILARITY_INNER_CROP_PADDING,
) -> Image.Image:
    """Remove a contiguous blank left margin without touching plotted ink."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    ink_columns = np.flatnonzero(np.any(np.any(rgb < 250, axis=2), axis=0))
    if ink_columns.size == 0:
        return image
    left = max(0, int(ink_columns[0]) - max(0, int(padding)))
    return image.crop((left, 0, image.width, image.height))


def _title_line_centers(image: Image.Image) -> List[float]:
    """Locate title text rows above a heatmap's chromatic panel."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    chroma = np.ptp(rgb.astype(np.int16), axis=2)
    active_rows = np.flatnonzero(
        np.count_nonzero(chroma >= 20, axis=1)
        >= max(4, round(image.width * 0.08))
    )
    panel_top = int(active_rows[0]) if active_rows.size else image.height
    dark = np.all(rgb[:panel_top] < 112, axis=2)
    text_rows = np.flatnonzero(
        np.count_nonzero(dark, axis=1) >= max(3, round(image.width * 0.01))
    )
    if text_rows.size == 0:
        return []
    groups = np.split(text_rows, np.flatnonzero(np.diff(text_rows) > 3) + 1)
    return [float(group[0] + group[-1]) / 2.0 for group in groups if group.size >= 3]


def _draw_single_row_backbone_label(image: Any, label: str) -> np.ndarray:
    """Draw a backbone name at the Pearson line using outer-label typography."""
    output = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    if not label:
        return np.asarray(output, dtype=np.uint8).copy()
    line_centers = _title_line_centers(output)
    if not line_centers:
        return np.asarray(output, dtype=np.uint8).copy()
    target_y = line_centers[1] if len(line_centers) >= 2 else line_centers[0]
    draw = ImageDraw.Draw(output)
    font = _font(SIMILARITY_OUTER_FONT_SIZE, bold=True)
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    axis_center, _axis_width = _heatmap_axis_geometry(output)
    draw.text(
        (
            round(axis_center - text_width / 2.0 - bounds[0]),
            round(target_y - text_height / 2.0 - bounds[1]),
        ),
        label,
        fill="#222222",
        font=font,
    )
    return np.asarray(output, dtype=np.uint8).copy()


def _draw_multi_row_backbone_label(
    image: Image.Image,
    reference: Image.Image,
    label: str,
    *,
    font_size: int = SIMILARITY_OUTER_FONT_SIZE,
) -> Image.Image:
    """Draw a row's backbone above its first plot at the Pearson title line."""
    if not label:
        return image
    line_centers = _title_line_centers(reference)
    if not line_centers:
        return image
    target_y = line_centers[1] if len(line_centers) >= 2 else line_centers[0]
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = _font(font_size, bold=True)
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    axis_center, _axis_width = _heatmap_axis_geometry(output)
    draw.text(
        (
            round(axis_center - text_width / 2.0 - bounds[0]),
            round(target_y - text_height / 2.0 - bounds[1]),
        ),
        label,
        fill="#222222",
        font=font,
    )
    return output


def _compose_similarity_metrics(value: Any) -> Any:
    cells = gallery_cells(value)
    columns = len(SIMILARITY_COLUMN_LABELS)
    rows: List[List[Cell]] = []
    labels: List[str] = []
    for start in range(0, len(cells), columns):
        source_row = cells[start : start + columns]
        source_row.extend(Cell(None) for _ in range(columns - len(source_row)))
        row = [source_row[index] for index in SIMILARITY_COLUMN_ORDER]
        # Keep the overview-to-metrics group separation, then remove only the
        # redundant Matplotlib left margin at the three internal metric seams:
        # CKA | Cosine | CCA | Inv_MSE.
        for column in range(2, columns):
            if row[column].image is not None:
                row[column].image = _crop_left_white_margin(row[column].image)
        labels.append(next((cell.caption for cell in row if cell.caption), ""))
        for cell in row:
            cell.caption = ""
        rows.append(row)
    source_widths = [
        max(
            (
                row[column].image.width
                for row in rows
                if row[column].image is not None
            ),
            default=1,
        )
        for column in range(columns)
    ]
    column_label_centers: List[float] = []
    for column, target_width in enumerate(source_widths):
        candidates: List[Tuple[int, float]] = []
        for row in rows:
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
    # Every renderer uses the same fixed-size Matplotlib heatmap axes, while
    # titles, tick labels, and colorbars change the tight-cropped raster bounds.
    # Pad each source raster and allocate columns by those source widths so the
    # composite applies one common scale factor to the actual heatmap axes.
    for row in rows:
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
    single_row_label = labels[0] if len(rows) == 1 and labels else ""
    if len(rows) > 1:
        total_gap = 2 + SIMILARITY_METRIC_COLUMN_GAP * (columns - 2)
        output_scale = (
            (DEFAULT_CANVAS_WIDTH - total_gap) / sum(source_widths)
        )
        source_label_font_size = max(
            1,
            round(SIMILARITY_OUTER_FONT_SIZE / output_scale),
        )
        for row, label in zip(rows, labels):
            first_image = row[0].image
            reference_image = next(
                (cell.image for cell in row[1:] if cell.image is not None),
                None,
            )
            if first_image is not None and reference_image is not None:
                row[0].image = _draw_multi_row_backbone_label(
                    first_image,
                    reference_image,
                    label,
                    font_size=source_label_font_size,
                )
    composed = compose_sections(
        [
            Section(
                columns=columns,
                cells=[],
                fixed_rows=rows,
                column_width_weights=[float(width) for width in source_widths],
                column_gaps=[
                    2,
                    *[SIMILARITY_METRIC_COLUMN_GAP] * (columns - 2),
                ],
                row_labels=None,
                row_labels_bold=True,
                column_labels=list(SIMILARITY_COLUMN_LABELS),
                column_label_centers=column_label_centers,
                column_labels_below=True,
                column_labels_bold=True,
            )
        ],
        outer_font_size=SIMILARITY_OUTER_FONT_SIZE,
        left_label_to_image_gap=SIMILARITY_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=SIMILARITY_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=SIMILARITY_BOTTOM_LABEL_TO_IMAGE_GAP,
        heatmap_row_gap=SIMILARITY_ROW_GAP,
    )
    return (
        _draw_single_row_backbone_label(composed, single_row_label)
        if single_row_label
        else composed
    )


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip()
    try:
        return (_DATASET_ORDER.index(key), key)
    except ValueError:
        return (len(_DATASET_ORDER), key)


def build_similarity_tab(
    *,
    controller: Any,
    dataset_choices: Sequence[str],
    backbone_choices: Sequence[str],
    initial_backbone: str | None,
    stage2_modes: List[str],
    stage2_scene_ids: List[str],
    initial_stage2_scene: str | None,
    stage2_metric_row_pose: str,
    stage2_metric_row_point: str,
    stage2_token_row_special: str,
    stage2_token_row_camera: str,
    stage2_token_row_patch: str,
    stage2_token_row_special_patch_mean: str,
    stage2_token_row_all: str,
    heatmaps_resolve_filters_cb: Any,
    refresh_stage2_full_cb: Any,
) -> Dict[str, Any]:
    sorted_backbone_choices = sorted(
        [str(x) for x in backbone_choices],
        key=backbone_publication_sort_key,
    )
    allowed_datasets = {"mean", *HEATMAP_REAL_DATASETS}
    sorted_dataset_choices = sorted(
        [str(x) for x in dataset_choices if str(x) in allowed_datasets],
        key=_dataset_sort_key,
    )
    default_dataset = (
        "mean"
        if "mean" in sorted_dataset_choices
        else (sorted_dataset_choices[0] if sorted_dataset_choices else None)
    )
    default_scene_ids = (
        ["mean"] if default_dataset == "mean" else list(stage2_scene_ids)
    )
    default_scene = (
        "mean" if default_dataset == "mean" else initial_stage2_scene
    )

    with gr.Tab("Similarity") as tab_stage2:
        gr.Markdown("## Similarity Metrics")
        gr.Markdown(
            "The all-metric mean always normalizes each metric to `[0,1]` before averaging. "
            "Similarity panels display `1 - similarity`, so lower values indicate more similar representations. "
            "Dataset **`mean`**: virtual cross-dataset view (element-wise mean of each dataset's `mean` "
            "similarity; Scene locked to `mean`)."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                stage2_backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[
                        (backbone_display_name(item), item)
                        for item in sorted_backbone_choices
                    ],
                    value=sorted_backbone_choices,
                )
                stage2_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=sorted_dataset_choices,
                    value=default_dataset,
                )
                stage2_scene_dd = gr.Dropdown(
                    label="Scene",
                    choices=default_scene_ids,
                    value=default_scene,
                    interactive=default_dataset != "mean",
                )
                _hm_vm, _hm_m, _hm_t, _, _hm_tc, _hm_hs, _hm_hi, _hm_po = heatmaps_resolve_filters_cb(
                    controller, default_scene, "absolute", None, None
                )
                physical_token_regions = {
                    str(stage2_token_row_special),
                    str(stage2_token_row_camera),
                    str(stage2_token_row_patch),
                    str(stage2_token_row_special_patch_mean),
                    str(stage2_token_row_all),
                }
                _hm_tc = [
                    item for item in _hm_tc if str(item) in physical_token_regions
                ]
                _hm_t = [item for item in _hm_t if str(item) in physical_token_regions]
                if stage2_token_row_all in _hm_tc:
                    _hm_t = [stage2_token_row_all]
                stage2_metric_rows_filter = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    visible=False,
                )
                stage2_value_mode = gr.Dropdown(
                    choices=["absolute"],
                    value="absolute",
                    visible=False,
                )
                stage2_pct_clip_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=100,
                    visible=False,
                )
                stage2_token_rows_filter = gr.CheckboxGroup(
                    label="Token 1 − similarity regions",
                    choices=[
                        (
                            "special-patch-mean"
                            if str(item) == stage2_token_row_special_patch_mean
                            else str(item),
                            item,
                        )
                        for item in _hm_tc
                    ],
                    value=_hm_t,
                    interactive=_hm_hi,
                )
                stage2_sim_gamma_slider = gr.State(1.0)
                stage2_sim_cutoff_slider = gr.State(0.0)
                stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                stage2_image = gr.Image(
                    label="mixed_units: 7-metric mean · selected 1 − similarity maps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        stage2_inputs = [
            stage2_dataset_dd,
            stage2_backbone_filter,
            stage2_scene_dd,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
        ]
        raw_gallery_count = 8 * len(stage2_modes)

        def _compose_raw_result(raw: Sequence[Any]) -> Tuple[Any, ...]:
            combined: List[Any] = []
            for index in range(raw_gallery_count):
                if index % 8 != 2:
                    continue
                value, _ = update_value(raw[index])
                if isinstance(value, list):
                    combined.extend(value)
            return (
                _compose_similarity_metrics(combined),
                raw[raw_gallery_count * 2],
                *raw[raw_gallery_count * 2 + 1 :],
            )

        def _compose_refresh(*args: Any) -> Tuple[Any, ...]:
            return _compose_raw_result(refresh_stage2_full_cb(*args))

        stage2_outputs = [
            stage2_image,
            stage2_info,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
        ]

        if default_scene_ids:
            stage2_refresh_events: List[Any] = []

            def _bind_refresh(
                trigger: Any,
                *,
                with_trigger_mode: bool = True,
                use_cancels: bool = True,
                track_event: bool = True,
            ) -> None:
                kwargs: Dict[str, Any] = {
                    "inputs": stage2_inputs,
                    "outputs": stage2_outputs,
                    "show_progress": "minimal",
                }
                if with_trigger_mode:
                    kwargs["trigger_mode"] = "always_last"
                if use_cancels and stage2_refresh_events:
                    kwargs["cancels"] = list(stage2_refresh_events)
                event = trigger(_compose_refresh, **kwargs)
                if track_event:
                    stage2_refresh_events.append(event)

            _bind_refresh(stage2_scene_dd.change)
            _bind_refresh(stage2_token_rows_filter.change)
            _bind_refresh(tab_stage2.select, with_trigger_mode=False)
        else:
            stage2_info.value = (
                f"No Stage2/Stage3 reports found under "
                f"sim={controller.analysis_similarity_dir} and skip={controller.analysis_skip_dir}"
            )

    return {
        "tab": tab_stage2,
        "stage2_backbone_filter": stage2_backbone_filter,
        "stage2_dataset_dd": stage2_dataset_dd,
        "stage2_scene_dd": stage2_scene_dd,
        "stage2_image": stage2_image,
        "compose_refresh_result": _compose_refresh,
        "compose_raw_result": _compose_raw_result,
        "stage2_info": stage2_info,
        "stage2_value_mode": stage2_value_mode,
        "stage2_pct_clip_slider": stage2_pct_clip_slider,
        "stage2_metric_rows_filter": stage2_metric_rows_filter,
        "stage2_token_rows_filter": stage2_token_rows_filter,
        "stage2_sim_gamma_slider": stage2_sim_gamma_slider,
        "stage2_sim_cutoff_slider": stage2_sim_cutoff_slider,
        "stage2_inputs": stage2_inputs,
        "stage2_outputs": stage2_outputs,
    }

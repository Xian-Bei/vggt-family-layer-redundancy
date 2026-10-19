from __future__ import annotations

from typing import Any, Dict, Sequence

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)
from analysis.vis.composite import (
    Cell,
    Section,
    _font,
    compose_sections,
    gallery_cells,
)

# Remaining typography and outer-label spacing (pixels).
REMAINING_SUBPLOT_FONT_SIZE = 24
REMAINING_OUTER_FONT_SIZE = 38
# Matplotlib heatmaps are rasterized before the composite scales them down.
# This point size lands at the same final visual size as the 28 px backbone name.
REMAINING_GAP_ANNOTATION_FONT_SIZE = 38
REMAINING_ANNOTATION_COLOR = "#777777"
REMAINING_LEFT_LABEL_TO_IMAGE_GAP = 0
REMAINING_TOP_LABEL_TO_IMAGE_GAP = 0
REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP = 0
# The inward-facing Matplotlib margins are cropped before composition, so the
# remaining gap can stay positive without covering the endpoint tick labels.
REMAINING_WITHIN_BACKBONE_GAP = 2
REMAINING_BETWEEN_BACKBONE_GAP = 8
REMAINING_BETWEEN_POINT_GAP = 2
REMAINING_INNER_CROP_PADDING = 2
# Match the 38 pt Matplotlib MAE annotation after its 150 DPI rasterization.
REMAINING_ARROW_FONT_SIZE = round(REMAINING_GAP_ANNOTATION_FONT_SIZE * 150 / 72)
REMAINING_ARROW_WIDTH = 20
REMAINING_COMPARISON_LINE_SPACING = 10

REMAINING_COLUMN_LABELS = (
    "Original\nDegradation",
    "Marginal\nDegradation",
)


def _bottom_axis_label_center(image: Image.Image) -> float:
    """Locate the visual center of the bottom ``layer index i`` label."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    dark = np.all(rgb < 112, axis=2)
    bottom_start = int(image.height * 0.72)
    populated_rows = np.flatnonzero(np.any(dark[bottom_start:], axis=1))
    if populated_rows.size == 0:
        return 0.5
    populated_rows = populated_rows + bottom_start
    breaks = np.flatnonzero(np.diff(populated_rows) > 3)
    label_start_index = int(breaks[-1] + 1) if breaks.size else 0
    label_rows = populated_rows[label_start_index:]
    populated_columns = np.flatnonzero(np.any(dark[label_rows, :], axis=0))
    if populated_columns.size == 0:
        return 0.5
    center = (float(populated_columns[0]) + float(populated_columns[-1]) + 1.0) / 2.0
    return center / float(image.width)


def _crop_white_side(
    image: Image.Image,
    *,
    side: str,
    padding: int = REMAINING_INNER_CROP_PADDING,
) -> Image.Image:
    """Crop a contiguous white outer margin while preserving a small guard."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    ink_columns = np.flatnonzero(np.any(np.any(rgb < 250, axis=2), axis=0))
    if ink_columns.size == 0:
        return image
    guard = max(0, int(padding))
    if side == "left":
        left = max(0, int(ink_columns[0]) - guard)
        return image.crop((left, 0, image.width, image.height))
    if side == "right":
        right = min(image.width, int(ink_columns[-1]) + guard + 1)
        return image.crop((0, 0, right, image.height))
    raise ValueError(f"Unsupported crop side: {side!r}")


def _draw_pruning_arrows(
    image: Image.Image,
    *,
    degradation_box: tuple[int, int, int, int],
    marginal_centers: Sequence[float],
    comparison_labels: Sequence[str],
) -> Image.Image:
    """Connect one centered degradation plot to its two marginal plots."""
    if len(marginal_centers) != 2 or len(comparison_labels) != 2:
        return image
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = _font(REMAINING_ARROW_FONT_SIZE, bold=True)
    left, top, right, bottom = degradation_box
    start_x = left + (right - left) * 0.56
    end_x = float(output.width - 3)

    for direction, target_y, raw_label in zip(
        (-1.0, 1.0),
        marginal_centers,
        comparison_labels,
    ):
        start_y = float(top + 4 if direction < 0 else bottom - 4)
        # A quadratic Bezier through this virtual corner gives a rounded 90°
        # elbow: vertical tangent at the source, horizontal tangent at the tip.
        corner = (start_x, float(target_y))
        points = []
        for step in range(31):
            t = step / 30.0
            omt = 1.0 - t
            x = (
                omt**2 * start_x
                + 2.0 * omt * t * corner[0]
                + t**2 * end_x
            )
            y = (
                omt**2 * start_y
                + 2.0 * omt * t * corner[1]
                + t**2 * float(target_y)
            )
            points.append((round(x), round(y)))
        draw.line(
            points,
            fill=REMAINING_ANNOTATION_COLOR,
            width=REMAINING_ARROW_WIDTH,
            joint="curve",
        )
        arrow_length = 60
        arrow_half_height = 38
        draw.polygon(
            [
                (round(end_x), round(target_y)),
                (round(end_x - arrow_length), round(target_y - arrow_half_height)),
                (round(end_x - arrow_length), round(target_y + arrow_half_height)),
            ],
            fill=REMAINING_ANNOTATION_COLOR,
        )

        label = str(raw_label).strip()
        if not label:
            continue
        bounds = draw.multiline_textbbox(
            (0, 0),
            label,
            font=font,
            spacing=REMAINING_COMPARISON_LINE_SPACING,
            align="center",
        )
        label_width = bounds[2] - bounds[0]
        label_height = bounds[3] - bounds[1]
        label_x = max(4, round((output.width - label_width) / 2.0))
        if direction < 0:
            label_y = max(4, round(min(top, target_y) - label_height - 16))
        else:
            label_y = min(
                output.height - label_height - 8,
                round(max(bottom, target_y) + 16),
            )
        draw.multiline_text(
            (label_x, label_y),
            label,
            fill=REMAINING_ANNOTATION_COLOR,
            font=font,
            spacing=REMAINING_COMPARISON_LINE_SPACING,
            align="center",
        )
    return output


def _compose_remaining(
    gallery_values: Sequence[Any],
    backbone_labels: Sequence[str],
    *,
    first_row_only: bool = False,
    draw_comparison_arrows: bool = True,
) -> Any:
    pair_size = len(REMAINING_COLUMN_LABELS)
    backbone_cells = [gallery_cells(value) for value in gallery_values]
    if first_row_only:
        backbone_cells = [cells[:pair_size] for cells in backbone_cells]
    row_counts = [
        (len(cells) + pair_size - 1) // pair_size
        for cells in backbone_cells
    ]
    row_count = max([1 if backbone_labels else 0, *row_counts])
    rows: list[list[Cell]] = []
    comparison_labels: list[list[str]] = [
        [] for _ in backbone_labels
    ]
    for row_index in range(row_count):
        cells: list[Cell] = []
        for backbone_index, _backbone in enumerate(backbone_labels):
            available = (
                backbone_cells[backbone_index]
                if backbone_index < len(backbone_cells)
                else []
            )
            start = row_index * pair_size
            group = available[start : start + pair_size]
            group.extend(Cell(None) for _ in range(pair_size - len(group)))
            encoded_caption = next(
                (cell.caption.strip() for cell in group if cell.caption.strip()),
                "",
            )
            _point_label, _separator, comparison_label = encoded_caption.partition(
                "\n"
            )
            comparison_labels[backbone_index].append(comparison_label.strip())
            for cell in group:
                cell.caption = ""
            if group[0].image is not None:
                group[0].image = _crop_white_side(group[0].image, side="right")
            if group[1].image is not None:
                group[1].image = _crop_white_side(group[1].image, side="left")
            cells.extend(group)
        image_heights = [
            cell.image.height
            for cell in cells
            if cell.image is not None
        ]
        if image_heights:
            shared_height = max(image_heights)
            for cell in cells:
                image = cell.image
                if image is None or image.height >= shared_height:
                    continue
                padded = Image.new("RGB", (image.width, shared_height), "white")
                padded.paste(image, (0, shared_height - image.height))
                cell.image = padded
        rows.append(cells)
    cells: list[Cell] = []
    for backbone_index in range(len(backbone_labels)):
        degradation_column = backbone_index * pair_size
        marginal_column = degradation_column + 1
        degradation = next(
            (
                row[degradation_column]
                for row in rows
                if row[degradation_column].image is not None
            ),
            Cell(None),
        )
        marginal_images = [
            row[marginal_column].image
            for row in rows
            if row[marginal_column].image is not None
        ]
        if marginal_images:
            marginal_width = max(image.width for image in marginal_images)
            marginal_height = sum(image.height for image in marginal_images)
            marginal_height += REMAINING_BETWEEN_POINT_GAP * (
                len(marginal_images) - 1
            )
            stacked_marginal = Image.new(
                "RGB",
                (marginal_width, marginal_height),
                "white",
            )
            y = 0
            marginal_centers: list[float] = []
            for marginal_index, marginal_image in enumerate(marginal_images):
                target_fraction = (
                    0.30
                    if marginal_index == 0
                    else 0.70
                    if marginal_index == len(marginal_images) - 1
                    else 0.50
                )
                marginal_centers.append(
                    y + marginal_image.height * target_fraction
                )
                stacked_marginal.paste(
                    marginal_image,
                    ((marginal_width - marginal_image.width) // 2, y),
                )
                y += marginal_image.height + REMAINING_BETWEEN_POINT_GAP
            marginal = Cell(stacked_marginal)
        else:
            marginal_centers = []
            marginal = Cell(None)
        if degradation.image is not None and marginal_images:
            degradation_canvas = Image.new(
                "RGB",
                (degradation.image.width, marginal_height),
                "white",
            )
            degradation_y = (marginal_height - degradation.image.height) // 2
            degradation_canvas.paste(degradation.image, (0, degradation_y))
            if draw_comparison_arrows:
                degradation.image = _draw_pruning_arrows(
                    degradation_canvas,
                    degradation_box=(
                        0,
                        degradation_y,
                        degradation.image.width,
                        degradation_y + degradation.image.height,
                    ),
                    marginal_centers=marginal_centers,
                    comparison_labels=comparison_labels[backbone_index],
                )
            else:
                degradation.image = degradation_canvas
        cells.extend([degradation, marginal])
    groups = [(backbone, pair_size) for backbone in backbone_labels]
    columns = list(REMAINING_COLUMN_LABELS) * len(backbone_labels)
    return compose_sections(
        [
            Section(
                columns=len(cells),
                cells=[],
                column_width_weights=[
                    float(cell.image.width)
                    if cell.image is not None and cell.image.width > 0
                    else 1.0
                    for cell in cells
                ],
                column_gaps=[
                    (
                        REMAINING_WITHIN_BACKBONE_GAP
                        if column % len(REMAINING_COLUMN_LABELS)
                        < len(REMAINING_COLUMN_LABELS) - 1
                        else REMAINING_BETWEEN_BACKBONE_GAP
                    )
                    for column in range(max(0, len(cells) - 1))
                ],
                fixed_rows=[cells],
                group_headings=groups,
                group_heading_anchor_boundaries=[
                    group_index * len(REMAINING_COLUMN_LABELS) + 1
                    for group_index in range(len(groups))
                ],
                column_labels=columns,
                column_label_centers=[
                    _bottom_axis_label_center(cell.image)
                    if cell.image is not None
                    else 0.5
                    for cell in cells
                ],
                column_labels_below=True,
                column_labels_bold=True,
            )
        ],
        outer_font_size=REMAINING_OUTER_FONT_SIZE,
        left_label_to_image_gap=REMAINING_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=REMAINING_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def build_remaining_tab(
    *,
    dataset_choices: Sequence[str],
    backbone_choices: Sequence[str],
    initial_dataset: str,
    refresh_remaining_cb: Any,
) -> Dict[str, Any]:
    """Build the compact original-vs-marginal 7-metric comparison tab."""

    choices = [
        str(dataset)
        for dataset in dataset_choices
        if str(dataset).strip().lower() != "min"
    ]
    backbones = sorted(
        [str(backbone) for backbone in backbone_choices],
        key=backbone_publication_sort_key,
    )
    selected = "mean" if "mean" in choices else (choices[0] if choices else "")

    with gr.Tab("Remaining") as tab:
        gr.Markdown("## Remaining")
        gr.Markdown(
            "Each configured point produces one row. Each model heading spans two columns that "
            "show the original **7-metric mean** and its **remaining marginal** "
            "heatmap. Both use scene `mean`."
        )
        dataset = gr.Dropdown(
            label="Dataset",
            choices=choices,
            value=selected,
        )
        image = gr.Image(
            label="Original / marginal 7-metric mean",
            show_label=False,
            type="numpy",
            interactive=False,
            elem_classes=["results-media", "composite-plot-image"],
        )
        info = gr.Textbox(label="Displayed counts", interactive=False)

        def refresh(dataset_value: str):
            values = refresh_remaining_cb(dataset_value)
            gallery_values = values[: len(backbones)]
            return (
                _compose_remaining(
                    gallery_values,
                    [backbone_display_name(item) for item in backbones],
                ),
                values[-1],
            )

        outputs = [image, info]
        dataset.change(
            refresh,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab.select(
            refresh,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "dataset": dataset,
        "image": image,
        "info": info,
    }

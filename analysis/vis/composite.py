from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CANVAS_WIDTH = 2400


@dataclass
class Cell:
    image: Image.Image | None
    caption: str = ""


@dataclass
class Section:
    columns: int
    cells: list[Cell]
    column_width_weights: list[float] | None = None
    column_gaps: list[int] | None = None
    fixed_rows: list[list[Cell]] | None = None
    row_labels: list[str] | None = None
    row_labels_bold: bool = False
    row_label_centers: list[float] | None = None
    column_labels: list[str] | None = None
    column_label_columns: int | None = None
    column_labels_below: bool = False
    column_labels_bold: bool = False
    column_label_centers: list[float] | None = None
    heading: str = ""
    heading_y_offset: int = 0
    caption_bold: bool = False
    group_headings: list[tuple[str, int]] | None = None
    group_heading_anchor_columns: list[float] | None = None
    group_heading_anchor_boundaries: list[int] | None = None
    group_headings_below: bool = False
    row_banners: dict[int, Any] | None = None
    row_banners_align_content: bool = False


def update_value(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, dict) and value.get("__type__") == "update":
        return value.get("value"), value
    return value, {}


def to_image(value: Any) -> Image.Image | None:
    value, _ = update_value(value)
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim not in (2, 3) or array.size == 0:
            return None
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                finite = array[np.isfinite(array)]
                if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
                    array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if path.is_file():
            with Image.open(path) as image:
                return image.convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return to_image(value["path"])
    return None


def gallery_cells(value: Any) -> list[Cell]:
    value, _ = update_value(value)
    items = value if isinstance(value, list) else ([] if value is None else [value])
    cells: list[Cell] = []
    for item in items:
        raw, caption = (item if isinstance(item, tuple) and len(item) == 2 else (item, ""))
        image = to_image(raw)
        if image is not None:
            cells.append(Cell(image=image, caption=str(caption or "")))
    return cells


def crop_horizontal_white_margins(
    image: Image.Image,
    *,
    padding: int = 10,
    threshold: int = 250,
) -> Image.Image:
    """Trim blank side margins while preserving normalized x-axis metadata."""
    source = image.convert("RGB")
    pixels = np.asarray(source)
    ink_columns = np.flatnonzero(np.any(pixels < threshold, axis=(0, 2)))
    if not ink_columns.size:
        return source
    left = max(0, int(ink_columns[0]) - max(0, int(padding)))
    right = min(source.width, int(ink_columns[-1]) + max(0, int(padding)) + 1)
    if left == 0 and right == source.width:
        return source

    cropped = source.crop((left, 0, right, source.height))
    cropped.info.update(source.info)
    for key in ("axes_center_x", "axes_left_x", "axes_right_x"):
        if key not in source.info:
            continue
        old_x = float(source.info[key]) * source.width
        cropped.info[key] = float(np.clip(
            (old_x - left) / cropped.width,
            0.0,
            1.0,
        ))
    return cropped


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ) if bold else (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _caption_lines(caption: str, cell_width: int, font_size: int) -> list[str]:
    if not caption.strip():
        return []
    width = max(12, int(cell_width / max(1, font_size * 0.58)))
    return [
        line
        for paragraph in caption.strip().splitlines()
        for line in (textwrap.wrap(paragraph, width=width, break_long_words=False) or [""])
    ]


def _section_rows(section: Section) -> Iterable[list[Cell]]:
    if section.fixed_rows is not None:
        yield from section.fixed_rows
        return
    for start in range(0, len(section.cells), section.columns):
        row = list(section.cells[start : start + section.columns])
        row.extend(Cell(image=None) for _ in range(section.columns - len(row)))
        yield row


def compose_sections(
    sections: Sequence[Section],
    *,
    width: int = DEFAULT_CANVAS_WIDTH,
    outer_font_size: int,
    left_label_to_image_gap: int,
    top_label_to_image_gap: int,
    bottom_label_to_image_gap: int,
    heatmap_column_gap: int = 2,
    heatmap_row_gap: int = 2,
    section_gap: int = 8,
) -> np.ndarray:
    rendered_rows: list[Image.Image] = []
    spacing_after: list[int] = []
    for section in sections:
        cell_font_size = int(outer_font_size)
        cell_font = _font(cell_font_size, bold=section.caption_bold)
        rows = list(_section_rows(section))
        row_font_size = int(outer_font_size)
        row_font = _font(row_font_size, bold=section.row_labels_bold)
        label_width = 0
        if section.row_labels:
            measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            label_width = max(
                measure.textbbox((0, 0), label, font=row_font)[2]
                for label in section.row_labels
            ) + 12
        x_offset = label_width + (left_label_to_image_gap if label_width else 0)
        content_width = width - x_offset
        column_gaps = [int(heatmap_column_gap)] * max(0, section.columns - 1)
        if (
            section.column_gaps is not None
            and len(section.column_gaps) == max(0, section.columns - 1)
        ):
            column_gaps = [int(gap) for gap in section.column_gaps]
        total_column_gap = sum(column_gaps)
        cell_width = max(
            1,
            (content_width - total_column_gap)
            // section.columns,
        )
        column_widths = [cell_width] * section.columns
        if (
            section.column_width_weights is not None
            and len(section.column_width_weights) == section.columns
            and all(
                np.isfinite(weight) and float(weight) > 0.0
                for weight in section.column_width_weights
            )
        ):
            available_width = max(
                section.columns,
                content_width - total_column_gap,
            )
            weights = np.asarray(section.column_width_weights, dtype=np.float64)
            exact_widths = available_width * weights / float(weights.sum())
            column_widths = np.maximum(1, np.floor(exact_widths)).astype(int).tolist()
        column_starts: list[int] = []
        cursor = x_offset
        for column, column_width in enumerate(column_widths):
            column_starts.append(cursor)
            cursor += column_width
            if column < len(column_gaps):
                cursor += column_gaps[column]
        if section.heading:
            font_size = int(outer_font_size)
            font = _font(font_size, bold=True)
            height = font_size + 6
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            bounds = draw.textbbox((0, 0), section.heading, font=font)
            draw.text(
                (
                    max(0, (width - (bounds[2] - bounds[0])) // 2),
                    int(section.heading_y_offset),
                ),
                section.heading,
                fill="#111111",
                font=font,
            )
            rendered_rows.append(image)
            spacing_after.append(top_label_to_image_gap)
        group_row: Image.Image | None = None
        if section.group_headings:
            font_size = int(outer_font_size)
            font = _font(font_size, bold=True)
            height = font_size + 6
            group_row = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(group_row)
            start = 0
            for group_index, (label, span) in enumerate(section.group_headings):
                end = min(section.columns, start + span)
                if start >= end:
                    break
                span_width = sum(column_widths[start:end]) + sum(
                    column_gaps[start : max(start, end - 1)]
                )
                x = column_starts[start]
                bounds = draw.textbbox((0, 0), label, font=font)
                heading_center = x + span_width / 2.0
                if (
                    section.group_heading_anchor_boundaries is not None
                    and group_index
                    < len(section.group_heading_anchor_boundaries)
                ):
                    boundary = int(
                        section.group_heading_anchor_boundaries[group_index]
                    )
                    if 0 < boundary < section.columns:
                        heading_center = (
                            column_starts[boundary]
                            - column_gaps[boundary - 1] / 2.0
                        )
                elif (
                    section.group_heading_anchor_columns is not None
                    and group_index < len(section.group_heading_anchor_columns)
                ):
                    anchor = float(section.group_heading_anchor_columns[group_index])
                    lower = int(np.floor(anchor))
                    upper = int(np.ceil(anchor))
                    if 0 <= lower < section.columns and 0 <= upper < section.columns:
                        fraction = anchor - lower

                        def _axis_center(column: int) -> float:
                            center = (
                                section.column_label_centers[column]
                                if section.column_label_centers is not None
                                and column < len(section.column_label_centers)
                                else 0.5
                            )
                            return column_starts[column] + column_widths[column] * center

                        heading_center = (
                            (1.0 - fraction) * _axis_center(lower)
                            + fraction * _axis_center(upper)
                        )
                draw.text(
                    (
                        max(0, round(heading_center - (bounds[2] - bounds[0]) / 2)),
                        0,
                    ),
                    label,
                    fill="#111111",
                    font=font,
                )
                start = end
            if not section.group_headings_below:
                rendered_rows.append(group_row)
                spacing_after.append(top_label_to_image_gap)
        column_row: Image.Image | None = None
        if section.column_labels:
            label_columns = max(
                1,
                int(section.column_label_columns or section.columns),
            )
            if label_columns == section.columns:
                label_widths = column_widths
                label_starts = column_starts
            else:
                label_cell_width = max(
                    1,
                    (
                        content_width
                        - heatmap_column_gap * (label_columns - 1)
                    )
                    // label_columns,
                )
                label_widths = [label_cell_width] * label_columns
                label_starts = [
                    x_offset + column * (label_cell_width + heatmap_column_gap)
                    for column in range(label_columns)
                ]
            font_size = int(outer_font_size)
            font = _font(font_size, bold=section.column_labels_bold)
            wrapped = [
                _caption_lines(label, label_widths[column], font_size)
                for column, label in enumerate(
                    section.column_labels[:label_columns]
                )
            ]
            line_height = font_size + 2
            measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            max_line_bottom = max(
                (
                    measure.textbbox((0, 0), line, font=font)[3]
                    for lines in wrapped
                    for line in lines
                ),
                default=line_height,
            )
            max_line_count = max((len(lines) for lines in wrapped), default=1)
            column_row = Image.new(
                "RGB",
                (
                    width,
                    (max_line_count - 1) * line_height
                    + max_line_bottom
                    + 2,
                ),
                "white",
            )
            draw = ImageDraw.Draw(column_row)
            for column, lines in enumerate(wrapped):
                x = label_starts[column]
                label_cell_width = label_widths[column]
                center = (
                    section.column_label_centers[column]
                    if section.column_label_centers is not None
                    else 0.5
                )
                for line_index, line in enumerate(lines):
                    bounds = draw.textbbox((0, 0), line, font=font)
                    draw.text(
                        (
                            x
                            + max(
                                0,
                                (
                                    label_cell_width * center
                                    - (bounds[2] - bounds[0]) / 2
                                ),
                            ),
                            line_index * line_height,
                        ),
                        line,
                        fill="#222222",
                        font=font,
                    )
            if not section.column_labels_below:
                rendered_rows.append(column_row)
                spacing_after.append(top_label_to_image_gap)
        for row_index, cells in enumerate(rows):
            if section.row_banners and row_index in section.row_banners:
                banner = to_image(section.row_banners[row_index])
                if banner is not None:
                    if banner.width != width:
                        banner_height = max(
                            1, round(banner.height * width / banner.width)
                        )
                        banner = banner.resize(
                            (width, banner_height), Image.Resampling.LANCZOS
                        )
                    if section.row_banners_align_content:
                        # Remove only horizontal white padding, keeping the
                        # legend's original font size when it fits the plots.
                        ink_columns = np.flatnonzero(
                            np.any(np.asarray(banner) < 250, axis=(0, 2))
                        )
                        if ink_columns.size:
                            banner = banner.crop((
                                max(0, int(ink_columns[0]) - 2), 0,
                                min(banner.width, int(ink_columns[-1]) + 3),
                                banner.height,
                            ))
                        if banner.width > content_width:
                            banner = banner.resize(
                                (content_width, max(1, round(
                                    banner.height * content_width / banner.width
                                ))),
                                Image.Resampling.LANCZOS,
                            )
                        aligned = Image.new("RGB", (width, banner.height), "white")
                        aligned.paste(
                            banner,
                            (x_offset + (content_width - banner.width) // 2, 0),
                        )
                        banner = aligned
                    rendered_rows.append(banner)
                    spacing_after.append(top_label_to_image_gap)
            scaled: list[tuple[Image.Image | None, list[str]]] = []
            image_height = caption_height = 0
            for column, cell in enumerate(cells):
                resolved_cell_width = column_widths[column]
                image = cell.image
                if image is not None:
                    height = max(
                        1,
                        round(image.height * resolved_cell_width / image.width),
                    )
                    image = image.resize(
                        (resolved_cell_width, height),
                        Image.Resampling.LANCZOS,
                    )
                    image_height = max(image_height, height)
                lines = _caption_lines(
                    cell.caption,
                    resolved_cell_width,
                    cell_font_size,
                )
                if lines and image is not None:
                    caption_height = max(caption_height, len(lines) * (cell_font_size + 5) + 12)
                scaled.append((image, lines))
            row_height = max(1, image_height + caption_height)
            canvas = Image.new("RGB", (width, row_height), "white")
            draw = ImageDraw.Draw(canvas)
            for column, (image, lines) in enumerate(scaled):
                x = column_starts[column]
                resolved_cell_width = column_widths[column]
                if image is not None:
                    canvas.paste(image, (x, max(0, (image_height - image.height) // 2)))
                for line_index, line in enumerate(lines):
                    bounds = draw.textbbox((0, 0), line, font=cell_font)
                    text_y = (
                        max(
                            0,
                            (image_height - len(lines) * (cell_font_size + 5)) // 2,
                        )
                        if image is None
                        else image_height + 6
                    )
                    draw.text(
                        (
                            x + max(
                                0,
                                (resolved_cell_width - (bounds[2] - bounds[0])) // 2,
                            ),
                            text_y + line_index * (cell_font_size + 5),
                        ),
                        line,
                        fill="#222222",
                        font=cell_font,
                    )
            if section.row_labels and row_index < len(section.row_labels):
                label = section.row_labels[row_index]
                bounds = draw.textbbox((0, 0), label, font=row_font)
                center = (
                    section.row_label_centers[row_index]
                    if section.row_label_centers is not None
                    and row_index < len(section.row_label_centers)
                    else 0.5
                )
                target_y = row_height * float(np.clip(center, 0.0, 1.0))
                draw.text(
                    (
                        max(0, (label_width - (bounds[2] - bounds[0])) // 2),
                        max(
                            0,
                            round(
                                target_y
                                - (bounds[3] - bounds[1]) / 2
                                - bounds[1]
                            ),
                        ),
                    ),
                    label,
                    fill="#222222",
                    font=row_font,
                    align="center",
                )
            rendered_rows.append(canvas)
            last = row_index == len(rows) - 1
            spacing_after.append(
                bottom_label_to_image_gap
                if last and column_row is not None and section.column_labels_below
                else section_gap if last
                else heatmap_row_gap
            )
        if column_row is not None and section.column_labels_below:
            rendered_rows.append(column_row)
            spacing_after.append(
                top_label_to_image_gap
                if group_row is not None and section.group_headings_below
                else section_gap
            )
        if group_row is not None and section.group_headings_below:
            rendered_rows.append(group_row)
            spacing_after.append(section_gap)
    if not rendered_rows:
        raise ValueError("No default plot images were returned.")
    height = sum(row.height for row in rendered_rows) + sum(
        spacing_after[: max(0, len(rendered_rows) - 1)]
    )
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for index, row in enumerate(rendered_rows):
        canvas.paste(row, (0, y))
        y += row.height
        if index < len(rendered_rows) - 1:
            y += spacing_after[index]
    return np.asarray(canvas, dtype=np.uint8).copy()

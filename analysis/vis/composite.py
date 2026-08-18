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
    fixed_rows: list[list[Cell]] | None = None
    row_labels: list[str] | None = None
    row_labels_bold: bool = False
    column_labels: list[str] | None = None
    column_labels_below: bool = False
    column_labels_bold: bool = False
    heading: str = ""
    caption_bold: bool = False
    group_headings: list[tuple[str, int]] | None = None


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
        cell_width = max(
            1,
            (content_width - heatmap_column_gap * (section.columns - 1))
            // section.columns,
        )
        if section.heading:
            font_size = int(outer_font_size)
            font = _font(font_size, bold=True)
            height = font_size + 6
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            bounds = draw.textbbox((0, 0), section.heading, font=font)
            draw.text(
                (max(0, (width - (bounds[2] - bounds[0])) // 2), 0),
                section.heading,
                fill="#111111",
                font=font,
            )
            rendered_rows.append(image)
            spacing_after.append(top_label_to_image_gap)
        if section.group_headings:
            font_size = int(outer_font_size)
            font = _font(font_size, bold=True)
            height = font_size + 6
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            start = 0
            for label, span in section.group_headings:
                span_width = cell_width * span + heatmap_column_gap * (span - 1)
                x = x_offset + start * (cell_width + heatmap_column_gap)
                bounds = draw.textbbox((0, 0), label, font=font)
                draw.text(
                    (x + max(0, (span_width - (bounds[2] - bounds[0])) // 2), 0),
                    label,
                    fill="#111111",
                    font=font,
                )
                start += span
            rendered_rows.append(image)
            spacing_after.append(top_label_to_image_gap)
        column_row: Image.Image | None = None
        if section.column_labels:
            font_size = int(outer_font_size)
            font = _font(font_size, bold=section.column_labels_bold)
            wrapped = [
                _caption_lines(label, cell_width, font_size)
                for label in section.column_labels[: section.columns]
            ]
            line_height = font_size + 2
            column_row = Image.new(
                "RGB",
                (width, max((len(lines) for lines in wrapped), default=1) * line_height + 2),
                "white",
            )
            draw = ImageDraw.Draw(column_row)
            for column, lines in enumerate(wrapped):
                x = x_offset + column * (cell_width + heatmap_column_gap)
                for line_index, line in enumerate(lines):
                    bounds = draw.textbbox((0, 0), line, font=font)
                    draw.text(
                        (x + max(0, (cell_width - (bounds[2] - bounds[0])) // 2), line_index * line_height),
                        line,
                        fill="#222222",
                        font=font,
                    )
            if not section.column_labels_below:
                rendered_rows.append(column_row)
                spacing_after.append(top_label_to_image_gap)
        for row_index, cells in enumerate(rows):
            scaled: list[tuple[Image.Image | None, list[str]]] = []
            image_height = caption_height = 0
            for cell in cells:
                image = cell.image
                if image is not None:
                    height = max(1, round(image.height * cell_width / image.width))
                    image = image.resize((cell_width, height), Image.Resampling.LANCZOS)
                    image_height = max(image_height, height)
                lines = _caption_lines(cell.caption, cell_width, cell_font_size)
                if lines:
                    caption_height = max(caption_height, len(lines) * (cell_font_size + 5) + 12)
                scaled.append((image, lines))
            row_height = max(1, image_height + caption_height)
            canvas = Image.new("RGB", (width, row_height), "white")
            draw = ImageDraw.Draw(canvas)
            for column, (image, lines) in enumerate(scaled):
                x = x_offset + column * (cell_width + heatmap_column_gap)
                if image is not None:
                    canvas.paste(image, (x, max(0, (image_height - image.height) // 2)))
                for line_index, line in enumerate(lines):
                    bounds = draw.textbbox((0, 0), line, font=cell_font)
                    draw.text(
                        (x + max(0, (cell_width - (bounds[2] - bounds[0])) // 2), image_height + 6 + line_index * (cell_font_size + 5)),
                        line,
                        fill="#222222",
                        font=cell_font,
                    )
            if section.row_labels and row_index < len(section.row_labels):
                label = section.row_labels[row_index]
                bounds = draw.textbbox((0, 0), label, font=row_font)
                draw.text(
                    (
                        max(0, (label_width - (bounds[2] - bounds[0])) // 2),
                        max(0, (row_height - (bounds[3] - bounds[1])) // 2),
                    ),
                    label,
                    fill="#222222",
                    font=row_font,
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

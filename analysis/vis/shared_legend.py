from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw


SHARED_LEGEND_CAPTION = "__shared_overview_legend__"
SHARED_LEGEND_DPI = 150
LegendEntry = Tuple[Any, str]


def _legend_handle_proxy(handle: Any) -> Any:
    """Copy the visible style of a legend handle into a detached proxy."""
    if isinstance(handle, Line2D):
        return Line2D(
            [],
            [],
            color=handle.get_color(),
            linestyle=handle.get_linestyle(),
            linewidth=handle.get_linewidth(),
            marker=handle.get_marker(),
            markersize=handle.get_markersize(),
            markerfacecolor=handle.get_markerfacecolor(),
            markeredgecolor=handle.get_markeredgecolor(),
            markeredgewidth=handle.get_markeredgewidth(),
            alpha=handle.get_alpha(),
        )
    if isinstance(handle, Patch):
        return Patch(
            facecolor=handle.get_facecolor(),
            edgecolor=handle.get_edgecolor(),
            linewidth=handle.get_linewidth(),
            linestyle=handle.get_linestyle(),
            hatch=handle.get_hatch(),
            alpha=handle.get_alpha(),
        )
    return handle


def collect_and_remove_legends(
    fig: Any,
    entries: List[LegendEntry],
) -> None:
    """Merge every axes legend into ``entries`` and remove it from ``fig``."""
    seen = {label for _, label in entries}
    legends: List[Legend] = []
    for axis in getattr(fig, "axes", ()):
        primary = axis.get_legend()
        if isinstance(primary, Legend) and primary not in legends:
            legends.append(primary)
        for artist in getattr(axis, "artists", ()):
            if isinstance(artist, Legend) and artist not in legends:
                legends.append(artist)

    for legend in legends:
        handles = getattr(
            legend,
            "legend_handles",
            getattr(legend, "legendHandles", ()),
        )
        labels = [text.get_text() for text in legend.get_texts()]
        for handle, label in zip(handles, labels):
            label = str(label).strip()
            if not label or label.startswith("_") or label in seen:
                continue
            entries.append((_legend_handle_proxy(handle), label))
            seen.add(label)
        legend.remove()


def _ordered_entries(
    entries: Sequence[LegendEntry],
    trailing_labels: Iterable[str],
) -> List[LegendEntry]:
    trailing = tuple(str(label) for label in trailing_labels)
    trailing_set = set(trailing)
    main = [entry for entry in entries if entry[1] not in trailing_set]
    by_label = {label: handle for handle, label in entries}
    return [
        *main,
        *((by_label[label], label) for label in trailing if label in by_label),
    ]


def render_shared_legend_banner(
    entries: Sequence[LegendEntry],
    *,
    width: int = 2400,
    font_size_pixels: float = 26.0,
    max_columns: int = 6,
    trailing_labels: Iterable[str] = (),
    handlelength: float = 2.8,
    columnspacing: float = 2.0,
    fit_single_row: bool = False,
    frameon: bool = True,
) -> np.ndarray | None:
    """Render deduplicated legend entries into a full-width white banner."""
    ordered = _ordered_entries(entries, trailing_labels)
    if not ordered:
        return None
    columns = min(max(1, int(max_columns)), len(ordered))
    rows = int(math.ceil(len(ordered) / columns))
    dpi = SHARED_LEGEND_DPI
    height = max(70, 70 * rows)
    canvas_width = max(1, int(width))
    effective_columnspacing = max(0.0, float(columnspacing))
    margin_pixels = max(8.0, float(font_size_pixels) * 0.25)

    def _render(current_width: int, current_columnspacing: float):
        figure = Figure(
            figsize=(current_width / dpi, height / dpi),
            dpi=dpi,
            facecolor="white",
        )
        FigureCanvasAgg(figure)
        current_legend = figure.legend(
            handles=[handle for handle, _ in ordered],
            labels=[label for _, label in ordered],
            loc="center",
            ncol=columns,
            fontsize=float(font_size_pixels) * 72.0 / dpi,
            markerscale=1.35,
            handlelength=handlelength,
            columnspacing=current_columnspacing,
            handleheight=1.2,
            frameon=bool(frameon),
        )
        for text in current_legend.get_texts():
            text.set_fontweight("bold")
        figure.canvas.draw()
        return figure, current_legend

    fig, legend = _render(canvas_width, effective_columnspacing)
    if fit_single_row and rows == 1:
        legend_width = float(
            legend.get_window_extent(renderer=fig.canvas.get_renderer()).width
        )
        available_width = float(canvas_width) - 2.0 * margin_pixels
        if legend_width > available_width and columns > 1:
            spacing_reduction = (
                (legend_width - available_width)
                / ((columns - 1) * max(float(font_size_pixels), 1.0))
            )
            effective_columnspacing = max(
                0.05,
                effective_columnspacing - spacing_reduction,
            )
            fig, legend = _render(
                canvas_width,
                effective_columnspacing,
            )
            legend_width = float(
                legend.get_window_extent(
                    renderer=fig.canvas.get_renderer()
                ).width
            )
        if legend_width > available_width:
            canvas_width = int(
                math.ceil(legend_width + 2.0 * margin_pixels)
            )
            fig, legend = _render(
                canvas_width,
                effective_columnspacing,
            )
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    content = np.any(rgb < 250, axis=2)
    ys, _ = np.nonzero(content)
    if ys.size:
        y0 = max(0, int(ys.min()) - 8)
        y1 = min(rgb.shape[0], int(ys.max()) + 9)
        rgb = rgb[y0:y1]
    return rgb.copy()


def prepend_shared_legend(
    composite: np.ndarray,
    legend: Any | None,
    *,
    gap: int = 2,
) -> np.ndarray:
    """Place a legend image above a composite, matching its canvas width."""
    if legend is None:
        return composite
    legend_image = Image.fromarray(np.asarray(legend, dtype=np.uint8)).convert("RGB")
    width = int(composite.shape[1])
    if legend_image.width != width:
        height = max(1, round(legend_image.height * width / legend_image.width))
        legend_image = legend_image.resize((width, height), Image.Resampling.LANCZOS)
    banner = np.asarray(legend_image, dtype=np.uint8)
    spacer = np.full((max(0, int(gap)), width, 3), 255, dtype=np.uint8)
    return np.concatenate((banner, spacer, composite), axis=0)


def append_shared_legend_right(
    composite: np.ndarray,
    legend: Any | None,
    *,
    gap: int = 16,
) -> np.ndarray:
    """Place a tightly cropped legend to the right, centered vertically."""
    return append_shared_legends_right(
        composite,
        ((legend, 0.5),) if legend is not None else (),
        gap=gap,
    )


def append_shared_legends_right(
    composite: np.ndarray,
    legends: Sequence[tuple[Any, ...]],
    *,
    gap: int = 16,
    frameon: bool = True,
) -> np.ndarray:
    """Place cropped legends in one right sidebar by center or exact y bounds."""
    if not legends:
        return composite
    content = Image.fromarray(np.asarray(composite, dtype=np.uint8)).convert("RGB")
    cropped: list[tuple[Image.Image, float, tuple[float, float] | None]] = []
    for positioned_legend in legends:
        legend, center = positioned_legend[:2]
        bounds = (
            (float(positioned_legend[1]), float(positioned_legend[2]))
            if len(positioned_legend) >= 3
            else None
        )
        entry_count = (
            max(1, int(positioned_legend[3]))
            if len(positioned_legend) >= 4
            else None
        )
        legend_image = Image.fromarray(
            np.asarray(legend, dtype=np.uint8)
        ).convert("RGB")
        legend_array = np.asarray(legend_image, dtype=np.uint8)
        ink_rows = np.flatnonzero(np.any(legend_array < 250, axis=(1, 2)))
        ink_columns = np.flatnonzero(np.any(legend_array < 250, axis=(0, 2)))
        if ink_rows.size and ink_columns.size:
            legend_image = legend_image.crop((
                max(0, int(ink_columns[0]) - 8),
                max(0, int(ink_rows[0]) - 8),
                min(legend_image.width, int(ink_columns[-1]) + 9),
                min(legend_image.height, int(ink_rows[-1]) + 9),
            ))
        cropped.append((
            legend_image,
            float(np.clip(center, 0.0, 1.0)),
            bounds,
        ))
    sidebar_width = max(image.width for image, _, _ in cropped)
    height = max(content.height, *(image.height for image, _, _ in cropped))
    canvas = Image.new(
        "RGB",
        (content.width + max(0, int(gap)) + sidebar_width, height),
        "white",
    )
    canvas.paste(content, (0, (height - content.height) // 2))
    sidebar_x = content.width + max(0, int(gap))
    for legend_image, center, bounds in cropped:
        if bounds is not None:
            top = round(float(np.clip(bounds[0], 0.0, 1.0)) * height)
            bottom = round(float(np.clip(bounds[1], 0.0, 1.0)) * height)
            target_height = max(1, bottom - top)
            legend_image = _redistribute_vertical_legend(
                legend_image,
                target_height,
                expected_count=entry_count,
                frameon=frameon,
            )
            center = (top + bottom) / (2.0 * height)
        x = sidebar_x + (sidebar_width - legend_image.width) // 2
        y = round(center * height - legend_image.height / 2)
        y = min(max(0, y), height - legend_image.height)
        canvas.paste(legend_image, (x, y))
    return np.asarray(canvas, dtype=np.uint8).copy()


def _redistribute_vertical_legend(
    legend_image: Image.Image,
    target_height: int,
    *,
    expected_count: int | None = None,
    frameon: bool = True,
) -> Image.Image:
    """Fit a one-column legend to an exact height without stretching its text."""
    source = np.asarray(legend_image.convert("RGB"), dtype=np.uint8)
    ink = np.any(source < 245, axis=2)
    ys, xs = np.nonzero(ink)
    if not ys.size or target_height <= 1:
        return legend_image
    if frameon:
        source = source[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        # The caller already leaves a small white crop margin around a
        # frameless legend.  Preserve that full width: trimming it again and
        # then removing an assumed frame margin clips long labels at both
        # ends.
        source = source[ys.min() : ys.max() + 1]
    height, width = source.shape[:2]
    edge_margin = max(6, min(16, width // 20)) if frameon else 0
    interior = np.any(
        source[:, edge_margin : max(edge_margin + 1, width - edge_margin)] < 245,
        axis=2,
    )
    counts = interior.sum(axis=1)
    row_mask = (counts > 1) & (counts < max(2, int(interior.shape[1] * 0.7)))
    active = np.flatnonzero(row_mask)
    groups: list[tuple[int, int]] = []
    if active.size:
        start = previous = int(active[0])
        for value in active[1:]:
            value = int(value)
            if value - previous > 7:
                groups.append((start, previous + 1))
                start = value
            previous = value
        groups.append((start, previous + 1))
    if expected_count is not None and len(groups) > expected_count:
        scored = sorted(
            enumerate(groups),
            key=lambda item: (
                -int(counts[item[1][0] : item[1][1]].sum()),
                item[0],
            ),
        )[:expected_count]
        groups = [groups[index] for index, _ in sorted(scored)]
    if not groups:
        return legend_image.resize(
            (legend_image.width, target_height), Image.Resampling.LANCZOS
        )

    canvas = Image.new("RGB", (width, target_height), "white")
    available = max(1, target_height - 16)
    centers = [
        8.0 + available * (index + 0.5) / len(groups)
        for index in range(len(groups))
    ]
    for (top, bottom), center in zip(groups, centers):
        strip = Image.fromarray(
            source[
                top:bottom,
                edge_margin : width - edge_margin if edge_margin else width,
            ].copy(),
            mode="RGB",
        )
        y = max(1, min(target_height - strip.height - 1, round(center - strip.height / 2)))
        canvas.paste(strip, (edge_margin, y))
    if frameon:
        ImageDraw.Draw(canvas).rounded_rectangle(
            (0, 0, width - 1, target_height - 1),
            radius=4,
            outline=(204, 204, 204),
            width=1,
        )
    return canvas

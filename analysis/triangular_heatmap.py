"""Shared display conventions for upper-triangle heatmaps."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from matplotlib.ticker import MultipleLocator


TRIANGLE_EMPTY_LEGEND_LOC = "lower right"


def triangle_display_array(arr: np.ndarray) -> np.ndarray:
    """Display inclusive upper-triangle data in the top-left with a lower-left origin."""
    return np.asanyarray(arr).T


def triangle_extent(n: int) -> Tuple[float, float, float, float]:
    return (-0.5, float(n) - 0.5, -0.5, float(n) - 0.5)


def triangle_xy(i: int, j: int) -> Tuple[int, int]:
    """Map logical matrix coordinates ``(i, j)`` to display coordinates ``(x, y)``."""
    return int(i), int(j)


def imshow_upper_triangle(ax: Any, arr: np.ndarray, n: int, **imshow_kwargs: Any) -> Any:
    """Render upper-triangle heatmap data using the project-wide display convention."""
    kwargs = dict(imshow_kwargs)
    kwargs["origin"] = "lower"
    kwargs["extent"] = triangle_extent(n)
    return ax.imshow(triangle_display_array(arr), **kwargs)


def configure_triangle_axes(
    ax: Any,
    n: int,
    *,
    index_name: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    tick_step: int = 4,
    label_fontsize: Optional[float] = 16,
    label_fontweight: Optional[str] = None,
    tick_fontsize: Optional[float] = 14,
    tick_width: Optional[float] = None,
    tick_length: Optional[float] = None,
    bold_tick_labels: bool = False,
    use_major_locator: bool = False,
) -> None:
    """Configure axes for lower-left-origin, top-left-filled upper-triangle heatmaps."""
    step = max(1, int(tick_step))
    label_kwargs: dict[str, Any] = {}
    if label_fontsize is not None:
        label_kwargs["fontsize"] = label_fontsize
    if label_fontweight is not None:
        label_kwargs["fontweight"] = label_fontweight
    ax.set_xlabel(str(xlabel) if xlabel is not None else f"{index_name} index i", **label_kwargs)
    ax.set_ylabel(str(ylabel) if ylabel is not None else f"{index_name} index j", **label_kwargs)
    ax.set_xlim(-0.5, float(n) - 0.5)
    ax.set_ylim(-0.5, float(n) - 0.5)
    if use_major_locator:
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))
    else:
        ticks = list(range(0, int(n), step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    tick_kwargs: dict[str, Any] = {}
    if tick_fontsize is not None:
        tick_kwargs["labelsize"] = tick_fontsize
    if tick_width is not None:
        tick_kwargs["width"] = tick_width
    if tick_length is not None:
        tick_kwargs["length"] = tick_length
    if tick_kwargs:
        ax.tick_params(axis="both", which="major", **tick_kwargs)
    if bold_tick_labels:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")


def draw_symmetric_heatmap_panel(
    ax: Any,
    values: np.ndarray,
    *,
    title: str = "",
    colorbar_label: str = "",
    cmap: str = "coolwarm_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diagonal_only: bool = False,
    index_name: str = "unit",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title_fontsize: float = 12,
    tick_fontsize: float = 14,
    colorbar_tick_fontsize: float = 14,
) -> Tuple[Any, Any]:
    """Draw an upper-stored matrix as the project-standard symmetric heatmap."""
    upper = np.asarray(values, dtype=np.float64).copy()
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1]:
        raise ValueError(
            f"expected square upper-stored heatmap, got {upper.shape}"
        )
    n = int(upper.shape[0])
    upper[np.tril_indices(n, k=-1)] = np.nan
    displayed = upper.copy()
    for row in range(n):
        for col in range(row + 1, n):
            displayed[col, row] = upper[row, col]
    if bool(diagonal_only):
        displayed[~np.eye(n, dtype=bool)] = np.nan

    finite = displayed[np.isfinite(displayed)]
    resolved_vmin = (
        float(vmin)
        if vmin is not None
        else float(np.min(finite))
        if finite.size
        else 0.0
    )
    resolved_vmax = (
        float(vmax)
        if vmax is not None
        else float(np.max(finite))
        if finite.size
        else 1.0
    )
    if not np.isfinite(resolved_vmax) or resolved_vmax <= resolved_vmin:
        resolved_vmax = resolved_vmin + 1.0

    image = imshow_upper_triangle(
        ax,
        np.ma.masked_invalid(displayed),
        n,
        cmap=str(cmap),
        vmin=resolved_vmin,
        vmax=resolved_vmax,
    )
    configure_triangle_axes(
        ax,
        n,
        index_name=str(index_name),
        xlabel=xlabel,
        ylabel=ylabel,
        tick_fontsize=float(tick_fontsize),
    )
    if str(title).strip():
        ax.set_title(
            str(title),
            fontsize=float(title_fontsize),
            fontweight="bold",
        )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if str(colorbar_label).strip():
        colorbar.set_label(str(colorbar_label), fontsize=16)
    colorbar.ax.tick_params(labelsize=float(colorbar_tick_fontsize))
    return image, colorbar

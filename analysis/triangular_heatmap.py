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
    tick_step: int = 2,
    label_fontsize: Optional[float] = None,
    label_fontweight: Optional[str] = None,
    tick_fontsize: Optional[float] = None,
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
    ax.set_xlabel(f"{index_name} index i", **label_kwargs)
    ax.set_ylabel(f"{index_name} index j", **label_kwargs)
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

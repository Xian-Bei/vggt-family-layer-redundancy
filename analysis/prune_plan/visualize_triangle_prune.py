"""Visualize CKA heatmap with greedy dense-triangle corner markers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import numpy as np
import torch

from analysis.redundancy.metrics import upper_triangle_pack, upper_triangle_unpack_for_heatmap
from analysis.prune_plan.triangle_prune import TrianglePick, greedy_dense_triangles
from analysis.triangular_heatmap import (
    TRIANGLE_EMPTY_LEGEND_LOC,
    configure_triangle_axes,
    imshow_upper_triangle,
    triangle_xy,
)

TRAJECTORY_SCORE_MAX_SPAN = 48
TRAJECTORY_SCORE_SPAN_STEP = 2


def _relative_degradation_vmax(
    values: Sequence[np.ndarray],
    relative_degradation_cap_multiple: float,
) -> float:
    cap = float(relative_degradation_cap_multiple)
    if np.isfinite(cap) and cap >= 1.0:
        return cap
    finite_values: list[np.ndarray] = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64)
        finite = np.isfinite(arr)
        if np.any(finite):
            finite_values.append(arr[finite])
    if not finite_values:
        return 1.0
    return max(1.0, float(np.nanmax(np.concatenate(finite_values))))


def _format_relative_degradation_colorbar(cbar) -> None:
    cbar.formatter = FuncFormatter(lambda value, _pos: f"{value:g}x")
    cbar.update_ticks()


def _path_scores(score: Optional[np.ndarray], picks: Sequence[TrianglePick]) -> list[float]:
    if score is None:
        return []
    arr = np.asarray(score, dtype=np.float64)
    out: list[float] = []
    for pick in picks:
        if int(pick.span) > int(TRAJECTORY_SCORE_MAX_SPAN):
            continue
        i = int(pick.i)
        j = int(pick.j)
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
            val = float(arr[i, j])
            if np.isfinite(val):
                out.append(val)
    return out


def _path_span_scores(
    score: Optional[np.ndarray],
    picks: Sequence[TrianglePick],
    *,
    span_step: int = 1,
) -> list[tuple[int, float]]:
    if score is None:
        return []
    arr = np.asarray(score, dtype=np.float64)
    out: list[tuple[int, float]] = []
    step = max(1, int(span_step))
    for pick in picks:
        span = int(pick.span)
        if span > int(TRAJECTORY_SCORE_MAX_SPAN):
            continue
        if span % step != 0:
            continue
        i = int(pick.i)
        j = int(pick.j)
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
            val = float(arr[i, j])
            if np.isfinite(val):
                out.append((span, val))
    return out


def _path_span_average_score(
    score: Optional[np.ndarray],
    picks: Sequence[TrianglePick],
    *,
    span_step: int = 1,
) -> Optional[float]:
    vals = [val for _span, val in _path_span_scores(score, picks, span_step=span_step)]
    if not vals:
        return None
    return float(np.mean(vals))


def _path_average_score(score: Optional[np.ndarray], picks: Sequence[TrianglePick]) -> Optional[float]:
    vals = _path_scores(score, picks)
    if not vals:
        return None
    return float(np.mean(vals))


def _path_label(label: str, score: Optional[np.ndarray], picks: Sequence[TrianglePick]) -> str:
    avg = _path_average_score(score, picks)
    if avg is None:
        return label
    return f"{label} (avg={avg:.3f})"


def _display_token_region_label(region: str) -> str:
    region_str = str(region)
    if region_str == "special":
        return "register"
    return region_str


def render_gt_trajectory_score_plot(
    score: np.ndarray,
    trajectories: Sequence[tuple[str, Sequence[TrianglePick]]],
    *,
    out_dir: Path,
    backbone: str,
    relative_degradation_cap_multiple: float = 10.0,
) -> Optional[Path]:
    """Plot GT aggregate relative-degradation score along each selected trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    score_arr = np.asarray(score, dtype=np.float64)
    prepared: list[tuple[str, list[tuple[int, float]], Optional[float]]] = []
    span_step = max(1, int(TRAJECTORY_SCORE_SPAN_STEP))
    for name, picks in trajectories:
        points = sorted(
            _path_span_scores(score_arr, picks, span_step=span_step),
            key=lambda item: item[0],
        )
        avg = _path_span_average_score(score_arr, picks, span_step=span_step)
        if points:
            prepared.append((str(name), points, avg))
    if not prepared:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_cycle = plt.get_cmap("tab10")
    color_index = 0
    max_y = 1.0
    max_x = 1
    for name, points, avg in prepared:
        xs = [span for span, _score in points]
        ys = [val for _span, val in points]
        max_y = max(max_y, max(ys))
        max_x = max(max_x, max(xs))
        if name.lower().startswith("gt"):
            color = "red"
            linestyle = "--"
            marker = "o"
        else:
            color = color_cycle(color_index % 10)
            color_index += 1
            linestyle = "-"
            marker = "s"
        label = name if avg is None else f"{name} (avg={avg:.3f})"
        ax.plot(
            xs,
            ys,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            linewidth=1.7,
            color=color,
            label=label,
        )

    cap = _relative_degradation_vmax([score_arr], relative_degradation_cap_multiple)
    ax.set_title(
        (
            f"{backbone}: GT aggregate score along selected trajectories "
            f"(span ≤ {TRAJECTORY_SCORE_MAX_SPAN}, step={span_step})"
        ),
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Dropped units / span", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative degradation multiple (lower is better)", fontsize=12, fontweight="bold")
    ax.set_xlim(left=0, right=min(max_x, int(TRAJECTORY_SCORE_MAX_SPAN)) + 0.5)
    ax.set_ylim(bottom=1.0, top=max(cap, max_y) * 1.03)
    ax.xaxis.set_major_locator(MultipleLocator(span_step))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_gt_trajectory_scores.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _draw_finite_score_candidate_segments(
    ax,
    score: np.ndarray,
    *,
    label: str = "Span candidates",
) -> None:
    """Draw light candidate segments for every span with finite GT scores."""
    arr = np.asarray(score, dtype=np.float64)
    if arr.ndim != 2:
        return
    n = int(arr.shape[0])
    seen_label = False
    for span in range(1, n):
        segments: list[list[tuple[int, int]]] = []
        current_segment: list[tuple[int, int]] = []
        previous_i: Optional[int] = None
        for i in range(0, n - span):
            j = i + span
            if not np.isfinite(float(arr[i, j])):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                previous_i = None
                continue
            if previous_i is not None and i != previous_i + 1:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
            current_segment.append((i, j))
            previous_i = i
        if current_segment:
            segments.append(current_segment)
        for segment in segments:
            if len(segment) < 2:
                continue
            points = [triangle_xy(i, j) for i, j in segment]
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            ax.plot(
                xs,
                ys,
                "-",
                color="white",
                alpha=0.42,
                linewidth=1.65,
                label=label if not seen_label else None,
            )
            seen_label = True


def _draw_cka_trajectories(
    ax,
    score: np.ndarray,
    trajectories: Sequence[tuple[str, Sequence[TrianglePick]]],
    *,
    label_vertices: bool = False,
) -> bool:
    """Draw one or more CKA-selected trajectories and return whether anything was drawn."""
    if not trajectories:
        return False
    colors = plt.get_cmap("tab10")
    drawn = False
    for index, (name, picks) in enumerate(trajectories):
        if not picks:
            continue
        points = [triangle_xy(pick.i, pick.j) for pick in picks]
        xs = [x for x, _y in points]
        ys = [y for _x, y in points]
        if not xs:
            continue
        color = colors(index % 10)
        ax.plot(
            xs,
            ys,
            "-",
            color=color,
            alpha=0.95,
            linewidth=1.6,
            label=_path_label(str(name), score, picks),
        )
        ax.plot(
            xs,
            ys,
            "s",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=3.0,
            linestyle="None",
            label=f"{name} vertex" if label_vertices and index == 0 else None,
        )
        drawn = True
    return drawn


def render_budget_similarity_curve(
    picks: Sequence[TrianglePick],
    sim: np.ndarray,
    budget_drop_counts: Mapping[int, int],
    *,
    out_dir: Path,
    backbone: str,
    threshold: Optional[float],
    selection_label: str = "Endpoint-linked triangle",
) -> Path:
    """Plot selected corner similarity by drop count and mark requested budgets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [int(pick.span) for pick in picks]
    ys = [float(sim[int(pick.i), int(pick.j)]) for pick in picks]
    span_to_y = {x: y for x, y in zip(xs, ys)}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=str(selection_label),
    )
    colors = plt.get_cmap("tab10")
    marker_offsets: dict[int, int] = {}
    for index, (budget, drop_count) in enumerate(budget_drop_counts.items()):
        x = int(drop_count)
        y = 1.0 if x == 0 else span_to_y.get(x)
        if y is None:
            continue
        color = colors(index % 10)
        ax.scatter([x], [y], s=65, color=color, edgecolors="white", linewidths=0.8, zorder=4)
        overlap_index = marker_offsets.get(x, 0)
        marker_offsets[x] = overlap_index + 1
        ax.annotate(
            f"{int(budget)}%\n{x} units · sim={y:.4f}",
            (x, y),
            xytext=(0, -30 - 42 * ((index + overlap_index) % 2)),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=color,
            horizontalalignment="center",
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.88,
            },
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2},
            annotation_clip=False,
        )

    if threshold is None:
        title = f"{backbone}: endpoint-linked triangle corner similarity"
    else:
        title = (
            f"{backbone}: dense-triangle corner similarity "
            f"(threshold > {float(threshold):g})"
        )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Dropped units / triangle span", fontsize=14, fontweight="bold")
    ax.set_ylabel("Selected corner similarity", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("bold")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", prop={"size": 12, "weight": "bold"})
    fig.tight_layout()
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_all_span_corner_heatmap(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    picks: Sequence[TrianglePick],
    *,
    span_to_candidate_picks: Optional[Mapping[int, Sequence[TrianglePick]]] = None,
    out_dir: Path,
    backbone: str,
    tag: Optional[str] = None,
    title: Optional[str] = None,
    gt_score: Optional[np.ndarray] = None,
    gt_picks: Optional[Sequence[TrianglePick]] = None,
    selected_path_label: str = "Selected path",
) -> Path:
    """Render the similarity heatmap and mark the selected corner for every span."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(sim.shape[0])
    arr = _sim_to_heatmap_grid(np.asarray(sim, dtype=np.float64))
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = imshow_upper_triangle(ax, masked, n, cmap="viridis", vmin=0.0, vmax=1.0)
    if title is None:
        title = f"{backbone}: selected endpoint-linked triangle vertices by dropped-unit count"
    ax.set_title(title, fontsize=13, fontweight="bold")
    configure_triangle_axes(ax, n, index_name="unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    seen_candidate_label = False
    for pick in picks:
        span = int(pick.span)
        if span_to_candidate_picks is None:
            candidate_segments = [
                [(i, i + span) for i in range(0, n - span)]
            ]
        else:
            candidates = sorted(
                span_to_candidate_picks.get(span, ()),
                key=lambda candidate: int(candidate.i),
            )
            candidate_segments: list[list[tuple[int, int]]] = []
            current_segment: list[tuple[int, int]] = []
            previous_i: Optional[int] = None
            for candidate in candidates:
                i = int(candidate.i)
                j = int(candidate.j)
                if previous_i is not None and i != previous_i + 1:
                    if current_segment:
                        candidate_segments.append(current_segment)
                    current_segment = []
                current_segment.append((i, j))
                previous_i = i
            if current_segment:
                candidate_segments.append(current_segment)
        for segment in candidate_segments:
            if len(segment) < 2:
                continue
            points = [triangle_xy(i, j) for i, j in segment]
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            ax.plot(
                xs,
                ys,
                "-",
                color="white",
                alpha=0.42,
                linewidth=1.65,
                label="Span candidates" if not seen_candidate_label else None,
            )
            seen_candidate_label = True

    if gt_picks:
        gt_points = [triangle_xy(pick.i, pick.j) for pick in gt_picks]
        gt_xs = [x for x, _y in gt_points]
        gt_ys = [y for _x, y in gt_points]
        if gt_xs:
            ax.plot(
                gt_xs,
                gt_ys,
                "--",
                color="red",
                alpha=0.95,
                linewidth=1.8,
                label=_path_label("GT best path", gt_score, gt_picks),
            )
            ax.plot(
                gt_xs,
                gt_ys,
                "o",
                markerfacecolor="red",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=3.5,
                linestyle="None",
                label="GT best vertex",
            )

    selected_points = [triangle_xy(pick.i, pick.j) for pick in picks]
    selected_xs = [x for x, _y in selected_points]
    selected_ys = [y for _x, y in selected_points]
    selected_color = "#c000ff" if gt_picks else "red"
    selected_marker = "s" if gt_picks else "o"
    if selected_xs:
        ax.plot(
            selected_xs,
            selected_ys,
            "-",
            color=selected_color,
            alpha=0.95 if gt_picks else 0.85,
            linewidth=1.7 if gt_picks else 1.0,
            label=_path_label(selected_path_label, gt_score, picks),
        )
    for k, pick in enumerate(picks):
        x, y = triangle_xy(pick.i, pick.j)
        ax.plot(
            x,
            y,
            selected_marker,
            markerfacecolor=selected_color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=3.2 if gt_picks else 4,
            label="Selected vertex" if k == 0 else None,
        )

    if picks or gt_picks:
        ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=9)

    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    if tag:
        safe_tag = str(tag).replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"prune_plan_{safe_backbone}_{safe_tag}.png"
    else:
        out_path = out_dir / f"prune_plan_{safe_backbone}_all_corners.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path




def _draw_span_candidate_segments_from_picks(
    ax,
    *,
    n: int,
    picks: Sequence[TrianglePick],
    span_to_candidate_picks: Optional[Mapping[int, Sequence[TrianglePick]]] = None,
    label: str = "Span candidates",
) -> bool:
    seen_candidate_label = False
    for pick in picks:
        span = int(pick.span)
        if span_to_candidate_picks is None:
            candidate_segments = [[(i, i + span) for i in range(0, n - span)]]
        else:
            candidates = sorted(
                span_to_candidate_picks.get(span, ()),
                key=lambda candidate: int(candidate.i),
            )
            candidate_segments: list[list[tuple[int, int]]] = []
            current_segment: list[tuple[int, int]] = []
            previous_i: Optional[int] = None
            for candidate in candidates:
                i = int(candidate.i)
                j = int(candidate.j)
                if previous_i is not None and i != previous_i + 1:
                    if current_segment:
                        candidate_segments.append(current_segment)
                    current_segment = []
                current_segment.append((i, j))
                previous_i = i
            if current_segment:
                candidate_segments.append(current_segment)
        for segment in candidate_segments:
            if len(segment) < 2:
                continue
            points = [triangle_xy(i, j) for i, j in segment]
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            ax.plot(
                xs,
                ys,
                "-",
                color="white",
                alpha=0.42,
                linewidth=1.9,
                label=label if not seen_candidate_label else None,
            )
            seen_candidate_label = True
    return seen_candidate_label


def _draw_gt_path(
    ax,
    score: Optional[np.ndarray],
    picks: Sequence[TrianglePick],
    *,
    label: str = "GT best path",
    marker_label: Optional[str] = None,
) -> bool:
    if not picks:
        return False
    points = [triangle_xy(pick.i, pick.j) for pick in picks]
    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    if not xs:
        return False
    ax.plot(
        xs,
        ys,
        "--",
        color="red",
        alpha=0.95,
        linewidth=1.8,
        label=_path_label(label, score, picks),
    )
    ax.plot(
        xs,
        ys,
        "o",
        markerfacecolor="red",
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=3.2,
        linestyle="None",
        label=marker_label,
    )
    return True


def _draw_selected_path(
    ax,
    score: Optional[np.ndarray],
    picks: Sequence[TrianglePick],
    *,
    label: str = "CKA selected path",
    color: str = "#c000ff",
    marker: str = "s",
    marker_label: Optional[str] = None,
) -> bool:
    if not picks:
        return False
    points = [triangle_xy(pick.i, pick.j) for pick in picks]
    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    if not xs:
        return False
    ax.plot(
        xs,
        ys,
        "-",
        color=color,
        alpha=0.95,
        linewidth=1.7,
        label=_path_label(label, score, picks),
    )
    ax.plot(
        xs,
        ys,
        marker,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=3.0,
        linestyle="None",
        label=marker_label,
    )
    return True


def render_token_region_corner_grid(
    token_region_items: Sequence[
        tuple[str, np.ndarray, Sequence[TrianglePick], Optional[Mapping[int, Sequence[TrianglePick]]]]
    ],
    *,
    out_dir: Path,
    backbone: str,
    selected_token_region: str,
    gt_score: Optional[np.ndarray] = None,
    gt_picks: Optional[Sequence[TrianglePick]] = None,
    selection_label: str = "Endpoint-linked triangle",
) -> Path:
    """Render all token-region corner heatmaps in one 2x3 comparison figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not token_region_items:
        raise ValueError("token_region_items must contain at least one token region")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)
    used_axes = []
    last_im = None
    selected_region = str(selected_token_region)
    displayed_items = list(token_region_items[:5])

    for index, (region, sim, picks, candidates_by_span) in enumerate(displayed_items):
        row = 0 if index < 3 else 1
        col = index if index < 3 else index - 3
        ax = axes[row][col]
        used_axes.append(ax)
        n = int(sim.shape[0])
        arr = _sim_to_heatmap_grid(np.asarray(sim, dtype=np.float64))
        masked = np.ma.masked_invalid(arr)
        last_im = imshow_upper_triangle(ax, masked, n, cmap="viridis", vmin=0.0, vmax=1.0)
        display_region = _display_token_region_label(region)
        is_selected = str(region) == selected_region
        avg = _path_average_score(gt_score, picks)
        title = f"{display_region}"
        if is_selected:
            title = f"SELECTED: {title}"
        if avg is not None:
            title += f"\nCKA avg={avg:.3f}"
        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold" if is_selected else "normal",
            color="red" if is_selected else "black",
        )
        configure_triangle_axes(ax, n, index_name="unit")
        ax.tick_params(axis="both", labelsize=7)
        _draw_span_candidate_segments_from_picks(
            ax,
            n=n,
            picks=picks,
            span_to_candidate_picks=candidates_by_span,
        )
        if gt_picks:
            _draw_gt_path(ax, gt_score, gt_picks)
        _draw_selected_path(
            ax,
            gt_score,
            picks,
            label=f"CKA selected path ({display_region})",
            color="#c000ff",
        )

    for index in range(len(displayed_items), 6):
        row = 0 if index < 3 else 1
        col = index if index < 3 else index - 3
        axes[row][col].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=used_axes, fraction=0.025, pad=0.015)
        cbar.set_label("CKA similarity", fontsize=10)

    legend_ax = axes[1][2]
    if len(displayed_items) < 6:
        legend_ax.axis("off")
        handles = [
            Line2D([0], [0], color="white", linewidth=1.9, label="Span candidates"),
            Line2D([0], [0], color="#c000ff", marker="s", linewidth=1.7, label="CKA selected path"),
        ]
        if gt_picks:
            handles.insert(
                1,
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="--",
                    marker="o",
                    linewidth=1.8,
                    label=_path_label("GT best path", gt_score, gt_picks),
                ),
            )
        legend_ax.legend(handles=handles, loc="center", fontsize=10)

    fig.suptitle(
        f"{backbone}: token-region CKA corner-path comparison ({selection_label})",
        fontsize=16,
        fontweight="bold",
    )
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_token_corners.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _normalize_display_heatmap(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out, float("nan"), float("nan")
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return out, lo, hi


def render_gt_aggregate_corner_heatmap(
    score: np.ndarray,
    picks: Sequence[TrianglePick],
    *,
    out_dir: Path,
    backbone: str,
    metric_names: Sequence[str],
    cka_picks: Optional[Sequence[TrianglePick]] = None,
    cka_trajectories: Optional[Sequence[tuple[str, Sequence[TrianglePick]]]] = None,
    relative_degradation_cap_multiple: float = 10.0,
    component_aggregation: str = "mean",
) -> Path:
    """Render aggregate 1..10 skip-eval relative-degradation heatmap C and GT path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(score.shape[0])
    score_arr = np.asarray(score, dtype=np.float64)
    finite_score = np.isfinite(score_arr)
    raw_min = float(np.nanmin(score_arr[finite_score])) if np.any(finite_score) else float("nan")
    raw_max = float(np.nanmax(score_arr[finite_score])) if np.any(finite_score) else float("nan")
    arr = _sim_to_heatmap_grid(score_arr)
    masked = np.ma.masked_invalid(arr)
    vmax = _relative_degradation_vmax([score_arr], relative_degradation_cap_multiple)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = imshow_upper_triangle(ax, masked, n, cmap="viridis_r", vmin=1.0, vmax=vmax)
    ax.set_title(
        f"{backbone}: GT-best path on skip-eval relative-degradation heatmap",
        fontsize=13,
        fontweight="bold",
    )
    configure_triangle_axes(ax, n, index_name="unit")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _format_relative_degradation_colorbar(cbar)
    cbar.set_label(
        (
            f"Aggregate relative degradation multiple (1x = upper, {vmax:g}x = capped)"
            if float(relative_degradation_cap_multiple) >= 1.0
            else "Aggregate relative degradation multiple (1x = upper, uncapped)"
        ),
        fontsize=10,
    )
    _draw_finite_score_candidate_segments(ax, score_arr)

    gt_points = [triangle_xy(pick.i, pick.j) for pick in picks]
    gt_xs = [x for x, _y in gt_points]
    gt_ys = [y for _x, y in gt_points]
    if gt_xs:
        ax.plot(
            gt_xs,
            gt_ys,
            "--",
            color="red",
            alpha=0.95,
            linewidth=1.8,
            label=_path_label("GT best path", score_arr, picks),
        )
        ax.plot(
            gt_xs,
            gt_ys,
            "o",
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=3.5,
            linestyle="None",
            label="GT best vertex",
        )

    cka_items: list[tuple[str, Sequence[TrianglePick]]] = []
    if cka_trajectories is not None:
        cka_items.extend((str(name), picks) for name, picks in cka_trajectories)
    elif cka_picks:
        cka_items.append(("CKA selected path", cka_picks))
    drew_cka = _draw_cka_trajectories(ax, score_arr, cka_items)

    if metric_names:
        score_range = (
            f"C multiple range: [{raw_min:.4f}, {raw_max:.4f}]\n"
            if np.isfinite(raw_min) and np.isfinite(raw_max)
            else ""
        )
        ax.text(
            0.015,
            0.015,
            (
                f"{score_range}aggregation: {component_aggregation}\n"
                f"metrics: {', '.join(str(m) for m in metric_names)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#888888",
                "alpha": 0.82,
            },
        )
    if gt_xs or drew_cka:
        ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=9)

    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_gt_corners.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_gt_component_heatmaps(
    component_heatmaps: Mapping[str, np.ndarray],
    *,
    out_dir: Path,
    backbone: str,
    relative_degradation_cap_multiple: float = 10.0,
) -> Optional[Path]:
    """Render per-metric 1..10 relative-degradation heatmaps B used for aggregate C."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_grid: list[list[Optional[str]]] = [
        ["ape", "are", "rpe_trans", "rpe_rot"],
        ["acc", "comp", "chamfer_distance", "nc"],
    ]
    if not component_heatmaps:
        return None
    first = next(iter(component_heatmaps.values()))
    n = int(first.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(5.0 * 4, 4.4 * 2), squeeze=False)
    for row_idx, row in enumerate(metric_grid):
        for col_idx, metric in enumerate(row):
            ax = axes[row_idx][col_idx]
            if metric is None or metric not in component_heatmaps:
                ax.axis("off")
                continue
            component = np.asarray(component_heatmaps[metric], dtype=np.float64)
            vmax = _relative_degradation_vmax(
                [component],
                relative_degradation_cap_multiple,
            )
            arr = _sim_to_heatmap_grid(component)
            masked = np.ma.masked_invalid(arr)
            im = imshow_upper_triangle(ax, masked, n, cmap="viridis_r", vmin=1.0, vmax=vmax)
            ax.set_title(str(metric), fontsize=11, fontweight="bold")
            configure_triangle_axes(ax, n, index_name="unit")
            ax.tick_params(axis="both", labelsize=6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _format_relative_degradation_colorbar(cbar)
            cbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        f"{backbone}: skip-eval component relative-degradation heatmaps (B, lower is better)",
        fontsize=15,
        fontweight="bold",
    )
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_gt_components.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_gt_ranked_candidate_heatmap(
    score: np.ndarray,
    picks: Sequence[TrianglePick],
    *,
    out_dir: Path,
    backbone: str,
    cka_picks: Optional[Sequence[TrianglePick]] = None,
    cka_trajectories: Optional[Sequence[tuple[str, Sequence[TrianglePick]]]] = None,
) -> Path:
    """Render per-span min-max candidate brightness from aggregate skip-eval score C.

    For each span (diagonal), finite candidate scores are linearly rescaled
    within that span.  The best/lowest score receives brightness 1.0, the
    worst/highest score receives 0.0, and intermediate candidates preserve
    their absolute score gaps within the span.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(score, dtype=np.float64)
    n = int(values.shape[0])
    ranked = np.full((n, n), np.nan, dtype=np.float64)
    for span in range(1, n):
        candidates: list[tuple[int, int, float]] = []
        for i in range(0, n - span):
            j = i + span
            val = float(values[i, j])
            if np.isfinite(val):
                candidates.append((i, j, val))
        if not candidates:
            continue
        span_values = np.asarray([val for _i, _j, val in candidates], dtype=np.float64)
        lo = float(np.nanmin(span_values))
        hi = float(np.nanmax(span_values))
        if hi <= lo:
            for i, j, _val in candidates:
                ranked[i, j] = 1.0
        else:
            denom = hi - lo
            for i, j, val in candidates:
                ranked[i, j] = 1.0 - (float(val) - lo) / denom

    arr = _sim_to_heatmap_grid(ranked)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = imshow_upper_triangle(ax, masked, n, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(
        f"{backbone}: GT candidates min-max scaled within each dropped-unit count",
        fontsize=13,
        fontweight="bold",
    )
    configure_triangle_axes(ax, n, index_name="unit")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Within-span min-max brightness (1 = best/lowest score)", fontsize=10)
    _draw_finite_score_candidate_segments(ax, values)

    gt_points = [triangle_xy(pick.i, pick.j) for pick in picks]
    gt_xs = [x for x, _y in gt_points]
    gt_ys = [y for _x, y in gt_points]
    if gt_xs:
        ax.plot(
            gt_xs,
            gt_ys,
            "--",
            color="red",
            alpha=0.95,
            linewidth=1.8,
            label=_path_label("GT best path", values, picks),
        )
        ax.plot(
            gt_xs,
            gt_ys,
            "o",
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=3.5,
            linestyle="None",
            label=None,
        )
    cka_items: list[tuple[str, Sequence[TrianglePick]]] = []
    if cka_trajectories is not None:
        cka_items.extend((str(name), picks) for name, picks in cka_trajectories)
    elif cka_picks:
        cka_items.append(("CKA selected path", cka_picks))
    drew_cka = _draw_cka_trajectories(ax, values, cka_items)
    if gt_xs or drew_cka:
        ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=9)

    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_gt_ranked_candidates.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _ranked_candidate_brightness(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    n = int(values_arr.shape[0])
    ranked = np.full((n, n), np.nan, dtype=np.float64)
    for span in range(1, n):
        candidates: list[tuple[int, int, float]] = []
        for i in range(0, n - span):
            j = i + span
            val = float(values_arr[i, j])
            if np.isfinite(val):
                candidates.append((i, j, val))
        if not candidates:
            continue
        span_values = np.asarray([val for _i, _j, val in candidates], dtype=np.float64)
        lo = float(np.nanmin(span_values))
        hi = float(np.nanmax(span_values))
        if hi <= lo:
            for i, j, _val in candidates:
                ranked[i, j] = 1.0
        else:
            denom = hi - lo
            for i, j, val in candidates:
                ranked[i, j] = 1.0 - (float(val) - lo) / denom
    return ranked


def render_gt_corner_comparison_heatmap(
    score: np.ndarray,
    picks: Sequence[TrianglePick],
    *,
    out_dir: Path,
    backbone: str,
    metric_names: Sequence[str],
    cka_trajectories: Optional[Sequence[tuple[str, Sequence[TrianglePick]]]] = None,
    relative_degradation_cap_multiple: float = 10.0,
    component_aggregation: str = "mean",
) -> Path:
    """Render GT aggregate and within-span candidate brightness side-by-side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    score_arr = np.asarray(score, dtype=np.float64)
    n = int(score_arr.shape[0])
    finite_score = np.isfinite(score_arr)
    raw_min = float(np.nanmin(score_arr[finite_score])) if np.any(finite_score) else float("nan")
    raw_max = float(np.nanmax(score_arr[finite_score])) if np.any(finite_score) else float("nan")
    vmax = _relative_degradation_vmax([score_arr], relative_degradation_cap_multiple)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.7), squeeze=False)
    ax_score = axes[0][0]
    ax_ranked = axes[0][1]

    score_grid = _sim_to_heatmap_grid(score_arr)
    score_masked = np.ma.masked_invalid(score_grid)
    im_score = imshow_upper_triangle(
        ax_score,
        score_masked,
        n,
        cmap="viridis_r",
        vmin=1.0,
        vmax=vmax,
    )
    ax_score.set_title("GT aggregate relative degradation", fontsize=13, fontweight="bold")
    configure_triangle_axes(ax_score, n, index_name="unit")
    cbar_score = fig.colorbar(im_score, ax=ax_score, fraction=0.046, pad=0.04)
    _format_relative_degradation_colorbar(cbar_score)
    cbar_score.set_label(
        (
            f"Relative degradation multiple (1x = upper, {vmax:g}x = capped)"
            if float(relative_degradation_cap_multiple) >= 1.0
            else "Relative degradation multiple (1x = upper, uncapped)"
        ),
        fontsize=9,
    )

    ranked = _ranked_candidate_brightness(score_arr)
    ranked_grid = _sim_to_heatmap_grid(ranked)
    ranked_masked = np.ma.masked_invalid(ranked_grid)
    im_ranked = imshow_upper_triangle(
        ax_ranked,
        ranked_masked,
        n,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_ranked.set_title(
        "GT candidates min-max scaled within each span",
        fontsize=13,
        fontweight="bold",
    )
    configure_triangle_axes(ax_ranked, n, index_name="unit")
    cbar_ranked = fig.colorbar(im_ranked, ax=ax_ranked, fraction=0.046, pad=0.04)
    cbar_ranked.set_label("Within-span brightness (1 = best/lowest score)", fontsize=9)

    cka_items = list(cka_trajectories or [])
    for ax in (ax_score, ax_ranked):
        _draw_finite_score_candidate_segments(ax, score_arr)
        drew_gt = _draw_gt_path(ax, score_arr, picks)
        drew_cka = _draw_cka_trajectories(ax, score_arr, cka_items)
        if drew_gt or drew_cka:
            ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=8)

    if metric_names:
        score_range = (
            f"C multiple range: [{raw_min:.4f}, {raw_max:.4f}]\n"
            if np.isfinite(raw_min) and np.isfinite(raw_max)
            else ""
        )
        ax_score.text(
            0.015,
            0.015,
            (
                f"{score_range}aggregation: {component_aggregation}\n"
                f"metrics: {', '.join(str(m) for m in metric_names)}"
            ),
            transform=ax_score.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#888888",
                "alpha": 0.82,
            },
        )

    fig.suptitle(
        f"{backbone}: GT corner path and candidate-brightness comparison",
        fontsize=16,
        fontweight="bold",
    )
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_gt_corners_and_ranked_candidates.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_corner_overview_heatmap(
    token_region_items: Sequence[
        tuple[str, np.ndarray, Sequence[TrianglePick], Optional[Mapping[int, Sequence[TrianglePick]]]]
    ],
    *,
    out_dir: Path,
    backbone: str,
    selected_token_region: str,
    gt_score: Optional[np.ndarray] = None,
    gt_picks: Optional[Sequence[TrianglePick]] = None,
    metric_names: Sequence[str] = (),
    cka_trajectories: Optional[Sequence[tuple[str, Sequence[TrianglePick]]]] = None,
    relative_degradation_cap_multiple: float = 10.0,
    component_aggregation: str = "mean",
    selection_label: str = "Endpoint-linked triangle",
) -> Path:
    """Render token CKA corners plus GT aggregate/ranked candidates in one 2-column grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not token_region_items and gt_score is None:
        raise ValueError("Need at least one token-region panel or a GT score panel")

    panels: list[tuple[str, object]] = []
    for item in token_region_items:
        panels.append(("token", item))
    if gt_score is not None and gt_picks:
        panels.append(("gt_score", None))
        panels.append(("gt_ranked", None))

    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15.5, max(6.4, rows * 6.0)),
        squeeze=False,
    )

    selected_region = str(selected_token_region)
    score_arr = None if gt_score is None else np.asarray(gt_score, dtype=np.float64)
    gt_pick_list = list(gt_picks or [])
    cka_items = list(cka_trajectories or [])
    raw_min = raw_max = float("nan")
    vmax = 1.0
    if score_arr is not None:
        finite_score = np.isfinite(score_arr)
        raw_min = float(np.nanmin(score_arr[finite_score])) if np.any(finite_score) else float("nan")
        raw_max = float(np.nanmax(score_arr[finite_score])) if np.any(finite_score) else float("nan")
        vmax = _relative_degradation_vmax([score_arr], relative_degradation_cap_multiple)

    for panel_index, (panel_kind, payload) in enumerate(panels):
        row = panel_index // cols
        col = panel_index % cols
        ax = axes[row][col]

        if panel_kind == "token":
            region, sim, picks, candidates_by_span = payload  # type: ignore[misc]
            sim_arr = np.asarray(sim, dtype=np.float64)
            n = int(sim_arr.shape[0])
            grid = _sim_to_heatmap_grid(sim_arr)
            masked = np.ma.masked_invalid(grid)
            im = imshow_upper_triangle(ax, masked, n, cmap="viridis", vmin=0.0, vmax=1.0)
            display_region = _display_token_region_label(str(region))
            is_selected = str(region) == selected_region
            avg = _path_average_score(score_arr, picks) if score_arr is not None else None
            title = f"{display_region}"
            if is_selected:
                title = f"SELECTED: {title}"
            if avg is not None:
                title += f"\nCKA avg={avg:.3f}"
            ax.set_title(
                title,
                fontsize=12,
                fontweight="bold" if is_selected else "normal",
                color="red" if is_selected else "black",
            )
            configure_triangle_axes(ax, n, index_name="unit")
            ax.tick_params(axis="both", labelsize=7)
            _draw_span_candidate_segments_from_picks(
                ax,
                n=n,
                picks=picks,
                span_to_candidate_picks=candidates_by_span,
            )
            if score_arr is not None and gt_pick_list:
                _draw_gt_path(ax, score_arr, gt_pick_list)
            _draw_selected_path(
                ax,
                score_arr,
                picks,
                label=f"CKA selected path ({display_region})",
                color="#c000ff",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("CKA similarity", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            continue

        if score_arr is None:
            ax.axis("off")
            continue

        n = int(score_arr.shape[0])
        if panel_kind == "gt_score":
            score_grid = _sim_to_heatmap_grid(score_arr)
            score_masked = np.ma.masked_invalid(score_grid)
            im = imshow_upper_triangle(
                ax,
                score_masked,
                n,
                cmap="viridis_r",
                vmin=1.0,
                vmax=vmax,
            )
            ax.set_title("GT aggregate relative degradation", fontsize=12, fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _format_relative_degradation_colorbar(cbar)
            cbar.set_label(
                (
                    f"Relative degradation multiple (1x = upper, {vmax:g}x = capped)"
                    if float(relative_degradation_cap_multiple) >= 1.0
                    else "Relative degradation multiple (1x = upper, uncapped)"
                ),
                fontsize=8,
            )
            cbar.ax.tick_params(labelsize=7)
            if metric_names:
                score_range = (
                    f"C multiple range: [{raw_min:.4f}, {raw_max:.4f}]\n"
                    if np.isfinite(raw_min) and np.isfinite(raw_max)
                    else ""
                )
                ax.text(
                    0.015,
                    0.015,
                    (
                        f"{score_range}aggregation: {component_aggregation}\n"
                        f"metrics: {', '.join(str(m) for m in metric_names)}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#888888",
                        "alpha": 0.82,
                    },
                )
        else:
            ranked = _ranked_candidate_brightness(score_arr)
            ranked_grid = _sim_to_heatmap_grid(ranked)
            ranked_masked = np.ma.masked_invalid(ranked_grid)
            im = imshow_upper_triangle(
                ax,
                ranked_masked,
                n,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(
                "GT candidates min-max scaled within each span",
                fontsize=12,
                fontweight="bold",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Within-span brightness (1 = best/lowest score)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        configure_triangle_axes(ax, n, index_name="unit")
        ax.tick_params(axis="both", labelsize=7)
        _draw_finite_score_candidate_segments(ax, score_arr)
        drew_gt = _draw_gt_path(ax, score_arr, gt_pick_list)
        drew_cka = _draw_cka_trajectories(ax, score_arr, cka_items)
        if drew_gt or drew_cka:
            ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=6.5)

    for empty_index in range(len(panels), rows * cols):
        row = empty_index // cols
        col = empty_index % cols
        axes[row][col].axis("off")

    fig.suptitle(
        f"{backbone}: CKA token corners and GT corner comparison ({selection_label})",
        fontsize=16,
        fontweight="bold",
    )
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"prune_plan_{safe_backbone}_corner_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_span_unit_action_grid(
    *,
    num_units: int,
    span_to_drop_indices: Mapping[int, Sequence[int]],
    span_to_lora_indices: Mapping[int, Sequence[int]],
    span_to_cache_indices: Mapping[int, Sequence[int]],
    highlighted_rows: Optional[Mapping[int, Sequence[str]]] = None,
    out_dir: Path,
    backbone: str,
    lora_prefix: int,
    tag: Optional[str] = None,
    lora_label: str = "+LoRA unit",
    lora_marker: str = "+",
    title: Optional[str] = None,
    show_empty_overlays_in_legend: bool = True,
) -> Path:
    """Render an n*n grid: row=drop count, col=unit, cells mark drop/lora/cache."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(num_units)
    drop_cells: set[tuple[int, int]] = set()
    for span, indices in span_to_drop_indices.items():
        row = int(span)
        if row < 0 or row >= n:
            continue
        for idx in indices:
            col = int(idx)
            if 0 <= col < n:
                drop_cells.add((row, col))

    cache_cells: set[tuple[int, int]] = set()
    for span, indices in span_to_cache_indices.items():
        row = int(span)
        if row < 0 or row >= n:
            continue
        for idx in indices:
            col = int(idx)
            if 0 <= col < n:
                cache_cells.add((row, col))

    lora_cells: set[tuple[int, int]] = set()
    for span, indices in span_to_lora_indices.items():
        row = int(span)
        if row < 0 or row >= n:
            continue
        for idx in indices:
            col = int(idx)
            if 0 <= col < n:
                lora_cells.add((row, col))

    fig_size = max(8.0, n * 0.22)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    drop_color = "#ffb3b3"
    cache_color = "#00bcd4"
    lora_edge = "#ff9800"
    grid_edge = "#d0d0d0"
    highlight_edge = "#222222"
    highlight_face = "#fff59d"

    highlighted_rows = highlighted_rows or {}

    for row in range(n):
        for col in range(n):
            cell = (row, col)
            if cell in drop_cells:
                facecolor = drop_color
                edgecolor = grid_edge
                linewidth = 0.45
            else:
                facecolor = "white"
                edgecolor = grid_edge
                linewidth = 0.35
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
            )
            if cell in cache_cells:
                ax.add_patch(
                    Rectangle(
                        (col - 0.28, row - 0.28),
                        0.56,
                        0.56,
                        facecolor="none",
                        edgecolor=cache_color,
                        linewidth=0.9,
                    )
                )
            if cell in lora_cells:
                marker = str(lora_marker or "+")
                if marker == "+":
                    ax.plot(
                        [col - 0.26, col + 0.26],
                        [row, row],
                        color=lora_edge,
                        linewidth=1.0,
                    )
                    ax.plot(
                        [col, col],
                        [row - 0.26, row + 0.26],
                        color=lora_edge,
                        linewidth=1.0,
                    )
                elif marker.lower() in ("x", "×"):
                    ax.plot(
                        [col - 0.24, col + 0.24],
                        [row - 0.24, row + 0.24],
                        color=lora_edge,
                        linewidth=1.0,
                    )
                    ax.plot(
                        [col - 0.24, col + 0.24],
                        [row + 0.24, row - 0.24],
                        color=lora_edge,
                        linewidth=1.0,
                    )
                elif marker.lower() in ("o", "circle"):
                    ax.plot(
                        [col],
                        [row],
                        marker="o",
                        markersize=5.6,
                        markerfacecolor="none",
                        markeredgecolor=lora_edge,
                        markeredgewidth=1.0,
                        linestyle="None",
                    )
                else:
                    ax.plot(
                        [col],
                        [row],
                        marker=marker,
                        markersize=6,
                        color=lora_edge,
                        linestyle="None",
                    )
    for row, labels in sorted(highlighted_rows.items()):
        row_i = int(row)
        if row_i < 0 or row_i >= n:
            continue
        ax.add_patch(
            Rectangle(
                (-0.5, row_i - 0.5),
                float(n),
                1.0,
                facecolor=highlight_face,
                edgecolor="none",
                alpha=0.16,
                zorder=0.5,
            )
        )
        ax.add_patch(
            Rectangle(
                (-0.5, row_i - 0.5),
                float(n),
                1.0,
                facecolor="none",
                edgecolor=highlight_edge,
                linewidth=1.35,
                zorder=4,
            )
        )
        label = " / ".join(str(x) for x in labels if str(x).strip())
        if label:
            ax.text(
                n - 0.15,
                row_i,
                label,
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=highlight_edge,
                zorder=5,
            )
    if title is None:
        title = f"{backbone}: span-by-unit actions (LoRA prefix l{int(lora_prefix)})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("unit index")
    ax.set_ylabel("dropped units / span")
    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal", adjustable="box")
    legend_handles: list = [
        Patch(facecolor=drop_color, edgecolor=grid_edge, label="Dropped unit"),
    ]
    if cache_cells or show_empty_overlays_in_legend:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                markersize=7,
                markerfacecolor="none",
                markeredgecolor=cache_color,
                linestyle="None",
                label="Cache unit",
            )
        )
    if lora_cells or show_empty_overlays_in_legend:
        marker = str(lora_marker or "+")
        legend_marker = "x" if marker.lower() in ("x", "×") else ("o" if marker.lower() in ("o", "circle") else marker)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=legend_marker,
                markersize=9,
                color=lora_edge,
                markerfacecolor="none" if legend_marker == "o" else lora_edge,
                markeredgecolor=lora_edge,
                linestyle="None",
                label=str(lora_label),
            ),
        )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
    )

    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    safe_tag = (
        str(tag).replace("/", "_").replace(" ", "_")
        if tag
        else f"prune_plan_{safe_backbone}_span_unit_grid_l{int(lora_prefix)}"
    )
    out_path = out_dir / f"{safe_tag}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_prune_span_drop_grid(
    *,
    num_units: int,
    span_to_drop_indices: Mapping[int, Sequence[int]],
    out_dir: Path,
    backbone: str,
) -> Path:
    """Render the prune-only n*n grid: row=drop count, col=unit, red cells are dropped."""
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    return render_span_unit_action_grid(
        num_units=num_units,
        span_to_drop_indices=span_to_drop_indices,
        span_to_lora_indices={},
        span_to_cache_indices={},
        out_dir=out_dir,
        backbone=backbone,
        lora_prefix=0,
        tag=f"prune_plan_{safe_backbone}_drop_grid",
        title=f"{backbone}: prune span-by-unit drop grid",
        show_empty_overlays_in_legend=False,
    )


def _sim_to_heatmap_grid(sim: np.ndarray, fill_value: float = float("nan")) -> np.ndarray:
    n = int(sim.shape[0])
    packed = upper_triangle_pack(torch.from_numpy(sim.astype(np.float64)))
    grid = upper_triangle_unpack_for_heatmap(packed, n, fill_value)
    return np.asarray(grid, dtype=np.float64)


def _apply_similarity_cutoff_matrix(sim: np.ndarray, cutoff: float) -> np.ndarray:
    """Heatmap display only: match Heatmaps UI ``_similarity_packed_cutoff_01`` (values < cutoff -> 0)."""
    cutoff = float(np.clip(cutoff, 0.0, 1.0))
    if cutoff <= 0.0:
        return np.asarray(sim, dtype=np.float64)
    out = np.array(sim, dtype=np.float64, copy=True)
    mask = np.isfinite(out) & (out < cutoff)
    out[mask] = 0.0
    return out


def render_triangle_prune_heatmap(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    picks: Sequence[TrianglePick],
    *,
    out_dir: Path,
    threshold: Optional[float],
    tag: str = "triangle_prune",
    title: Optional[str] = None,
    lora_unit_indices: Optional[Sequence[int]] = None,
    cache_unit_indices: Optional[Sequence[int]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(sim.shape[0])
    display_cutoff = 0.0 if threshold is None else float(threshold)
    arr = _sim_to_heatmap_grid(_apply_similarity_cutoff_matrix(sim, display_cutoff))
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = imshow_upper_triangle(ax, masked, n, cmap="viridis", vmin=0.0, vmax=1.0)
    if title is None:
        if threshold is None:
            title = f"CKA + legal triangles (n={len(picks)})"
        else:
            title = f"CKA + dense triangles (tau={threshold}, n={len(picks)})"
    ax.set_title(title)
    configure_triangle_axes(ax, n, index_name="unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k, pick in enumerate(picks):
        x, y = triangle_xy(pick.i, pick.j)
        ax.plot(
            x,
            y,
            "o",
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.5,
            markersize=12,
            label=f"T{k} ({pick.i},{pick.j})" if k < 6 else None,
        )
    drop_idxs = sorted({idx for p in picks for idx in range(int(p.i) + 1, int(p.j) + 1) if idx >= 0})
    for k, idx in enumerate(drop_idxs):
        ax.plot(
            idx,
            idx,
            "_",
            color="#ff6b6b",
            markersize=13,
            markeredgewidth=2.0,
            label="Dropped unit" if k == 0 else None,
        )
    cache_idxs = sorted({int(x) for x in (cache_unit_indices or []) if int(x) >= 0})
    for k, idx in enumerate(cache_idxs):
        ax.plot(
            idx,
            idx,
            "s",
            markerfacecolor="none",
            markeredgecolor="#00e5ff",
            markeredgewidth=1.2,
            markersize=9,
            label="Cache unit" if k == 0 else None,
        )
    lora_idxs = sorted({int(x) for x in (lora_unit_indices or []) if int(x) >= 0})
    for k, idx in enumerate(lora_idxs):
        ax.plot(
            idx,
            idx,
            "+",
            color="#ffa500",
            markeredgewidth=1.8,
            markersize=10,
            label="LoRA unit" if k == 0 else None,
        )
    if picks or lora_idxs or drop_idxs or cache_idxs:
        ax.legend(loc=TRIANGLE_EMPTY_LEGEND_LOC, fontsize=9)

    safe_tag = tag.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{safe_tag}_triangles.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _synthetic_unit_ids(depth: int = 24) -> List[str]:
    ids: List[str] = []
    for i in range(depth):
        ids.append(f"frame_{i}")
        ids.append(f"global_{i}")
    return ids


def _synthetic_sim(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0.5, 1.0, size=(n, n)).astype(np.float64)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    # Dense block in upper triangle around units 10..20
    for i in range(8, 18):
        for j in range(i + 1, 22):
            sim[i, j] = sim[j, i] = rng.uniform(0.92, 0.99)
    return sim


def _find_npz_under(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    for pattern in ("redundancy_graph_all.npz", "redundancy_graph_v3.npz"):
        candidates = sorted(root.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Visualize triangle greedy picks on CKA heatmap.")
    p.add_argument(
        "--redundancy-root",
        type=Path,
        default=Path("analysis_outputs_v3/first"),
        help="Search for redundancy_graph_all.npz (or legacy redundancy_graph_v3.npz) under this tree.",
    )
    p.add_argument("--npz", type=Path, default=None, help="Explicit redundancy npz path.")
    p.add_argument("--out-dir", type=Path, default=Path("tmp/triangle_prune_viz"))
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--max-triangles", type=int, default=2)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic 48x48 CKA if no npz.")
    args = p.parse_args(list(argv) if argv is not None else None)

    npz_path = args.npz
    if npz_path is None:
        npz_path = _find_npz_under(args.redundancy_root)

    if npz_path is not None and npz_path.is_file():
        z = np.load(npz_path, allow_pickle=True)
        unit_ids = [str(u) for u in list(z["unit_ids"])]
        sim = np.asarray(z["cka_mean"], dtype=np.float64)
        tag = npz_path.parent.parent.name + "_" + npz_path.parent.name
    elif args.synthetic or npz_path is None:
        unit_ids = _synthetic_unit_ids(24)
        sim = _synthetic_sim(len(unit_ids))
        tag = "synthetic_vggt48"
        print("No redundancy graph npz found; using synthetic CKA matrix.", flush=True)
    else:
        raise FileNotFoundError(f"Missing npz: {npz_path}")

    picks, _ = greedy_dense_triangles(
        sim, unit_ids, threshold=float(args.threshold), max_triangles=int(args.max_triangles)
    )
    out = render_triangle_prune_heatmap(
        sim,
        unit_ids,
        picks,
        out_dir=args.out_dir,
        threshold=float(args.threshold),
        tag=tag,
    )
    print(f"Wrote {out} ({len(picks)} triangles)", flush=True)
    for k, pick in enumerate(picks):
        print(f"  [{k}] (i,j)=({pick.i},{pick.j}) span={pick.span} anchor={pick.unit_i}", flush=True)


if __name__ == "__main__":
    main()

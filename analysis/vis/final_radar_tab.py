"""Final-table recovery points rendered as one radar per backbone."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

from analysis.eval.summary_v3 import METRIC_HEADERS
from analysis.prune_plan import load_prune_plan_bundle
from analysis.schema.v3 import UNIFIED_METRIC_KEYS
from analysis.vis.display_names import backbone_display_name
from analysis.vis.final_results_tab import FINAL_TABLE_DATASET_AVERAGE_GROUPS
from analysis.vis.recovery_speed_accuracy_tab import (
    _RADAR_FIELDS,
    _figure_image,
)
from analysis.vis.summary_tab import (
    _BLOCK_PARAMS_HEADER,
    _INFERENCE_TIME_HEADER,
    _PEAK_VRAM_HEADER,
    _add_dataset_average_rows,
    _compression_for_row,
    _load_live_eval_rows,
    _mode_group_for_row,
    _numeric_cell_value,
    _recovery_method_for_row,
    _select_best_recovery_rows,
    _span_value_for_row,
)


FINAL_RADAR_DATASET = "four-dataset testing average"
FINAL_RADAR_RECOVERY_METHOD = "ghost_interval_sequential_autogroup"
FINAL_RADAR_BUDGETS: Tuple[Tuple[str, int], ...] = (
    ("vggt", 10),
    ("pi3", 16),
    ("vggt_omega", 18),
)

_SERIES_COLORS = {
    "upper": "#4c78a8",
    "full_fastvggt": "#b279a2",
    "full_quantvggt": "#72b7b2",
    "recovery": "#e45756",
}
_BACKBONE_LINE_COLORS = {
    "vggt": "#4c78a8",
    "pi3": "#59a14f",
    "vggt_omega": "#f28e2b",
}
_METHOD_LINESTYLES = {
    "upper": "-",
    "full_fastvggt": "--",
    "full_quantvggt": ":",
    "recovery": "-.",
}
_FINAL_RADAR_FIELDS = (
    *_RADAR_FIELDS[:2],
    ("rpe_trans", "RPEₜ↓\n(m)"),
    ("rpe_rot", "RPEᵣ↓\n(°)"),
    *_RADAR_FIELDS[4:7],
    ("inference_time_ms", "Latency\n(ms) ↓"),
    ("peak_allocated_gib", "Memory\n(GiB) ↓"),
    ("param_count", "Params.\n(M) ↓"),
)
_FINAL_RADAR_GROUPS = (
    ("Trajectory Errors", 0, 4, "#4c78a8"),
    ("Geometric Errors", 4, 7, "#59a14f"),
    ("System Efficiency", 7, 10, "#f28e2b"),
)
_RADAR_LEVELS = (1.0 / 3.0, 2.0 / 3.0, 1.0)

_AXIS_LABEL_FONTSIZE = 19.5
_LEVEL_LABEL_FONTSIZE = 14.3
_TITLE_FONTSIZE = 31.2
_METHOD_LEGEND_FONTSIZE = {False: 20.15, True: 18.85}
_GROUP_LEGEND_FONTSIZE = 18.85
_AXIS_LABEL_PAD = 30
_SIDE_AXIS_LABEL_PAD = 35
_TOP_AXIS_LABEL_PAD = 10
_BOTTOM_AXIS_LABEL_PAD = 17
_OUTER_LEVEL_LABEL_RADIUS = 1.06
_AXIS_LABEL_OFFSETS_POINTS = (
    (0, 0),     # APE: up 50, down 20 twice, then down 10
    (-50, -25), # ARE: left 50, down 10 twice, then down 5
    (-60, 0),   # RPE_t: left 50, then left 10
    (-60, 0),   # RPE_r: left 50, then left 10
    (-50, 25),  # Acc.: left 50, up 10 twice, then up 5
    (0, 10),    # Comp.: down 50, up 20 twice, then up 10 twice
    (50, 25),   # CD: right 50, up 10 twice, then up 5
    (60, 0),    # Latency: right 50, then right 10
    (60, 0),    # Memory: right 50, then right 10
    (50, -25),  # Params.: right 50, down 10 twice, then down 5
)


def _format_radar_level_value(field: str, value: float) -> str:
    if field in {"inference_time_ms", "param_count"}:
        return f"{value:.0f}"
    if field == "peak_allocated_gib":
        return f"{value:.1f}"
    return f"{value:.2f}"


def _row_to_radar_point(
    row: Any,
    *,
    state: str,
    budget: int,
    layer_count: int,
    plot: bool = True,
) -> Dict[str, Any]:
    raw_metric: Dict[str, float] = {}
    for metric_key in UNIFIED_METRIC_KEYS:
        header = METRIC_HEADERS[metric_key]
        raw_value = (
            row.raw_metrics.get(header)
            if isinstance(row.raw_metrics, Mapping)
            else None
        )
        value = (
            float(raw_value)
            if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value))
            else _numeric_cell_value(row.metrics.get(header, "—"))
        )
        if value is not None:
            raw_metric[metric_key] = float(value)
    point = {
        "backbone": row.backbone,
        "state": state,
        "budget": int(budget),
        "layer_count": int(layer_count),
        "plot": bool(plot),
        "mode": row.mode,
        "raw_metric": raw_metric,
        "inference_time_ms": _numeric_cell_value(
            row.metrics.get(_INFERENCE_TIME_HEADER, "—")
        ),
        "peak_allocated_gib": _numeric_cell_value(
            row.metrics.get(_PEAK_VRAM_HEADER, "—")
        ),
        "param_count": _numeric_cell_value(
            row.metrics.get(_BLOCK_PARAMS_HEADER, "—")
        ),
    }
    if "quantvggt" in state:
        # QuantVGGT currently has no real INT4 deployment.  Its simulated
        # latency and memory measurements are not comparable and must not be
        # plotted or used to establish radar bounds.
        point["inference_time_ms"] = None
        point["peak_allocated_gib"] = None
    return point


def _point_values(point: Mapping[str, Any]) -> List[float] | None:
    raw = point.get("raw_metric", {})
    values: List[float] = []
    for field, _label in _FINAL_RADAR_FIELDS:
        if "quantvggt" in str(point.get("state", "")) and field in {
            "inference_time_ms",
            "peak_allocated_gib",
        }:
            values.append(float("nan"))
            continue
        value = point.get(field) if field in {
            "inference_time_ms",
            "peak_allocated_gib",
            "param_count",
        } else raw.get(field) if isinstance(raw, Mapping) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return values


def _collect_final_radar_points(
    output_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Load exactly the rows used by Final Table and select its tested budgets."""
    header, rows, live_roots = _load_live_eval_rows(
        Path(output_root),
        prune_strategies=("retention",),
        include_compression=True,
        include_efficiency_metrics=True,
    )
    metric_headers = header[3:] if len(header) > 3 else []
    rows = _add_dataset_average_rows(
        rows,
        metric_headers,
        FINAL_TABLE_DATASET_AVERAGE_GROUPS,
    )
    rows = _select_best_recovery_rows(
        rows,
        include_lower=False,
        share_recovery_winner_across_variants=True,
    )
    result: Dict[str, List[Dict[str, Any]]] = {}
    messages: List[str] = []
    if not live_roots:
        return result, [f"No live eval records found under {output_root}."]

    for backbone, budget in FINAL_RADAR_BUDGETS:
        try:
            total_layers = int(
                load_prune_plan_bundle(Path(output_root) / backbone)["num_units"]
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            messages.append(
                f"{backbone}: unable to load total layer count: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        backbone_rows = [
            row
            for row in rows
            if row.dataset == FINAL_RADAR_DATASET and row.backbone == backbone
        ]
        upper_rows = [row for row in backbone_rows if row.mode == "upper"]
        fastvggt_rows = [
            row
            for row in backbone_rows
            if _mode_group_for_row(row.mode) == "upper"
            and _compression_for_row(row.mode) == "fastvggt"
        ]
        quantvggt_rows = [
            row
            for row in backbone_rows
            if backbone == "vggt"
            and _mode_group_for_row(row.mode) == "upper"
            and _compression_for_row(row.mode) == "quantvggt"
        ]
        recovery_rows = [
            row
            for row in backbone_rows
            if _mode_group_for_row(row.mode) == "recovery"
            and _span_value_for_row(row.mode) == str(budget)
            and _recovery_method_for_row(row.mode) == FINAL_RADAR_RECOVERY_METHOD
            and any(
                source == "retention" or source.startswith("retention_")
                for source in row.pruning_sources
            )
        ]
        upper_points = [
            _row_to_radar_point(
                row,
                state="upper",
                budget=budget,
                layer_count=total_layers,
            )
            for row in upper_rows
        ]
        fastvggt_points = [
            _row_to_radar_point(
                row,
                state="full_fastvggt",
                budget=budget,
                layer_count=total_layers,
            )
            for row in fastvggt_rows
        ]
        quantvggt_points = [
            _row_to_radar_point(
                row,
                state="full_quantvggt",
                budget=budget,
                layer_count=total_layers,
            )
            for row in quantvggt_rows
        ]
        recovery_points = [
            _row_to_radar_point(
                row,
                state="recovery",
                budget=budget,
                layer_count=total_layers - budget,
            )
            for row in recovery_rows
            if not _compression_for_row(row.mode)
        ]
        complete_upper = [point for point in upper_points if _point_values(point)]
        complete_fastvggt = [
            point for point in fastvggt_points if _point_values(point)
        ]
        complete_quantvggt = [
            point for point in quantvggt_points if _point_values(point)
        ]
        complete_recovery = [
            point for point in recovery_points if _point_values(point)
        ]
        if not complete_upper:
            messages.append(f"{backbone}: original model has incomplete radar data.")
            continue
        if not complete_fastvggt:
            messages.append(
                f"{backbone}: original + FastVGGT has incomplete radar data."
            )
            continue
        if backbone == "vggt" and not complete_quantvggt:
            messages.append(
                "vggt: original + QuantVGGT has incomplete radar data; "
                "omitting that optional curve."
            )
        if not complete_recovery:
            messages.append(
                f"{backbone}: recovery model at budget {budget} has incomplete radar data."
            )
            continue
        if len(complete_recovery) > 1:
            messages.append(
                f"{backbone}: multiple complete recovery rows at budget {budget}; "
                f"using {sorted(point['mode'] for point in complete_recovery)[0]}."
            )
        if len(complete_fastvggt) > 1:
            messages.append(
                f"{backbone}: multiple complete original + FastVGGT rows; "
                f"using {sorted(point['mode'] for point in complete_fastvggt)[0]}."
            )
        if len(complete_quantvggt) > 1:
            messages.append(
                "vggt: multiple complete original + QuantVGGT rows; "
                f"using {sorted(point['mode'] for point in complete_quantvggt)[0]}."
            )
        chosen_fastvggt = sorted(
            complete_fastvggt, key=lambda point: point["mode"]
        )[0]
        chosen_recovery = sorted(complete_recovery, key=lambda point: point["mode"])[0]
        recovery_base_mode = str(chosen_recovery["mode"])
        hidden_points: List[Dict[str, Any]] = []
        for compression, state in (
            ("fastvggt", "recovery_fastvggt"),
            ("quantvggt", "recovery_quantvggt"),
        ):
            candidates = [
                _row_to_radar_point(
                    row,
                    state=state,
                    budget=budget,
                    layer_count=total_layers - budget,
                    plot=False,
                )
                for row in recovery_rows
                if _compression_for_row(row.mode) == compression
                and str(row.mode).split("+", 1)[0] == recovery_base_mode
            ]
            complete_candidates = [
                point for point in candidates if _point_values(point)
            ]
            if complete_candidates:
                hidden_points.append(
                    sorted(complete_candidates, key=lambda point: point["mode"])[0]
                )
        selected_points = [complete_upper[0], chosen_fastvggt]
        if complete_quantvggt:
            selected_points.append(
                sorted(complete_quantvggt, key=lambda point: point["mode"])[0]
            )
        selected_points.append(chosen_recovery)
        result[backbone] = [*selected_points, *hidden_points]
        messages.append(
            f"{backbone}: selected {len(selected_points)} result(s), including "
            f"Recovery at budget {budget} "
            f"({chosen_recovery['mode']}); {len(hidden_points)} compressed Ours "
            "result(s) affect bounds only."
        )
    return result, messages


def _render_final_radar_figure(
    backbone: str,
    points: Sequence[Mapping[str, Any]],
    *,
    axis_minima: Sequence[float] | None = None,
    axis_maxima: Sequence[float] | None = None,
    combined: bool = False,
) -> Any | None:
    complete = [
        (point, _point_values(point))
        for point in points
        if bool(point.get("plot", True))
        and (
            not combined
            or str(point.get("state", "")) in {"upper", "recovery"}
        )
    ]
    complete = [(point, values) for point, values in complete if values is not None]
    if not complete:
        return None

    raw_matrix = np.asarray([values for _point, values in complete], dtype=float)
    minima = (
        np.asarray(axis_minima, dtype=float)
        if axis_minima is not None
        else np.nanmin(raw_matrix, axis=0)
    )
    maxima = (
        np.asarray(axis_maxima, dtype=float)
        if axis_maxima is not None
        else np.nanmax(raw_matrix, axis=0)
    )
    if minima.shape != (raw_matrix.shape[1],) or maxima.shape != (
        raw_matrix.shape[1],
    ):
        raise ValueError("radar axis bounds do not match the metric count")
    scores = np.zeros_like(raw_matrix, dtype=float)
    valid_axes = maxima > 0.0
    scores[:, valid_axes] = raw_matrix[:, valid_axes] / maxima[valid_axes]

    axis_count = len(_FINAL_RADAR_FIELDS)
    step = 2.0 * np.pi / axis_count
    angles = np.linspace(0.0, 2.0 * np.pi, axis_count, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(
        figsize=(12.0, 14.5) if combined else (10.0, 12.5),
        dpi=130,
        subplot_kw={"projection": "polar"},
    )
    for _name, start, stop, color in _FINAL_RADAR_GROUPS:
        ax.bar(
            float(np.mean(angles[start:stop])),
            1.0,
            width=step * (stop - start),
            bottom=0.0,
            color=color,
            alpha=0.13,
            edgecolor="none",
            linewidth=0.0,
            align="center",
            zorder=0,
        )

    labels = {
        "full_fastvggt": "FastVGGT",
        "full_quantvggt": "QuantVGGT",
        "recovery": "Ours",
    }
    legend_handles: List[Line2D] = []
    for index, (point, _values) in enumerate(complete):
        state = str(point.get("state", ""))
        point_backbone = str(point.get("backbone", backbone))
        closed_scores = np.concatenate([scores[index], scores[index, :1]])
        color = (
            _BACKBONE_LINE_COLORS.get(point_backbone, "#777777")
            if combined
            else _SERIES_COLORS.get(state, "#777777")
        )
        linestyle = _METHOD_LINESTYLES.get(state, "-") if combined else "-"
        layer_count = int(point.get("layer_count", 0))
        label = (
            backbone_display_name(point_backbone)
            if state == "upper"
            else labels.get(state, state)
        )
        label = f"{label} ({layer_count} layers)"
        if combined:
            if state != "upper":
                label = f"{backbone_display_name(point_backbone)} · {label}"
        ax.plot(
            closed_angles,
            closed_scores,
            color=color,
            linewidth=5.2,
            marker="o",
            markersize=8.0,
            linestyle=linestyle,
            alpha=0.94,
            label=label,
            zorder=3,
        )
        if combined and state == "recovery":
            ax.fill(
                closed_angles,
                closed_scores,
                facecolor=color,
                edgecolor="none",
                alpha=0.12,
                zorder=1,
            )
        legend_handles.append(
            Line2D(
                [], [], color=color, linewidth=5.2, marker="o", markersize=7.5,
                linestyle=linestyle, label=label,
            )
        )

    group_by_axis = {
        index: color
        for _name, start, stop, color in _FINAL_RADAR_GROUPS
        for index in range(start, stop)
    }
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    _gridlines, theta_labels = ax.set_thetagrids(
        np.degrees(angles),
        [label for _field, label in _FINAL_RADAR_FIELDS],
        fontsize=_AXIS_LABEL_FONTSIZE,
    )
    for index, label in enumerate(theta_labels):
        label.set_fontweight("bold")
        label.set_color(group_by_axis[index])
        label.set_linespacing(1.05)
        if 1 <= index <= 4:
            label.set_horizontalalignment("left")
        elif 6 <= index <= 9:
            label.set_horizontalalignment("right")
    ax.tick_params(axis="x", pad=_AXIS_LABEL_PAD)
    ax.xaxis.majorTicks[0].set_pad(_TOP_AXIS_LABEL_PAD)
    ax.xaxis.majorTicks[5].set_pad(_BOTTOM_AXIS_LABEL_PAD)
    # Push the labels on both sides farther away from the radar.  These include
    # RPE on the right and latency/memory on the left, which otherwise overlap
    # the enlarged plot annotations.
    for index in (2, 3, 7, 8):
        ax.xaxis.majorTicks[index].set_pad(_SIDE_AXIS_LABEL_PAD)
    # Apply explicit per-label translations only after configuring tick pads;
    # changing a tick's pad rebuilds its transform and would otherwise discard
    # these offsets.
    for index, label in enumerate(theta_labels):
        offset_x, offset_y = _AXIS_LABEL_OFFSETS_POINTS[index]
        label.set_transform(
            label.get_transform()
            + ScaledTranslation(
                offset_x / 72.0,
                offset_y / 72.0,
                fig.dpi_scale_trans,
            )
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(_RADAR_LEVELS)
    ax.set_yticklabels([""] * len(_RADAR_LEVELS))
    for axis_index, ((field, _label), angle) in enumerate(
        zip(_FINAL_RADAR_FIELDS, angles)
    ):
        axis_maximum = float(maxima[axis_index])
        for radius in _RADAR_LEVELS:
            label_radius = (
                _OUTER_LEVEL_LABEL_RADIUS if radius == _RADAR_LEVELS[-1] else radius
            )
            ax.text(
                float(angle),
                float(label_radius),
                _format_radar_level_value(field, radius * axis_maximum),
                ha="center",
                va="center",
                fontsize=_LEVEL_LABEL_FONTSIZE,
                fontweight="semibold",
                color=group_by_axis[axis_index],
                alpha=0.82,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.66,
                },
                zorder=4,
            )
    ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.48)
    ax.spines["polar"].set_color("#b8b8b8")
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_alpha(0.48)
    ax.set_title(
        "All Backbones" if combined else backbone_display_name(backbone),
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.35,
        pad=0,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98 if combined else 0.985),
        ncol=1,
        prop={"size": _METHOD_LEGEND_FONTSIZE[combined], "weight": "bold"},
        frameon=True,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor=color, alpha=0.32, label=name)
            for name, _start, _stop, color in _FINAL_RADAR_GROUPS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=1,
        fontsize=_GROUP_LEGEND_FONTSIZE,
        frameon=False,
    )
    # A fixed square plotting area keeps the doubled outer labels from making
    # tight_layout collapse the radar itself.  The surrounding figure space is
    # reserved for the outward-facing axis labels and stacked legends.
    ax.set_position(
        (0.28, 0.27, 0.44, 0.3641)
        if combined
        else (0.28, 0.29, 0.44, 0.352)
    )
    return fig


def _shared_radar_bounds(
    by_backbone: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Return per-axis bounds shared by every displayed backbone and method."""
    rows = [
        values
        for backbone, _budget in FINAL_RADAR_BUDGETS
        for point in by_backbone.get(backbone, ())
        for values in [_point_values(point)]
        if values is not None
    ]
    if not rows:
        return None
    matrix = np.asarray(rows, dtype=float)
    return np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0)


def render_final_radars(
    output_root: Path,
) -> Tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    by_backbone, messages = _collect_final_radar_points(Path(output_root))
    shared_bounds = _shared_radar_bounds(by_backbone)
    images: List[np.ndarray | None] = []
    for backbone, _budget in FINAL_RADAR_BUDGETS:
        minima, maxima = shared_bounds if shared_bounds is not None else (None, None)
        images.append(
            _figure_image(
                _render_final_radar_figure(
                    backbone,
                    by_backbone.get(backbone, ()),
                    axis_minima=minima,
                    axis_maxima=maxima,
                )
            )
        )
    all_points = [
        point
        for backbone, _budget in FINAL_RADAR_BUDGETS
        for point in by_backbone.get(backbone, ())
    ]
    minima, maxima = shared_bounds if shared_bounds is not None else (None, None)
    images.append(
        _figure_image(
            _render_final_radar_figure(
                "all",
                all_points,
                axis_minima=minima,
                axis_maxima=maxima,
                combined=True,
            )
        )
    )
    messages.append(
        f"Rendered {sum(image is not None for image in images)}/"
        f"{len(FINAL_RADAR_BUDGETS) + 1} final radars from Final Table rows."
    )
    return images[0], images[1], images[2], images[3], "\n".join(messages)


def build_final_radar_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
) -> Dict[str, Any]:
    del blocks

    def redraw():
        return render_final_radars(Path(output_root))

    initial = render_final_radars(Path(output_root))
    with gr.Tab("Final Radar", id="final_radar") as tab:
        gr.Markdown(
            "## Final Radar\n"
            "Each backbone compares its original model, original + FastVGGT, and the "
            "uncompressed Recovery model at the budget selected in `main.sh`; VGGT "
            "also includes original + QuantVGGT. QuantVGGT latency and VRAM are omitted "
            "because no real INT4 deployment is available; those two simulated values "
            "also do not affect the shared axis bounds. Ours + FastVGGT and available "
            "Ours + QuantVGGT values affect the shared bounds but are not drawn. The "
            "fourth plot overlays only each backbone's original model and Ours, using "
            "one color per backbone and translucent fill for Ours only. Values "
            "come from the same live records and four-dataset average as Final Table. "
            "Every metric is lower-is-better. On every axis, zero maps to the center "
            "and the global worst across all range-setting results maps to the outer "
            "ring. The shared scales make curve size directly comparable across all "
            "four plots."
        )
        refresh_btn = gr.Button("Refresh latest data", variant="primary")
        status = gr.Textbox(label="Status", value=initial[4], interactive=False, lines=5)
        with gr.Row():
            images = [
                gr.Image(
                    value=initial[index],
                    label=f"{backbone_display_name(backbone)} · budget {budget}",
                    type="numpy",
                    interactive=False,
                    show_download_button=True,
                    elem_classes=["final-radar-image"],
                )
                for index, (backbone, budget) in enumerate(FINAL_RADAR_BUDGETS)
            ]
        with gr.Row():
            images.append(
                gr.Image(
                    value=initial[3],
                    label="All backbones · shared scale",
                    type="numpy",
                    interactive=False,
                    show_download_button=True,
                    elem_classes=["final-radar-image"],
                )
            )
        refresh_btn.click(
            redraw,
            inputs=None,
            outputs=[*images, status],
            show_progress="minimal",
        )
        tab.select(
            redraw,
            inputs=None,
            outputs=[*images, status],
            show_progress="hidden",
        )
    return {"tab": tab, "images": images, "status": status, "refresh_btn": refresh_btn}

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    METRIC_HEADERS,
    RECOVERY_METRIC_GRID,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    metric_value,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
)
from analysis.vis.recovery_overview_tab import (
    _BASELINE_STRATEGIES,
    _DATASET_AVERAGE_GROUPS,
    _dataset_group_average,
    _figure_to_rgb,
    _pruning_only_recovery_like_payload,
    discover_recovery_overview_datasets,
    discover_recovery_overview_options,
)
from analysis.vis.dataset_average_warning import format_incomplete_average_warning


_BACKBONE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
_PRUNE_STRATEGY_FAMILIES = ("gt", "ours")
_PRUNE_STRUCTURES = ("single", "double")


def _is_no_recovery_strategy(strategy_key: str) -> bool:
    return str(strategy_key).strip().lower().startswith("no_recovery")


def _recovery_state_style(strategy_key: str) -> Dict[str, Any]:
    """Encode recovery state without taking over color or line style."""
    if _is_no_recovery_strategy(strategy_key):
        return {
            "linewidth": 1.6,
            "alpha": 0.55,
            "markerfacecolor": "none",
            "markeredgecolor": None,
            "markeredgewidth": 1.5,
            "markersize": 7.0,
            "zorder": 2.5,
        }
    return {
        "linewidth": 2.8,
        "alpha": 0.95,
        "markerfacecolor": None,
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
        "markersize": 7.8,
        "zorder": 4.0,
    }


def _selected_dataset(payload: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    if dataset == "mean":
        value = payload.get("dataset_average", {})
    elif dataset in _DATASET_AVERAGE_GROUPS:
        value = _dataset_group_average(payload, dataset)
    else:
        datasets = payload.get("datasets", {})
        value = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    return value if isinstance(value, dict) else {}


def _global_metric_ranges(
    payloads: Sequence[Dict[str, Any]], strategy_keys: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    """Return one min/max range per metric shared by every selected backbone."""
    values: Dict[str, List[float]] = {key: [] for key in GT_MEAN_METRICS}
    for payload in payloads:
        upper = payload.get("upper_average", {})
        if isinstance(upper, dict):
            for metric in GT_MEAN_METRICS:
                value = metric_value(upper, metric)
                if value is not None and np.isfinite(value):
                    values[metric].append(float(value))
        strategies = payload.get("strategies", {})
        if not isinstance(strategies, dict):
            continue
        for strategy_key in strategy_keys:
            strategy = strategies.get(strategy_key, {})
            points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
            for point in points.values() if isinstance(points, dict) else ():
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                raw = point.get("raw_metric", point.get("metrics_mean", {}))
                if not isinstance(raw, dict):
                    continue
                for metric in GT_MEAN_METRICS:
                    value = metric_value(raw, metric)
                    if value is not None and np.isfinite(value):
                        values[metric].append(float(value))
    return {
        metric: (min(items), max(items))
        for metric, items in values.items()
        if items
    }


def _normalized_mean(raw: Dict[str, Any], ranges: Dict[str, Tuple[float, float]]) -> float | None:
    normalized: List[float] = []
    for metric in GT_MEAN_METRICS:
        value = metric_value(raw, metric)
        limits = ranges.get(metric)
        if value is None or limits is None:
            continue
        low, high = limits
        normalized.append(0.0 if high <= low else (float(value) - low) / (high - low))
    return float(np.mean(normalized)) if normalized else None


def _render_combined_figure(
    payloads: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    method_keys: Sequence[str],
    title: str,
    show_no_recovery: bool = True,
) -> Any | None:
    strategy_keys = tuple(
        dict.fromkeys(
            (*_BASELINE_STRATEGIES, *method_keys)
            if show_no_recovery
            else method_keys
        )
    )
    ranges = _global_metric_ranges([payload for _, payload in payloads], strategy_keys)
    fig, axes = plt.subplots(2, 4, figsize=(29.6, 11.6), squeeze=False)
    any_data = False
    backbone_handles: List[Line2D] = []
    present_styles: Dict[str, str] = {}

    for backbone_idx, (backbone, payload) in enumerate(payloads):
        color = _BACKBONE_COLORS[backbone_idx % len(_BACKBONE_COLORS)]
        backbone_handles.append(Line2D([], [], color=color, linewidth=2.4, label=backbone))
        strategies = payload.get("strategies", {})
        if not isinstance(strategies, dict):
            continue
        upper = payload.get("upper_average", {})
        for panel_idx, metric in enumerate(sum((tuple(row) for row in RECOVERY_METRIC_GRID), ())):
            ax = axes.flat[panel_idx]
            upper_value = (
                _normalized_mean(upper, ranges)
                if metric == "gt_mean" and isinstance(upper, dict)
                else metric_value(upper, metric) if isinstance(upper, dict) else None
            )
            for strategy_key in strategy_keys:
                strategy = strategies.get(strategy_key, {})
                if not isinstance(strategy, dict):
                    continue
                points = strategy.get("points", {})
                xs: List[int] = []
                ys: List[float] = []
                for drop_key, point in sorted(
                    points.items() if isinstance(points, dict) else (),
                    key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
                ):
                    if not str(drop_key).isdigit() or not isinstance(point, dict) or not point.get("available", False):
                        continue
                    raw = point.get("raw_metric", point.get("metrics_mean", {}))
                    if not isinstance(raw, dict):
                        continue
                    value = _normalized_mean(raw, ranges) if metric == "gt_mean" else metric_value(raw, metric)
                    if value is not None:
                        xs.append(int(drop_key))
                        ys.append(float(value))
                if not xs:
                    continue
                any_data = True
                display = str(strategy.get("display_name", strategy_key))
                present_styles.setdefault(strategy_key, display)
                linestyle = strategy_plot_linestyle(strategy_key)
                state_style = _recovery_state_style(strategy_key)
                (line,) = ax.plot(
                    xs, ys, color=color, linestyle=linestyle,
                    linewidth=state_style["linewidth"],
                    alpha=state_style["alpha"],
                    zorder=state_style["zorder"],
                )
                dashes = strategy_plot_dashes(strategy_key)
                if dashes is not None:
                    line.set_dashes(dashes)
                marker_facecolor = state_style["markerfacecolor"] or color
                marker_edgecolor = state_style["markeredgecolor"] or color
                ax.scatter(
                    xs, ys,
                    facecolors=marker_facecolor,
                    edgecolors=marker_edgecolor,
                    linewidths=state_style["markeredgewidth"],
                    marker=strategy_plot_marker(strategy_key),
                    s=state_style["markersize"] ** 2,
                    alpha=state_style["alpha"],
                    zorder=state_style["zorder"] + 0.2,
                )
            if upper_value is not None:
                ax.axhline(
                    float(upper_value),
                    color=color,
                    linestyle=":",
                    linewidth=2.2,
                    alpha=.9,
                    zorder=2,
                )

    for panel_idx, metric in enumerate(sum((tuple(row) for row in RECOVERY_METRIC_GRID), ())):
        ax = axes.flat[panel_idx]
        suffix = "globally normalized" if metric == "gt_mean" else "raw metric"
        ax.set_title(
            f"{METRIC_HEADERS.get(metric, metric)} · {suffix}",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Dropped units", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.grid(True, alpha=.25)
    if not any_data:
        plt.close(fig)
        return None
    style_handles = []
    for key, label in present_styles.items():
        state_style = _recovery_state_style(key)
        marker_facecolor = state_style["markerfacecolor"] or "#333333"
        marker_edgecolor = state_style["markeredgecolor"] or "#333333"
        handle = Line2D(
            [], [], color="#333333", linestyle=strategy_plot_linestyle(key),
            marker=strategy_plot_marker(key),
            linewidth=state_style["linewidth"], alpha=state_style["alpha"],
            markerfacecolor=marker_facecolor,
            markeredgecolor=marker_edgecolor,
            markeredgewidth=state_style["markeredgewidth"],
            markersize=state_style["markersize"], label=label,
        )
        dashes = strategy_plot_dashes(key)
        if dashes is not None:
            handle.set_dashes(dashes)
        style_handles.append(handle)
    upper_handle = Line2D(
        [], [], color="#333333", linestyle=":", linewidth=2.2, label="Upper"
    )
    fig.legend(
        handles=[*backbone_handles, *style_handles, upper_handle],
        loc="upper center",
        bbox_to_anchor=(.5, .958),
        ncol=max(1, min(6, len(backbone_handles) + 2)),
        fontsize=13,
        frameon=True,
    )
    fig.suptitle(title, fontsize=19, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return fig


def render_recovery_backbone_comparison(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "training average",
    prune_strategy: str = "ours",
    show_no_recovery: bool = True,
    prune_structures: Sequence[str] | None = None,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    backbones = list(dict.fromkeys(str(item) for item in (selected_backbones or ())))
    method = str(selected_method or "").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not method:
        return [], "Select one recovery method."
    strategy_family = str(prune_strategy or "").strip().lower()
    if strategy_family not in _PRUNE_STRATEGY_FAMILIES:
        return [], "Prune strategy must be gt or ours."
    structures = list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in (prune_structures if prune_structures is not None else _PRUNE_STRUCTURES)
            if str(item).strip().lower() in _PRUNE_STRUCTURES
        )
    )
    if not structures:
        return [], "Select at least one pruning structure (single or double)."
    variants = [f"{strategy_family}_{structure}" for structure in structures]
    method_keys = [f"{method}_{variant}" for variant in variants]
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    warnings: List[str] = []
    messages: List[str] = []
    for backbone in backbones:
        root = Path(output_root) / backbone
        try:
            pruning = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)
            merged: Dict[str, Any] = {"upper_average": {}, "strategies": {}}
            missing_variants: List[str] = []
            for variant, structure in zip(variants, structures):
                recovery = collect_recovery_eval_payload(
                    experiment_root=root,
                    prune_strategy=variant,
                )
                if dataset in _DATASET_AVERAGE_GROUPS:
                    recovery_datasets = recovery.get("datasets", {})
                    method_datasets = (
                        [
                            name
                            for name, payload in recovery_datasets.items()
                            if isinstance(payload, dict)
                            and isinstance(payload.get("strategies"), dict)
                            and method in payload["strategies"]
                        ]
                        if isinstance(recovery_datasets, dict)
                        else []
                    )
                    warning = format_incomplete_average_warning(
                        owner=f"{backbone} / {method} / {variant}",
                        scope=dataset,
                        expected=_DATASET_AVERAGE_GROUPS[dataset],
                        available=method_datasets,
                    )
                    if warning:
                        warnings.append(warning)
                recovery = add_recovery_comparison_strategies(
                    recovery, pruning, variant
                )
                payload = _selected_dataset(recovery, dataset)
                strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
                if not isinstance(strategies, dict) or method not in strategies:
                    payload = _pruning_only_recovery_like_payload(
                        pruning, dataset=dataset, prune_strategy=variant
                    )
                    strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
                    missing_variants.append(variant)
                upper = payload.get("upper_average", {}) if isinstance(payload, dict) else {}
                if not merged["upper_average"] and isinstance(upper, dict):
                    merged["upper_average"] = upper
                if not isinstance(strategies, dict):
                    continue
                for source_key, strategy_payload in strategies.items():
                    if not isinstance(strategy_payload, dict):
                        continue
                    if source_key == method:
                        target_key = f"{method}_{variant}"
                        display_name = f"{method.removeprefix('recovery_')} · {variant}"
                    elif str(source_key).startswith("no_recovery"):
                        target_key = f"no_recovery_{structure}"
                        display_name = f"No recovery · {variant}"
                    else:
                        continue
                    merged["strategies"][target_key] = {
                        **strategy_payload,
                        "display_name": display_name,
                    }
            if missing_variants:
                messages.append(
                    f"{backbone}: recovery method missing for {', '.join(missing_variants)}; "
                    "showing available pruning baselines"
                )
            loaded.append((backbone, merged))
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    fig = _render_combined_figure(
        loaded, method_keys=method_keys,
        title=(
            f"Backbone comparison: {method.removeprefix('recovery_')} · "
            f"{strategy_family} ({', '.join(structures)}) · {dataset}"
        ),
        show_no_recovery=bool(show_no_recovery),
    )
    if fig is None:
        return [], "\n".join([*warnings, *messages, "No matching data to plot."])
    try:
        image = _figure_to_rgb(fig)
    finally:
        plt.close(fig)
    messages.append(
        f"Rendered {len(loaded)} backbone(s) in one 2×4 figure; first 7 metrics are raw and the mean uses shared global ranges."
    )
    return [(image, "Backbone comparison")], "\n".join([*warnings, *messages])


def build_recovery_backbone_comparison_tab(*, blocks: gr.Blocks, output_root: Path) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = discover_recovery_overview_datasets(Path(output_root))
    default_method = (
        "recovery_ghost"
        if "recovery_ghost" in methods
        else methods[0] if methods else None
    )

    def redraw(
        selected_backbones, selected_method, dataset, prune_strategy,
        prune_structures, show_no_recovery,
    ):
        images, status_value = render_recovery_backbone_comparison(
            Path(output_root), selected_backbones, selected_method, dataset,
            prune_strategy, show_no_recovery, prune_structures,
        )
        return (images[0][0] if images else None), status_value

    with gr.Tab("Recovery Backbone Comparison", id="recovery_backbone_comparison") as tab:
        gr.Markdown("## Recovery Backbone Comparison")
        gr.Markdown(
            "All selected backbones share one 2×4 figure and each backbone has its own color. "
            "The first seven panels show raw metrics. The 7-metric mean normalizes every metric "
            "with the range shared by all selected backbones before averaging."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(label="Backbones", choices=backbones, value=backbones)
                prune_strategy_dd = gr.Dropdown(
                    label="Prune strategy", choices=list(_PRUNE_STRATEGY_FAMILIES),
                    value="ours", multiselect=False,
                )
                prune_structure_cb = gr.CheckboxGroup(
                    label="Pruning structure",
                    choices=list(_PRUNE_STRUCTURES),
                    value=list(_PRUNE_STRUCTURES),
                )
                method_dd = gr.Dropdown(
                    label="Recovery method",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=default_method, multiselect=False,
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset", choices=datasets, value="training average", multiselect=False
                )
                show_no_recovery_cb = gr.Checkbox(
                    label="Show No recovery",
                    value=True,
                )
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=4)
            with gr.Column(scale=5):
                image = gr.Image(
                    label="Combined recovery comparison",
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "recovery-backbone-comparison-image"],
                )

        inputs = [
            backbone_cb, method_dd, dataset_dd, prune_strategy_dd,
            prune_structure_cb, show_no_recovery_cb,
        ]
        for event in (
            backbone_cb.change, method_dd.change, dataset_dd.change,
            prune_strategy_dd.change, prune_structure_cb.change,
            show_no_recovery_cb.change,
        ):
            event(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        refresh_btn.click(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        blocks.load(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal")

    return {
        "tab": tab, "backbone_cb": backbone_cb, "method_dd": method_dd,
        "prune_strategy_dd": prune_strategy_dd,
        "prune_structure_cb": prune_structure_cb, "dataset_dd": dataset_dd,
        "show_no_recovery_cb": show_no_recovery_cb,
        "image": image, "status": status,
    }

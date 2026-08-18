from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from analysis.vis.display_names import backbone_display_name
from analysis.vis.composite import Cell, Section, compose_sections, to_image

from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    METRIC_HEADERS,
    PUBLICATION_METRIC_SUBPLOT_HEIGHT,
    PUBLICATION_METRIC_SUBPLOT_WIDTH,
    RECOVERY_METRIC_GRID,
    STRATEGY_PLOT_COLORS,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    metric_value,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
)
from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.recovery.shared_results import (
    PRUNE_FAMILY_STRATEGIES,
    best_recovery_result,
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
from analysis.vis.dataset_average_warning import (
    dataset_display_choices,
    format_incomplete_average_warning,
)

# Recovery Remaining typography and outer-label spacing (pixels).
RECOVERY_REMAINING_SUBPLOT_FONT_SIZE = 18
RECOVERY_REMAINING_OUTER_FONT_SIZE = 26
RECOVERY_REMAINING_LEFT_LABEL_TO_IMAGE_GAP = 0
RECOVERY_REMAINING_TOP_LABEL_TO_IMAGE_GAP = 0
RECOVERY_REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP = 0


_PRUNING_OVERVIEW_COLOR_KEYS = (
    "ours",
    "gt",
    "replaceme",
    "inv_mse",
    "cosine",
    "cca",
    "shortgpt",
    "simpler",
)
_BACKBONE_COLORS = tuple(
    STRATEGY_PLOT_COLORS[key] for key in _PRUNING_OVERVIEW_COLOR_KEYS
)
_PRUNE_STRATEGY_FAMILIES = ("gt", "ours")
_MIN_PRUNE_STRATEGY = "min"
_PRUNE_STRUCTURES = ("single", "double")
_TESTTIME_MARKERS = ("X", "D", "P", "*", "h", "v", "<", ">")
_RECOVERY_METHOD_LEGEND_LABELS = {
    "recovery_ghost_interval_sequential": "Ours",
}
_COMPRESSION_LEGEND_LABELS = {
    "fastvggt_m0_r0p9": "FastVGGT",
}


def _compose_recovery_remaining(value: Any) -> np.ndarray:
    image = to_image(value)
    if image is None:
        raise ValueError("No recovery-remaining image was returned.")
    return compose_sections(
        [Section(columns=1, cells=[Cell(image=image)])],
        outer_font_size=RECOVERY_REMAINING_OUTER_FONT_SIZE,
        left_label_to_image_gap=RECOVERY_REMAINING_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=RECOVERY_REMAINING_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=RECOVERY_REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def _recovery_method_legend_label(method: str) -> str:
    return _RECOVERY_METHOD_LEGEND_LABELS.get(
        str(method), str(method).removeprefix("recovery_")
    )


def _structure_legend_label(structure: str) -> str:
    return "Single" if str(structure) == "single" else "Multi"


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


def _strategy_selectors_by_drop(root: Path, strategy: str) -> Dict[int, str]:
    bundle = load_prune_plan_bundle(Path(root))
    strategy_record = bundle_strategy_records(bundle).get(str(strategy), {})
    points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
    plans = bundle_plan_records(bundle)
    selected: Dict[int, str] = {}
    for drop_key, plan_name in points.items() if isinstance(points, dict) else ():
        if not str(drop_key).isdigit():
            continue
        record = plans.get(str(plan_name))
        if not isinstance(record, dict):
            continue
        selected[int(drop_key)] = physical_pruning_selector_from_record(record)
    return selected


def _load_testtime_points(
    root: Path,
    *,
    recovery_method: str,
    prune_strategy: str,
    dataset: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load test-time summaries matching one recovery and pruning configuration."""
    root = Path(root)
    method_roots = {
        path.parent.parent.parent.name: path
        for path in root.glob(f"*/eval/testtime/{recovery_method}")
        if path.is_dir()
    }
    if not method_roots:
        return [], []
    selectors_by_drop = _strategy_selectors_by_drop(root, prune_strategy)
    if not selectors_by_drop:
        return [], []
    if dataset == "mean":
        expected_datasets = tuple(sorted(method_roots))
    elif dataset in _DATASET_AVERAGE_GROUPS:
        expected_datasets = tuple(_DATASET_AVERAGE_GROUPS[dataset])
    else:
        expected_datasets = (dataset,)

    aggregated: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for dataset_name in expected_datasets:
        method_root = method_roots.get(dataset_name)
        if method_root is None:
            continue
        for compression_root in sorted(path for path in method_root.iterdir() if path.is_dir()):
            compression = compression_root.name
            for drop_count, selector in selectors_by_drop.items():
                summary_path = compression_root / selector / "summary.json"
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(summary, dict):
                    continue
                if str(summary.get("selector", "")) != selector:
                    continue
                if str(summary.get("recovery_plan_name", "")) != recovery_method:
                    continue
                try:
                    recorded_drop = int(summary.get("drop_count"))
                except (TypeError, ValueError):
                    continue
                if recorded_drop != drop_count:
                    continue
                metrics = summary.get("metrics_mean", {})
                if not isinstance(metrics, dict):
                    continue
                key = (compression, drop_count, selector)
                item = aggregated.setdefault(
                    key,
                    {
                        "compression": compression,
                        "display_name": compression,
                        "drop_count": drop_count,
                        "selector": selector,
                        "metric_values": {},
                        "datasets": [],
                    },
                )
                item["datasets"].append(dataset_name)
                for metric_key, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        item["metric_values"].setdefault(str(metric_key), []).append(float(value))

    points: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for item in sorted(
        aggregated.values(),
        key=lambda value: (str(value["compression"]), int(value["drop_count"])),
    ):
        raw_metric = {
            key: float(np.mean(values))
            for key, values in item.pop("metric_values").items()
            if values
        }
        if not raw_metric:
            continue
        points.append({**item, "raw_metric": raw_metric})
        if dataset in _DATASET_AVERAGE_GROUPS:
            warning = format_incomplete_average_warning(
                owner=(
                    f"testtime / {item['compression']} / {recovery_method} / "
                    f"{prune_strategy} / drop {item['drop_count']}"
                ),
                scope=dataset,
                expected=expected_datasets,
                available=item["datasets"],
            )
            if warning:
                warnings.append(warning)
    return points, warnings


def _filter_testtime_to_family_winners(
    payload: Dict[str, Any],
    *,
    root: Path,
    recovery_method: str,
    families: Sequence[str],
    selection_datasets: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only selectors selected by the test-time scheduling policy."""
    points = payload.get("testtime_points", ())
    if not isinstance(points, list) or not points:
        return payload, []
    drop_counts = sorted({
        int(point["drop_count"])
        for point in points
        if isinstance(point, dict) and isinstance(point.get("drop_count"), int)
    })
    winners: Dict[int, Dict[str, str]] = {}
    failures: List[str] = []
    for drop_count in drop_counts:
        for family in families:
            strategies = PRUNE_FAMILY_STRATEGIES.get(str(family))
            if not strategies:
                continue
            try:
                winner = best_recovery_result(
                    Path(root),
                    drop_count=drop_count,
                    datasets=selection_datasets,
                    recovery_plan_name=recovery_method,
                    strategies=strategies,
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                failures.append(f"{family} drop {drop_count}: {exc}")
                continue
            winners.setdefault(drop_count, {})[str(family)] = str(winner["selector"])

    filtered: List[Dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("drop_count"), int):
            continue
        drop_count = int(point["drop_count"])
        selector = str(point.get("selector", ""))
        matched_families = [
            family
            for family, winner_selector in winners.get(drop_count, {}).items()
            if selector == winner_selector
        ]
        if matched_families:
            filtered.append({
                **point,
                "testtime_winner_families": matched_families,
            })
    return {**payload, "testtime_points": filtered}, failures


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
        testtime_points = payload.get("testtime_points", ())
        for point in testtime_points if isinstance(testtime_points, list) else ():
            raw = point.get("raw_metric", {}) if isinstance(point, dict) else {}
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


def _strict_normalized_mean(
    raw: Dict[str, Any], ranges: Dict[str, Tuple[float, float]]
) -> float | None:
    """Score a candidate only when all seven metrics are present."""
    if any(metric_value(raw, metric) is None or metric not in ranges for metric in GT_MEAN_METRICS):
        return None
    return _normalized_mean(raw, ranges)


def _collapse_min_prune_strategies(
    payload: Dict[str, Any], *, method: str, structures: Sequence[str],
    ranges: Dict[str, Tuple[float, float]] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Pairwise-select GT/Ours points without mixing metrics across candidates."""
    strategies = payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return payload, []
    ranges = ranges or _global_metric_ranges([payload], list(strategies))
    selected: Dict[str, Any] = {}
    method_keys: List[str] = []
    for structure in structures:
        target_method = f"{method}_min_{structure}"
        method_keys.append(target_method)
        winners: Dict[str, Any] = {}
        winner_variants: Dict[str, str] = {}
        drop_keys = {
            str(drop_key)
            for family in _PRUNE_STRATEGY_FAMILIES
            for drop_key in strategies.get(
                f"{method}_{family}_{structure}", {}
            ).get("points", {})
            if isinstance(strategies.get(f"{method}_{family}_{structure}"), dict)
        }
        for drop_key in drop_keys:
            choices: List[Tuple[float, str, Dict[str, Any]]] = []
            for family in _PRUNE_STRATEGY_FAMILIES:
                variant = f"{family}_{structure}"
                source = strategies.get(f"{method}_{variant}", {})
                point = source.get("points", {}).get(drop_key) if isinstance(source, dict) else None
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                raw = point.get("raw_metric", point.get("metrics_mean", {}))
                score = _strict_normalized_mean(raw, ranges) if isinstance(raw, dict) else None
                if score is not None:
                    choices.append((score, variant, point))
            if choices:
                _, variant, point = min(choices, key=lambda item: (item[0], item[1]))
                winners[drop_key] = {**deepcopy(point), "min_prune_strategy": variant}
                winner_variants[drop_key] = variant
        if winners:
            selected[target_method] = {
                "display_name": (
                    f"{_recovery_method_legend_label(method)} · "
                    f"{_structure_legend_label(structure)}"
                ),
                "points": winners,
            }

        baseline_points: Dict[str, Any] = {}
        for drop_key, variant in winner_variants.items():
            baseline = strategies.get(f"no_recovery_{variant}", {})
            point = baseline.get("points", {}).get(drop_key) if isinstance(baseline, dict) else None
            if isinstance(point, dict) and point.get("available", False):
                baseline_points[drop_key] = {
                    **deepcopy(point), "min_prune_strategy": variant,
                }
        baseline_drop_keys = {
            str(drop_key)
            for family in _PRUNE_STRATEGY_FAMILIES
            for drop_key in strategies.get(
                f"no_recovery_{family}_{structure}", {}
            ).get("points", {})
            if isinstance(
                strategies.get(f"no_recovery_{family}_{structure}"), dict
            )
        }
        for drop_key in baseline_drop_keys.difference(winner_variants):
            choices: List[Tuple[float, str, Dict[str, Any]]] = []
            for family in _PRUNE_STRATEGY_FAMILIES:
                variant = f"{family}_{structure}"
                baseline = strategies.get(f"no_recovery_{variant}", {})
                point = (
                    baseline.get("points", {}).get(drop_key)
                    if isinstance(baseline, dict) else None
                )
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                raw = point.get("raw_metric", point.get("metrics_mean", {}))
                score = (
                    _strict_normalized_mean(raw, ranges)
                    if isinstance(raw, dict) else None
                )
                if score is not None:
                    choices.append((score, variant, point))
            if choices:
                _, variant, point = min(
                    choices, key=lambda item: (item[0], item[1])
                )
                baseline_points[drop_key] = {
                    **deepcopy(point),
                    "min_prune_strategy": variant,
                    "no_recovery_fallback": True,
                }
        if baseline_points:
            selected[f"no_recovery_{structure}"] = {
                "display_name": f"No Recovery · {_structure_legend_label(structure)}",
                "points": baseline_points,
            }

    testtime_winners: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for point in payload.get("testtime_points", ()):
        if not isinstance(point, dict):
            continue
        compression = str(point.get("compression", ""))
        drop_count = point.get("drop_count")
        if compression and isinstance(drop_count, int):
            grouped.setdefault((compression, drop_count), []).append(point)
    for candidates in grouped.values():
        choices = []
        for point in candidates:
            raw = point.get("raw_metric", {})
            score = _strict_normalized_mean(raw, ranges) if isinstance(raw, dict) else None
            if score is not None:
                choices.append((score, str(point.get("prune_strategy", "")), point))
        if choices:
            _, _, winner = min(choices, key=lambda item: (item[0], item[1]))
            testtime_winners.append(deepcopy(winner))
    return {
        **payload,
        "strategies": selected,
        "testtime_points": testtime_winners,
    }, method_keys


def _render_combined_figure(
    payloads: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    method_keys: Sequence[str],
    show_no_recovery: bool = True,
    total_units_by_backbone: Dict[str, int] | None = None,
    halve_x_axis: bool = False,
    metric_ranges: Dict[str, Tuple[float, float]] | None = None,
) -> Any | None:
    axis_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    tick_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    legend_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    strategy_keys = tuple(
        dict.fromkeys(
            (*_BASELINE_STRATEGIES, *method_keys)
            if show_no_recovery
            else method_keys
        )
    )
    ranges = metric_ranges or _global_metric_ranges(
        [payload for _, payload in payloads], strategy_keys
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(
            PUBLICATION_METRIC_SUBPLOT_WIDTH * 4,
            PUBLICATION_METRIC_SUBPLOT_HEIGHT * 2,
        ),
        squeeze=False,
    )
    any_data = False
    backbone_handles: List[Line2D] = []
    present_styles: Dict[str, str] = {}
    compression_names = sorted(
        {
            str(point.get("compression"))
            for _, payload in payloads
            for point in payload.get("testtime_points", ())
            if isinstance(point, dict) and point.get("compression")
        }
    )
    compression_markers = {
        name: _TESTTIME_MARKERS[index % len(_TESTTIME_MARKERS)]
        for index, name in enumerate(compression_names)
    }
    present_compressions: Dict[str, str] = {}

    for backbone_idx, (backbone, payload) in enumerate(payloads):
        color = _BACKBONE_COLORS[backbone_idx % len(_BACKBONE_COLORS)]
        backbone_handles.append(
            Line2D(
                [], [], color=color, linewidth=2.4,
                label=backbone_display_name(backbone),
            )
        )
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
                xs: List[float] = []
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
                        drop_count = int(drop_key)
                        total_units = (total_units_by_backbone or {}).get(backbone)
                        x_value = total_units - drop_count if total_units is not None else drop_count
                        xs.append(float(x_value) / 2.0 if halve_x_axis else float(x_value))
                        ys.append(float(value))
                if not xs:
                    continue
                if total_units_by_backbone is not None:
                    ordered = sorted(zip(xs, ys))
                    xs, ys = [item[0] for item in ordered], [item[1] for item in ordered]
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
            testtime_points = payload.get("testtime_points", ())
            for point in testtime_points if isinstance(testtime_points, list) else ():
                if not isinstance(point, dict):
                    continue
                raw = point.get("raw_metric", {})
                compression = str(point.get("compression", ""))
                drop_count = point.get("drop_count")
                if not isinstance(raw, dict) or not compression or not isinstance(drop_count, int):
                    continue
                value = _normalized_mean(raw, ranges) if metric == "gt_mean" else metric_value(raw, metric)
                if value is None:
                    continue
                total_units = (total_units_by_backbone or {}).get(backbone)
                x_value = total_units - drop_count if total_units is not None else drop_count
                x_value = float(x_value) / 2.0 if halve_x_axis else float(x_value)
                any_data = True
                present_compressions.setdefault(
                    compression, str(point.get("display_name", compression))
                )
                ax.scatter(
                    [x_value], [float(value)],
                    facecolors=color,
                    edgecolors="black",
                    linewidths=1.1,
                    marker=compression_markers[compression],
                    s=112,
                    alpha=.98,
                    zorder=7.0,
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
        ax.set_ylabel(
            METRIC_HEADERS.get(metric, metric),
            fontsize=axis_font_size,
            fontweight="bold",
        )
        x_label = "Remaining layers" if total_units_by_backbone is not None else "Dropped units"
        ax.tick_params(
            axis="both", labelsize=tick_font_size
        )
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
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
    compression_handles = [
        Line2D(
            [], [], color="#333333", linestyle="None",
            marker=compression_markers[key], markersize=9,
            markerfacecolor="#999999", markeredgecolor="black",
            label=_COMPRESSION_LEGEND_LABELS.get(key, label),
        )
        for key, label in present_compressions.items()
    ]
    legend = fig.legend(
        handles=[*backbone_handles, *style_handles, *compression_handles, upper_handle],
        loc="upper center",
        bbox_to_anchor=(.5, .958),
        ncol=max(
            1,
            len(backbone_handles) + len(style_handles)
            + len(compression_handles) + 1,
        ),
        fontsize=legend_font_size,
        markerscale=1.35,
        handlelength=2.8,
        handleheight=1.2,
        frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    shared_x_label = f"{x_label} ÷ 2" if halve_x_axis else x_label
    fig.supxlabel(
        shared_x_label,
        fontsize=axis_font_size,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, .035, 1, .91))
    return fig


def render_recovery_backbone_remaining_comparison(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "training average",
    prune_strategy: str = "ours",
    show_no_recovery: bool = True,
    prune_structures: Sequence[str] | None = None,
    *,
    halve_x_axis: bool = False,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Render the backbone comparison against units left after pruning."""
    backbones = list(dict.fromkeys(str(item) for item in (selected_backbones or ())))
    method = str(selected_method or "").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not method:
        return [], "Select one recovery method."
    strategy_family = str(prune_strategy or "").strip().lower()
    if strategy_family not in (*_PRUNE_STRATEGY_FAMILIES, _MIN_PRUNE_STRATEGY):
        return [], "Prune strategy must be gt, ours, or min."
    structures = list(
        dict.fromkeys(
            str(item).strip().lower()
            for item in (prune_structures if prune_structures is not None else _PRUNE_STRUCTURES)
            if str(item).strip().lower() in _PRUNE_STRUCTURES
        )
    )
    if not structures:
        return [], "Select at least one pruning structure (single or double)."
    variants = (
        [
            f"{family}_{structure}"
            for structure in structures
            for family in _PRUNE_STRATEGY_FAMILIES
        ]
        if strategy_family == _MIN_PRUNE_STRATEGY
        else [f"{strategy_family}_{structure}" for structure in structures]
    )
    method_keys = [f"{method}_{variant}" for variant in variants]
    selection_datasets = tuple(_DATASET_AVERAGE_GROUPS["training average"])
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    total_units_by_backbone: Dict[str, int] = {}
    warnings: List[str] = []
    messages: List[str] = []
    min_metric_ranges: Dict[str, Tuple[float, float]] | None = None
    for backbone in backbones:
        root = Path(output_root) / backbone
        try:
            total_units = int(load_prune_plan_bundle(root).get("num_units", 0))
            if total_units <= 0:
                raise ValueError("prune plan bundle has no positive num_units")
            total_units_by_backbone[backbone] = total_units
            pruning = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)
            merged: Dict[str, Any] = {
                "upper_average": {}, "strategies": {}, "testtime_points": []
            }
            seen_testtime: set[Tuple[str, int, str]] = set()
            missing_variants: List[str] = []
            for variant in variants:
                structure = variant.rsplit("_", 1)[-1]
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
                        display_name = (
                            f"{_recovery_method_legend_label(method)} · "
                            f"{_structure_legend_label(structure)}"
                        )
                    elif str(source_key).startswith("no_recovery"):
                        target_key = (
                            f"no_recovery_{variant}"
                            if strategy_family == _MIN_PRUNE_STRATEGY
                            else f"no_recovery_{structure}"
                        )
                        display_name = (
                            f"No Recovery · {_structure_legend_label(structure)}"
                        )
                    else:
                        continue
                    merged["strategies"][target_key] = {
                        **strategy_payload,
                        "display_name": display_name,
                    }
                testtime_points, testtime_warnings = _load_testtime_points(
                    root,
                    recovery_method=method,
                    prune_strategy=variant,
                    dataset=dataset,
                )
                warnings.extend(f"{backbone}: {warning}" for warning in testtime_warnings)
                for point in testtime_points:
                    point = {**point, "prune_strategy": variant}
                    identity = (
                        str(point.get("compression", "")),
                        int(point.get("drop_count", -1)),
                        str(point.get("selector", "")),
                    )
                    if identity in seen_testtime:
                        continue
                    seen_testtime.add(identity)
                    merged["testtime_points"].append(point)
            if missing_variants:
                messages.append(
                    f"{backbone}: recovery method missing for {', '.join(missing_variants)}; "
                    "showing available pruning baselines"
                )
            winner_families = (
                _PRUNE_STRATEGY_FAMILIES
                if strategy_family == _MIN_PRUNE_STRATEGY
                else (strategy_family,)
            )
            merged, winner_failures = _filter_testtime_to_family_winners(
                merged,
                root=root,
                recovery_method=method,
                families=winner_families,
                selection_datasets=selection_datasets,
            )
            if winner_failures:
                messages.append(
                    f"{backbone}: test-time family selection skipped "
                    + "; ".join(winner_failures)
                )
            loaded.append((backbone, merged))
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    if strategy_family == _MIN_PRUNE_STRATEGY:
        candidate_keys = [
            str(key)
            for _, payload in loaded
            for key in payload.get("strategies", {})
            if isinstance(payload.get("strategies", {}), dict)
        ]
        shared_ranges = _global_metric_ranges(
            [payload for _, payload in loaded], candidate_keys,
        )
        min_metric_ranges = shared_ranges
        collapsed: List[Tuple[str, Dict[str, Any]]] = []
        for backbone, payload in loaded:
            selected_payload, method_keys = _collapse_min_prune_strategies(
                payload, method=method, structures=structures, ranges=shared_ranges,
            )
            collapsed.append((backbone, selected_payload))
        loaded = collapsed
    fig = _render_combined_figure(
        loaded, method_keys=method_keys,
        show_no_recovery=bool(show_no_recovery),
        total_units_by_backbone=total_units_by_backbone,
        halve_x_axis=bool(halve_x_axis),
        metric_ranges=min_metric_ranges,
    )
    if fig is None:
        return [], "\n".join([*warnings, *messages, "No matching data to plot."])
    try:
        image = _figure_to_rgb(fig)
    finally:
        plt.close(fig)
    messages.append(
        f"Rendered {len(loaded)} backbone(s) in one 2×4 figure with "
        f"{sum(len(payload.get('testtime_points', ())) for _, payload in loaded)} matching "
        "test-time point(s); first 7 metrics are raw and the mean uses shared global ranges."
    )
    return [(image, "Backbone comparison by remaining units")], "\n".join(
        [*warnings, *messages]
    )


def build_recovery_backbone_remaining_comparison_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = discover_recovery_overview_datasets(Path(output_root))
    default_method = (
        "recovery_ghost_interval_sequential"
        if "recovery_ghost_interval_sequential" in methods
        else methods[0] if methods else None
    )

    def redraw(
        selected_backbones, selected_method, dataset, prune_strategy,
        prune_structures, show_no_recovery, halve_x_axis,
    ):
        images, status_value = render_recovery_backbone_remaining_comparison(
            Path(output_root), selected_backbones, selected_method, dataset,
            prune_strategy, show_no_recovery, prune_structures,
            halve_x_axis=halve_x_axis,
        )
        return (
            _compose_recovery_remaining(images[0][0]) if images else None,
            status_value,
        )

    tab_label = "Recovery Remaining"
    tab_id = "recovery_backbone_remaining_comparison"
    with gr.Tab(tab_label, id=tab_id) as tab:
        gr.Markdown(f"## {tab_label}")
        gr.Markdown(
            "All selected backbones share one 2×4 figure and each backbone has its own color. "
            "The first seven panels show raw metrics. The 7-metric mean normalizes every metric "
            "with the range shared by all selected backbones before averaging. Matching test-time "
            "compression results are overlaid as isolated points, with marker shape identifying "
            "the compression configuration. Min compares GT and Ours separately for single and "
            "double pruning at every drop count, using the 7-metric mean to keep each winning "
            "point's metrics together. No recovery follows that winner where recovery exists, "
            "and otherwise uses its own 7-metric-mean winner."
            + " The x-axis is the number of remaining units, inferred from each backbone's prune plan."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                prune_strategy_dd = gr.Dropdown(
                    label="Prune strategy",
                    choices=[*_PRUNE_STRATEGY_FAMILIES, _MIN_PRUNE_STRATEGY],
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
                    label="Dataset",
                    choices=dataset_display_choices(datasets),
                    value="training average",
                    multiselect=False,
                )
                show_no_recovery_cb = gr.Checkbox(
                    label="Show No recovery",
                    value=True,
                )
                halve_x_axis_cb = gr.Checkbox(
                    label="Divide x-axis by 2",
                    value=False,
                    info="Available points use even x values, so the scaled axis remains integral.",
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
            prune_structure_cb, show_no_recovery_cb, halve_x_axis_cb,
        ]
        for event in (
            backbone_cb.change, method_dd.change, dataset_dd.change,
            prune_strategy_dd.change, prune_structure_cb.change,
            show_no_recovery_cb.change, halve_x_axis_cb.change,
        ):
            event(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        refresh_btn.click(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        blocks.load(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal")

    return {
        "tab": tab, "backbone_cb": backbone_cb, "method_dd": method_dd,
        "prune_strategy_dd": prune_strategy_dd,
        "prune_structure_cb": prune_structure_cb, "dataset_dd": dataset_dd,
        "show_no_recovery_cb": show_no_recovery_cb,
        "halve_x_axis_cb": halve_x_axis_cb,
        "image": image, "status": status,
    }

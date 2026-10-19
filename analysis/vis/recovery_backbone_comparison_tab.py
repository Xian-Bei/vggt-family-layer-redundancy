from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, MultipleLocator

from analysis.vis.display_names import backbone_display_name
from analysis.vis.composite import Cell, Section, compose_sections, to_image

from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    METRIC_HEADERS,
    PUBLICATION_DETAIL_AXES_BOX_ASPECT,
    PUBLICATION_DETAIL_SUBPLOT_HEIGHT,
    PUBLICATION_METRIC_SUBPLOT_WIDTH,
    RECOVERY_METRIC_GRID,
    STRATEGY_PLOT_COLORS,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    metric_or_gt_mean_value,
    metric_value,
    move_y_axis_label_to_top,
    set_y_axis_max_decimals,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
)
from analysis.redundancy.skip_metric_scaling import (
    normalize_raw_metric_against_upper_worst,
)
from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_pruning_strategy_key,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.recovery.shared_results import (
    PRUNE_FAMILY_STRATEGIES,
    best_recovery_result,
    seven_metric_mean_score,
    single_double_mean_decision,
)
from analysis.vis.recovery_tab import (
    _ALL_DATASET_AVERAGE_GROUPS as _DATASET_AVERAGE_GROUPS,
    _dataset_group_average,
    _figure_to_rgb,
    _seven_metric_mean,
    _pruning_only_recovery_like_payload,
    discover_recovery_overview_options,
)
from analysis.vis.dataset_average_warning import (
    format_incomplete_average_warning,
)

# Recovery Remaining typography and outer-label spacing (pixels).
RECOVERY_REMAINING_SUBPLOT_FONT_SIZE = 24
RECOVERY_REMAINING_OUTER_FONT_SIZE = 32
RECOVERY_REMAINING_LEFT_LABEL_TO_IMAGE_GAP = 0
RECOVERY_REMAINING_TOP_LABEL_TO_IMAGE_GAP = 0
RECOVERY_REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP = 0


_PRUNING_OVERVIEW_COLOR_KEYS = (
    "CKA",
    "retention",
    "replaceme",
    "inv_mse",
    "cosine",
    "cca",
    "shortgpt",
    "gardener",
    "simpler",
)
_BACKBONE_COLORS = tuple(
    STRATEGY_PLOT_COLORS[key] for key in _PRUNING_OVERVIEW_COLOR_KEYS
)
_PRUNE_STRATEGY_FAMILIES = ("retention", "CKA")
_PRUNE_STRUCTURES = ("single", "double")
_DISPLAYED_STRUCTURE_MIN_FAMILIES = ("retention",)
_RECOVERY_METHOD_LEGEND_LABELS = {
    "recovery_ghost_interval_sequential": "CKA",
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


def _recovery_method_legend_label(
    method: str, prune_strategy: str = "CKA"
) -> str:
    strategy_family = canonical_pruning_strategy_key(prune_strategy)
    if strategy_family == "retention":
        return "Degradation"
    if strategy_family == "CKA":
        return "CKA"
    return _RECOVERY_METHOD_LEGEND_LABELS.get(
        str(method), str(method).removeprefix("recovery_")
    )


def _structure_legend_label(structure: str) -> str:
    return "Single" if str(structure) == "single" else "Multi"


def _is_no_recovery_strategy(strategy_key: str) -> bool:
    return str(strategy_key).strip().lower().startswith("no_recovery")


def _prune_family_plot_style(strategy_key: str) -> Dict[str, Any]:
    """Encode the fixed pruning family after single/double is collapsed."""
    key = str(strategy_key).strip().lower()
    if key == "retention_min" or key.endswith("_retention_min"):
        return {"linestyle": "-", "marker": "o"}
    if key == "cka_min" or key.endswith("_cka_min"):
        return {"linestyle": "-", "marker": "s"}
    return {
        "linestyle": strategy_plot_linestyle(strategy_key),
        "marker": strategy_plot_marker(strategy_key),
    }


def _recovery_state_style(strategy_key: str) -> Dict[str, Any]:
    """Encode recovery state without taking over color or line style."""
    retention_width_boost = (
        0.4 if str(strategy_key).strip().lower().endswith("retention_min") else 0.0
    )
    if _is_no_recovery_strategy(strategy_key):
        return {
            "linewidth": 1.6 + retention_width_boost,
            "alpha": 0.55,
            "markerfacecolor": "none",
            "markeredgecolor": None,
            "markeredgewidth": 1.5,
            "markersize": 7.0,
            "zorder": 2.5,
        }
    return {
        "linewidth": 2.8 + retention_width_boost,
        "alpha": 0.95,
        "markerfacecolor": None,
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
        "markersize": 7.8,
        "zorder": 4.0,
    }


def _overlapping_upper_linestyle(
    overlap_index: int, overlap_count: int
) -> Any:
    """Distribute coincident dotted bounds evenly across one dash cycle."""
    if int(overlap_index) == 0 or int(overlap_count) <= 1:
        return ":"
    pattern = tuple(float(value) for value in plt.rcParams["lines.dotted_pattern"])
    return (sum(pattern) * int(overlap_index) / int(overlap_count), pattern)


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


def _main_calibration_structure_selection(
    *,
    root: Path,
    recovery_method: str,
    family: str,
    selection_datasets: Sequence[str],
) -> Tuple[Dict[str, str], List[str]]:
    """Select a physical-plan structure once from the main calibration results."""
    strategies = PRUNE_FAMILY_STRATEGIES.get(str(family), ())
    if not strategies:
        return {}, [f"unsupported pruning family: {family}"]
    drop_counts = sorted({
        drop_count
        for strategy in strategies
        for drop_count in _strategy_selectors_by_drop(Path(root), strategy)
    })
    selected: Dict[str, str] = {}
    failures: List[str] = []
    for drop_count in drop_counts:
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
        selected[str(drop_count)] = str(winner["strategy"])
    return selected, failures


def _shared_profile_maxima(
    payloads: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Take each metric's largest profile maximum across displayed backbones."""
    maxima: Dict[str, float] = {}
    for metric in GT_MEAN_METRICS:
        values: List[float] = []
        for payload in payloads:
            bounds = payload.get("normalization_bounds", {})
            metric_bounds = bounds.get(metric, {}) if isinstance(bounds, dict) else {}
            maximum = (
                metric_bounds.get("max")
                if isinstance(metric_bounds, dict)
                else None
            )
            if (
                isinstance(maximum, (int, float))
                and np.isfinite(float(maximum))
            ):
                values.append(float(maximum))
        if values:
            maxima[metric] = max(values)
    return maxima


def _shared_profile_normalized_mean(
    point: Dict[str, Any],
    payload: Dict[str, Any],
    shared_maxima: Dict[str, float],
) -> float | None:
    """Normalize seven raw metrics against shared maxima and local upper values."""
    raw = point.get("raw_metric", point.get("metrics_mean", {}))
    upper = payload.get("upper_average", {})
    if not isinstance(raw, dict) or not isinstance(upper, dict):
        return None
    normalized: List[float] = []
    for metric in GT_MEAN_METRICS:
        value = metric_value(raw, metric)
        upper_value = metric_value(upper, metric)
        maximum = shared_maxima.get(metric)
        if (
            value is None
            or upper_value is None
            or maximum is None
            or not np.isfinite(float(value))
            or not np.isfinite(float(upper_value))
        ):
            return None
        normalized.append(float(
            normalize_raw_metric_against_upper_worst(
                np.asarray([float(value)], dtype=np.float64),
                upper_baseline=float(upper_value),
                worst_value=float(maximum),
                clamp_better_than_upper=False,
            )[0]
        ))
    return float(np.mean(normalized))


def _collapse_structure_min_strategy(
    payload: Dict[str, Any],
    *,
    method: str,
    family: str,
    calibration_selection: Mapping[str, str] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply one main-calibration single/double winner to every plotted scope."""
    strategies = payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return payload, []
    variants = tuple(f"{family}_{structure}" for structure in _PRUNE_STRUCTURES)
    def _select_curve(source_keys: Sequence[str]) -> Dict[str, Any]:
        drop_keys = {
            str(drop_key)
            for source_key in source_keys
            for drop_key in (
                strategies.get(source_key, {}).get("points", {})
                if isinstance(strategies.get(source_key), dict)
                else {}
            )
        }
        winners: Dict[str, Any] = {}
        for drop_key in drop_keys:
            if calibration_selection is not None:
                selected_variant = calibration_selection.get(drop_key)
                if selected_variant not in variants:
                    continue
                source_key = source_keys[variants.index(selected_variant)]
                source = strategies.get(source_key, {})
                point = (
                    source.get("points", {}).get(drop_key)
                    if isinstance(source, dict)
                    else None
                )
                if isinstance(point, dict) and point.get("available", False):
                    winners[drop_key] = {
                        **deepcopy(point),
                        "min_prune_strategy": selected_variant,
                        "selection_scope": "main_calibration",
                    }
                continue
            choices: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
            for structure, variant, source_key in zip(
                _PRUNE_STRUCTURES, variants, source_keys
            ):
                source = strategies.get(source_key, {})
                point = (
                    source.get("points", {}).get(drop_key)
                    if isinstance(source, dict)
                    else None
                )
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                raw = point.get("raw_metric", point.get("metrics_mean", {}))
                score = seven_metric_mean_score(raw) if isinstance(raw, dict) else None
                if score is not None:
                    choices[structure] = (score, variant, point)
            if choices:
                if set(choices) == {"single", "double"}:
                    decision = single_double_mean_decision(
                        choices["single"][0], choices["double"][0]
                    )
                    _, variant, point = choices[str(decision["winner"])]
                else:
                    _, variant, point = min(
                        choices.values(), key=lambda item: (item[0], item[1])
                    )
                winners[drop_key] = {
                    **deepcopy(point),
                    "min_prune_strategy": variant,
                }
        return winners

    target_method = f"{method}_{family}_min"
    selected: Dict[str, Any] = {}
    recovery_points = _select_curve(
        [f"{method}_{variant}" for variant in variants]
    )
    if recovery_points:
        selected[target_method] = {
            "display_name": _recovery_method_legend_label(method, family),
            "points": recovery_points,
        }
    baseline_points = _select_curve(
        [f"no_recovery_{variant}" for variant in variants]
    )
    if baseline_points:
        selected["no_recovery"] = {
            "display_name": "No Recovery",
            "points": baseline_points,
        }

    return {
        **payload,
        "strategies": selected,
        "testtime_points": [],
    }, [target_method]


def _calibration_color(
    base_color: Any, size: int, selected_sizes: Sequence[int] | None = None,
) -> tuple[float, float, float]:
    """Rank displayed sizes from a light tint to a dark shade of one hue."""
    rgb = to_rgb(base_color)
    if size == 0:
        strength = 0.22
    else:
        sizes = sorted({int(n) for n in (selected_sizes if selected_sizes is not None else (50, 100, 150)) if int(n) > 0} | {size})
        rank = sizes.index(size) / (len(sizes) - 1) if len(sizes) > 1 else 1.0
        strength = 0.50 + 0.95 * rank
    if strength <= 1.0:
        return tuple(1.0 - strength * (1.0 - channel) for channel in rgb)
    return tuple(channel * (2.0 - strength) for channel in rgb)


def _render_combined_figure(
    payloads: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    method_keys: Sequence[str],
    show_no_recovery: bool = True,
    total_units_by_backbone: Dict[str, int] | None = None,
    halve_x_axis: bool = False,
    show_upper: bool = True,
    show_backbone_legend: bool = True,
    calibration_size_multiplier: int = 1,
    x_axis_left: float | None = None,
) -> Any | None:
    if show_no_recovery:
        payloads = [(name, payload if payload.get("calibration_size") else _fill_missing_recovery_points(payload, method_keys))
                    for name, payload in payloads]
    axis_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    tick_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    legend_font_size = float(RECOVERY_REMAINING_SUBPLOT_FONT_SIZE)
    row_label_font_size = float(RECOVERY_REMAINING_OUTER_FONT_SIZE)
    baseline_keys = tuple(dict.fromkeys(
        str(key)
        for _, payload in payloads
        for key in (
            payload.get("strategies", {}).keys()
            if isinstance(payload.get("strategies"), dict)
            else ()
        )
        if _is_no_recovery_strategy(str(key))
    ))
    strategy_keys = tuple(dict.fromkeys(
        (*baseline_keys, *method_keys) if show_no_recovery else method_keys
    ))
    shared_profile_maxima = _shared_profile_maxima(
        [payload for _, payload in payloads]
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(
            PUBLICATION_METRIC_SUBPLOT_WIDTH * 4,
            PUBLICATION_DETAIL_SUBPLOT_HEIGHT * 2,
        ),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_box_aspect(PUBLICATION_DETAIL_AXES_BOX_ASPECT)
    plotted_metrics = tuple(
        metric for row in RECOVERY_METRIC_GRID for metric in row if metric != "gt_mean"
    )
    legend_ax = axes[1, 3]
    legend_ax.set_axis_off()
    upper_values_by_payload: List[List[float | None]] = []
    upper_y_overlap_totals: List[Dict[float, int]] = [
        {} for _ in plotted_metrics
    ]
    for backbone, payload in payloads:
        upper = payload.get("upper_average", {})
        normalized_upper = payload.get("normalized_upper_average", {})
        upper_values: List[float | None] = []
        for panel_idx, metric in enumerate(plotted_metrics):
            value = (
                metric_or_gt_mean_value(normalized_upper, "gt_mean")
                if metric == "gt_mean" and isinstance(normalized_upper, dict)
                else metric_value(upper, metric) if isinstance(upper, dict) else None
            )
            upper_value = float(value) if value is not None else None
            upper_values.append(upper_value)
            if upper_value is not None:
                totals = upper_y_overlap_totals[panel_idx]
                totals[upper_value] = totals.get(upper_value, 0) + 1
        upper_values_by_payload.append(upper_values)
    any_data = False
    backbone_handles: List[Line2D] = []
    present_styles: Dict[str, str] = {}
    upper_y_overlap_indices: List[Dict[float, int]] = [
        {} for _ in plotted_metrics
    ]
    backbone_ids = list(dict.fromkeys(
        str(payload.get("backbone", name)) for name, payload in payloads
    ))
    drawn_baselines: set[tuple[str, int, str]] = set()
    legend_backbones: set[str] = set()
    sizes = sorted({int(p["calibration_size"]) for _, p in payloads if p.get("calibration_size")})
    for backbone_idx, (backbone, payload) in enumerate(payloads):
        backbone_id = str(payload.get("backbone", backbone))
        base_color = _BACKBONE_COLORS[backbone_ids.index(backbone_id) % len(_BACKBONE_COLORS)]
        color = (
            _calibration_color(base_color, int(payload["calibration_size"]), sizes)
            if payload.get("calibration_size") else base_color
        )
        total_units = (total_units_by_backbone or {}).get(backbone)
        if backbone_id not in legend_backbones:
            legend_backbones.add(backbone_id)
            backbone_handles.append(
                Line2D(
                    [], [], color=base_color, linewidth=2.4,
                    label=backbone_display_name(backbone_id),
                )
            )
        strategies = payload.get("strategies", {})
        if not isinstance(strategies, dict):
            continue
        for panel_idx, metric in enumerate(plotted_metrics):
            ax = axes.flat[panel_idx]
            upper_value = upper_values_by_payload[backbone_idx][panel_idx]
            upper_y_overlap_index = 0
            upper_y_overlap_count = 0
            if upper_value is not None:
                indices = upper_y_overlap_indices[panel_idx]
                upper_y_overlap_index = indices.get(upper_value, 0)
                indices[upper_value] = upper_y_overlap_index + 1
                upper_y_overlap_count = upper_y_overlap_totals[panel_idx][upper_value]
            for strategy_key in strategy_keys:
                baseline_id = (backbone_id, panel_idx, strategy_key)
                is_baseline = _is_no_recovery_strategy(strategy_key)
                if is_baseline and baseline_id in drawn_baselines:
                    continue
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
                    if metric == "gt_mean":
                        value = _shared_profile_normalized_mean(
                            point, payload, shared_profile_maxima
                        )
                        if value is None:
                            value = _seven_metric_mean(point, payload)
                    else:
                        value = metric_value(raw, metric)
                    if value is not None:
                        drop_count = int(drop_key)
                        x_value = (
                            drop_count
                            if payload.get("point_axis") == "remaining"
                            else total_units - drop_count
                            if total_units is not None
                            else drop_count
                        )
                        xs.append(float(x_value) / 2.0 if halve_x_axis else float(x_value))
                        ys.append(float(value))
                if not xs:
                    continue
                if is_baseline:
                    drawn_baselines.add(baseline_id)
                if total_units is not None and upper_value is not None:
                    full_x = float(total_units) / (2.0 if halve_x_axis else 1.0)
                    # Use the same full-model measurement as the horizontal bound.
                    curve = dict(zip(xs, ys))
                    curve[full_x] = upper_value
                    xs, ys = list(curve), list(curve.values())
                if total_units_by_backbone is not None:
                    ordered = sorted(zip(xs, ys))
                    xs, ys = [item[0] for item in ordered], [item[1] for item in ordered]
                any_data = True
                display = str(strategy.get("display_name", strategy_key))
                present_styles.setdefault(strategy_key, display)
                family_style = _prune_family_plot_style(strategy_key)
                linestyle = "--" if is_baseline else "-"
                state_style = _recovery_state_style(strategy_key.replace("no_recovery", "recovery"))
                curve_color = _calibration_color(base_color, 0) if is_baseline else color
                (line,) = ax.plot(
                    xs, ys, color=curve_color, linestyle=linestyle,
                    linewidth=state_style["linewidth"],
                    alpha=state_style["alpha"],
                    zorder=state_style["zorder"],
                )
                marker_facecolor = state_style["markerfacecolor"] or curve_color
                marker_edgecolor = state_style["markeredgecolor"] or curve_color
                ax.scatter(
                    xs, ys,
                    facecolors=marker_facecolor,
                    edgecolors=marker_edgecolor,
                    linewidths=state_style["markeredgewidth"],
                    marker=family_style["marker"],
                    s=state_style["markersize"] ** 2,
                    alpha=state_style["alpha"],
                    zorder=state_style["zorder"] + 0.2,
                )
            if show_upper and upper_value is not None:
                ax.axhline(
                    float(upper_value),
                    color=color,
                    linestyle=_overlapping_upper_linestyle(
                        upper_y_overlap_index, upper_y_overlap_count
                    ),
                    linewidth=2.2,
                    alpha=.9,
                    zorder=2,
                )

    for panel_idx, metric in enumerate(plotted_metrics):
        ax = axes.flat[panel_idx]
        move_y_axis_label_to_top(
            ax,
            METRIC_HEADERS.get(metric, metric),
            fontsize=axis_font_size,
        )
        x_label = "Remaining units" if total_units_by_backbone is not None else "Dropped Layers"
        if halve_x_axis:
            x_label = f"{x_label} ÷ 2"
        ax.set_xlabel(
            x_label,
            fontsize=axis_font_size,
            fontweight="bold",
        )
        ax.tick_params(
            axis="both", labelsize=tick_font_size
        )
        # Recovery Remaining and Ablation Size share this raw-metric figure.
        set_y_axis_max_decimals(ax, 2)
        if total_units_by_backbone:
            scale = 2.0 if halve_x_axis else 1.0
            left = 1.0 / scale
            right = max(total_units_by_backbone.values()) / scale
            ax.set_xlim(left=left, right=right)
            ticks = sorted({left, right, *np.arange(4.0, right, 4.0)})
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        else:
            ax.set_xlim(left=0 if x_axis_left is None else float(x_axis_left))
            ax.xaxis.set_major_locator(MultipleLocator(4))
        for tick_label in (
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ):
            tick_label.set_fontsize(tick_font_size)
            tick_label.set_fontweight("bold")
        for offset_text in (
            ax.xaxis.get_offset_text(),
            ax.yaxis.get_offset_text(),
        ):
            offset_text.set_fontsize(tick_font_size)
            offset_text.set_fontweight("bold")
        ax.grid(True, alpha=.25)
    if not any_data:
        plt.close(fig)
        return None
    style_handles = []
    present_families = []
    if any(str(key).lower().endswith("_retention_min") for key in present_styles):
        present_families.append(("retention_min", "Degradation"))
    if any(str(key).lower().endswith("_cka_min") for key in present_styles):
        present_families.append(("CKA_min", "CKA"))
    for key, label in present_families:
        # In averaged ablation plots, the colored calibration handles below
        # already carry Degradation's circular marker.
        if not show_backbone_legend and label == "Degradation":
            continue
        family_style = _prune_family_plot_style(key)
        handle = Line2D(
            [], [], color="#333333", linestyle=family_style["linestyle"],
            marker=family_style["marker"], linewidth=2.2,
            markerfacecolor="#333333", markeredgecolor="#333333",
            markersize=7.4, label=label,
        )
        style_handles.append(handle)
    if not present_families:
        present_structures = []
        if any(str(key).lower().endswith("_single") for key in present_styles):
            present_structures.append(("structure_single", "Single"))
        if any(
            str(key).lower().endswith(("_double", "_multi"))
            for key in present_styles
        ):
            present_structures.append(("structure_double", "Multi"))
        for key, label in present_structures:
            handle = Line2D(
                [], [], color="#333333",
                linestyle=strategy_plot_linestyle(key),
                marker=strategy_plot_marker(key), linewidth=2.2,
                markerfacecolor="#333333", markeredgecolor="#333333",
                markersize=7.4, label=label,
            )
            dashes = strategy_plot_dashes(key)
            if dashes is not None:
                handle.set_dashes(dashes)
            style_handles.append(handle)
    upper_handles = (
        [Line2D(
            [], [], color="#333333", linestyle=":", linewidth=2.2,
            label="Upper",
        )]
        if show_upper
        else []
    )
    shade_handles = []
    shade_base_color = (
        _BACKBONE_COLORS[0] if not show_backbone_legend else "#333333"
    )
    if baseline_keys and show_no_recovery:
        shade_handles.append(Line2D(
            [], [], color=_calibration_color(shade_base_color, 0), linestyle="--",
            marker="o", markerfacecolor="none",
            markeredgecolor=_calibration_color(shade_base_color, 0),
            linewidth=2.4, label="No Recovery",
        ))
    if sizes:
        shade_handles.extend(
            Line2D([], [],
                   color=_calibration_color(shade_base_color, size, sizes),
                   marker="o",
                   markerfacecolor=_calibration_color(
                       shade_base_color, size, sizes
                   ),
                   markeredgecolor=_calibration_color(
                       shade_base_color, size, sizes
                   ),
                   markersize=7.4, linewidth=2.4,
                   label=(
                       f"{size * calibration_size_multiplier} scenes"
                       if calibration_size_multiplier != 1
                       else f"{size} scenes/domain"
                   ))
            for size in sizes
        )
    fig.tight_layout()
    displayed_backbone_handles = backbone_handles if show_backbone_legend else []
    legend_handles = [
        *displayed_backbone_handles,
        *shade_handles,
        *style_handles,
        *upper_handles,
    ]
    detailed_legend = bool(
        len(backbone_handles) > 1
        or len(sizes) > 1
        or len(method_keys) > 1
        or (baseline_keys and show_no_recovery)
        or (not show_backbone_legend and legend_handles)
    )
    if not detailed_legend:
        if displayed_backbone_handles:
            legend_ax.text(
                0.5,
                0.5,
                displayed_backbone_handles[0].get_label(),
                ha="center",
                va="center",
                transform=legend_ax.transAxes,
                fontsize=row_label_font_size,
                fontweight="bold",
            )
        return fig
    # Auto-discovered sizes can add more rows than the original three-size UI.
    panel_height_pt = legend_ax.get_window_extent().height * 72.0 / fig.dpi
    legend_font_size = min(
        legend_font_size,
        panel_height_pt / (0.9375 * (len(legend_handles) + 2)),
    )
    def create_legend(labelspacing: float) -> Any:
        return fig.legend(
            handles=legend_handles,
            title="Backbone (color)" if show_backbone_legend else None,
            loc="center",
            bbox_to_anchor=(.5, .5),
            bbox_transform=legend_ax.transAxes,
            ncol=1,
            fontsize=legend_font_size,
            markerscale=1.35,
            handlelength=3.2,
            handleheight=1.0,
            handletextpad=0.9,
            labelspacing=labelspacing,
            borderpad=0.75,
            borderaxespad=0.0,
            columnspacing=1.4,
            frameon=True,
        )

    legend = create_legend(0.48)
    # Match the legend frame to the neighboring axes' plot box (labels are
    # outside that box), distributing every row evenly over the full height.
    if len(legend_handles) > 1:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        target_height = legend_ax.get_window_extent(renderer).height
        current_height = legend.get_window_extent(renderer).height
        font_height = legend_font_size * fig.dpi / 72.0
        spacing = max(
            0.0,
            0.48
            + (target_height - current_height)
            / (font_height * (len(legend_handles) - 1)),
        )
        legend.remove()
        legend = create_legend(spacing)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(legend_font_size)
        legend.get_title().set_fontweight("bold")
    return fig


def _fill_missing_recovery_points(
    payload: Dict[str, Any], method_keys: Sequence[str],
) -> Dict[str, Any]:
    """Join baseline-only coordinates to recovery curves without replacing recovery."""
    result = deepcopy(payload)
    strategies = result.get("strategies", {})
    for baseline_key in list(strategies):
        if not _is_no_recovery_strategy(baseline_key):
            continue
        suffix = baseline_key.removeprefix("no_recovery")
        targets = [key for key in method_keys if key.endswith(suffix)]
        for key in targets:
            target = strategies.setdefault(key, {"points": {}})
            points = target.setdefault("points", {})
            for drop, baseline in strategies[baseline_key].get("points", {}).items():
                current = points.get(drop, {})
                if not current.get("available", False) and baseline.get("available", False):
                    points[drop] = deepcopy(baseline)
    return result


_AVERAGE_BACKBONES = ("vggt", "pi3", "vggt_omega")


def _average_backbone_payloads(
    payloads,
    totals,
    selected_backbones=_AVERAGE_BACKBONES,
    *,
    coordinate: str = "dropped",
):
    """Equal-weight raw means at strictly matching dropped/remaining coordinates."""
    if coordinate not in {"dropped", "remaining"}:
        raise ValueError("coordinate must be either 'dropped' or 'remaining'")
    selected_backbones = list(dict.fromkeys(selected_backbones))
    if not selected_backbones:
        return [], {}, ["Select at least one backbone."]
    grouped = {}
    for name, payload in payloads:
        size = int(payload.get("calibration_size", 50))
        backbone = payload.get("backbone", name)
        grouped.setdefault(size, {})[backbone] = (name, payload)
    output, output_totals, messages = [], {}, []

    def mean_maps(maps):
        common = set.intersection(*(set(m) for m in maps))
        result = {}
        for metric in common:
            values = [m[metric] for m in maps]
            if all(isinstance(v, (float, int)) and np.isfinite(v) for v in values):
                result[metric] = sum(values) / len(values)
        return result

    for size, records in sorted(grouped.items()):
        missing = set(selected_backbones) - records.keys()
        if missing:
            messages.append(f"{size} scenes/domain: selected-backbone mean skipped; missing {', '.join(sorted(missing))}.")
            continue
        entries = [records[b] for b in selected_backbones]
        units = {totals[name] for name, _ in entries}
        sources = [p for _, p in entries]
        strategies = {}
        keys = set.union(*(set(p.get("strategies", {})) for p in sources))
        skipped = 0
        for key in sorted(keys):
            maps = []
            for name, payload in entries:
                source_points = payload.get("strategies", {}).get(key, {}).get("points", {})
                if coordinate == "remaining":
                    total = int(totals[name])
                    source_points = {
                        str(total - int(drop)): point
                        for drop, point in source_points.items()
                        if str(drop).isdigit()
                    }
                maps.append(source_points)
            points = {}
            for point_coordinate in set.union(*(set(m) for m in maps)):
                candidates = [m.get(point_coordinate, {}) for m in maps]
                if not all(p.get("available", False) for p in candidates):
                    skipped += 1
                    continue
                raw_maps = [p.get("raw_metric", p.get("metrics_mean", {})) for p in candidates]
                raw = mean_maps(raw_maps)
                if not raw:
                    skipped += 1
                    continue
                points[point_coordinate] = {"available": True, "raw_metric": raw, "num_backbones": len(selected_backbones)}
            if points:
                strategies[key] = {"points": points}
        if skipped:
            messages.append(f"{size} scenes/domain: skipped {skipped} strategy/budget points without all selected backbones.")
        name = f"backbone_mean_{size}"
        if strategies:
            average_payload = {
                "backbone": f"Mean ({len(selected_backbones)} backbone{'s' if len(selected_backbones) != 1 else ''})",
                "is_backbone_average": True,
                "strategies": strategies,
                "upper_average": mean_maps([p.get("upper_average", {}) for p in sources]),
            }
            if coordinate == "remaining":
                average_payload["point_axis"] = "remaining"
            else:
                average_payload["calibration_size"] = size
            output.append((name, average_payload))
            if len(units) == 1:
                output_totals[name] = units.pop()
    return output, output_totals, messages


def discover_calibration_sizes(output_root: Path) -> list[str]:
    """Keep the main 50-scene baseline; discover ablation sizes with evaluations."""
    root = Path(output_root)
    samples_root = root.with_name(root.name + "_samples_ablation")
    sizes = {50}
    for directory in samples_root.glob("samples_*"):
        suffix = directory.name.removeprefix("samples_")
        if suffix.isdigit() and 0 < int(suffix) <= 150 and any(
            directory.glob("*/*/eval/recovery/*/*/summary.json")
        ):
            sizes.add(int(suffix))
    return [str(size) for size in sorted(sizes)]


def _calibration_choices(output_root: Path) -> list[tuple[str, str]]:
    return [(f"{size} ({'main experiment' if size == '50' else 'sample ablation'})", size)
            for size in discover_calibration_sizes(output_root)]


def render_recovery_backbone_remaining_comparison(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "same-dataset testing average",
    show_no_recovery: bool = True,
    *,
    halve_x_axis: bool = False,
    calibration_sizes: Sequence[str | int] | None = None,
    samples_output_root: Path | None = None,
    average_backbones: bool = False,
    x_axis: str = "remaining",
    prune_families: Sequence[str] | None = None,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Render the backbone comparison against remaining or dropped units."""
    backbones = list(dict.fromkeys(str(item) for item in (selected_backbones or ())))
    method = str(selected_method or "").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not method:
        return [], "Select one recovery method."
    sizes = list(dict.fromkeys(str(size) for size in calibration_sizes)) if calibration_sizes is not None else ["50"]
    if not sizes:
        return [], "Select at least one calibration size."
    if any(not size.isdigit() or not 0 < int(size) <= 150 for size in sizes):
        return [], "Calibration sizes must be integers from 1 to 150 scenes per domain."
    if x_axis not in {"remaining", "dropped"}:
        return [], "x_axis must be either 'remaining' or 'dropped'."
    sizes = list(dict.fromkeys(str(int(size)) for size in sizes))
    samples_root = Path(samples_output_root) if samples_output_root is not None else Path(output_root).with_name(Path(output_root).name + "_samples_ablation")
    if prune_families is None:
        displayed_families = (
            _PRUNE_STRATEGY_FAMILIES
            if x_axis == "remaining" and calibration_sizes is None
            else _DISPLAYED_STRUCTURE_MIN_FAMILIES
        )
    else:
        requested_families = {str(item) for item in prune_families}
        unsupported = requested_families.difference(_PRUNE_STRATEGY_FAMILIES)
        if unsupported:
            return [], "Unsupported pruning criteria: " + ", ".join(sorted(unsupported))
        displayed_families = tuple(
            family for family in _PRUNE_STRATEGY_FAMILIES
            if family in requested_families
        )
        if not displayed_families:
            return [], "Select at least one pruning criterion."
    variants = [
        f"{family}_{structure}"
        for family in displayed_families
        for structure in _PRUNE_STRUCTURES
    ]
    method_keys = [
        f"{method}_{family}_min"
        for family in displayed_families
    ]
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    total_units_by_backbone: Dict[str, int] = {}
    warnings: List[str] = []
    messages: List[str] = []
    for backbone, size in ((backbone, size) for backbone in backbones for size in sizes):
        root = Path(output_root) / backbone
        result_root = root if size == "50" else samples_root / f"samples_{int(size):03d}" / backbone
        series = backbone if calibration_sizes is None else f"{backbone}_{size}"
        label = f"{backbone_display_name(backbone)} · {size} scenes/domain"
        try:
            total_units = int(load_prune_plan_bundle(root).get("num_units", 0))
            if total_units <= 0:
                raise ValueError("prune plan bundle has no positive num_units")
            total_units_by_backbone[series] = total_units
            pruning = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)
            merged: Dict[str, Any] = {
                "upper_average": {},
                "normalized_upper_average": {},
                "normalization_bounds": {},
                "strategies": {},
                "testtime_points": [],
            }
            if calibration_sizes is not None:
                merged.update(calibration_size=int(size), series_label=label, backbone=backbone)
            missing_variants: List[str] = []
            for variant in variants:
                structure = variant.rsplit("_", 1)[-1]
                recovery = collect_recovery_eval_payload(
                    experiment_root=result_root,
                    prune_strategy=variant,
                    **({"prune_plan_root": root} if size != "50" else {}),
                )
                # Ablation folders contain evaluations only, not upper bounds.
                # Use the same full-model baseline for every calibration size.
                if size != "50":
                    for name, item in recovery.get("datasets", {}).items():
                        item["upper_average"] = pruning.get("datasets", {}).get(name, {}).get("upper_average", {})
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
                        owner=f"{label} / {method} / {variant}",
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
                    if calibration_sizes is not None:
                        continue
                upper = payload.get("upper_average", {}) if isinstance(payload, dict) else {}
                if not merged["upper_average"] and isinstance(upper, dict):
                    merged["upper_average"] = upper
                normalized_upper = (
                    payload.get("normalized_upper_average", {})
                    if isinstance(payload, dict)
                    else {}
                )
                if (
                    not merged["normalized_upper_average"]
                    and isinstance(normalized_upper, dict)
                ):
                    merged["normalized_upper_average"] = normalized_upper
                normalization_bounds = (
                    payload.get("normalization_bounds", {})
                    if isinstance(payload, dict)
                    else {}
                )
                if (
                    not merged["normalization_bounds"]
                    and isinstance(normalization_bounds, dict)
                ):
                    merged["normalization_bounds"] = normalization_bounds
                if not isinstance(strategies, dict):
                    continue
                for source_key, strategy_payload in strategies.items():
                    if not isinstance(strategy_payload, dict):
                        continue
                    if source_key == method:
                        target_key = f"{method}_{variant}"
                        display_name = (
                            f"{_recovery_method_legend_label(method, variant)} · "
                            f"{_structure_legend_label(structure)}"
                        )
                    elif str(source_key).startswith("no_recovery"):
                        target_key = f"no_recovery_{variant}"
                        display_name = (
                            f"No Recovery · {_structure_legend_label(structure)}"
                        )
                    else:
                        continue
                    merged["strategies"][target_key] = {
                        **strategy_payload,
                        "display_name": display_name,
                    }
            if missing_variants:
                messages.append(
                    f"{label}: recovery method missing for {', '.join(missing_variants)}; "
                    + ("missing calibration results skipped" if calibration_sizes is not None else "showing available pruning baselines")
                )
            if merged["strategies"]:
                loaded.append((series, merged))
        except Exception as exc:
            messages.append(f"{label}: {type(exc).__name__}: {exc}")
    collapsed: List[Tuple[str, Dict[str, Any]]] = []
    calibration_selection_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    selection_datasets = _DATASET_AVERAGE_GROUPS["training average"]
    for series, payload in loaded:
        source_backbone = str(payload.get("backbone", series))
        combined_strategies: Dict[str, Any] = {}
        for family in displayed_families:
            cache_key = (source_backbone, family)
            if cache_key not in calibration_selection_cache:
                calibration_selection, selection_failures = (
                    _main_calibration_structure_selection(
                        root=Path(output_root) / source_backbone,
                        recovery_method=method,
                        family=family,
                        selection_datasets=selection_datasets,
                    )
                )
                calibration_selection_cache[cache_key] = calibration_selection
                messages.extend(
                    f"{backbone_display_name(source_backbone)} calibration selection: {failure}"
                    for failure in selection_failures
                )
            selected_payload, _ = _collapse_structure_min_strategy(
                payload,
                method=method,
                family=family,
                calibration_selection=calibration_selection_cache[cache_key],
            )
            family_strategies = selected_payload.get("strategies", {})
            if isinstance(family_strategies, dict):
                recovery_key = f"{method}_{family}_min"
                if recovery_key in family_strategies:
                    combined_strategies[recovery_key] = family_strategies[recovery_key]
                if "no_recovery" in family_strategies:
                    baseline_key = f"no_recovery_{family}_min"
                    combined_strategies[baseline_key] = {
                        **family_strategies["no_recovery"],
                        "display_name": f"No Recovery · {_recovery_method_legend_label(method, family)}",
                    }
        collapsed.append((series, {
            **payload,
            "strategies": combined_strategies,
            "testtime_points": [],
        }))
    loaded = collapsed
    if average_backbones:
        loaded, total_units_by_backbone, average_messages = _average_backbone_payloads(
            loaded, total_units_by_backbone, backbones,
            coordinate=x_axis,
        )
        messages.extend(average_messages)
        coordinate_label = "remaining-layer count" if x_axis == "remaining" else "drop budget"
        messages.append(f"Equal-weight arithmetic mean of raw metrics over {', '.join(backbone_display_name(b) for b in backbones)} at each shared {coordinate_label} and calibration size (no normalization).")
    fig = _render_combined_figure(
        loaded, method_keys=method_keys,
        show_no_recovery=bool(show_no_recovery),
        total_units_by_backbone=(
            total_units_by_backbone if x_axis == "remaining" else None
        ),
        halve_x_axis=bool(halve_x_axis),
        show_upper=calibration_sizes is None,
        show_backbone_legend=not (
            calibration_sizes is not None and average_backbones
        ),
        calibration_size_multiplier=(
            2 if calibration_sizes is not None else 1
        ),
        x_axis_left=(
            12.0
            if calibration_sizes is not None and x_axis == "dropped"
            else None
        ),
    )
    if fig is None:
        return [], "\n".join([*warnings, *messages, "No matching data to plot."])
    try:
        image = _figure_to_rgb(fig)
    finally:
        plt.close(fig)
    messages.append(
        f"Rendered {len(loaded)} series in one 2×4 figure with "
        "seven raw metrics and a factored backbone/color and calibration/shade legend."
    )
    axis_caption = "remaining units" if x_axis == "remaining" else "dropped layers"
    return [(image, f"Backbone comparison by {axis_caption}")], "\n".join(
        [*warnings, *messages]
    )


def build_recovery_backbone_remaining_comparison_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    default_method = (
        "recovery_ghost_interval_sequential_autogroup"
        if "recovery_ghost_interval_sequential_autogroup" in methods
        else methods[0] if methods else None
    )

    def redraw(
        selected_backbones, selected_method, dataset,
        show_no_recovery, halve_x_axis, prune_families, average_backbones,
    ):
        images, status_value = render_recovery_backbone_remaining_comparison(
            Path(output_root), selected_backbones, selected_method, dataset,
            show_no_recovery,
            halve_x_axis=halve_x_axis,
            average_backbones=average_backbones,
            x_axis="remaining",
            prune_families=prune_families,
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
            "All selected backbones share one 2×4 figure, with one hue per backbone. "
            "Seven panels show raw metrics; the bottom-right panel holds the legend. "
            "The Degradation and CKA curves each select the method's best "
            "single/double point on the main calibration datasets at every drop count "
            "using the raw seven-metric mean with a 0.10 Double-improvement margin, then reuse that physical pruning "
            "choice on every test dataset. Each matching No Recovery baseline uses "
            "the same calibration-selected pruning choice."
            + " The x-axis is the number of remaining units, inferred from each backbone's prune plan."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                method_dd = gr.Dropdown(
                    label="Recovery method",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=default_method, multiselect=False,
                )
                prune_families_cb = gr.CheckboxGroup(
                    label="Pruning criteria",
                    choices=[
                        ("Degradation", "retention"),
                        ("CKA", "CKA"),
                    ],
                    value=["retention", "CKA"],
                    info="Each criterion independently selects its best single/double result at every point.",
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=[
                        ("Training", "training average"),
                        ("Same-Dataset Generalization", "same-dataset testing average"),
                        ("Cross-Dataset Generalization", "cross-dataset testing average"),
                    ],
                    value="same-dataset testing average",
                    multiselect=False,
                )
                show_no_recovery_cb = gr.Checkbox(
                    label="Show No recovery",
                    value=True,
                )
                average_backbones_cb = gr.Checkbox(
                    label="Average selected backbones (raw metrics)",
                    value=False,
                    info="Equally averages checked Backbones at shared remaining-layer counts. Points missing any selected model are skipped.",
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
            backbone_cb, method_dd, dataset_dd,
            show_no_recovery_cb, halve_x_axis_cb,
            prune_families_cb, average_backbones_cb,
        ]
        for event in (
            backbone_cb.change, method_dd.change, dataset_dd.change,
            show_no_recovery_cb.change, halve_x_axis_cb.change,
            prune_families_cb.change, average_backbones_cb.change,
        ):
            event(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        refresh_btn.click(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")
        blocks.load(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal")

    return {
        "tab": tab, "backbone_cb": backbone_cb, "method_dd": method_dd,
        "dataset_dd": dataset_dd,
        "show_no_recovery_cb": show_no_recovery_cb,
        "halve_x_axis_cb": halve_x_axis_cb,
        "prune_families_cb": prune_families_cb,
        "average_backbones_cb": average_backbones_cb,
        "image": image, "status": status,
    }


def build_recovery_ablation_size_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    calibration_choices = _calibration_choices(Path(output_root))
    default_method = (
        "recovery_ghost_interval_sequential_autogroup"
        if "recovery_ghost_interval_sequential_autogroup" in methods
        else methods[0] if methods else None
    )

    def redraw(
        selected_backbones, selected_method, dataset,
        show_no_recovery, halve_x_axis, average_backbones, calibration_sizes,
    ):
        images, status_value = render_recovery_backbone_remaining_comparison(
            Path(output_root), selected_backbones, selected_method, dataset,
            show_no_recovery,
            halve_x_axis=halve_x_axis,
            calibration_sizes=calibration_sizes,
            average_backbones=average_backbones,
            x_axis="dropped",
        )
        return (
            _compose_recovery_remaining(images[0][0]) if images else None,
            status_value,
        )

    with gr.Tab("Ablation Size", id="recovery_ablation_size") as tab:
        gr.Markdown("## Ablation Size")
        gr.Markdown(
            "Compare calibration-set sizes in one 2×4 raw-metric figure. "
            "Each backbone uses one hue, larger calibration sets use darker shades, "
            "and No Recovery uses the lightest shade. The x-axis is the number of "
            "dropped layers. The 50-scenes/domain series uses the main experiment; "
            "other sizes use <output_root>_samples_ablation/samples_<size>. All sizes "
            "reuse the Single/Double selector chosen from the main experiment's "
            "calibration results so only recovery fitting-set size changes."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                method_dd = gr.Dropdown(
                    label="Recovery method",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=default_method,
                    multiselect=False,
                )
                calibration_sizes_cb = gr.CheckboxGroup(
                    label="Calibration scenes per domain",
                    choices=calibration_choices,
                    value=[size for _, size in calibration_choices],
                    info="All discovered sizes are selected by default. Shades are re-ranked on selection changes; the largest displayed size is darkest. Counts are per domain; missing results are skipped.",
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=[
                        ("Training", "training average"),
                        ("Same-Dataset Generalization", "same-dataset testing average"),
                        ("Cross-Dataset Generalization", "cross-dataset testing average"),
                    ],
                    value="same-dataset testing average",
                    multiselect=False,
                )
                show_no_recovery_cb = gr.Checkbox(
                    label="Show No recovery",
                    value=True,
                )
                average_backbones_cb = gr.Checkbox(
                    label="Average selected backbones (raw metrics)",
                    value=True,
                    info="Equally averages the checked Backbones. One checked model shows its original values; points missing any selected model are skipped.",
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
                    label="Combined ablation-size comparison",
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "recovery-backbone-comparison-image"],
                )

        inputs = [
            backbone_cb, method_dd, dataset_dd,
            show_no_recovery_cb, halve_x_axis_cb,
            average_backbones_cb, calibration_sizes_cb,
        ]
        for event in (
            backbone_cb.change, method_dd.change, dataset_dd.change,
            show_no_recovery_cb.change, halve_x_axis_cb.change,
            calibration_sizes_cb.change, average_backbones_cb.change,
        ):
            event(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal", trigger_mode="always_last")

        def refresh(*values):
            choices = _calibration_choices(Path(output_root))
            available = {value for _, value in choices}
            selected = [size for size in (values[-1] or []) if size in available]
            result = redraw(*values[:-1], selected)
            return (*result, gr.update(choices=choices, value=selected))

        refresh_btn.click(refresh, inputs=inputs, outputs=[image, status, calibration_sizes_cb], show_progress="minimal", trigger_mode="always_last")
        blocks.load(redraw, inputs=inputs, outputs=[image, status], show_progress="minimal")

    return {
        "tab": tab, "backbone_cb": backbone_cb, "method_dd": method_dd,
        "dataset_dd": dataset_dd,
        "show_no_recovery_cb": show_no_recovery_cb,
        "halve_x_axis_cb": halve_x_axis_cb,
        "calibration_sizes_cb": calibration_sizes_cb,
        "average_backbones_cb": average_backbones_cb,
        "image": image, "status": status,
    }

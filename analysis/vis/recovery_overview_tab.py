from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)

from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    RECOVERY_METHOD_ORDER,
    RECOVERY_METRIC_GRID,
    _with_dataset_average,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    metric_value,
    render_strategy_metric_figure,
)
from analysis.vis.dataset_average_warning import (
    DATASET_AVERAGE_GROUPS,
    format_incomplete_average_warning,
)
from analysis.vis.pruning_plan_tab import _mean_backbone_plot_payloads


_BACKBONE_ORDER = ("mapanything_v1", "mapanything", "vggt_omega", "pi3", "vggt")
_PRUNE_STRATEGIES = (
    "ours_double", "ours_single", "gt_double", "gt_single",
)
_BASELINE_STRATEGIES = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
)
_DATASET_AVERAGE_GROUPS = DATASET_AVERAGE_GROUPS
_MEAN_BACKBONE = "mean"
_MIN_PRUNE_STRATEGY = "min"


def _seven_metric_mean(point: Dict[str, Any], payload: Dict[str, Any]) -> float | None:
    """Return the normalized seven-metric mean used to choose one whole point."""
    normalized = point.get("normalized_metric", {})
    if isinstance(normalized, dict):
        values = [metric_value(normalized, metric) for metric in GT_MEAN_METRICS]
        if all(value is not None and np.isfinite(value) for value in values):
            return float(np.mean(values))
    raw = point.get("raw_metric", point.get("metrics_mean", {}))
    if not isinstance(raw, dict):
        return None
    bounds = payload.get("normalization_bounds", {})
    values: List[float] = []
    for metric in GT_MEAN_METRICS:
        value = metric_value(raw, metric)
        if value is None or not np.isfinite(value):
            return None
        metric_bounds = bounds.get(metric, {}) if isinstance(bounds, dict) else {}
        low = metric_bounds.get("min") if isinstance(metric_bounds, dict) else None
        high = metric_bounds.get("max") if isinstance(metric_bounds, dict) else None
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high > low:
            values.append((float(value) - float(low)) / (float(high) - float(low)))
        else:
            values.append(float(value))
    return float(np.mean(values))


def _minimum_strategy_plot_payload(
    candidates: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    tie_no_recovery_to_winner: bool = True,
) -> Dict[str, Any]:
    """Select one complete strategy point per drop using its 7-metric mean."""
    valid = [(name, payload) for name, payload in candidates if isinstance(payload, dict)]
    if not valid:
        return {}
    result = deepcopy(valid[0][1])
    result["strategies"] = {}
    strategy_keys = {
        str(key)
        for _, payload in valid
        for key in payload.get("strategies", {})
        if isinstance(payload.get("strategies", {}), dict)
    }
    winner_sources: Dict[str, Dict[str, str]] = {}
    primary_keys = (
        {key for key in strategy_keys if not key.startswith("no_recovery")}
        if tie_no_recovery_to_winner else strategy_keys
    )
    for strategy_key in primary_keys:
        choices: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = {}
        display_name = strategy_key
        for source_name, payload in valid:
            strategies = payload.get("strategies", {})
            strategy = strategies.get(strategy_key, {}) if isinstance(strategies, dict) else {}
            if not isinstance(strategy, dict):
                continue
            display_name = str(strategy.get("display_name", display_name))
            points = strategy.get("points", {})
            for drop_key, point in points.items() if isinstance(points, dict) else ():
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                score = _seven_metric_mean(point, payload)
                if score is not None:
                    choices.setdefault(str(drop_key), []).append((score, source_name, point))
        selected_points: Dict[str, Any] = {}
        for drop_key, point_choices in choices.items():
            _, source_name, point = min(point_choices, key=lambda item: (item[0], item[1]))
            selected_points[drop_key] = {
                **deepcopy(point), "min_prune_strategy": source_name,
            }
            winner_sources.setdefault(strategy_key, {})[drop_key] = source_name
        if selected_points:
            result["strategies"][strategy_key] = {
                "display_name": display_name, "points": selected_points,
            }
    if tie_no_recovery_to_winner:
        candidate_map = dict(valid)
        for strategy_key, winners in winner_sources.items():
            if not strategy_key.startswith("recovery_"):
                continue
            baseline_points: Dict[str, Any] = {}
            baseline_drop_keys = {
                str(drop_key)
                for _, payload in valid
                for key, baseline in payload.get("strategies", {}).items()
                if str(key).startswith("no_recovery") and isinstance(baseline, dict)
                for drop_key in baseline.get("points", {})
            }
            for drop_key, source_name in winners.items():
                source_strategies = candidate_map[source_name].get("strategies", {})
                if not isinstance(source_strategies, dict):
                    continue
                baseline = next(
                    (
                        value for key, value in source_strategies.items()
                        if str(key).startswith("no_recovery") and isinstance(value, dict)
                    ),
                    {},
                )
                point = baseline.get("points", {}).get(drop_key)
                if isinstance(point, dict) and point.get("available", False):
                    baseline_points[drop_key] = {
                        **deepcopy(point), "min_prune_strategy": source_name,
                    }
            for drop_key in baseline_drop_keys.difference(winners):
                choices: List[Tuple[float, str, Dict[str, Any]]] = []
                for source_name, payload in valid:
                    source_strategies = payload.get("strategies", {})
                    if not isinstance(source_strategies, dict):
                        continue
                    baseline = next(
                        (
                            value for key, value in source_strategies.items()
                            if str(key).startswith("no_recovery")
                            and isinstance(value, dict)
                        ),
                        {},
                    )
                    point = baseline.get("points", {}).get(drop_key)
                    if not isinstance(point, dict) or not point.get("available", False):
                        continue
                    score = _seven_metric_mean(point, payload)
                    if score is not None:
                        choices.append((score, source_name, point))
                if choices:
                    _, source_name, point = min(
                        choices, key=lambda item: (item[0], item[1])
                    )
                    baseline_points[drop_key] = {
                        **deepcopy(point),
                        "min_prune_strategy": source_name,
                        "no_recovery_fallback": True,
                    }
            if baseline_points:
                result["strategies"][f"no_recovery__{strategy_key}"] = {
                    "display_name": (
                        f"No recovery · {strategy_key.removeprefix('recovery_')} winner"
                    ),
                    "points": baseline_points,
                }
    return result


def _sort_key(name: str, preferred: Sequence[str]) -> Tuple[int, str]:
    key = str(name)
    try:
        return (preferred.index(key), key)
    except ValueError:
        return (len(preferred), key)


def discover_recovery_overview_options(output_root: Path) -> Tuple[List[str], List[str]]:
    """Discover controls from plans and result folders, including unfinished runs."""
    root = Path(output_root)
    backbones: List[str] = []
    methods = set()
    if root.is_dir():
        for backbone_dir in (path for path in root.iterdir() if path.is_dir()):
            from_results = {
                path.name
                for path in backbone_dir.glob("*/eval/recovery/recovery_*")
                if path.is_dir() and path.name in RECOVERY_METHOD_ORDER
            }
            from_results.update(
                path.name
                for path in backbone_dir.glob("*/eval/recovery/*/recovery_*")
                if path.is_dir() and path.name in RECOVERY_METHOD_ORDER
            )
            from_plans = {
                path.stem
                for path in backbone_dir.glob("recovery_*.json")
                if path.stem in RECOVERY_METHOD_ORDER
            }
            from_bundle = set()
            bundle_path = backbone_dir / "recovery_plan_all.json"
            if bundle_path.is_file():
                try:
                    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    bundle = {}
                bundle_methods = bundle.get("methods", {}) if isinstance(bundle, dict) else {}
                if isinstance(bundle_methods, dict):
                    from_bundle = {
                        str(name)
                        for name in bundle_methods
                        if str(name) in RECOVERY_METHOD_ORDER
                    }
            found = from_results | from_plans | from_bundle
            if found or (backbone_dir / "prune_plan_all.json").is_file():
                backbones.append(backbone_dir.name)
                methods.update(found)
    return (
        sorted(set(backbones), key=lambda name: _sort_key(name, _BACKBONE_ORDER)),
        sorted(methods, key=lambda name: _sort_key(name, RECOVERY_METHOD_ORDER)),
    )


def discover_recovery_overview_prune_strategies(output_root: Path) -> List[str]:
    """Discover supported strategy views with shared recovery results."""
    root = Path(output_root)
    strategies = set()
    if root.is_dir():
        for backbone_root in (path for path in root.iterdir() if path.is_dir()):
            try:
                bundle = load_prune_plan_bundle(backbone_root)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            available_selectors = {
                selector_root.name
                for recovery_root in backbone_root.glob("*/eval/recovery")
                if recovery_root.is_dir()
                for selector_root in recovery_root.iterdir()
                if selector_root.is_dir()
                and any(
                    method.is_dir() and method.name in RECOVERY_METHOD_ORDER
                    for method in selector_root.iterdir()
                )
            }
            checkpoint_root = backbone_root / "recovery"
            if checkpoint_root.is_dir():
                available_selectors.update(
                    selector_root.name
                    for selector_root in checkpoint_root.iterdir()
                    if selector_root.is_dir()
                    and any(
                        method.is_dir() and method.name in RECOVERY_METHOD_ORDER
                        for method in selector_root.iterdir()
                    )
                )
            plans = bundle_plan_records(bundle)
            for strategy, strategy_record in bundle_strategy_records(bundle).items():
                if strategy not in _PRUNE_STRATEGIES:
                    continue
                points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
                selectors = set()
                for plan_name in points.values() if isinstance(points, dict) else ():
                    record = plans.get(str(plan_name))
                    if isinstance(record, dict):
                        selectors.add(physical_pruning_selector_from_record(record))
                if not available_selectors or selectors.intersection(available_selectors):
                    strategies.add(str(strategy))
    return sorted(strategies, key=lambda name: _sort_key(name, _PRUNE_STRATEGIES))


def discover_recovery_overview_datasets(output_root: Path) -> List[str]:
    """Discover datasets with a recovery evaluation directory."""
    root = Path(output_root)
    datasets = {
        path.parent.parent.name
        for path in root.glob("*/*/eval/recovery")
        if path.is_dir()
    }
    return [*_DATASET_AVERAGE_GROUPS, *sorted(datasets)]


def _dataset_group_average(payload: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Average the available datasets belonging to a named overview group."""
    dataset_payloads = payload.get("datasets", {})
    if not isinstance(dataset_payloads, dict):
        return {}
    selected = {
        name: dataset_payloads[name]
        for name in _DATASET_AVERAGE_GROUPS.get(dataset, ())
        if name in dataset_payloads
    }
    average = _with_dataset_average({"datasets": selected}).get(
        "dataset_average", {}
    )
    strategies = average.get("strategies", {}) if isinstance(average, dict) else {}
    if isinstance(strategies, dict):
        for strategy_key, strategy_payload in strategies.items():
            if not str(strategy_key).startswith("no_recovery") or not isinstance(
                strategy_payload, dict
            ):
                continue
            points = strategy_payload.get("points", {})
            if isinstance(points, dict):
                strategy_payload["points"] = {
                    str(drop_key): point
                    for drop_key, point in points.items()
                    if str(drop_key).isdigit()
                    and int(drop_key) % 2 == 0
                }
    return average


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    content = np.any(rgb < 250, axis=2)
    ys, xs = np.nonzero(content)
    if ys.size and xs.size:
        pad = 10
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
        rgb = rgb[y0:y1, x0:x1]
    return rgb.copy()


def _pruning_only_recovery_like_payload(
    pruning_payload: Dict[str, Any],
    *,
    dataset: str,
    prune_strategy: str = "",
) -> Dict[str, Any]:
    """Build a recovery-like plotting payload from pruning-only summaries.

    This keeps the same rendering path as normal recovery plots so that
    missing-recovery backbones still show upper + no-recovery curves.
    """
    if dataset == "mean":
        source = pruning_payload.get("dataset_average", {})
    elif dataset in _DATASET_AVERAGE_GROUPS:
        source = _dataset_group_average(pruning_payload, dataset)
    else:
        datasets = pruning_payload.get("datasets", {})
        source = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    if not isinstance(source, dict):
        return {}

    source_strategies = source.get("strategies", {})
    if not isinstance(source_strategies, dict):
        source_strategies = {}

    # Mirror the same baseline family names used by recovery overview.
    mapping: List[Tuple[str, str, str]] = []
    effective_prune_strategy = "" if prune_strategy == "legacy" else prune_strategy
    if effective_prune_strategy and effective_prune_strategy in source_strategies:
        source_strategy = source_strategies.get(effective_prune_strategy, {})
        display_name = (
            str(source_strategy.get("display_name", effective_prune_strategy))
            if isinstance(source_strategy, dict)
            else effective_prune_strategy
        )
        mapping.append((effective_prune_strategy, "no_recovery", f"No recovery · {display_name}"))
    elif "ours_single" in source_strategies:
        mapping.append(("ours_single", "no_recovery_single", "No recovery · CKA single"))
    if not effective_prune_strategy and "ours_double" in source_strategies:
        mapping.append(("ours_double", "no_recovery_double", "No recovery · CKA double"))
    if not mapping and "ours" in source_strategies:
        mapping.append(("ours", "no_recovery", "No recovery"))

    strategies: Dict[str, Any] = {}
    for src_key, dst_key, display_name in mapping:
        src_payload = source_strategies.get(src_key, {})
        points = src_payload.get("points", {}) if isinstance(src_payload, dict) else {}
        if not isinstance(points, dict) or not points:
            continue
        strategies[dst_key] = {
            "display_name": display_name,
            "points": points,
        }

    return {
        "upper_average": source.get("upper_average", {}),
        "normalized_upper_average": source.get(
            "normalized_upper_average", {}
        ),
        "normalization_bounds": source.get("normalization_bounds", {}),
        "strategies": strategies,
    }


def render_recovery_overview(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    dataset: str = "training average",
    normalize_metrics: bool = True,
    prune_strategy: str = "",
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Reload summaries and redraw one recovery comparison figure per backbone."""
    backbones = list(dict.fromkeys(str(name) for name in (selected_backbones or [])))
    methods = list(dict.fromkeys(str(name) for name in (selected_methods or [])))
    dataset = str(dataset or "training average").strip()
    prune_strategy = str(prune_strategy or "").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not methods:
        return [], "Select at least one recovery method."

    available_backbones, _ = discover_recovery_overview_options(Path(output_root))
    if _MEAN_BACKBONE in backbones and not available_backbones:
        return [], "No backbones are available to average."

    gallery: List[Tuple[np.ndarray, str]] = []
    warnings: List[str] = []
    messages: List[str] = []
    for backbone in backbones:
        try:
            contributors = (
                available_backbones
                if backbone == _MEAN_BACKBONE
                else [backbone]
            )
            contributor_plots: List[Dict[str, Any]] = []
            contributor_fallbacks: List[Dict[str, Any]] = []
            contributor_candidate_plots: List[List[Tuple[str, Dict[str, Any]]]] = []
            contributor_candidate_fallbacks: List[List[Tuple[str, Dict[str, Any]]]] = []
            skipped_profiles: List[str] = []
            for contributor in contributors:
                backbone_root = Path(output_root) / contributor
                pruning = collect_pruning_eval_payload(
                    experiment_root=backbone_root,
                    prune_plan_root=backbone_root,
                )
                strategy_candidates = (
                    _PRUNE_STRATEGIES
                    if prune_strategy == _MIN_PRUNE_STRATEGY
                    else (prune_strategy,)
                )
                candidate_plots: List[Tuple[str, Dict[str, Any]]] = []
                candidate_fallbacks: List[Tuple[str, Dict[str, Any]]] = []
                for candidate in strategy_candidates:
                    recovery = collect_recovery_eval_payload(
                        experiment_root=backbone_root,
                        prune_strategy=None if candidate == "legacy" else candidate,
                    )
                    if dataset in _DATASET_AVERAGE_GROUPS:
                        recovery_datasets = recovery.get("datasets", {})
                        for method in methods:
                            method_datasets = (
                                [
                                    name for name, payload in recovery_datasets.items()
                                    if isinstance(payload, dict)
                                    and isinstance(payload.get("strategies"), dict)
                                    and method in payload["strategies"]
                                ] if isinstance(recovery_datasets, dict) else []
                            )
                            warning = format_incomplete_average_warning(
                                owner=f"{contributor} / {method} / {candidate}",
                                scope=dataset,
                                expected=_DATASET_AVERAGE_GROUPS[dataset],
                                available=method_datasets,
                            )
                            if warning:
                                warnings.append(warning)
                    recovery = add_recovery_comparison_strategies(
                        recovery, pruning, None if candidate == "legacy" else candidate,
                    )
                    skipped_profiles.extend(
                        f"{contributor}:{profile}"
                        for profile in recovery.get("skipped_normalization_datasets", [])
                    )
                    if dataset == "mean":
                        candidate_plot = recovery.get("dataset_average", {})
                    elif dataset in _DATASET_AVERAGE_GROUPS:
                        candidate_plot = _dataset_group_average(recovery, dataset)
                    else:
                        dataset_payloads = recovery.get("datasets", {})
                        candidate_plot = (
                            dataset_payloads.get(dataset, {})
                            if isinstance(dataset_payloads, dict) else {}
                        )
                    candidate_plots.append((candidate, candidate_plot))
                    candidate_fallbacks.append((
                        candidate,
                        _pruning_only_recovery_like_payload(
                            pruning, dataset=dataset, prune_strategy=candidate,
                        ),
                    ))
                contributor_candidate_plots.append(candidate_plots)
                contributor_candidate_fallbacks.append(candidate_fallbacks)
                title_scope = "averaged over datasets" if dataset == "mean" else dataset
                contributor_plots.append(
                    _minimum_strategy_plot_payload(candidate_plots)
                    if prune_strategy == _MIN_PRUNE_STRATEGY else candidate_plots[0][1]
                )
                contributor_fallbacks.append(
                    _minimum_strategy_plot_payload(candidate_fallbacks)
                    if prune_strategy == _MIN_PRUNE_STRATEGY else candidate_fallbacks[0][1]
                )

            if backbone == _MEAN_BACKBONE and prune_strategy == _MIN_PRUNE_STRATEGY:
                plot_payload = _minimum_strategy_plot_payload([
                    (
                        candidate,
                        _mean_backbone_plot_payloads([
                            dict(candidate_list).get(candidate, {})
                            for candidate_list in contributor_candidate_plots
                        ]),
                    )
                    for candidate in _PRUNE_STRATEGIES
                ])
            else:
                plot_payload = (
                    _mean_backbone_plot_payloads(contributor_plots)
                    if backbone == _MEAN_BACKBONE else contributor_plots[0]
                )
            available = (
                plot_payload.get("strategies", {})
                if isinstance(plot_payload, dict)
                else {}
            )
            selected = [
                key for key in available
                if isinstance(available, dict) and str(key).startswith("no_recovery")
            ] + [
                key for key in methods
                if isinstance(available, dict) and key in available
            ]
            fig = render_strategy_metric_figure(
                payload=plot_payload,
                title=f"{backbone}: {prune_strategy or 'legacy'} recovery · {title_scope}",
                metric_grid=RECOVERY_METRIC_GRID,
                strategy_keys=selected,
                legend_fontsize=11,
                normalize_metrics=bool(normalize_metrics),
                always_normalize_metric_mean=True,
                include_upper_anchor=False,
                subplot_width=7.4,
            )
            if fig is None:
                if backbone == _MEAN_BACKBONE and prune_strategy == _MIN_PRUNE_STRATEGY:
                    fallback_plot_payload = _minimum_strategy_plot_payload([
                        (
                            candidate,
                            _mean_backbone_plot_payloads([
                                dict(candidate_list).get(candidate, {})
                                for candidate_list in contributor_candidate_fallbacks
                            ]),
                        )
                        for candidate in _PRUNE_STRATEGIES
                    ])
                else:
                    fallback_plot_payload = (
                        _mean_backbone_plot_payloads(contributor_fallbacks)
                        if backbone == _MEAN_BACKBONE else contributor_fallbacks[0]
                    )
                fallback_available = (
                    fallback_plot_payload.get("strategies", {})
                    if isinstance(fallback_plot_payload, dict)
                    else {}
                )
                fallback_selected = [
                    key
                    for key in (*_BASELINE_STRATEGIES, *methods)
                    if isinstance(fallback_available, dict) and key in fallback_available
                ]
                fig = render_strategy_metric_figure(
                    payload=fallback_plot_payload,
                    title=f"{backbone}: {prune_strategy or 'legacy'} recovery · {title_scope}",
                    metric_grid=RECOVERY_METRIC_GRID,
                    strategy_keys=fallback_selected if fallback_selected else None,
                    legend_fontsize=11,
                    normalize_metrics=bool(normalize_metrics),
                    always_normalize_metric_mean=True,
                    include_upper_anchor=False,
                    subplot_width=7.4,
                )

                if fig is None:
                    messages.append(
                        f"{backbone}: no matching recovery data (and no pruning baseline fallback)"
                    )
                    continue

                messages.append(
                    f"{backbone}: recovery missing; rendered upper + no-recovery baselines from pruning summaries"
                )
            try:
                gallery.append((_figure_to_rgb(fig), backbone))
            finally:
                plt.close(fig)
            missing = [method for method in methods if method not in available]
            suffix = f" (missing: {', '.join(missing)})" if missing else ""
            if skipped_profiles:
                suffix += f" (missing pruning profile: {', '.join(skipped_profiles)})"
            messages.append(
                f"{backbone} / {dataset} / {prune_strategy or 'legacy'}: redrawn from current summaries "
                f"| normalized={bool(normalize_metrics)}"
                f" | backbones={','.join(contributors)}{suffix}"
            )
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    return gallery, "\n".join([*warnings, *messages])


def build_recovery_overview_tab(*, blocks: gr.Blocks, output_root: Path) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = discover_recovery_overview_datasets(Path(output_root))
    prune_strategies = discover_recovery_overview_prune_strategies(Path(output_root))
    default_prune_strategy = prune_strategies[0] if prune_strategies else _PRUNE_STRATEGIES[0]

    def _redraw(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
        normalize_metrics: bool,
        prune_strategy: str,
    ):
        return render_recovery_overview(
            Path(output_root), selected_backbones, selected_methods, dataset,
            normalize_metrics, prune_strategy,
        )

    def _refresh_options(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
        normalize_metrics: bool,
        prune_strategy: str,
    ):
        latest_backbones, latest_methods = discover_recovery_overview_options(
            Path(output_root)
        )
        latest_datasets = discover_recovery_overview_datasets(Path(output_root))
        latest_prune_strategies = discover_recovery_overview_prune_strategies(
            Path(output_root)
        )
        backbone_value = [
            name
            for name in (selected_backbones or [])
            if name == _MEAN_BACKBONE or name in latest_backbones
        ] or latest_backbones
        method_value = [
            name for name in (selected_methods or []) if name in latest_methods
        ] or latest_methods
        dataset_value = (
            dataset if dataset in latest_datasets else "training average"
        )
        prune_strategy_value = (
            prune_strategy
            if prune_strategy in (*_PRUNE_STRATEGIES, _MIN_PRUNE_STRATEGY)
            else (
                latest_prune_strategies[0]
                if latest_prune_strategies
                else _PRUNE_STRATEGIES[0]
            )
        )
        gallery_value, status_value = render_recovery_overview(
            Path(output_root), backbone_value, method_value, dataset_value,
            normalize_metrics, prune_strategy_value,
        )
        return (
            gr.update(choices=[_MEAN_BACKBONE, *latest_backbones], value=backbone_value),
            gr.update(
                choices=[
                    (name.removeprefix("recovery_"), name)
                    for name in latest_methods
                ],
                value=method_value,
            ),
            gr.update(choices=latest_datasets, value=dataset_value),
            gr.update(
                choices=[*_PRUNE_STRATEGIES, _MIN_PRUNE_STRATEGY],
                value=prune_strategy_value,
            ),
            gallery_value,
            status_value,
        )

    with gr.Tab("Recovery Overview", id="recovery_overview") as tab:
        gr.Markdown("## Recovery Overview")
        gr.Markdown(
            "Compare recovery methods across backbones. Every update reloads the latest "
            "evaluation summaries and redraws the plots; saved PNG files are not used. "
            "The normalization control affects only individual metrics; the 7-metric "
            "mean is always normalized per metric before averaging."
            " Choosing min selects one complete strategy per drop from the displayed "
            "7-metric mean; individual metrics are never selected independently. Its "
            "No recovery curve follows that winner where recovery exists, and otherwise "
            "uses its own 7-metric-mean winner."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[_MEAN_BACKBONE, *backbones],
                    value=[_MEAN_BACKBONE, *backbones],
                )
                prune_strategy_dd = gr.Dropdown(
                    label="Prune strategy",
                    choices=[*_PRUNE_STRATEGIES, _MIN_PRUNE_STRATEGY],
                    value=default_prune_strategy,
                    multiselect=False,
                )
                method_cb = gr.CheckboxGroup(
                    label="Recovery methods",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=methods,
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=datasets,
                    value="training average",
                    multiselect=False,
                )
                normalize_metrics = gr.Checkbox(
                    label="Normalize individual metrics to [0,1]",
                    value=True,
                    info="The 7-metric mean is always normalized per metric.",
                )
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=4)
            with gr.Column(scale=5):
                gallery = gr.Gallery(
                    label="Recovery comparison by backbone",
                    columns=1,
                    height="auto",
                    object_fit="contain",
                    elem_classes=[
                        "results-gallery",
                        "recovery-eval-gallery",
                        "recovery-overview-gallery",
                    ],
                )

        for event in (
            backbone_cb.change,
            method_cb.change,
            dataset_dd.change,
            normalize_metrics.change,
            prune_strategy_dd.change,
        ):
            event(
                _redraw,
                inputs=[backbone_cb, method_cb, dataset_dd, normalize_metrics, prune_strategy_dd],
                outputs=[gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
        refresh_btn.click(
            _refresh_options,
            inputs=[backbone_cb, method_cb, dataset_dd, normalize_metrics, prune_strategy_dd],
            outputs=[backbone_cb, method_cb, dataset_dd, prune_strategy_dd, gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _redraw,
            inputs=[backbone_cb, method_cb, dataset_dd, normalize_metrics, prune_strategy_dd],
            outputs=[gallery, status],
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "backbone_cb": backbone_cb,
        "method_cb": method_cb,
        "prune_strategy_dd": prune_strategy_dd,
        "dataset_dd": dataset_dd,
        "normalize_metrics": normalize_metrics,
        "gallery": gallery,
        "status": status,
    }

from __future__ import annotations

import json
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
    RECOVERY_METHOD_ORDER,
    RECOVERY_METRIC_GRID,
    _with_dataset_average,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    render_strategy_metric_figure,
)
from analysis.vis.dataset_average_warning import (
    DATASET_AVERAGE_GROUPS,
    format_incomplete_average_warning,
)
from analysis.vis.pruning_plan_tab import _mean_backbone_plot_payloads


_BACKBONE_ORDER = ("vggt_omega", "pi3", "vggt")
_PRUNE_STRATEGY_ORDER = (
    "ours_double", "ours_single", "gt_double", "gt_single",
    "cosine_double", "cosine_single", "inv_mse_double", "inv_mse_single",
    "cca_double", "cca_single",
    "replaceme_double", "replaceme_single", "shortgpt", "simpler", "legacy",
)
_BASELINE_STRATEGIES = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
)
_DATASET_AVERAGE_GROUPS = DATASET_AVERAGE_GROUPS
_MEAN_BACKBONE = "mean"


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
    """Discover strategy views whose selectors have shared recovery results."""
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
                points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
                selectors = set()
                for plan_name in points.values() if isinstance(points, dict) else ():
                    record = plans.get(str(plan_name))
                    if isinstance(record, dict):
                        selectors.add(physical_pruning_selector_from_record(record))
                if not available_selectors or selectors.intersection(available_selectors):
                    strategies.add(str(strategy))
    return sorted(strategies, key=lambda name: _sort_key(name, _PRUNE_STRATEGY_ORDER))


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
            skipped_profiles: List[str] = []
            for contributor in contributors:
                backbone_root = Path(output_root) / contributor
                pruning = collect_pruning_eval_payload(
                    experiment_root=backbone_root,
                    prune_plan_root=backbone_root,
                )
                recovery = collect_recovery_eval_payload(
                    experiment_root=backbone_root,
                    prune_strategy=None if prune_strategy == "legacy" else prune_strategy,
                )
                if dataset in _DATASET_AVERAGE_GROUPS:
                    recovery_datasets = recovery.get("datasets", {})
                    for method in methods:
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
                            owner=f"{contributor} / {method}",
                            scope=dataset,
                            expected=_DATASET_AVERAGE_GROUPS[dataset],
                            available=method_datasets,
                        )
                        if warning:
                            warnings.append(warning)
                recovery = add_recovery_comparison_strategies(
                    recovery,
                    pruning,
                    None if prune_strategy == "legacy" else prune_strategy,
                )
                skipped_profiles.extend(
                    f"{contributor}:{profile}"
                    for profile in recovery.get("skipped_normalization_datasets", [])
                )
                if dataset == "mean":
                    contributor_plot = recovery.get("dataset_average", {})
                    title_scope = "averaged over datasets"
                elif dataset in _DATASET_AVERAGE_GROUPS:
                    contributor_plot = _dataset_group_average(recovery, dataset)
                    title_scope = dataset
                else:
                    dataset_payloads = recovery.get("datasets", {})
                    contributor_plot = (
                        dataset_payloads.get(dataset, {})
                        if isinstance(dataset_payloads, dict)
                        else {}
                    )
                    title_scope = dataset
                contributor_plots.append(contributor_plot)
                contributor_fallbacks.append(
                    _pruning_only_recovery_like_payload(
                        pruning,
                        dataset=dataset,
                        prune_strategy=prune_strategy,
                    )
                )

            plot_payload = (
                _mean_backbone_plot_payloads(contributor_plots)
                if backbone == _MEAN_BACKBONE
                else contributor_plots[0]
            )
            available = (
                plot_payload.get("strategies", {})
                if isinstance(plot_payload, dict)
                else {}
            )
            selected = [
                key
                for key in (*_BASELINE_STRATEGIES, *methods)
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
                fallback_plot_payload = (
                    _mean_backbone_plot_payloads(contributor_fallbacks)
                    if backbone == _MEAN_BACKBONE
                    else contributor_fallbacks[0]
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
    default_prune_strategy = prune_strategies[0] if prune_strategies else "legacy"

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
            if prune_strategy in latest_prune_strategies
            else (latest_prune_strategies[0] if latest_prune_strategies else "legacy")
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
            gr.update(choices=latest_prune_strategies or ["legacy"], value=prune_strategy_value),
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
                    choices=prune_strategies or ["legacy"],
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

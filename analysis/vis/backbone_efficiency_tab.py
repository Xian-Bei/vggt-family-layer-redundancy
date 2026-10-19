from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from analysis.vis.display_names import backbone_display_name

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.cache_replay import get_cache_replay_engine
from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    bundle_plan_records,
    bundle_physical_plan_records,
    bundle_strategy_records,
    canonical_pruning_strategy_key,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.prune_plan.eval_plots import (
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
)
from analysis.vis.recovery_backbone_comparison_tab import (
    _BACKBONE_COLORS,
    _PRUNE_STRATEGY_FAMILIES,
    _PRUNE_STRUCTURES,
    _selected_dataset,
)
from analysis.vis.recovery_tab import (
    _DATASET_AVERAGE_GROUPS,
    _pruning_only_recovery_like_payload,
    discover_recovery_overview_datasets,
    discover_recovery_overview_options,
)
from analysis.vis.dataset_average_warning import (
    dataset_display_choices,
    dataset_display_name,
    format_incomplete_average_warning,
)


_PERFORMANCE_METRICS = (
    ("inference_time_ms", "Inference Time", "ms"),
    ("infer_peak_vram_gb", "Peak VRAM", "GB"),
)
_STRUCTURE_MARKERS = {"single": "o", "double": "s"}


def _unique_parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    return total


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]) -> Dict[int, torch.nn.Parameter]:
    return {id(parameter): parameter for parameter in parameters}


def _record_unit_ids(record: Mapping[str, Any], catalog: Sequence[Any]) -> List[str]:
    explicit = record.get("dropped_unit_ids", ())
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        values = [str(value) for value in explicit if str(value)]
        if values:
            return values
    drop = [int(value) for value in record.get("drop", ())]
    unit_ids = [str(spec.unit_id) for spec in catalog]
    if any(index < 0 or index >= len(unit_ids) for index in drop):
        raise ValueError(f"drop indices are outside the model unit catalog: {drop}")
    return [unit_ids[index] for index in drop]


def count_parameter_series(
    *,
    model: torch.nn.Module,
    adapter: Any,
    accounting_parameters: Iterable[torch.nn.Parameter],
    bundle: Mapping[str, Any],
    selectors_by_drop: Mapping[int, Sequence[str]],
) -> Tuple[Dict[int, Tuple[int, int]], List[str]]:
    """Count exact full/aggregator parameters for the selected physical plans."""
    all_parameters = _unique_parameters(model.parameters())
    aggregator_parameters = _unique_parameters(accounting_parameters)
    catalog = adapter.layer_unit_catalog(model)
    records = bundle_physical_plan_records(dict(bundle))
    series: Dict[int, Tuple[int, int]] = {
        0: (sum(p.numel() for p in all_parameters.values()),
            sum(p.numel() for p in aggregator_parameters.values()))
    }
    warnings: List[str] = []
    for drop_count, selectors in sorted(selectors_by_drop.items()):
        candidates: set[Tuple[int, int]] = set()
        for selector in dict.fromkeys(str(value) for value in selectors if str(value)):
            record = records.get(selector)
            if not isinstance(record, dict):
                warnings.append(f"drop {drop_count}: plan record missing for {selector}")
                continue
            removed: Dict[int, torch.nn.Parameter] = {}
            for unit_id in _record_unit_ids(record, catalog):
                block = adapter.block_for_unit(model, unit_id)
                removed.update(_unique_parameters(block.parameters()))
            removed_full = sum(
                parameter.numel()
                for identity, parameter in removed.items()
                if identity in all_parameters
            )
            removed_aggregator = sum(
                parameter.numel()
                for identity, parameter in removed.items()
                if identity in aggregator_parameters
            )
            candidates.add(
                (series[0][0] - int(removed_full), series[0][1] - int(removed_aggregator))
            )
        if len(candidates) == 1:
            series[int(drop_count)] = next(iter(candidates))
        elif len(candidates) > 1:
            warnings.append(
                f"drop {drop_count}: selected plans have different parameter counts; point skipped"
            )
    return series, warnings


def compute_parameter_series_for_backbone(
    backbone: str,
    bundle: Mapping[str, Any],
    selectors_by_drop: Mapping[int, Sequence[str]],
) -> Tuple[Dict[int, Tuple[int, int]], List[str]]:
    """Load one pristine backbone on CPU and calculate parameter curves live."""
    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    engine = get_cache_replay_engine(backbone)
    model = adapter.load_pretrained(
        hf_model=adapter.default_hf_model(), device="cpu", dtype=None, eval_mode=True
    )
    try:
        return count_parameter_series(
            model=model,
            adapter=adapter,
            accounting_parameters=engine.parameter_accounting_parameters(model),
            bundle=bundle,
            selectors_by_drop=selectors_by_drop,
        )
    finally:
        del model
        gc.collect()


def _selected_variants(strategy: str, structures: Sequence[str]) -> List[str]:
    return [f"{strategy}_{structure}" for structure in structures]


def _load_backbone_efficiency_payload(
    root: Path,
    *,
    method: str,
    dataset: str,
    variants: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    pruning = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)
    merged: Dict[str, Any] = {"upper_average": {}, "strategies": {}}
    messages: List[str] = []
    for variant in variants:
        structure = variant.rsplit("_", 1)[-1]
        recovery = collect_recovery_eval_payload(
            experiment_root=root, prune_strategy=variant
        )
        if dataset in _DATASET_AVERAGE_GROUPS:
            recovery_datasets = recovery.get("datasets", {})
            available = (
                [
                    name for name, item in recovery_datasets.items()
                    if isinstance(item, dict)
                    and isinstance(item.get("strategies"), dict)
                    and method in item["strategies"]
                ]
                if isinstance(recovery_datasets, dict) else []
            )
            warning = format_incomplete_average_warning(
                owner=f"{method} / {variant}", scope=dataset,
                expected=_DATASET_AVERAGE_GROUPS[dataset], available=available,
            )
            if warning:
                messages.append(warning)
        recovery = add_recovery_comparison_strategies(recovery, pruning, variant)
        payload = _selected_dataset(recovery, dataset)
        strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
        if not isinstance(strategies, dict) or method not in strategies:
            payload = _pruning_only_recovery_like_payload(
                pruning, dataset=dataset, prune_strategy=variant
            )
            strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
            messages.append(f"{variant}: recovery method has no matching data")
        upper = payload.get("upper_average", {}) if isinstance(payload, dict) else {}
        if not merged["upper_average"] and isinstance(upper, dict):
            merged["upper_average"] = dict(upper)
        if not isinstance(strategies, dict):
            continue
        for source_key, strategy_payload in strategies.items():
            if not isinstance(strategy_payload, dict):
                continue
            if source_key == method:
                target_key = f"{method}_{variant}"
            elif str(source_key).startswith("no_recovery"):
                target_key = f"no_recovery_{structure}"
            else:
                continue
            merged["strategies"][target_key] = strategy_payload
    for structure in dict.fromkeys(variant.rsplit("_", 1)[-1] for variant in variants):
        recovery_keys = [
            f"{method}_{variant}" for variant in variants if variant.endswith(f"_{structure}")
        ]
        recovery_drops = {
            str(drop)
            for key in recovery_keys
            for drop, point in merged["strategies"].get(key, {}).get("points", {}).items()
            if isinstance(point, dict) and point.get("available", False)
        }
        baseline_keys = [f"no_recovery_{structure}"]
        baseline_drops = {
            str(drop)
            for key in baseline_keys
            for drop, point in merged["strategies"].get(key, {}).get("points", {}).items()
            if isinstance(point, dict) and point.get("available", False)
        }
        missing = sorted(
            baseline_drops.difference(recovery_drops),
            key=lambda value: int(value) if value.isdigit() else 10**9,
        )
        if missing:
            messages.append(
                f"{structure}: recovery missing at drop counts {', '.join(missing)}; pruning remains visible"
            )
    return merged, messages


def _point_metric(point: Mapping[str, Any], key: str) -> float | None:
    metrics = point.get("metrics_mean", {})
    value = metrics.get(key) if isinstance(metrics, Mapping) else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _series_from_strategy(
    payload: Mapping[str, Any], strategy_key: str, metric: str
) -> Tuple[List[int], List[float], Dict[int, str]]:
    strategy = payload.get("strategies", {}).get(strategy_key, {})
    points = strategy.get("points", {}) if isinstance(strategy, Mapping) else {}
    values: List[Tuple[int, float, str]] = []
    for drop_key, point in points.items() if isinstance(points, Mapping) else ():
        if not isinstance(point, Mapping) or not point.get("available", False):
            continue
        value = _point_metric(point, metric)
        selector = str(point.get("selector", ""))
        if str(drop_key).isdigit() and value is not None:
            values.append((int(drop_key), value, selector))
    values.sort()
    return (
        [item[0] for item in values],
        [item[1] for item in values],
        {item[0]: item[2] for item in values if item[2]},
    )


def _selectors_for_parameter_count(
    payload: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    method: str,
    strategy: str,
    structures: Sequence[str],
) -> Dict[int, List[str]]:
    selectors: Dict[int, List[str]] = {}
    strategy_records = bundle_strategy_records(dict(bundle))
    plan_records = bundle_plan_records(dict(bundle))
    for structure in structures:
        recovery_key = (
            f"{method}_{strategy}_{structure}"
        )
        for key in (recovery_key, f"no_recovery_{structure}"):
            curve = payload.get("strategies", {}).get(key, {})
            points = curve.get("points", {}) if isinstance(curve, Mapping) else {}
            for drop_key, point in points.items() if isinstance(points, Mapping) else ():
                if not str(drop_key).isdigit() or not isinstance(point, Mapping):
                    continue
                drop_count = int(drop_key)
                selector = str(point.get("selector", ""))
                variant = str(point.get("min_prune_strategy", ""))
                if not variant:
                    variant = f"{strategy}_{structure}"
                if not selector and variant:
                    strategy_record = strategy_records.get(variant, {})
                    strategy_points = (
                        strategy_record.get("points", {})
                        if isinstance(strategy_record, Mapping) else {}
                    )
                    plan_name = strategy_points.get(str(drop_count)) if isinstance(strategy_points, Mapping) else None
                    plan_record = plan_records.get(str(plan_name))
                    if isinstance(plan_record, dict):
                        selector = physical_pruning_selector_from_record(plan_record)
                if selector:
                    selectors.setdefault(drop_count, []).append(selector)
    return selectors


def _format_parameter_axis(ax: Any) -> None:
    from matplotlib.ticker import FuncFormatter

    def formatter(value: float, _position: int) -> str:
        if abs(value) >= 1e9:
            return f"{value / 1e9:.2g}B"
        return f"{value / 1e6:.0f}M"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render_backbone_efficiency(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "training average",
    prune_strategy: str = "CKA",
    prune_structures: Sequence[str] | None = None,
) -> Tuple[np.ndarray | None, str]:
    backbones = list(dict.fromkeys(str(value) for value in (selected_backbones or ())))
    method = str(selected_method or "").strip()
    strategy = canonical_pruning_strategy_key(prune_strategy or "")
    structures = list(dict.fromkeys(
        str(value).strip().lower()
        for value in (prune_structures if prune_structures is not None else _PRUNE_STRUCTURES)
        if str(value).strip().lower() in _PRUNE_STRUCTURES
    ))
    if not backbones:
        return None, "Select at least one backbone."
    if not method:
        return None, "Select one recovery method."
    if strategy not in _PRUNE_STRATEGY_FAMILIES:
        return None, "Prune strategy must be retention or CKA."
    if not structures:
        return None, "Select at least one pruning structure."

    variants = _selected_variants(strategy, structures)
    loaded: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    messages: List[str] = []
    for backbone in backbones:
        root = Path(output_root) / backbone
        try:
            bundle = load_prune_plan_bundle(root)
            payload, notes = _load_backbone_efficiency_payload(
                root, method=method, dataset=dataset, variants=variants,
            )
            loaded.append((backbone, payload, bundle))
            messages.extend(f"{backbone}: {note}" for note in notes)
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")

    if not loaded:
        return None, "\n".join([*messages, "No matching backbone data to plot."])

    parameter_series: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for backbone, payload, bundle in loaded:
        selectors_by_drop = _selectors_for_parameter_count(
            payload, bundle, method=method, strategy=strategy, structures=structures
        )
        try:
            series, notes = compute_parameter_series_for_backbone(
                backbone, bundle, selectors_by_drop
            )
            parameter_series[backbone] = series
            messages.extend(f"{backbone}: {note}" for note in notes)
        except Exception as exc:
            messages.append(f"{backbone}: parameter count failed: {type(exc).__name__}: {exc}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), dpi=130)
    any_data = False
    for index, (backbone, payload, _bundle) in enumerate(loaded):
        color = _BACKBONE_COLORS[index % len(_BACKBONE_COLORS)]
        params = parameter_series.get(backbone, {})
        if params:
            xs = sorted(params)
            axes[0].plot(xs, [params[x][0] for x in xs], color=color, linestyle="--", linewidth=2.2)
            axes[0].plot(xs, [params[x][1] for x in xs], color=color, linestyle="-", linewidth=2.2)
            any_data = True
        upper = payload.get("upper_average", {})
        for axis_offset, (metric, _title, _unit) in enumerate(_PERFORMANCE_METRICS, start=1):
            upper_value = upper.get(metric) if isinstance(upper, Mapping) else None
            for structure in structures:
                marker = _STRUCTURE_MARKERS[structure]
                recovery_key = (
                    f"{method}_{strategy}_{structure}"
                )
                for key, linestyle in ((recovery_key, "-"), (f"no_recovery_{structure}", "--")):
                    xs, ys, _ = _series_from_strategy(payload, key, metric)
                    if isinstance(upper_value, (int, float)) and math.isfinite(float(upper_value)):
                        xs = [0, *xs]
                        ys = [float(upper_value), *ys]
                    if xs:
                        axes[axis_offset].plot(
                            xs, ys, color=color, linestyle=linestyle, marker=marker,
                            linewidth=2.2, markersize=5.5, alpha=.95,
                        )
                        any_data = True

    titles = ("Parameters", "Inference Time", "Peak VRAM")
    ylabels = ("Parameters", "Time (ms)", "Peak VRAM (GB)")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Dropped layers")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_xlim(left=0)
        ax.grid(True, alpha=.25)
    _format_parameter_axis(axes[0])

    backbone_handles = [
        Line2D(
            [], [], color=_BACKBONE_COLORS[i % len(_BACKBONE_COLORS)],
            linewidth=3, label=backbone_display_name(name),
        )
        for i, (name, _, _) in enumerate(loaded)
    ]
    semantic_handles = [
        Line2D([], [], color="#333333", linestyle="-", label="Recovery / Aggregator"),
        Line2D([], [], color="#333333", linestyle="--", label="Pruning / Full model"),
        *[
            Line2D([], [], color="#333333", linestyle="None", marker=_STRUCTURE_MARKERS[s], label=s.title())
            for s in structures
        ],
    ]
    fig.legend(
        handles=[*backbone_handles, *semantic_handles], loc="upper center",
        bbox_to_anchor=(.5, .99), ncol=min(7, len(backbone_handles) + len(semantic_handles)),
        frameon=True,
    )
    fig.suptitle(
        f"Backbone Efficiency · {method.removeprefix('recovery_')} · "
        f"{strategy} · {dataset_display_name(dataset)}",
        fontsize=18, fontweight="bold", y=1.07,
    )
    fig.tight_layout(rect=(0, 0, 1, .9))
    if not any_data:
        plt.close(fig)
        return None, "\n".join([*messages, "No matching data to plot."])
    try:
        image = _figure_to_rgb(fig)
    finally:
        plt.close(fig)
    messages.append(f"Rendered {len(loaded)} backbone(s); parameters were recalculated from live models.")
    return image, "\n".join(messages)


def build_backbone_efficiency_tab(*, blocks: gr.Blocks, output_root: Path) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = discover_recovery_overview_datasets(Path(output_root))
    default_method = (
        "recovery_ghost_interval_sequential"
        if "recovery_ghost_interval_sequential" in methods
        else methods[0] if methods else None
    )

    def redraw(selected_backbones, selected_method, dataset, strategy, structures):
        return render_backbone_efficiency(
            Path(output_root), selected_backbones, selected_method, dataset,
            strategy, structures,
        )

    with gr.Tab("Backbone Efficiency", id="backbone_efficiency") as tab:
        gr.Markdown("## Backbone Efficiency")
        gr.Markdown(
            "Compare live parameter counts with stored pruning/recovery inference time and peak VRAM. "
            "Colors identify backbones; performance uses solid recovery and dashed pruning curves."
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
                    choices=list(_PRUNE_STRATEGY_FAMILIES),
                    value="CKA", multiselect=False,
                )
                prune_structure_cb = gr.CheckboxGroup(
                    label="Pruning structure", choices=list(_PRUNE_STRUCTURES),
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
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=6)
            with gr.Column(scale=5):
                image = gr.Image(
                    label="Backbone efficiency", type="numpy", interactive=False,
                    elem_classes=["results-media", "backbone-efficiency-image"],
                )

        inputs = [
            backbone_cb, method_dd, dataset_dd, prune_strategy_dd, prune_structure_cb
        ]
        for event in (
            backbone_cb.change, method_dd.change, dataset_dd.change,
            prune_strategy_dd.change, prune_structure_cb.change,
        ):
            event(
                redraw, inputs=inputs, outputs=[image, status],
                show_progress="full", trigger_mode="always_last",
            )
        refresh_btn.click(
            redraw, inputs=inputs, outputs=[image, status],
            show_progress="full", trigger_mode="always_last",
        )
        blocks.load(
            redraw, inputs=inputs, outputs=[image, status], show_progress="full"
        )
    return {
        "tab": tab, "backbone_cb": backbone_cb,
        "prune_strategy_dd": prune_strategy_dd,
        "prune_structure_cb": prune_structure_cb, "method_dd": method_dd,
        "dataset_dd": dataset_dd, "image": image, "status": status,
    }

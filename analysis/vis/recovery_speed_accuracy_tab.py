from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.prune_plan import load_prune_plan_bundle
from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    METRIC_HEADERS,
    OURS_CURVE_ZORDER,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    metric_value,
)
from analysis.vis.backbone_efficiency_tab import (
    compute_parameter_series_for_backbone,
)
from analysis.vis.dataset_average_warning import (
    dataset_display_choices,
    format_incomplete_average_warning,
)
from analysis.vis.display_names import backbone_display_name
from analysis.vis.recovery_backbone_comparison_tab import (
    _BACKBONE_COLORS,
    _DATASET_AVERAGE_GROUPS,
    _filter_testtime_to_family_winners,
    _load_testtime_points,
    _main_calibration_structure_selection,
    _selected_dataset,
    _collapse_structure_min_strategy,
)
from analysis.vis.recovery_tab import (
    discover_recovery_overview_options,
)


_BENCHMARK_FILENAME = "inference_benchmark_v3.json"
_FASTVGGT_PREFIX = "fastvggt_"
_FASTVGGT_PATTERN = re.compile(
    r"^fastvggt_m(?P<start>\d+)_r(?P<whole>\d+)p(?P<fraction>\d+)$"
)
_RETENTION_VARIANTS = ("retention_single", "retention_double")
_FULL_BENCHMARK_COMPRESSIONS = {
    "full": "none",
    "full_fastvggt": "fastvggt",
    "full_quantvggt": "quantvggt",
}
_POINT_STYLES: Dict[str, Dict[str, Any]] = {
    "upper": {
        "marker": "o",
        "label": "Org",
        "face": "color",
        "hatch": None,
        "zorder": 5,
    },
    "before": {
        "marker": "o",
        "label": "Ours",
        "face": "white",
        "hatch": "//////",
        "zorder": OURS_CURVE_ZORDER,
    },
    "after": {
        "marker": "o",
        "label": "Ours(FastVGGT)",
        "face": "white",
        "hatch": "//",
        "zorder": OURS_CURVE_ZORDER + 1.0,
    },
}
_TABLE_STATE_LABELS = {
    "upper": "Original upper",
    "full_fastvggt": "Full + FastVGGT",
    "full_quantvggt": "Full + QuantVGGT",
    "before": "Recovery",
    "after": "Recovery + FastVGGT",
}
_FULL_TABLE_STATES = frozenset({"upper", "full_fastvggt", "full_quantvggt"})
_TABLE_HEADERS = (
    "Backbone",
    "Pruned layers",
    "Result",
    *(METRIC_HEADERS.get(metric, metric) for metric in GT_MEAN_METRICS),
    "Parameters ↓",
    "Inference time (ms) ↓",
    "Peak VRAM (GiB) ↓",
)


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _point_metrics(point: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("raw_metric", "metrics_mean"):
        value = point.get(key, {})
        if isinstance(value, Mapping) and value:
            return dict(value)
    return {}


def _benchmark_datasets(root: Path, dataset: str) -> Tuple[str, ...]:
    if dataset in _DATASET_AVERAGE_GROUPS:
        return tuple(_DATASET_AVERAGE_GROUPS[dataset])
    if dataset == "mean":
        return tuple(
            sorted(
                {
                    path.parents[2].name
                    for path in Path(root).glob(
                        f"*/eval/testtime/{_BENCHMARK_FILENAME}"
                    )
                }
            )
        )
    return (str(dataset),)


def _benchmark_alias_matches(
    result: Mapping[str, Any], *, drop_count: int, prune_strategy: str
) -> bool:
    selection = result.get("selection", {})
    aliases = selection.get("aliases", ()) if isinstance(selection, Mapping) else ()
    for alias in aliases if isinstance(aliases, list) else ():
        if not isinstance(alias, Mapping):
            continue
        try:
            recorded_drop = int(alias.get("drop_count"))
        except (TypeError, ValueError):
            continue
        if recorded_drop == drop_count and str(
            alias.get("selected_strategy", "")
        ) == str(prune_strategy):
            return True
    return False


def _benchmark_compression_matches(
    result: Mapping[str, Any], *, state: str, compression_name: str
) -> bool:
    compression = result.get("compression", {})
    if not isinstance(compression, Mapping):
        return False
    if state == "before":
        return compression.get("name") == "none"
    match = _FASTVGGT_PATTERN.fullmatch(str(compression_name))
    if match is None or compression.get("name") != "fastvggt":
        return False
    expected_ratio = float(f"{match.group('whole')}.{match.group('fraction')}")
    try:
        return int(compression.get("merge_start")) == int(
            match.group("start")
        ) and math.isclose(float(compression.get("merge_ratio")), expected_ratio)
    except (TypeError, ValueError):
        return False


def _full_compression_output_tag(compression: Mapping[str, Any]) -> str:
    name = str(compression.get("name", "")).strip().lower()
    if name == "fastvggt":
        ratio = f"{float(compression.get('merge_ratio', 0.9)):g}".replace(".", "p")
        return f"fastvggt_m{int(compression.get('merge_start', 0))}_r{ratio}"
    if name == "quantvggt":
        return f"quantvggt_{str(compression.get('dtype', 'w4a4')).strip().lower()}"
    if name == "none":
        return ""
    raise ValueError(f"unsupported full-model compression record: {dict(compression)!r}")


def _matching_benchmark_result(
    payload: Mapping[str, Any],
    *,
    recovery_method: str,
    drop_count: int,
    selector: str,
    prune_strategy: str,
    state: str,
    compression_name: str,
) -> Mapping[str, Any] | None:
    expected_variant = "recovery" if state == "before" else "recovery_fastvggt"
    results = payload.get("results", {})
    for result in results.values() if isinstance(results, Mapping) else ():
        if not isinstance(result, Mapping):
            continue
        if (
            result.get("variant") != expected_variant
            or str(result.get("selector", "")) != selector
            or str(result.get("recovery_plan_name", "")) != recovery_method
            or not _benchmark_alias_matches(
                result,
                drop_count=drop_count,
                prune_strategy=prune_strategy,
            )
            or not _benchmark_compression_matches(
                result,
                state=state,
                compression_name=compression_name,
            )
        ):
            continue
        return result
    return None


def _load_benchmark_measurement(
    root: Path,
    *,
    recovery_method: str,
    dataset: str,
    drop_count: int,
    selector: str,
    prune_strategy: str,
    state: str,
    compression_name: str,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    """Aggregate exclusive-GPU scene records into forward latency."""
    expected_datasets = _benchmark_datasets(Path(root), dataset)
    available_datasets: List[str] = []
    latencies_ms: List[float] = []
    peak_allocated_gib: List[float] = []
    for dataset_name in expected_datasets:
        path = Path(root) / dataset_name / "eval" / "testtime" / _BENCHMARK_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or str(payload.get("dataset", "")) != dataset_name
        ):
            continue
        result = _matching_benchmark_result(
            payload,
            recovery_method=recovery_method,
            drop_count=drop_count,
            selector=selector,
            prune_strategy=prune_strategy,
            state=state,
            compression_name=compression_name,
        )
        if result is None:
            continue
        scenes = result.get("scenes", ())
        dataset_latencies: List[float] = []
        dataset_peaks: List[float] = []
        valid = isinstance(scenes, list) and bool(scenes)
        for scene in scenes if isinstance(scenes, list) else ():
            memory = scene.get("memory_gib", {}) if isinstance(scene, Mapping) else {}
            try:
                latency = float(scene["latency_ms"])
                frames = int(scene["num_frames"])
                peak = float(memory["peak_allocated"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if (
                not math.isfinite(latency)
                or latency <= 0.0
                or frames <= 0
                or not math.isfinite(peak)
                or peak < 0.0
            ):
                valid = False
                break
            dataset_latencies.append(latency)
            dataset_peaks.append(peak)
        if not valid:
            continue
        available_datasets.append(dataset_name)
        latencies_ms.extend(dataset_latencies)
        peak_allocated_gib.extend(dataset_peaks)

    warnings: List[str] = []
    if dataset in _DATASET_AVERAGE_GROUPS:
        warning = format_incomplete_average_warning(
            owner=(f"inference benchmark / {state} / {selector} / drop {drop_count}"),
            scope=dataset,
            expected=expected_datasets,
            available=available_datasets,
        )
        if warning:
            warnings.append(warning)
    elif not available_datasets:
        warnings.append(
            f"inference benchmark / {state} / {selector} / drop {drop_count}: "
            f"no matching result for {dataset}"
        )
    if not latencies_ms:
        return None, warnings
    return {
        "inference_time_ms": float(np.mean(latencies_ms)),
        "peak_allocated_gib": float(np.mean(peak_allocated_gib)),
        "benchmark_scene_count": len(latencies_ms),
        "benchmark_datasets": available_datasets,
        "timing_source": "exclusive_gpu_benchmark",
    }, warnings


def _load_full_benchmark_measurement(
    root: Path,
    *,
    dataset: str,
    variant: str = "full",
) -> Tuple[Dict[str, Any] | None, List[str]]:
    """Aggregate full-model records produced by the exclusive-GPU benchmark."""
    if variant not in _FULL_BENCHMARK_COMPRESSIONS:
        raise ValueError(f"unsupported full-model benchmark variant: {variant!r}")
    expected_compression = _FULL_BENCHMARK_COMPRESSIONS[variant]
    expected_datasets = _benchmark_datasets(Path(root), dataset)
    available_datasets: List[str] = []
    latencies_ms: List[float] = []
    peak_allocated_gib: List[float] = []
    compression_record: Dict[str, Any] | None = None
    for dataset_name in expected_datasets:
        path = Path(root) / dataset_name / "eval" / "testtime" / _BENCHMARK_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or str(payload.get("dataset", "")) != dataset_name
        ):
            continue
        results = payload.get("results", {})
        full_result = next(
            (
                result
                for result in (
                    results.values() if isinstance(results, Mapping) else ()
                )
                if isinstance(result, Mapping)
                and result.get("variant") == variant
                and str(result.get("selector", "")) == "full"
                and result.get("selection") is None
                and result.get("recovery_plan_name") is None
                and result.get("checkpoint") is None
                and isinstance(result.get("compression"), Mapping)
                and result["compression"].get("name") == expected_compression
                and (
                    compression_record is None
                    or _full_compression_output_tag(result["compression"])
                    == _full_compression_output_tag(compression_record)
                )
            ),
            None,
        )
        if full_result is None:
            continue
        if compression_record is None:
            compression_record = dict(full_result["compression"])
        scenes = full_result.get("scenes", ())
        dataset_latencies: List[float] = []
        dataset_peaks: List[float] = []
        valid = isinstance(scenes, list) and bool(scenes)
        for scene in scenes if isinstance(scenes, list) else ():
            memory = scene.get("memory_gib", {}) if isinstance(scene, Mapping) else {}
            try:
                latency = float(scene["latency_ms"])
                frames = int(scene["num_frames"])
                peak = float(memory["peak_allocated"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if (
                not math.isfinite(latency)
                or latency <= 0.0
                or frames <= 0
                or not math.isfinite(peak)
                or peak < 0.0
            ):
                valid = False
                break
            dataset_latencies.append(latency)
            dataset_peaks.append(peak)
        if not valid:
            continue
        available_datasets.append(dataset_name)
        latencies_ms.extend(dataset_latencies)
        peak_allocated_gib.extend(dataset_peaks)

    warnings: List[str] = []
    if dataset in _DATASET_AVERAGE_GROUPS:
        warning = format_incomplete_average_warning(
            owner=f"inference benchmark / {variant}",
            scope=dataset,
            expected=expected_datasets,
            available=available_datasets,
        )
        if warning:
            warnings.append(warning)
    elif not available_datasets:
        warnings.append(
            f"inference benchmark / {variant}: no matching result for {dataset}"
        )
    if not latencies_ms:
        return None, warnings
    return {
        "inference_time_ms": float(np.mean(latencies_ms)),
        "peak_allocated_gib": float(np.mean(peak_allocated_gib)),
        "benchmark_scene_count": len(latencies_ms),
        "benchmark_datasets": available_datasets,
        "timing_source": "exclusive_gpu_benchmark",
        "compression": compression_record or {"name": expected_compression},
    }, warnings


def _load_full_accuracy_metrics(
    root: Path,
    *,
    dataset: str,
    variant: str,
    compression: Mapping[str, Any],
) -> Tuple[Dict[str, float] | None, List[str]]:
    """Aggregate standalone full-model compression accuracy eval records."""
    if variant not in {"full_fastvggt", "full_quantvggt"}:
        raise ValueError(f"unsupported compressed accuracy variant: {variant!r}")
    expected_name = _FULL_BENCHMARK_COMPRESSIONS[variant]
    if str(compression.get("name", "")).strip().lower() != expected_name:
        raise ValueError(
            f"accuracy variant {variant!r} does not match compression {compression!r}"
        )
    compression_tag = _full_compression_output_tag(compression)
    run_name = f"upper_{compression_tag}"
    expected_datasets = _benchmark_datasets(Path(root), dataset)
    available_datasets: List[str] = []
    metric_samples: Dict[str, List[float]] = {
        metric: [] for metric in GT_MEAN_METRICS
    }
    for dataset_name in expected_datasets:
        path = Path(root) / dataset_name / "eval" / run_name / "eval_record_v3.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, Mapping):
            continue
        extra = record.get("extra", {})
        recorded_compression = (
            extra.get("inference_compression", {})
            if isinstance(extra, Mapping)
            else {}
        )
        average = record.get("average", {})
        try:
            recorded_compression_tag = (
                _full_compression_output_tag(recorded_compression)
                if isinstance(recorded_compression, Mapping)
                else ""
            )
        except (TypeError, ValueError):
            continue
        if (
            record.get("schema_version") != "v3"
            or str(record.get("dataset", "")) != dataset_name
            or not isinstance(extra, Mapping)
            or str(extra.get("eval_mode", "")).strip().lower() != "upper"
            or not isinstance(recorded_compression, Mapping)
            or recorded_compression_tag != compression_tag
            or not isinstance(average, dict)
        ):
            continue
        values = {
            metric: metric_value(average, metric) for metric in GT_MEAN_METRICS
        }
        if any(value is None for value in values.values()):
            continue
        available_datasets.append(dataset_name)
        for metric, value in values.items():
            assert value is not None
            metric_samples[metric].append(float(value))

    warnings: List[str] = []
    owner = f"accuracy eval / {variant}"
    if dataset in _DATASET_AVERAGE_GROUPS:
        warning = format_incomplete_average_warning(
            owner=owner,
            scope=dataset,
            expected=expected_datasets,
            available=available_datasets,
        )
        if warning:
            warnings.append(warning)
    elif not available_datasets:
        warnings.append(f"{owner}: no matching result for {dataset}")
    if not available_datasets:
        return None, warnings
    return {
        metric: float(np.mean(values)) for metric, values in metric_samples.items()
    }, warnings


def _matching_recovery_point(
    payload: Mapping[str, Any],
    *,
    method: str,
    drop_count: int,
    selector: str,
) -> Dict[str, Any] | None:
    strategies = payload.get("strategies", {})
    strategy = strategies.get(method, {}) if isinstance(strategies, Mapping) else {}
    points = strategy.get("points", {}) if isinstance(strategy, Mapping) else {}
    point = points.get(str(int(drop_count))) if isinstance(points, Mapping) else None
    if not isinstance(point, Mapping) or not point.get("available", False):
        return None
    # Dataset/group averaging intentionally drops non-numeric selector fields.
    # The payload has already been restricted to one pruning variant, so an
    # absent selector is safe; when present (single-dataset view), keep the
    # exact-match guard against stale recovery output.
    recorded_selector = str(point.get("selector", ""))
    if recorded_selector and recorded_selector != str(selector):
        return None
    return dict(point)


def _load_backbone_tradeoff_points(
    root: Path,
    *,
    backbone: str,
    recovery_method: str,
    dataset: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load Org and every matched Retention/FastVGGT point pair."""
    root = Path(root)
    bundle = load_prune_plan_bundle(root)
    messages: List[str] = []

    pruning = collect_pruning_eval_payload(
        experiment_root=root,
        prune_plan_root=root,
    )
    pruning_dataset = _selected_dataset(pruning, dataset)
    upper = pruning_dataset.get("upper_average", {})
    upper_metrics = dict(upper) if isinstance(upper, Mapping) else {}

    before_payloads: Dict[str, Dict[str, Any]] = {}
    compressed_candidates: List[Dict[str, Any]] = []
    for variant in _RETENTION_VARIANTS:
        recovery = collect_recovery_eval_payload(
            experiment_root=root,
            prune_strategy=variant,
        )
        selected = _selected_dataset(recovery, dataset)
        before_payloads[variant] = selected
        compressed, warnings = _load_testtime_points(
            root,
            recovery_method=recovery_method,
            prune_strategy=variant,
            dataset=dataset,
        )
        messages.extend(f"{backbone}: {warning}" for warning in warnings)
        compressed_candidates.extend(
            {**point, "prune_strategy": variant}
            for point in compressed
            if str(point.get("compression", "")).lower().startswith(_FASTVGGT_PREFIX)
        )

    filtered, failures = _filter_testtime_to_family_winners(
        {"testtime_points": compressed_candidates},
        root=root,
        recovery_method=recovery_method,
        families=("retention",),
        selection_datasets=_DATASET_AVERAGE_GROUPS["training average"],
    )
    messages.extend(f"{backbone}: {failure}" for failure in failures)
    compressed_points = sorted(
        (
            point
            for point in filtered.get("testtime_points", ())
            if isinstance(point, dict)
        ),
        key=lambda point: (
            int(point.get("drop_count", 10**9)),
            str(point.get("selector", "")),
        ),
    )

    selectors_by_drop: Dict[int, List[str]] = {}
    for point in compressed_points:
        drop_count = point.get("drop_count")
        selector = str(point.get("selector", ""))
        if isinstance(drop_count, int) and selector:
            selectors_by_drop.setdefault(drop_count, []).append(selector)

    parameter_series, parameter_warnings = compute_parameter_series_for_backbone(
        backbone,
        bundle,
        selectors_by_drop,
    )
    messages.extend(f"{backbone}: {warning}" for warning in parameter_warnings)

    result: List[Dict[str, Any]] = []
    upper_params = parameter_series.get(0)
    upper_measurement, upper_warnings = _load_full_benchmark_measurement(
        root,
        dataset=dataset,
    )
    messages.extend(f"{backbone}: {warning}" for warning in upper_warnings)
    if upper_metrics and upper_params is not None and upper_measurement is not None:
        result.append(
            {
                "backbone": backbone,
                "state": "upper",
                "drop_count": 0,
                "selector": "upper",
                **upper_measurement,
                "param_count": int(upper_params[0]),
                "raw_metric": upper_metrics,
            }
        )
    else:
        messages.append(
            f"{backbone}: Upper metrics, parameter count, or full benchmark is unavailable"
        )

    standalone_variants = ["full_fastvggt"]
    if backbone == "vggt":
        standalone_variants.append("full_quantvggt")
    if upper_params is not None:
        for standalone_variant in standalone_variants:
            measurement, benchmark_warnings = _load_full_benchmark_measurement(
                root,
                dataset=dataset,
                variant=standalone_variant,
            )
            messages.extend(
                f"{backbone}: {warning}" for warning in benchmark_warnings
            )
            if measurement is None:
                continue
            compression = measurement.get("compression", {})
            if not isinstance(compression, Mapping):
                compression = {}
            accuracy, accuracy_warnings = _load_full_accuracy_metrics(
                root,
                dataset=dataset,
                variant=standalone_variant,
                compression=compression,
            )
            messages.extend(
                f"{backbone}: {warning}" for warning in accuracy_warnings
            )
            result.append(
                {
                    "backbone": backbone,
                    "state": standalone_variant,
                    "drop_count": 0,
                    "selector": "full",
                    **measurement,
                    "param_count": int(upper_params[0]),
                    "raw_metric": accuracy or {},
                }
            )

    for after in compressed_points:
        drop_count = after.get("drop_count")
        selector = str(after.get("selector", ""))
        variant = str(after.get("prune_strategy", ""))
        if not isinstance(drop_count, int) or not selector:
            continue
        parameter_pair = parameter_series.get(drop_count)
        if parameter_pair is None:
            messages.append(
                f"{backbone}: parameter count unavailable for drop {drop_count} ({selector})"
            )
            continue
        before = _matching_recovery_point(
            before_payloads.get(variant, {}),
            method=recovery_method,
            drop_count=drop_count,
            selector=selector,
        )
        if before is None:
            messages.append(
                f"{backbone}: uncompressed recovery point missing for drop "
                f"{drop_count} ({selector})"
            )
            continue
        before_measurement, benchmark_warnings = _load_benchmark_measurement(
            root,
            recovery_method=recovery_method,
            dataset=dataset,
            drop_count=drop_count,
            selector=selector,
            prune_strategy=variant,
            state="before",
            compression_name="none",
        )
        messages.extend(f"{backbone}: {warning}" for warning in benchmark_warnings)
        after_measurement, benchmark_warnings = _load_benchmark_measurement(
            root,
            recovery_method=recovery_method,
            dataset=dataset,
            drop_count=drop_count,
            selector=selector,
            prune_strategy=variant,
            state="after",
            compression_name=str(after.get("compression", "")),
        )
        messages.extend(f"{backbone}: {warning}" for warning in benchmark_warnings)
        if before_measurement is None or after_measurement is None:
            messages.append(
                f"{backbone}: exclusive benchmark pair unavailable for drop "
                f"{drop_count} ({selector}); the table falls back to synchronized "
                "eval timing/VRAM and the scatter omits the unbenchmarked point(s)"
            )
        common = {
            "backbone": backbone,
            "drop_count": drop_count,
            "selector": selector,
            "prune_strategy": variant,
            # FastVGGT patches attention execution but introduces no parameters.
            "param_count": int(parameter_pair[0]),
        }
        result.append(
            {
                **common,
                **(before_measurement or {}),
                "state": "before",
                "raw_metric": _point_metrics(before),
            }
        )
        result.append(
            {
                **common,
                **(after_measurement or {}),
                "state": "after",
                "raw_metric": _point_metrics(after),
            }
        )

    interval_count = len(
        {
            (point["drop_count"], point["selector"])
            for point in result
            if point["state"] == "after"
        }
    )
    messages.append(
        f"{backbone}: loaded Upper and {interval_count} Retention interval pair(s)"
    )
    return result, list(dict.fromkeys(messages))


def _scatter_area(param_count: int, max_param_count: int) -> float:
    if max_param_count <= 0:
        return 120.0
    # Matplotlib's `s` is marker area. Squaring the parameter ratio makes marker
    # radius, rather than area, proportional to the model parameter count.
    ratio = float(param_count) / float(max_param_count)
    return max(18.0, 720.0 * ratio * ratio)


def _format_param_count(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.0f}M"


def _render_tradeoff_figure(points: Sequence[Mapping[str, Any]]) -> Any | None:
    timed_points = [
        dict(point)
        for point in points
        if str(point.get("state", "")) in _POINT_STYLES
        if isinstance(point.get("param_count"), int)
        and int(point["param_count"]) > 0
        and isinstance(point.get("inference_time_ms"), (int, float))
        and math.isfinite(float(point["inference_time_ms"]))
        and float(point["inference_time_ms"]) > 0.0
    ]
    if not timed_points:
        return None
    max_params = max(int(point["param_count"]) for point in timed_points)
    backbones = list(dict.fromkeys(str(point["backbone"]) for point in timed_points))
    colors = {
        backbone: _BACKBONE_COLORS[index % len(_BACKBONE_COLORS)]
        for index, backbone in enumerate(backbones)
    }

    fig, axes = plt.subplots(
        1,
        len(GT_MEAN_METRICS),
        figsize=(35.0, 6.2),
        dpi=130,
        squeeze=False,
    )
    axes_row = axes[0]
    any_data = False
    # Large bubbles are drawn first so smaller parameter-count points remain
    # visible when nearby measurements overlap.
    drawing_order = sorted(
        timed_points,
        key=lambda point: (
            -int(point["param_count"]),
            str(point.get("state", "")),
        ),
    )
    for metric, ax in zip(GT_MEAN_METRICS, axes_row):
        for point in drawing_order:
            raw = point.get("raw_metric", {})
            value = metric_value(raw, metric) if isinstance(raw, dict) else None
            if value is None or not math.isfinite(float(value)):
                continue
            state = str(point.get("state", ""))
            style = _POINT_STYLES.get(state)
            if style is None:
                continue
            color = colors[str(point["backbone"])]
            facecolor = color if style["face"] == "color" else style["face"]
            ax.scatter(
                [float(point["inference_time_ms"])],
                [float(value)],
                s=_scatter_area(int(point["param_count"]), max_params),
                marker=style["marker"],
                facecolors=facecolor,
                edgecolors=color,
                linewidths=1.8,
                hatch=style["hatch"],
                alpha=0.88,
                zorder=style["zorder"],
            )
            any_data = True
        ax.set_title(METRIC_HEADERS.get(metric, metric), fontsize=14, fontweight="bold")
        ax.set_xlabel("Inference time (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("")
        ax.set_ylim(bottom=0.0)
        ax.set_box_aspect(1.0)
        ax.margins(x=0.1)
        ax.grid(True, alpha=0.24)
        ax.tick_params(axis="both", labelsize=10)
    if not any_data:
        plt.close(fig)
        return None

    backbone_handles = [
        Line2D(
            [],
            [],
            color=colors[backbone],
            marker="o",
            linestyle="None",
            markersize=9,
            label=backbone_display_name(backbone),
        )
        for backbone in backbones
    ]
    state_handles = [
        axes_row[0].scatter(
            [],
            [],
            s=120.0,
            marker="o",
            facecolors="#777777",
            edgecolors="#333333",
            linewidths=1.5,
            label=_POINT_STYLES["upper"]["label"],
        ),
        axes_row[0].scatter(
            [],
            [],
            s=120.0,
            marker="o",
            facecolors="white",
            edgecolors="#333333",
            linewidths=1.5,
            hatch=_POINT_STYLES["before"]["hatch"],
            label=_POINT_STYLES["before"]["label"],
        ),
        axes_row[0].scatter(
            [],
            [],
            s=120.0,
            marker="o",
            facecolors="white",
            edgecolors="#333333",
            linewidths=1.5,
            hatch=_POINT_STYLES["after"]["hatch"],
            label=_POINT_STYLES["after"]["label"],
        ),
    ]
    unique_params = sorted({int(point["param_count"]) for point in timed_points})
    reference_indices = sorted({0, len(unique_params) // 2, len(unique_params) - 1})
    size_handles = [
        axes_row[0].scatter(
            [],
            [],
            s=_scatter_area(unique_params[index], max_params),
            facecolors="none",
            edgecolors="#444444",
            linewidths=1.3,
            label=f"{_format_param_count(unique_params[index])} params",
        )
        for index in reference_indices
    ]
    fig.legend(
        handles=[*backbone_handles, *state_handles, *size_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(9, len(backbone_handles) + len(state_handles) + len(size_handles)),
        fontsize=11,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def _fastvggt_alias_candidates(
    root: Path,
    *,
    recovery_method: str,
) -> Dict[Tuple[int, str], set[Tuple[str, str]]]:
    """Collect physical FastVGGT aliases from every successfully published dataset."""
    candidates: Dict[Tuple[int, str], set[Tuple[str, str]]] = {}
    for path in Path(root).glob(f"*/eval/testtime/{_BENCHMARK_FILENAME}"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results", {}) if isinstance(payload, Mapping) else {}
        for result in results.values() if isinstance(results, Mapping) else ():
            if (
                not isinstance(result, Mapping)
                or result.get("variant") != "recovery_fastvggt"
                or str(result.get("recovery_plan_name", "")) != recovery_method
            ):
                continue
            selector = str(result.get("selector", ""))
            selection = result.get("selection", {})
            aliases = (
                selection.get("aliases", ()) if isinstance(selection, Mapping) else ()
            )
            for alias in aliases if isinstance(aliases, list) else ():
                if not isinstance(alias, Mapping):
                    continue
                family_raw = str(alias.get("family", ""))
                family = "CKA" if family_raw.lower() == "cka" else family_raw.lower()
                if family not in {"CKA", "retention"}:
                    continue
                try:
                    drop_count = int(alias.get("drop_count"))
                except (TypeError, ValueError):
                    continue
                strategy = str(alias.get("selected_strategy", ""))
                if selector and strategy:
                    candidates.setdefault((drop_count, family), set()).add(
                        (strategy, selector)
                    )
    return candidates


def _fastvggt_family_aliases(
    root: Path,
    *,
    recovery_method: str,
    family: str,
) -> Dict[int, Dict[str, str]]:
    """Resolve every unambiguous benchmarked FastVGGT alias for one family."""
    candidates = _fastvggt_alias_candidates(
        root,
        recovery_method=recovery_method,
    )
    resolved: Dict[int, Dict[str, str]] = {}
    for (drop_count, candidate_family), values in sorted(candidates.items()):
        if candidate_family != family:
            continue
        if len(values) != 1:
            raise ValueError(
                f"inconsistent FastVGGT aliases for drop {drop_count} / {family}: "
                f"{sorted(values)}"
            )
        strategy, selector = next(iter(values))
        resolved[int(drop_count)] = {
            "selected_strategy": strategy,
            "selector": selector,
        }
    if not resolved:
        raise FileNotFoundError(f"no FastVGGT aliases found for {family}")
    return resolved


def _minimum_fastvggt_aliases(
    root: Path,
    *,
    recovery_method: str,
) -> Tuple[int, Dict[str, Dict[str, str]]]:
    """Resolve CKA/Retention aliases at the smallest benchmarked FastVGGT drop."""
    candidates = _fastvggt_alias_candidates(
        root,
        recovery_method=recovery_method,
    )
    complete_drops = sorted(
        {
            drop_count
            for drop_count, family in candidates
            if family == "CKA" and (drop_count, "retention") in candidates
        }
    )
    if not complete_drops:
        raise FileNotFoundError("no FastVGGT drop has both CKA and Retention aliases")
    drop_count = complete_drops[0]
    resolved: Dict[str, Dict[str, str]] = {}
    for family in ("CKA", "retention"):
        values = candidates[(drop_count, family)]
        if len(values) != 1:
            raise ValueError(
                f"inconsistent FastVGGT aliases for drop {drop_count} / {family}: "
                f"{sorted(values)}"
            )
        strategy, selector = next(iter(values))
        resolved[family] = {
            "selected_strategy": strategy,
            "selector": selector,
        }
    return drop_count, resolved


def _figure9_family_winner_curves(
    root: Path,
    *,
    recovery_method: str,
    dataset: str,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Load Figure 9's independently collapsed Single/Double family curves."""
    pruning = collect_pruning_eval_payload(
        experiment_root=root,
        prune_plan_root=root,
    )
    merged: Dict[str, Any] = {
        "upper_average": {},
        "normalized_upper_average": {},
        "normalization_bounds": {},
        "strategies": {},
        "testtime_points": [],
    }
    for family in ("CKA", "retention"):
        for structure in ("single", "double"):
            variant = f"{family}_{structure}"
            recovery = collect_recovery_eval_payload(
                experiment_root=root,
                prune_strategy=variant,
            )
            payload = _selected_dataset(
                add_recovery_comparison_strategies(recovery, pruning, variant),
                dataset,
            )
            for key in (
                "upper_average",
                "normalized_upper_average",
                "normalization_bounds",
            ):
                value = payload.get(key, {}) if isinstance(payload, Mapping) else {}
                if not merged[key] and isinstance(value, Mapping):
                    merged[key] = dict(value)
            strategies = (
                payload.get("strategies", {}) if isinstance(payload, Mapping) else {}
            )
            method_payload = (
                strategies.get(recovery_method)
                if isinstance(strategies, Mapping)
                else None
            )
            if isinstance(method_payload, Mapping):
                merged["strategies"][f"{recovery_method}_{variant}"] = dict(
                    method_payload
                )

    winners: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for family in ("CKA", "retention"):
        calibration_selection, _selection_failures = (
            _main_calibration_structure_selection(
                root=root,
                recovery_method=recovery_method,
                family=family,
                selection_datasets=_DATASET_AVERAGE_GROUPS["training average"],
            )
        )
        selected, _keys = _collapse_structure_min_strategy(
            merged,
            method=recovery_method,
            family=family,
            calibration_selection=calibration_selection,
        )
        strategies = selected.get("strategies", {})
        strategy = (
            strategies.get(f"{recovery_method}_{family}_min", {})
            if isinstance(strategies, Mapping)
            else {}
        )
        points = strategy.get("points", {}) if isinstance(strategy, Mapping) else {}
        winners[family] = {
            str(drop): dict(point)
            for drop, point in (points.items() if isinstance(points, Mapping) else ())
            if isinstance(point, Mapping) and point.get("available", False)
        }
    return winners, pruning


def _figure9_family_winners(
    root: Path,
    *,
    recovery_method: str,
    dataset: str,
    drop_count: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load Figure 9's independent Single/Double winner at one drop."""
    curves, pruning = _figure9_family_winner_curves(
        root,
        recovery_method=recovery_method,
        dataset=dataset,
    )
    winners: Dict[str, Dict[str, Any]] = {}
    for family in ("CKA", "retention"):
        point = curves.get(family, {}).get(str(drop_count))
        if point is None:
            raise FileNotFoundError(
                f"Figure 9 has no {family} winner at drop {drop_count}"
            )
        winners[family] = point
    return winners, pruning


def _load_backbone_radar_points(
    root: Path,
    *,
    backbone: str,
    recovery_method: str,
    dataset: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load Org and the minimum-span Retention point plus all-span axis maxima."""
    root = Path(root)
    messages: List[str] = []
    aliases_by_drop = _fastvggt_family_aliases(
        root,
        recovery_method=recovery_method,
        family="retention",
    )
    winner_curves, pruning = _figure9_family_winner_curves(
        root,
        recovery_method=recovery_method,
        dataset=dataset,
    )
    retention_curve = winner_curves.get("retention", {})
    for drop_count, alias in aliases_by_drop.items():
        point = retention_curve.get(str(drop_count))
        if point is None:
            raise FileNotFoundError(
                f"Figure 9 has no retention winner at drop {drop_count}"
            )
        expected_strategy = alias["selected_strategy"]
        actual_strategy = str(point.get("min_prune_strategy", ""))
        if actual_strategy != expected_strategy:
            raise ValueError(
                f"Figure 9 winner {actual_strategy!r} does not match benchmark alias "
                f"{expected_strategy!r} for retention drop {drop_count}"
            )

    selector_slots = {
        index: [alias["selector"]]
        for index, alias in enumerate(aliases_by_drop.values(), start=1)
    }
    parameter_series, parameter_warnings = compute_parameter_series_for_backbone(
        backbone,
        load_prune_plan_bundle(root),
        selector_slots,
    )
    messages.extend(f"{backbone}: {warning}" for warning in parameter_warnings)

    selected_pruning = _selected_dataset(pruning, dataset)
    upper = selected_pruning.get("upper_average", {})
    upper_metrics = dict(upper) if isinstance(upper, Mapping) else {}
    upper_measurement, upper_warnings = _load_full_benchmark_measurement(
        root,
        dataset=dataset,
    )
    messages.extend(f"{backbone}: {warning}" for warning in upper_warnings)
    retention_points: List[Dict[str, Any]] = []
    for slot, (drop_count, alias) in enumerate(aliases_by_drop.items(), start=1):
        measurement, warnings = _load_benchmark_measurement(
            root,
            recovery_method=recovery_method,
            dataset=dataset,
            drop_count=drop_count,
            selector=alias["selector"],
            prune_strategy=alias["selected_strategy"],
            state="before",
            compression_name="none",
        )
        messages.extend(f"{backbone}: {warning}" for warning in warnings)
        parameter_pair = parameter_series.get(slot)
        if measurement is None or parameter_pair is None:
            messages.append(
                f"{backbone}: radar retention point unavailable at drop {drop_count}"
            )
            continue
        retention_points.append(
            {
                "backbone": backbone,
                "state": "before",
                "radar_family": "retention",
                "drop_count": drop_count,
                "selector": alias["selector"],
                "prune_strategy": alias["selected_strategy"],
                **measurement,
                "param_count": int(parameter_pair[0]),
                "raw_metric": _point_metrics(retention_curve[str(drop_count)]),
            }
        )
    if not retention_points:
        raise FileNotFoundError("no complete Retention span is available for radar bounds")

    axis_maxima: Dict[str, float] = {}
    for field, _label in _RADAR_FIELDS:
        values = [_radar_raw_value(point, field) for point in retention_points]
        if any(value is None for value in values):
            raise ValueError(f"Retention span values are incomplete for {field}")
        maximum = max(float(value) for value in values if value is not None)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"invalid Retention span maximum for {field}: {maximum}")
        axis_maxima[field] = maximum

    displayed_ours = min(retention_points, key=lambda point: int(point["drop_count"]))
    result: List[Dict[str, Any]] = []
    upper_params = parameter_series.get(0)
    if upper_metrics and upper_measurement is not None and upper_params is not None:
        result.append(
            {
                "backbone": backbone,
                "state": "upper",
                "radar_family": "Org",
                "drop_count": 0,
                "selector": "upper",
                **upper_measurement,
                "param_count": int(upper_params[0]),
                "raw_metric": upper_metrics,
                "radar_axis_maxima": axis_maxima,
            }
        )
    result.append({**displayed_ours, "radar_axis_maxima": axis_maxima})
    messages.append(
        f"{backbone}: radar displays Retention drop {displayed_ours['drop_count']} and "
        f"uses zero-to-maximum bounds from discovered Retention drops "
        f"{[int(point['drop_count']) for point in retention_points]}"
    )
    return result, list(dict.fromkeys(messages))


_RADAR_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("ape", METRIC_HEADERS["ape"]),
    ("are", METRIC_HEADERS["are"]),
    ("rpe_trans", METRIC_HEADERS["rpe_trans"]),
    ("rpe_rot", METRIC_HEADERS["rpe_rot"]),
    ("acc", METRIC_HEADERS["acc"]),
    ("comp", METRIC_HEADERS["comp"]),
    ("cd", METRIC_HEADERS["cd"]),
    ("inference_time_ms", "Inference\nTime (ms) ↓"),
    ("peak_allocated_gib", "Peak VRAM\n(GiB) ↓"),
    ("param_count", "Parameters (M) ↓"),
)
_RADAR_GROUPS: Tuple[Tuple[str, int, int, str], ...] = (
    ("Estimation metrics", 0, 4, "#4c78a8"),
    ("Point-cloud metrics", 4, 7, "#59a14f"),
    ("Efficiency metrics", 7, 10, "#f28e2b"),
)


def _radar_raw_value(point: Mapping[str, Any], field: str) -> float | None:
    if field in {"inference_time_ms", "param_count", "peak_allocated_gib"}:
        value = point.get(field)
        if field == "peak_allocated_gib" and value is None:
            raw = point.get("raw_metric", {})
            value = raw.get("infer_peak_vram_gb") if isinstance(raw, Mapping) else None
    else:
        raw = point.get("raw_metric", {})
        value = metric_value(raw, field) if isinstance(raw, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _render_tradeoff_radar_figure(
    points: Sequence[Mapping[str, Any]],
) -> Any | None:
    """Render the selected backbones' full models on one shared-scale radar."""
    complete: List[Dict[str, Any]] = []
    for point in points:
        state = str(point.get("state", ""))
        if state != "upper":
            continue
        values = [_radar_raw_value(point, field) for field, _label in _RADAR_FIELDS]
        if any(value is None for value in values):
            continue
        complete.append({**dict(point), "_radar_raw": [float(v) for v in values]})
    if not complete:
        return None

    raw_matrix = np.stack(
        [np.asarray(point["_radar_raw"], dtype=float) for point in complete], axis=0
    )
    shared_axis_minima = np.min(raw_matrix, axis=0)
    shared_axis_maxima = np.max(raw_matrix, axis=0)
    shared_axis_ranges = shared_axis_maxima - shared_axis_minima
    for point in complete:
        raw_values = np.asarray(point.pop("_radar_raw"), dtype=float)
        normalized = np.full_like(raw_values, 0.5, dtype=float)
        varying = shared_axis_ranges > 0.0
        normalized[varying] = 0.5 + 0.4 * (
            (raw_values[varying] - shared_axis_minima[varying])
            / shared_axis_ranges[varying]
        )
        point["radar_score"] = normalized.clip(0.5, 0.9)

    backbones = list(dict.fromkeys(str(point["backbone"]) for point in complete))
    colors = {
        backbone: _BACKBONE_COLORS[index % len(_BACKBONE_COLORS)]
        for index, backbone in enumerate(backbones)
    }
    axis_count = len(_RADAR_FIELDS)
    step = 2.0 * np.pi / axis_count
    angles = np.linspace(0.0, 2.0 * np.pi, axis_count, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(
        figsize=(8.4, 7.6),
        dpi=130,
        subplot_kw={"projection": "polar"},
    )
    for _group_name, start, stop, group_color in _RADAR_GROUPS:
        center = float(np.mean(angles[start:stop]))
        ax.bar(
            center,
            1.0,
            width=step * (stop - start),
            bottom=0.0,
            color=group_color,
            alpha=0.13,
            edgecolor="none",
            linewidth=0.0,
            align="center",
            zorder=0,
        )

    for point in complete:
        scores = np.asarray(point["radar_score"], dtype=float)
        closed_scores = np.concatenate([scores, scores[:1]])
        color = colors[str(point["backbone"])]
        ax.plot(
            closed_angles,
            closed_scores,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            alpha=0.94,
            label=backbone_display_name(str(point["backbone"])),
            zorder=3,
        )

    group_by_axis = {
        index: color
        for _name, start, stop, color in _RADAR_GROUPS
        for index in range(start, stop)
    }
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    _gridlines, theta_labels = ax.set_thetagrids(
        np.degrees(angles),
        [label for _field, label in _RADAR_FIELDS],
        fontsize=10,
    )
    for index, label in enumerate(theta_labels):
        label.set_fontweight("bold")
        label.set_color(group_by_axis[index])
        label.set_linespacing(1.05)
    ax.tick_params(axis="x", pad=28)
    for index in (0, 5):
        ax.xaxis.majorTicks[index].set_pad(15)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_yticklabels(("", "", "", "", ""))
    ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.48)
    ax.spines["polar"].set_color("#b8b8b8")
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_alpha(0.48)

    fig.legend(
        handles=[
            Line2D(
                [], [], color=colors[backbone], linewidth=2.4, marker="o",
                markersize=4, label=backbone_display_name(backbone),
            )
            for backbone in backbones
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(backbones),
        prop={"size": 11, "weight": "bold"},
        frameon=True,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor=color, alpha=0.32, label=name)
            for name, _start, _stop, color in _RADAR_GROUPS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(_RADAR_GROUPS),
        fontsize=10,
        frameon=False,
    )
    fig.text(
        0.5,
        0.018,
        "Per-axis scaling: best = 0.50 · worst = 0.90 · inner is better",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    return fig


def _collect_tradeoff_points(
    output_root: Path,
    backbones: Sequence[str],
    method: str,
    dataset: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    points: List[Dict[str, Any]] = []
    messages: List[str] = []
    for backbone in backbones:
        try:
            loaded, notes = _load_backbone_tradeoff_points(
                Path(output_root) / backbone,
                backbone=backbone,
                recovery_method=method,
                dataset=dataset,
            )
            points.extend(loaded)
            messages.extend(notes)
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    return points, messages


def _collect_radar_points(
    output_root: Path,
    backbones: Sequence[str],
    method: str,
    dataset: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    points: List[Dict[str, Any]] = []
    messages: List[str] = []
    for backbone in backbones:
        try:
            loaded, notes = _load_backbone_radar_points(
                Path(output_root) / backbone,
                backbone=backbone,
                recovery_method=method,
                dataset=dataset,
            )
            points.extend(loaded)
            messages.extend(notes)
        except Exception as exc:
            messages.append(f"{backbone} radar: {type(exc).__name__}: {exc}")
    return points, messages


def _figure_image(fig: Any | None) -> np.ndarray | None:
    if fig is None:
        return None
    try:
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)


def _tradeoff_table_rows(
    points: Sequence[Mapping[str, Any]],
) -> List[List[Any]]:
    """Format full-model and recovery benchmark points for two-decimal display."""
    backbone_order = {
        backbone: index
        for index, backbone in enumerate(
            dict.fromkeys(str(point.get("backbone", "")) for point in points)
        )
    }
    state_order = {
        "upper": 0,
        "full_fastvggt": 1,
        "full_quantvggt": 2,
        "before": 3,
        "after": 4,
    }
    ordered_points = sorted(
        (
            point
            for point in points
            if str(point.get("state", "")) in _TABLE_STATE_LABELS
        ),
        key=lambda point: (
            backbone_order.get(str(point.get("backbone", "")), len(backbone_order)),
            0 if str(point.get("state", "")) in _FULL_TABLE_STATES else 1,
            state_order[str(point.get("state", ""))]
            if str(point.get("state", "")) in _FULL_TABLE_STATES
            else 0,
            int(point.get("drop_count", 10**9)),
            str(point.get("selector", "")),
            state_order[str(point.get("state", ""))],
        ),
    )

    rows: List[List[Any]] = []
    for point in ordered_points:
        state = str(point["state"])
        raw = point.get("raw_metric", {})
        raw_metrics = raw if isinstance(raw, Mapping) else {}
        metric_values = [metric_value(raw_metrics, metric) for metric in GT_MEAN_METRICS]
        inference_time_ms = point.get(
            "inference_time_ms", raw_metrics.get("inference_time_ms")
        )
        peak_vram_gib = point.get(
            "peak_allocated_gib", raw_metrics.get("infer_peak_vram_gb")
        )

        def formatted(value: Any) -> str:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return "—"
            return f"{float(value):.2f}"

        drop_count = (
            "—" if state in _FULL_TABLE_STATES else str(point.get("drop_count", ""))
        )
        rows.append(
            [
                backbone_display_name(str(point.get("backbone", ""))),
                drop_count,
                _TABLE_STATE_LABELS[state],
                *(formatted(value) for value in metric_values),
                _format_param_count(int(point["param_count"]))
                if isinstance(point.get("param_count"), int)
                else "—",
                formatted(inference_time_ms),
                formatted(peak_vram_gib),
            ]
        )
    return rows


def _validated_selection(
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
) -> Tuple[List[str], str, str | None]:
    backbones = list(dict.fromkeys(str(value) for value in (selected_backbones or ())))
    method = str(selected_method or "").strip()
    if not backbones:
        return [], method, "Select at least one backbone."
    if not method:
        return backbones, method, "Select one recovery method."
    return backbones, method, None


def render_recovery_speed_accuracy(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "testing average",
) -> Tuple[np.ndarray | None, str]:
    backbones, method, error = _validated_selection(selected_backbones, selected_method)
    if error:
        return None, error
    points, messages = _collect_tradeoff_points(output_root, backbones, method, dataset)

    fig = _render_tradeoff_figure(points)
    if fig is None:
        return None, "\n".join([*messages, "No matching data to plot."])
    image = _figure_image(fig)
    timed_point_count = sum(
        isinstance(point.get("inference_time_ms"), (int, float))
        and math.isfinite(float(point["inference_time_ms"]))
        and float(point["inference_time_ms"]) > 0.0
        for point in points
    )
    messages.append(
        f"Rendered {timed_point_count} point(s). Ours timings use the exclusive-GPU "
        "CUDA-event "
        "benchmark; Org timing uses the legacy synchronized eval record. Bubble radius "
        "is proportional to full model parameter count."
    )
    return image, "\n".join(messages)


def _render_recovery_speed_accuracy_outputs(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_method: str | None,
    dataset: str = "same-dataset testing average",
) -> Tuple[np.ndarray | None, List[List[Any]], str]:
    backbones, method, error = _validated_selection(selected_backbones, selected_method)
    if error:
        return None, [], error
    points, messages = _collect_tradeoff_points(output_root, backbones, method, dataset)
    table_rows = _tradeoff_table_rows(points)
    radar_points = [point for point in points if point.get("state") == "upper"]
    radar = _figure_image(_render_tradeoff_radar_figure(radar_points))
    if radar is None:
        messages.append("No complete full-model nine-axis radar data to plot.")
    messages.append(
        f"Rendered {len(radar_points)} full-model radar point(s) and "
        f"{len(table_rows)} table row(s). On every radar axis, the best selected "
        "full model maps to 0.5 and the worst maps to 0.9; inner is better."
    )
    return radar, table_rows, "\n".join(messages)


def build_recovery_speed_accuracy_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    default_method = (
        "recovery_ghost_interval_sequential_autogroup"
        if "recovery_ghost_interval_sequential_autogroup" in methods
        else methods[0]
        if methods
        else None
    )
    dataset_groups = [
        "training average",
        "same-dataset testing average",
        "cross-dataset testing average",
    ]
    default_dataset = "same-dataset testing average"

    def redraw(selected_backbones, selected_method, dataset):
        return _render_recovery_speed_accuracy_outputs(
            Path(output_root), selected_backbones, selected_method, dataset
        )

    tab_label = "Recovery Speed–Accuracy"
    with gr.Tab(tab_label, id="recovery_speed_accuracy") as tab:
        gr.Markdown(f"## {tab_label}")
        gr.Markdown(
            "The radar overlays the selected backbones' full models on one shared scale. "
            "The axes are grouped into estimation, point-cloud, and efficiency metrics. "
            "For each axis, the best selected full model maps to 0.5 and the worst maps "
            "to 0.9, so inner is better. "
            "The table reports every Org, standalone full-model FastVGGT/QuantVGGT, "
            "Recovery, and Recovery + FastVGGT result. Standalone compression rows "
            "combine their upper-compression accuracy eval with the matching forward-only "
            "benchmark; accuracy cells remain blank until that eval is complete. Table "
            "values prefer the exclusive-GPU benchmark and fall back to synchronized eval "
            "timing/VRAM when that benchmark is unavailable."
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
                    choices=[
                        (name.removeprefix("recovery_"), name) for name in methods
                    ],
                    value=default_method,
                    multiselect=False,
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=dataset_display_choices(dataset_groups),
                    value=default_dataset,
                    multiselect=False,
                )
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=7)
            with gr.Column(scale=5):
                radar_image = gr.Image(
                    label="Recovery speed–accuracy radar plots",
                    type="numpy",
                    interactive=False,
                    elem_classes=[
                        "results-media",
                        "recovery-speed-accuracy-radar-image",
                    ],
                )
                results_table = gr.Dataframe(
                    headers=list(_TABLE_HEADERS),
                    datatype=["str"] * len(_TABLE_HEADERS),
                    value=[],
                    label="Full / Recovery / compression measurements",
                    interactive=False,
                    wrap=True,
                    show_row_numbers=False,
                    show_search="filter",
                    pinned_columns=3,
                    max_height=560,
                    elem_classes=["recovery-speed-accuracy-table"],
                )

        inputs = [backbone_cb, method_dd, dataset_dd]
        for event in (backbone_cb.change, method_dd.change, dataset_dd.change):
            event(
                redraw,
                inputs=inputs,
                outputs=[radar_image, results_table, status],
                show_progress="full",
                trigger_mode="always_last",
            )
        refresh_btn.click(
            redraw,
            inputs=inputs,
            outputs=[radar_image, results_table, status],
            show_progress="full",
            trigger_mode="always_last",
        )
        blocks.load(
            redraw,
            inputs=inputs,
            outputs=[radar_image, results_table, status],
            show_progress="full",
        )

    return {
        "tab": tab,
        "backbone_cb": backbone_cb,
        "method_dd": method_dd,
        "dataset_dd": dataset_dd,
        "radar_image": radar_image,
        "results_table": results_table,
        "status": status,
    }

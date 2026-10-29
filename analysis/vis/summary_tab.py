"""Tab: interactive table collected directly from live v3 eval records."""

from __future__ import annotations

import html
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gradio as gr

from analysis.eval.summary_v3 import (
    AGG_PARAMS_KEY,
    DEFAULT_SUMMARY_PRUNE_STRATEGIES,
    METRIC_HEADERS,
    _fmt_value,
    collect_summary_parameter_counts,
    collect_summary_records,
)
from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    expand_pruning_strategy_keys,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.schema.v3 import UNIFIED_METRIC_KEYS
from analysis.recovery.shared_results import (
    seven_metric_mean_score,
    single_double_mean_decision,
)
from analysis.vis.dataset_average_warning import (
    GENERALIZATION_DATASET_AVERAGE_GROUPS,
    dataset_display_choices,
    dataset_display_name,
)

_BACKBONE_ORDER: Tuple[str, ...] = (
    "vggt",
    "pi3",
    "vggt_omega",
)

_DATASET_ORDER: Tuple[str, ...] = (
    "training average",
    "same-dataset testing average",
    "cross-dataset testing average",
    "four-dataset testing average",
    "two-dataset testing average",
    "scannet_dev50",
    "scans_dev50",
    "scannet_test50",
    "scans_test50",
    "7scenes_test",
    "7scenes",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "waymo_validation",
    "nuscenes_train",
    "waymo_training",
)

_PCT_RE = re.compile(r"\s*\(\*\*(.+?)\*\*\)")
_NUMBER_RE = re.compile(r"^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_PLAN_MODE_RE = re.compile(r"^(?:prune_span(\d+)|plan_budget(\d+)|plan_t(\d+))(?:_.*)?$")
_SPAN_IN_MODE_RE = re.compile(r"(?:^|/)prune_span(\d+)(?:$|/|_)")
_RECOVERY_MODE_RE = re.compile(r"(?:^|/)prune_span\d+[_/](?P<plan>[^/+]+)")
_INTERVAL_SELECTOR_PREFIX_RE = re.compile(
    r"^(?P<selector>i\d+_j\d+(?:_i\d+_j\d+)*)(?P<suffix>.*)$"
)
_INTERVAL_COMPONENT_RE = re.compile(r"i(?P<start>\d+)_j(?P<end>\d+)")
_BENCHMARK_FILENAME = "inference_benchmark_v3.json"
_BLOCK_PARAMS_HEADER = "Block params (M) ↓"
_INFERENCE_TIME_HEADER = "Latency (ms) ↓"
_PEAK_VRAM_HEADER = "Peak VRAM (GiB) ↓"
_BENCHMARK_HEADERS = (_INFERENCE_TIME_HEADER, _PEAK_VRAM_HEADER)
_ONE_DECIMAL_HEADERS = {
    _BLOCK_PARAMS_HEADER,
    _INFERENCE_TIME_HEADER,
    _PEAK_VRAM_HEADER,
}
_DEFAULT_RECOVERY_METHOD = "ghost_interval_sequential"
_SEVEN_METRIC_HEADERS: Tuple[str, ...] = tuple(
    METRIC_HEADERS[key] for key in UNIFIED_METRIC_KEYS
)
_RECOVERY_PRUNING_FAMILIES: Tuple[Tuple[str, ...], ...] = (
    ("CKA_double", "CKA_single"),
    ("retention_double", "retention_single"),
)
_COMPRESSION_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("FastVGGT", "fastvggt"),
    ("QuantVGGT", "quantvggt"),
)
_PRUNING_STRATEGY_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("CKA", "CKA"),
    ("Degradation", "retention"),
    ("ReplaceMe", "replaceme"),
)
_PRUNING_STRUCTURE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Single", "single"),
    ("Double", "double"),
)
_MODE_KIND_ORDER: Tuple[str, ...] = (
    "upper",
    "lower",
    "recovery",
    "linearpatch",
    "ghost",
    "ghost_interval",
    "ghost_interval_sequential",
    "ghost_cos",
    "ghost_mse_more",
    "ghost_mse_cos_more",
    "ghost_mse_cka_more",
    "ghost_mse_cos_cka_more",
    "ghost_mse_cos_balanced_more",
    "ghost_mse_cka_balanced_more",
    "ghost_mse_cos_cka_balanced_more",
    "ghost_mse_cka_ema_more",
    "ghost_mse_cos_cka_ema_half_more",
    "ghost_mse_cos_cka_ema_one_more",
    "ghost_mse_cka",
    "ghost_cka_only",
    "ghost_cka_ridge",
    "replaceme_ls",
    "replaceme_cos",
    "prunecomp",
    "lora",
    "train",
    "model",
)


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (_DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(_DATASET_ORDER), name)


def _mode_sort_key(name: str) -> Tuple[int, int, str]:
    key = str(name or "").strip()
    if key == "upper":
        return (0, 0, key)
    if key.startswith("upper+"):
        return (0, 1, key)
    interval = _interval_mode_parts(key)
    if interval is not None:
        return (1, interval[0], key)
    span_match = _SPAN_IN_MODE_RE.search(key)
    if span_match:
        return (1, int(span_match.group(1)), key)
    m = _PLAN_MODE_RE.match(key)
    if m:
        return (1, int(next(g for g in m.groups() if g is not None)), key)
    if key.startswith("prune_span") or key.startswith("plan"):
        return (1, 10_000, key)
    return (2, 0, key)


def _mode_kind_sort_key(kind: str) -> Tuple[int, str]:
    key = str(kind or "").strip().lower()
    try:
        return (_MODE_KIND_ORDER.index(key), key)
    except ValueError:
        return (len(_MODE_KIND_ORDER), key)


def _mode_kind_for_row(mode: str) -> str:
    key = str(mode or "").strip()
    low = key.lower()
    if low == "upper" or low.startswith("upper+"):
        return "upper"
    recovery_method = _recovery_method_for_row(key)
    if recovery_method:
        return recovery_method
    if low == "lower" or low.startswith("lower/") or low.startswith("prune_span") or low.startswith("plan_"):
        return "lower"
    interval = _interval_mode_parts(key.split("+", 1)[0])
    if interval is not None and not interval[2]:
        return "lower"
    if low == "model" or low.startswith("model/"):
        return "model"
    return "model"


def _recovery_method_for_row(mode: str) -> str:
    """Return a recovery method without its ``recovery_`` prefix."""
    base = str(mode or "").strip().split("+", 1)[0]
    interval = _interval_mode_parts(base)
    if interval is not None and interval[2].startswith("recovery_"):
        return interval[2].removeprefix("recovery_")
    recovery_match = _RECOVERY_MODE_RE.search(base)
    if recovery_match:
        return recovery_match.group("plan").removeprefix("recovery_")
    return ""


def _mode_group_for_row(mode: str) -> str:
    """Collapse concrete recovery methods into one top-level filter group."""
    kind = _mode_kind_for_row(mode)
    return "recovery" if _recovery_method_for_row(mode) else kind


def _default_recovery_methods(available: Sequence[str]) -> List[str]:
    """Default to the current target method, and never auto-select alternatives."""
    return (
        [_DEFAULT_RECOVERY_METHOD]
        if _DEFAULT_RECOVERY_METHOD in {str(item) for item in available}
        else []
    )


def _span_value_for_row(mode: str) -> str:
    key = str(mode or "").strip()
    match = _SPAN_IN_MODE_RE.search(key)
    if match:
        return str(int(match.group(1)))
    interval = _interval_mode_parts(key)
    if interval is not None:
        return str(interval[0])
    if _mode_kind_for_row(key) == "upper":
        return ""
    return key


def _interval_mode_parts(mode: str) -> Optional[Tuple[int, str, str]]:
    """Return dropped-layer count, selector, and non-selector mode suffix."""
    match = _INTERVAL_SELECTOR_PREFIX_RE.match(str(mode or "").strip())
    if match is None:
        return None

    selector = match.group("selector")
    intervals = [
        (int(component.group("start")), int(component.group("end")))
        for component in _INTERVAL_COMPONENT_RE.finditer(selector)
    ]
    if not intervals or any(start > end for start, end in intervals):
        return None

    merged: List[List[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    drop_count = sum(end - start + 1 for start, end in merged)
    suffix = match.group("suffix").lstrip("_/")
    return drop_count, selector, suffix


def _format_mode_cell(
    mode: str,
    pruning_sources: Sequence[str] = (),
    *,
    show_replaceme_unrecovered: bool = False,
) -> str:
    """Show a drop count whose tooltip identifies its pruning strategy."""
    interval = _interval_mode_parts(mode)
    if interval is None:
        return html.escape(_plain_mode_cell(mode))

    drop_count, _selector, suffix = interval
    sources = [str(source) for source in pruning_sources if str(source).strip()]
    if sources:
        source_label = ", ".join(dict.fromkeys(sources))
        escaped_source = html.escape(source_label, quote=True)
        count = (
            f'<abbr class="summary-drop-count" title="{escaped_source}" '
            f'aria-label="{drop_count} dropped layers; pruning strategy '
            f'{escaped_source}">{drop_count}</abbr>'
        )
    else:
        count = f'<span class="summary-drop-count">{drop_count}</span>'
    if not suffix:
        replaceme_structures = [
            structure.title()
            for structure in _pruning_structures_for_sources(sources)
            if any(
                source == f"replaceme_{structure}"
                for source in sources
            )
        ]
        if show_replaceme_unrecovered and replaceme_structures:
            structure_label = "/".join(replaceme_structures)
            return (
                f'{count}<span class="summary-mode-suffix"> · '
                f'ReplaceMe {html.escape(structure_label)} · no recovery</span>'
            )
        return count
    return (
        f'{count}<span class="summary-mode-suffix"> · '
        f'{html.escape(_short_mode_suffix(suffix))}</span>'
    )


def _short_mode_suffix(suffix: str) -> str:
    """Shorten recovery method labels while preserving compression variants."""
    method, separator, compression = str(suffix or "").partition("+")
    parts = [part for part in method.split("_") if part]
    short_method = "_".join(parts[-2:]) if len(parts) > 2 else method
    return f"{short_method}{separator}{compression}" if separator else short_method


def _plain_mode_cell(mode: str) -> str:
    """Return the visible mode text without the table's HTML decoration."""
    interval = _interval_mode_parts(mode)
    if interval is None:
        key = str(mode or "")
        method = _recovery_method_for_row(key)
        if not method:
            return key
        variant = _recovery_variant_for_row(key)
        suffix = f"{method}+{variant}" if variant else method
        return f"{_span_value_for_row(key)} · {_short_mode_suffix(suffix)}"
    drop_count, _selector, suffix = interval
    return (
        f"{drop_count} · {_short_mode_suffix(suffix)}"
        if suffix
        else str(drop_count)
    )


def _latex_escape_cell(value: Any) -> str:
    """Escape a plain table value for use as one LaTeX tabular cell."""
    text = str(value if value is not None else "").replace("\n", " ").strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _span_sort_key(span: str) -> Tuple[int, int, str]:
    key = str(span or "").strip()
    try:
        return (0, int(key), key)
    except ValueError:
        return (1, 0, key)


def _default_span_values(spans: Sequence[str]) -> List[str]:
    """Select drop counts from 12 onward on initial load."""
    selected: List[str] = []
    for span in spans:
        try:
            if int(str(span)) >= 12:
                selected.append(str(span))
        except ValueError:
            continue
    return selected


def _metric_key_for_header(header: str) -> Optional[str]:
    raw_header = str(header or "").strip()
    for metric_key in UNIFIED_METRIC_KEYS:
        if raw_header == METRIC_HEADERS.get(metric_key):
            return metric_key

    key = raw_header.lower()
    key = key.replace("↓", "").replace("↑", "").replace("(°)", "").strip()
    key = re.sub(r"\s+", " ", key)
    if key.startswith("ape"):
        return "ape"
    if key.startswith("are"):
        return "are"
    if key.startswith("rpe trans"):
        return "rpe_trans"
    if key.startswith("rpe rot"):
        return "rpe_rot"
    if key.startswith("acc"):
        return "acc"
    if key.startswith("comp"):
        return "comp"
    if key == "cd":
        return "cd"
    return None


def _metric_higher_is_better(header: str, metric_key: str) -> bool:
    text = str(header or "")
    if "↑" in text:
        return True
    if "↓" in text:
        return False
    return False


def _format_header_cell(header: str) -> str:
    """Render the two RPE suffixes as real HTML subscripts in summary tables."""
    metric_key = _metric_key_for_header(header)
    if metric_key == "rpe_trans":
        return "RPE<sub>t</sub>↓ (m)"
    if metric_key == "rpe_rot":
        return "RPE<sub>r</sub>↓ (°)"
    return html.escape(str(header))


def _format_training_metric(value: float, epoch: int) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.2f} (e{int(epoch)})"


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    backbone: str
    mode: str
    metrics: Dict[str, str]
    pruning_sources: Tuple[str, ...] = ()
    raw_metrics: Optional[Mapping[str, float]] = None


def collect_summary_rows(
    output_root: Path,
    *,
    prune_strategies: Sequence[str] = DEFAULT_SUMMARY_PRUNE_STRATEGIES,
    dataset_average_groups: Mapping[str, Sequence[str]] = GENERALIZATION_DATASET_AVERAGE_GROUPS,
    include_compression: bool = True,
    include_efficiency_metrics: bool = True,
    separate_pruning_structures: bool = False,
    winner_only_recovery: bool = False,
) -> Tuple[List[str], List[SummaryRow]]:
    """Collect the same table data for Gradio and the static project website."""
    header, rows, _ = _load_live_eval_rows(
        output_root,
        prune_strategies=prune_strategies,
        include_compression=include_compression,
        include_efficiency_metrics=include_efficiency_metrics,
    )
    rows = _add_dataset_average_rows(rows, header[3:], dataset_average_groups)
    if not separate_pruning_structures:
        rows = _select_best_recovery_rows(rows)
    elif winner_only_recovery:
        rows = _select_best_recovery_rows(
            rows, include_lower=False, share_recovery_winner_across_variants=True
        )
    return header, rows


def _load_pruning_source_map(
    backbone_root: Path,
    *,
    prune_strategies: Sequence[str] = DEFAULT_SUMMARY_PRUNE_STRATEGIES,
) -> Dict[str, Tuple[str, ...]]:
    """Map plan aliases and physical selectors to their pruning strategies."""
    try:
        bundle = load_prune_plan_bundle(Path(backbone_root))
        plan_records = bundle_plan_records(bundle)
        strategy_records = bundle_strategy_records(bundle)
        strategies = expand_pruning_strategy_keys(
            prune_strategies,
            available=strategy_records.keys(),
        )
    except Exception:
        return {}

    sources: Dict[str, List[str]] = {}
    for strategy in strategies:
        strategy_record = strategy_records.get(strategy, {})
        points = (
            strategy_record.get("points", {})
            if isinstance(strategy_record, Mapping)
            else {}
        )
        for plan_alias in points.values() if isinstance(points, Mapping) else ():
            alias = str(plan_alias)
            plan_record = plan_records.get(alias)
            if not isinstance(plan_record, Mapping):
                continue
            try:
                selector = physical_pruning_selector_from_record(plan_record)
            except Exception:
                continue
            for key in (alias, selector):
                bucket = sources.setdefault(key, [])
                if strategy not in bucket:
                    bucket.append(strategy)
    return {key: tuple(values) for key, values in sources.items()}


def _pruning_sources_for_mode(
    mode: str,
    source_map: Mapping[str, Sequence[str]],
) -> Tuple[str, ...]:
    base = str(mode or "").strip().split("+", 1)[0]
    interval = _interval_mode_parts(base)
    candidates = [interval[1]] if interval is not None else []
    candidates.extend(
        selector
        for selector in source_map
        if base == selector
        or base.startswith(f"{selector}_recovery_")
        or base.startswith(f"{selector}/recovery_")
    )
    for candidate in dict.fromkeys(candidates):
        values = source_map.get(candidate, ())
        if values:
            return tuple(dict.fromkeys(str(value) for value in values))
    return ()


def _benchmark_compression_tag(compression: Mapping[str, Any]) -> str:
    name = str(compression.get("name", "none")).strip().lower()
    if name in ("", "none"):
        return ""
    if name == "fastvggt":
        try:
            merge_start = int(compression.get("merge_start", 0))
            merge_ratio = float(compression.get("merge_ratio", 0.9))
        except (TypeError, ValueError):
            return ""
        ratio = f"{merge_ratio:g}".replace(".", "p")
        return f"fastvggt_m{merge_start}_r{ratio}"
    if name == "quantvggt":
        dtype = str(compression.get("dtype", "w4a4")).strip().lower()
        return f"quantvggt_{dtype}"
    return name


def _benchmark_mode_label(result: Mapping[str, Any]) -> Optional[str]:
    """Translate a benchmark variant into the matching summary-table mode."""
    variant = str(result.get("variant", "")).strip().lower()
    selector = str(result.get("selector", "")).strip()
    compression = result.get("compression", {})
    compression_tag = (
        _benchmark_compression_tag(compression)
        if isinstance(compression, Mapping)
        else ""
    )
    if variant == "full":
        return "upper" if selector == "full" and not compression_tag else None
    if variant in {"full_fastvggt", "full_quantvggt"}:
        return (
            f"upper+{compression_tag}"
            if selector == "full" and compression_tag
            else None
        )
    if variant == "pruning":
        return selector or None
    if variant in {
        "recovery",
        "recovery_fastvggt",
        "recovery_quantvggt",
    }:
        recovery_plan = str(result.get("recovery_plan_name", "")).strip()
        if not selector or not recovery_plan:
            return None
        base = f"{selector}_{recovery_plan}"
        return f"{base}+{compression_tag}" if compression_tag else base
    return None


def _benchmark_result_metrics(result: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """Average validated raw scene measurements from one benchmark result."""
    scenes = result.get("scenes", ())
    if not isinstance(scenes, list) or not scenes:
        return None
    latencies: List[float] = []
    peak_vram: List[float] = []
    for scene in scenes:
        if not isinstance(scene, Mapping):
            return None
        memory = scene.get("memory_gib", {})
        if not isinstance(memory, Mapping):
            return None
        try:
            latency = float(scene["latency_ms"])
            frames = int(scene["num_frames"])
            peak = float(memory["peak_allocated"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(latency)
            or latency <= 0.0
            or frames <= 0
            or not math.isfinite(peak)
            or peak < 0.0
        ):
            return None
        latencies.append(latency)
        peak_vram.append(peak)
    return {
        _INFERENCE_TIME_HEADER: f"{sum(latencies) / len(latencies):.1f}",
        _PEAK_VRAM_HEADER: f"{sum(peak_vram) / len(peak_vram):.1f}",
    }


def _load_benchmark_metrics(backbone_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Load run_testtime_benchmark measurements keyed by dataset and mode."""
    measurements: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in Path(backbone_root).glob(
        f"*/eval/testtime/{_BENCHMARK_FILENAME}"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = path.parents[2].name
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or str(payload.get("backbone", "")) != Path(backbone_root).name
            or str(payload.get("dataset", "")) != dataset
        ):
            continue
        results = payload.get("results", {})
        for result in results.values() if isinstance(results, Mapping) else ():
            if not isinstance(result, Mapping):
                continue
            mode = _benchmark_mode_label(result)
            metrics = _benchmark_result_metrics(result)
            if mode and metrics:
                measurements[(dataset, mode)] = metrics
    return measurements


def _load_live_eval_rows(
    output_root: Path,
    *,
    prune_strategies: Sequence[str] = DEFAULT_SUMMARY_PRUNE_STRATEGIES,
    include_compression: bool = True,
    include_efficiency_metrics: bool = True,
) -> Tuple[List[str], List[SummaryRow], List[Path]]:
    """Build table rows from eval_record_v3.json and the current plan bundle."""
    root = Path(output_root)
    if not root.is_dir():
        return [], [], []

    metric_keys = [*UNIFIED_METRIC_KEYS]
    if include_efficiency_metrics:
        metric_keys.append(AGG_PARAMS_KEY)
    metric_headers = [
        *(
            _BLOCK_PARAMS_HEADER
            if key == AGG_PARAMS_KEY
            else METRIC_HEADERS.get(key, key)
            for key in metric_keys
        ),
        *(_BENCHMARK_HEADERS if include_efficiency_metrics else ()),
    ]
    header = ["Dataset", "Backbone", "Mode", *metric_headers]
    rows: List[SummaryRow] = []
    live_roots: List[Path] = []
    for backbone_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: _backbone_sort_key(path.name),
    ):
        grouped = collect_summary_records(
            backbone_dir,
            prune_plan_root=backbone_dir,
            prune_strategies=prune_strategies,
        )
        if not grouped:
            continue
        live_roots.append(backbone_dir)
        upper_params: Optional[int] = None
        lower_params: Dict[str, int] = {}
        if include_efficiency_metrics:
            upper_params, lower_params = collect_summary_parameter_counts(
                backbone_dir,
                grouped,
            )
        pruning_source_map = _load_pruning_source_map(
            backbone_dir,
            prune_strategies=prune_strategies,
        )
        benchmark_metrics = (
            _load_benchmark_metrics(backbone_dir)
            if include_efficiency_metrics
            else {}
        )
        for dataset, modes in grouped.items():
            for mode, record in modes.items():
                if not include_compression and _compression_for_row(str(mode)):
                    continue
                metrics = {
                    METRIC_HEADERS.get(key, key): _fmt_value(
                        key,
                        float(record.average.get(key, float("nan"))),
                    )
                    for key in UNIFIED_METRIC_KEYS
                }
                if include_efficiency_metrics:
                    params = (
                        upper_params
                        if mode == "upper" or mode.startswith("upper+")
                        else lower_params.get(
                            mode, lower_params.get(mode.split("+", 1)[0])
                        )
                    )
                    metrics[_BLOCK_PARAMS_HEADER] = (
                        f"{float(params) / 1_000_000.0:.1f}"
                        if params is not None
                        else "—"
                    )
                    metrics.update(
                        benchmark_metrics.get(
                            (str(dataset), str(mode)),
                            {header: "—" for header in _BENCHMARK_HEADERS},
                        )
                    )
                rows.append(
                    SummaryRow(
                        dataset=str(dataset),
                        backbone=backbone_dir.name,
                        mode=str(mode),
                        metrics=metrics,
                        pruning_sources=_pruning_sources_for_mode(
                            str(mode), pruning_source_map
                        ),
                        raw_metrics={
                            METRIC_HEADERS.get(key, key): float(
                                record.average.get(key, float("nan"))
                            )
                            for key in UNIFIED_METRIC_KEYS
                        },
                    )
                )
    return header, rows, live_roots


def _numeric_cell_value(raw: str) -> Optional[float]:
    """Return the leading numeric value, ignoring any existing percentage suffix."""
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _numeric_cell_parts(raw: str) -> Optional[Tuple[float, str]]:
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value, text[match.end() :].strip()


def _mean_cell_value(values: Sequence[str]) -> str:
    numeric_values: List[float] = []
    suffixes: List[str] = []
    for raw in values:
        parts = _numeric_cell_parts(raw)
        if parts is None:
            continue
        value, suffix = parts
        numeric_values.append(value)
        if suffix:
            suffixes.append(suffix)
    if not numeric_values:
        return "—"
    mean_value = sum(numeric_values) / len(numeric_values)
    suffix = suffixes[0] if suffixes and all(s == suffixes[0] for s in suffixes) else ""
    return f"{mean_value:.2f}{suffix}"


def _add_dataset_average_rows(
    rows: Sequence[SummaryRow],
    metric_headers: Sequence[str],
    dataset_average_groups: Mapping[str, Sequence[str]] = (
        GENERALIZATION_DATASET_AVERAGE_GROUPS
    ),
) -> List[SummaryRow]:
    """Add the configured dataset averages and discard legacy ``mean``."""
    virtual_datasets = {
        "mean",
        *GENERALIZATION_DATASET_AVERAGE_GROUPS,
        *dataset_average_groups,
    }
    base_rows = [row for row in rows if row.dataset not in virtual_datasets]
    average_rows: List[SummaryRow] = []
    for average_name, datasets in dataset_average_groups.items():
        dataset_set = set(datasets)
        grouped: Dict[Tuple[str, str], List[SummaryRow]] = {}
        for row in base_rows:
            if row.dataset in dataset_set:
                grouped.setdefault((row.backbone, row.mode), []).append(row)

        for (backbone, mode), group_rows in sorted(
            grouped.items(),
            key=lambda item: (
                _backbone_sort_key(item[0][0]),
                _mode_sort_key(item[0][1]),
            ),
        ):
            average_raw_metrics: Dict[str, float] = {}
            for metric_header in _SEVEN_METRIC_HEADERS:
                raw_values = [
                    float(row.raw_metrics[metric_header])
                    for row in group_rows
                    if row.raw_metrics is not None
                    and metric_header in row.raw_metrics
                    and math.isfinite(float(row.raw_metrics[metric_header]))
                ]
                if raw_values:
                    average_raw_metrics[metric_header] = (
                        sum(raw_values) / len(raw_values)
                    )
            average_rows.append(
                SummaryRow(
                    dataset=average_name,
                    backbone=backbone,
                    mode=mode,
                    metrics={
                        metric_header: _mean_cell_value(
                            [
                                row.metrics.get(metric_header, "—")
                                for row in group_rows
                            ]
                        )
                        for metric_header in metric_headers
                    },
                    pruning_sources=tuple(
                        dict.fromkeys(
                            source
                            for row in group_rows
                            for source in row.pruning_sources
                        )
                    ),
                    raw_metrics=average_raw_metrics,
                )
            )
    return [*average_rows, *base_rows]


def _seven_metric_values(row: SummaryRow) -> Optional[Tuple[float, ...]]:
    """Return one complete seven-metric point in the canonical column order."""
    if row.raw_metrics is not None:
        raw_values = tuple(
            row.raw_metrics.get(header) for header in _SEVEN_METRIC_HEADERS
        )
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in raw_values
        ):
            return tuple(
                float(value)
                for value in raw_values
                if isinstance(value, (int, float))
            )
    values = tuple(
        _numeric_cell_value(row.metrics.get(header, "—"))
        for header in _SEVEN_METRIC_HEADERS
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values if value is not None)


def _recovery_variant_for_row(mode: str) -> str:
    """Keep separately benchmarked compression variants in separate groups."""
    key = str(mode or "").strip()
    return key.split("+", 1)[1] if "+" in key else ""


def _compression_for_row(mode: str) -> str:
    """Return the supported inference-compression family encoded in a mode."""
    variant = _recovery_variant_for_row(mode).lower()
    for _, compression in _COMPRESSION_CHOICES:
        if variant == compression or variant.startswith(f"{compression}_"):
            return compression
    return ""


def _compression_is_visible(
    mode: str,
    selected_compressions: Sequence[str],
) -> bool:
    """Uncompressed rows are always visible; compressed rows require opt-in."""
    compression = _compression_for_row(mode)
    return not compression or compression in {
        str(item) for item in (selected_compressions or ())
    }


def _pruning_families_for_sources(
    sources: Sequence[str],
) -> Tuple[str, ...]:
    families: List[str] = []
    for source in sources:
        key = str(source)
        family = next(
            (
                value
                for _label, value in _PRUNING_STRATEGY_CHOICES
                if key == value or key.startswith(f"{value}_")
            ),
            key,
        )
        if family and family not in families:
            families.append(family)
    return tuple(sorted(families))


def _pruning_structures_for_sources(
    sources: Sequence[str],
) -> Tuple[str, ...]:
    """Return the Single/Double structures represented by pruning sources."""
    structures: List[str] = []
    for source in sources:
        key = str(source).lower()
        for _label, structure in _PRUNING_STRUCTURE_CHOICES:
            if key == structure or key.endswith(f"_{structure}"):
                if structure not in structures:
                    structures.append(structure)
    return tuple(structures)


def _select_best_recovery_rows(
    rows: Sequence[SummaryRow],
    *,
    include_lower: bool = True,
    share_recovery_winner_across_variants: bool = False,
) -> List[SummaryRow]:
    """Select by the shared calibration seven-metric rule and reuse everywhere.

    ``share_recovery_winner_across_variants`` makes compressed recovery rows
    follow the winner selected from the uncompressed calibration result.  This
    is the test-time policy: compression does not independently reselect the
    Single/Double pruning structure.
    """

    def candidate_group(row: SummaryRow) -> Optional[Tuple[str, str, str, str, Tuple[str, ...]]]:
        method = _recovery_method_for_row(row.mode)
        mode_group = _mode_group_for_row(row.mode)
        pruning_families = _pruning_families_for_sources(row.pruning_sources)
        # ReplaceMe selects Single/Double from pure-pruning calibration errors
        # before fitting LS.  It does not run recovered calibration evaluation,
        # so its held-out recovery row must not be discarded here.
        if method == "replaceme_ls" or pruning_families == ("replaceme",):
            return None
        if method:
            candidate_kind = "recovery"
        elif include_lower and mode_group == "lower" and row.pruning_sources:
            candidate_kind = "lower"
        else:
            return None
        variant = _recovery_variant_for_row(row.mode)
        if candidate_kind == "recovery" and share_recovery_winner_across_variants:
            variant = ""
        return (
            candidate_kind,
            _span_value_for_row(row.mode),
            method,
            variant,
            pruning_families,
        )

    calibration_by_backbone: Dict[str, List[SummaryRow]] = {}
    for row in rows:
        if row.dataset == "training average":
            calibration_by_backbone.setdefault(row.backbone, []).append(row)

    winner_mode_by_group: Dict[
        Tuple[str, str, str, str, str, Tuple[str, ...]], str
    ] = {}
    for backbone, context_rows in calibration_by_backbone.items():
        candidates: Dict[Tuple[str, str, str, str, Tuple[str, ...]], List[SummaryRow]] = {}
        for row in context_rows:
            group = candidate_group(row)
            if group is None:
                continue
            if (
                share_recovery_winner_across_variants
                and group[0] == "recovery"
                and _recovery_variant_for_row(row.mode)
            ):
                continue
            candidates.setdefault(group, []).append(row)

        for group, group_rows in candidates.items():
            scored: List[Tuple[float, str, SummaryRow]] = []
            for row in group_rows:
                values = _seven_metric_values(row)
                if values is not None:
                    score = seven_metric_mean_score(dict(zip(UNIFIED_METRIC_KEYS, values)))
                    if score is not None:
                        scored.append((score, row.mode, row))
            by_structure = {
                structure: item
                for item in scored
                for structure in _pruning_structures_for_sources(item[2].pruning_sources)
                if structure in {"single", "double"}
            }
            if set(by_structure) == {"single", "double"}:
                decision = single_double_mean_decision(
                    by_structure["single"][0], by_structure["double"][0]
                )
                chosen = by_structure[str(decision["winner"])][2]
            else:
                chosen = (
                    min(scored, key=lambda item: (item[0], item[1]))[2]
                    if scored
                    else min(group_rows, key=lambda row: _mode_sort_key(row.mode))
                )
            winner_mode_by_group[(backbone, *group)] = chosen.mode.split("+", 1)[0]

    selected_ids: set[int] = set()
    candidate_ids: set[int] = set()
    for row in rows:
        group = candidate_group(row)
        if group is None:
            continue
        winner_mode = winner_mode_by_group.get((row.backbone, *group))
        if winner_mode is None:
            # Never use a held-out dataset as an implicit fallback selector.
            continue
        candidate_ids.add(id(row))
        if row.mode.split("+", 1)[0] == winner_mode:
            selected_ids.add(id(row))

    return [
        row
        for row in rows
        if id(row) not in candidate_ids or id(row) in selected_ids
    ]


def _format_base_cell_text(raw: str, *, decimals: int = 2) -> str:
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return html.escape(text)
    try:
        value = float(match.group(1))
    except ValueError:
        return html.escape(text)
    if not math.isfinite(value):
        return html.escape(text)
    suffix = text[match.end() :]
    return html.escape(f"{value:.{int(decimals)}f}{suffix}")


def _format_change_badge(
    change: Optional[float],
    *,
    percentage: bool,
    higher_is_better: bool,
    decimals: int = 2,
) -> str:
    if change is None or not math.isfinite(change):
        return ""
    if percentage:
        label = (
            f"{change:+.{int(decimals)}f}%"
            if change
            else f"{0.0:.{int(decimals)}f}%"
        )
    else:
        label = (
            f"{change:+.{int(decimals)}f}"
            if change
            else f"{0.0:.{int(decimals)}f}"
        )
    cls = "summary-pct"
    if change:
        improved = change > 0 if higher_is_better else change < 0
        cls += " summary-pct-down" if improved else " summary-pct-up"
    return f'<span class="{cls}">{html.escape(label)}</span>'


def _format_cell(
    raw: str,
    *,
    pct_change: Optional[float] = None,
    abs_change: Optional[float] = None,
    higher_is_better: bool = False,
    decimals: int = 2,
) -> str:
    text = str(raw or "").strip()
    if not text or text == "—":
        return '<span class="summary-na">—</span>'
    base = _format_base_cell_text(text, decimals=decimals)
    if abs_change is not None and math.isfinite(abs_change):
        return (
            f"{base}"
            f"{_format_change_badge(abs_change, percentage=False, higher_is_better=higher_is_better, decimals=decimals)}"
        )
    if pct_change is not None and math.isfinite(pct_change):
        return (
            f"{base}"
            f"{_format_change_badge(pct_change, percentage=True, higher_is_better=higher_is_better, decimals=decimals)}"
        )
    return base


def _render_summary_html(
    headers: Sequence[str],
    rows: Sequence[SummaryRow],
    *,
    metric_headers: Sequence[str],
    show_percentages: bool = False,
    show_absolute_changes: bool = False,
    reference_rows: Optional[Sequence[SummaryRow]] = None,
    show_replaceme_unrecovered: bool = False,
) -> str:
    if not rows:
        return (
            '<div class="summary-empty">No rows match the current filters. '
            "Select at least one backbone and one dataset.</div>"
        )

    display_headers = [
        "Dropped layers / Mode" if str(header) == "Mode" else str(header)
        for header in headers
    ]
    head_cells = (
        '<th class="summary-col-select">Select</th>'
        + "".join(f"<th>{_format_header_cell(h)}</th>" for h in display_headers)
    )
    dataset_rowspans: Dict[int, int] = {}
    backbone_rowspans: Dict[int, int] = {}
    idx = 0
    while idx < len(rows):
        dataset = rows[idx].dataset
        end = idx + 1
        while end < len(rows) and rows[end].dataset == dataset:
            end += 1
        dataset_rowspans[idx] = end - idx
        bb_idx = idx
        while bb_idx < end:
            backbone = rows[bb_idx].backbone
            bb_end = bb_idx + 1
            while (
                bb_end < end
                and rows[bb_end].dataset == dataset
                and rows[bb_end].backbone == backbone
            ):
                bb_end += 1
            backbone_rowspans[bb_idx] = bb_end - bb_idx
            bb_idx = bb_end
        idx = end

    upper_metrics = {
        (row.dataset, row.backbone): row.metrics
        for row in (reference_rows if reference_rows is not None else rows)
        if row.mode == "upper"
    }

    body_rows: List[str] = []
    for row_idx, row in enumerate(rows):
        # Export only metric columns; the leading Dataset/Backbone/Mode cells
        # remain useful in the browser but are deliberately omitted from the
        # copied LaTeX rows.
        latex_cells: List[str] = []
        dataset_cell = ""
        rowspan = dataset_rowspans.get(row_idx)
        if rowspan is not None:
            dataset_label = dataset_display_name(row.dataset)
            escaped_dataset_label = html.escape(dataset_label).replace("\n", "<br>")
            dataset_cell = (
                f'<td class="summary-col-dataset" rowspan="{rowspan}">'
                f"{escaped_dataset_label}</td>"
            )
        backbone_cell = ""
        bb_rowspan = backbone_rowspans.get(row_idx)
        if bb_rowspan is not None:
            backbone_cell = (
                f'<td class="summary-col-backbone" rowspan="{bb_rowspan}">'
                f"{html.escape(row.backbone)}</td>"
            )
        mode_cls = "summary-mode-upper" if row.mode == "upper" else "summary-mode-plan"
        baseline_metrics = upper_metrics.get((row.dataset, row.backbone), {})
        metric_cells_list: List[str] = []
        for metric_header in metric_headers:
            raw = row.metrics.get(metric_header, "—")
            pct_change: Optional[float] = None
            abs_change: Optional[float] = None
            metric_key = _metric_key_for_header(metric_header)
            higher_is_better = (
                _metric_higher_is_better(metric_header, metric_key)
                if metric_key is not None
                else False
            )
            value = _numeric_cell_value(raw)
            baseline = _numeric_cell_value(baseline_metrics.get(metric_header, "—"))
            cell_decimals = 1 if metric_header in _ONE_DECIMAL_HEADERS else 2
            if row.mode == "upper":
                pass
            elif show_absolute_changes:
                if value is not None and baseline is not None:
                    abs_change = value - baseline
            elif show_percentages:
                if value is not None and baseline is not None and baseline != 0:
                    pct_change = (value - baseline) / abs(baseline) * 100.0
            metric_cells_list.append(
                f"<td>{_format_cell(raw, pct_change=pct_change, abs_change=abs_change, higher_is_better=higher_is_better, decimals=cell_decimals)}</td>"
            )
            plain_metric = _PCT_RE.sub("", str(raw or "")).strip() or "—"
            if plain_metric != "—":
                numeric_parts = _numeric_cell_parts(plain_metric)
                if numeric_parts is not None:
                    number, suffix = numeric_parts
                    plain_metric = f"{number:.{cell_decimals}f}{suffix}"
            if abs_change is not None and math.isfinite(abs_change):
                plain_metric += f" ({abs_change:+.{cell_decimals}f})"
            elif pct_change is not None and math.isfinite(pct_change):
                plain_metric += f" ({pct_change:+.{cell_decimals}f}%)"
            latex_cells.append(_latex_escape_cell(plain_metric))
        metric_cells = "".join(metric_cells_list)
        latex_row = " & ".join(latex_cells) + r" \\"
        body_rows.append(
            "<tr>"
            '<td class="summary-col-select">'
            '<input type="checkbox" class="summary-row-select" '
            f'data-latex="{html.escape(latex_row, quote=True)}" '
            f'aria-label="Select {html.escape(row.backbone, quote=True)} '
            f'{html.escape(_plain_mode_cell(row.mode), quote=True)}" />'
            "</td>"
            f"{dataset_cell}"
            f"{backbone_cell}"
            f'<td class="summary-col-mode {mode_cls}">'
            f"{_format_mode_cell(row.mode, row.pruning_sources, show_replaceme_unrecovered=show_replaceme_unrecovered)}</td>"
            f"{metric_cells}"
            "</tr>"
        )

    return (
        '<div class="summary-table-wrap">'
        '<table class="summary-table">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _best_training_metrics_for_dataset(
    epochs: Sequence[Dict[str, Any]],
    dataset: str,
    metric_headers: Sequence[str],
) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for header in metric_headers:
        metric_key = _metric_key_for_header(header)
        if not metric_key:
            metrics[str(header)] = "—"
            continue
        higher_is_better = _metric_higher_is_better(header, metric_key)
        best_value: Optional[float] = None
        best_epoch: Optional[int] = None
        for item in epochs:
            if not isinstance(item, dict):
                continue
            raw_epoch = item.get("epoch", 0)
            try:
                epoch = int(raw_epoch)
            except (TypeError, ValueError):
                epoch = 0
            ds_metrics = item.get(dataset)
            if not isinstance(ds_metrics, dict) or metric_key not in ds_metrics:
                continue
            try:
                value = float(ds_metrics[metric_key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if best_value is None:
                best_value = value
                best_epoch = epoch
            elif higher_is_better and value > best_value:
                best_value = value
                best_epoch = epoch
            elif not higher_is_better and value < best_value:
                best_value = value
                best_epoch = epoch
        metrics[str(header)] = (
            _format_training_metric(best_value, int(best_epoch or 0))
            if best_value is not None
            else "—"
        )
    return metrics


def _load_training_epochs_from_csv(path: Path) -> List[Dict[str, Any]]:
    epochs_by_id: Dict[int, Dict[str, Any]] = {}
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("tag", "")).strip() != "train":
                    continue
                dataset = str(row.get("dataset", "")).strip()
                metric = str(row.get("metric", "")).strip()
                if not dataset or not metric:
                    continue
                try:
                    epoch = int(row.get("epoch", 0))
                    value = float(row.get("value", ""))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                epoch_row = epochs_by_id.setdefault(epoch, {"epoch": epoch})
                ds_metrics = epoch_row.setdefault(dataset, {})
                if isinstance(ds_metrics, dict):
                    ds_metrics[metric] = value
    except OSError:
        return []
    return [epochs_by_id[k] for k in sorted(epochs_by_id)]


def _load_training_best_rows(
    *,
    output_root: Path,
    metric_headers: Sequence[str],
) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    root = Path(output_root)
    if not root.is_dir():
        return rows
    for backbone_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _backbone_sort_key(p.name)):
        train_roots = [backbone_dir / "train", backbone_dir / "scannet_dev50" / "train"]
        for train_root in train_roots:
            if not train_root.is_dir():
                continue
            for run_dir in sorted((p for p in train_root.iterdir() if p.is_dir()), key=lambda p: _mode_sort_key(p.name)):
                eval_path = run_dir / "eval_metrics_history.csv"
                if not eval_path.is_file():
                    continue
                epochs = _load_training_epochs_from_csv(eval_path)
                if not epochs:
                    continue
                datasets = sorted(
                    {
                        str(k)
                        for item in epochs
                        if isinstance(item, dict)
                        for k, v in item.items()
                        if k != "epoch" and isinstance(v, dict)
                    },
                    key=_dataset_sort_key,
                )
                for dataset in datasets:
                    rows.append(
                        SummaryRow(
                            dataset=dataset,
                            backbone=backbone_dir.name,
                            mode=run_dir.name,
                            metrics=_best_training_metrics_for_dataset(
                                epochs,
                                dataset,
                                metric_headers,
                            ),
                        )
                    )
    return rows


def build_summary_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str | None = None,
    tab_name: str = "Eval Summary",
    prune_strategies: Sequence[str] = DEFAULT_SUMMARY_PRUNE_STRATEGIES,
    dataset_average_groups: Mapping[str, Sequence[str]] = (
        GENERALIZATION_DATASET_AVERAGE_GROUPS
    ),
    default_datasets: Optional[Sequence[str]] = None,
    default_recovery_methods: Optional[Sequence[str]] = None,
    default_spans_with_recovery: bool = False,
    include_compression: bool = True,
    include_efficiency_metrics: bool = True,
    separate_pruning_structures: bool = False,
    winner_only_recovery: bool = False,
    show_replaceme_unrecovered: bool = False,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    tab_name = str(tab_name or "Eval Summary").strip() or "Eval Summary"
    prune_strategies = tuple(str(item) for item in prune_strategies)
    dataset_average_groups = {
        str(name): tuple(str(dataset) for dataset in datasets)
        for name, datasets in dataset_average_groups.items()
    }

    def _load_rows_from_disk(training_loaded: bool) -> Tuple[List[str], List[str], List[SummaryRow], List[Path], int]:
        fresh_header, fresh_rows, live_roots = _load_live_eval_rows(
            output_root,
            prune_strategies=prune_strategies,
            include_compression=include_compression,
            include_efficiency_metrics=include_efficiency_metrics,
        )
        fresh_metric_headers = fresh_header[3:] if len(fresh_header) > 3 else []
        training_rows: List[SummaryRow] = []
        if training_loaded:
            training_rows = _load_training_best_rows(
                output_root=output_root,
                metric_headers=fresh_metric_headers,
            )
        rows = _add_dataset_average_rows(
            [*fresh_rows, *training_rows],
            fresh_metric_headers,
            dataset_average_groups,
        )
        if not separate_pruning_structures:
            rows = _select_best_recovery_rows(rows)
        elif winner_only_recovery:
            rows = _select_best_recovery_rows(
                rows,
                include_lower=False,
                share_recovery_winner_across_variants=True,
            )
        return fresh_header, fresh_metric_headers, rows, live_roots, len(training_rows)

    header, metric_headers, initial_rows, _, _ = _load_rows_from_disk(False)
    backbone_choices = sorted({r.backbone for r in initial_rows}, key=_backbone_sort_key)
    dataset_choices = sorted({r.dataset for r in initial_rows}, key=_dataset_sort_key)
    span_choices = sorted(
        {span for r in initial_rows for span in [_span_value_for_row(r.mode)] if span},
        key=_span_sort_key,
    )
    default_span_values = (
        sorted(
            {
                span
                for row in initial_rows
                if _mode_group_for_row(row.mode) == "recovery"
                for span in [_span_value_for_row(row.mode)]
                if span
            },
            key=_span_sort_key,
        )
        if default_spans_with_recovery
        else _default_span_values(span_choices)
    )
    mode_group_choices = sorted(
        {_mode_group_for_row(r.mode) for r in initial_rows},
        key=_mode_kind_sort_key,
    )
    recovery_method_choices = sorted(
        {
            method
            for row in initial_rows
            for method in [_recovery_method_for_row(row.mode)]
            if method
        },
        key=_mode_kind_sort_key,
    )
    default_recovery_method_values = (
        [
            str(name)
            for name in default_recovery_methods
            if str(name) in recovery_method_choices
        ]
        if default_recovery_methods is not None
        else _default_recovery_methods(recovery_method_choices)
    )
    default_dataset_values = (
        [str(name) for name in default_datasets if str(name) in dataset_choices]
        if default_datasets is not None
        else [
            name
            for name in dataset_average_groups
            if name in dataset_choices
        ] or list(dataset_choices)
    )
    pruning_strategy_choices = [
        (label, value)
        for label, value in _PRUNING_STRATEGY_CHOICES
        if any(
            strategy == value or strategy.startswith(f"{value}_")
            for strategy in prune_strategies
        )
    ]
    default_pruning_strategy_values = [
        value for _label, value in pruning_strategy_choices
    ]
    default_pruning_structure_values = [
        value for _label, value in _PRUNING_STRUCTURE_CHOICES
    ]

    def _choices_for_rows(
        rows: Sequence[SummaryRow],
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        return (
            sorted({r.backbone for r in rows}, key=_backbone_sort_key),
            sorted({r.dataset for r in rows}, key=_dataset_sort_key),
            sorted(
                {span for r in rows for span in [_span_value_for_row(r.mode)] if span},
                key=_span_sort_key,
            ),
            sorted(
                {_mode_group_for_row(r.mode) for r in rows},
                key=_mode_kind_sort_key,
            ),
            sorted(
                {
                    method
                    for row in rows
                    for method in [_recovery_method_for_row(row.mode)]
                    if method
                },
                key=_mode_kind_sort_key,
            ),
        )

    def _valid_values(selected: Sequence[str], choices: Sequence[str]) -> List[str]:
        selected_set = {str(x) for x in (selected or [])}
        return [str(x) for x in choices if str(x) in selected_set]

    def _filter_rows(
        rows: Sequence[SummaryRow],
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_spans: Sequence[str],
        selected_mode_groups: Sequence[str],
        selected_recovery_methods: Sequence[str],
        selected_compressions: Sequence[str],
        selected_pruning_strategies: Sequence[str],
        selected_pruning_structures: Sequence[str],
    ) -> List[SummaryRow]:
        bb_set = {str(x) for x in (selected_backbones or [])}
        ds_set = {str(x) for x in (selected_datasets or [])}
        span_set = {str(x) for x in (selected_spans or [])}
        mode_group_set = {str(x) for x in (selected_mode_groups or [])}
        recovery_method_set = {
            str(x) for x in (selected_recovery_methods or [])
        }
        pruning_strategy_set = {
            str(x) for x in (selected_pruning_strategies or [])
        }
        pruning_structure_set = {
            str(x) for x in (selected_pruning_structures or [])
        }
        if not bb_set or not ds_set or not mode_group_set:
            return []
        filtered: List[SummaryRow] = []
        for r in rows:
            if r.backbone not in bb_set or r.dataset not in ds_set:
                continue
            if include_compression:
                if not _compression_is_visible(r.mode, selected_compressions):
                    continue
            elif _compression_for_row(r.mode):
                continue
            if r.pruning_sources and not any(
                source == strategy or source.startswith(f"{strategy}_")
                for source in r.pruning_sources
                for strategy in pruning_strategy_set
            ):
                continue
            structures = _pruning_structures_for_sources(r.pruning_sources)
            if structures and not pruning_structure_set.intersection(structures):
                continue
            mode_group = _mode_group_for_row(r.mode)
            if mode_group not in mode_group_set:
                continue
            if (
                mode_group == "recovery"
                and _recovery_method_for_row(r.mode) not in recovery_method_set
            ):
                continue
            if mode_group != "upper" and _span_value_for_row(r.mode) not in span_set:
                continue
            filtered.append(r)
        return sorted(
            filtered,
            key=lambda r: (
                _dataset_sort_key(r.dataset),
                _backbone_sort_key(r.backbone),
                _mode_sort_key(r.mode),
            ),
        )

    def _refresh(
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_spans: Sequence[str],
        selected_mode_groups: Sequence[str],
        selected_recovery_methods: Sequence[str],
        selected_compressions: Sequence[str],
        selected_pruning_strategies: Sequence[str],
        selected_pruning_structures: Sequence[str],
        training_loaded: bool,
        show_percentages: bool,
        show_absolute_changes: bool,
    ):
        fresh_header, fresh_metric_headers, source_rows, live_roots, training_count = _load_rows_from_disk(
            bool(training_loaded)
        )
        (
            cur_backbone_choices,
            cur_dataset_choices,
            cur_span_choices,
            cur_mode_group_choices,
            cur_recovery_method_choices,
        ) = _choices_for_rows(source_rows)
        bb_values = _valid_values(selected_backbones, cur_backbone_choices)
        ds_values = _valid_values(selected_datasets, cur_dataset_choices)
        span_values = _valid_values(selected_spans, cur_span_choices)
        mode_group_values = _valid_values(
            selected_mode_groups, cur_mode_group_choices
        )
        recovery_method_values = _valid_values(
            selected_recovery_methods, cur_recovery_method_choices
        )
        rows = _filter_rows(
            source_rows,
            bb_values,
            ds_values,
            span_values,
            mode_group_values,
            recovery_method_values,
            selected_compressions,
            selected_pruning_strategies,
            selected_pruning_structures,
        )
        table_html = _render_summary_html(
            fresh_header,
            rows,
            metric_headers=fresh_metric_headers,
            show_percentages=bool(show_percentages),
            show_absolute_changes=bool(show_absolute_changes),
            reference_rows=source_rows,
            show_replaceme_unrecovered=show_replaceme_unrecovered,
        )
        status = (
            f"{len(rows)} rows · "
            f"{len(bb_values)}/{len(cur_backbone_choices)} backbones · "
            f"{len(ds_values)}/{len(cur_dataset_choices)} datasets · "
            f"{len(span_values)}/{len(cur_span_choices)} drop counts · "
            f"{len(mode_group_values)}/{len(cur_mode_group_choices)} modes · "
            f"{len(recovery_method_values)}/{len(cur_recovery_method_choices)} "
            "recovery methods · "
            f"{len(selected_pruning_strategies or ())}/"
            f"{len(pruning_strategy_choices)} pruning strategies"
        )
        if separate_pruning_structures:
            status += (
                f" · {len(selected_pruning_structures or ())}/"
                f"{len(_PRUNING_STRUCTURE_CHOICES)} pruning structures"
            )
        if include_compression:
            status += (
                f" · {len(selected_compressions or ())}/"
                f"{len(_COMPRESSION_CHOICES)} compression types"
            )
        if training_loaded:
            status += f" · training best rows={training_count}"
        if not live_roots:
            status = f"No live eval_record_v3.json results found under: {output_root}"
        return (
            gr.update(choices=cur_backbone_choices, value=bb_values),
            gr.update(
                choices=dataset_display_choices(cur_dataset_choices),
                value=ds_values,
            ),
            gr.update(choices=cur_span_choices, value=span_values),
            gr.update(choices=cur_mode_group_choices, value=mode_group_values),
            gr.update(
                choices=cur_recovery_method_choices,
                value=recovery_method_values,
            ),
            table_html,
            status,
        )

    def _toggle_training_rows(
        training_loaded: bool,
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_spans: Sequence[str],
        selected_mode_groups: Sequence[str],
        selected_recovery_methods: Sequence[str],
        selected_compressions: Sequence[str],
        selected_pruning_strategies: Sequence[str],
        selected_pruning_structures: Sequence[str],
        show_percentages: bool,
        show_absolute_changes: bool,
    ):
        next_loaded = not bool(training_loaded)
        fresh_header, fresh_metric_headers, source_rows, live_roots, training_count = _load_rows_from_disk(
            next_loaded
        )
        (
            next_backbone_choices,
            next_dataset_choices,
            next_span_choices,
            next_mode_group_choices,
            next_recovery_method_choices,
        ) = _choices_for_rows(source_rows)

        bb_set = {str(x) for x in (selected_backbones or [])}
        ds_set = {str(x) for x in (selected_datasets or [])}
        span_set = {str(x) for x in (selected_spans or [])}
        mode_group_set = {str(x) for x in (selected_mode_groups or [])}
        recovery_method_set = {
            str(x) for x in (selected_recovery_methods or [])
        }
        if next_loaded:
            bb_values = sorted(bb_set | set(next_backbone_choices), key=_backbone_sort_key)
            ds_values = sorted(ds_set | set(next_dataset_choices), key=_dataset_sort_key)
            span_values = sorted(span_set | set(next_span_choices), key=_span_sort_key)
            mode_group_values = sorted(
                mode_group_set | set(next_mode_group_choices),
                key=_mode_kind_sort_key,
            )
            recovery_method_values = [
                x
                for x in next_recovery_method_choices
                if x in recovery_method_set
            ]
        else:
            bb_values = [x for x in next_backbone_choices if x in bb_set]
            ds_values = [x for x in next_dataset_choices if x in ds_set]
            span_values = [x for x in next_span_choices if x in span_set]
            mode_group_values = [
                x for x in next_mode_group_choices if x in mode_group_set
            ]
            recovery_method_values = [
                x
                for x in next_recovery_method_choices
                if x in recovery_method_set
            ]
            if not bb_values:
                bb_values = list(next_backbone_choices)
            if not ds_values:
                ds_values = list(next_dataset_choices)
            if not span_values:
                span_values = list(next_span_choices)
            if not mode_group_values:
                mode_group_values = list(next_mode_group_choices)

        rows = _filter_rows(
            source_rows,
            bb_values,
            ds_values,
            span_values,
            mode_group_values,
            recovery_method_values,
            selected_compressions,
            selected_pruning_strategies,
            selected_pruning_structures,
        )
        table_html = _render_summary_html(
            fresh_header,
            rows,
            metric_headers=fresh_metric_headers,
            show_percentages=bool(show_percentages),
            show_absolute_changes=bool(show_absolute_changes),
            reference_rows=source_rows,
            show_replaceme_unrecovered=show_replaceme_unrecovered,
        )
        status = (
            f"{len(rows)} rows · "
            f"{len(bb_values)}/{len(next_backbone_choices)} backbones · "
            f"{len(ds_values)}/{len(next_dataset_choices)} datasets · "
            f"{len(span_values)}/{len(next_span_choices)} drop counts · "
            f"{len(mode_group_values)}/{len(next_mode_group_choices)} modes · "
            f"{len(recovery_method_values)}/{len(next_recovery_method_choices)} "
            "recovery methods · "
            f"{len(selected_pruning_strategies or ())}/"
            f"{len(pruning_strategy_choices)} pruning strategies"
        )
        if separate_pruning_structures:
            status += (
                f" · {len(selected_pruning_structures or ())}/"
                f"{len(_PRUNING_STRUCTURE_CHOICES)} pruning structures"
            )
        if include_compression:
            status += (
                f" · {len(selected_compressions or ())}/"
                f"{len(_COMPRESSION_CHOICES)} compression types"
            )
        if next_loaded:
            status += f" · training best rows={training_count}"
        if not live_roots:
            status = f"No live eval_record_v3.json results found under: {output_root}"
        return (
            next_loaded,
            gr.update(
                value="Hide training eval best"
                if next_loaded
                else "Load training eval best"
            ),
            gr.update(choices=next_backbone_choices, value=bb_values),
            gr.update(
                choices=dataset_display_choices(next_dataset_choices),
                value=ds_values,
            ),
            gr.update(choices=next_span_choices, value=span_values),
            gr.update(choices=next_mode_group_choices, value=mode_group_values),
            gr.update(
                choices=next_recovery_method_choices,
                value=recovery_method_values,
            ),
            table_html,
            status,
        )

    with gr.Tab(tab_name, id=tab_id) as tab_summary:
        source_description = (
            f"Source: live `eval_record_v3.json` under `{output_root}` "
            "(reloaded from disk on each refresh; Markdown summaries are not "
            "used)."
        )
        if include_efficiency_metrics:
            source_description += (
                " Resource columns additionally use "
                "`eval/testtime/inference_benchmark_v3.json` from the same root."
            )
        compression_description = (
            " FastVGGT and QuantVGGT results are opt-in below."
            if include_compression
            else ""
        )
        pruning_description = (
            (
                " Unrecovered Single and Double candidates are shown separately; "
                "recovery rows keep only the calibration winner for each drop "
                "count using the raw seven-metric mean with a 0.10 "
                "Double-improvement margin."
            )
            if separate_pruning_structures and winner_only_recovery
            else (
                " Single and double pruning candidates are shown separately and can "
                "be selected below."
                if separate_pruning_structures
                else (
                " For each drop count, CKA/Degradation single/double candidates "
                "are reduced using the raw seven-metric mean with a 0.10 "
                "Double-improvement margin."
                )
            )
        )
        gr.Markdown(
            f"## {tab_name}\n{source_description}{pruning_description}"
            f"{compression_description}"
        )
        training_loaded_state = gr.State(False)
        with gr.Row():
            with gr.Column(scale=1, min_width=240, elem_classes=["control-column"]):
                summary_training_btn = gr.Button(
                    value="Load training eval best",
                    variant="secondary",
                )
                summary_percentage_cb = gr.Checkbox(
                    label="Show change vs upper (%)",
                    value=False,
                )
                summary_absolute_cb = gr.Checkbox(
                    label="Show change vs upper (abs)",
                    value=False,
                )
                summary_backbone_cb = gr.CheckboxGroup(
                    label="Backbone",
                    choices=backbone_choices,
                    value=backbone_choices,
                )
                summary_dataset_cb = gr.CheckboxGroup(
                    label="Dataset",
                    choices=dataset_display_choices(dataset_choices),
                    value=default_dataset_values,
                )
                summary_span_cb = gr.CheckboxGroup(
                    label="Dropped layers",
                    choices=span_choices,
                    value=default_span_values,
                )
                summary_mode_group_cb = gr.CheckboxGroup(
                    label="Mode",
                    choices=mode_group_choices,
                    value=mode_group_choices,
                )
                summary_recovery_method_cb = gr.CheckboxGroup(
                    label="Recovery methods",
                    choices=recovery_method_choices,
                    value=default_recovery_method_values,
                )
                if include_compression:
                    summary_compression_cb: Optional[gr.CheckboxGroup] = (
                        gr.CheckboxGroup(
                            label="Inference compression",
                            choices=list(_COMPRESSION_CHOICES),
                            value=[],
                        )
                    )
                    summary_compression_input: Any = summary_compression_cb
                else:
                    summary_compression_cb = None
                    summary_compression_input = gr.State([])
                summary_pruning_strategy_cb = gr.CheckboxGroup(
                    label="Pruning strategy",
                    choices=pruning_strategy_choices,
                    value=default_pruning_strategy_values,
                )
                if separate_pruning_structures:
                    summary_pruning_structure_cb: Optional[gr.CheckboxGroup] = (
                        gr.CheckboxGroup(
                            label="Pruning structure",
                            choices=list(_PRUNING_STRUCTURE_CHOICES),
                            value=default_pruning_structure_values,
                        )
                    )
                    summary_pruning_structure_input: Any = (
                        summary_pruning_structure_cb
                    )
                else:
                    summary_pruning_structure_cb = None
                    summary_pruning_structure_input = gr.State(
                        default_pruning_structure_values
                    )
                summary_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                summary_copy_latex_btn = gr.Button(
                    value="Copy selected rows as LaTeX",
                    variant="secondary",
                )
                summary_html = gr.HTML(
                    value=_render_summary_html(
                        header,
                        _filter_rows(
                            initial_rows,
                            backbone_choices,
                            default_dataset_values,
                            default_span_values,
                            mode_group_choices,
                            default_recovery_method_values,
                            [],
                            default_pruning_strategy_values,
                            default_pruning_structure_values,
                        ),
                        metric_headers=metric_headers,
                        show_replaceme_unrecovered=show_replaceme_unrecovered,
                    ),
                    elem_id=f"{tab_id or 'eval-summary'}-summary-table",
                    elem_classes=["summary-panel"],
                )

        summary_copy_latex_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=f"""
            () => {{
                const root = document.getElementById({json.dumps(f"{tab_id or 'eval-summary'}-summary-table")});
                if (!root) return;
                const selected = Array.from(root.querySelectorAll('.summary-row-select:checked'));
                if (selected.length === 0) {{
                    window.alert('Select at least one table row first.');
                    return;
                }}
                const latex = selected.map((box) => box.dataset.latex || '').join('\\n');
                navigator.clipboard.writeText(latex).catch(() => {{
                    window.prompt('Copy the selected LaTeX rows:', latex);
                }});
            }}
            """,
            queue=False,
            show_progress="hidden",
        )

        summary_inputs = [
            summary_backbone_cb,
            summary_dataset_cb,
            summary_span_cb,
            summary_mode_group_cb,
            summary_recovery_method_cb,
            summary_compression_input,
            summary_pruning_strategy_cb,
            summary_pruning_structure_input,
            training_loaded_state,
            summary_percentage_cb,
            summary_absolute_cb,
        ]
        summary_outputs = [
            summary_backbone_cb,
            summary_dataset_cb,
            summary_span_cb,
            summary_mode_group_cb,
            summary_recovery_method_cb,
            summary_html,
            summary_status,
        ]
        refresh_controls = [
            summary_backbone_cb,
            summary_dataset_cb,
            summary_span_cb,
            summary_mode_group_cb,
            summary_recovery_method_cb,
            summary_pruning_strategy_cb,
            summary_percentage_cb,
            summary_absolute_cb,
        ]
        if summary_pruning_structure_cb is not None:
            refresh_controls.append(summary_pruning_structure_cb)
        if summary_compression_cb is not None:
            refresh_controls.append(summary_compression_cb)
        for ctrl in refresh_controls:
            ctrl.change(
                _refresh,
                inputs=summary_inputs,
                outputs=summary_outputs,
                show_progress="hidden",
            )
        summary_training_btn.click(
            _toggle_training_rows,
            inputs=[
                training_loaded_state,
                summary_backbone_cb,
                summary_dataset_cb,
                summary_span_cb,
                summary_mode_group_cb,
                summary_recovery_method_cb,
                summary_compression_input,
                summary_pruning_strategy_cb,
                summary_pruning_structure_input,
                summary_percentage_cb,
                summary_absolute_cb,
            ],
            outputs=[
                training_loaded_state,
                summary_training_btn,
                summary_backbone_cb,
                summary_dataset_cb,
                summary_span_cb,
                summary_mode_group_cb,
                summary_recovery_method_cb,
                summary_html,
                summary_status,
            ],
            show_progress="minimal",
        )
        tab_summary.select(
            _refresh,
            inputs=summary_inputs,
            outputs=summary_outputs,
            show_progress="hidden",
        )

    return {
        "tab": tab_summary,
        "mode_cb": summary_mode_group_cb,
        "span_cb": summary_span_cb,
        "recovery_method_cb": summary_recovery_method_cb,
        "compression_cb": summary_compression_cb,
        "pruning_strategy_cb": summary_pruning_strategy_cb,
        "pruning_structure_cb": summary_pruning_structure_cb,
        "dataset_cb": summary_dataset_cb,
        "copy_latex_btn": summary_copy_latex_btn,
        "table_html": summary_html,
    }

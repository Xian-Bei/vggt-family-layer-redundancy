from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_physical_pruning_selector,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.redundancy.skip_metric_scaling import (
    SEVEN_METRIC_KEYS,
    apply_minmax_bounds,
    minmax_normalize_finite,
)

PRUNING_METRIC_GRID = (
    ("ape", "are", "rpe_trans", "rpe_rot"),
    ("acc", "comp", "chamfer_distance", "gt_mean"),
)
PRUNING_MEAN_METRIC_GRID = (("gt_mean",),)
RECOVERY_METRIC_GRID = (
    ("ape", "are", "rpe_trans", "rpe_rot"),
    ("acc", "comp", "chamfer_distance", "gt_mean"),
)
GT_MEAN_METRICS = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")
DEFAULT_KNEE_DEGRADATION_THRESHOLD = 0.1
METRIC_HEADERS = {
    "ape": "APE ↓",
    "are": "ARE (deg) ↓",
    "rpe_trans": "RPE trans ↓",
    "rpe_rot": "RPE rot (deg) ↓",
    "acc": "Acc ↓",
    "comp": "Comp ↓",
    "chamfer_distance": "CD ↓",
    "cd": "CD ↓",
    "nc": "NC ↑",
    "gt_mean": "7-metric mean degradation ↓",
}
HIGHER_IS_BETTER = frozenset({"depth_delta1", "nc"})
STRATEGY_PLOT_ORDER = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
    "gt_drop",
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_cos",
    "recovery_ghost_cos_more",
    "recovery_ghost_cos_scale",
    "recovery_ghost_mse_more",
    "recovery_ghost_mse_cos_more",
    "recovery_ghost_mse_cka_more",
    "recovery_ghost_mse_cos_cka_more",
    "recovery_ghost_mse_cos_balanced_more",
    "recovery_ghost_mse_cka_balanced_more",
    "recovery_ghost_mse_cos_cka_balanced_more",
    "recovery_ghost_mse_cka_ema_more",
    "recovery_ghost_mse_cos_cka_ema_half_more",
    "recovery_ghost_mse_cos_cka_ema_one_more",
    "recovery_ghost_mse_cka",
    "recovery_ghost_cka_only",
    "recovery_ghost_cka_ridge",
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_replaceme_cos_more",
    "recovery_replaceme_cos_scale",
    "recovery_prunecomp",
    "recovery_lora",
    "recovery_train",
    "gt_single",
    "gt_double",
    "ours_single",
    "ours_double",
    "cosine_single",
    "cosine_double",
    "inv_mse_single",
    "inv_mse_double",
    "replaceme_single",
    "replaceme_double",
    "shortgpt",
    "simpler",
)
RECOVERY_METHOD_ORDER = tuple(
    method for method in STRATEGY_PLOT_ORDER if method.startswith("recovery_")
)
STRATEGY_PLOT_COLORS = {
    "ours": "#2ca02c",
    "ours_single": "#2ca02c",
    "ours_double": "#2ca02c",
    "cosine": "#ff7f0e",
    "cosine_single": "#ff7f0e",
    "cosine_double": "#ff7f0e",
    "gt": "#1f77b4",
    "gt_single": "#1f77b4",
    "gt_double": "#1f77b4",
    "replaceme": "#d62728",
    "replaceme_single": "#d62728",
    "replaceme_double": "#d62728",
    "inv_mse": "#9467bd",
    "inv_mse_single": "#9467bd",
    "inv_mse_double": "#9467bd",
    "shortgpt": "#8c564b",
    "simpler": "#e377c2",
    "no_recovery_single": "#1f77b4",
    "no_recovery_double": "#1f77b4",
    "no_recovery": "#1f77b4",
    "gt_drop": "#7f7f7f",
    "recovery_linearpatch": "#ff7f0e",
    "recovery_ghost": "#9467bd",
    "recovery_ghost_interval": "#2ca02c",
    "recovery_ghost_interval_sequential": "#17becf",
    "recovery_ghost_cos": "#17becf",
    "recovery_ghost_cos_more": "#0e7490",
    "recovery_ghost_cos_scale": "#67e8f9",
    "recovery_ghost_mse_more": "#b45309",
    "recovery_ghost_mse_cos_more": "#ea580c",
    "recovery_ghost_mse_cka_more": "#7c3aed",
    "recovery_ghost_mse_cos_cka_more": "#c026d3",
    "recovery_ghost_mse_cos_balanced_more": "#fb923c",
    "recovery_ghost_mse_cka_balanced_more": "#a78bfa",
    "recovery_ghost_mse_cos_cka_balanced_more": "#e879f9",
    "recovery_ghost_mse_cka_ema_more": "#2563eb",
    "recovery_ghost_mse_cos_cka_ema_half_more": "#0891b2",
    "recovery_ghost_mse_cos_cka_ema_one_more": "#059669",
    "recovery_ghost_mse_cka": "#0f766e",
    "recovery_ghost_cka_only": "#db2777",
    "recovery_ghost_cka_ridge": "#84cc16",
    "recovery_replaceme_ls": "#8c564b",
    "recovery_replaceme_cos": "#d62728",
    "recovery_replaceme_cos_more": "#be185d",
    "recovery_replaceme_cos_scale": "#f9a8d4",
    "recovery_prunecomp": "#2ca02c",
    "recovery_lora": "#7f7f7f",
    "recovery_train": "#bcbd22",
}

PRUNING_METHOD_LABELS = {
    "gt": "GT",
    "ours": "Ours_CKA",
    "replaceme": "ReplaceMe",
    "shortgpt": "ShortGPT",
    "simpler": "SIMPLER",
    "cosine": "Ours_Cosine",
    "inv_mse": "Ours_INV_MSE",
}
OURS_PRUNING_FAMILIES = frozenset({"ours", "cosine", "inv_mse"})


def pruning_strategy_family(strategy_key: str) -> str:
    key = str(strategy_key).strip().lower()
    for suffix in ("_single", "_double"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def pruning_average_strategy_keys(
    strategy_keys: Sequence[str],
    group: str,
) -> tuple[str, ...]:
    """Select a method-family or interval-structure average-plot group."""
    normalized_group = str(group).strip().lower().replace("-", "_")
    if normalized_group not in {
        "all",
        "ours",
        "ours_single",
        "ours_multi",
        "others",
        "single",
        "non_single",
    }:
        raise ValueError(
            f"unsupported pruning average strategy group {group!r}; "
            "expected all, ours, ours_single, ours_multi, others, single, "
            "or non_single"
        )
    if normalized_group in {"single", "non_single"}:
        return pruning_structure_strategy_keys(strategy_keys, normalized_group)
    if normalized_group in {"ours_single", "ours_multi"}:
        want_single = normalized_group == "ours_single"
        return tuple(
            str(strategy_key)
            for strategy_key in strategy_keys
            if (
                pruning_strategy_family(str(strategy_key)) == "gt"
                or pruning_strategy_family(str(strategy_key))
                in OURS_PRUNING_FAMILIES
            )
            and strategy_is_single_interval(str(strategy_key)) == want_single
        )
    selected: list[str] = []
    for strategy_key in strategy_keys:
        key = str(strategy_key)
        family = pruning_strategy_family(key)
        if (
            normalized_group == "all"
            or family == "gt"
            or (
                normalized_group == "ours"
                and family in OURS_PRUNING_FAMILIES
            )
            or (
                normalized_group == "others"
                and (
                    family == "ours"
                    or family not in OURS_PRUNING_FAMILIES
                )
            )
        ):
            selected.append(key)
    return tuple(selected)


def pruning_structure_strategy_keys(
    strategy_keys: Sequence[str],
    group: str,
) -> tuple[str, ...]:
    """Select Single or the comparison-focused Multi pruning strategies."""
    normalized_group = str(group).strip().lower().replace("-", "_")
    if normalized_group not in {"single", "non_single"}:
        raise ValueError(
            f"unsupported pruning structure group {group!r}; "
            "expected single or non_single"
        )
    want_single = normalized_group == "single"
    return tuple(
        str(strategy_key)
        for strategy_key in strategy_keys
        if strategy_is_single_interval(str(strategy_key)) == want_single
        and (
            want_single
            or pruning_strategy_family(str(strategy_key))
            not in {"cosine", "inv_mse"}
        )
    )


def strategy_is_single_interval(strategy_key: str) -> bool:
    """Treat explicit single variants and SIMPLER as single-interval methods."""
    key = str(strategy_key).strip().lower()
    return key.endswith("_single") or pruning_strategy_family(key) == "simpler"


def strategy_plot_linestyle(strategy_key: str) -> str:
    """Encode Single versus Multi structure independently from method color."""
    key = str(strategy_key).strip().lower()
    return "--" if key == "no_recovery" or strategy_is_single_interval(key) else "-"


def strategy_plot_dashes(strategy_key: str) -> Optional[tuple[float, ...]]:
    """Return a high-contrast dash sequence for non-solid strategy lines."""
    linestyle = strategy_plot_linestyle(strategy_key)
    if linestyle == "--":
        return (5.0, 2.0)
    return None


def strategy_plot_marker(strategy_key: str) -> str:
    """Encode pruning structure in markers as well as line style."""
    return "o" if strategy_is_single_interval(strategy_key) else "s"


def strategy_plot_marker_offset(strategy_key: str) -> float:
    """Horizontally dodge markers without moving their underlying data line."""
    return -0.12 if strategy_is_single_interval(strategy_key) else 0.12


def strategy_plot_zorder(strategy_key: str) -> float:
    """Keep dashed/dotted curves visible above potentially coincident solids."""
    return 3.0 if strategy_plot_linestyle(strategy_key) != "-" else 2.0


def metric_aliases(metric: str) -> tuple[str, ...]:
    key = str(metric)
    if key == "chamfer_distance":
        return ("chamfer_distance", "cd")
    if key == "cd":
        return ("cd", "chamfer_distance")
    return (key,)


def metric_value(values: dict[str, Any], metric: str) -> Optional[float]:
    for key in metric_aliases(metric):
        value = values.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def load_upper_average_for_result_root(result_root: Path) -> dict[str, float]:
    result_root = Path(result_root)
    candidates = [
        result_root.parent / "upper" / "eval_record_v3.json",
        result_root.parent.parent / "upper" / "eval_record_v3.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        average = raw.get("average", {})
        if isinstance(average, dict):
            return {
                str(k): float(v)
                for k, v in average.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            }
    return {}


def load_interval_summary_records(result_root: Path) -> dict[str, dict[str, Any]]:
    """Load multi-interval summaries keyed by canonical physical selector."""
    records: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(Path(result_root).glob("*/summary.json")):
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        selector_raw = str(raw.get("selector") or summary_path.parent.name).strip()
        try:
            selector = canonical_physical_pruning_selector(selector_raw)
        except ValueError:
            continue
        metrics_mean = raw.get("metrics_mean", {})
        if not selector or not isinstance(metrics_mean, dict):
            continue
        candidate = {
            "selector": selector,
            "stored_selector": selector_raw,
            "summary_path": str(summary_path),
            "metrics_mean": {
                str(k): float(v)
                for k, v in metrics_mean.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            },
            "num_scenes": (
                float(raw["num_scenes"])
                if isinstance(raw.get("num_scenes"), (int, float))
                and math.isfinite(float(raw.get("num_scenes")))
                else None
            ),
            "drop_count": (
                int(raw["drop_count"])
                if isinstance(raw.get("drop_count"), (int, float))
                else None
            ),
        }
        existing = records.get(selector)
        # Prefer the canonical directory when both historical interval orders
        # exist; otherwise retain the first valid summary deterministically.
        if existing is None or (
            summary_path.parent.name == selector
            and Path(str(existing.get("summary_path", ""))).parent.name != selector
        ):
            records[selector] = candidate
    return records


def metrics_relative_to_upper(
    metrics_mean: dict[str, Any],
    upper_average: dict[str, Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric_key, value in metrics_mean.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        upper = metric_value(upper_average, str(metric_key))
        if upper is None or upper <= 0.0:
            continue
        if str(metric_key) in HIGHER_IS_BETTER:
            ratio = upper / max(float(value), 1e-8)
        else:
            ratio = float(value) / upper
        if math.isfinite(ratio):
            out[str(metric_key)] = float(max(1.0, ratio))
    return out


def _strategy_points_sorted(strategy_record: dict[str, Any]) -> list[tuple[int, str]]:
    points = strategy_record.get("points", {})
    if not isinstance(points, dict):
        return []
    out: list[tuple[int, str]] = []
    for drop_key, selector in sorted(
        points.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if str(drop_key).isdigit():
            out.append((int(drop_key), str(selector)))
    return out


def collect_pruning_eval_payload(
    *,
    experiment_root: Path,
    prune_plan_root: Path,
    strategies: Optional[dict[str, dict[str, Any]]] = None,
    datasets: Optional[Sequence[str]] = None,
    bundle: Optional[dict[str, Any]] = None,
    min_drop: Optional[int] = None,
    max_drop: Optional[int] = None,
    drop_step: Optional[int] = None,
    eval_selectors: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    if bundle is None:
        bundle = load_prune_plan_bundle(prune_plan_root)
    if strategies is None:
        strategies = bundle_strategy_records(bundle)
    plan_records = bundle_plan_records(bundle)
    requested_eval_selectors = (
        None
        if eval_selectors is None
        else {
            canonical_physical_pruning_selector(value)
            for value in eval_selectors
        }
    )
    source_datasets_raw = bundle.get("dataset", "")
    if isinstance(source_datasets_raw, str):
        gt_route_datasets = {
            name.strip() for name in source_datasets_raw.split(",") if name.strip()
        }
    elif isinstance(source_datasets_raw, Sequence):
        gt_route_datasets = {
            str(name).strip() for name in source_datasets_raw if str(name).strip()
        }
    else:
        gt_route_datasets = set()
    requested_datasets = None if datasets is None else {str(name) for name in datasets}
    dataset_payloads: dict[str, Any] = {}
    for result_root in sorted(Path(experiment_root).glob("*/eval/pruning")):
        summaries = load_interval_summary_records(result_root)
        if not summaries:
            continue
        dataset = result_root.parent.parent.name
        if requested_datasets is not None and dataset not in requested_datasets:
            continue
        upper_average = load_upper_average_for_result_root(result_root)
        relative_by_selector = {
            selector: metrics_relative_to_upper(summary.get("metrics_mean", {}), upper_average)
            for selector, summary in summaries.items()
            if str(selector).startswith("i") and "_j" in str(selector)
        }
        normalization_bounds: dict[str, dict[str, float]] = {}
        for metric_key in SEVEN_METRIC_KEYS:
            values = [
                metric_value(relative, metric_key)
                for relative in relative_by_selector.values()
            ]
            finite_values = [float(value) for value in values if value is not None]
            if not finite_values:
                continue
            _normalized, meta = minmax_normalize_finite(np.asarray(finite_values))
            normalization_bounds[metric_key] = {
                "min": float(meta["min"]),
                "max": float(meta["max"]),
            }
        strategy_payload: dict[str, Any] = {}
        for strategy_key, strategy_record in strategies.items():
            display_name = str(strategy_record.get("display_name", strategy_key))
            points_payload: dict[str, Any] = {}
            for drop_count, plan_selector in _strategy_points_sorted(strategy_record):
                if min_drop is not None and drop_count < int(min_drop):
                    continue
                if max_drop is not None and drop_count > int(max_drop):
                    continue
                if drop_step is not None and int(drop_step) > 0:
                    base = int(min_drop or 0)
                    if (drop_count - base) % int(drop_step) != 0:
                        continue
                record = plan_records.get(plan_selector)
                if isinstance(record, dict):
                    selector = physical_pruning_selector_from_record(record)
                else:
                    try:
                        selector = canonical_physical_pruning_selector(
                            plan_selector
                        )
                    except ValueError as exc:
                        raise FileNotFoundError(
                            f"strategy {strategy_key!r} references missing plan "
                            f"record {plan_selector!r}"
                        ) from exc
                if (
                    requested_eval_selectors is not None
                    and selector not in requested_eval_selectors
                ):
                    continue
                summary = summaries.get(selector)
                if summary is None:
                    points_payload[str(drop_count)] = {
                        "available": False,
                        "selector": selector,
                        "drop_count": int(drop_count),
                        "skip_reason": "missing_exact_summary",
                    }
                    continue
                metrics_mean = dict(summary.get("metrics_mean", {}))
                relative = metrics_relative_to_upper(metrics_mean, upper_average)
                points_payload[str(drop_count)] = {
                    "available": True,
                    "selector": selector,
                    "drop_count": int(drop_count),
                    "summary_path": str(summary.get("summary_path", "")),
                    "num_scenes": summary.get("num_scenes"),
                    "metrics_mean": metrics_mean,
                    "relative_to_upper": relative,
                    "normalized_relative_to_upper": {
                        metric_key: apply_minmax_bounds(
                            value,
                            minimum=normalization_bounds[metric_key]["min"],
                            maximum=normalization_bounds[metric_key]["max"],
                        )
                        for metric_key in SEVEN_METRIC_KEYS
                        for value in [metric_value(relative, metric_key)]
                        if value is not None and metric_key in normalization_bounds
                    },
                }
            strategy_payload[strategy_key] = {
                "display_name": display_name,
                "points": points_payload,
            }
        dataset_payloads[dataset] = {
            "result_root": str(result_root),
            "upper_average": upper_average,
            "normalization_bounds": normalization_bounds,
            "strategies": strategy_payload,
        }
    return _with_dataset_average(
        {
            "schema_version": "strategy_eval_v1",
            "kind": "pruning",
            "gt_route_datasets": sorted(gt_route_datasets),
            "datasets": dataset_payloads,
        }
    )


def collect_recovery_eval_payload(*, experiment_root: Path) -> dict[str, Any]:
    expected_selector_by_drop: dict[int, str] = {}
    try:
        current_bundle = load_prune_plan_bundle(Path(experiment_root))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        current_bundle = {}
    spans = current_bundle.get("spans", {}) if isinstance(current_bundle, dict) else {}
    if isinstance(spans, dict):
        for drop_key, record in spans.items():
            if not str(drop_key).isdigit() or not isinstance(record, dict):
                continue
            try:
                expected_selector_by_drop[int(drop_key)] = (
                    physical_pruning_selector_from_record(record)
                )
            except (KeyError, TypeError, ValueError):
                continue

    datasets: dict[str, Any] = {}
    for recovery_root in sorted(Path(experiment_root).glob("*/eval/recovery")):
        if not recovery_root.is_dir():
            continue
        dataset = recovery_root.parent.parent.name
        upper_average = load_upper_average_for_result_root(recovery_root)
        strategy_payload: dict[str, Any] = {}
        for method_root in sorted(p for p in recovery_root.iterdir() if p.is_dir()):
            if method_root.name not in RECOVERY_METHOD_ORDER:
                continue
            summaries = load_interval_summary_records(method_root)
            if not summaries:
                continue
            points_payload: dict[str, Any] = {}
            for selector, summary in sorted(
                summaries.items(),
                key=lambda item: (
                    int(item[1].get("drop_count") or 10**9),
                    item[0],
                ),
            ):
                drop_count = summary.get("drop_count")
                if drop_count is None:
                    continue
                expected_selector = expected_selector_by_drop.get(int(drop_count))
                if expected_selector is not None and selector != expected_selector:
                    # A method directory can retain summaries from an older
                    # pruning plan.  Never let lexicographic selector order
                    # decide which same-budget result appears in the current
                    # recovery overview.
                    continue
                metrics_mean = dict(summary.get("metrics_mean", {}))
                points_payload[str(int(drop_count))] = {
                    "available": True,
                    "selector": selector,
                    "drop_count": int(drop_count),
                    "summary_path": str(summary.get("summary_path", "")),
                    "num_scenes": summary.get("num_scenes"),
                    "metrics_mean": metrics_mean,
                    "relative_to_upper": metrics_relative_to_upper(metrics_mean, upper_average),
                }
            if points_payload:
                strategy_payload[method_root.name] = {
                    "display_name": method_root.name.removeprefix("recovery_"),
                    "points": points_payload,
                }
        if strategy_payload:
            datasets[dataset] = {
                "result_root": str(recovery_root),
                "upper_average": upper_average,
                "strategies": strategy_payload,
            }
    return _with_dataset_average(
        {"schema_version": "strategy_eval_v1", "kind": "recovery", "datasets": datasets}
    )


def add_recovery_comparison_strategies(
    recovery_payload: dict[str, Any],
    pruning_payload: dict[str, Any],
) -> dict[str, Any]:
    """Add matched GT and unrecovered-pruning baselines to recovery plots."""
    recovery_datasets = recovery_payload.get("datasets", {})
    pruning_datasets = pruning_payload.get("datasets", {})
    if not isinstance(recovery_datasets, dict) or not isinstance(pruning_datasets, dict):
        return recovery_payload

    gt_route_datasets_raw = pruning_payload.get("gt_route_datasets", ())
    gt_route_datasets = (
        {str(name) for name in gt_route_datasets_raw}
        if isinstance(gt_route_datasets_raw, Sequence)
        and not isinstance(gt_route_datasets_raw, (str, bytes))
        else set()
    )
    for dataset, recovery_dataset in recovery_datasets.items():
        pruning_dataset = pruning_datasets.get(dataset)
        if not isinstance(recovery_dataset, dict) or not isinstance(pruning_dataset, dict):
            continue
        recovery_strategies = recovery_dataset.get("strategies", {})
        pruning_strategies = pruning_dataset.get("strategies", {})
        if not isinstance(recovery_strategies, dict) or not isinstance(pruning_strategies, dict):
            continue

        recovery_drop_counts: set[str] = set()
        for strategy in recovery_strategies.values():
            if not isinstance(strategy, dict):
                continue
            points = strategy.get("points", {})
            if not isinstance(points, dict):
                continue
            recovery_drop_counts.update(
                str(drop_key)
                for drop_key, point in points.items()
                if isinstance(point, dict) and bool(point.get("available", False))
            )
        gt_source_key = (
            "gt_double" if "gt_double" in pruning_strategies else "gt"
        )
        comparison_specs: list[tuple[str, str, str]] = []
        if "ours_single" in pruning_strategies:
            comparison_specs.append(
                (
                    "ours_single",
                    "no_recovery_single",
                    "No recovery · CKA single",
                )
            )
        if "ours_double" in pruning_strategies:
            comparison_specs.append(
                (
                    "ours_double",
                    "no_recovery_double",
                    "No recovery · CKA double",
                )
            )
        # Backward compatibility for plan bundles written before explicit
        # single/double CKA strategy variants were introduced.
        if not comparison_specs and "ours" in pruning_strategies:
            comparison_specs.append(("ours", "no_recovery", "No recovery"))
        if dataset in gt_route_datasets:
            comparison_specs.insert(0, (gt_source_key, "gt_drop", "GT route"))
        for source_key, comparison_key, display_name in comparison_specs:
            source = pruning_strategies.get(source_key)
            if not isinstance(source, dict):
                continue
            source_points = source.get("points", {})
            if not isinstance(source_points, dict):
                continue
            # The unrecovered pruning baseline is complete and should show its
            # full curve. GT remains restricted to recovery-tested drop counts
            # so that it is only shown where the comparison is matched.
            comparison_points = {
                str(drop_key): deepcopy(point)
                for drop_key, point in source_points.items()
                if isinstance(point, dict)
                and bool(point.get("available", False))
                and (
                    comparison_key.startswith("no_recovery")
                    or str(drop_key) in recovery_drop_counts
                )
            }
            if comparison_points:
                recovery_strategies[comparison_key] = {
                    "display_name": display_name,
                    "points": comparison_points,
                }

    for dataset_payload in recovery_datasets.values():
        if isinstance(dataset_payload, dict):
            _normalize_strategy_points_for_dataset(dataset_payload)

    # Rebuild the mean from the per-dataset payload after all recovery and
    # no-recovery curves have been normalized together.  Do not replace these
    # baselines with pruning_payload["dataset_average"]: that payload was
    # normalized against a different strategy population, so mixing it with
    # recovery-normalized curves can reverse the 7-metric mean ordering even
    # when all seven raw no-recovery metrics are worse.
    merged = _with_dataset_average(recovery_payload)
    recovery_average = merged.get("dataset_average", {})
    recovery_average_strategies = (
        recovery_average.get("strategies", {})
        if isinstance(recovery_average, dict)
        else {}
    )
    if isinstance(recovery_average_strategies, dict):
        for strategy_key, strategy_payload in recovery_average_strategies.items():
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
                    and 2 <= int(drop_key) <= 24
                    and int(drop_key) % 2 == 0
                }

    gt_average_is_comparable = bool(recovery_datasets) and all(
        dataset in gt_route_datasets
        and isinstance(dataset_payload, dict)
        and "gt_drop" in dataset_payload.get("strategies", {})
        for dataset, dataset_payload in recovery_datasets.items()
    )
    if not gt_average_is_comparable:
        average = merged.get("dataset_average", {})
        if isinstance(average, dict):
            strategies_average = average.get("strategies", {})
            if isinstance(strategies_average, dict):
                strategies_average.pop("gt_drop", None)
    return merged


def _normalize_strategy_points_for_dataset(dataset_payload: dict[str, Any]) -> None:
    """Apply common per-metric bounds to every strategy shown for one dataset."""
    strategies = dataset_payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return
    values_by_metric: dict[str, list[float]] = {key: [] for key in SEVEN_METRIC_KEYS}
    for strategy_payload in strategies.values():
        points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
        if not isinstance(points, dict):
            continue
        for point in points.values():
            if not isinstance(point, dict) or not bool(point.get("available", False)):
                continue
            relative = point.get("relative_to_upper", {})
            if not isinstance(relative, dict):
                continue
            for metric_key in SEVEN_METRIC_KEYS:
                value = metric_value(relative, metric_key)
                if value is not None:
                    values_by_metric[metric_key].append(float(value))

    bounds: dict[str, dict[str, float]] = {}
    for metric_key, values in values_by_metric.items():
        if not values:
            continue
        _normalized, meta = minmax_normalize_finite(np.asarray(values))
        bounds[metric_key] = {"min": float(meta["min"]), "max": float(meta["max"])}
    dataset_payload["normalization_bounds"] = bounds

    for strategy_payload in strategies.values():
        points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
        if not isinstance(points, dict):
            continue
        for point in points.values():
            if not isinstance(point, dict) or not bool(point.get("available", False)):
                continue
            relative = point.get("relative_to_upper", {})
            if not isinstance(relative, dict):
                continue
            point["normalized_relative_to_upper"] = {
                metric_key: apply_minmax_bounds(
                    value,
                    minimum=bounds[metric_key]["min"],
                    maximum=bounds[metric_key]["max"],
                )
                for metric_key in SEVEN_METRIC_KEYS
                for value in [metric_value(relative, metric_key)]
                if value is not None and metric_key in bounds
            }


def _with_dataset_average(payload: dict[str, Any]) -> dict[str, Any]:
    metric_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    relative_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    normalized_relative_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    upper_values: dict[str, list[float]] = {}
    for dataset_payload in payload.get("datasets", {}).values():
        upper_average = dataset_payload.get("upper_average", {})
        if isinstance(upper_average, dict):
            for metric_key, value in upper_average.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_values.setdefault("__upper__", {}).setdefault("0", {}).setdefault(
                        str(metric_key), []
                    ).append(float(value))
                    upper_values.setdefault(str(metric_key), []).append(float(value))
        strategies = dataset_payload.get("strategies", {})
        if not isinstance(strategies, dict):
            continue
        for strategy_key, strategy_payload in strategies.items():
            points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
            if not isinstance(points, dict):
                continue
            for drop_key, point in points.items():
                if not isinstance(point, dict) or not bool(point.get("available", False)):
                    continue
                metrics_mean = point.get("metrics_mean", {})
                if not isinstance(metrics_mean, dict):
                    continue
                for metric_key, value in metrics_mean.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        metric_values.setdefault(str(strategy_key), {}).setdefault(
                            str(drop_key), {}
                        ).setdefault(str(metric_key), []).append(float(value))
                relative = point.get("relative_to_upper", {})
                if isinstance(relative, dict):
                    for metric_key, value in relative.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            relative_values.setdefault(str(strategy_key), {}).setdefault(
                                str(drop_key), {}
                            ).setdefault(str(metric_key), []).append(float(value))
                normalized_relative = point.get("normalized_relative_to_upper", {})
                if isinstance(normalized_relative, dict):
                    for metric_key, value in normalized_relative.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            normalized_relative_values.setdefault(str(strategy_key), {}).setdefault(
                                str(drop_key), {}
                            ).setdefault(str(metric_key), []).append(float(value))

    strategies_average: dict[str, Any] = {}
    strategy_names = sorted(k for k in metric_values.keys() if k != "__upper__")
    display_names = _display_names(payload)
    for strategy_key in strategy_names:
        points_payload: dict[str, Any] = {}
        for drop_key, metrics_for_point in sorted(
            metric_values.get(strategy_key, {}).items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            points_payload[str(drop_key)] = {
                "available": True,
                "drop_count": int(drop_key),
                "metrics_mean": {
                    metric_key: float(np.mean(values))
                    for metric_key, values in sorted(metrics_for_point.items())
                    if values
                },
                "relative_to_upper": {
                    metric_key: float(np.mean(values))
                    for metric_key, values in sorted(
                        relative_values.get(strategy_key, {}).get(str(drop_key), {}).items()
                    )
                    if values
                },
                "normalized_relative_to_upper": {
                    metric_key: float(np.mean(values))
                    for metric_key, values in sorted(
                        normalized_relative_values.get(strategy_key, {}).get(str(drop_key), {}).items()
                    )
                    if values
                },
            }
        strategies_average[strategy_key] = {
            "display_name": display_names.get(strategy_key, strategy_key),
            "points": points_payload,
        }
    payload["dataset_average"] = {
        "upper_average": {
            metric_key: float(np.mean(values))
            for metric_key, values in sorted(upper_values.items())
            if values
        },
        "strategies": strategies_average,
        "num_datasets": len(payload.get("datasets", {})),
    }
    return payload


def _display_names(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for dataset_payload in payload.get("datasets", {}).values():
        strategies = dataset_payload.get("strategies", {})
        if not isinstance(strategies, dict):
            continue
        for key, strategy_payload in strategies.items():
            if isinstance(strategy_payload, dict):
                out[str(key)] = str(strategy_payload.get("display_name", key))
    return out


def write_recovery_eval_plots(
    *,
    experiment_root: Path,
    backbone: str,
    payload: dict[str, Any],
) -> list[Path]:
    written: list[Path] = []
    datasets = payload.get("datasets", {})
    if isinstance(datasets, dict):
        for dataset, dataset_payload in sorted(datasets.items()):
            if not isinstance(dataset_payload, dict):
                continue
            result_root_raw = str(dataset_payload.get("result_root", "")).strip()
            if not result_root_raw:
                continue
            result_root = Path(result_root_raw)
            safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
            out_path = result_root.parent / f"recovery_plan_{safe_ds}_recovery_drop.png"
            path = _write_strategy_metric_plot(
                payload=dataset_payload,
                out_path=out_path,
                title=f"{backbone} / {dataset}: recovery strategies vs upper",
                metric_grid=RECOVERY_METRIC_GRID,
            )
            if path is not None:
                written.append(path)
                for legacy_path in (
                    result_root / f"recovery_plan_{safe_ds}_recovery_drop.png",
                    result_root / f"{safe_ds}_recovery_drop.png",
                ):
                    if legacy_path.is_file():
                        legacy_path.unlink()
    average = payload.get("dataset_average", {})
    if isinstance(average, dict):
        safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
        out_path = (
            Path(experiment_root)
            / f"recovery_plan_{safe_backbone}_recovery_drop.png"
        )
        path = _write_strategy_metric_plot(
            payload=average,
            out_path=out_path,
            title=f"{backbone}: recovery strategies averaged over datasets",
            metric_grid=RECOVERY_METRIC_GRID,
        )
        if path is not None:
            written.append(path)
            legacy_path = (
                Path(experiment_root)
                / f"prune_plan_{safe_backbone}_recovery_drop.png"
            )
            if legacy_path.is_file():
                legacy_path.unlink()
    return written


def write_source_pruning_eval_plots(
    *,
    experiment_root: Path,
    backbone: str,
    payload: dict[str, Any],
) -> list[Path]:
    """Write pruning curves only for the brute-force source datasets."""
    source_names_raw = payload.get("gt_route_datasets", ())
    source_names = (
        {str(name) for name in source_names_raw}
        if isinstance(source_names_raw, Sequence)
        and not isinstance(source_names_raw, (str, bytes))
        else set()
    )
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return []
    source_payload = deepcopy(payload)
    source_payload["datasets"] = {
        str(dataset): dataset_payload
        for dataset, dataset_payload in datasets.items()
        if str(dataset) in source_names and isinstance(dataset_payload, dict)
    }
    if not source_payload["datasets"]:
        return []
    source_payload = _with_dataset_average(source_payload)

    written: list[Path] = []
    for dataset, dataset_payload in sorted(source_payload["datasets"].items()):
        result_root_raw = str(dataset_payload.get("result_root", "")).strip()
        if not result_root_raw:
            continue
        safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
        paths = _write_split_pruning_group_plots(
            payload=dataset_payload,
            out_dir=Path(result_root_raw).parent,
            filename_prefix=safe_ds,
            title_prefix=f"{backbone} / {dataset}",
            metric_grid=PRUNING_METRIC_GRID,
        )
        written.extend(paths)

    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    average = source_payload.get("dataset_average", {})
    if isinstance(average, dict):
        path = _write_strategy_metric_plot(
            payload=average,
            out_path=(
                Path(experiment_root)
                / f"prune_plan_{safe_backbone}_pruning_drop.png"
            ),
            title=f"{backbone}: pruning strategies averaged over source datasets",
            metric_grid=PRUNING_METRIC_GRID,
        )
        if path is not None:
            written.append(path)
    return written


def unavailable_pruning_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return missing
    for dataset, dataset_payload in sorted(datasets.items()):
        strategies = (
            dataset_payload.get("strategies", {})
            if isinstance(dataset_payload, dict)
            else {}
        )
        if not isinstance(strategies, dict):
            continue
        for strategy, strategy_payload in sorted(strategies.items()):
            points = (
                strategy_payload.get("points", {})
                if isinstance(strategy_payload, dict)
                else {}
            )
            if not isinstance(points, dict):
                continue
            for drop_count, point in sorted(
                points.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
            ):
                if isinstance(point, dict) and bool(point.get("available", False)):
                    continue
                missing.append(
                    {
                        "dataset": str(dataset),
                        "strategy": str(strategy),
                        "drop_count": int(drop_count),
                        "selector": (
                            str(point.get("selector", ""))
                            if isinstance(point, dict)
                            else ""
                        ),
                    }
                )
    return missing


def write_pruning_eval_plots(
    *,
    experiment_root: Path,
    backbone: str,
    payload: dict[str, Any],
    write_dataset_plots: bool,
    average_filename: Optional[str] = None,
    average_title: Optional[str] = None,
    average_strategy_group: str = "all",
    require_complete: bool = True,
) -> list[Path]:
    """Write selected pruning curves, optionally with one strict group average."""
    missing = unavailable_pruning_points(payload)
    if require_complete and missing:
        preview = ", ".join(
            f"{item['dataset']}:{item['strategy']}@{item['drop_count']}"
            f"={item['selector']}"
            for item in missing[:12]
        )
        suffix = "" if len(missing) <= 12 else f" (+{len(missing) - 12} more)"
        raise FileNotFoundError(
            f"cannot plot incomplete pruning results: {preview}{suffix}"
        )

    written: list[Path] = []
    datasets = payload.get("datasets", {})
    if write_dataset_plots and isinstance(datasets, dict):
        for dataset, dataset_payload in sorted(datasets.items()):
            if not isinstance(dataset_payload, dict):
                continue
            result_root_raw = str(dataset_payload.get("result_root", "")).strip()
            if not result_root_raw:
                continue
            safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
            paths = _write_split_pruning_group_plots(
                payload=dataset_payload,
                out_dir=Path(result_root_raw).parent,
                filename_prefix=safe_ds,
                title_prefix=f"{backbone} / {dataset}",
                metric_grid=PRUNING_MEAN_METRIC_GRID,
            )
            written.extend(paths)

    if average_filename:
        average = payload.get("dataset_average", {})
        if isinstance(average, dict):
            average_strategies = average.get("strategies", {})
            average_strategy_keys = pruning_average_strategy_keys(
                (
                    tuple(average_strategies)
                    if isinstance(average_strategies, dict)
                    else ()
                ),
                average_strategy_group,
            )
            path = _write_strategy_metric_plot(
                payload=average,
                out_path=Path(experiment_root) / str(average_filename),
                title=(
                    str(average_title)
                    if average_title
                    else f"{backbone}: pruning strategies averaged over datasets"
                ),
                metric_grid=PRUNING_MEAN_METRIC_GRID,
                strategy_keys=average_strategy_keys,
                compact_pruning_legend=True,
            )
            if path is not None:
                written.append(path)
    return written


def _write_split_pruning_group_plots(
    *,
    payload: dict[str, Any],
    out_dir: Path,
    filename_prefix: str,
    title_prefix: str,
    metric_grid: Sequence[Sequence[str]],
) -> list[Path]:
    """Write per-dataset method-family and interval-structure pruning plots."""
    strategies = payload.get("strategies", {})
    strategy_keys = tuple(strategies) if isinstance(strategies, dict) else ()
    written: list[Path] = []
    plot_groups = (
        (
            "ours_single",
            "Ours methods + GT (Single)",
            pruning_average_strategy_keys(strategy_keys, "ours_single"),
        ),
        (
            "ours_multi",
            "Ours methods + GT (Multi)",
            pruning_average_strategy_keys(strategy_keys, "ours_multi"),
        ),
        (
            "ours",
            "Ours methods + GT (Single and Multi)",
            pruning_average_strategy_keys(strategy_keys, "ours"),
        ),
        (
            "others",
            "Ours_CKA + GT + other baselines (Single and Multi)",
            pruning_average_strategy_keys(strategy_keys, "others"),
        ),
    )
    for group, title_group, selected_keys in plot_groups:
        path = _write_strategy_metric_plot(
            payload=payload,
            out_path=Path(out_dir)
            / f"{filename_prefix}_{group}_pruning_drop.png",
            title=f"{title_prefix}: {title_group} vs upper",
            metric_grid=metric_grid,
            strategy_keys=selected_keys,
            compact_pruning_legend=True,
        )
        if path is not None:
            written.append(path)

    if written:
        for legacy_group in ("", "single", "non_single"):
            separator = "_" if legacy_group else ""
            legacy_path = (
                Path(out_dir)
                / f"{filename_prefix}{separator}{legacy_group}_pruning_drop.png"
            )
            if legacy_path.is_file():
                legacy_path.unlink()
    return written


def write_gt_degradation_plot(
    *,
    experiment_root: Path,
    backbone: str,
    payload: dict[str, Any],
) -> Optional[Path]:
    """Write the source-dataset GT route plot without the redundant pruning plots."""
    average = payload.get("dataset_average", {})
    if not isinstance(average, dict):
        return None
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    strategies = average.get("strategies", {})
    strategy_key = (
        "gt_double"
        if isinstance(strategies, dict) and "gt_double" in strategies
        else "gt"
    )
    return _write_strategy_metric_plot(
        payload=average,
        out_path=(
            Path(experiment_root)
            / f"prune_plan_{safe_backbone}_gt_degradation.png"
        ),
        title=f"{backbone}: GT degradation averaged over datasets",
        metric_grid=(("gt_mean",),),
        strategy_keys=(strategy_key,),
    )


def gt_mean_degradation_series(
    payload: dict[str, Any],
    *,
    strategy_key: str = "gt",
) -> list[tuple[int, float]]:
    """Extract the same averaged GT curve used by the gt_mean plot panel."""
    average = payload.get("dataset_average", {})
    if not isinstance(average, dict):
        return []
    strategies = average.get("strategies", {})
    if not isinstance(strategies, dict):
        return []
    strategy = strategies.get(str(strategy_key), {})
    points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
    if not isinstance(points, dict):
        return []
    series: list[tuple[int, float]] = [(0, 0.0)]
    for drop_key, point in sorted(
        points.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if not str(drop_key).isdigit() or not isinstance(point, dict):
            continue
        if not bool(point.get("available", False)):
            continue
        relative = point.get("normalized_relative_to_upper", {})
        if not isinstance(relative, dict):
            continue
        values = [metric_value(relative, metric) for metric in GT_MEAN_METRICS]
        finite = [float(value) for value in values if value is not None]
        if finite:
            series.append((int(drop_key), float(np.mean(finite))))
    return series if len(series) > 1 else []


def select_max_drop_below_mean_degradation(
    payload: dict[str, Any],
    *,
    strategy_key: str = "ours",
    threshold: float = DEFAULT_KNEE_DEGRADATION_THRESHOLD,
    drop_step: int = 1,
) -> dict[str, Any]:
    """Select the largest strategy drop count whose normalized 7-metric mean is allowed."""
    limit = float(threshold)
    if not math.isfinite(limit) or not 0.0 <= limit <= 1.0:
        raise ValueError(f"degradation threshold must be finite and in [0, 1], got {threshold}")
    if isinstance(drop_step, bool) or int(drop_step) <= 0:
        raise ValueError(f"drop_step must be a positive integer, got {drop_step}")
    step = int(drop_step)
    average = payload.get("dataset_average", {})
    strategies = average.get("strategies", {}) if isinstance(average, dict) else {}
    if not isinstance(strategies, dict) or str(strategy_key) not in strategies:
        raise ValueError(f"pruning payload has no strategy {strategy_key!r}")
    strategy = strategies.get(str(strategy_key), {}) if isinstance(strategies, dict) else {}
    points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
    if not isinstance(points, dict):
        points = {}

    candidates: list[tuple[int, float, str]] = [(0, 0.0, "")]
    for drop_key, point in points.items():
        if not str(drop_key).isdigit() or not isinstance(point, dict):
            continue
        if int(drop_key) % step != 0:
            continue
        if not bool(point.get("available", False)):
            continue
        normalized = point.get("normalized_relative_to_upper", {})
        if not isinstance(normalized, dict):
            continue
        values = [metric_value(normalized, metric) for metric in GT_MEAN_METRICS]
        finite = [float(value) for value in values if value is not None]
        if len(finite) != len(GT_MEAN_METRICS):
            continue
        mean_degradation = float(np.mean(finite))
        if mean_degradation <= limit:
            candidates.append(
                (int(drop_key), mean_degradation, str(point.get("selector", "")))
            )
    drop_count, mean_degradation, selector = max(candidates, key=lambda item: item[0])
    return {
        "schema_version": "pruning_knee_v1",
        "method": "max_drop_at_or_below_normalized_7_metric_mean_threshold",
        "strategy": str(strategy_key),
        "threshold": limit,
        "drop_step": step,
        "drop_count": int(drop_count),
        "plan_name": f"prune_span{drop_count}" if drop_count > 0 else "upper",
        **({"selector": selector} if selector else {}),
        "mean_normalized_degradation": mean_degradation,
        "num_metrics": len(GT_MEAN_METRICS),
    }


def write_pruning_strategy_knee_plot(
    *,
    experiment_root: Path,
    backbone: str,
    payload: dict[str, Any],
    knee_points: dict[str, dict[str, Any]],
    threshold: float = DEFAULT_KNEE_DEGRADATION_THRESHOLD,
) -> Optional[Path]:
    """Plot pruning-strategy degradation curves and their selected knees."""
    average = payload.get("dataset_average", {})
    strategies = average.get("strategies", {}) if isinstance(average, dict) else {}
    if not isinstance(strategies, dict):
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for knee plot: {exc}", flush=True)
        return None

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    plotted = False
    for strategy_candidates, label in (
        (("ours_double", "ours"), "Ours_CKA"),
        (("cosine_double", "cosine"), "Ours_Cosine"),
        (("gt_double", "gt"), "GT"),
        (("replaceme_double", "replaceme"), "ReplaceMe"),
        (("inv_mse_double", "inv_mse"), "Ours_INV_MSE"),
        (("shortgpt",), "ShortGPT"),
        (("simpler",), "SIMPLER"),
    ):
        strategy_key = next(
            (key for key in strategy_candidates if key in strategies),
            strategy_candidates[0],
        )
        strategy = strategies.get(strategy_key, {})
        points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
        if not isinstance(points, dict):
            continue
        xs: list[int] = [0]
        ys: list[float] = [0.0]
        for drop_key, point in sorted(
            points.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            if not str(drop_key).isdigit() or not isinstance(point, dict):
                continue
            if not bool(point.get("available", False)):
                continue
            normalized = point.get("normalized_relative_to_upper", {})
            if not isinstance(normalized, dict):
                continue
            values = [metric_value(normalized, metric) for metric in GT_MEAN_METRICS]
            finite = [float(value) for value in values if value is not None]
            if len(finite) != len(GT_MEAN_METRICS):
                continue
            xs.append(int(drop_key))
            ys.append(float(np.mean(finite)))
        if len(xs) <= 1:
            continue
        plotted = True
        color = STRATEGY_PLOT_COLORS.get(strategy_key)
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=2.2, color=color, label=label)
        knee = knee_points.get(strategy_key, {})
        drop_count = knee.get("drop_count") if isinstance(knee, dict) else None
        if isinstance(drop_count, int):
            ax.axvline(
                drop_count,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=f"{label} knee = {drop_count}",
            )
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(
        float(threshold),
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label=f"threshold = {float(threshold):g}",
    )
    ax.set_title(
        f"{backbone}: pruning strategy knees",
        fontweight="bold",
    )
    ax.set_xlabel("Dropped units")
    ax.set_ylabel("7-metric mean normalized degradation")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    safe_backbone = str(backbone).replace("/", "_").replace(" ", "_")
    out_path = (
        Path(experiment_root)
        / f"prune_plan_{safe_backbone}_strategy_knees.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_strategy_metric_figure(
    *,
    payload: dict[str, Any],
    title: str,
    metric_grid: Sequence[Sequence[str]],
    strategy_keys: Optional[Sequence[str]] = None,
    compact_pruning_legend: bool = False,
    legend_fontsize: float = 8.0,
) -> Optional[Any]:
    """Render a strategy metric figure without reading or writing a cached PNG."""
    strategies = payload.get("strategies", {})
    if not isinstance(strategies, dict) or not strategies:
        return None
    if strategy_keys is not None:
        requested = {str(key) for key in strategy_keys}
        strategies = {
            str(key): value
            for key, value in strategies.items()
            if str(key) in requested
        }
        if not strategies:
            return None
    order_index = {key: idx for idx, key in enumerate(STRATEGY_PLOT_ORDER)}
    strategies = dict(
        sorted(
            strategies.items(),
            key=lambda item: (
                order_index.get(str(item[0]), len(order_index)),
                str(item[0]),
            ),
        )
    )
    upper_average = payload.get("upper_average", {})
    if not isinstance(upper_average, dict):
        upper_average = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for strategy eval plot: {exc}", flush=True)
        return None

    legend_fontsize = max(1.0, float(legend_fontsize))
    rows = len(metric_grid)
    cols = len(metric_grid[0])
    figure_width = 7.4 if cols == 1 else 6.2 * cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figure_width, 5.8 * rows),
        squeeze=False,
    )
    any_data = False
    for row_idx, row in enumerate(metric_grid):
        for col_idx, metric_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            upper_value = 0.0 if metric_key == "gt_mean" else metric_value(upper_average, metric_key)
            has_data = upper_value is not None
            if upper_value is not None:
                ax.axhline(
                    float(upper_value),
                    linestyle="--",
                    color="#333333",
                    linewidth=1.2,
                    alpha=0.7,
                    label="upper",
                )
                ax.scatter([0], [float(upper_value)], color="#333333", marker="*", s=90, zorder=4)
            all_xs: list[int] = []
            for strategy_key, strategy_payload in strategies.items():
                if not isinstance(strategy_payload, dict):
                    continue
                display_name = str(strategy_payload.get("display_name", strategy_key))
                points = strategy_payload.get("points", {})
                if not isinstance(points, dict):
                    continue
                xs: list[int] = []
                ys: list[float] = []
                if upper_value is not None:
                    xs.append(0)
                    ys.append(float(upper_value))
                for drop_key, point in sorted(
                    points.items(),
                    key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
                ):
                    if not str(drop_key).isdigit() or not isinstance(point, dict):
                        continue
                    if not bool(point.get("available", False)):
                        continue
                    if metric_key == "gt_mean":
                        relative = point.get("normalized_relative_to_upper", {})
                        if not isinstance(relative, dict):
                            continue
                        relative_values = [
                            metric_value(relative, name) for name in GT_MEAN_METRICS
                        ]
                        finite_values = [
                            float(value) for value in relative_values if value is not None
                        ]
                        value = (
                            float(np.mean(finite_values)) if finite_values else None
                        )
                    else:
                        metrics_mean = point.get("metrics_mean", {})
                        if not isinstance(metrics_mean, dict):
                            continue
                        value = metric_value(metrics_mean, metric_key)
                    if value is None:
                        continue
                    xs.append(int(drop_key))
                    ys.append(float(value))
                    all_xs.append(int(drop_key))
                if len(xs) <= (1 if upper_value is not None else 0):
                    continue
                has_data = True
                any_data = True
                color = STRATEGY_PLOT_COLORS.get(str(strategy_key))
                linestyle = strategy_plot_linestyle(str(strategy_key))
                line_zorder = strategy_plot_zorder(str(strategy_key))
                (line,) = ax.plot(
                    xs,
                    ys,
                    linestyle=linestyle,
                    linewidth=2.6 if linestyle != "-" else 2.0,
                    color=color,
                    alpha=0.9,
                    label=display_name,
                    zorder=line_zorder,
                )
                dashes = strategy_plot_dashes(str(strategy_key))
                if dashes is not None:
                    line.set_dashes(dashes)

                marker_offset = strategy_plot_marker_offset(str(strategy_key))
                marker_xs = [
                    float(x) if int(x) == 0 else float(x) + marker_offset
                    for x in xs
                ]
                is_single = strategy_is_single_interval(str(strategy_key))
                ax.scatter(
                    marker_xs,
                    ys,
                    marker=strategy_plot_marker(str(strategy_key)),
                    s=48,
                    facecolors="none" if is_single else color,
                    edgecolors=color if is_single else "white",
                    linewidths=1.5 if is_single else 0.8,
                    alpha=1.0,
                    zorder=line_zorder + 2.0,
                )
            if not has_data:
                ax.axis("off")
                continue
            ax.set_title(METRIC_HEADERS.get(metric_key, metric_key), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            if all_xs:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            if compact_pruning_legend:
                method_handles = []
                seen_families: set[str] = set()
                for strategy_key in strategies:
                    family = pruning_strategy_family(strategy_key)
                    if family in seen_families:
                        continue
                    seen_families.add(family)
                    method_handles.append(
                        Line2D(
                            [],
                            [],
                            color=STRATEGY_PLOT_COLORS.get(
                                str(strategy_key), "#333333"
                            ),
                            marker="o",
                            linestyle="None",
                            markersize=7,
                            label=PRUNING_METHOD_LABELS.get(
                                family,
                                str(family),
                            ),
                        )
                    )
                style_handles = []
                strategy_names = {str(key).lower() for key in strategies}
                if any(strategy_is_single_interval(key) for key in strategy_names):
                    style_handles.append(
                        Line2D(
                            [], [], color="#222222", linestyle="--",
                            dashes=(5.0, 2.0), marker="o",
                            markerfacecolor="none", markeredgecolor="#222222",
                            linewidth=2.6, label="Single",
                        )
                    )
                if any(
                    not strategy_is_single_interval(key)
                    for key in strategy_names
                ):
                    style_handles.append(
                        Line2D(
                            [], [], color="#222222", linestyle="-",
                            marker="s", markerfacecolor="#222222",
                            linewidth=2.0, label="Multi",
                        )
                    )
                if method_handles:
                    method_legend = ax.legend(
                        handles=method_handles,
                        title="Method (color)",
                        loc="upper left",
                        fontsize=legend_fontsize,
                        title_fontsize=legend_fontsize,
                    )
                    ax.add_artist(method_legend)
                if style_handles:
                    style_legend = ax.legend(
                        handles=style_handles,
                        title="Structure (line + marker)",
                        loc="upper left",
                        fontsize=legend_fontsize,
                        title_fontsize=legend_fontsize,
                    )
                    if method_handles:
                        # Keep both compact legend blocks in the upper-left
                        # without drawing the structure legend over the method
                        # legend.
                        fig.canvas.draw()
                        renderer = fig.canvas.get_renderer()
                        method_bbox = method_legend.get_window_extent(
                            renderer=renderer
                        ).transformed(ax.transAxes.inverted())
                        style_legend.set_bbox_to_anchor(
                            (0.0, method_bbox.y0 - 0.015),
                            transform=ax.transAxes,
                        )
            else:
                ax.legend(loc="best", fontsize=legend_fontsize)
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.97), pad=0.7, h_pad=1.0, w_pad=0.8)
    return fig


def _write_strategy_metric_plot(
    *,
    payload: dict[str, Any],
    out_path: Path,
    title: str,
    metric_grid: Sequence[Sequence[str]],
    strategy_keys: Optional[Sequence[str]] = None,
    compact_pruning_legend: bool = False,
) -> Optional[Path]:
    fig = render_strategy_metric_figure(
        payload=payload,
        title=title,
        metric_grid=metric_grid,
        strategy_keys=strategy_keys,
        compact_pruning_legend=compact_pruning_legend,
    )
    if fig is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.close(fig)
    except Exception:
        pass
    return out_path

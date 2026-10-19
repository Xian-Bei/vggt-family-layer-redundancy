from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from analysis.metric_labels import SEVEN_METRIC_LABELS

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_pruning_strategy_key,
    canonical_physical_pruning_selector,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.redundancy.skip_metric_scaling import (
    GT_METRIC_KEYS,
    normalize_raw_metric_against_upper_worst,
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
PUBLICATION_METRIC_SUBPLOT_WIDTH = 7.4
PUBLICATION_METRIC_SUBPLOT_HEIGHT = 5.8
PUBLICATION_TICK_LABEL_FONTSIZE = 13.0
PUBLICATION_AXIS_LABEL_FONTSIZE = 15.0
PUBLICATION_LEGEND_FONTSIZE = 15.0
PUBLICATION_FLAT_AXES_BOX_ASPECT = 0.704
# Shared flatter geometry for publication figures 5--9 (axes height / width).
PUBLICATION_COMPARISON_AXES_BOX_ASPECT = 0.55
PUBLICATION_COMPARISON_SUBPLOT_HEIGHT = 4.8
PUBLICATION_DETAIL_SUBPLOT_HEIGHT = 6.0
PUBLICATION_DETAIL_AXES_BOX_ASPECT = (
    PUBLICATION_COMPARISON_AXES_BOX_ASPECT
    * PUBLICATION_DETAIL_SUBPLOT_HEIGHT
    / PUBLICATION_COMPARISON_SUBPLOT_HEIGHT
)
GT_MEAN_METRICS = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")
POSE_SELECTION_METRICS = ("ape", "are", "rpe_trans", "rpe_rot")
PRUNING_NORMALIZATION_PROFILE_FILENAME = "normalization_profile.json"
PRUNING_NORMALIZATION_PROFILE_SCHEMA_VERSION = "stage2_upper_zero_worst_all_intervals_v3"
DEFAULT_NORMALIZATION_MAX_DROP = 48
CALIBRATION_DATASET_BY_DATASET = {
    "scannet_dev50": "scannet_dev50",
    "scannet_test50": "scannet_dev50",
    "7scenes_train": "7scenes_train",
    "7scenes_test": "7scenes_train",
    "nuscenes_train": "nuscenes_train",
    "nuscenes_val": "nuscenes_train",
    "waymo_training": "waymo_training",
    "waymo_validation": "waymo_training",
}
METRIC_HEADERS = {
    **SEVEN_METRIC_LABELS,
    "gt_mean": "7-metric mean ↓",
}


def pose_selection_score(metrics: dict[str, Any]) -> Optional[float]:
    """Return the direct four-pose-metric sum used for Single/Double selection."""
    values = [metric_value(metrics, metric) for metric in POSE_SELECTION_METRICS]
    if not all(value is not None and math.isfinite(float(value)) for value in values):
        return None
    return float(sum(float(value) for value in values if value is not None))


def set_y_axis_max_decimals(axis: Any, decimals: int = 2) -> None:
    """Format every y tick with one shared precision up to ``decimals``."""
    from matplotlib.ticker import FuncFormatter  # type: ignore

    max_places = max(0, int(decimals))
    finite_ticks = [
        float(value)
        for value in axis.get_yticks()
        if math.isfinite(float(value))
    ]
    places = max_places
    for candidate in range(max_places + 1):
        if all(
            math.isclose(
                value,
                round(value, candidate),
                rel_tol=1e-9,
                abs_tol=10.0 ** (-(candidate + 7)),
            )
            for value in finite_ticks
        ):
            places = candidate
            break

    def _format_tick(value: float, _position: int) -> str:
        rounded = round(float(value), places)
        if rounded == 0:
            rounded = 0.0
        return f"{rounded:.{places}f}"

    axis.yaxis.set_major_formatter(FuncFormatter(_format_tick))


def move_y_axis_label_to_top(
    axis: Any,
    label: str,
    *,
    fontsize: float,
) -> None:
    """Draw a metric label inside the top center without changing layout."""
    axis.set_ylabel(
        str(label),
        fontsize=float(fontsize),
        fontweight="bold",
    )
    # Retain the original ylabel as an invisible layout placeholder so moving
    # the visible text does not change subplot widths or inter-panel spacing.
    axis.yaxis.label.set_alpha(0.0)
    for artist in list(axis.texts):
        if artist.get_gid() == "metric-label-top":
            artist.remove()
    axis.text(
        0.5,
        0.98,
        str(label),
        ha="center",
        va="top",
        transform=axis.transAxes,
        fontsize=float(fontsize),
        fontweight="bold",
        zorder=20,
        gid="metric-label-top",
    )


STRATEGY_PLOT_ORDER = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
    "gt_drop",
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_interval_sequential_grouped",
    "recovery_ghost_interval_sequential_autogroup",
    "recovery_replaceme_cos",
    "recovery_replaceme_ls",
    "recovery_streamline_ffn",
    "recovery_lora",
    "recovery_train",
    "retention_single",
    "retention_double",
    "CKA_single",
    "CKA_double",
    "cosine_single",
    "cosine_double",
    "inv_mse_single",
    "inv_mse_double",
    "cca_single",
    "cca_double",
    "replaceme_single",
    "replaceme_double",
    "shortgpt",
    "gardener",
    "simpler",
)
RECOVERY_METHOD_ORDER = tuple(
    method for method in STRATEGY_PLOT_ORDER if method.startswith("recovery_")
)
STRATEGY_PLOT_COLORS = {
    "CKA": "#d62728",
    "CKA_single": "#d62728",
    "CKA_double": "#d62728",
    "cosine": "#ff7f0e",
    "cosine_single": "#ff7f0e",
    "cosine_double": "#ff7f0e",
    "retention": "#1f77b4",
    "retention_single": "#1f77b4",
    "retention_double": "#1f77b4",
    "replaceme": "#2ca02c",
    "replaceme_single": "#2ca02c",
    "replaceme_double": "#2ca02c",
    "inv_mse": "#9467bd",
    "inv_mse_single": "#9467bd",
    "inv_mse_double": "#9467bd",
    "cca": "#17becf",
    "cca_single": "#17becf",
    "cca_double": "#17becf",
    "shortgpt": "#8c564b",
    "gardener": "#bcbd22",
    "simpler": "#e377c2",
    "no_recovery_single": "#1f77b4",
    "no_recovery_double": "#1f77b4",
    "no_recovery": "#1f77b4",
    "gt_drop": "#7f7f7f",
    "recovery_linearpatch": "#ff7f0e",
    "recovery_ghost": "#9467bd",
    "recovery_ghost_interval": "#2ca02c",
    "recovery_ghost_interval_sequential": "#aec7e8",
    "recovery_ghost_interval_sequential_grouped": "#c49c94",
    "recovery_ghost_interval_sequential_autogroup": "#d62728",
    "recovery_replaceme_ls": "#8c564b",
    "recovery_replaceme_cos": "#17becf",
    "recovery_streamline_ffn": "#e377c2",
    "recovery_lora": "#7f7f7f",
    "recovery_train": "#bcbd22",
}

PRUNING_METHOD_LABELS = {
    "retention": "Retention",
    "CKA": "CKA",
    "replaceme": "ReplaceMe",
    "shortgpt": "ShortGPT",
    "gardener": "Gardener",
    "simpler": "SIMPLER",
    "cosine": "Cosine",
    "inv_mse": "INV_MSE",
    "cca": "CCA",
}
CKA_PRUNING_FAMILIES = frozenset({"CKA", "cosine", "inv_mse", "cca"})


def pruning_strategy_family(strategy_key: str) -> str:
    key = canonical_pruning_strategy_key(strategy_key)
    for suffix in ("_single", "_double"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def pruning_average_strategy_keys(
    strategy_keys: Sequence[str],
    group: str,
) -> tuple[str, ...]:
    """Select a method-family or interval-structure average-plot group."""
    normalized_group = str(group).strip().casefold().replace("-", "_")
    if normalized_group not in {
        "all",
        "cka",
        "cka_single",
        "cka_multi",
        "others",
        "single",
        "non_single",
    }:
        raise ValueError(
            f"unsupported pruning average strategy group {group!r}; "
            "expected all, CKA, CKA_single, CKA_multi, others, single, "
            "or non_single"
        )
    if normalized_group in {"single", "non_single"}:
        return pruning_structure_strategy_keys(strategy_keys, normalized_group)
    if normalized_group in {"cka_single", "cka_multi"}:
        want_single = normalized_group == "cka_single"
        return tuple(
            str(strategy_key)
            for strategy_key in strategy_keys
            if (
                pruning_strategy_family(str(strategy_key)) == "retention"
                or pruning_strategy_family(str(strategy_key))
                in CKA_PRUNING_FAMILIES
            )
            and strategy_is_single_interval(str(strategy_key)) == want_single
        )
    selected: list[str] = []
    for strategy_key in strategy_keys:
        key = str(strategy_key)
        family = pruning_strategy_family(key)
        if (
            normalized_group == "all"
            or family == "retention"
            or (
                normalized_group == "cka"
                and family in CKA_PRUNING_FAMILIES
            )
            or (
                normalized_group == "others"
                and family not in CKA_PRUNING_FAMILIES
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
            not in {"cosine", "inv_mse", "cca"}
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
    return -0.24 if strategy_is_single_interval(strategy_key) else 0.24


OURS_CURVE_ZORDER = 20.0
OURS_MARKER_ZORDER = 22.0


def display_name_is_ours(display_name: str) -> bool:
    """Identify publication curves that belong to the Ours family."""
    label = str(display_name).strip()
    return label == "Ours" or label.startswith(("Ours(", "Ours ", "Ours·"))


def strategy_plot_zorder(strategy_key: str) -> float:
    """Keep dashed/dotted curves visible above potentially coincident solids."""
    if "ghost_interval_sequential_autogroup" in str(strategy_key).lower():
        return OURS_CURVE_ZORDER
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


def metric_or_gt_mean_value(
    values: dict[str, Any], metric: str
) -> Optional[float]:
    if metric != "gt_mean":
        return metric_value(values, metric)
    finite = [
        value
        for name in GT_MEAN_METRICS
        for value in [metric_value(values, name)]
        if value is not None
    ]
    return float(np.mean(finite)) if finite else None


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
        ratio = float(value) / upper
        if math.isfinite(ratio):
            out[str(metric_key)] = float(max(1.0, ratio))
    return out


def _fit_pruning_normalization_profile(
    *,
    summaries: dict[str, dict[str, Any]],
    upper_average: dict[str, Any],
    result_root: Path,
    max_drop_count: Optional[int],
) -> dict[str, Any]:
    """Fit visualization bounds from all Stage-2 contiguous intervals in scope."""
    maxima: dict[str, float] = {}
    counts: dict[str, int] = {}
    worst_sources: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for selector, summary in summaries.items():
        match = re.fullmatch(r"i([1-9]\d*)_j([1-9]\d*)", str(selector))
        if match is None or not isinstance(summary, dict):
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            continue
        drop_count = end - start + 1
        if max_drop_count is not None and drop_count > int(max_drop_count):
            continue
        candidates.append(
            {
                "selector": str(selector),
                "drop_count": drop_count,
                "metrics_mean": summary.get("metrics_mean", {}),
                "summary_path": summary.get("summary_path", ""),
            }
        )
    for metric_key in GT_METRIC_KEYS:
        values: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            drop_count = candidate.get("drop_count")
            if not isinstance(drop_count, (int, float)):
                continue
            metrics = candidate.get("metrics_mean", {})
            current = metric_value(metrics, metric_key) if isinstance(metrics, dict) else None
            if current is None:
                continue
            if math.isfinite(float(current)):
                values.append((float(current), candidate))
        if values:
            maximum, source = max(
                values,
                key=lambda item: (
                    item[0],
                    int(item[1].get("drop_count", 0)),
                    str(item[1].get("selector", "")),
                ),
            )
            maxima[metric_key] = float(maximum)
            counts[metric_key] = len(values)
            worst_sources[metric_key] = {
                "value": float(maximum),
                "drop_count": int(source.get("drop_count", 0)),
                "selector": str(source.get("selector", "")),
                "summary_path": str(source.get("summary_path", "")),
            }
    signature_payload = {
        "upper_average": {
            key: metric_value(upper_average, key) for key in GT_METRIC_KEYS
        },
        "max_drop_budget": max_drop_count,
        "candidates": [
            {
                "drop_count": int(candidate.get("drop_count", 0)),
                "selector": str(candidate.get("selector", "")),
                "metrics_mean": candidate.get("metrics_mean", {}),
            }
            for candidate in candidates
        ],
    }
    input_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "schema_version": PRUNING_NORMALIZATION_PROFILE_SCHEMA_VERSION,
        "method": "stage2_brute_force_all_interval_raw_metric_max",
        "source": str(result_root),
        "source_dataset": Path(result_root).parent.parent.name,
        "input_signature": input_signature,
        "max_drop_budget": max_drop_count,
        "candidate_scope": "stage2_contiguous_intervals_with_drop_count_at_most_max_drop_budget",
        "upper_average": {
            key: value
            for key in GT_METRIC_KEYS
            for value in [metric_value(upper_average, key)]
            if value is not None
        },
        "maximum_brute_force_metric": maxima,
        # Compatibility aliases for older consumers.
        "maximum_selected_pruning_metric": maxima,
        "maximum_single_interval_metric": maxima,
        "worst_source_by_metric": worst_sources,
        "candidate_points": [
            {
                "drop_count": int(candidate.get("drop_count", 0)),
                "selector": str(candidate.get("selector", "")),
                "summary_path": str(candidate.get("summary_path", "")),
            }
            for candidate in candidates
        ],
        "candidate_counts": counts,
    }


def _write_normalization_profile(result_root: Path, profile: dict[str, Any]) -> Path:
    output = Path(result_root) / PRUNING_NORMALIZATION_PROFILE_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return output


def _load_normalization_profile(result_root: Path) -> Optional[dict[str, Any]]:
    path = Path(result_root) / PRUNING_NORMALIZATION_PROFILE_FILENAME
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict):
        return None
    if saved.get("schema_version") != PRUNING_NORMALIZATION_PROFILE_SCHEMA_VERSION:
        return None
    # Stage 2 is the sole authority for normalization bounds.  Profiles from
    # older plan-dependent schemes are deliberately ignored.
    saved["profile_path"] = str(path)
    return saved


def _normalization_profile_maxima(profile: dict[str, Any]) -> dict[str, Any]:
    maxima = profile.get(
        "maximum_brute_force_metric",
        profile.get(
            "maximum_selected_pruning_metric",
            profile.get("maximum_single_interval_metric", {}),
        ),
    )
    return maxima if isinstance(maxima, dict) else {}


def _adapt_normalization_profile_to_dataset(
    profile: dict[str, Any],
    *,
    dataset: str,
    upper_average: dict[str, Any],
) -> dict[str, Any]:
    """Reuse a calibration Stage-2 span, anchored at this split's upper model."""
    source_upper = profile.get("upper_average", {})
    source_maxima = _normalization_profile_maxima(profile)
    if not isinstance(source_upper, dict):
        source_upper = {}
    maxima: dict[str, float] = {}
    adapted_upper: dict[str, float] = {}
    for metric in GT_METRIC_KEYS:
        source_min = metric_value(source_upper, metric)
        source_max = metric_value(source_maxima, metric)
        target_min = metric_value(upper_average, metric)
        if source_min is None or source_max is None or target_min is None:
            continue
        span = max(0.0, float(source_max) - float(source_min))
        adapted_upper[metric] = float(target_min)
        maxima[metric] = float(target_min) + span
    result = deepcopy(profile)
    result["normalization_source_dataset"] = str(
        profile.get("source_dataset", dataset)
    )
    result["applied_dataset"] = str(dataset)
    result["source_upper_average"] = deepcopy(source_upper)
    result["source_maximum_brute_force_metric"] = deepcopy(source_maxima)
    result["upper_average"] = adapted_upper
    result["maximum_brute_force_metric"] = maxima
    result["maximum_selected_pruning_metric"] = maxima
    result["maximum_single_interval_metric"] = maxima
    return result


def _normalization_profile_for_dataset(
    *,
    experiment_root: Path,
    dataset: str,
    result_root: Path,
    upper_average: dict[str, Any],
) -> Optional[dict[str, Any]]:
    source_dataset = CALIBRATION_DATASET_BY_DATASET.get(str(dataset), str(dataset))
    source_root = (
        Path(result_root)
        if source_dataset == str(dataset)
        else Path(experiment_root) / source_dataset / "eval" / "pruning"
    )
    profile = _load_normalization_profile(source_root)
    if profile is None:
        return None
    return _adapt_normalization_profile_to_dataset(
        profile,
        dataset=str(dataset),
        upper_average=upper_average,
    )


def _apply_metric_profile_to_point(
    point: dict[str, Any],
    *,
    upper_average: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    metrics = point.get("metrics_mean", {})
    if not isinstance(metrics, dict):
        return
    maxima = _normalization_profile_maxima(profile)
    if not isinstance(maxima, dict):
        maxima = {}
    raw: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for metric_key in GT_METRIC_KEYS:
        current = metric_value(metrics, metric_key)
        upper = metric_value(upper_average, metric_key)
        if current is None or upper is None:
            continue
        raw_value = float(current)
        if math.isfinite(raw_value):
            raw[metric_key] = raw_value
        maximum = maxima.get(metric_key)
        if maximum is not None and math.isfinite(raw_value):
            normalized[metric_key] = float(
                normalize_raw_metric_against_upper_worst(
                    np.asarray([raw_value], dtype=np.float64),
                    upper_baseline=float(upper),
                    worst_value=float(maximum),
                    clamp_better_than_upper=False,
                )[0]
            )
    point["raw_metric"] = raw
    point["normalized_metric"] = normalized
    point["normalized_relative_to_upper"] = dict(normalized)


def _normalize_metric_map(
    values: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, float]:
    maxima = _normalization_profile_maxima(profile)
    upper_average = profile.get("upper_average", {})
    if not isinstance(maxima, dict):
        return {}
    if not isinstance(upper_average, dict):
        return {}
    normalized: dict[str, float] = {}
    for metric_key in GT_METRIC_KEYS:
        value = metric_value(values, metric_key)
        upper = metric_value(upper_average, metric_key)
        maximum = maxima.get(metric_key)
        if value is None or upper is None or maximum is None:
            continue
        normalized[metric_key] = float(
            normalize_raw_metric_against_upper_worst(
                np.asarray([float(value)], dtype=np.float64),
                upper_baseline=float(upper),
                worst_value=float(maximum),
                clamp_better_than_upper=False,
            )[0]
        )
    return normalized


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
                point_payload = {
                    "available": True,
                    "selector": selector,
                    "drop_count": int(drop_count),
                    "summary_path": str(summary.get("summary_path", "")),
                    "num_scenes": summary.get("num_scenes"),
                    "metrics_mean": metrics_mean,
                    "relative_to_upper": relative,
                }
                points_payload[str(drop_count)] = point_payload
            strategy_payload[strategy_key] = {
                "display_name": display_name,
                "points": points_payload,
            }
        normalization_profile = _normalization_profile_for_dataset(
            experiment_root=Path(experiment_root),
            dataset=dataset,
            result_root=result_root,
            upper_average=upper_average,
        )
        if normalization_profile is None and (
            CALIBRATION_DATASET_BY_DATASET.get(dataset, dataset) == dataset
        ):
            fallback = _fit_pruning_normalization_profile(
                summaries=summaries,
                upper_average=upper_average,
                result_root=result_root,
                max_drop_count=DEFAULT_NORMALIZATION_MAX_DROP,
            )
            normalization_profile = _adapt_normalization_profile_to_dataset(
                fallback,
                dataset=dataset,
                upper_average=upper_average,
            )
        normalization_profile = normalization_profile or {}
        for strategy in strategy_payload.values():
            points = strategy.get("points", {}) if isinstance(strategy, dict) else {}
            for point in points.values() if isinstance(points, dict) else ():
                if not isinstance(point, dict) or not point.get("available", False):
                    continue
                _apply_metric_profile_to_point(
                    point,
                    upper_average=upper_average,
                    profile=normalization_profile,
                )
        dataset_payloads[dataset] = {
            "result_root": str(result_root),
            "upper_average": upper_average,
            "normalized_upper_average": _normalize_metric_map(
                upper_average, normalization_profile
            ),
            "normalization_profile": normalization_profile,
            "normalization_bounds": {
                metric: {
                    "min": float(metric_value(upper_average, metric) or 0.0),
                    "max": float(maximum),
                }
                for metric, maximum in _normalization_profile_maxima(
                    normalization_profile
                ).items()
            },
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


def collect_recovery_eval_payload(
    *,
    experiment_root: Path,
    prune_strategy: Optional[str] = None,
    prune_plan_root: Optional[Path] = None,
) -> dict[str, Any]:
    selected_prune_strategy = canonical_pruning_strategy_key(prune_strategy or "")
    expected_selector_by_drop: dict[int, str] = {}
    try:
        current_bundle = load_prune_plan_bundle(Path(prune_plan_root or experiment_root))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        current_bundle = {}
    expected_records: dict[str, Any] = {}
    if selected_prune_strategy:
        strategies = bundle_strategy_records(current_bundle) if current_bundle else {}
        strategy_record = strategies.get(selected_prune_strategy, {})
        points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
        plan_records = bundle_plan_records(current_bundle) if current_bundle else {}
        if isinstance(points, dict):
            expected_records = {
                str(drop_key): plan_records.get(str(selector))
                for drop_key, selector in points.items()
            }
    else:
        spans = current_bundle.get("spans", {}) if isinstance(current_bundle, dict) else {}
        if isinstance(spans, dict):
            expected_records = spans
    for drop_key, record in expected_records.items():
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
        summaries_by_method: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        allowed_selectors = set(expected_selector_by_drop.values())
        for selector_root in sorted(path for path in recovery_root.iterdir() if path.is_dir()):
            if selector_root.name.startswith("recovery_"):
                # Compatibility with the legacy method/selector layout.
                if selector_root.name not in RECOVERY_METHOD_ORDER:
                    continue
                for selector, summary in load_interval_summary_records(selector_root).items():
                    summaries_by_method.setdefault(selector_root.name, []).append(
                        (selector, summary)
                    )
                continue
            if selected_prune_strategy and selector_root.name == selected_prune_strategy:
                # Compatibility with the short-lived strategy/method/selector layout.
                for method_root in selector_root.iterdir():
                    if method_root.is_dir() and method_root.name in RECOVERY_METHOD_ORDER:
                        for selector, summary in load_interval_summary_records(method_root).items():
                            summaries_by_method.setdefault(method_root.name, []).append(
                                (selector, summary)
                            )
                continue
            if allowed_selectors and selector_root.name not in allowed_selectors:
                continue
            for method_root in sorted(path for path in selector_root.iterdir() if path.is_dir()):
                if method_root.name not in RECOVERY_METHOD_ORDER:
                    continue
                summary_path = method_root / "summary.json"
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(summary, dict):
                    summaries_by_method.setdefault(method_root.name, []).append(
                        (selector_root.name, {**summary, "summary_path": str(summary_path)})
                    )
        for method_name, summaries in sorted(summaries_by_method.items()):
            points_payload: dict[str, Any] = {}
            for selector, summary in sorted(
                summaries,
                key=lambda item: (int(item[1].get("drop_count") or 10**9), item[0]),
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
                strategy_payload[method_name] = {
                    "display_name": method_name.removeprefix("recovery_"),
                    "points": points_payload,
                }
        if strategy_payload:
            datasets[dataset] = {
                "result_root": str(recovery_root),
                "upper_average": upper_average,
                "strategies": strategy_payload,
            }
    payload = _with_dataset_average(
        {"schema_version": "strategy_eval_v1", "kind": "recovery", "datasets": datasets}
    )
    payload["prune_strategy"] = selected_prune_strategy
    return payload


def add_recovery_comparison_strategies(
    recovery_payload: dict[str, Any],
    pruning_payload: dict[str, Any],
    prune_strategy: Optional[str] = None,
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
        retention_source_key = (
            "retention_double"
            if "retention_double" in pruning_strategies
            else "retention"
        )
        selected_prune_strategy = canonical_pruning_strategy_key(
            prune_strategy or ""
        )
        comparison_specs: list[tuple[str, str, str]] = []
        if selected_prune_strategy and selected_prune_strategy in pruning_strategies:
            source_payload = pruning_strategies.get(selected_prune_strategy, {})
            display_name = (
                str(source_payload.get("display_name", selected_prune_strategy))
                if isinstance(source_payload, dict)
                else selected_prune_strategy
            )
            comparison_specs.append(
                (selected_prune_strategy, "no_recovery", f"No recovery · {display_name}")
            )
        elif "CKA_single" in pruning_strategies:
            comparison_specs.append(
                (
                    "CKA_single",
                    "no_recovery_single",
                    "No recovery · CKA single",
                )
            )
        if not selected_prune_strategy and "CKA_double" in pruning_strategies:
            comparison_specs.append(
                (
                    "CKA_double",
                    "no_recovery_double",
                    "No recovery · CKA double",
                )
            )
        if not comparison_specs and "CKA" in pruning_strategies:
            comparison_specs.append(("CKA", "no_recovery", "No recovery · CKA"))
        if dataset in gt_route_datasets:
            comparison_specs.insert(
                0, (retention_source_key, "gt_drop", "GT route")
            )
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

    missing_profile_datasets: list[str] = []
    for dataset, dataset_payload in recovery_datasets.items():
        if not isinstance(dataset_payload, dict):
            continue
        pruning_dataset = pruning_datasets.get(dataset, {})
        profile = (
            pruning_dataset.get("normalization_profile", {})
            if isinstance(pruning_dataset, dict)
            else {}
        )
        maxima = (
            profile.get(
                "maximum_brute_force_metric",
                profile.get(
                    "maximum_selected_pruning_metric",
                    profile.get("maximum_single_interval_metric", {}),
                ),
            )
            if isinstance(profile, dict)
            else {}
        )
        if not isinstance(maxima, dict) or not maxima:
            dataset_payload["normalization_skip_reason"] = (
                "missing_complete_pruning_normalization_profile"
            )
            missing_profile_datasets.append(str(dataset))
            continue
        missing_metrics = [metric for metric in GT_METRIC_KEYS if metric not in maxima]
        if missing_metrics:
            dataset_payload["normalization_incomplete_metrics"] = missing_metrics
        dataset_payload["normalization_profile"] = deepcopy(profile)
        dataset_payload["normalization_bounds"] = {
            metric: {
                "min": float(metric_value(profile.get("upper_average", {}), metric) or 0.0),
                "max": float(maxima[metric]),
            }
            for metric in GT_METRIC_KEYS
            if metric in maxima
        }
        upper_average = profile.get("upper_average", {})
        if not isinstance(upper_average, dict) or not upper_average:
            upper_average = dataset_payload.get("upper_average", {})
        if isinstance(upper_average, dict):
            dataset_payload["normalized_upper_average"] = _normalize_metric_map(
                upper_average, profile
            )
        strategies = dataset_payload.get("strategies", {})
        if not isinstance(upper_average, dict) or not isinstance(strategies, dict):
            continue
        for strategy_payload in strategies.values():
            points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
            if not isinstance(points, dict):
                continue
            for point in points.values():
                if isinstance(point, dict) and bool(point.get("available", False)):
                    _apply_metric_profile_to_point(
                        point,
                        upper_average=upper_average,
                        profile=profile,
                    )
    recovery_payload["skipped_normalization_datasets"] = missing_profile_datasets

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


def _with_dataset_average(payload: dict[str, Any]) -> dict[str, Any]:
    metric_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    relative_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    normalized_relative_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    plotted_metric_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        "raw_metric": {},
        "normalized_metric": {},
    }
    upper_values: dict[str, list[float]] = {}
    normalized_upper_values: dict[str, list[float]] = {}
    normalization_bound_values: dict[str, list[float]] = {}
    normalization_bound_min_values: dict[str, list[float]] = {}
    for dataset_payload in payload.get("datasets", {}).values():
        upper_average = dataset_payload.get("upper_average", {})
        if isinstance(upper_average, dict):
            for metric_key, value in upper_average.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_values.setdefault("__upper__", {}).setdefault("0", {}).setdefault(
                        str(metric_key), []
                    ).append(float(value))
                    upper_values.setdefault(str(metric_key), []).append(float(value))
        normalized_upper = dataset_payload.get("normalized_upper_average", {})
        if isinstance(normalized_upper, dict):
            for metric_key, value in normalized_upper.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    normalized_upper_values.setdefault(str(metric_key), []).append(
                        float(value)
                    )
        normalization_bounds = dataset_payload.get("normalization_bounds", {})
        if isinstance(normalization_bounds, dict):
            for metric_key, bounds in normalization_bounds.items():
                minimum = bounds.get("min") if isinstance(bounds, dict) else None
                maximum = bounds.get("max") if isinstance(bounds, dict) else None
                if isinstance(minimum, (int, float)) and math.isfinite(float(minimum)):
                    normalization_bound_min_values.setdefault(str(metric_key), []).append(
                        float(minimum)
                    )
                if isinstance(maximum, (int, float)) and math.isfinite(float(maximum)):
                    normalization_bound_values.setdefault(str(metric_key), []).append(
                        float(maximum)
                    )
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
                for field, field_values in plotted_metric_values.items():
                    metric_map = point.get(field, {})
                    if not isinstance(metric_map, dict):
                        continue
                    for metric_key, value in metric_map.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            field_values.setdefault(str(strategy_key), {}).setdefault(
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
                **{
                    field: {
                        metric_key: float(np.mean(values))
                        for metric_key, values in sorted(
                            field_values.get(strategy_key, {}).get(str(drop_key), {}).items()
                        )
                        if values
                    }
                    for field, field_values in plotted_metric_values.items()
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
        "normalized_upper_average": {
            metric_key: float(np.mean(values))
            for metric_key, values in sorted(normalized_upper_values.items())
            if values
        },
        "normalization_bounds": {
            metric_key: {
                "min": float(np.mean(normalization_bound_min_values.get(metric_key, [0.0]))),
                "max": float(np.mean(values)),
            }
            for metric_key, values in sorted(normalization_bound_values.items())
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
            "CKA_single",
            "Similarity methods + Retention (Single)",
            pruning_average_strategy_keys(strategy_keys, "CKA_single"),
        ),
        (
            "CKA_multi",
            "Similarity methods + Retention (Multi)",
            pruning_average_strategy_keys(strategy_keys, "CKA_multi"),
        ),
        (
            "CKA",
            "Similarity methods + Retention (Single and Multi)",
            pruning_average_strategy_keys(strategy_keys, "CKA"),
        ),
        (
            "others",
            "Retention + other baselines (Single and Multi)",
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
        "retention_double"
        if isinstance(strategies, dict) and "retention_double" in strategies
        else "retention"
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
    strategy_key: str = "retention",
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
    normalized_upper = average.get("normalized_upper_average", {})
    upper_value = (
        metric_or_gt_mean_value(normalized_upper, "gt_mean")
        if isinstance(normalized_upper, dict)
        else None
    )
    series: list[tuple[int, float]] = (
        [(0, float(upper_value))] if upper_value is not None else []
    )
    for drop_key, point in sorted(
        points.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if not str(drop_key).isdigit() or not isinstance(point, dict):
            continue
        if not bool(point.get("available", False)):
            continue
        relative = point.get("normalized_metric", {})
        if not isinstance(relative, dict):
            continue
        values = [metric_value(relative, metric) for metric in GT_MEAN_METRICS]
        finite = [float(value) for value in values if value is not None]
        if finite:
            series.append((int(drop_key), float(np.mean(finite))))
    return series if len(series) > 1 else []


def render_strategy_metric_figure(
    *,
    payload: dict[str, Any],
    title: str,
    metric_grid: Sequence[Sequence[str]],
    strategy_keys: Optional[Sequence[str]] = None,
    compact_pruning_legend: bool = False,
    legend_fontsize: float = 8.0,
    normalize_metrics: Optional[bool] = None,
    always_normalize_metric_mean: bool = False,
    include_upper_anchor: bool = True,
    subplot_width: Optional[float] = None,
    subplot_box_aspect: Optional[float] = None,
    strategy_style_overrides: Optional[dict[str, dict[str, Any]]] = None,
    default_linewidth: Optional[float] = None,
    upper_linewidth: float = 1.2,
    draw_solid_before_dashed: bool = False,
    dodge_structure_markers: bool = True,
    shared_xlabel: Optional[str] = "Dropped Layers",
    bottom_row_xlabel: Optional[str] = None,
    all_axes_xlabel: bool = False,
    show_figure_title: bool = True,
    show_metric_titles: bool = True,
    metric_labels_as_ylabels: bool = False,
    metric_ylabel: Optional[str] = None,
    y_tick_percentage: bool = False,
    y_tick_max_decimals: Optional[int] = None,
    cap_y_at_normalization_max: bool = True,
    minimum_y_axis_upper: Optional[float] = None,
    fixed_y_axis_upper: Optional[float] = None,
    upper_anchor_bottom_margin_fraction: Optional[float] = None,
    x_tick_step: Optional[float] = None,
    tick_label_fontsize: float = 10.0,
    axis_label_fontsize: float = 11.0,
    legend_markerscale: float = 1.0,
    legend_handlelength: float = 2.0,
    legend_fontweight: str = "normal",
    legend_loc: str = "best",
    upper_label: str = "upper",
    legend_order: Optional[Sequence[str]] = None,
    compact_method_labels: Optional[dict[str, str]] = None,
    compact_method_order: Optional[Sequence[str]] = None,
    compact_structure_labels: Optional[Sequence[str]] = None,
    compact_legend_titles: tuple[str, str] = (
        "Method (color)",
        "Structure (line + marker)",
    ),
    combine_compact_legend: bool = False,
    shared_ylabel: Optional[str] = None,
    figure_height_per_row: float = PUBLICATION_METRIC_SUBPLOT_HEIGHT,
    fixed_layout_margins: Optional[tuple[float, float, float, float]] = None,
    fixed_layout_wspace: Optional[float] = None,
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
    if draw_solid_before_dashed:
        strategies = dict(sorted(
            strategies.items(),
            key=lambda item: strategy_plot_linestyle(str(item[0])) != "-",
        ))
    upper_average = payload.get("upper_average", {})
    if not isinstance(upper_average, dict):
        upper_average = {}
    normalized_upper_average = payload.get("normalized_upper_average", {})
    if not isinstance(normalized_upper_average, dict):
        normalized_upper_average = {}
    normalization_bounds = payload.get("normalization_bounds", {})
    if not isinstance(normalization_bounds, dict):
        normalization_bounds = {}
    modern_scaling = normalize_metrics is not None
    normalized_view = bool(normalize_metrics) if modern_scaling else False
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from matplotlib.ticker import (  # type: ignore
            MaxNLocator,
            MultipleLocator,
            PercentFormatter,
        )
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for strategy eval plot: {exc}", flush=True)
        return None

    legend_fontsize = max(1.0, float(legend_fontsize))
    rows = len(metric_grid)
    cols = len(metric_grid[0])
    figure_width = (
        max(1.0, float(subplot_width)) * cols
        if subplot_width is not None
        else (
            PUBLICATION_METRIC_SUBPLOT_WIDTH
            if cols == 1
            else 6.2 * cols
        )
    )
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figure_width, max(1.0, float(figure_height_per_row)) * rows),
        squeeze=False,
    )
    any_data = False
    style_overrides = strategy_style_overrides or {}
    method_labels = compact_method_labels or PRUNING_METHOD_LABELS

    def _style_legend_text(legend: Any) -> None:
        for text in legend.get_texts():
            text.set_fontweight(str(legend_fontweight))
        legend.get_title().set_fontweight(str(legend_fontweight))

    normalization_states = [
        bool(
            normalized_view
            or (always_normalize_metric_mean and metric_key == "gt_mean")
        )
        for row in metric_grid
        for metric_key in row
    ]
    for row_idx, row in enumerate(metric_grid):
        for col_idx, metric_key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if subplot_box_aspect is not None:
                ax.set_box_aspect(float(subplot_box_aspect))
            metric_normalized_view = bool(
                normalized_view
                or (always_normalize_metric_mean and metric_key == "gt_mean")
            )
            value_field = (
                "normalized_metric"
                if metric_normalized_view
                else "raw_metric"
            )
            if modern_scaling:
                upper_value = metric_or_gt_mean_value(
                    normalized_upper_average
                    if metric_normalized_view
                    else upper_average,
                    metric_key,
                )
            else:
                upper_value = 0.0 if metric_key == "gt_mean" else metric_value(upper_average, metric_key)
            has_data = upper_value is not None
            if upper_value is not None:
                ax.axhline(
                    float(upper_value),
                    linestyle="--",
                    color="#333333",
                    linewidth=float(upper_linewidth),
                    alpha=0.7,
                    label=str(upper_label),
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
                if include_upper_anchor and upper_value is not None:
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
                    if modern_scaling:
                        metric_map = point.get(value_field, {})
                        if not isinstance(metric_map, dict):
                            continue
                        if metric_key == "gt_mean":
                            component_values = [
                                metric_value(metric_map, name) for name in GT_MEAN_METRICS
                            ]
                            finite_values = [
                                float(item) for item in component_values if item is not None
                            ]
                            value = float(np.mean(finite_values)) if finite_values else None
                        else:
                            value = metric_value(metric_map, metric_key)
                    elif metric_key == "gt_mean":
                        relative = point.get("normalized_relative_to_upper", {})
                        if not isinstance(relative, dict):
                            continue
                        relative_values = [metric_value(relative, name) for name in GT_MEAN_METRICS]
                        finite_values = [float(item) for item in relative_values if item is not None]
                        value = float(np.mean(finite_values)) if finite_values else None
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
                if len(xs) <= (1 if include_upper_anchor and upper_value is not None else 0):
                    continue
                has_data = True
                any_data = True
                style_override = style_overrides.get(str(strategy_key), {})
                color = style_override.get(
                    "color", STRATEGY_PLOT_COLORS.get(str(strategy_key))
                )
                linestyle = str(
                    style_override.get(
                        "linestyle", strategy_plot_linestyle(str(strategy_key))
                    )
                )
                linewidth = float(
                    style_override.get(
                        "linewidth",
                        default_linewidth
                        if default_linewidth is not None
                        else (2.6 if linestyle != "-" else 2.0),
                    )
                )
                alpha = float(style_override.get("alpha", 0.9))
                line_zorder = float(
                    style_override.get(
                        "zorder", strategy_plot_zorder(str(strategy_key))
                    )
                )
                if display_name_is_ours(display_name):
                    line_zorder = max(line_zorder, OURS_CURVE_ZORDER)
                (line,) = ax.plot(
                    xs,
                    ys,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha,
                    label=display_name,
                    zorder=line_zorder,
                )
                dashes = style_override.get(
                    "dashes", strategy_plot_dashes(str(strategy_key))
                )
                if dashes is not None:
                    line.set_dashes(dashes)

                structure_marker_offset = (
                    strategy_plot_marker_offset(str(strategy_key))
                    if dodge_structure_markers
                    else 0.0
                )
                marker_offset = structure_marker_offset + float(
                    style_override.get("marker_offset", 0.0)
                )
                marker_xs = [
                    float(x) if int(x) == 0 else float(x) + marker_offset
                    for x in xs
                ]
                is_single = strategy_is_single_interval(str(strategy_key))
                marker_zorder = float(
                    style_override.get("marker_zorder", line_zorder + 2.0)
                )
                if display_name_is_ours(display_name):
                    marker_zorder = max(marker_zorder, OURS_MARKER_ZORDER)
                ax.scatter(
                    marker_xs,
                    ys,
                    marker=strategy_plot_marker(str(strategy_key)),
                    s=float(style_override.get("markersize", 48)),
                    facecolors="none" if is_single else color,
                    edgecolors=color if is_single else "white",
                    linewidths=float(
                        style_override.get(
                            "markeredgewidth", 1.5 if is_single else 0.8
                        )
                    ),
                    alpha=float(style_override.get("marker_alpha", 1.0)),
                    zorder=marker_zorder,
                )
            if not has_data:
                ax.axis("off")
                continue
            metric_title = METRIC_HEADERS.get(metric_key, metric_key)
            if show_metric_titles:
                ax.set_title(metric_title, fontsize=11, fontweight="bold")
            if metric_labels_as_ylabels:
                ax.set_ylabel(
                    str(metric_ylabel or metric_title),
                    fontsize=float(axis_label_fontsize),
                    fontweight="bold",
                )
            if bottom_row_xlabel and (
                all_axes_xlabel or row_idx == rows - 1
            ):
                ax.set_xlabel(
                    str(bottom_row_xlabel),
                    fontsize=float(axis_label_fontsize),
                    fontweight="bold",
                )
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            if metric_key == "gt_mean":
                component_maxima = [
                    bounds.get("max")
                    for name in GT_MEAN_METRICS
                    for bounds in [normalization_bounds.get(name)]
                    if isinstance(bounds, dict)
                    and isinstance(bounds.get("max"), (int, float))
                    and math.isfinite(float(bounds["max"]))
                ]
                raw_axis_upper = (
                    float(np.mean(component_maxima))
                    if component_maxima
                    else None
                )
            else:
                metric_bounds = normalization_bounds.get(metric_key, {})
                raw_maximum = (
                    metric_bounds.get("max")
                    if isinstance(metric_bounds, dict)
                    else None
                )
                raw_axis_upper = (
                    float(raw_maximum)
                    if isinstance(raw_maximum, (int, float))
                    and math.isfinite(float(raw_maximum))
                    else None
                )
            axis_upper = 1.0 if metric_normalized_view and raw_axis_upper is not None else raw_axis_upper
            if cap_y_at_normalization_max and axis_upper is not None:
                ax.set_ylim(top=axis_upper)
            if minimum_y_axis_upper is not None:
                ax.set_ylim(
                    top=max(
                        float(minimum_y_axis_upper),
                        float(ax.get_ylim()[1]),
                    )
                )
            if fixed_y_axis_upper is not None:
                ax.set_ylim(top=float(fixed_y_axis_upper))
            if (
                upper_anchor_bottom_margin_fraction is not None
                and upper_value is not None
            ):
                margin_fraction = max(
                    0.0,
                    float(upper_anchor_bottom_margin_fraction),
                )
                current_top = float(ax.get_ylim()[1])
                upper_scalar = float(upper_value)
                span_above_upper = current_top - upper_scalar
                if span_above_upper > 0.0:
                    ax.set_ylim(
                        bottom=(
                            upper_scalar
                            - margin_fraction * span_above_upper
                        )
                    )
            if y_tick_percentage:
                ax.yaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0)
                )
            elif y_tick_max_decimals is not None:
                set_y_axis_max_decimals(ax, y_tick_max_decimals)
            if all_xs:
                if x_tick_step is not None and float(x_tick_step) > 0:
                    ax.xaxis.set_major_locator(
                        MultipleLocator(float(x_tick_step))
                    )
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.tick_params(axis="both", labelsize=float(tick_label_fontsize))
            if compact_pruning_legend:
                method_handles = []
                seen_families: set[str] = set()
                legend_strategy_keys = list(strategies)
                if compact_method_order is not None:
                    method_order = {
                        str(family): index
                        for index, family in enumerate(compact_method_order)
                    }
                    legend_strategy_keys.sort(
                        key=lambda key: method_order.get(
                            pruning_strategy_family(key), len(method_order)
                        )
                    )
                for strategy_key in legend_strategy_keys:
                    family = pruning_strategy_family(strategy_key)
                    if family in seen_families:
                        continue
                    seen_families.add(family)
                    method_handles.append(
                        Line2D(
                            [],
                            [],
                            color=style_overrides.get(
                                str(strategy_key), {}
                            ).get(
                                "color",
                                STRATEGY_PLOT_COLORS.get(
                                    str(strategy_key), "#333333"
                                ),
                            ),
                            marker="o",
                            linestyle="None",
                            markersize=7,
                            label=method_labels.get(
                                family,
                                str(family),
                            ),
                        )
                    )
                style_handles = []
                strategy_names = {str(key).lower() for key in strategies}
                shown_structure_labels = (
                    None
                    if compact_structure_labels is None
                    else {str(label) for label in compact_structure_labels}
                )
                multi_structure_label = (
                    "Double"
                    if shown_structure_labels is not None
                    and "Double" in shown_structure_labels
                    else "Multi"
                )
                if (
                    shown_structure_labels is None
                    or "Single" in shown_structure_labels
                ) and any(
                    strategy_is_single_interval(key) for key in strategy_names
                ):
                    single_key = next(
                        key
                        for key in strategies
                        if strategy_is_single_interval(key)
                    )
                    single_override = style_overrides.get(single_key, {})
                    single_linestyle = str(
                        single_override.get(
                            "linestyle", strategy_plot_linestyle(single_key)
                        )
                    )
                    single_handle = Line2D(
                        [], [],
                        color=single_override.get("color", "#222222"),
                        linestyle=single_linestyle,
                        marker="o", markerfacecolor="none",
                        markeredgecolor=single_override.get("color", "#222222"),
                        linewidth=(
                            float(default_linewidth)
                            if default_linewidth is not None
                            else float(single_override.get("linewidth", 2.6))
                        ),
                        alpha=float(single_override.get("alpha", 1.0)),
                        label="Single",
                    )
                    single_dashes = single_override.get(
                        "dashes", strategy_plot_dashes(single_key)
                    )
                    if single_dashes is not None:
                        single_handle.set_dashes(single_dashes)
                    style_handles.append(single_handle)
                if (
                    shown_structure_labels is None
                    or multi_structure_label in shown_structure_labels
                ) and any(
                    not strategy_is_single_interval(key)
                    for key in strategy_names
                ):
                    style_handles.append(
                        Line2D(
                            [], [], color="#222222", linestyle="-",
                            marker="s", markerfacecolor="#222222",
                            linewidth=(
                                float(default_linewidth)
                                if default_linewidth is not None
                                else 2.0
                            ),
                            label=multi_structure_label,
                        )
                    )
                if combine_compact_legend and (method_handles or style_handles):
                    combined_legend = ax.legend(
                        handles=[*method_handles, *style_handles],
                        loc="upper left",
                        fontsize=legend_fontsize,
                        markerscale=float(legend_markerscale),
                        handlelength=float(legend_handlelength),
                    )
                    _style_legend_text(combined_legend)
                elif method_handles:
                    method_legend = ax.legend(
                        handles=method_handles,
                        title=str(compact_legend_titles[0]),
                        loc="upper left",
                        fontsize=legend_fontsize,
                        title_fontsize=legend_fontsize,
                        markerscale=float(legend_markerscale),
                        handlelength=float(legend_handlelength),
                    )
                    _style_legend_text(method_legend)
                    ax.add_artist(method_legend)
                if style_handles and not combine_compact_legend:
                    style_legend = ax.legend(
                        handles=style_handles,
                        title=str(compact_legend_titles[1]),
                        loc="upper left",
                        fontsize=legend_fontsize,
                        title_fontsize=legend_fontsize,
                        markerscale=float(legend_markerscale),
                        handlelength=float(legend_handlelength),
                    )
                    _style_legend_text(style_legend)
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
                legend_kwargs: dict[str, Any] = {}
                if legend_order is not None:
                    handles, labels = ax.get_legend_handles_labels()
                    preferred = {
                        str(label): index for index, label in enumerate(legend_order)
                    }
                    ordered_entries = []
                    seen_labels: set[str] = set()
                    for original_index, (handle, label) in enumerate(
                        zip(handles, labels)
                    ):
                        label = str(label)
                        if label in seen_labels:
                            continue
                        seen_labels.add(label)
                        ordered_entries.append(
                            (
                                preferred.get(label, len(preferred)),
                                original_index,
                                handle,
                                label,
                            )
                        )
                    ordered_entries.sort(key=lambda item: (item[0], item[1]))
                    legend_kwargs = {
                        "handles": [item[2] for item in ordered_entries],
                        "labels": [item[3] for item in ordered_entries],
                    }
                legend = ax.legend(
                    loc=str(legend_loc),
                    fontsize=legend_fontsize,
                    markerscale=float(legend_markerscale),
                    handlelength=float(legend_handlelength),
                    **legend_kwargs,
                )
                _style_legend_text(legend)
    for row_idx, row in enumerate(metric_grid):
        for col_idx in range(len(row), cols):
            axes[row_idx][col_idx].axis("off")
    if not any_data:
        plt.close(fig)
        return None
    figure_title = title
    if modern_scaling and normalization_states:
        if all(normalization_states):
            normalization_note = "Metrics: normalized"
        elif not any(normalization_states):
            normalization_note = "Metrics: raw"
        else:
            normalization_note = "Individual metrics: raw · 7-metric mean: normalized"
        figure_title = f"{title}\n{normalization_note}"
    if show_figure_title:
        fig.suptitle(figure_title, fontsize=14, fontweight="bold", y=0.995)
    bottom = 0.01
    if shared_xlabel:
        fig.supxlabel(
            str(shared_xlabel),
            fontsize=float(axis_label_fontsize),
            fontweight="bold",
        )
        bottom = 0.035
    if shared_ylabel:
        fig.supylabel(
            str(shared_ylabel),
            fontsize=float(axis_label_fontsize),
            fontweight="bold",
        )
    top = 0.97 if show_figure_title else 0.99
    if fixed_layout_margins is None:
        fig.tight_layout(
            rect=(0.01, bottom, 0.99, top),
            pad=0.7,
            h_pad=1.0,
            w_pad=0.8,
        )
    else:
        left, fixed_bottom, right, fixed_top = (
            float(value) for value in fixed_layout_margins
        )
        fig.subplots_adjust(
            left=left,
            bottom=fixed_bottom,
            right=right,
            top=fixed_top,
            wspace=(
                None
                if fixed_layout_wspace is None
                else float(fixed_layout_wspace)
            ),
        )
    return fig


def emphasize_figure(fig: Any, *, linewidth_scale: float = 1.25) -> None:
    """Bold every label and slightly strengthen every visible figure stroke."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.spines import Spine
    from matplotlib.text import Text

    scale = max(1.0, float(linewidth_scale))
    for text in fig.findobj(match=Text):
        text.set_fontweight("bold")
    for line in fig.findobj(match=Line2D):
        line.set_linewidth(line.get_linewidth() * scale)
        line.set_markeredgewidth(line.get_markeredgewidth() * scale)
    for collection in fig.findobj(match=Collection):
        widths = collection.get_linewidths()
        if len(widths):
            collection.set_linewidths(np.asarray(widths) * scale)
    for patch in fig.findobj(match=Patch):
        if not isinstance(patch, Spine) and patch.get_linewidth() > 0:
            patch.set_linewidth(patch.get_linewidth() * scale)
    for axis in getattr(fig, "axes", ()):
        for spine in axis.spines.values():
            spine.set_linewidth(spine.get_linewidth() * scale)


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

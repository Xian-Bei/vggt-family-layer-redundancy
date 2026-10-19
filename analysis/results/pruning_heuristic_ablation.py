"""Manifest and reporting helpers for pruning-heuristic ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_physical_pruning_selector,
    load_prune_plan_bundle,
)
from analysis.prune_plan.eval_plots import collect_pruning_eval_payload
from analysis.redundancy.skip_metric_scaling import GT_METRIC_KEYS


SCHEMA_VERSION = "pruning_heuristic_ablation_v1"
MANIFEST_FILENAME = "pruning_heuristic_ablation_manifest.json"
RESULT_JSON_FILENAME = "pruning_heuristic_ablation_results.json"
RESULT_CSV_FILENAME = "pruning_heuristic_ablation_results.csv"
STRATEGIES = (
    "retention_single",
    "retention_double",
    "CKA_single",
    "CKA_double",
)
VARIANT_ORDER = (
    "baseline",
    "no_single_double",
)
VARIANT_TOLERANCES = {
    "baseline": 0.005,
    "no_single_double": 0.0,
}


def _split(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.replace(",", " ").split()
    else:
        values = [str(value) for value in raw]
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_selector_or_raw(value: str) -> str:
    try:
        return canonical_physical_pruning_selector(value)
    except ValueError:
        return str(value)


def _existing_selectors(
    source_output_root: Path,
    backbones: Sequence[str],
    datasets: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for backbone in backbones:
        result[str(backbone)] = {}
        for dataset in datasets:
            pruning_root = (
                Path(source_output_root)
                / str(backbone)
                / str(dataset)
                / "eval"
                / "pruning"
            )
            result[str(backbone)][str(dataset)] = sorted(
                _canonical_selector_or_raw(path.name)
                for path in pruning_root.iterdir()
                if path.is_dir() and (path / "summary.json").is_file()
            ) if pruning_root.is_dir() else []
    return result


def init_manifest(
    *,
    output_root: Path,
    source_output_root: Path,
    backbones: Sequence[str],
    training_datasets: Sequence[str],
    heldout_datasets: Sequence[str],
    min_drop: int,
    max_drop: int,
    drop_step: int,
    variants: Sequence[str] = VARIANT_ORDER,
    strategies: Sequence[str] = STRATEGIES,
) -> Path:
    output_root = Path(output_root)
    source_output_root = Path(source_output_root)
    manifest_path = output_root / MANIFEST_FILENAME
    variant_names = list(dict.fromkeys(str(value) for value in variants))
    strategy_names = list(dict.fromkeys(str(value) for value in strategies))
    unknown_variants = sorted(set(variant_names) - set(VARIANT_ORDER))
    unknown_strategies = sorted(set(strategy_names) - set(STRATEGIES))
    if unknown_variants or unknown_strategies:
        raise ValueError(
            f"Unknown variants/strategies: {unknown_variants}/{unknown_strategies}"
        )

    expected = {
        "source_output_root": str(source_output_root.resolve()),
        "backbones": list(backbones),
        "training_datasets": list(training_datasets),
        "heldout_datasets": list(heldout_datasets),
        "min_drop": int(min_drop),
        "max_drop": int(max_drop),
        "drop_step": int(drop_step),
        "variants": {
            variant: {
                "single_double_comparison_tolerance": VARIANT_TOLERANCES[variant],
            }
            for variant in variant_names
        },
        "strategies": strategy_names,
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Existing ablation manifest has incompatible settings: {mismatches}"
            )
        return manifest_path

    datasets = [*training_datasets, *heldout_datasets]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **expected,
        "token_region_candidates": ["all", "special_patch_mean"],
        "initial_existing_selectors": _existing_selectors(
            source_output_root, backbones, datasets
        ),
        "eval_events": [],
    }
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def record_eval_event(
    *,
    output_root: Path,
    variant: str,
    backbone: str,
    selector: str,
    datasets: Sequence[str],
    phase: str,
) -> None:
    manifest_path = Path(output_root) / MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    event = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "variant": str(variant),
        "backbone": str(backbone),
        "selector": str(selector),
        "datasets": list(datasets),
        "phase": str(phase),
    }
    events = manifest.setdefault("eval_events", [])
    if not isinstance(events, list):
        raise ValueError("manifest eval_events must be a list")
    identity = {
        key: event[key]
        for key in ("variant", "backbone", "selector", "datasets", "phase")
    }
    if not any(
        isinstance(existing, dict)
        and all(existing.get(key) == value for key, value in identity.items())
        for existing in events
    ):
        events.append(event)
        _atomic_write_json(manifest_path, manifest)


def _selector_origin(
    manifest: Mapping[str, Any],
    *,
    variant: str,
    backbone: str,
    dataset: str,
    selector: str,
) -> str:
    initial = manifest.get("initial_existing_selectors", {})
    selectors = (
        initial.get(backbone, {}).get(dataset, [])
        if isinstance(initial, dict)
        and isinstance(initial.get(backbone), dict)
        else []
    )
    if selector in selectors:
        return "preexisting_main"
    events = manifest.get("eval_events", [])
    producers = [
        str(event.get("variant"))
        for event in events
        if isinstance(event, dict)
        and event.get("backbone") == backbone
        and event.get("selector") == selector
        and dataset in event.get("datasets", [])
    ] if isinstance(events, list) else []
    if not producers:
        return "unknown"
    return (
        "generated_by_ablation"
        if producers[0] == variant
        else "reused_from_earlier_variant"
    )


def _mean(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _winner_token_region(
    bundle: Mapping[str, Any], strategy: str, drop_count: int
) -> str:
    strategy_record = bundle_strategy_records(dict(bundle)).get(strategy, {})
    points = strategy_record.get("points", {}) if isinstance(strategy_record, dict) else {}
    plan_key = points.get(str(drop_count)) if isinstance(points, dict) else None
    record = bundle_plan_records(dict(bundle)).get(str(plan_key), {})
    if strategy.startswith("CKA_") and isinstance(record, dict):
        selection = record.get("token_region_selection", {})
        if isinstance(selection, dict):
            return str(selection.get("winner_token_region", ""))
    return "all" if strategy.startswith("retention_") else ""


def _validate_bundle(
    bundle: Mapping[str, Any],
    *,
    variant: str,
    backbone: str,
    expected_strategies: Sequence[str] = STRATEGIES,
) -> None:
    strategies = bundle_strategy_records(dict(bundle))
    if set(strategies) != set(expected_strategies):
        raise ValueError(
            f"{variant}/{backbone}: expected exactly {list(expected_strategies)}, "
            f"got {sorted(strategies)}"
        )
    expected = VARIANT_TOLERANCES[variant]
    controls = bundle.get("extra", {}).get("similarity_selection_controls", {})
    if not isinstance(controls, dict):
        raise ValueError(f"{variant}/{backbone}: missing similarity_selection_controls")
    actual = float(controls.get("single_double_comparison_tolerance", math.nan))
    if actual != expected:
        raise ValueError(
            f"{variant}/{backbone}: tolerance mismatch {actual} != {expected}"
        )
    for strategy in set(expected_strategies).intersection(("CKA_single", "CKA_double")):
        points = strategies.get(strategy, {}).get("points", {})
        for drop_count in points if isinstance(points, dict) else ():
            if not _winner_token_region(bundle, strategy, int(drop_count)):
                raise ValueError(
                    f"{variant}/{backbone}: {strategy}@{drop_count} has no "
                    "token-region selection provenance"
                )


def collect_rows(
    *, output_root: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(str(manifest["source_output_root"]))
    min_drop = int(manifest["min_drop"])
    max_drop = int(manifest["max_drop"])
    drop_step = int(manifest["drop_step"])
    training = set(str(value) for value in manifest["training_datasets"])
    heldout = set(str(value) for value in manifest["heldout_datasets"])
    requested_datasets = [*manifest["training_datasets"], *manifest["heldout_datasets"]]
    variants = list(manifest["variants"])
    strategies = list(manifest["strategies"])
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for variant in variants:
        for backbone in manifest["backbones"]:
            plan_root = Path(output_root) / variant / backbone
            bundle = load_prune_plan_bundle(plan_root)
            _validate_bundle(
                bundle,
                variant=variant,
                backbone=backbone,
                expected_strategies=strategies,
            )
            payload = collect_pruning_eval_payload(
                experiment_root=source_root / backbone,
                prune_plan_root=plan_root,
                bundle=bundle,
                strategies={key: bundle["strategies"][key] for key in strategies},
                datasets=requested_datasets,
                min_drop=min_drop,
                max_drop=max_drop,
                drop_step=drop_step,
            )
            for dataset in requested_datasets:
                dataset_payload = payload.get("datasets", {}).get(dataset)
                if not isinstance(dataset_payload, dict):
                    missing.append({
                        "variant": variant,
                        "backbone": backbone,
                        "dataset": dataset,
                        "reason": "missing_dataset_payload",
                    })
                    continue
                scope = "training" if dataset in training else "heldout" if dataset in heldout else "unknown"
                for strategy in strategies:
                    strategy_payload = dataset_payload.get("strategies", {}).get(strategy, {})
                    points = strategy_payload.get("points", {}) if isinstance(strategy_payload, dict) else {}
                    for drop_count in range(min_drop, max_drop + 1, drop_step):
                        point = points.get(str(drop_count)) if isinstance(points, dict) else None
                        if not isinstance(point, dict) or not point.get("available", False):
                            missing.append({
                                "variant": variant,
                                "backbone": backbone,
                                "dataset": dataset,
                                "strategy": strategy,
                                "drop_count": drop_count,
                                "selector": point.get("selector", "") if isinstance(point, dict) else "",
                                "reason": "missing_summary",
                            })
                            continue
                        raw = point.get("metrics_mean", {})
                        normalized = point.get("normalized_relative_to_upper", {})
                        selector = str(point.get("selector", ""))
                        row: dict[str, Any] = {
                            "variant": variant,
                            "backbone": backbone,
                            "scope": scope,
                            "dataset": dataset,
                            "strategy": strategy,
                            "family": "CKA" if strategy.startswith("CKA_") else "retention",
                            "structure": "single" if strategy.endswith("_single") else "double",
                            "drop_count": drop_count,
                            "selector": selector,
                            "winner_token_region": _winner_token_region(bundle, strategy, drop_count),
                            "eval_origin": _selector_origin(
                                manifest,
                                variant=variant,
                                backbone=backbone,
                                dataset=dataset,
                                selector=selector,
                            ),
                            "num_scenes": point.get("num_scenes"),
                            "summary_path": point.get("summary_path", ""),
                        }
                        for metric in GT_METRIC_KEYS:
                            row[f"raw_{metric}"] = raw.get(metric) if isinstance(raw, dict) else None
                            row[f"normalized_degradation_{metric}"] = (
                                normalized.get(metric) if isinstance(normalized, dict) else None
                            )
                        row["gt_mean"] = _mean(
                            row[f"normalized_degradation_{metric}"]
                            for metric in GT_METRIC_KEYS
                        )
                        rows.append(row)
    baseline = {
        (row["backbone"], row["dataset"], row["strategy"], row["drop_count"]): row["gt_mean"]
        for row in rows
        if row["variant"] == "baseline"
    }
    for row in rows:
        base = baseline.get(
            (row["backbone"], row["dataset"], row["strategy"], row["drop_count"])
        )
        row["delta_gt_mean_vs_baseline"] = (
            None if base is None or row["gt_mean"] is None else float(row["gt_mean"] - base)
        )
    return rows, missing


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["variant"]),
            str(row["backbone"]),
            str(row["scope"]),
            str(row["strategy"]),
            int(row["drop_count"]),
        )].append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        variant, backbone, scope, strategy, drop_count = key
        record: dict[str, Any] = {
            "variant": variant,
            "backbone": backbone,
            "scope": scope,
            "strategy": strategy,
            "drop_count": drop_count,
            "num_datasets": len(group),
            "gt_mean": _mean(row.get("gt_mean") for row in group),
            "delta_gt_mean_vs_baseline": _mean(
                row.get("delta_gt_mean_vs_baseline") for row in group
            ),
        }
        for metric in GT_METRIC_KEYS:
            field = f"normalized_degradation_{metric}"
            record[field] = _mean(row.get(field) for row in group)
        raw_sums = []
        for row in group:
            values = [row.get(f"raw_{metric}") for metric in GT_METRIC_KEYS]
            if all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in values
            ):
                raw_sums.append(sum(float(value) for value in values))
        record["raw_metric_sum"] = _mean(raw_sums)
        aggregates.append(record)
    return aggregates


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "variant", "backbone", "scope", "dataset", "strategy", "family",
        "structure", "drop_count", "selector", "winner_token_region",
        "eval_origin", "num_scenes", "summary_path",
        *(f"raw_{metric}" for metric in GT_METRIC_KEYS),
        *(f"normalized_degradation_{metric}" for metric in GT_METRIC_KEYS),
        "gt_mean", "delta_gt_mean_vs_baseline",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(output_root: Path, aggregates: Sequence[Mapping[str, Any]]) -> list[str]:
    figure_root = Path(output_root) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {
        "baseline": "#111111",
        "no_single_double": "#1f77b4",
    }
    labels = {
        "baseline": "Similarity tolerance enabled",
        "no_single_double": "No similarity Single/Double tolerance",
    }
    backbones = sorted({str(row["backbone"]) for row in aggregates})

    def curve(scope: str, strategy: str, variant: str, backbone: str | None = None):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in aggregates:
            if row["scope"] != scope or row["strategy"] != strategy or row["variant"] != variant:
                continue
            if backbone is not None and row["backbone"] != backbone:
                continue
            if row.get("gt_mean") is not None:
                grouped[int(row["drop_count"])].append(float(row["gt_mean"]))
        return [(drop, float(np.mean(values))) for drop, values in sorted(grouped.items())]

    for backbone in backbones:
        for scope in ("training", "heldout"):
            for family, prefix in (("retention", "retention"), ("CKA", "CKA")):
                fig, axis = plt.subplots(figsize=(8.2, 4.8))
                for variant in VARIANT_ORDER:
                    strategy = f"{prefix}_double"
                    points = sorted(
                        (
                            (int(row["drop_count"]), float(row["gt_mean"]))
                            for row in aggregates
                            if row["backbone"] == backbone
                            and row["scope"] == scope
                            and row["variant"] == variant
                            and row["strategy"] == strategy
                            and row.get("gt_mean") is not None
                        ),
                        key=lambda item: item[0],
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [point[1] for point in points],
                            marker="o",
                            color=colors[variant],
                            label=labels[variant],
                        )
                single_strategy = f"{prefix}_single"
                single = sorted(
                    (
                        (int(row["drop_count"]), float(row["gt_mean"]))
                        for row in aggregates
                        if row["backbone"] == backbone
                        and row["scope"] == scope
                        and row["variant"] == "baseline"
                        and row["strategy"] == single_strategy
                        and row.get("gt_mean") is not None
                    ),
                    key=lambda item: item[0],
                )
                if single:
                    axis.plot(
                        [point[0] for point in single],
                        [point[1] for point in single],
                        linestyle="--",
                        color="#7f7f7f",
                        label="single reference",
                    )
                axis.set_title(f"{backbone} · {scope} · {family}")
                axis.set_xlabel("Dropped units")
                axis.set_ylabel("Normalized 7-metric degradation ↓")
                axis.grid(alpha=0.25)
                axis.legend()
                fig.tight_layout()
                path = figure_root / f"{backbone}_{scope}_{family.lower()}.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                written.append(str(path))

    families = (("Retention", "retention"), ("CKA", "CKA"))
    scopes = (("Training", "training"), ("Held-out", "heldout"))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)
    for row_index, (scope_label, scope) in enumerate(scopes):
        for column_index, (family_label, prefix) in enumerate(families):
            axis = axes[row_index, column_index]
            for variant in VARIANT_ORDER:
                points = curve(scope, f"{prefix}_double", variant)
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o", color=colors[variant], label=labels[variant],
                )
            single = curve(scope, f"{prefix}_single", "baseline")
            axis.plot(
                [point[0] for point in single],
                [point[1] for point in single],
                linestyle="--", color="#7f7f7f", label="Single reference",
            )
            axis.set_title(f"{scope_label} · {family_label}")
            axis.set_xlabel("Dropped units")
            axis.set_ylabel("Normalized 7-metric degradation ↓")
            axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Pruning heuristic ablation: absolute degradation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figure_root / "heuristic_ablation_absolute_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True, sharey=True)
    for row_index, (scope_label, scope) in enumerate(scopes):
        for column_index, (family_label, prefix) in enumerate(families):
            axis = axes[row_index, column_index]
            baseline = dict(curve(scope, f"{prefix}_double", "baseline"))
            for variant in VARIANT_ORDER[1:]:
                points = [
                    (drop, value - baseline[drop])
                    for drop, value in curve(scope, f"{prefix}_double", variant)
                    if drop in baseline
                ]
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o", color=colors[variant], label=labels[variant],
                )
            axis.axhline(0.0, color="#111111", linewidth=1.0, linestyle="--")
            axis.set_title(f"{scope_label} · {family_label}")
            axis.set_xlabel("Dropped units")
            axis.set_ylabel("Δ degradation vs. baseline (lower is better)")
            axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Effect of disabling pruning heuristics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figure_root / "heuristic_ablation_delta_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(
        2, len(backbones), figsize=(5.2 * len(backbones), 8.0), squeeze=False,
        sharex=True,
    )
    for row_index, (family_label, prefix) in enumerate(families):
        for column_index, backbone in enumerate(backbones):
            axis = axes[row_index, column_index]
            for variant in VARIANT_ORDER:
                points = curve("heldout", f"{prefix}_double", variant, backbone)
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o", color=colors[variant], label=labels[variant],
                )
            single = curve("heldout", f"{prefix}_single", "baseline", backbone)
            axis.plot(
                [point[0] for point in single],
                [point[1] for point in single],
                linestyle="--", color="#7f7f7f", label="Single reference",
            )
            axis.set_title(f"{backbone} · {family_label}")
            axis.set_xlabel("Dropped units")
            axis.set_ylabel("Held-out normalized degradation ↓")
            axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Held-out pruning heuristic ablation by backbone", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = figure_root / "heuristic_ablation_heldout_by_backbone.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(str(path))
    return written


def _write_four_row_raw_sum_figure(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Compare similarity tolerance enabled vs. disabled on held-out data."""
    figure_root = Path(output_root) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#111111",
        "no_single_double": "#1f77b4",
    }
    labels = {
        "baseline": "Similarity tolerance enabled",
        "no_single_double": "No similarity Single/Double tolerance",
    }
    variants = VARIANT_ORDER
    backbones = sorted({str(row["backbone"]) for row in rows})
    row_specs = (
        ("Same-dataset", ("scannet_test50", "nuscenes_val"), "ScanNet test + nuScenes val", "Retention", "retention_double"),
        ("Same-dataset", ("scannet_test50", "nuscenes_val"), "ScanNet test + nuScenes val", "CKA", "CKA_double"),
        ("Cross-data", ("7scenes_test", "waymo_validation"), "7-Scenes test + Waymo validation", "Retention", "retention_double"),
        ("Cross-data", ("7scenes_test", "waymo_validation"), "7-Scenes test + Waymo validation", "CKA", "CKA_double"),
    )
    fig, axes = plt.subplots(
        len(row_specs),
        len(backbones),
        figsize=(5.3 * len(backbones), 4.0 * len(row_specs)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, (
        generalization_label,
        datasets,
        dataset_caption,
        family_label,
        strategy,
    ) in enumerate(row_specs):
        for column_index, backbone in enumerate(backbones):
            axis = axes[row_index, column_index]
            for variant in variants:
                values_by_drop: dict[int, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row["dataset"] not in datasets
                        or row["strategy"] != strategy
                        or row["variant"] != variant
                        or row["backbone"] != backbone
                        or int(row["drop_count"]) > 22
                    ):
                        continue
                    values = [row.get(f"raw_{metric}") for metric in GT_METRIC_KEYS]
                    if all(
                        isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in values
                    ):
                        values_by_drop[int(row["drop_count"])].append(
                            sum(float(value) for value in values)
                        )
                points = sorted(
                    (drop_count, float(np.mean(values)))
                    for drop_count, values in values_by_drop.items()
                )
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    color=colors[variant],
                    label=labels[variant],
                )
            axis.set_title(
                f"{backbone} · {generalization_label} · {family_label}\n"
                f"{dataset_caption}"
            )
            axis.set_xlabel("Dropped units")
            axis.set_ylabel("Sum of 7 raw metrics ↓")
            axis.grid(alpha=0.25)
            axis.relim()
            axis.autoscale_view(tight=True)
            axis.margins(x=0.02, y=0.05)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Held-out generalization: similarity Single/Double tolerance ablation",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = figure_root / "heuristic_ablation_heldout_by_backbone.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [str(path)]


def _write_main_results_raw_sum_figure(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_datasets: Sequence[str],
    same_dataset_eval_datasets: Sequence[str],
) -> list[str]:
    """Plot main-result calibration and same-dataset Double-plan curves."""
    figure_root = Path(output_root) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#111111",
        "no_single_double": "#1f77b4",
    }
    labels = {
        "baseline": "Similarity tolerance enabled",
        "no_single_double": "No similarity Single/Double tolerance",
    }
    variants = tuple(
        variant
        for variant in VARIANT_ORDER
        if any(row["variant"] == variant for row in rows)
    )
    backbones = sorted({str(row["backbone"]) for row in rows})
    row_specs = (
        ("Calibration", set(calibration_datasets), "Retention", "retention_double"),
        ("Calibration", set(calibration_datasets), "CKA", "CKA_double"),
        (
            "Same-dataset eval",
            set(same_dataset_eval_datasets),
            "Retention",
            "retention_double",
        ),
        ("Same-dataset eval", set(same_dataset_eval_datasets), "CKA", "CKA_double"),
    )
    fig, axes = plt.subplots(
        len(row_specs),
        len(backbones),
        figsize=(5.3 * len(backbones), 4.0 * len(row_specs)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, (scope_label, datasets, family_label, strategy) in enumerate(row_specs):
        for column_index, backbone in enumerate(backbones):
            axis = axes[row_index, column_index]
            for variant in variants:
                values_by_drop: dict[int, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row["dataset"] not in datasets
                        or row["strategy"] != strategy
                        or row["variant"] != variant
                        or row["backbone"] != backbone
                    ):
                        continue
                    values = [row.get(f"raw_{metric}") for metric in GT_METRIC_KEYS]
                    if all(
                        isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in values
                    ):
                        values_by_drop[int(row["drop_count"])].append(
                            sum(float(value) for value in values)
                        )
                points = sorted(
                    (drop_count, float(np.mean(values)))
                    for drop_count, values in values_by_drop.items()
                )
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    color=colors[variant],
                    label=labels[variant],
                )
            axis.set_title(f"{backbone} · {scope_label} · {family_label}")
            axis.set_xlabel("Dropped units")
            axis.set_ylabel("Mean sum of 7 raw metrics ↓")
            axis.grid(alpha=0.25)
            axis.relim()
            axis.autoscale_view(tight=True)
            axis.margins(x=0.02, y=0.05)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Main-results pruning heuristic ablation (Double plans)", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = figure_root / "heuristic_ablation_main_results.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [str(path)]


def summarize(output_root: Path, *, plot_mode: str = "heldout-generalization") -> dict[str, Any]:
    output_root = Path(output_root)
    manifest = _load_json(output_root / MANIFEST_FILENAME)
    rows, missing = collect_rows(output_root=output_root, manifest=manifest)
    if missing:
        preview = ", ".join(
            f"{item.get('variant')}/{item.get('backbone')}/{item.get('dataset')}/"
            f"{item.get('strategy', '?')}@{item.get('drop_count', '?')}"
            for item in missing[:12]
        )
        raise FileNotFoundError(
            f"Ablation results are incomplete: {preview}"
            + (f" (+{len(missing) - 12} more)" if len(missing) > 12 else "")
        )
    aggregates = aggregate_rows(rows)
    if plot_mode == "main-results":
        figures = _write_main_results_raw_sum_figure(
            output_root,
            rows,
            calibration_datasets=manifest["training_datasets"],
            same_dataset_eval_datasets=manifest["heldout_datasets"],
        )
    else:
        figures = _write_four_row_raw_sum_figure(output_root, rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            key: manifest[key]
            for key in (
                "source_output_root", "backbones", "training_datasets",
                "heldout_datasets", "min_drop", "max_drop", "drop_step",
                "variants", "strategies", "token_region_candidates",
            )
        },
        "provenance": {
            "manifest": str(output_root / MANIFEST_FILENAME),
            "eval_events": manifest.get("eval_events", []),
        },
        "missing": [],
        "rows": rows,
        "aggregates": aggregates,
        "figures": figures,
    }
    _atomic_write_json(output_root / RESULT_JSON_FILENAME, payload)
    _write_csv(output_root / RESULT_CSV_FILENAME, rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-manifest")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--source-output-root", type=Path, required=True)
    init.add_argument("--backbones", required=True)
    init.add_argument("--training-datasets", required=True)
    init.add_argument("--heldout-datasets", required=True)
    init.add_argument("--min-drop", type=int, required=True)
    init.add_argument("--max-drop", type=int, required=True)
    init.add_argument("--drop-step", type=int, required=True)
    init.add_argument("--variants", default=",".join(VARIANT_ORDER))
    init.add_argument("--strategies", default=",".join(STRATEGIES))
    event = subparsers.add_parser("record-eval")
    event.add_argument("--output-root", type=Path, required=True)
    event.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    event.add_argument("--backbone", required=True)
    event.add_argument("--selector", required=True)
    event.add_argument("--datasets", required=True)
    event.add_argument("--phase", required=True)
    report = subparsers.add_parser("summarize")
    report.add_argument("--output-root", type=Path, required=True)
    report.add_argument(
        "--plot-mode",
        choices=("heldout-generalization", "main-results"),
        default="heldout-generalization",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-manifest":
        init_manifest(
            output_root=args.output_root,
            source_output_root=args.source_output_root,
            backbones=_split(args.backbones),
            training_datasets=_split(args.training_datasets),
            heldout_datasets=_split(args.heldout_datasets),
            min_drop=args.min_drop,
            max_drop=args.max_drop,
            drop_step=args.drop_step,
            variants=_split(args.variants),
            strategies=_split(args.strategies),
        )
    elif args.command == "record-eval":
        record_eval_event(
            output_root=args.output_root,
            variant=args.variant,
            backbone=args.backbone,
            selector=args.selector,
            datasets=_split(args.datasets),
            phase=args.phase,
        )
    else:
        payload = summarize(args.output_root, plot_mode=args.plot_mode)
        print(
            f"Wrote {len(payload['rows'])} rows, "
            f"{len(payload['aggregates'])} aggregates, and "
            f"{len(payload['figures'])} figures"
        )


if __name__ == "__main__":
    main()

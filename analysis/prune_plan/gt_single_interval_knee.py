from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.prune_plan import interval_selector, is_interval_selector
from analysis.prune_plan.build_prune_plan_common import parse_dataset_list
from analysis.prune_plan.build_prune_plan_v3 import (
    GT_AGGREGATE_METRICS,
    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP,
    _load_gt_aggregate_for_datasets,
    select_gt_single_interval_path,
)


DEFAULT_THRESHOLD = 0.1
SELECTION_SCHEMA = "gt_single_interval_knee_v1"
SELECTION_FILENAME = "remaining_skip_gt_knee.json"
PLOT_FILENAME = "remaining_skip_gt_knee.png"


def round_degradation_for_selection(value: float) -> float:
    return float(
        Decimal(str(float(value))).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float | None:
    aliases = ("chamfer_distance", "cd") if metric == "chamfer_distance" else (metric,)
    for key in aliases:
        value = metrics.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    return None


def _dataset_inputs(
    *,
    plan_root: Path,
    dataset: str,
    expected_unit_ids: Sequence[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    pruning_root = Path(plan_root) / str(dataset) / "eval" / "pruning"
    manifest_path = pruning_root / "skip_eval_manifest_v3.json"
    upper_path = pruning_root.parent / "upper" / "eval_record_v3.json"
    manifest = _load_json(manifest_path)
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{manifest_path}: config must be an object")
    unit_ids = [str(unit_id) for unit_id in config.get("unit_ids", [])]
    if not unit_ids:
        raise ValueError(f"{manifest_path}: unit_ids is empty")
    if expected_unit_ids is not None and unit_ids != [str(x) for x in expected_unit_ids]:
        raise ValueError(
            f"{manifest_path}: unit catalog differs across GT source datasets"
        )
    upper = _load_json(upper_path)
    upper_scenes = upper.get("scenes", {})
    if not isinstance(upper_scenes, dict) or not upper_scenes:
        raise ValueError(f"{upper_path}: upper scene results are empty")
    expected_num_scenes = len(upper_scenes)

    summaries: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(pruning_root.glob("*/summary.json")):
        raw = _load_json(summary_path)
        selector = str(raw.get("selector") or summary_path.parent.name).strip()
        if not is_interval_selector(selector):
            continue
        summaries[selector] = raw
    required_selectors = {
        interval_selector(start, end)
        for start in range(1, len(unit_ids))
        for end in range(start, len(unit_ids))
    }
    missing = sorted(required_selectors.difference(summaries))
    if missing:
        raise ValueError(
            f"{pruning_root}: missing {len(missing)} required single-interval "
            f"summaries, examples={missing[:8]}"
        )

    digest_rows: list[dict[str, Any]] = []
    for selector in sorted(required_selectors):
        summary = summaries[selector]
        num_scenes = int(summary.get("num_scenes", -1))
        if num_scenes != expected_num_scenes:
            raise ValueError(
                f"{pruning_root / selector}: num_scenes={num_scenes}, "
                f"expected upper coverage={expected_num_scenes}"
            )
        metrics = summary.get("metrics_mean", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"{pruning_root / selector}: metrics_mean is invalid")
        clean_metrics: dict[str, float] = {}
        for metric in GT_AGGREGATE_METRICS:
            value = _metric_value(metrics, metric)
            if value is None:
                raise ValueError(
                    f"{pruning_root / selector}: required GT metric {metric!r} is missing"
                )
            clean_metrics[str(metric)] = value
        digest_rows.append(
            {
                "selector": selector,
                "num_scenes": num_scenes,
                "metrics": clean_metrics,
            }
        )
    digest_raw = json.dumps(
        digest_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return unit_ids, {
        "dataset": str(dataset),
        "pruning_root": str(pruning_root),
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": str(manifest.get("fingerprint", "")),
        "upper_path": str(upper_path),
        "num_scenes": expected_num_scenes,
        "num_single_intervals": len(required_selectors),
        "summary_digest_sha256": hashlib.sha256(
            digest_raw.encode("utf-8")
        ).hexdigest(),
    }


def select_gt_single_interval_knee(
    *,
    plan_root: Path,
    datasets: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    drop_step: int = 1,
) -> dict[str, Any]:
    dataset_names = [str(name).strip() for name in datasets if str(name).strip()]
    if not dataset_names:
        raise ValueError("at least one GT source dataset is required")
    limit = float(threshold)
    if not math.isfinite(limit) or not 0.0 <= limit <= 1.0:
        raise ValueError(f"threshold must be finite and in [0,1], got {threshold}")
    if isinstance(drop_step, bool) or int(drop_step) <= 0:
        raise ValueError(f"drop_step must be a positive integer, got {drop_step}")
    step = int(drop_step)

    unit_ids: list[str] | None = None
    inputs: list[dict[str, Any]] = []
    for dataset in dataset_names:
        current_ids, input_record = _dataset_inputs(
            plan_root=Path(plan_root),
            dataset=dataset,
            expected_unit_ids=unit_ids,
        )
        if unit_ids is None:
            unit_ids = current_ids
        inputs.append(input_record)
    assert unit_ids is not None

    aggregate, _components, _picks, meta, _per_dataset, _per_dataset_aggregate = (
        _load_gt_aggregate_for_datasets(
            plan_root=Path(plan_root),
            datasets=dataset_names,
            unit_ids=unit_ids,
            metric_selector="all",
        )
    )
    path_drops, path_scores, path_extra = select_gt_single_interval_path(aggregate)
    curve: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for drop_count in sorted(path_drops):
        components = path_extra[int(drop_count)]["component_intervals"]
        if len(components) != 1:
            raise RuntimeError(
                "canonical GT single-interval path returned a non-single plan"
            )
        component = components[0]
        start = int(component["start"])
        end = int(component["end"])
        raw_score = float(path_scores[int(drop_count)])
        if not math.isfinite(raw_score):
            continue
        record = {
            "drop_count": int(drop_count),
            "selector": interval_selector(start, end),
            "start": start,
            "end": end,
            "mean_normalized_degradation": raw_score,
        }
        curve.append(record)
        comparison_score = round_degradation_for_selection(raw_score)
        if drop_count % step == 0 and comparison_score <= limit:
            eligible.append(record)
    if not eligible:
        raise ValueError(
            "No positive single-interval GT candidate satisfies the knee threshold "
            f"{limit:g} with drop_step={step}"
        )
    selected = max(
        eligible,
        key=lambda record: (
            int(record["drop_count"]),
            -float(record["mean_normalized_degradation"]),
            -int(record["start"]),
            -int(record["end"]),
        ),
    )
    return {
        "schema_version": SELECTION_SCHEMA,
        "method": "max_drop_at_or_below_gt_single_interval_threshold",
        "strategy": "gt_single_interval",
        "threshold": limit,
        "score_rounding": {
            "decimal_places": 2,
            "mode": "ROUND_HALF_UP",
            "applied_before_threshold_comparison": True,
        },
        "drop_step": step,
        "datasets": dataset_names,
        "metrics": [str(metric) for metric in GT_AGGREGATE_METRICS],
        "drop_count": int(selected["drop_count"]),
        "selector": str(selected["selector"]),
        "start": int(selected["start"]),
        "end": int(selected["end"]),
        "mean_normalized_degradation": float(
            selected["mean_normalized_degradation"]
        ),
        "num_units": len(unit_ids),
        "unit_ids": list(unit_ids),
        "single_interval_only": True,
        "path_selection": "canonical_pruning_plan_gt_single_interval_path",
        "cross_budget_interval_overlap_required": bool(
            GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
        ),
        "cross_budget_scan_direction": "large_to_small",
        "protected_unit_zero": True,
        "tie_policy": "lowest_start_then_lowest_end",
        "curve": curve,
        "inputs": inputs,
        "gt_aggregation": meta,
    }


def write_selection(path: Path, selection: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(selection), indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def write_knee_plot(path: Path, selection: Mapping[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = [
        record
        for record in selection.get("curve", [])
        if isinstance(record, Mapping)
    ]
    if not curve:
        raise ValueError("GT single-interval knee selection has no curve to plot")
    xs = [int(record["drop_count"]) for record in curve]
    ys = [float(record["mean_normalized_degradation"]) for record in curve]
    knee_x = int(selection["drop_count"])
    knee_y = float(selection["mean_normalized_degradation"])
    threshold = float(selection["threshold"])
    backbone = str(selection.get("backbone", "")).strip() or "backbone"
    datasets = ", ".join(str(name) for name in selection.get("datasets", []))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(
        xs,
        ys,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4.2,
        label="Best GT single interval per drop count",
    )
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold = {threshold:g}",
    )
    ax.axvline(
        knee_x,
        color="#16a34a",
        linestyle="--",
        linewidth=1.8,
        label=f"Knee = {knee_x} dropped units",
    )
    ax.scatter(
        [knee_x],
        [knee_y],
        s=90,
        color="#16a34a",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    ax.annotate(
        f"  knee: {knee_x} units\n  interval: {selection['selector']}\n"
        f"  mean: {knee_y:.4f}",
        xy=(knee_x, knee_y),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9.5,
        color="#14532d",
    )
    x_ticks = sorted(
        {
            int(round(value))
            for value in ax.get_xticks()
            if math.isfinite(float(value)) and value >= 0
        }
        | {knee_x}
    )
    y_ticks = sorted(
        {
            float(value)
            for value in ax.get_yticks()
            if math.isfinite(float(value)) and value >= 0
        }
        | {threshold}
    )
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Dropped units")
    ax.set_ylabel("Mean normalized degradation across 7 GT metrics")
    ax.set_title(
        f"{backbone}: GT single-interval knee\nsource datasets: {datasets}",
        fontweight="bold",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select one GT single-interval knee from first-pass eval/pruning results."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--drop-step", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    selection = select_gt_single_interval_knee(
        plan_root=Path(args.root),
        datasets=parse_dataset_list(str(args.datasets)),
        threshold=float(args.threshold),
        drop_step=int(args.drop_step),
    )
    if not args.no_write:
        plot_path = write_knee_plot(
            (
                Path(args.plot_output)
                if args.plot_output is not None
                else Path(args.root) / PLOT_FILENAME
            ),
            {
                **selection,
                "backbone": Path(args.root).name,
            },
        )
        selection["backbone"] = Path(args.root).name
        selection["plot_path"] = str(plot_path)
        write_selection(
            Path(args.output) if args.output is not None else Path(args.root) / SELECTION_FILENAME,
            selection,
        )
    print(f"{selection['selector']} {selection['drop_count']}")


if __name__ == "__main__":
    main()

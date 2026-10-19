"""Finalize pointwise token-region pruning candidates into one canonical bundle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    bundle_plan_records,
    bundle_strategy_records,
    physical_pruning_selector_from_record,
)
from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    _fit_pruning_normalization_profile,
    load_interval_summary_records,
    load_upper_average_for_result_root,
    metric_value,
)
from analysis.redundancy.skip_metric_scaling import (
    normalize_raw_metric_against_upper_worst,
)
from analysis.prune_plan.visualize_triangle_prune import (
    render_prune_strategy_drop_grids,
)


POINTWISE_TOKEN_REGION_STRATEGIES = (
    "CKA_single",
    "CKA_double",
    "cosine_single",
    "cosine_double",
    "inv_mse_single",
    "inv_mse_double",
    "cca_single",
    "cca_double",
)
TOKEN_REGION_ALL = "all"
TOKEN_REGION_SPECIAL_PATCH_MEAN = "special_patch_mean"
TOKEN_REGION_TIE_PRIORITY = (
    TOKEN_REGION_SPECIAL_PATCH_MEAN,
    TOKEN_REGION_ALL,
)


def _split_list(value: str | Sequence[str]) -> list[str]:
    raw = value if isinstance(value, str) else ",".join(str(item) for item in value)
    return list(
        dict.fromkeys(
            item.strip()
            for item in str(raw).replace(",", " ").split()
            if item.strip()
        )
    )


def _load_bundle_file(path: Path) -> dict[str, Any]:
    bundle_path = Path(path)
    try:
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing token-region candidate bundle: {bundle_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Candidate bundle must contain an object: {bundle_path}")
    if str(raw.get("schema_version", "")) != PRUNE_PLAN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid candidate bundle schema at {bundle_path}: "
            f"{raw.get('schema_version')!r}"
        )
    if not isinstance(raw.get("strategies"), dict):
        raise ValueError(f"Candidate bundle has no strategies object: {bundle_path}")
    return raw


def _validate_compatible_bundles(
    all_bundle: Mapping[str, Any],
    special_patch_mean_bundle: Mapping[str, Any],
) -> None:
    for field in ("backbone", "dataset", "num_units", "unit_ids"):
        if all_bundle.get(field) != special_patch_mean_bundle.get(field):
            raise ValueError(
                f"Token-region candidate bundles disagree on {field}: "
                f"{all_bundle.get(field)!r} != {special_patch_mean_bundle.get(field)!r}"
            )
    special_strategies = set(bundle_strategy_records(dict(special_patch_mean_bundle)))
    unexpected = sorted(
        special_strategies - set(POINTWISE_TOKEN_REGION_STRATEGIES)
    )
    if unexpected:
        raise ValueError(
            "special_patch_mean candidate contains strategies that must remain all-only: "
            f"{unexpected}"
        )


def _maximum_drop_budget(bundle: Mapping[str, Any]) -> Optional[int]:
    extra = bundle.get("extra", {})
    declared = extra.get("max_drop_budget") if isinstance(extra, dict) else None
    if isinstance(declared, int) and declared > 0:
        return int(declared)
    drop_counts = [
        int(drop_key)
        for strategy in bundle_strategy_records(dict(bundle)).values()
        for drop_key in (
            strategy.get("points", {})
            if isinstance(strategy.get("points", {}), dict)
            else {}
        )
        if str(drop_key).isdigit()
    ]
    return max(drop_counts) if drop_counts else None


def _selection_profiles(
    *,
    experiment_root: Path,
    datasets: Sequence[str],
    max_drop_count: Optional[int],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        result_root = Path(experiment_root) / str(dataset) / "eval" / "pruning"
        summaries = load_interval_summary_records(result_root)
        if not summaries:
            raise FileNotFoundError(
                f"No pruning summaries available for selection dataset {dataset!r}: "
                f"{result_root}"
            )
        upper_average = load_upper_average_for_result_root(result_root)
        profile = _fit_pruning_normalization_profile(
            summaries=summaries,
            upper_average=upper_average,
            result_root=result_root,
            max_drop_count=max_drop_count,
        )
        maxima = profile.get(
            "maximum_brute_force_metric",
            profile.get("maximum_single_interval_metric", {}),
        )
        missing = [
            metric
            for metric in GT_MEAN_METRICS
            if metric_value(upper_average, metric) is None
            or metric_value(maxima, metric) is None
        ]
        if missing:
            raise ValueError(
                f"Incomplete normalization profile for {dataset!r}; "
                f"missing metrics: {missing}"
            )
        profiles[str(dataset)] = {
            "result_root": str(result_root),
            "summaries": summaries,
            "upper_average": upper_average,
            "normalization_profile": profile,
        }
    return profiles


def _candidate_selection_score(
    *,
    selector: str,
    profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    dataset_scores: dict[str, float] = {}
    for dataset, entry in profiles.items():
        summaries = entry.get("summaries", {})
        summary = summaries.get(selector) if isinstance(summaries, dict) else None
        if not isinstance(summary, dict):
            raise FileNotFoundError(
                f"Missing evaluated pruning selector {selector!r} for "
                f"selection dataset {dataset!r}"
            )
        metrics = summary.get("metrics_mean", {})
        upper = entry.get("upper_average", {})
        profile = entry.get("normalization_profile", {})
        maxima = (
            profile.get(
                "maximum_brute_force_metric",
                profile.get("maximum_single_interval_metric", {}),
            )
            if isinstance(profile, dict)
            else {}
        )
        normalized: list[float] = []
        for metric in GT_MEAN_METRICS:
            current = metric_value(metrics, metric) if isinstance(metrics, dict) else None
            baseline = metric_value(upper, metric) if isinstance(upper, dict) else None
            maximum = metric_value(maxima, metric) if isinstance(maxima, dict) else None
            if current is None or baseline is None or maximum is None:
                raise ValueError(
                    f"Incomplete {metric!r} metric for selector {selector!r} "
                    f"on selection dataset {dataset!r}"
                )
            value = float(normalize_raw_metric_against_upper_worst(
                np.asarray([float(current)], dtype=np.float64),
                upper_baseline=float(baseline),
                worst_value=float(maximum),
                clamp_better_than_upper=False,
            )[0])
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite normalized {metric!r} for selector {selector!r} "
                    f"on selection dataset {dataset!r}"
                )
            normalized.append(float(value))
        dataset_scores[str(dataset)] = float(np.mean(normalized))
    return float(np.mean(list(dataset_scores.values()))), dataset_scores


def _strategy_points(strategy: Mapping[str, Any], name: str) -> dict[str, str]:
    points = strategy.get("points", {})
    if not isinstance(points, dict):
        raise ValueError(f"Strategy {name!r} has an invalid points object")
    return {str(key): str(value) for key, value in points.items() if str(key).isdigit()}


def _budget_is_selected(
    drop_count: int,
    *,
    min_drop: Optional[int],
    max_drop: Optional[int],
    drop_step: Optional[int],
) -> bool:
    count = int(drop_count)
    if min_drop is not None and count < int(min_drop):
        return False
    if max_drop is not None and count > int(max_drop):
        return False
    if drop_step is not None:
        step = int(drop_step)
        if step <= 0:
            raise ValueError("drop_step must be positive")
        if (count - int(min_drop or 0)) % step != 0:
            return False
    return True


def _copy_selected_record(
    *,
    final_bundle: dict[str, Any],
    final_plan_key: str,
    source_record: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    record = copy.deepcopy(dict(source_record))
    record["selector"] = str(final_plan_key)
    record["token_region_selection"] = copy.deepcopy(dict(selection))
    intervals = final_bundle.setdefault("intervals", {})
    sets = final_bundle.setdefault("sets", {})
    if not isinstance(intervals, dict) or not isinstance(sets, dict):
        raise ValueError("Final bundle intervals/sets sections must be objects")
    intervals.pop(str(final_plan_key), None)
    sets[str(final_plan_key)] = record


def _rebuild_cka_spans(bundle: dict[str, Any]) -> None:
    strategies = bundle_strategy_records(bundle)
    plan_records = bundle_plan_records(bundle)
    cka_double = strategies.get("CKA_double", {})
    points = _strategy_points(cka_double, "CKA_double") if cka_double else {}
    spans: dict[str, dict[str, Any]] = {}
    for drop_key, plan_key in points.items():
        record = plan_records.get(plan_key)
        if isinstance(record, dict):
            spans[str(int(drop_key))] = copy.deepcopy(record)
    bundle["spans"] = {
        key: spans[key] for key in sorted(spans, key=int)
    }


def _drop_grid_items(bundle: Mapping[str, Any]) -> list[tuple[str, dict[int, list[int]]]]:
    plan_records = bundle_plan_records(dict(bundle))
    items: list[tuple[str, dict[int, list[int]]]] = []
    for strategy_key, strategy in bundle_strategy_records(dict(bundle)).items():
        drops: dict[int, list[int]] = {}
        for drop_key, plan_key in _strategy_points(strategy, strategy_key).items():
            record = plan_records.get(plan_key)
            if isinstance(record, dict):
                drops[int(drop_key)] = [int(value) for value in record.get("drop", [])]
        if drops:
            items.append((str(strategy.get("display_name", strategy_key)), drops))
    return items


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def finalize_token_region_plan(
    *,
    experiment_root: Path,
    all_bundle_path: Path,
    selection_datasets: Sequence[str],
    output_path: Path,
    special_patch_mean_bundle_path: Optional[Path] = None,
    viz_out_dir: Optional[Path] = None,
    min_drop: Optional[int] = None,
    max_drop: Optional[int] = None,
    drop_step: Optional[int] = None,
) -> Path:
    """Select the lower training degradation for the four dual-region families."""
    datasets = _split_list(selection_datasets)
    if not datasets:
        raise ValueError("At least one selection dataset is required")
    if drop_step is not None and int(drop_step) <= 0:
        raise ValueError("drop_step must be positive")
    if min_drop is not None and max_drop is not None and int(min_drop) > int(max_drop):
        raise ValueError("min_drop must not exceed max_drop")
    all_bundle = _load_bundle_file(all_bundle_path)
    special_bundle = (
        _load_bundle_file(special_patch_mean_bundle_path)
        if special_patch_mean_bundle_path is not None
        else None
    )
    if special_bundle is not None:
        _validate_compatible_bundles(all_bundle, special_bundle)

    final_bundle = copy.deepcopy(all_bundle)
    all_strategies = bundle_strategy_records(all_bundle)
    special_strategies = (
        bundle_strategy_records(special_bundle) if special_bundle is not None else {}
    )
    all_records = bundle_plan_records(all_bundle)
    special_records = bundle_plan_records(special_bundle) if special_bundle is not None else {}
    profiles = (
        _selection_profiles(
            experiment_root=Path(experiment_root),
            datasets=datasets,
            max_drop_count=_maximum_drop_budget(all_bundle),
        )
        if special_bundle is not None and special_strategies
        else {}
    )

    winner_counts = {TOKEN_REGION_ALL: 0, TOKEN_REGION_SPECIAL_PATCH_MEAN: 0}
    final_strategies = final_bundle.get("strategies", {})
    if not isinstance(final_strategies, dict):
        raise ValueError("Final bundle strategies section must be an object")
    for strategy_key in POINTWISE_TOKEN_REGION_STRATEGIES:
        all_strategy = all_strategies.get(strategy_key)
        special_strategy = special_strategies.get(strategy_key)
        if all_strategy is None and special_strategy is None:
            continue
        if not isinstance(all_strategy, dict) or not isinstance(special_strategy, dict):
            raise ValueError(
                f"Dual-region strategy {strategy_key!r} must be present in both candidates"
            )
        all_points = _strategy_points(all_strategy, strategy_key)
        special_points = _strategy_points(special_strategy, strategy_key)
        if set(all_points) != set(special_points):
            raise ValueError(
                f"Dual-region strategy {strategy_key!r} has different drop budgets: "
                f"all={sorted(all_points, key=int)}, "
                f"special_patch_mean={sorted(special_points, key=int)}"
            )
        final_strategy = copy.deepcopy(all_strategy)
        selected_drop_keys = [
            drop_key
            for drop_key in sorted(all_points, key=int)
            if _budget_is_selected(
                int(drop_key),
                min_drop=min_drop,
                max_drop=max_drop,
                drop_step=drop_step,
            )
        ]
        final_intervals = final_bundle.get("intervals", {})
        final_sets = final_bundle.get("sets", {})
        if not isinstance(final_intervals, dict) or not isinstance(final_sets, dict):
            raise ValueError("Final bundle intervals/sets sections must be objects")
        for plan_key in all_points.values():
            final_intervals.pop(plan_key, None)
            final_sets.pop(plan_key, None)
        final_strategy["points"] = {}
        final_scores: dict[str, float] = {}
        candidate_sources = {
            TOKEN_REGION_ALL: copy.deepcopy(all_strategy.get("source")),
            TOKEN_REGION_SPECIAL_PATCH_MEAN: copy.deepcopy(
                special_strategy.get("source")
            ),
        }
        for drop_key in selected_drop_keys:
            all_plan_key = all_points[drop_key]
            special_plan_key = special_points[drop_key]
            all_record = all_records.get(all_plan_key)
            special_record = special_records.get(special_plan_key)
            if not isinstance(all_record, dict) or not isinstance(special_record, dict):
                raise FileNotFoundError(
                    f"Missing candidate plan record for {strategy_key}@{drop_key}"
                )
            selectors = {
                TOKEN_REGION_ALL: physical_pruning_selector_from_record(all_record),
                TOKEN_REGION_SPECIAL_PATCH_MEAN: (
                    physical_pruning_selector_from_record(special_record)
                ),
            }
            scores: dict[str, float] = {}
            dataset_scores: dict[str, dict[str, float]] = {}
            for token_region, selector in selectors.items():
                score, per_dataset = _candidate_selection_score(
                    selector=selector,
                    profiles=profiles,
                )
                scores[token_region] = score
                dataset_scores[token_region] = per_dataset
            winner_region = min(
                TOKEN_REGION_TIE_PRIORITY,
                key=lambda region: (
                    float(scores[region]),
                    TOKEN_REGION_TIE_PRIORITY.index(region),
                ),
            )
            winner_counts[winner_region] += 1
            winner_strategy = (
                all_strategy
                if winner_region == TOKEN_REGION_ALL
                else special_strategy
            )
            winner_record = (
                all_record
                if winner_region == TOKEN_REGION_ALL
                else special_record
            )
            winner_proxy_scores = winner_strategy.get("scores", {})
            if isinstance(winner_proxy_scores, dict):
                proxy_score = winner_proxy_scores.get(drop_key)
                if isinstance(proxy_score, (int, float)) and math.isfinite(
                    float(proxy_score)
                ):
                    final_scores[drop_key] = float(proxy_score)
            selection = {
                "method": "minimum_training_normalized_gt_degradation",
                "winner_token_region": winner_region,
                "winner_selector": selectors[winner_region],
                "candidate_selectors": selectors,
                "candidate_scores": scores,
                "candidate_dataset_scores": dataset_scores,
                "selection_datasets": list(datasets),
                "metrics": list(GT_MEAN_METRICS),
                "tie_priority": list(TOKEN_REGION_TIE_PRIORITY),
            }
            _copy_selected_record(
                final_bundle=final_bundle,
                final_plan_key=all_plan_key,
                source_record=winner_record,
                selection=selection,
            )
            final_strategy.setdefault("points", {})[drop_key] = all_plan_key
        final_strategy["scores"] = final_scores
        final_strategy["source"] = {
            "kind": "pointwise_token_region_selection",
            "candidate_token_regions": [
                TOKEN_REGION_ALL,
                TOKEN_REGION_SPECIAL_PATCH_MEAN,
            ],
            "selection": "minimum_training_normalized_gt_degradation",
            "selection_datasets": list(datasets),
            "metrics": list(GT_MEAN_METRICS),
            "tie_priority": list(TOKEN_REGION_TIE_PRIORITY),
            "selected_budget_filter": {
                "min_drop": min_drop,
                "max_drop": max_drop,
                "drop_step": drop_step,
            },
            "candidate_paths_individually_preserve_cross_budget_constraints": True,
            "final_cross_budget_path_constraint_guaranteed": False,
            "candidate_sources": candidate_sources,
        }
        final_strategies[strategy_key] = final_strategy

    _rebuild_cka_spans(final_bundle)
    extra = final_bundle.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("Final bundle extra section must be an object")
    extra.update(
        {
            "token_region": "strategy_specific",
            "similarity_source": {
                "kind": "strategy_specific_token_region_candidates",
                TOKEN_REGION_ALL: copy.deepcopy(
                    all_bundle.get("extra", {}).get("similarity_source")
                    if isinstance(all_bundle.get("extra"), dict)
                    else None
                ),
                **(
                    {
                        TOKEN_REGION_SPECIAL_PATCH_MEAN: copy.deepcopy(
                            special_bundle.get("extra", {}).get(
                                "similarity_source"
                            )
                            if isinstance(special_bundle.get("extra"), dict)
                            else None
                        )
                    }
                    if special_bundle is not None
                    else {}
                ),
            },
            "token_region_policy": {
                "fixed_all_strategies": sorted(
                    set(all_strategies) - set(POINTWISE_TOKEN_REGION_STRATEGIES)
                ),
                "pointwise_selected_strategies": sorted(
                    set(all_strategies).intersection(
                        POINTWISE_TOKEN_REGION_STRATEGIES
                    )
                ),
                "candidate_token_regions": [
                    TOKEN_REGION_ALL,
                    TOKEN_REGION_SPECIAL_PATCH_MEAN,
                ]
                if special_bundle is not None
                else [TOKEN_REGION_ALL],
                "selection_datasets": list(datasets),
                "selection_metric": "seven_metric_normalized_gt_degradation_mean",
                "optimization_direction": "minimize",
                "tie_priority": list(TOKEN_REGION_TIE_PRIORITY),
                "winner_counts": winner_counts,
                "selected_budget_filter": {
                    "min_drop": min_drop,
                    "max_drop": max_drop,
                    "drop_step": drop_step,
                },
            },
            "token_region_candidate_bundles": {
                TOKEN_REGION_ALL: str(Path(all_bundle_path)),
                **(
                    {
                        TOKEN_REGION_SPECIAL_PATCH_MEAN: str(
                            Path(special_patch_mean_bundle_path)
                        )
                    }
                    if special_patch_mean_bundle_path is not None
                    else {}
                ),
            },
        }
    )
    spans = final_bundle.get("spans", {})
    span_counts = [
        int(key)
        for key in spans
        if isinstance(spans, dict) and str(key).isdigit()
    ]
    extra["max_available_span"] = max(span_counts, default=0)
    extra["available_spans"] = len(span_counts)
    cross_budget = extra.get("cross_budget_path_constraint", {})
    if isinstance(cross_budget, dict):
        cross_budget["pointwise_token_region_selected_families"] = False
        cross_budget["pointwise_selection_may_switch_region_between_budgets"] = True
    _atomic_write_json(Path(output_path), final_bundle)

    if viz_out_dir is not None:
        items = _drop_grid_items(final_bundle)
        if items:
            backbone = str(final_bundle.get("backbone", "backbone"))
            safe_backbone = backbone.replace("/", "_").replace(" ", "_")
            render_prune_strategy_drop_grids(
                num_units=int(final_bundle.get("num_units", 0)),
                strategy_drop_indices=items,
                out_dir=Path(viz_out_dir),
                backbone=backbone,
                tag=f"prune_plan_{safe_backbone}_drop_grid",
                title=f"{backbone}: finalized pointwise token-region pruning plans",
                strategy_evidence=None,
            )
    print(
        f"Wrote finalized pruning bundle {output_path} "
        f"(all_wins={winner_counts[TOKEN_REGION_ALL]}, "
        f"special_patch_mean_wins={winner_counts[TOKEN_REGION_SPECIAL_PATCH_MEAN]})",
        flush=True,
    )
    return Path(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select all vs special_patch_mean per strategy/drop budget and write "
            "the canonical pruning bundle."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--all-bundle", type=Path, required=True)
    parser.add_argument("--special-patch-mean-bundle", type=Path)
    parser.add_argument("--selection-datasets", required=True)
    parser.add_argument("--min-drop", type=int)
    parser.add_argument("--max-drop", type=int)
    parser.add_argument("--drop-step", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(PRUNE_PLAN_BUNDLE_FILENAME),
    )
    parser.add_argument("--viz-out-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    finalize_token_region_plan(
        experiment_root=args.experiment_root,
        all_bundle_path=args.all_bundle,
        special_patch_mean_bundle_path=args.special_patch_mean_bundle,
        selection_datasets=_split_list(args.selection_datasets),
        output_path=args.output,
        viz_out_dir=args.viz_out_dir,
        min_drop=args.min_drop,
        max_drop=args.max_drop,
        drop_step=args.drop_step,
    )


if __name__ == "__main__":
    main()

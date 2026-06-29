"""Build percentage-budget prune-plan groups from dense CKA triangles."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, Optional, Union

from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    BUDGET_PLAN_BUNDLE_FILENAME,
    BUDGET_PLAN_BUNDLE_SCHEMA_VERSION,
)
from analysis.prune_plan.training_cache_units import required_cache_unit_ids
from analysis.prune_plan.build_prune_plan_v3_backup import (
    _build_plan_from_prefix,
    _default_plan_root,
    _unit_id_to_index,
    load_similarity,
)
from analysis.redundancy.similarity_v3 import (
    DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
    REDUNDANCY_GRAPH_REGIONS,
    parse_exclude_token_regions,
)
from analysis.prune_plan.triangle_prune import (
    CORNER_RANKING_TOLERANCE,
    MIN_CORNER_LEGALITY_TOLERANCE,
    TrianglePick,
    scan_min_corner_triangles_with_candidates,
)
from analysis.prune_plan.visualize_triangle_prune import (
    render_all_span_corner_heatmap,
    render_budget_similarity_curve,
    render_span_unit_action_grid,
)
from analysis.schema.v3 import stage_dir


BudgetInput = Union[str, int, Sequence[Union[str, int]]]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build percentage-budget prune-plan bundle as plan_budget_all.json."
    )
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, or mean/min to aggregate available datasets.",
    )
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--metric", type=str, default="cka")
    p.add_argument(
        "--token-region",
        type=str,
        choices=(*REDUNDANCY_GRAPH_REGIONS, "mean", "min"),
        default="all",
    )
    p.add_argument("--exclude-token-regions", type=str, default="")
    p.add_argument(
        "--pruning-budget",
        type=str,
        nargs="+",
        required=True,
        help="Integer percentages in [0,100], separated by commas and/or spaces.",
    )
    p.add_argument(
        "--max-lora-anchors",
        type=int,
        default=1,
        help="Emit l=1..L for every budget, using the legacy LoRA prefix semantics.",
    )
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--prune-plan-root", type=Path, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Explicit bundle output path (default: prune-plan-root/{BUDGET_PLAN_BUNDLE_FILENAME}).",
    )
    p.add_argument(
        "--viz-out-dir",
        type=Path,
        default=None,
        help="Curve output directory (default: prune-plan-root).",
    )
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--cache-device", type=str, default="cuda")
    return p


def parse_pruning_budgets(value: BudgetInput) -> List[int]:
    """Parse comma/whitespace-separated budgets, deduplicating in input order."""
    if isinstance(value, str):
        raw_values = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    elif isinstance(value, int):
        raw_values = [str(value)]
    else:
        raw_values = [
            part
            for item in value
            for part in re.split(r"[\s,]+", str(item).strip())
            if part
        ]
    if not raw_values:
        raise ValueError("pruning_budget must contain at least one integer percentage")

    budgets: List[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        if not re.fullmatch(r"[+-]?\d+", raw):
            raise ValueError(f"pruning_budget entries must be integers, got {raw!r}")
        budget = int(raw)
        if budget < 0 or budget > 100:
            raise ValueError(f"pruning_budget entries must be in [0, 100], got {budget}")
        if budget not in seen:
            seen.add(budget)
            budgets.append(budget)
    return budgets


def rounded_budget_units(num_units: int, pruning_budget: int) -> int:
    """Convert an integer percentage to a unit count using round-half-up."""
    n = int(num_units)
    budget = int(pruning_budget)
    if n <= 0:
        raise ValueError(f"num_units must be positive, got {n}")
    if budget < 0 or budget > 100:
        raise ValueError(f"pruning_budget must be an integer in [0, 100], got {budget}")
    return (n * budget + 50) // 100


def _plan_group(pruning_budget: int) -> str:
    return f"plan_budget{int(pruning_budget)}"


def _plan_name(pruning_budget: int, lora_prefix: int) -> str:
    return f"{_plan_group(pruning_budget)}_l{int(lora_prefix)}"


def _pick_for_drop_count(
    scan_by_span: Mapping[int, TrianglePick],
    drop_count: int,
) -> Optional[TrianglePick]:
    if int(drop_count) == 0:
        return None
    if int(drop_count) < 0:
        raise ValueError(f"drop_count must be non-negative, got {int(drop_count)}")
    pick = scan_by_span.get(int(drop_count))
    if pick is None:
        raise ValueError(
            f"No min-corner legal triangle is available for drop_count={int(drop_count)}"
        )
    return pick


def run_build_prune_plan(
    *,
    experiment: str,
    dataset: str,
    pruning_budgets: BudgetInput,
    backbone: str = "vggt",
    metric: str = "cka",
    token_region: str = "all",
    exclude_token_regions: Sequence[str] = DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
    max_lora_anchors: int = 1,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    prune_plan_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    viz_out_dir: Optional[Path] = None,
    hf_model: str = "",
    cache_device: str = "cuda",
) -> List[Path]:
    """Build every budget/LoRA-prefix combination and return written plan paths."""
    budgets = parse_pruning_budgets(pruning_budgets)
    max_lora = int(max_lora_anchors)
    if max_lora < 1:
        raise ValueError(f"max_lora_anchors must be >= 1, got {max_lora}")

    unit_ids, sim, similarity_source = load_similarity(
        experiment=experiment,
        dataset=dataset,
        token_region=token_region,
        metric=metric,
        exclude_token_regions=exclude_token_regions,
    )
    num_units = int(sim.shape[0])
    budget_drop_counts = {
        budget: rounded_budget_units(num_units, budget) for budget in budgets
    }
    scan, eligible_candidates_by_span, rejected_by_legality_by_span = (
        scan_min_corner_triangles_with_candidates(
            sim,
            unit_ids,
            legality_tolerance=MIN_CORNER_LEGALITY_TOLERANCE,
            ranking_tolerance=CORNER_RANKING_TOLERANCE,
        )
    )
    scan_by_span = {int(pick.span): pick for pick in scan}
    max_available_span = max(scan_by_span, default=0)

    skipped = [
        (budget, drop_count)
        for budget, drop_count in budget_drop_counts.items()
        if drop_count >= num_units or (drop_count > 0 and drop_count not in scan_by_span)
    ]
    skipped_budgets = {budget for budget, _drop_count in skipped}
    if skipped:
        details = ", ".join(f"{budget}%->{count} units" for budget, count in skipped)
        print(
            f"Skipping budgets without min-corner legal triangles "
            f"(available spans={len(scan)}/{num_units - 1}, "
            f"max available span={max_available_span}, "
            f"legality_tolerance={MIN_CORNER_LEGALITY_TOLERANCE:g}, "
            f"ranking_tolerance={CORNER_RANKING_TOLERANCE:g}): {details}",
            flush=True,
        )
    available_budgets = [budget for budget in budgets if budget not in skipped_budgets]
    available_drop_counts = {
        budget: budget_drop_counts[budget] for budget in available_budgets
    }

    plan_root = prune_plan_root or _default_plan_root(backbone)
    plan_root.mkdir(parents=True, exist_ok=True)
    bundle_path = output_path or (plan_root / BUDGET_PLAN_BUNDLE_FILENAME)
    visualization_root = (
        viz_out_dir or (output_path.parent if output_path is not None else plan_root)
    )
    corner_heatmap_path = render_all_span_corner_heatmap(
        sim,
        unit_ids,
        scan,
        span_to_candidate_picks=eligible_candidates_by_span,
        span_to_rejected_picks=rejected_by_legality_by_span,
        out_dir=visualization_root,
        backbone=backbone,
    )
    print(f"Wrote all-span corner heatmap {corner_heatmap_path}", flush=True)

    source_cache = (
        ""
        if dataset.strip().lower() in {"mean", "min"}
        else str(stage_dir(experiment, dataset, "cache"))
    )
    span_records: dict[str, dict] = {}

    if scan:
        adapter = get_backbone(backbone)
        hf = str(hf_model or adapter.default_hf_model())
        model = adapter.load_pretrained(
            hf_model=hf, device=str(cache_device), dtype=None, eval_mode=True
        )
        for lora_prefix in range(1, max_lora + 1):
            span_to_drop_indices: dict[int, list[int]] = {}
            span_to_lora_indices: dict[int, list[int]] = {}
            span_to_cache_indices: dict[int, list[int]] = {}
            for scan_pick in scan:
                span = int(scan_pick.span)
                scan_plan, scan_lora_indices = _build_plan_from_prefix(
                    picks_prefix=[scan_pick],
                    lora_prefix=lora_prefix,
                    lora_rank=int(lora_rank),
                    lora_alpha=float(lora_alpha),
                    backbone=backbone,
                    dataset=dataset,
                    num_units=num_units,
                    source_cache=source_cache,
                )
                scan_cache_unit_ids = sorted(
                    required_cache_unit_ids(scan_plan, model) if scan_plan.lora else set(),
                    key=_unit_id_to_index,
                )
                scan_cache_unit_indices = [
                    _unit_id_to_index(uid) for uid in scan_cache_unit_ids
                ]
                span_to_drop_indices[span] = [int(idx) for idx in scan_plan.drop]
                span_to_lora_indices[span] = [int(idx) for idx in scan_lora_indices]
                span_to_cache_indices[span] = scan_cache_unit_indices

                record = span_records.setdefault(
                    str(span),
                    {
                        "span": span,
                        "i": int(scan_pick.i),
                        "j": int(scan_pick.j),
                        "unit_i": str(scan_pick.unit_i),
                        "unit_j": str(scan_pick.unit_j),
                        "drop": [int(idx) for idx in scan_plan.drop],
                        "dropped_unit_ids": [
                            str(unit_ids[int(idx)])
                            for idx in scan_plan.drop
                            if 0 <= int(idx) < len(unit_ids)
                        ],
                        "corner_similarity": float(sim[int(scan_pick.i), int(scan_pick.j)]),
                        "lora": {},
                    },
                )
                record["lora"][str(lora_prefix)] = {
                    "lora": {str(int(k)): asdict(v) for k, v in scan_plan.lora.items()},
                    "lora_indices": [int(idx) for idx in scan_lora_indices],
                    "cache_unit_ids": [str(uid) for uid in scan_cache_unit_ids],
                    "cache_unit_indices": scan_cache_unit_indices,
                }
            action_grid_path = render_span_unit_action_grid(
                num_units=num_units,
                span_to_drop_indices=span_to_drop_indices,
                span_to_lora_indices=span_to_lora_indices,
                span_to_cache_indices=span_to_cache_indices,
                highlighted_rows={
                    int(drop_count): [
                        f"{int(budget)}%"
                        for budget, count in available_drop_counts.items()
                        if int(count) == int(drop_count)
                    ]
                    for drop_count in sorted(set(available_drop_counts.values()))
                },
                out_dir=visualization_root,
                backbone=backbone,
                lora_prefix=lora_prefix,
            )
            print(f"Wrote span-unit action grid {action_grid_path}", flush=True)

    budget_records: dict[str, dict] = {}
    for budget in budgets:
        drop_count = int(budget_drop_counts[budget])
        available = budget in available_budgets
        skip_reason = ""
        if not available:
            if drop_count >= num_units:
                skip_reason = "drop_count must be smaller than num_units"
            elif drop_count > 0 and drop_count not in scan_by_span:
                skip_reason = "no min-corner legal triangle for requested drop_count"
            else:
                skip_reason = "budget is unavailable"
        budget_records[str(int(budget))] = {
            "budget_percent": int(budget),
            "drop_count": drop_count,
            "available": bool(available),
            "skip_reason": skip_reason,
            "selector_group": _plan_group(budget),
            "selector_names": (
                [_plan_name(budget, lora_prefix) for lora_prefix in range(1, max_lora + 1)]
                if available
                else []
            ),
        }

    bundle = {
        "schema_version": BUDGET_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": str(backbone),
        "dataset": str(dataset),
        "num_units": int(num_units),
        "source_cache": source_cache,
        "budgets": budget_records,
        "spans": {
            str(span): span_records[str(span)]
            for span in sorted((int(k) for k in span_records.keys()))
        },
        "lora_prefixes": [int(x) for x in range(1, max_lora + 1)],
        "extra": {
            "similarity_source": similarity_source,
            "metric": str(metric),
            "token_region": str(token_region),
            "exclude_token_regions": [str(x) for x in exclude_token_regions],
            "triangle_legality": "corner_is_minimum_with_tolerance",
            "min_corner_legality_tolerance": float(MIN_CORNER_LEGALITY_TOLERANCE),
            "corner_ranking_tolerance": float(CORNER_RANKING_TOLERANCE),
            "dense_triangle_selection": "min_corner_endpoint_linked_descending",
            "max_available_span": int(max_available_span),
            "available_spans": len(scan),
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
        },
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(
        f"Wrote {bundle_path} "
        f"(budgets={','.join(str(b) for b in budgets)}, lora_prefixes=1..{max_lora})",
        flush=True,
    )

    curve_path = render_budget_similarity_curve(
        scan,
        sim,
        available_drop_counts,
        out_dir=visualization_root,
        backbone=backbone,
        threshold=None,
    )
    print(
        f"Wrote budget curve {curve_path} "
        f"(available spans={len(scan)}, max available span={max_available_span})",
        flush=True,
    )
    return [bundle_path]


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_build_prune_plan(
        experiment=args.experiment,
        dataset=args.dataset,
        pruning_budgets=args.pruning_budget,
        backbone=args.backbone,
        metric=args.metric,
        token_region=args.token_region,
        exclude_token_regions=parse_exclude_token_regions(args.exclude_token_regions),
        max_lora_anchors=args.max_lora_anchors,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        prune_plan_root=args.prune_plan_root,
        output_path=args.output,
        viz_out_dir=args.viz_out_dir,
        hf_model=args.hf_model,
        cache_device=args.cache_device,
    )


if __name__ == "__main__":
    main()

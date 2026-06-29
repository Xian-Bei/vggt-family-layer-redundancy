"""Build percentage-budget prune-plan groups from dense CKA triangles."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Union

from analysis.backbones.registry import get_backbone
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
    TrianglePick,
    scan_nested_dense_triangles_from_max,
)
from analysis.prune_plan.visualize_triangle_prune import (
    render_budget_similarity_curve,
    render_triangle_prune_heatmap,
)
from analysis.schema.v3 import stage_dir


DENSE_TRIANGLE_THRESHOLD = 0.80
DENSE_TRIANGLE_THRESHOLD_FLOOR = 0.80
DENSE_TRIANGLE_THRESHOLD_STEP = 0.01
BudgetInput = Union[str, int, Sequence[Union[str, int]]]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build percentage-budget prune-plan groups as plan_budgetN_lL.json."
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
        help="Explicit output path; only valid for one budget with max-lora-anchors=1.",
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
    scan: Sequence[TrianglePick],
    drop_count: int,
) -> Optional[TrianglePick]:
    if int(drop_count) == 0:
        return None
    if int(drop_count) < 0 or int(drop_count) > len(scan):
        raise ValueError(f"No dense triangle is available for drop_count={int(drop_count)}")
    pick = scan[int(drop_count) - 1]
    if int(pick.span) != int(drop_count):
        raise AssertionError(
            f"Dense-triangle scan is not consecutive at drop_count={int(drop_count)}"
        )
    return pick


def _scan_with_adaptive_threshold(
    sim,
    unit_ids: Sequence[str],
    *,
    num_units: int,
    budget_drop_counts: Sequence[int],
) -> tuple[List[TrianglePick], float]:
    """Scan dense triangles, lowering threshold only for the requested max budget."""
    valid_requested_drop_counts = [
        int(count)
        for count in budget_drop_counts
        if 0 < int(count) < int(num_units)
    ]
    threshold = float(DENSE_TRIANGLE_THRESHOLD)
    floor = float(DENSE_TRIANGLE_THRESHOLD_FLOOR)
    step = float(DENSE_TRIANGLE_THRESHOLD_STEP)
    if floor > threshold:
        raise ValueError(
            f"DENSE_TRIANGLE_THRESHOLD_FLOOR must be <= DENSE_TRIANGLE_THRESHOLD, "
            f"got {floor:g} > {threshold:g}"
        )
    if step <= 0:
        raise ValueError(f"DENSE_TRIANGLE_THRESHOLD_STEP must be > 0, got {step}")

    target_drop_count = max(valid_requested_drop_counts, default=0)
    best_scan: List[TrianglePick] = []
    best_threshold = threshold
    while threshold >= floor:
        scan = scan_nested_dense_triangles_from_max(
            sim,
            unit_ids,
            threshold=threshold,
        )
        best_scan = scan
        best_threshold = threshold
        if len(scan) >= target_drop_count:
            if threshold != float(DENSE_TRIANGLE_THRESHOLD):
                print(
                    f"Lowered dense-triangle threshold from "
                    f"{DENSE_TRIANGLE_THRESHOLD:g} to {threshold:g} so "
                    f"the maximum requested budget "
                    f"({target_drop_count} units) is supported.",
                    flush=True,
                )
            return scan, threshold
        next_threshold = round(threshold - step, 10)
        if next_threshold < floor and threshold != floor:
            threshold = floor
        else:
            threshold = next_threshold

    print(
        f"Could not support the maximum requested budget ({target_drop_count} units) "
        f"after lowering dense-triangle threshold to floor {floor:g}; "
        f"unsupported budgets will be skipped.",
        flush=True,
    )
    return best_scan, best_threshold


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
    if output_path is not None and (len(budgets) != 1 or max_lora != 1):
        raise ValueError("--output requires exactly one budget and --max-lora-anchors=1")

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
    scan, effective_threshold = _scan_with_adaptive_threshold(
        sim,
        unit_ids,
        num_units=num_units,
        budget_drop_counts=list(budget_drop_counts.values()),
    )

    skipped = [
        (budget, drop_count)
        for budget, drop_count in budget_drop_counts.items()
        if drop_count >= num_units or (drop_count > 0 and drop_count > len(scan))
    ]
    skipped_budgets = {budget for budget, _drop_count in skipped}
    if skipped:
        details = ", ".join(f"{budget}%->{count} units" for budget, count in skipped)
        print(
            f"Skipping budgets beyond the dense-triangle curve (last valid drop count: "
            f"{len(scan)}/{num_units - 1}, threshold>{effective_threshold:g}): {details}",
            flush=True,
        )
    available_budgets = [budget for budget in budgets if budget not in skipped_budgets]
    available_drop_counts = {
        budget: budget_drop_counts[budget] for budget in available_budgets
    }

    selected = {
        budget: _pick_for_drop_count(scan, drop_count)
        for budget, drop_count in available_drop_counts.items()
    }
    plan_root = prune_plan_root or _default_plan_root(backbone)
    plan_root.mkdir(parents=True, exist_ok=True)
    visualization_root = (
        viz_out_dir or (output_path.parent if output_path is not None else plan_root)
    )

    written: List[Path] = []
    if available_budgets:
        source_cache = (
            ""
            if dataset.strip().lower() in {"mean", "min"}
            else str(stage_dir(experiment, dataset, "cache"))
        )
        adapter = get_backbone(backbone)
        hf = str(hf_model or adapter.default_hf_model())
        model = adapter.load_pretrained(
            hf_model=hf, device=str(cache_device), dtype=None, eval_mode=True
        )

    for budget in available_budgets:
        drop_count = budget_drop_counts[budget]
        pick = selected[budget]
        picks = [] if pick is None else [pick]
        for lora_prefix in range(1, max_lora + 1):
            plan, lora_indices = _build_plan_from_prefix(
                picks_prefix=picks,
                lora_prefix=lora_prefix,
                lora_rank=int(lora_rank),
                lora_alpha=float(lora_alpha),
                backbone=backbone,
                dataset=dataset,
                num_units=num_units,
                source_cache=source_cache,
            )
            plan.extra.update(
                {
                    "similarity_source": similarity_source,
                    "pruning_budget_percent": budget,
                    "requested_drop_count": drop_count,
                    "dense_triangle_threshold": effective_threshold,
                    "dense_triangle_initial_threshold": DENSE_TRIANGLE_THRESHOLD,
                    "dense_triangle_threshold_floor": DENSE_TRIANGLE_THRESHOLD_FLOOR,
                    "dense_triangle_threshold_step": DENSE_TRIANGLE_THRESHOLD_STEP,
                    "dense_triangle_selection": "nested_from_max_span",
                    "selected_corner_similarity": (
                        1.0 if pick is None else float(sim[int(pick.i), int(pick.j)])
                    ),
                    "lora_prefix": lora_prefix,
                }
            )
            cache_unit_ids = sorted(
                required_cache_unit_ids(plan, model) if plan.lora else set(),
                key=_unit_id_to_index,
            )
            cache_unit_indices = [_unit_id_to_index(uid) for uid in cache_unit_ids]
            plan.extra["cache_unit_ids"] = cache_unit_ids
            plan.extra["cache_unit_indices"] = cache_unit_indices

            name = _plan_name(budget, lora_prefix)
            out_path = output_path or (plan_root / f"{name}.json")
            plan.save(out_path)
            written.append(out_path)
            print(
                f"Wrote {out_path} (budget={budget}%, drop={len(plan.drop)}/{num_units}, "
                f"lora_prefix={lora_prefix}, lora={len(plan.lora)})",
                flush=True,
            )
            heatmap_path = render_triangle_prune_heatmap(
                sim,
                unit_ids,
                picks,
                out_dir=visualization_root,
                threshold=effective_threshold,
                tag=f"{name}_{backbone}",
                lora_unit_indices=lora_indices,
                cache_unit_indices=cache_unit_indices,
            )
            print(f"Wrote plan heatmap {heatmap_path}", flush=True)

    curve_path = render_budget_similarity_curve(
        scan,
        sim,
        available_drop_counts,
        out_dir=visualization_root,
        backbone=backbone,
        threshold=effective_threshold,
    )
    print(f"Wrote budget curve {curve_path} (last valid drop count={len(scan)})", flush=True)
    return written


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

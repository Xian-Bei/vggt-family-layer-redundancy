from __future__ import annotations

import argparse
from pathlib import Path

from analysis.prune_plan import (
    bundle_strategy_records,
    expand_pruning_strategy_keys,
    load_prune_plan_bundle,
)
from analysis.prune_plan.eval_plots import (
    collect_pruning_eval_payload,
    write_pruning_eval_plots,
)


def _split(raw: str) -> list[str]:
    return [
        value.strip()
        for value in str(raw or "").replace(",", " ").split()
        if value.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render strict pruning metric plots for selected plan points."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--prune-plan-root", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--strategies", required=True)
    parser.add_argument("--selectors", default="")
    parser.add_argument("--min-drop", type=int, required=True)
    parser.add_argument("--max-drop", type=int, required=True)
    parser.add_argument("--drop-step", type=int, required=True)
    parser.add_argument("--dataset-plots", action="store_true")
    parser.add_argument("--average-filename", default="")
    parser.add_argument("--average-title", default="")
    parser.add_argument(
        "--average-strategy-group",
        choices=(
            "all",
            "ours",
            "ours_single",
            "ours_multi",
            "others",
            "single",
            "non_single",
        ),
        default="all",
        help=(
            "Filter only the dataset-average plot. ours_single/ours_multi contain "
            "GT plus the four Ours families at one structure; others contains "
            "GT, Ours_CKA, and other baselines at all available structures."
        ),
    )
    args = parser.parse_args()

    bundle = load_prune_plan_bundle(args.prune_plan_root)
    all_strategies = bundle_strategy_records(bundle)
    requested = expand_pruning_strategy_keys(
        _split(args.strategies),
        available=all_strategies.keys(),
    )
    strategies = {}
    for key in requested:
        normalized = key.lower()
        if normalized not in all_strategies:
            raise FileNotFoundError(
                f"strategy {key!r} is not present in prune plan bundle"
            )
        strategies[normalized] = all_strategies[normalized]
    payload = collect_pruning_eval_payload(
        experiment_root=args.experiment_root,
        prune_plan_root=args.prune_plan_root,
        strategies=strategies,
        datasets=_split(args.datasets),
        bundle=bundle,
        min_drop=args.min_drop,
        max_drop=args.max_drop,
        drop_step=args.drop_step,
        eval_selectors=_split(args.selectors) or None,
    )
    requested_datasets = set(_split(args.datasets))
    found_datasets = set(payload.get("datasets", {}))
    missing_datasets = sorted(requested_datasets - found_datasets)
    if missing_datasets:
        raise FileNotFoundError(
            "missing pruning result roots for datasets: "
            + ", ".join(missing_datasets)
        )
    paths = write_pruning_eval_plots(
        experiment_root=args.experiment_root,
        backbone=args.backbone,
        payload=payload,
        write_dataset_plots=bool(args.dataset_plots),
        average_filename=str(args.average_filename or "") or None,
        average_title=str(args.average_title or "") or None,
        average_strategy_group=str(args.average_strategy_group),
        require_complete=True,
    )
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_physical_pruning_selector,
    load_prune_plan_bundle,
    physical_pruning_selector_aliases,
    physical_pruning_selector_from_record,
    pruning_intervals_from_drop,
    expand_pruning_strategy_keys,
)


@dataclass(frozen=True)
class SelectedPlan:
    selector: str
    num_intervals: int
    references: tuple[str, ...]


def selected_plans(
    prune_plan_root: Path,
    *,
    strategies: Sequence[str],
    min_drop: int,
    max_drop: int,
    drop_step: int,
    explicit_selectors: Optional[Sequence[str]] = None,
) -> list[SelectedPlan]:
    bundle = load_prune_plan_bundle(prune_plan_root)
    plan_records = bundle_plan_records(bundle)
    strategy_records = bundle_strategy_records(bundle)
    references: dict[str, list[str]] = {}
    interval_counts: dict[str, int] = {}

    if explicit_selectors:
        physical_records = {
            physical_pruning_selector_from_record(record): record
            for record in plan_records.values()
        }
        for raw in explicit_selectors:
            selector = canonical_physical_pruning_selector(raw)
            record = physical_records.get(selector)
            if not isinstance(record, dict):
                raise FileNotFoundError(
                    f"physical selector {selector!r} is not present in prune plan bundle"
                )
            references.setdefault(selector, []).append("explicit")
            interval_counts[selector] = len(
                pruning_intervals_from_drop(record.get("drop", []))
            )
    else:
        for strategy_raw in expand_pruning_strategy_keys(
            strategies,
            available=strategy_records.keys(),
        ):
            strategy = str(strategy_raw).strip().lower()
            strategy_record = strategy_records.get(strategy)
            if not isinstance(strategy_record, dict):
                raise FileNotFoundError(
                    f"strategy {strategy_raw!r} is not present in prune plan bundle"
                )
            points = strategy_record.get("points", {})
            if not isinstance(points, dict):
                continue
            for drop_key, plan_key_raw in points.items():
                if not str(drop_key).isdigit():
                    continue
                drop_count = int(drop_key)
                if drop_count < int(min_drop) or drop_count > int(max_drop):
                    continue
                if (drop_count - int(min_drop)) % int(drop_step) != 0:
                    continue
                plan_key = str(plan_key_raw)
                record = plan_records.get(plan_key)
                if not isinstance(record, dict):
                    raise FileNotFoundError(
                        f"strategy {strategy!r} references missing plan {plan_key!r}"
                    )
                selector = physical_pruning_selector_from_record(record)
                references.setdefault(selector, []).append(
                    f"{strategy}@{drop_count}"
                )
                interval_counts[selector] = len(
                    pruning_intervals_from_drop(record.get("drop", []))
                )

    return [
        SelectedPlan(
            selector=selector,
            num_intervals=int(interval_counts[selector]),
            references=tuple(sorted(set(refs))),
        )
        for selector, refs in references.items()
    ]


def existing_summary_path(result_root: Path, selector: str) -> Optional[Path]:
    for alias in physical_pruning_selector_aliases(selector):
        path = Path(result_root) / alias / "summary.json"
        if path.is_file():
            return path
    return None


def missing_plans_for_dataset(
    *,
    result_root: Path,
    plans: Sequence[SelectedPlan],
    training_dataset: bool,
    dataset: str,
) -> list[SelectedPlan]:
    missing: list[SelectedPlan] = []
    for plan in plans:
        if existing_summary_path(result_root, plan.selector) is not None:
            continue
        if training_dataset and int(plan.num_intervals) == 1:
            refs = ", ".join(plan.references)
            raise FileNotFoundError(
                f"training single-interval result is unexpectedly missing: "
                f"dataset={dataset} selector={plan.selector} plans={refs}"
            )
        missing.append(plan)
    return missing


def _split(raw: str) -> list[str]:
    return [
        value.strip()
        for value in str(raw or "").replace(",", " ").split()
        if value.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve missing physical pruning selectors for one dataset."
    )
    parser.add_argument("--prune-plan-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategies", required=True)
    parser.add_argument("--selectors", default="")
    parser.add_argument("--min-drop", type=int, required=True)
    parser.add_argument("--max-drop", type=int, required=True)
    parser.add_argument("--drop-step", type=int, required=True)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    if args.drop_step <= 0:
        raise ValueError("--drop-step must be positive")
    plans = selected_plans(
        args.prune_plan_root,
        strategies=_split(args.strategies),
        min_drop=args.min_drop,
        max_drop=args.max_drop,
        drop_step=args.drop_step,
        explicit_selectors=_split(args.selectors) or None,
    )
    missing = missing_plans_for_dataset(
        result_root=args.result_root,
        plans=plans,
        training_dataset=bool(args.training),
        dataset=args.dataset,
    )
    for plan in missing:
        print(plan.selector)


if __name__ == "__main__":
    main()

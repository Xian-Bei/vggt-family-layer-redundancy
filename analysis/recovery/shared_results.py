from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analysis.prune_plan import (
    load_named_prune_plan,
    physical_pruning_selector_from_drop,
    strategy_interval_selectors,
)

_DERIVED_GHOST_METHODS = {
    "recovery_ghost_sequential",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}


def canonical_selector(plan_root: Path, selector: str) -> str:
    plan = load_named_prune_plan(Path(plan_root), str(selector))
    return physical_pruning_selector_from_drop(plan.drop)


def selectors_for_strategies(
    plan_root: Path,
    strategies: list[str],
    requested_selectors: list[str],
) -> list[str]:
    """Return the stable, canonical selector union for strategy/drop requests."""
    selected: list[str] = []
    for strategy in strategies:
        for value in requested_selectors:
            match = re.fullmatch(r"prune_span(\d+)", value)
            if match is None:
                raise ValueError(
                    "requested recovery selectors must be prune_spanN aliases; "
                    f"got {value!r}"
                )
            drop_count = int(match.group(1))
            plans = strategy_interval_selectors(
                Path(plan_root),
                strategies=[strategy],
                min_drop=drop_count,
                max_drop=drop_count,
                drop_step=1,
            )
            if len(plans) != 1:
                raise ValueError(
                    f"expected exactly one {strategy!r} plan at drop "
                    f"{drop_count}, got {plans}"
                )
            canonical = canonical_selector(Path(plan_root), plans[0])
            if canonical not in selected:
                selected.append(canonical)
    return selected


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def recovery_result_complete(
    output_dir: Path,
    *,
    plan_root: Path,
    backbone: str,
    selector: str,
    recovery_plan_name: str,
) -> bool:
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoints" / "latest.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return False
    config = _load_object(output_dir / "config.json")
    if not config:
        config = _load_object(output_dir / "recovery_summary.json")
    if not config or str(config.get("backbone", "")).strip() != str(backbone).strip():
        return False
    recorded_recovery = str(
        config.get("recovery_plan_name") or config.get("method") or ""
    ).strip()
    expected_recovery = str(recovery_plan_name).strip()
    if recorded_recovery != expected_recovery and not (
        output_dir.is_symlink()
        and expected_recovery in _DERIVED_GHOST_METHODS
        and recorded_recovery == "recovery_ghost"
    ):
        return False
    recorded_selector = str(config.get("prune_plan_name", "")).strip()
    if not recorded_selector:
        return False
    try:
        expected = canonical_selector(plan_root, selector)
        recorded = canonical_selector(plan_root, recorded_selector)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return recorded == expected


def eval_result_complete(
    output_dir: Path,
    *,
    selector: str,
    recovery_plan_name: str,
) -> bool:
    output_dir = Path(output_dir)
    record = _load_object(output_dir / "eval_record_v3.json")
    summary = _load_object(output_dir / "summary.json")
    if not record or not summary:
        return False
    if str(summary.get("selector", "")).strip() != str(selector).strip():
        return False
    recorded_recovery = str(summary.get("recovery_plan_name", "")).strip()
    expected_recovery = str(recovery_plan_name).strip()
    return recorded_recovery == expected_recovery or (
        output_dir.is_symlink()
        and expected_recovery in _DERIVED_GHOST_METHODS
        and recorded_recovery == "recovery_ghost"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    recovery = subparsers.add_parser("recovery-complete")
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.add_argument("--plan-root", type=Path, required=True)
    recovery.add_argument("--backbone", required=True)
    recovery.add_argument("--selector", required=True)
    recovery.add_argument("--recovery-plan-name", required=True)
    evaluation = subparsers.add_parser("eval-complete")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--selector", required=True)
    evaluation.add_argument("--recovery-plan-name", required=True)
    canonical = subparsers.add_parser("canonical-selector")
    canonical.add_argument("--plan-root", type=Path, required=True)
    canonical.add_argument("--selector", required=True)
    args = parser.parse_args()
    if args.command == "recovery-complete":
        complete = recovery_result_complete(
            args.output_dir,
            plan_root=args.plan_root,
            backbone=args.backbone,
            selector=args.selector,
            recovery_plan_name=args.recovery_plan_name,
        )
    elif args.command == "eval-complete":
        complete = eval_result_complete(
            args.output_dir,
            selector=args.selector,
            recovery_plan_name=args.recovery_plan_name,
        )
    else:
        print(canonical_selector(args.plan_root, args.selector))
        return
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()

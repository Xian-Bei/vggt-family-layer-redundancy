from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    canonical_pruning_strategy_key,
    drop_from_physical_pruning_selector,
    load_named_prune_plan,
    load_prune_plan_bundle,
    physical_pruning_selector_from_drop,
    physical_pruning_selector_from_record,
    strategy_interval_selectors,
)
from analysis.prune_plan.eval_plots import (
    GT_MEAN_METRICS,
    metric_value,
)
from analysis.prune_plan.recovery_plan import (
    TOKEN_GROUPED_RECOVERY_BASE_METHODS,
    list_recovery_plan_names,
    recovery_method_semantics_version,
)

_DERIVED_GHOST_METHODS = {
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}
_LEGACY_SINGLE_COMPONENT_SEMANTICS_COMPAT = {
    "recovery_replaceme_cos",
}
_SINGLE_COMPONENT_VERSION_COMPAT = {
    # Before the official current-target behavior was restored, multi-interval
    # ReplaceMe Cosine used Full-model targets. The two objectives are identical
    # for a single interval, so those artifacts remain reusable.
    "recovery_replaceme_cos": {"current_source_full_target_v2"},
}
PRUNE_FAMILY_STRATEGIES = {
    "CKA": ("CKA_single", "CKA_double"),
    "retention": ("retention_single", "retention_double"),
}
SINGLE_DOUBLE_MEAN_MARGIN = 0.1
SINGLE_DOUBLE_SELECTION_DATASETS = (
    "scannet_dev50",
    "7scenes_train",
    "nuscenes_train",
    "waymo_training",
)


def seven_metric_mean_score(metrics: dict[str, Any]) -> float | None:
    """Return the direct arithmetic mean of the seven raw error metrics."""
    values = [metric_value(metrics, metric) for metric in GT_MEAN_METRICS]
    if not all(value is not None for value in values):
        return None
    numeric = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in numeric):
        return None
    return float(sum(numeric) / len(numeric))


def single_double_mean_decision(
    single_score: float,
    double_score: float,
    *,
    margin: float = SINGLE_DOUBLE_MEAN_MARGIN,
) -> dict[str, Any]:
    """Prefer Single unless Double lowers the seven-metric mean beyond margin."""
    single = float(single_score)
    double = float(double_score)
    threshold = float(margin)
    if not all(math.isfinite(value) for value in (single, double, threshold)):
        raise ValueError("single, double, and margin must be finite")
    if threshold < 0.0:
        raise ValueError("margin must be non-negative")
    double_improvement = single - double
    winner = "double" if double_improvement > threshold else "single"
    return {
        "winner": winner,
        "single": single,
        "double": double,
        "gap": double - single,
        "double_improvement": double_improvement,
        "margin": threshold,
        "margin_override": 0.0 < double_improvement <= threshold,
    }


def _validate_single_prune_family(strategies: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(strategy).strip() for strategy in strategies))
    matching_families = [
        family
        for family, family_strategies in PRUNE_FAMILY_STRATEGIES.items()
        if requested and set(requested).issubset(family_strategies)
    ]
    if len(matching_families) != 1:
        raise ValueError(
            "recovery selection candidates must be single/double variants from "
            "exactly one pruning family"
        )
    return requested


def best_recovery_result(
    plan_root: Path,
    *,
    drop_count: int,
    datasets: Sequence[str],
    recovery_plan_name: str,
    strategies: Sequence[str],
) -> dict[str, Any]:
    """Select Single/Double by raw seven-metric mean with a Single margin."""
    strategies = _validate_single_prune_family(strategies)
    requested_datasets = tuple(dict.fromkeys(str(name).strip() for name in datasets if str(name).strip()))
    if not requested_datasets:
        raise ValueError("at least one selection dataset is required")
    if int(drop_count) <= 0:
        raise ValueError(f"drop_count must be positive, got {drop_count}")
    recovery_plan_name = str(recovery_plan_name).strip()
    if not recovery_plan_name:
        raise ValueError("recovery_plan_name is required")

    root = Path(plan_root)
    bundle = load_prune_plan_bundle(root)
    strategy_records = bundle_strategy_records(bundle)
    plan_records = bundle_plan_records(bundle)

    candidates: list[dict[str, Any]] = []
    failures: list[str] = []
    for order, strategy in enumerate(strategies):
        strategy = str(strategy)
        strategy_record = strategy_records.get(strategy, {})
        strategy_points = (
            strategy_record.get("points", {})
            if isinstance(strategy_record, dict)
            else {}
        )
        plan_name = (
            strategy_points.get(str(int(drop_count)))
            if isinstance(strategy_points, dict)
            else None
        )
        plan_record = plan_records.get(str(plan_name)) if plan_name is not None else None
        if not isinstance(plan_record, dict):
            failures.append(f"{strategy}: missing prune plan at drop {drop_count}")
            continue
        selector = physical_pruning_selector_from_record(plan_record)
        scores: list[float] = []
        missing: list[str] = []
        for dataset in requested_datasets:
            method_root = (
                root / dataset / "eval" / "recovery" / selector / recovery_plan_name
            )
            summary = _load_object(method_root / "summary.json")
            recorded_recovery = str(summary.get("recovery_plan_name", "")).strip()
            try:
                recorded_drop = int(summary.get("drop_count", -1))
            except (TypeError, ValueError):
                recorded_drop = -1
            recovery_matches = recorded_recovery == recovery_plan_name or (
                method_root.is_symlink()
                and _symlink_recovery_alias_matches(
                    expected=recovery_plan_name,
                    recorded=recorded_recovery,
                    selector=selector,
                    backbone=str(bundle.get("backbone", "")),
                )
            )
            if (
                not summary
                or str(summary.get("selector", "")).strip() != selector
                or recorded_drop != int(drop_count)
                or not recovery_matches
            ):
                missing.append(dataset)
                continue
            metrics = summary.get("metrics_mean", {})
            score = seven_metric_mean_score(metrics) if isinstance(metrics, dict) else None
            if score is None:
                missing.append(dataset)
                continue
            scores.append(float(score))
        if missing or len(scores) != len(requested_datasets):
            failures.append(f"{strategy}: missing complete metrics for {','.join(missing)}")
            continue
        candidates.append(
            {
                "strategy": strategy,
                "selector": selector,
                "drop_count": int(drop_count),
                "score": float(sum(scores) / len(scores)),
                "datasets": list(requested_datasets),
                "recovery_plan_name": recovery_plan_name,
                "order": order,
            }
        )
    if not candidates:
        detail = "; ".join(failures) if failures else "no candidate strategies"
        raise ValueError(f"no complete recovered candidate at drop {drop_count}: {detail}")
    by_structure = {
        structure: candidate
        for candidate in candidates
        for structure in (
            "single" if str(candidate["strategy"]).endswith("_single") else
            "double" if str(candidate["strategy"]).endswith("_double") else "",
        )
        if structure
    }
    if set(by_structure) == {"single", "double"}:
        decision = single_double_mean_decision(
            float(by_structure["single"]["score"]),
            float(by_structure["double"]["score"]),
        )
        best = by_structure[str(decision["winner"])]
        best = {
            **best,
            "single_score": decision["single"],
            "double_score": decision["double"],
            "selection_margin": decision["margin"],
            "margin_override": decision["margin_override"],
        }
    else:
        best = min(
            candidates,
            key=lambda item: (float(item["score"]), int(item["order"])),
        )
    best.pop("order", None)
    return best


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


def _split_list(value: str) -> list[str]:
    return [item for item in re.split(r"[\s,]+", value) if item]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_single_contiguous_selector(selector: str) -> bool:
    """Return whether a canonical physical selector has exactly one component."""
    try:
        drop = sorted(
            {int(index) for index in drop_from_physical_pruning_selector(selector)}
        )
    except (TypeError, ValueError):
        return False
    return bool(drop) and drop == list(range(drop[0], drop[-1] + 1))


def _symlink_recovery_alias_matches(
    *, expected: str, recorded: str, selector: str, backbone: str = ""
) -> bool:
    """Validate only exact recovery aliases that the runners create."""
    aliases: set[str] = set()
    if (
        str(backbone).strip().lower() == "pi3"
        and expected in TOKEN_GROUPED_RECOVERY_BASE_METHODS
    ):
        aliases.add(TOKEN_GROUPED_RECOVERY_BASE_METHODS[expected])
    if _is_single_contiguous_selector(selector):
        candidates = {expected, *aliases}
        if any(candidate in _DERIVED_GHOST_METHODS for candidate in candidates):
            aliases.add("recovery_ghost")
    return recorded in aliases


def _semantics_compatible(
    *, method: str, recorded_version: str, selector: str
) -> bool:
    expected = recovery_method_semantics_version(method)
    recorded = str(recorded_version).strip()
    if not expected or recorded == expected:
        return True
    if not _is_single_contiguous_selector(selector):
        return False
    if not recorded and method in _LEGACY_SINGLE_COMPONENT_SEMANTICS_COMPAT:
        return True
    return recorded in _SINGLE_COMPONENT_VERSION_COMPAT.get(method, set())


def recovery_result_complete(
    output_dir: Path,
    *,
    plan_root: Path,
    backbone: str,
    selector: str,
    recovery_plan_name: str,
    expected_fit_signature: str = "",
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
    if expected_fit_signature and str(config.get("fit_signature", "")) != str(
        expected_fit_signature
    ):
        return False
    recorded_recovery = str(
        config.get("recovery_plan_name") or config.get("method") or ""
    ).strip()
    expected_recovery = str(recovery_plan_name).strip()
    if (
        str(backbone).strip().lower() == "pi3"
        and expected_recovery in TOKEN_GROUPED_RECOVERY_BASE_METHODS
        and not output_dir.is_symlink()
    ):
        return False
    symlink_alias = (
        output_dir.is_symlink()
        and _symlink_recovery_alias_matches(
            expected=expected_recovery,
            recorded=recorded_recovery,
            selector=selector,
            backbone=backbone,
        )
    )
    if recorded_recovery != expected_recovery and not symlink_alias:
        return False
    recorded_selector = str(config.get("prune_plan_name", "")).strip()
    if not recorded_selector:
        return False
    try:
        expected = canonical_selector(plan_root, selector)
        recorded = canonical_selector(plan_root, recorded_selector)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    if recorded != expected:
        return False
    return _semantics_compatible(
        method=(recorded_recovery if symlink_alias else expected_recovery),
        recorded_version=str(config.get("recovery_semantics_version", "")),
        selector=expected,
    )


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
    if (
        str(record.get("backbone", "")).strip().lower() == "pi3"
        and expected_recovery in TOKEN_GROUPED_RECOVERY_BASE_METHODS
        and not output_dir.is_symlink()
    ):
        return False
    symlink_alias = (
        output_dir.is_symlink()
        and _symlink_recovery_alias_matches(
            expected=expected_recovery,
            recorded=recorded_recovery,
            selector=selector,
            backbone=str(record.get("backbone", "")),
        )
    )
    recovery_matches = recorded_recovery == expected_recovery or symlink_alias
    if not recovery_matches:
        return False
    return _semantics_compatible(
        method=(recorded_recovery if symlink_alias else expected_recovery),
        recorded_version=str(summary.get("recovery_semantics_version", "")),
        selector=str(selector),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    recovery = subparsers.add_parser("recovery-complete")
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.add_argument("--plan-root", type=Path, required=True)
    recovery.add_argument("--backbone", required=True)
    recovery.add_argument("--selector", required=True)
    recovery.add_argument("--recovery-plan-name", required=True)
    recovery.add_argument("--fit-signature", default="")
    evaluation = subparsers.add_parser("eval-complete")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--selector", required=True)
    evaluation.add_argument("--recovery-plan-name", required=True)
    canonical = subparsers.add_parser("canonical-selector")
    canonical.add_argument("--plan-root", type=Path, required=True)
    canonical.add_argument("--selector", required=True)
    strategy_selectors = subparsers.add_parser("selectors-for-strategies")
    strategy_selectors.add_argument("--plan-root", type=Path, required=True)
    strategy_selectors.add_argument("--strategies", required=True)
    strategy_selectors.add_argument("--selectors", required=True)
    recovery_names = subparsers.add_parser("recovery-plan-names")
    recovery_names.add_argument("--plan-root", type=Path, required=True)
    best_recovery = subparsers.add_parser("best-recovery-result")
    best_recovery.add_argument("--plan-root", type=Path, required=True)
    best_recovery.add_argument("--drop-count", type=int, required=True)
    best_recovery.add_argument("--datasets", required=True)
    best_recovery.add_argument("--recovery-plan-name", required=True)
    best_recovery.add_argument(
        "--prune-family",
        type=canonical_pruning_strategy_key,
        choices=tuple(PRUNE_FAMILY_STRATEGIES),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.command == "recovery-complete":
        complete = recovery_result_complete(
            args.output_dir,
            plan_root=args.plan_root,
            backbone=args.backbone,
            selector=args.selector,
            recovery_plan_name=args.recovery_plan_name,
            expected_fit_signature=args.fit_signature,
        )
    elif args.command == "eval-complete":
        complete = eval_result_complete(
            args.output_dir,
            selector=args.selector,
            recovery_plan_name=args.recovery_plan_name,
        )
    elif args.command == "selectors-for-strategies":
        strategies = _split_list(args.strategies)
        selectors = _split_list(args.selectors)
        if not strategies:
            parser.error("--strategies must contain at least one strategy")
        if not selectors:
            parser.error("--selectors must contain at least one selector")
        try:
            selected = selectors_for_strategies(
                args.plan_root, strategies, selectors
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(",".join(selected))
        return
    elif args.command == "recovery-plan-names":
        print(",".join(list_recovery_plan_names(args.plan_root)))
        return
    elif args.command == "best-recovery-result":
        try:
            result = best_recovery_result(
                args.plan_root,
                drop_count=args.drop_count,
                datasets=_split_list(args.datasets),
                recovery_plan_name=args.recovery_plan_name,
                strategies=PRUNE_FAMILY_STRATEGIES[args.prune_family],
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            f"{result['strategy']}\t{result['selector']}\t{float(result['score']):.9f}"
        )
        return
    else:
        print(canonical_selector(args.plan_root, args.selector))
        return
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()

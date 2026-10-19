"""Persist fixed normalization profiles from Stage-2 brute-force intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analysis.prune_plan.eval_plots import (
    DEFAULT_NORMALIZATION_MAX_DROP,
    GT_METRIC_KEYS,
    _fit_pruning_normalization_profile,
    _write_normalization_profile,
    load_interval_summary_records,
    load_upper_average_for_result_root,
    metric_value,
)


def _split(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument(
        "--max-drop",
        type=int,
        default=DEFAULT_NORMALIZATION_MAX_DROP,
        help="Largest contiguous-interval drop count used for visualization bounds (default: 48).",
    )
    args = parser.parse_args(argv)
    for dataset in sorted(_split(args.datasets)):
        result_root = args.plan_root / dataset / "eval" / "pruning"
        summaries = load_interval_summary_records(result_root)
        upper_average = load_upper_average_for_result_root(result_root)
        profile = _fit_pruning_normalization_profile(
            summaries=summaries,
            upper_average=upper_average,
            result_root=result_root,
            max_drop_count=args.max_drop,
        )
        maxima = profile.get("maximum_brute_force_metric", {})
        missing = [
            metric
            for metric in GT_METRIC_KEYS
            if metric_value(maxima, metric) is None
            or metric_value(upper_average, metric) is None
        ]
        if missing:
            raise ValueError(
                f"Incomplete Stage-2 normalization profile for {dataset}: "
                f"missing {missing}"
            )
        path = _write_normalization_profile(result_root, profile)
        print(f"{dataset}\t{path}")


if __name__ == "__main__":
    main()

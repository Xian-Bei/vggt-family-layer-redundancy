from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from analysis.prune_plan import interval_selector
from analysis.prune_plan.build_prune_plan_common import parse_dataset_list
from analysis.prune_plan.build_prune_plan_v3 import (
    _load_gt_aggregate_for_datasets,
    select_retention_single_interval_path,
)


def _unit_ids(plan_root: Path, datasets: Sequence[str]) -> list[str]:
    expected: list[str] | None = None
    for dataset in datasets:
        path = plan_root / dataset / "eval" / "pruning" / "skip_eval_manifest_v3.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        current = [str(unit_id) for unit_id in config.get("unit_ids", [])]
        if not current:
            raise ValueError(f"{path}: unit_ids is empty")
        if expected is not None and current != expected:
            raise ValueError(f"{path}: unit catalog differs across GT source datasets")
        expected = current
    if expected is None:
        raise ValueError("at least one GT source dataset is required")
    return expected


def select_retention_single_interval_at_point(
    *, plan_root: Path, datasets: Sequence[str], point: int
) -> str:
    dataset_names = [str(name).strip() for name in datasets if str(name).strip()]
    if isinstance(point, bool) or int(point) <= 0:
        raise ValueError(f"point must be a positive integer, got {point}")
    unit_ids = _unit_ids(Path(plan_root), dataset_names)
    aggregate, *_ = _load_gt_aggregate_for_datasets(
        plan_root=Path(plan_root),
        datasets=dataset_names,
        unit_ids=unit_ids,
        metric_selector="all",
    )
    path_drops, _path_scores, path_extra = select_retention_single_interval_path(aggregate)
    drop_count = int(point)
    if drop_count not in path_drops:
        available = sorted(int(value) for value in path_drops)
        raise ValueError(
            f"point={drop_count} is unavailable; valid drop counts are {available}"
        )
    components = path_extra[drop_count]["component_intervals"]
    if len(components) != 1:
        raise RuntimeError(
            "canonical Retention single-interval path returned a non-single plan"
        )
    component = components[0]
    return interval_selector(int(component["start"]), int(component["end"]))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Map a manual drop count to its canonical Retention interval."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--point", type=int, required=True)
    args = parser.parse_args(argv)
    selector = select_retention_single_interval_at_point(
        plan_root=args.root,
        datasets=parse_dataset_list(args.datasets),
        point=args.point,
    )
    print(f"{selector} {args.point}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.prune_plan import prune_plan_bundle_path


def load_knee_point(root: Path, *, strategy: str = "ours") -> dict[str, Any]:
    path = prune_plan_bundle_path(Path(root))
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"prune plan bundle does not exist: {path}") from exc
    knee_points = bundle.get("knee_points", {})
    knee = knee_points.get(str(strategy), {}) if isinstance(knee_points, dict) else {}
    if not isinstance(knee, dict):
        raise ValueError(f"{path}: knee_points.{strategy} must be an object")
    plan_name = knee.get("plan_name")
    drop_count = knee.get("drop_count")
    if not isinstance(plan_name, str) or not plan_name.strip():
        raise ValueError(f"{path}: knee_points.{strategy}.plan_name is missing")
    if not isinstance(drop_count, int) or isinstance(drop_count, bool) or drop_count <= 0:
        raise ValueError(
            f"{path}: knee_points.{strategy}.drop_count must be a positive integer"
        )
    return knee


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read the automatically selected pruning knee point."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strategy", default="ours")
    args = parser.parse_args(argv)
    knee = load_knee_point(args.root, strategy=args.strategy)
    print(f"{knee['plan_name']} {knee['drop_count']}")


if __name__ == "__main__":
    main()

"""Backbone environment bootstrap (PYTHONPATH) for shell scripts and stages."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pythonpath_for_backbone(backbone: str, *, repo_root_path: Path | None = None) -> List[str]:
    from analysis.backbones.registry import get_backbone

    root = repo_root_path or repo_root()
    adapter = get_backbone(backbone)
    return [str(p) for p in adapter.pythonpath_entries(root)]


def apply_pythonpath(backbone: str, *, repo_root_path: Path | None = None) -> str:
    """Prepend adapter paths and return the joined PYTHONPATH string."""
    root = repo_root_path or repo_root()
    entries = pythonpath_for_backbone(backbone, repo_root_path=root)
    existing = os.environ.get("PYTHONPATH", "")
    merged: List[str] = []
    for p in entries + [str(root)]:
        if p and p not in merged:
            merged.append(p)
    if existing:
        for p in existing.split(":"):
            if p and p not in merged:
                merged.append(p)
    joined = ":".join(merged)
    os.environ["PYTHONPATH"] = joined
    for p in reversed(entries):
        if p not in sys.path:
            sys.path.insert(0, p)
    return joined


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Print or apply backbone PYTHONPATH.")
    p.add_argument("--backbone", required=True)
    p.add_argument("--apply", action="store_true", help="Set os.environ PYTHONPATH")
    args = p.parse_args()
    if args.apply:
        print(apply_pythonpath(args.backbone))
    else:
        print(":".join(pythonpath_for_backbone(args.backbone)))


if __name__ == "__main__":
    main()

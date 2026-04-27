#!/usr/bin/env python3
"""
Temporary migration script for skip per-scene logs.

What it does for each `*_skip.json` file:
1) Reads `fusion_by_mode.both.gt_delta_upper_by_metric`
2) Writes it into `global_skip_range_loss_upper_by_metric_by_mode["both"]`
3) Removes `fusion_by_mode`
4) Removes `fusion_modes` (and `fusion_models` if present)

Usage:
  python scripts/fix_skip_perscene_both_mode.py \
    --dir analysis_outputs/scannet_redundancy_skip/per_scene

Optional:
  --dry-run   : only report, do not write files
  --backup    : save `<file>.bak` before rewriting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def migrate_one(path: Path, dry_run: bool, backup: bool) -> tuple[bool, str]:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"read/parse failed: {exc}"

    fusion_by_mode = data.get("fusion_by_mode")
    if not isinstance(fusion_by_mode, dict):
        return False, "missing or invalid fusion_by_mode"

    both_obj = fusion_by_mode.get("both")
    if not isinstance(both_obj, dict):
        return False, "missing fusion_by_mode.both"

    gt_delta = both_obj.get("gt_delta_upper_by_metric")
    if not isinstance(gt_delta, dict):
        return False, "missing fusion_by_mode.both.gt_delta_upper_by_metric"

    gskip = data.get("global_skip_range_loss_upper_by_metric_by_mode")
    if gskip is None:
        gskip = {}
        data["global_skip_range_loss_upper_by_metric_by_mode"] = gskip
    if not isinstance(gskip, dict):
        return False, "global_skip_range_loss_upper_by_metric_by_mode is not an object"

    # Move/overwrite target location with both-mode metrics.
    gskip["both"] = gt_delta

    # Remove old/now-redundant fields.
    data.pop("fusion_by_mode", None)
    data.pop("fusion_modes", None)
    data.pop("fusion_models", None)  # tolerate possible typo/legacy naming

    if dry_run:
        return True, "would migrate"

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, "migrated"


def main() -> int:
    parser = argparse.ArgumentParser(description="Move both-mode skip metrics in per-scene logs.")
    parser.add_argument(
        "--dir",
        default="analysis_outputs/scannet_redundancy_skip/per_scene",
        help="Directory containing *_skip.json files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--backup", action="store_true", help="Create .bak before overwrite")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[ERROR] invalid directory: {target_dir}")
        return 2

    files = sorted(target_dir.glob("*_skip.json"))
    if not files:
        print(f"[ERROR] no *_skip.json files found in: {target_dir}")
        return 2

    ok = 0
    fail = 0
    for f in files:
        success, msg = migrate_one(f, dry_run=args.dry_run, backup=args.backup)
        tag = "OK" if success else "FAIL"
        print(f"[{tag}] {f.name}: {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone. success={ok}, failed={fail}, total={len(files)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

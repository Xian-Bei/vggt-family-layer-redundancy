#!/usr/bin/env python3
"""
Move eval visualization outputs from malformed scene paths to the correct layout.

Buggy exports used scene_id like ``['chess/seq-01`` so paths became::
    eval_root/['chess/seq-01/plot.png
instead of::
    eval_root/chess/seq-01/plot.png

NRGBD scenes (no seq subdir) look like ``['kitchen/plot.png`` -> ``kitchen/plot.png``.

Metrics (metrics.json) and eval_record_v3.json were already written to the correct
paths; this script only relocates files still under ``['...`` directories.

Usage:
    python scripts/fix_eval_scene_dir_names.py --dry-run
    python scripts/fix_eval_scene_dir_names.py --apply
    python scripts/fix_eval_scene_dir_names.py --apply --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _fix_segment(name: str) -> str:
    if name.startswith("['"):
        return name[2:]
    return name


def corrected_path(path: Path) -> Path:
    return Path(*[_fix_segment(part) for part in path.parts])


def path_has_malformed_segment(path: Path) -> bool:
    return any(part.startswith("['") for part in path.parts)


def collect_malformed_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and path_has_malformed_segment(p):
            files.append(p)
    return sorted(files)


def collect_malformed_dirs(root: Path) -> List[Path]:
    dirs: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir() and path_has_malformed_segment(p):
            dirs.append(p)
    return sorted(dirs, key=lambda x: len(x.parts), reverse=True)


def plan_moves(
    root: Path, *, overwrite: bool
) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    moves: List[Tuple[Path, Path]] = []
    warnings: List[str] = []
    for src in collect_malformed_files(root):
        dst = corrected_path(src)
        if src == dst:
            continue
        if not str(dst).startswith(str(root.resolve())):
            warnings.append(f"skip (escapes root): {src}")
            continue
        if dst.exists() and dst.resolve() != src.resolve():
            if overwrite:
                warnings.append(f"overwrite: {dst.relative_to(root)}")
            else:
                warnings.append(f"skip existing: {dst.relative_to(root)}")
                continue
        moves.append((src, dst))
    return moves, warnings


def apply_moves(moves: Iterable[Tuple[Path, Path]], *, dry_run: bool, overwrite: bool) -> int:
    n = 0
    for src, dst in moves:
        rel = dst
        print(f"{'[dry-run] ' if dry_run else ''}{src} -> {dst}")
        if dry_run:
            n += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and overwrite:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        n += 1
    return n


def _dir_is_empty(d: Path) -> bool:
    try:
        next(d.iterdir())
        return False
    except StopIteration:
        return True
    except OSError:
        return False


def remove_empty_malformed_dirs(root: Path, *, dry_run: bool) -> int:
    removed = 0
    while True:
        batch = 0
        for d in collect_malformed_dirs(root):
            if not d.exists() or not _dir_is_empty(d):
                continue
            print(f"{'[dry-run] rmdir ' if dry_run else 'rmdir '}{d}")
            if not dry_run:
                d.rmdir()
            batch += 1
        removed += batch
        if batch == 0:
            break
    return removed


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo / "analysis_outputs_v3",
        help="analysis_outputs_v3 root (default: repo/analysis_outputs_v3)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform moves (default: dry-run only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace destination files if they already exist",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    dry_run = not args.apply
    if dry_run:
        print("DRY RUN (pass --apply to execute)\n")

    moves, warnings = plan_moves(root, overwrite=args.overwrite)
    for w in warnings:
        print(f"warn: {w}")
    print(f"\nPlanned file moves: {len(moves)}\n")
    n_moves = apply_moves(moves, dry_run=dry_run, overwrite=args.overwrite)
    n_rm = remove_empty_malformed_dirs(root, dry_run=dry_run)
    print(f"\nDone: {n_moves} file(s) moved, {n_rm} empty malformed dir(s) removed.")
    if dry_run:
        print("Re-run with --apply to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

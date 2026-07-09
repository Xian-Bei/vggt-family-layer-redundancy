#!/usr/bin/env python3
"""Rewrite v3 redundancy NPZ files: neg_mse -> inv_mse.

For each redundancy_graph_*.npz:
  neg_mse_scenes = -MSE_scenes
  inv_mse_scenes = 1 / (1 + MSE_scenes)
  inv_mse_mean = 1 / (1 + mean(MSE_scenes, axis=0))

The original neg_mse_* keys are removed and the file is overwritten atomically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ROOTS = (
    "analysis_outputs_v3",
    "analysis_outputs_v3_half",
    "analysis_outputs_v3_scannetdev",
    "analysis_outputs_v3_all",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved redundancy neg_mse metrics to inv_mse in-place."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Specific redundancy_graph_*.npz files or output roots. "
            "Defaults to existing analysis_outputs_v3* roots."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be updated without writing.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a sibling .bak copy before overwriting.",
    )
    parser.add_argument(
        "--keep-neg-mse",
        action="store_true",
        help="Keep neg_mse_* keys instead of renaming/removing them.",
    )
    return parser.parse_args()


def _iter_npz(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("redundancy_graph_*.npz"))
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def _convert_file(path: Path, *, dry_run: bool, backup: bool, keep_neg_mse: bool) -> str:
    with np.load(path, allow_pickle=True) as npz:
        files = list(npz.files)
        if "neg_mse_scenes" not in files and "neg_mse_mean" not in files:
            if "inv_mse_scenes" in files or "inv_mse_mean" in files:
                return "skip already-converted"
            return "skip no-neg-mse"

        payload = {key: npz[key] for key in files}

    if "neg_mse_scenes" in payload:
        neg_scenes = np.asarray(payload["neg_mse_scenes"], dtype=np.float64)
        mse_scenes = -neg_scenes
        inv_scenes = 1.0 / (1.0 + mse_scenes)
        inv_mean = 1.0 / (1.0 + np.mean(mse_scenes, axis=0))
        payload["inv_mse_scenes"] = inv_scenes
        payload["inv_mse_mean"] = inv_mean
        status = "update scenes+mean"
    else:
        neg_mean = np.asarray(payload["neg_mse_mean"], dtype=np.float64)
        mse_mean = -neg_mean
        payload["inv_mse_mean"] = 1.0 / (1.0 + mse_mean)
        status = "update mean-only"

    if not keep_neg_mse:
        payload.pop("neg_mse_scenes", None)
        payload.pop("neg_mse_mean", None)

    if dry_run:
        return f"dry-run {status}"

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    tmp_path = path.with_name(path.name + ".tmp")
    np.savez(tmp_path, **payload)
    saved_tmp = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if saved_tmp != tmp_path:
        saved_tmp.replace(tmp_path)
    tmp_path.replace(path)
    return status


def main() -> int:
    args = _parse_args()
    roots = args.paths or [Path(root) for root in DEFAULT_ROOTS]
    npz_paths = _iter_npz(roots)
    if not npz_paths:
        print("No redundancy_graph_*.npz files found.")
        return 1

    counts: dict[str, int] = {}
    for path in npz_paths:
        status = _convert_file(
            path,
            dry_run=bool(args.dry_run),
            backup=bool(args.backup),
            keep_neg_mse=bool(args.keep_neg_mse),
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {path}")

    print("Summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

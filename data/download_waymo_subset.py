#!/usr/bin/env python3
"""Download a reproducible Waymo Perception subset from Google Cloud Storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=stdin,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _split_seed(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _list_tfrecords(bucket: str, split: str) -> list[str]:
    prefix = bucket.rstrip("/")
    query = f"{prefix}/{split}/*.tfrecord"
    result = _run(["gcloud", "storage", "ls", query])
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(path for path in paths if path.endswith(".tfrecord"))


def _sample(paths: list[str], *, seed: int, split: str, count: int) -> list[str]:
    if len(paths) < count:
        raise RuntimeError(f"Split {split!r} has only {len(paths)} tfrecords; need {count}.")
    rng = random.Random(_split_seed(seed, split))
    return sorted(rng.sample(paths, count))


def _scene_id(path: str) -> str:
    """Match extract_waymo.py's scene-directory identity (not just filename)."""
    name = Path(path).name
    if not name.startswith("segment-") or "_" not in name:
        raise ValueError(f"Unrecognized Waymo segment name: {name}")
    return name.split("-", 1)[1].split("_", 1)[0]


def _exclude_scenes(paths: list[str], roots: list[Path]) -> list[str]:
    excluded = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing exclusion directory: {root}")
        excluded.update(p.name for p in root.iterdir() if p.is_dir())
    # Multiple segments can share the ID used by the extractor. Select at most
    # one file per ID so two downloads cannot overwrite one output scene.
    unique = {}
    for path in sorted(paths):
        scene = _scene_id(path)
        if scene not in excluded:
            unique.setdefault(scene, path)
    return list(unique.values())


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _download(selected: list[str], dest_dir: Path, manifest_path: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gcloud",
            "storage",
            "cp",
            "--no-clobber",
            "--read-paths-from-stdin",
            "--manifest-path",
            str(manifest_path),
            str(dest_dir) + "/",
        ],
        stdin="".join(f"{path}\n" for path in selected),
    )


def _existing_count(selected: list[str], dest_dir: Path) -> int:
    return sum(1 for path in selected if (dest_dir / Path(path).name).is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default="gs://waymo_open_dataset_v_1_4_3/individual_files",
        help="Waymo GCS root containing training/validation/testing folders.",
    )
    parser.add_argument("--target-root", default="data/waymo_raw", help="Local output root.")
    parser.add_argument("--manifest-dir", default=None, help="Manifest directory. Defaults to target-root/manifests.")
    parser.add_argument("--seed", type=int, default=20260616, help="Sampling seed.")
    parser.add_argument("--count", type=int, default=50, help="Scenes per split.")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--dry-run", action="store_true", help="Only list and sample; do not download.")
    parser.add_argument("--exclude-scene-root", type=Path, action="append", default=[],
                        help="Exclude scene IDs in this extracted split directory; repeatable.")
    args = parser.parse_args()

    target_root = Path(args.target_root)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else target_root / "manifests"
    summary: dict[str, object] = {
        "bucket": args.bucket,
        "seed": args.seed,
        "count_per_split": args.count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for split in args.splits:
        print(f"[waymo] listing {split} from {args.bucket}", flush=True)
        all_paths = _list_tfrecords(args.bucket, split)
        candidates = _exclude_scenes(all_paths, args.exclude_scene_root) if args.exclude_scene_root else all_paths
        selected = _sample(candidates, seed=args.seed, split=split, count=args.count)
        if args.exclude_scene_root:
            selection_path = manifest_dir / f"{split}_extra_selection.json"
            selection = {
                "seed": args.seed, "split": split,
                "scene_ids": [_scene_id(path) for path in selected],
                "source_paths": selected,
                "exclude_scene_roots": [str(root.resolve()) for root in args.exclude_scene_root],
            }
            if selection_path.is_file():
                previous = json.loads(selection_path.read_text())
                if previous != selection:
                    raise RuntimeError(f"Selection changed; refusing to mix datasets in {target_root}")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")

        _write_lines(manifest_dir / f"{split}_all.txt", all_paths)
        _write_lines(manifest_dir / f"{split}_selected.txt", selected)
        _write_lines(manifest_dir / f"{split}_selected_basenames.txt", [Path(p).name for p in selected])

        dest_dir = target_root / split
        transfer_manifest = manifest_dir / f"{split}_gcloud_transfer_manifest.csv"
        if not args.dry_run:
            print(f"[waymo] downloading {len(selected)} {split} scenes -> {dest_dir}", flush=True)
            _download(selected, dest_dir, transfer_manifest)

        existing = _existing_count(selected, dest_dir)
        summary["splits"][split] = {
            "available": len(all_paths),
            "selected": len(selected),
            "downloaded_or_present": existing,
            "target_dir": str(dest_dir),
            "selected_manifest": str(manifest_dir / f"{split}_selected.txt"),
        }
        print(f"[waymo] {split}: selected={len(selected)} present={existing}", flush=True)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "subset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[waymo] wrote {manifest_dir / 'subset_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

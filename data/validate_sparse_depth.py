"""Validate extracted sparse depth and optionally write RGB overlay previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

try:
    from analysis.datasets.sparse_depth import load_sparse_depth, sparse_depth_stats
except ImportError:
    from sparse_depth import load_sparse_depth, sparse_depth_stats


def _image_for_depth(path: Path) -> Path:
    parts = list(path.parts)
    idx = parts.index("depth_points")
    parts[idx] = "image"
    return Path(*parts).with_suffix(".jpg")


def _write_overlay(depth_path: Path, out_path: Path, max_points: int = 20_000) -> None:
    item = load_sparse_depth(depth_path)
    image_path = _image_for_depth(depth_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing matching RGB image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if image.size != (item.image_width, item.image_height):
        raise ValueError(f"Image/depth size mismatch: {image_path} {image.size} vs {item.image_width}x{item.image_height}")
    draw = ImageDraw.Draw(image, "RGBA")
    order = np.arange(item.count)
    if order.size > int(max_points):
        order = order[np.linspace(0, order.size - 1, int(max_points), dtype=np.int64)]
    lo, hi = float(item.depth.min()), float(item.depth.max())
    denom = max(hi - lo, 1e-6)
    for i in order:
        t = (float(item.depth[i]) - lo) / denom
        color = (int(255 * (1.0 - t)), int(255 * t), 64, 210)
        x, y = int(round(float(item.u[i]))), int(round(float(item.v[i])))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Extracted nuScenes/Waymo root or split root")
    ap.add_argument("--preview-dir", type=Path, default=None)
    ap.add_argument("--max-previews", type=int, default=8)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(args.root.glob("**/depth_points/*/*.npz"))
    if not paths:
        raise SystemExit(f"No sparse depth files under {args.root}")
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for path in paths:
        try:
            item = load_sparse_depth(path)
            stats = sparse_depth_stats(item)
            if item.count < int(args.min_samples):
                raise ValueError(f"only {item.count} samples (< {args.min_samples})")
            image_path = _image_for_depth(path)
            if not image_path.is_file():
                raise FileNotFoundError(f"matching RGB missing: {image_path}")
            with Image.open(image_path) as image:
                if image.size != (item.image_width, item.image_height):
                    raise ValueError(f"RGB size {image.size} does not match sparse depth")
            rows.append({"path": str(path), **stats})
        except Exception as exc:
            failures.append(f"{path}: {exc}")

    if args.preview_dir is not None:
        count = min(max(0, int(args.max_previews)), len(paths))
        indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64) if count else []
        for idx in indices:
            path = paths[int(idx)]
            rel = path.relative_to(args.root).with_suffix(".jpg")
            _write_overlay(path, args.preview_dir / rel)

    report = {
        "root": str(args.root),
        "files": len(paths),
        "valid_files": len(rows),
        "failures": failures,
        "total_samples": int(sum(int(r["samples"]) for r in rows)),
        "mean_samples_per_frame": float(np.mean([int(r["samples"]) for r in rows])) if rows else 0.0,
        "frames": rows,
    }
    report_path = args.report or (args.root / "sparse_depth_validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sparse-depth] valid={len(rows)}/{len(paths)} report={report_path}")
    if failures:
        for failure in failures[:20]:
            print(f"[error] {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

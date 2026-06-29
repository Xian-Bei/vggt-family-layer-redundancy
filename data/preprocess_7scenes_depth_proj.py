#!/usr/bin/env python3
"""Generate 7Scenes ``frame-*.depth.proj.png`` (RGB-aligned depth) from ``frame-*.depth.png``.

Geometry matches the SimpleRecon-style convention used by FastVGGT ``eval/data.py``
(``D_TO_RGB`` depth sensor -> RGB sensor).

By default ``setup_7scenes_fastvggt_aligned.sh`` processes **all** sequences listed in
both ``TrainSplit.txt`` and ``TestSplit.txt`` for every scene. Use ``--only-test`` when
you intentionally want to limit work to TestSplit-only (faster, smaller disk churn).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


FOCAL_LENGTH_RGB = 525.0
FOCAL_LENGTH_DEPTH = 585.0
IMG_W = 640
IMG_H = 480

# Calibrated transform depth_sensor -> rgb_sensor
# Source convention aligned with Simplerecon preprocessing.
D_TO_RGB = np.array(
    [
        [9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
        [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
        [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _parse_split(scene_dir: Path, split_name: str) -> list[str]:
    split_file = scene_dir / split_name
    if not split_file.exists():
        return []
    seqs: list[str] = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # Typical line: "sequence 1"
        digits = "".join(c for c in raw if c.isdigit())
        if not digits:
            continue
        seqs.append(f"seq-{int(digits):02d}")
    return seqs


def _register_depth(depth_mm: np.ndarray) -> np.ndarray:
    depth_m = depth_mm.astype(np.float64) / 1000.0
    h, w = depth_m.shape

    ys, xs = np.mgrid[0:h, 0:w]
    x = xs.astype(np.float64).reshape(-1) + 0.5
    y = ys.astype(np.float64).reshape(-1) + 0.5
    d = depth_m.reshape(-1)

    valid = (d > 0.0) & (d < 100.0)
    if not np.any(valid):
        return np.zeros((IMG_H, IMG_W), dtype=np.uint16)

    x = x[valid]
    y = y[valid]
    d = d[valid]

    x = (x - (w / 2.0)) / FOCAL_LENGTH_DEPTH * d
    y = (y - (h / 2.0)) / FOCAL_LENGTH_DEPTH * d

    eye = np.vstack([x, y, d, np.ones_like(d)])
    eye_rgb = D_TO_RGB @ eye

    z = eye_rgb[2]
    nz = z > 1e-9
    if not np.any(nz):
        return np.zeros((IMG_H, IMG_W), dtype=np.uint16)

    eye_rgb = eye_rgb[:, nz]
    z = z[nz]

    u = eye_rgb[0] / z * FOCAL_LENGTH_RGB + (IMG_W / 2.0)
    v = eye_rgb[1] / z * FOCAL_LENGTH_RGB + (IMG_H / 2.0)

    registered = np.full((IMG_H, IMG_W), 2e3, dtype=np.float64)
    for uu, vv, dd in zip(u, v, z):
        xi = int(round(float(uu)))
        yi = int(round(float(vv)))
        if 0 <= xi < IMG_W and 0 <= yi < IMG_H:
            if dd < registered[yi, xi]:
                registered[yi, xi] = dd

    registered[registered > 1e3] = 0.0
    return (registered * 1000.0).astype(np.uint16)


def process_scene(scene_dir: Path, only_test: bool, skip_existing: bool) -> tuple[int, int]:
    splits = ["TestSplit.txt"] if only_test else ["TrainSplit.txt", "TestSplit.txt"]
    seqs: set[str] = set()
    for sp in splits:
        seqs.update(_parse_split(scene_dir, sp))

    done = 0
    total = 0
    for seq in sorted(seqs):
        seq_dir = scene_dir / seq
        if not seq_dir.exists():
            continue
        for depth_path in sorted(seq_dir.glob("frame-*.depth.png")):
            total += 1
            out_path = depth_path.with_name(depth_path.name.replace(".depth.png", ".depth.proj.png"))
            if skip_existing and out_path.exists():
                done += 1
                continue
            depth_mm = np.array(Image.open(depth_path), dtype=np.uint16)
            proj = _register_depth(depth_mm)
            Image.fromarray(proj, mode="I;16").save(out_path)
            done += 1
    return done, total


def main() -> int:
    p = argparse.ArgumentParser(description="Generate 7Scenes frame-*.depth.proj.png files.")
    p.add_argument("--root", type=Path, default=Path("data/7scenes"), help="7Scenes root directory.")
    p.add_argument(
        "--only-test",
        action="store_true",
        help="Only sequences in TestSplit.txt (skip TrainSplit). Default setup uses full train+test.",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip if *.depth.proj.png already exists.")
    args = p.parse_args()

    scene_names = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    total_done = 0
    total_all = 0
    for name in scene_names:
        scene_dir = args.root / name
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            continue
        done, cnt = process_scene(scene_dir, only_test=args.only_test, skip_existing=args.skip_existing)
        total_done += done
        total_all += cnt
        print(f"[scene] {name}: processed {done}/{cnt}", flush=True)
    print(f"[all] processed {total_done}/{total_all}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

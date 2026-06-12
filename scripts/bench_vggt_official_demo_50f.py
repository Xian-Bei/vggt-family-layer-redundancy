#!/usr/bin/env python3
"""
Benchmark 50-frame inference using official VGGT demo / README code path.

Uses:
  - backbones/vggt/vggt (facebookresearch/vggt)
  - load_and_preprocess_images (official)
  - demo_gradio-style autocast (bf16/fp16 on Ampere+)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OFFICIAL_VGGT_ROOT = REPO / "backbones" / "vggt"
sys.path.insert(0, str(OFFICIAL_VGGT_ROOT))
sys.path.insert(0, str(REPO))

from analysis.eval.scannet_eval import SceneData, discover_scannet_scenes, select_scene_image_paths_fastvggt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device: torch.device) -> float:
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for official demo benchmark")

    max_frames = 50
    data_root = REPO / "data/scans_dev50"
    scenes = {s.scene_id: s for s in discover_scannet_scenes(data_root)}
    scene_id = "scene0016_00" if "scene0016_00" in scenes else sorted(scenes.keys())[0]
    scene = select_scene_image_paths_fastvggt(scenes[scene_id], max_frames)
    image_names = [str(p) for p in scene.image_paths]
    print(f"official vggt root={OFFICIAL_VGGT_ROOT}")
    print(f"scene={scene.scene_id} n_images={len(image_names)}")

    # Official README / demo_gradio mixed-precision policy
    cap = torch.cuda.get_device_capability()[0]
    amp_dtype = torch.bfloat16 if cap >= 8 else torch.float16
    print(f"device={device} amp_dtype={amp_dtype} capability={cap}.x")

    print("loading VGGT.from_pretrained('facebook/VGGT-1B') ...")
    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    sync(device)
    print(f"  load={time.perf_counter() - t0:.3f}s")

    print("\n=== load_and_preprocess_images (official, mode=crop default) ===")
    prep_times = []
    images = None
    for i in range(3):
        dt = bench(lambda: load_and_preprocess_images(image_names), device)
        prep_times.append(dt)
        if images is None:
            images = load_and_preprocess_images(image_names)
    print(f"  mean={sum(prep_times)/len(prep_times):.3f}s shape={tuple(images.shape)}")

    images = images.to(device)

    print("\n=== warmup: model(images) with autocast (official demo) ===")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
        bench(lambda: model(images), device)

    print("\n=== model(images) WITH autocast — official README / demo_gradio ===")
    times = []
    for i in range(5):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            dt = bench(lambda: model(images), device)
        print(f"  run {i + 1}: {dt:.3f}s")
        times.append(dt)
    print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

    print("\n=== model(images) WITHOUT autocast — fp32 compute (LaRe skip_gt incremental style) ===")
    times = []
    for i in range(3):
        with torch.no_grad():
            dt = bench(lambda: model(images), device)
        print(f"  run {i + 1}: {dt:.3f}s")
        times.append(dt)
    print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

    print("\n=== aggregator(images.unsqueeze(0)) WITH autocast ===")
    images_batched = images.unsqueeze(0)
    times = []
    for i in range(3):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            dt = bench(lambda: model.aggregator(images_batched), device)
        print(f"  run {i + 1}: {dt:.3f}s")
        times.append(dt)
    print(f"  mean={sum(times)/len(times):.3f}s")


if __name__ == "__main__":
    main()

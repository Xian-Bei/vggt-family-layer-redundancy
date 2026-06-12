#!/usr/bin/env python3
"""Benchmark full VGGT forward vs skip_gt_eval incremental replay (50 frames)."""

from __future__ import annotations

import gc
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.aggregator_compat import aggregator_supports_return_layers
from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.registry import get_dataset
from analysis.eval.eval_utils import get_vgg_input_imgs
from analysis.eval.scannet_eval import SceneData, load_scene_images
from analysis.redundancy.skip_gt_eval_v3 import (
    _build_initial_tokens_vggt_family,
    _build_rope_positions_vggt_family,
    _required_prediction_layers,
    _run_unit_vggt_family,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_once(fn, device: torch.device, warmup: bool = False) -> float:
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    dt = time.perf_counter() - t0
    if not warmup:
        print(f"  {dt:.3f}s")
    return dt


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = REPO / "data/scans_dev50"
    max_frames = 50
    backbone = "vggt"

    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    ds = get_dataset("scannet")

    batch = next(
        ds.iter_scenes(data_root, max_scenes=1, max_frames=max_frames, split=None)
    )
    scene_dir = Path(batch.meta["scene_dir"])
    sel_paths = [Path(p) for p in batch.meta["image_paths"]]
    if isinstance(batch.images, torch.Tensor):
        images_array = batch.images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    else:
        scene = SceneData(scene_id=batch.scene_id, scene_dir=scene_dir, image_paths=sel_paths)
        images_array = np.stack(load_scene_images(scene, mode="crop"))

    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    images_batched = vgg_input.unsqueeze(0).to(device)
    print(
        f"scene={batch.scene_id} frames={images_batched.shape[1]} "
        f"hw={images_batched.shape[-2]}x{images_batched.shape[-1]} device={device}"
    )

    print("loading model...")
    t0 = time.perf_counter()
    hf = adapter.default_hf_model()
    model = adapter.load_pretrained(hf_model=hf, device=str(device), eval_mode=True)
    sync(device)
    print(f"  load={time.perf_counter() - t0:.3f}s")

    unit_ids = [u.unit_id for u in adapter.layer_unit_catalog(model)]
    unit_count = len(unit_ids)
    required = set(_required_prediction_layers(model))

    with torch.no_grad():
        print("\n=== warmup (1x) ===")
        bench_once(lambda: model(images_batched), device, warmup=True)

        print("\n=== model.forward (full, incl patch_embed + aggregator + heads) ===")
        times = []
        for i in range(5):
            print(f"run {i + 1}:", end=" ")
            times.append(bench_once(lambda: model(images_batched), device))
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

        print("\n=== aggregator.forward only ===")
        times = []
        for i in range(5):
            print(f"run {i + 1}:", end=" ")
            times.append(bench_once(lambda: model.aggregator(images_batched), device))
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

        print("\n=== patch_embed only (_build_initial_tokens) ===")
        times = []
        for i in range(5):
            print(f"run {i + 1}:", end=" ")
            times.append(
                bench_once(lambda: _build_initial_tokens_vggt_family(model, images_batched), device)
            )
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

        def incremental_replay_all_units(*, sync_per_unit: bool, use_autocast: bool) -> None:
            ac_kw = autocast_device_dtype(device)
            ctx = torch.autocast(**ac_kw) if use_autocast else nullcontext()
            with ctx:
                pos_frame, pos_global = _build_rope_positions_vggt_family(model, images_batched)
                tokens = _build_initial_tokens_vggt_family(model, images_batched)
                sparse_layers: dict[int, torch.Tensor] = {}
                pend_depth = None
                pend_frame = None
                for u in range(unit_count):
                    tokens, depth_idx, frame_out, global_out = _run_unit_vggt_family(
                        model=model,
                        tokens_bspc=tokens,
                        unit_idx=u,
                        pos_frame=pos_frame,
                        pos_global=pos_global,
                        skip=False,
                    )
                    if sync_per_unit:
                        sync(device)
                    if u % 2 == 0:
                        pend_depth = depth_idx
                        pend_frame = frame_out
                    else:
                        if int(depth_idx) in required:
                            sparse_layers[int(depth_idx)] = torch.cat([pend_frame, global_out], dim=-1)
                        pend_depth = None
                        pend_frame = None

        print("\n=== incremental replay (48 units, autocast, no per-unit sync) ===")
        times = []
        for i in range(5):
            print(f"run {i + 1}:", end=" ")
            times.append(
                bench_once(lambda: incremental_replay_all_units(sync_per_unit=False, use_autocast=True), device)
            )
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

        print("\n=== incremental replay (48 units, fp32, no per-unit sync) ===")
        times = []
        for i in range(3):
            print(f"run {i + 1}:", end=" ")
            times.append(
                bench_once(lambda: incremental_replay_all_units(sync_per_unit=False, use_autocast=False), device)
            )
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

        print("\n=== incremental replay (48 units, fp32, sync AFTER EACH unit) ===")
        times = []
        for i in range(3):
            print(f"run {i + 1}:", end=" ")
            times.append(
                bench_once(lambda: incremental_replay_all_units(sync_per_unit=True, use_autocast=False), device)
            )
        print(f"  mean={sum(times)/len(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

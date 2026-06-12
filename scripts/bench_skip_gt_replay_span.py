#!/usr/bin/env python3
"""Time skip_gt_eval incremental replay per span (with autocast) on one scene."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.registry import get_dataset
from analysis.eval.eval_utils import get_vgg_input_imgs
from analysis.eval.prediction_eval import build_fastvggt_scene_eval_context
from analysis.eval.scannet_eval import SceneData, load_scene_images
from analysis.redundancy.skip_gt_eval_v3 import (
    _compute_span_metrics_incremental_for_scene,
    _required_prediction_layers,
    iter_unit_spans,
)


def main() -> None:
    device = torch.device("cuda")
    backbone = "vggt"
    max_frames = 50
    max_span = 0
    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    ds = get_dataset("scannet")
    batch = next(ds.iter_scenes(REPO / "data/scans_dev50", max_scenes=1, max_frames=max_frames, split=None))

    scene_dir = Path(batch.meta["scene_dir"])
    sel_paths = [Path(p) for p in batch.meta["image_paths"]]
    if isinstance(batch.images, torch.Tensor):
        images_array = batch.images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    else:
        scene = SceneData(scene_id=batch.scene_id, scene_dir=scene_dir, image_paths=sel_paths)
        images_array = np.stack(load_scene_images(scene, mode="crop"))
    vgg_input, _, _ = get_vgg_input_imgs(images_array)
    images_batched = vgg_input.unsqueeze(0).to(device)
    scene_meta = SceneData(scene_id=batch.scene_id, scene_dir=scene_dir, image_paths=sel_paths)
    eval_ctx = build_fastvggt_scene_eval_context(
        scene_meta, REPO / "data/scans_dev50", None, image_hw=(int(vgg_input.shape[-2]), int(vgg_input.shape[-1]))
    )

    model = adapter.load_pretrained(hf_model=adapter.default_hf_model(), device=str(device), eval_mode=True)
    model.eval()
    unit_ids = [u.unit_id for u in adapter.layer_unit_catalog(model)]
    unit_count = len(unit_ids)
    spans = list(iter_unit_spans(unit_count, max_span=max_span))
    spans_by_i: dict[int, list[tuple[int, int]]] = {}
    for i, j in spans:
        if i != 0:
            continue
        spans_by_i.setdefault(i, []).append((i, j))
        if len(spans_by_i[i]) >= 5:
            break

    print(f"scene={batch.scene_id} frames={images_batched.shape[1]} spans_at_i0={len(spans_by_i[0])}")
    _compute_span_metrics_incremental_for_scene(
        model=model,
        images_batched=images_batched,
        required_depth_layers=_required_prediction_layers(model),
        spans_by_i=spans_by_i,
        scene_eval_ctx=eval_ctx,
        scene_meta=scene_meta,
        unit_count=unit_count,
        unit_ids=unit_ids,
        progress=False,
        timing=True,
        scene_id=str(batch.scene_id),
    )


if __name__ == "__main__":
    main()

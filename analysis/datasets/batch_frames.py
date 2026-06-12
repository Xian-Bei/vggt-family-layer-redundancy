"""Frame-count helpers for EvalSampleBatch."""

from __future__ import annotations

from typing import Any

import torch

from analysis.datasets.contracts import EvalSampleBatch


def count_eval_frames(batch: EvalSampleBatch) -> int:
    """Number of frames selected for this scene (after plugin sampling + max_frames cap)."""
    paths = batch.meta.get("image_paths")
    if paths:
        return len(paths)
    fids = batch.meta.get("frame_ids")
    if fids:
        return len(fids)
    if batch.frame_ids:
        return len(batch.frame_ids)

    images: Any = batch.images
    if isinstance(images, torch.Tensor):
        if images.dim() == 5:
            return int(images.shape[1])
        if images.dim() == 4:
            return int(images.shape[0])
    if isinstance(images, list):
        return len(images)
    if isinstance(images, dict) and "img" in images:
        img = images["img"]
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                return int(img.shape[0])
            if img.dim() == 5:
                return int(img.shape[1])
    return 0

"""Shared helpers for cache-first training (dummy images, geometry inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


def infer_train_cache_geometry(
    row: Dict[str, Any],
    model: nn.Module,
    get_layer_cpu: Callable[[int], torch.Tensor],
    *,
    cache_dir: Optional[Path] = None,
) -> Tuple[int, int, int, int]:
    """Return (height, width, seq_len, patch_start_idx)."""
    from analysis.redundancy.cache_layout_v3 import require_minimal_cache_geometry

    require_minimal_cache_geometry(row, cache_dir=cache_dir or Path("."))

    teacher_pred = row.get("pred") or {}
    hw = row.get("image_hw")
    if isinstance(teacher_pred, dict) and "depth" in teacher_pred:
        td = teacher_pred["depth"]
        if isinstance(td, torch.Tensor):
            if td.dim() == 5:
                h, w = int(td.shape[2]), int(td.shape[3])
            elif td.dim() == 4:
                h, w = int(td.shape[-2]), int(td.shape[-1])
            else:
                raise ValueError(f"Unsupported teacher depth shape: {tuple(td.shape)}")
            seq_len = int(row.get("num_frames", 1))
            patch_start = int(row.get("patch_start_idx", getattr(model, "patch_start_idx", 0)))
            return h, w, seq_len, patch_start
    if hw is not None:
        h, w = int(hw[0]), int(hw[1])
        seq_len = int(row.get("num_frames", 1))
        patch_start = int(row.get("patch_start_idx", getattr(model, "patch_start_idx", 0)))
        return h, w, seq_len, patch_start

    from analysis.redundancy.cache_layout_v3 import is_prune_plan_minimal_scene_metadata

    if is_prune_plan_minimal_scene_metadata(row):
        raise RuntimeError(
            "Minimal training cache metadata is missing image_hw. "
            "Re-run cache with PRUNE_PLAN_ROOT and --save-cache enabled."
        )

    layer0 = get_layer_cpu(0)
    patch_start = int(row.get("patch_start_idx", getattr(model, "patch_start_idx", 0)))
    patch_tokens = int(layer0.shape[2]) - patch_start
    agg = getattr(model, "aggregator", None)
    ps = int(getattr(agg, "patch_size", getattr(model, "patch_size", 14)))
    inferred = None
    for cand in ((392, 518), (518, 392), (518, 518), (512, 512)):
        ph, pw = cand[0] // ps, cand[1] // ps
        if ph * pw == patch_tokens:
            inferred = cand
            break
    if inferred is None:
        side = int(round(patch_tokens**0.5))
        inferred = (side * ps, side * ps)
    h, w = inferred
    seq_len = int(row.get("num_frames", layer0.shape[1]))
    return h, w, seq_len, patch_start


def build_vggt_family_dummy_images(
    *,
    seq_len: int,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros((1, seq_len, 3, h, w), device=device, dtype=torch.float32)


def build_pi3_dummy_images(
    *,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros((1, 1, 3, h, w), device=device, dtype=torch.float32)

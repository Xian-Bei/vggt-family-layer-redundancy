"""Sparse driving-depth rasterization aligned with each backbone's RGB preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from analysis.datasets.sparse_depth import SparseDepth, load_sparse_depth


def load_raw_eval_images(image_paths: Sequence[Path | str]) -> torch.Tensor:
    """Load scene RGB paths to ``[S, 3, H, W]`` float tensor in ``[0, 1]`` (no resize)."""
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    if not image_paths:
        raise ValueError("At least one image path is required")
    images: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    for raw_path in image_paths:
        img = Image.open(raw_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        tensor = to_tensor(img.convert("RGB"))
        shapes.add((int(tensor.shape[1]), int(tensor.shape[2])))
        images.append(tensor)
    if len(shapes) != 1:
        raise ValueError(f"Scene images must share one size, got {sorted(shapes)}")
    return torch.stack(images, dim=0)


def vggt_crop_output_hw(image_width: int, image_height: int, target_width: int = 518) -> Tuple[int, int]:
    """Mirror ``vggt.utils.load_fn.load_and_preprocess_images(mode='crop')``."""
    new_height = round(int(image_height) * (int(target_width) / int(image_width)) / 14) * 14
    return min(int(new_height), int(target_width)), int(target_width)


def _aspect_crop_dims(image_height: int, image_width: int) -> Tuple[int, int, int, int]:
    """Return ``(crop_top, crop_left, crop_h, crop_w)`` for Omega aspect-ratio crop."""
    h, w = int(image_height), int(image_width)
    ratio = h / max(w, 1)
    if ratio < 0.5:
        crop_w = min(w, max(1, int(round(h / 0.5))))
        left = max((w - crop_w) // 2, 0)
        return 0, left, h, crop_w
    if ratio > 2.0:
        crop_h = min(h, max(1, int(round(w * 2.0))))
        top = max((h - crop_h) // 2, 0)
        return top, 0, crop_h, w
    return 0, 0, h, w


def omega_balanced_output_hw(
    image_height: int,
    image_width: int,
    *,
    image_resolution: int = 512,
    patch_size: int = 16,
) -> Tuple[int, int]:
    """Mirror ``vggt_omega_adapter._balanced_target_shape`` after aspect crop."""
    crop_top, crop_left, crop_h, crop_w = _aspect_crop_dims(image_height, image_width)
    del crop_top, crop_left
    ratio = float(crop_h) / max(float(crop_w), 1.0)
    token_number = (int(image_resolution) // int(patch_size)) ** 2
    w_patches = max(1, int(round(math.sqrt(token_number / max(ratio, 1e-8)))))
    h_patches = max(1, int(round(token_number / max(float(w_patches), 1.0))))
    return h_patches * patch_size, w_patches * patch_size


def _rasterize_zbuffer(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    *,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    flat = np.full(out_h * out_w, np.inf, dtype=np.float32)
    np.minimum.at(flat, y[valid] * out_w + x[valid], depth[valid])
    out = flat.reshape(out_h, out_w)
    out[~np.isfinite(out)] = 0.0
    return out


def rasterize_for_vggt_crop(item: SparseDepth, output_hw: Tuple[int, int]) -> np.ndarray:
    """Map original-image samples through VGGT resize/crop and apply a z-buffer."""
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    expected_h, expected_w = vggt_crop_output_hw(item.image_width, item.image_height, target_width=out_w)
    if (out_h, out_w) != (expected_h, expected_w):
        raise ValueError(
            f"VGGT input shape mismatch for {item.image_width}x{item.image_height}: "
            f"expected {(expected_h, expected_w)}, got {(out_h, out_w)}"
        )
    resized_h = round(item.image_height * (out_w / item.image_width) / 14) * 14
    crop_top = max(0, (int(resized_h) - out_h) // 2)
    x = np.rint(item.u * (out_w / item.image_width)).astype(np.int64)
    y = np.rint(item.v * (resized_h / item.image_height) - crop_top).astype(np.int64)
    return _rasterize_zbuffer(x, y, item.depth, out_h=out_h, out_w=out_w)


def rasterize_for_omega_balanced(item: SparseDepth, output_hw: Tuple[int, int]) -> np.ndarray:
    """Map original-image samples through Omega balanced resize and apply a z-buffer."""
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    expected_h, expected_w = omega_balanced_output_hw(item.image_height, item.image_width)
    if (out_h, out_w) != (expected_h, expected_w):
        raise ValueError(
            f"Omega input shape mismatch for {item.image_width}x{item.image_height}: "
            f"expected {(expected_h, expected_w)}, got {(out_h, out_w)}"
        )
    crop_top, crop_left, crop_h, crop_w = _aspect_crop_dims(item.image_height, item.image_width)
    u = item.u - float(crop_left)
    v = item.v - float(crop_top)
    x = np.rint(u * (out_w / crop_w)).astype(np.int64)
    y = np.rint(v * (out_h / crop_h)).astype(np.int64)
    return _rasterize_zbuffer(x, y, item.depth, out_h=out_h, out_w=out_w)


def load_driving_depth_stack_vggt(
    depth_paths: Sequence[Path | str],
    *,
    output_hw: Tuple[int, int],
) -> tuple[np.ndarray, int]:
    maps = []
    valid_frames = 0
    for path in depth_paths:
        item = load_sparse_depth(path)
        depth = rasterize_for_vggt_crop(item, output_hw)
        maps.append(depth)
        if np.count_nonzero(depth) > 0:
            valid_frames += 1
    if not maps:
        raise ValueError("No sparse driving-depth paths provided")
    return np.stack(maps, axis=0).astype(np.float32, copy=False), int(valid_frames)


def load_driving_depth_stack_omega(
    depth_paths: Sequence[Path | str],
    *,
    output_hw: Tuple[int, int],
) -> tuple[np.ndarray, int]:
    maps = []
    valid_frames = 0
    for path in depth_paths:
        item = load_sparse_depth(path)
        depth = rasterize_for_omega_balanced(item, output_hw)
        maps.append(depth)
        if np.count_nonzero(depth) > 0:
            valid_frames += 1
    if not maps:
        raise ValueError("No sparse driving-depth paths provided")
    return np.stack(maps, axis=0).astype(np.float32, copy=False), int(valid_frames)


def load_driving_depth_stack(
    depth_paths: Sequence[Path | str],
    *,
    output_hw: Tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Backward-compatible alias for VGGT-crop sparse depth rasterization."""
    return load_driving_depth_stack_vggt(depth_paths, output_hw=output_hw)

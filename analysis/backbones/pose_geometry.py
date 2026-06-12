"""
Pose decode and depth unprojection aligned with each backbone's official demo.

Vendor copies live under ``analysis.backbones.vendor_inference`` (from
``backbones/*/demo_gradio.py``). Stages import this module instead of
``analysis.redundancy.pose_geometry``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _squeeze_batch_like_demo(arr: np.ndarray) -> np.ndarray:
    """Match demo_gradio numpy conversion: drop leading batch dim when B==1."""
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _depth_or_conf_to_shw(x: np.ndarray, name: str) -> np.ndarray:
    """Normalize depth/conf to (S, H, W) for demo-aligned unprojection."""
    d = x.astype(np.float64)
    while d.ndim >= 4 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 4:
        if d.shape[1] == 1:
            d = d[:, 0, :, :]
        elif d.shape[-1] == 1:
            d = d[..., 0]
        else:
            raise ValueError(f"{name}: expected channel-last or (S,1,H,W), got {d.shape}")
    if d.ndim != 3:
        raise ValueError(f"{name}: expected (S,H,W), got shape {d.shape}")
    return d


def _depth_to_demo_layout(depth: torch.Tensor) -> np.ndarray:
    """Depth as numpy; Omega demo uses ``depth_map[..., 0]`` → (S, H, W, 1)."""
    d = depth.detach().float().cpu().numpy()
    if d.ndim == 5:
        d = _squeeze_batch_like_demo(d)
    if d.ndim == 4 and d.shape[-1] == 1:
        return d
    if d.ndim == 4:
        return d[..., None]
    if d.ndim == 3:
        return d[..., None]
    shw = _depth_or_conf_to_shw(d, "depth")
    return shw[..., None]


def _extrinsic_intrinsic_to_demo_numpy(
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    extr = extrinsic.detach().float().cpu().numpy()
    intr = intrinsic.detach().float().cpu().numpy()
    extr = _squeeze_batch_like_demo(extr)
    intr = _squeeze_batch_like_demo(intr)
    if extr.ndim == 4 and extr.shape[-2:] == (4, 4):
        extr = extr[..., :3, :4]
    return extr, intr


def decode_pose_enc_to_extrinsics_intrinsics(
    pose_enc: torch.Tensor,
    image_size_hw: Tuple[int, int],
    *,
    backbone: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode pose using the same functions as each backbone's demo_gradio.py."""
    h, w = int(image_size_hw[0]), int(image_size_hw[1])
    key = (backbone or "").strip().lower()

    if key == "vggt_omega":
        from analysis.backbones.vendor_inference.vggt_omega_pose_enc import encoding_to_camera

        extrinsic, intrinsics = encoding_to_camera(pose_enc, (h, w))
        return extrinsic.float(), intrinsics.float() if intrinsics is not None else None

    from analysis.backbones.vendor_inference.vggt_pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(h, w))
    return extrinsic.float(), intrinsics.float() if intrinsics is not None else None


def unproject_depth_payload(
    depth: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    *,
    backbone: Optional[str] = None,
) -> np.ndarray:
    """Unproject depth to world points [S, H, W, 3] using each backbone's demo code path."""
    key = (backbone or "").strip().lower()
    depth_np = _depth_to_demo_layout(depth)
    extr_np, intr_np = _extrinsic_intrinsic_to_demo_numpy(extrinsic, intrinsic)

    if key == "vggt_omega":
        from analysis.backbones.vendor_inference.vggt_omega_unproject import (
            unproject_depth_map_to_point_map,
        )

        return unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)

    from analysis.backbones.vendor_inference.vggt_geometry import unproject_depth_map_to_point_map

    return unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)

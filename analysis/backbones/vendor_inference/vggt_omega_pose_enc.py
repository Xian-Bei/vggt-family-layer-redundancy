"""
VGGT-Omega pose decode — copied from the official demo import path.

Source: ``backbones/vggt-omega/demo_gradio.py`` imports
``encoding_to_camera`` from ``vggt_omega.utils.pose_enc`` (see ~L22, L50–53).
``encoding_to_camera`` and ``quat_to_mat`` below are verbatim from
``backbones/vggt-omega/vggt_omega/utils/pose_enc.py`` (L29–52) and
``backbones/vggt-omega/vggt_omega/utils/rotation.py`` (L13–42).
"""

from __future__ import annotations

import torch


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternion Order: XYZW (scalar-last)."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True):
    """Decode VGGT-Omega pose encoding into extrinsics and intrinsics."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics

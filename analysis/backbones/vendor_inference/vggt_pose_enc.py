"""
VGGT pose decode — copied from the official demo import path.

Source: ``backbones/vggt/demo_gradio.py`` imports
``pose_encoding_to_extri_intri`` from ``vggt.utils.pose_enc`` (see ~L24, L79).
``pose_encoding_to_extri_intri`` and ``quat_to_mat`` below are verbatim from
``backbones/vggt/vggt/utils/pose_enc.py`` (L62–124) and
``backbones/vggt/vggt/utils/rotation.py`` (L14–44).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    """
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


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    return standardize_quaternion(out[..., [1, 2, 3, 0]])


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to the official VGGT 9D pose encoding."""
    if pose_encoding_type == "absT_quaR_FoV":
        r = extrinsics[:, :, :3, :3]
        t = extrinsics[:, :, :3, 3]
        quat = mat_to_quat(r)
        h, w = image_size_hw
        fov_h = 2 * torch.atan((h / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((w / 2) / intrinsics[..., 0, 0])
        return torch.cat([t, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    raise NotImplementedError


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    """Convert a pose encoding back to camera extrinsics and intrinsics (demo default)."""
    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

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
    else:
        raise NotImplementedError

    return extrinsics, intrinsics

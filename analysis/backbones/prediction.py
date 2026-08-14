"""Shared prediction normalization helpers used by backbone adapters."""

from __future__ import annotations

from typing import Any, Dict

import torch

from analysis.backbones.contracts import PredictionBundle
from analysis.backbones.pose_geometry import (
    decode_pose_enc_to_extrinsics_intrinsics,
    unproject_depth_payload,
)


def bundle_from_pose_depth(
    *,
    raw: Dict[str, Any],
    images: torch.Tensor,
    backbone_name: str,
    defer_depth_unprojection: bool = False,
) -> PredictionBundle:
    """Normalize VGGT-like raw pose/depth outputs into a PredictionBundle."""
    pose_enc = raw.get("pose_enc") if isinstance(raw, dict) else None
    depth = raw.get("depth") if isinstance(raw, dict) else None
    depth_conf = raw.get("depth_conf") if isinstance(raw, dict) else None
    points = None
    camera_poses = None
    raw_out = dict(raw) if isinstance(raw, dict) else {}
    if isinstance(pose_enc, torch.Tensor):
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        extrinsic_w2c, intrinsic = decode_pose_enc_to_extrinsics_intrinsics(
            pose_enc,
            image_hw,
            backbone=backbone_name,
        )
        if isinstance(depth, torch.Tensor) and intrinsic is not None:
            if defer_depth_unprojection:
                raw_out["deferred_unprojection_backbone"] = str(backbone_name)
                raw_out["deferred_unprojection_extrinsic"] = extrinsic_w2c
                raw_out["deferred_unprojection_intrinsic"] = intrinsic
            else:
                world_np = unproject_depth_payload(
                    depth,
                    extrinsic_w2c,
                    intrinsic,
                    backbone=backbone_name,
                )
                points = torch.from_numpy(world_np).to(
                    device=depth.device, dtype=depth.dtype
                )
                if points.dim() == 4:
                    points = points.unsqueeze(0)
        bs, seq = int(extrinsic_w2c.shape[0]), int(extrinsic_w2c.shape[1])
        last_row = torch.zeros(
            bs,
            seq,
            1,
            4,
            device=extrinsic_w2c.device,
            dtype=extrinsic_w2c.dtype,
        )
        last_row[..., 0, 3] = 1.0
        w2c4 = torch.cat([extrinsic_w2c, last_row], dim=-2)
        camera_poses = torch.linalg.inv(w2c4)

    if isinstance(points, torch.Tensor):
        raw_out["world_points_from_depth"] = points
    return PredictionBundle(
        pose_enc=pose_enc if isinstance(pose_enc, torch.Tensor) else None,
        depth=depth if isinstance(depth, torch.Tensor) else None,
        depth_conf=depth_conf if isinstance(depth_conf, torch.Tensor) else None,
        points=points if isinstance(points, torch.Tensor) else None,
        camera_poses=camera_poses if isinstance(camera_poses, torch.Tensor) else None,
        raw=raw_out,
    )


def bundle_from_pi3_raw(raw: Dict[str, Any]) -> PredictionBundle:
    """Normalize Pi3 raw predictions; dense depth is intentionally absent."""
    return PredictionBundle(
        points=raw.get("points"),
        local_points=raw.get("local_points"),
        conf=raw.get("conf"),
        camera_poses=raw.get("camera_poses"),
        raw=dict(raw),
    )

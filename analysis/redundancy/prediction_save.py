"""
Prediction payload helpers.

Mainline usage:
- Stage 1 / Stage 2 build in-memory payloads for GT evaluation and visualization export.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from analysis.backbones.pose_geometry import decode_pose_enc_to_extrinsics_intrinsics


def build_scene_prediction_payload(
    pred_cpu: Dict[str, torch.Tensor],
    image_hw: Tuple[int, int],
    *,
    storage_dtype: torch.dtype,
    backbone: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the shared in-memory prediction payload used by GT eval and visualization."""
    H, W = image_hw
    payload: Dict[str, Any] = {"image_hw": [H, W]}
    if backbone:
        payload["backbone"] = str(backbone)
    if "pose_enc" in pred_cpu:
        pe = pred_cpu["pose_enc"]
        extri, intri = decode_pose_enc_to_extrinsics_intrinsics(
            pe, image_size_hw=(H, W), backbone=backbone
        )
        payload["pose_enc"] = pe.to(dtype=storage_dtype).detach().cpu()
        payload["extrinsic"] = extri.float().cpu()
        payload["intrinsic"] = intri.float().cpu() if intri is not None else None
    if "world_points_from_depth" in pred_cpu and pred_cpu["world_points_from_depth"].numel() > 0:
        payload["world_points_from_depth"] = (
            pred_cpu["world_points_from_depth"].to(dtype=storage_dtype).detach().cpu()
        )
    # Exclude point-head outputs (world_points / world_points_conf) across the project.
    for key in ("depth", "depth_conf"):
        if key in pred_cpu and pred_cpu[key].numel() > 0:
            payload[key] = pred_cpu[key].to(dtype=storage_dtype).detach().cpu()
    return payload

"""
Prediction payload helpers.

Mainline usage:
- Stage 1 / Stage 2 build in-memory payloads for GT evaluation and visualization export.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def build_scene_prediction_payload(pred_cpu: Dict[str, torch.Tensor], image_hw: Tuple[int, int]) -> Dict[str, Any]:
    """Build the shared in-memory prediction payload used by GT eval and visualization."""
    H, W = image_hw
    payload: Dict[str, Any] = {"image_hw": [H, W]}
    if "pose_enc" in pred_cpu:
        pe = pred_cpu["pose_enc"]
        extri, intri = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W))
        payload["pose_enc"] = pe.half()
        payload["extrinsic"] = extri.float().cpu()
        payload["intrinsic"] = intri.float().cpu() if intri is not None else None
    for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
        if key in pred_cpu and pred_cpu[key].numel() > 0:
            payload[key] = pred_cpu[key].half()
    return payload

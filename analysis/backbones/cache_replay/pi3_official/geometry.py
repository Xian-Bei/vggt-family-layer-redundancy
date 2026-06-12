"""Small subset of Pi3 geometry utilities used by the official loss.

Vendored to support :class:`~analysis.backbones.cache_replay.pi3_official.loss.Pi3Loss`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def homogenize_points(x: torch.Tensor) -> torch.Tensor:
    """Append homogeneous 1 to the last dimension."""
    ones = torch.ones_like(x[..., :1])
    return torch.cat([x, ones], dim=-1)


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3].unsqueeze(-1)
    R_inv = R.transpose(-2, -1)
    t_inv = -torch.matmul(R_inv, t)
    bottom = torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
    return torch.cat([torch.cat([R_inv, t_inv], dim=-1), bottom], dim=-2)


def depth_edge(
    depth: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    """Edge mask from depth discontinuities (ported from Pi3 utils)."""
    if depth.dim() < 2:
        return torch.zeros_like(depth, dtype=torch.bool)

    if atol is None and rtol is None:
        # Default from Pi3 loss call site.
        rtol = 0.03
    atol_f = float(atol) if atol is not None else 0.0
    rtol_f = float(rtol) if rtol is not None else 0.0

    d = depth.float()
    while d.dim() < 4:
        d = d.unsqueeze(0)
    # [B,1,H,W]
    if d.shape[-1] <= 1 or d.shape[-2] <= 1:
        out = torch.zeros(d.shape[:-1], device=d.device, dtype=torch.bool)
        return out.reshape(depth.shape[:-1])

    k = int(kernel_size)
    pad = k // 2
    d_pad = F.pad(d, (pad, pad, pad, pad), mode="replicate")
    # Neighborhood min/max.
    unfold = d_pad.unfold(-2, k, 1).unfold(-2, k, 1)  # [..., H, W, k, k]
    d_min = unfold.amin(dim=(-1, -2))
    d_max = unfold.amax(dim=(-1, -2))
    thresh = atol_f + rtol_f * d.clamp_min(1e-6)
    edge = (d_max - d_min) > thresh
    edge = edge.reshape(depth.shape)
    if mask is not None:
        edge = edge & mask.to(edge.device)
    return edge


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


class IdentityBlock(nn.Module):
    def forward(self, x, pos=None):  # noqa: D401
        return x


@dataclass
class PruneConfig:
    scheme_name: str = "baseline0_none"
    prune_head_ratio: float = 0.0
    prune_ffn_ratio: float = 0.0
    global_skip_layers: Sequence[int] = ()
    global_head_layers: Sequence[int] = ()
    global_ffn_layers: Sequence[int] = ()


def prune_config_from_scheme_dict(d: Dict[str, Any]) -> PruneConfig:
    """Build :class:`PruneConfig` from a scheme object (e.g. in ``schemes.example.json``)."""
    return PruneConfig(
        scheme_name=str(d.get("scheme_name", "unnamed_scheme")),
        prune_head_ratio=float(d.get("prune_head_ratio", 0.0)),
        prune_ffn_ratio=float(d.get("prune_ffn_ratio", 0.0)),
        global_skip_layers=list(d.get("global_skip_layers", [])),
        global_head_layers=list(d.get("global_head_layers", [])),
        global_ffn_layers=list(d.get("global_ffn_layers", [])),
    )


def _head_importance(attn_module: nn.Module) -> torch.Tensor:
    dim = attn_module.qkv.in_features
    heads = attn_module.num_heads
    head_dim = dim // heads
    q_w = attn_module.qkv.weight[:dim].detach().reshape(heads, head_dim, dim)
    return q_w.abs().mean(dim=(1, 2))


def prune_attention_heads(attn_module: nn.Module, ratio: float) -> List[int]:
    if ratio <= 0:
        return []
    dim = attn_module.qkv.in_features
    heads = attn_module.num_heads
    head_dim = dim // heads
    n_prune = int(heads * ratio)
    if n_prune <= 0:
        return []

    importance = _head_importance(attn_module)
    prune_ids = torch.argsort(importance)[:n_prune].tolist()
    with torch.no_grad():
        for hid in prune_ids:
            q_start = hid * head_dim
            q_end = (hid + 1) * head_dim
            for chunk in range(3):
                row_start = chunk * dim + q_start
                row_end = chunk * dim + q_end
                attn_module.qkv.weight[row_start:row_end].zero_()
                if attn_module.qkv.bias is not None:
                    attn_module.qkv.bias[row_start:row_end].zero_()
            # Zero out output channels receiving this head.
            attn_module.proj.weight[:, q_start:q_end].zero_()
    return prune_ids


def prune_ffn_channels(mlp_module: nn.Module, ratio: float) -> List[int]:
    if ratio <= 0:
        return []
    hidden = mlp_module.fc1.weight.shape[0]
    n_prune = int(hidden * ratio)
    if n_prune <= 0:
        return []
    importance = mlp_module.fc1.weight.detach().abs().mean(dim=1)
    prune_ids = torch.argsort(importance)[:n_prune].tolist()
    with torch.no_grad():
        mlp_module.fc1.weight[prune_ids].zero_()
        if mlp_module.fc1.bias is not None:
            mlp_module.fc1.bias[prune_ids].zero_()
        mlp_module.fc2.weight[:, prune_ids].zero_()
    return prune_ids


def skip_layers(blocks: nn.ModuleList, layer_ids: Sequence[int]) -> List[int]:
    """Replace selected global blocks with identity (tensor passthrough). Caller should ``model.to(device)`` after."""
    used = []
    for idx in layer_ids:
        if 0 <= idx < len(blocks):
            blocks[idx] = IdentityBlock()
            used.append(idx)
    return used


def apply_structured_pruning(model: nn.Module, config: PruneConfig) -> Dict[str, Sequence[int]]:
    pruned: Dict[str, Sequence[int]] = {
        "scheme_name": [config.scheme_name],
        "global_heads_layers": [],
        "global_ffn_layers": [],
        "global_skip_layers": [],
    }

    global_head_layers: List[int] = []
    global_ffn_layers: List[int] = []
    for idx, blk in enumerate(model.aggregator.global_blocks):
        if hasattr(blk, "attn") and idx in set(config.global_head_layers):
            _ = prune_attention_heads(blk.attn, config.prune_head_ratio)
            global_head_layers.append(idx)
        if hasattr(blk, "mlp") and idx in set(config.global_ffn_layers):
            _ = prune_ffn_channels(blk.mlp, config.prune_ffn_ratio)
            global_ffn_layers.append(idx)

    skipped = skip_layers(model.aggregator.global_blocks, config.global_skip_layers)

    pruned["global_heads_layers"] = global_head_layers
    pruned["global_ffn_layers"] = global_ffn_layers
    pruned["global_skip_layers"] = skipped
    return pruned


def short_recovery_train(
    student: nn.Module,
    teacher: nn.Module,
    steps: int = 20,
    lr: float = 1e-5,
    image_size: int = 518,
    seq_len: int = 2,
    batch_size: int = 1,
) -> None:
    student.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    for _ in range(steps):
        images = torch.rand(batch_size, seq_len, 3, image_size, image_size)
        with torch.no_grad():
            teacher_out = teacher(images)
        student_out = student(images)

        loss = 0.0
        for key in ("pose_enc", "depth", "world_points"):
            if key in teacher_out and key in student_out:
                loss = loss + torch.nn.functional.mse_loss(student_out[key], teacher_out[key])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


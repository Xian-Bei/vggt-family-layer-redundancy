"""Shared LoRA folding used by deployment-time compression adapters."""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.recovery.train.model import LoRALinear


def _replace_child(parent: nn.Module, name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def _merged_lora_linear(module: LoRALinear) -> nn.Linear:
    base = module.base
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    delta = torch.matmul(module.lora_b.weight.float(), module.lora_a.weight.float())
    merged.weight.data.copy_(base.weight.data + delta.to(base.weight.dtype) * float(module.scale))
    if base.bias is not None:
        merged.bias.data.copy_(base.bias.data)
    merged.weight.requires_grad_(False)
    if merged.bias is not None:
        merged.bias.requires_grad_(False)
    return merged


def merge_lora_linears(module: nn.Module) -> int:
    """Fold every LaRe LoRA projection in place and return the replacement count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            _replace_child(module, name, _merged_lora_linear(child))
            count += 1
        else:
            count += merge_lora_linears(child)
    return count

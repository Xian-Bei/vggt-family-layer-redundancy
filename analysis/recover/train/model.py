from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import math

import torch
import torch.nn as nn

from analysis.backbones.registry import get_backbone


class SkipIdentityBlock(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear modules."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False).to(
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False).to(
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        # Keep pretrained projection frozen; only LoRA branches are trainable.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))
        return base_out + lora_out * self.scale

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def _replace_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList):
        parent[int(child_name)] = new_module
        return
    setattr(parent, child_name, new_module)


def _iter_named_linears_for_lora(block: nn.Module) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """Return candidate attention projections to receive LoRA adapters."""
    candidates: List[Tuple[nn.Module, str, nn.Linear]] = []
    include_names = {"qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"}
    for parent in block.modules():
        for child_name, child in parent.named_children():
            if child_name not in include_names:
                continue
            if isinstance(child, LoRALinear):
                # Already wrapped.
                continue
            if isinstance(child, nn.Linear):
                candidates.append((parent, child_name, child))
    return candidates


def inject_lora_attn(
    block: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[nn.Parameter]:
    # Generic adapter: wrap common attention projections with LoRA linear layers.
    params: List[nn.Parameter] = []
    for parent, child_name, linear in _iter_named_linears_for_lora(block):
        wrapped = LoRALinear(linear, rank=int(rank), alpha=float(alpha), dropout=float(dropout))
        _replace_child_module(parent, child_name, wrapped)
        params.extend(list(wrapped.lora_a.parameters()))
        params.extend(list(wrapped.lora_b.parameters()))
    return params


def build_skipped_layer_indices(intervals: Sequence[Tuple[int, int]]) -> List[int]:
    out: set[int] = set()
    for l, r in intervals:
        for idx in range(int(l), int(r) + 1):
            out.add(idx)
    return sorted(out)


def count_unique_params(modules: Iterable[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for m in modules:
        for p in m.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += int(p.numel())
    return total


def build_model(
    *,
    model_name: str,
    backend: str,
    hf_model: str,
    device: str,
    unit_plan=None,
    ckpt_path: Path | None = None,
    train_dir: Path | None = None,
    epoch: int | None = None,
    eval_mode: bool = True,
):
    del model_name, unit_plan, ckpt_path, train_dir, epoch
    adapter = get_backbone(backend)
    return adapter.load_pretrained(hf_model=hf_model, device=device, eval_mode=eval_mode)

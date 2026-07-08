"""FastVGGT token merging applied to an already-restored LaRe VGGT model."""

from __future__ import annotations

import importlib.util
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.inference_compression.contracts import (
    InferenceCompressionAdapter,
    InferenceCompressionConfig,
)
from analysis.inference_compression.lora import merge_lora_linears


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _token_merge_function() -> Callable[..., Tuple[Callable, Callable]]:
    source = _repo_root() / "submodules" / "FastVGGT" / "merging" / "merge.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"FastVGGT submodule is missing {source}. Run: git submodule update --init submodules/FastVGGT"
        )
    spec = importlib.util.spec_from_file_location("lare_fastvggt_merge", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FastVGGT token merge implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.token_merge_bipartite2d


def _fast_attention_forward(attn: nn.Module, x: torch.Tensor, pos=None) -> torch.Tensor:
    batch, num_tokens, channels = x.shape
    qkv = (
        attn.qkv(x)
        .reshape(batch, num_tokens, 3, attn.num_heads, attn.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None:
        q = attn.rope(q, pos)
        k = attn.rope(k, pos)

    merge, unmerge = _token_merge_function()(
        x,
        int(attn._fastvggt_patch_width),
        int(attn._fastvggt_patch_height),
        2,
        2,
        int(num_tokens * float(attn._fastvggt_merge_ratio)),
        False,
        torch.Generator(device=x.device).manual_seed(33),
        enable_protection=True,
    )
    q_shape = q.shape
    q_in = q.permute(0, 2, 1, 3).reshape(batch, num_tokens, channels)
    k_in = k.permute(0, 2, 1, 3).reshape(batch, num_tokens, channels)
    v_in = v.permute(0, 2, 1, 3).reshape(batch, num_tokens, channels)
    q_out, k_out, v_out = merge(
        q_in,
        mode="mean",
        extra_tensors=k_in,
        extra_tensors_2=v_in,
    )
    merged_tokens = int(q_out.shape[1])
    q = q_out.reshape(batch, merged_tokens, q_shape[1], q_shape[3]).permute(0, 2, 1, 3)
    k = k_out.reshape(batch, merged_tokens, q_shape[1], q_shape[3]).permute(0, 2, 1, 3)
    v = v_out.reshape(batch, merged_tokens, q_shape[1], q_shape[3]).permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=attn.attn_drop.p if attn.training else 0.0,
    )
    out = out.transpose(1, 2).reshape(batch, merged_tokens, channels)
    out = attn.proj(out)
    out = attn.proj_drop(out)
    return unmerge(out)


class FastVGGTInferenceAdapter(InferenceCompressionAdapter):
    name = "fastvggt"
    supported_backbones = frozenset({"vggt"})

    def validate_config(self, config: InferenceCompressionConfig) -> None:
        start = int(config.fastvggt_merge_start)
        ratio = float(config.fastvggt_merge_ratio)
        if start < 0:
            raise ValueError(f"FASTVGGT_MERGE_START must be >= 0, got {start}")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"FASTVGGT_MERGE_RATIO must be in [0, 1], got {ratio}")

    def apply(
        self,
        model: nn.Module,
        *,
        backbone_name: str,
        config: InferenceCompressionConfig,
    ) -> nn.Module:
        self.validate_backbone(backbone_name)
        self.validate_config(config)
        start = int(config.fastvggt_merge_start)
        ratio = float(config.fastvggt_merge_ratio)

        aggregator = getattr(model, "aggregator", None)
        global_blocks = getattr(aggregator, "global_blocks", None)
        if aggregator is None or global_blocks is None:
            raise TypeError("FastVGGT requires a VGGT model with aggregator.global_blocks")
        if start >= len(global_blocks):
            raise ValueError(
                f"FASTVGGT_MERGE_START={start} is outside the {len(global_blocks)} global blocks"
            )

        model.eval()
        merged_count = merge_lora_linears(model)
        patched = []
        for index, block in enumerate(global_blocks):
            if index < start or not hasattr(block, "attn"):
                continue
            attn = block.attn
            required = ("qkv", "q_norm", "k_norm", "num_heads", "head_dim", "proj", "proj_drop")
            missing = [name for name in required if not hasattr(attn, name)]
            if missing:
                raise TypeError(f"VGGT global block {index} attention is missing {missing}")
            attn._fastvggt_merge_ratio = ratio
            attn.forward = types.MethodType(_fast_attention_forward, attn)
            patched.append(attn)

        patch_size = int(getattr(aggregator, "patch_size", 14))

        def _sync_patch_grid(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise ValueError("FastVGGT aggregator pre-hook expected images as its first argument")
            images = args[0]
            patch_height = int(images.shape[-2]) // patch_size
            patch_width = int(images.shape[-1]) // patch_size
            for attn in patched:
                attn._fastvggt_patch_height = patch_height
                attn._fastvggt_patch_width = patch_width

        aggregator.register_forward_pre_hook(_sync_patch_grid)
        record = config.as_record()
        record["merged_lora_linears"] = merged_count
        model._inference_compression = record
        return model

    def output_tag(self, config: InferenceCompressionConfig) -> str:
        ratio = f"{float(config.fastvggt_merge_ratio):g}".replace(".", "p")
        return f"fastvggt_m{int(config.fastvggt_merge_start)}_r{ratio}"

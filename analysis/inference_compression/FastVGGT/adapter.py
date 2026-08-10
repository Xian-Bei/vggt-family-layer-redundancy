"""FastVGGT token merging applied to restored VGGT, Pi3, and VGGT-Omega models."""

from __future__ import annotations

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
    # Upstream FastVGGT assumes exactly five special tokens per image. Pi3 also
    # uses five, while VGGT-Omega uses camera + 16 register tokens. Load the
    # vendored implementation with that constant promoted to an argument; this
    # keeps the algorithm identical without modifying the git submodule.
    code = source.read_text(encoding="utf-8")
    replacements = {
        "    enable_protection: bool = False,\n) -> Tuple[Callable, Callable]:":
            "    enable_protection: bool = False,\n    num_special_tokens: int = 5,\n) -> Tuple[Callable, Callable]:",
        "tokens_per_img = w * h + 5": "tokens_per_img = w * h + num_special_tokens",
        '"Token count doesn\'t match (w*h+5)*num_imgs"':
            '"Token count does not match the patch grid and special-token count"',
        "torch.arange(5, device=metric.device)":
            "torch.arange(num_special_tokens, device=metric.device)",
        "* tokens_per_img + 5": "* tokens_per_img + num_special_tokens",
    }
    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"FastVGGT merge source changed; missing expected fragment: {old!r}")
        code = code.replace(old, new)
    module = types.ModuleType("lare_fastvggt_merge")
    module.__file__ = str(source)
    exec(compile(code, str(source), "exec"), module.__dict__)
    return module.token_merge_bipartite2d


def _fast_attention_forward(
    attn: nn.Module,
    x: torch.Tensor,
    pos=None,
    attn_bias=None,
    xpos=None,
    rope=None,
    **_kwargs: Any,
) -> torch.Tensor:
    if attn_bias is not None:
        raise ValueError("FastVGGT token merging does not support an attention bias")
    batch, num_tokens, channels = x.shape
    head_dim = channels // int(attn.num_heads)
    qkv = (
        attn.qkv(x)
        .reshape(batch, num_tokens, 3, attn.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q).to(v.dtype)
    if k_norm is not None:
        k = k_norm(k).to(v.dtype)
    # VGGT and Pi3 own their RoPE module and receive positions as pos/xpos.
    # Omega instead receives a precomputed (sin, cos) tuple from its block.
    external_rope = rope
    if external_rope is not None and hasattr(attn, "apply_rope"):
        q, k = attn.apply_rope(q, k, external_rope)
    else:
        internal_rope = getattr(attn, "rope", None)
        position = pos if pos is not None else xpos
        if internal_rope is not None:
            q = internal_rope(q, position)
            k = internal_rope(k, position)

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
        num_special_tokens=int(attn._fastvggt_num_special_tokens),
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
    supported_backbones = frozenset({"vggt", "pi3", "vggt_omega"})

    @staticmethod
    def _global_blocks(
        model: nn.Module, backbone_name: str
    ) -> tuple[list[nn.Module], nn.Module, int, int]:
        """Return global blocks, image-input owner, patch size, and special-token count."""
        if backbone_name == "pi3":
            decoder = getattr(model, "decoder", None)
            if decoder is None:
                raise TypeError("FastVGGT requires Pi3 model.decoder")
            return (
                list(decoder[1::2]), model, int(getattr(model, "patch_size", 14)),
                int(getattr(model, "patch_start_idx", 5)),
            )

        aggregator = getattr(model, "aggregator", None)
        if aggregator is None:
            raise TypeError(f"FastVGGT requires {backbone_name}.aggregator")
        if backbone_name == "vggt_omega":
            all_blocks = getattr(aggregator, "inter_frame_blocks", None)
            attention_types = getattr(aggregator, "inter_frame_attention_types", None)
            if all_blocks is None or not isinstance(attention_types, (list, tuple)):
                raise TypeError("FastVGGT requires Omega inter-frame blocks and attention types")
            # Register-only Omega blocks never receive patch tokens and therefore
            # must retain their ordinary attention implementation.
            blocks = [
                block for index, block in enumerate(all_blocks)
                if index < len(attention_types) and str(attention_types[index]) == "global"
            ]
        else:
            blocks = list(getattr(aggregator, "global_blocks", []))
        special_tokens = int(
            getattr(aggregator, "patch_token_start", getattr(aggregator, "patch_start_idx", 5))
        )
        return blocks, aggregator, int(getattr(aggregator, "patch_size", 14)), special_tokens

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

        global_blocks, image_input_owner, patch_size, num_special_tokens = self._global_blocks(
            model, backbone_name
        )
        if not global_blocks:
            raise TypeError(f"FastVGGT found no patch-aware global blocks for {backbone_name}")
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
            required = ("qkv", "num_heads", "proj", "proj_drop")
            missing = [name for name in required if not hasattr(attn, name)]
            if missing:
                raise TypeError(
                    f"{backbone_name} global block {index} attention is missing {missing}"
                )
            attn._fastvggt_merge_ratio = ratio
            attn._fastvggt_num_special_tokens = num_special_tokens
            attn.forward = types.MethodType(_fast_attention_forward, attn)
            patched.append(attn)

        def _sync_patch_grid(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise ValueError("FastVGGT aggregator pre-hook expected images as its first argument")
            images = args[0]
            patch_height = int(images.shape[-2]) // patch_size
            patch_width = int(images.shape[-1]) // patch_size
            for attn in patched:
                attn._fastvggt_patch_height = patch_height
                attn._fastvggt_patch_width = patch_width

        image_input_owner.register_forward_pre_hook(_sync_patch_grid)
        record = config.as_record()
        record["merged_lora_linears"] = merged_count
        model._inference_compression = record
        return model

    def output_tag(self, config: InferenceCompressionConfig) -> str:
        ratio = f"{float(config.fastvggt_merge_ratio):g}".replace(".", "p")
        return f"fastvggt_m{int(config.fastvggt_merge_start)}_r{ratio}"

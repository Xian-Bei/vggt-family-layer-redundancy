#!/usr/bin/env python3
"""
Visualize how VGGT frame/global block attention and MLP parameter counts and
forward MACs change with the number of input images.

Assumptions:
- Uses the current default VGGT / Aggregator hyperparameters from this repo.
- Counts one forward pass with batch size 1.
- Reports MACs (multiply-accumulate operations), not FLOPs. A rough FLOP count
  is about 2x MACs for the matmul-heavy terms used here.
- Includes attention module params (`qkv`, `q_norm`, `k_norm`, `proj`) and MLP
  params (`fc1`, `fc2`), but excludes surrounding block LayerNorm / LayerScale
  because the request focuses on attention vs MLP.
- MAC estimates cover the dominant linear / attention matmul terms and ignore
  lower-order costs such as LayerNorm, GELU, dropout, and RoPE application.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CurrentVGGTConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    qk_norm: bool = True
    init_values: float = 0.01

    @property
    def patch_tokens_per_image(self) -> int:
        grid = self.img_size // self.patch_size
        if grid * self.patch_size != self.img_size:
            raise ValueError(
                f"img_size={self.img_size} is not divisible by patch_size={self.patch_size}; "
                "the current script assumes a square integer patch grid."
            )
        return grid * grid

    @property
    def special_tokens_per_image(self) -> int:
        # 1 camera token + num_register_tokens
        return 1 + self.num_register_tokens

    @property
    def tokens_per_image(self) -> int:
        return self.patch_tokens_per_image + self.special_tokens_per_image


def attention_params(cfg: CurrentVGGTConfig) -> int:
    c = cfg.embed_dim
    head_dim = c // cfg.num_heads
    total = 0
    # qkv + proj weights
    total += 3 * c * c
    total += c * c
    # qkv + proj biases
    if cfg.qkv_bias:
        total += 3 * c
    if cfg.proj_bias:
        total += c
    # q_norm / k_norm are LayerNorm(head_dim) with weight + bias
    if cfg.qk_norm:
        total += 4 * head_dim
    return total


def mlp_params(cfg: CurrentVGGTConfig) -> int:
    c = cfg.embed_dim
    hidden_dim = int(c * cfg.mlp_ratio)
    total = 2 * c * hidden_dim
    if cfg.ffn_bias:
        total += hidden_dim + c
    return total


def attention_macs(num_tokens: int, embed_dim: int) -> int:
    """
    Dominant attention MACs for one module forward:
    - qkv projection: 3 * N * C * C
    - output projection: N * C * C
    - qk^T and attn @ v: 2 * N * N * C
    """
    return 4 * num_tokens * embed_dim * embed_dim + 2 * num_tokens * num_tokens * embed_dim


def mlp_macs(num_tokens: int, embed_dim: int, mlp_ratio: float) -> int:
    hidden_dim = int(embed_dim * mlp_ratio)
    return 2 * num_tokens * embed_dim * hidden_dim


def _build_rows(cfg: CurrentVGGTConfig, image_counts: Iterable[int]) -> List[Dict[str, float]]:
    attn_param_count = attention_params(cfg)
    mlp_param_count = mlp_params(cfg)

    p = cfg.tokens_per_image
    rows: List[Dict[str, float]] = []
    for num_images in image_counts:
        if num_images <= 0:
            raise ValueError(f"num_images must be positive, got {num_images}")

        frame_tokens = p
        global_tokens = num_images * p

        frame_attn_macs = num_images * attention_macs(frame_tokens, cfg.embed_dim)
        global_attn_macs = attention_macs(global_tokens, cfg.embed_dim)
        # Frame/global MLPs both process the same total token count B*S*P for one input.
        frame_mlp_macs = num_images * mlp_macs(frame_tokens, cfg.embed_dim, cfg.mlp_ratio)
        global_mlp_macs = mlp_macs(global_tokens, cfg.embed_dim, cfg.mlp_ratio)

        rows.append(
            {
                "num_images": num_images,
                "patch_tokens_per_image": cfg.patch_tokens_per_image,
                "special_tokens_per_image": cfg.special_tokens_per_image,
                "tokens_per_image": cfg.tokens_per_image,
                "frame_attention_params_per_block": attn_param_count,
                "frame_mlp_params_per_block": mlp_param_count,
                "global_attention_params_per_block": attn_param_count,
                "global_mlp_params_per_block": mlp_param_count,
                "frame_attention_params_all_blocks": attn_param_count * cfg.depth,
                "frame_mlp_params_all_blocks": mlp_param_count * cfg.depth,
                "global_attention_params_all_blocks": attn_param_count * cfg.depth,
                "global_mlp_params_all_blocks": mlp_param_count * cfg.depth,
                "frame_attention_macs_per_block": frame_attn_macs,
                "frame_mlp_macs_per_block": frame_mlp_macs,
                "global_attention_macs_per_block": global_attn_macs,
                "global_mlp_macs_per_block": global_mlp_macs,
                "frame_attention_macs_all_blocks": frame_attn_macs * cfg.depth,
                "frame_mlp_macs_all_blocks": frame_mlp_macs * cfg.depth,
                "global_attention_macs_all_blocks": global_attn_macs * cfg.depth,
                "global_mlp_macs_all_blocks": global_mlp_macs * cfg.depth,
            }
        )
    return rows


def _to_billions(values: Iterable[float]) -> List[float]:
    return [float(v) / 1e9 for v in values]


def _plot(rows: List[Dict[str, float]], cfg: CurrentVGGTConfig, out_path: Path) -> None:
    x = [int(r["num_images"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle(
        "VGGT all-block-stack scaling under current defaults\n"
        f"img={cfg.img_size}, patch={cfg.patch_size}, embed_dim={cfg.embed_dim}, "
        f"depth={cfg.depth}, heads={cfg.num_heads}, mlp_ratio={cfg.mlp_ratio}",
        fontsize=13,
    )

    stack_param_specs = [
        ("frame_attention_params_all_blocks", "24x frame attention"),
        ("frame_mlp_params_all_blocks", "24x frame MLP"),
        ("global_attention_params_all_blocks", "24x global attention"),
        ("global_mlp_params_all_blocks", "24x global MLP"),
    ]
    ax = axes[0]
    for key, label in stack_param_specs:
        ax.plot(x, _to_billions(r[key] for r in rows), marker="o", linewidth=2, label=label)
    ax.set_title("All-block-stack parameters vs number of images")
    ax.set_xlabel("Number of input images")
    ax.set_ylabel("Parameters (billions)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    stack_compute_specs = [
        ("frame_attention_macs_all_blocks", "24x frame attention"),
        ("frame_mlp_macs_all_blocks", "24x frame MLP"),
        ("global_attention_macs_all_blocks", "24x global attention"),
        ("global_mlp_macs_all_blocks", "24x global MLP"),
    ]
    ax = axes[1]
    for key, label in stack_compute_specs:
        ax.plot(x, _to_billions(r[key] for r in rows), marker="o", linewidth=2, label=label)
    ax.set_title("All-block-stack forward compute vs number of images")
    ax.set_xlabel("Number of input images")
    ax.set_ylabel("MACs per input (GMACs)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-images", type=int, default=1, help="Minimum number of input images to plot.")
    p.add_argument("--max-images", type=int, default=1000, help="Maximum number of input images to plot.")
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("scripts/vggt_block_scaling"),
        help="Output prefix without extension; writes <prefix>.csv and <prefix>.png.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.min_images <= 0 or args.max_images < args.min_images:
        raise SystemExit(
            f"Invalid image range: min={args.min_images}, max={args.max_images}. "
            "Require 1 <= min <= max."
        )

    cfg = CurrentVGGTConfig()
    image_counts = range(args.min_images, args.max_images + 1)
    rows = _build_rows(cfg, image_counts)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    _write_csv(rows, csv_path)
    _plot(rows, cfg, png_path)

    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote figure to {png_path}")


if __name__ == "__main__":
    main()

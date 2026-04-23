"""
Stage 0: VGGT checkpoint-only weight statistics (no image data, no activation cache).

Writes ``weight_report.json`` and bar charts under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from vggt.models.vggt import VGGT

from .collect import compute_global_weight_stats


def _bar(values: list[float], labels: list[int], out_path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(values)), values, tick_label=[str(x) for x in labels])
    ax.set_title(title)
    ax.set_xlabel("global block index")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze VGGT global-block weights only (no dataset).")
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B", help="HuggingFace model id.")
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/scannet_weight_analysis"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.hf_model.strip():
        raise SystemExit("Stage 0 requires a non-empty --hf-model.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VGGT.from_pretrained(args.hf_model).cpu().eval()
    weight_stats = compute_global_weight_stats(model)

    report: Dict[str, Any] = {
        "hf_model": args.hf_model,
        "weight_stats": weight_stats,
    }
    (out_dir / "weight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ga = weight_stats["attention_stats"]["global"]
    sp = weight_stats["spectral_stats"]
    head_cos = [x["mean_head_cosine"] for x in ga]
    _bar(head_cos, list(range(len(head_cos))), out_dir / "bar_global_mean_head_cosine.png", "Global block Q-head cosine", "mean pairwise cos")

    stable_attn = [s["stable_rank"] for s in sp if s["module"] == "global_attn_qkv"]
    layers_attn = [s["layer"] for s in sp if s["module"] == "global_attn_qkv"]
    _bar(stable_attn, layers_attn, out_dir / "bar_global_attn_stable_rank.png", "Global attn qkv stable rank", "stable_rank")

    print(f"Wrote weight analysis to {out_dir}")


if __name__ == "__main__":
    main()

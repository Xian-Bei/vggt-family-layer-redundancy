from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


LAYER_RE = re.compile(r"baseline1_skip_l(\d+)$")


def _collect(baseline_dir: Path) -> List[Tuple[int, Dict[str, float]]]:
    rows: List[Tuple[int, Dict[str, float]]] = []
    for d in sorted(p for p in baseline_dir.iterdir() if p.is_dir()):
        m = LAYER_RE.match(d.name)
        if not m:
            continue
        layer = int(m.group(1))
        fp = d / "compare_vs_stage1.json"
        if not fp.exists():
            continue
        payload = json.loads(fp.read_text(encoding="utf-8"))
        delta = payload.get("aggregate_delta_gt_vs_stage1", {})
        rows.append((layer, delta))
    rows.sort(key=lambda x: x[0])
    return rows


def _plot_delta(rows: List[Tuple[int, Dict[str, float]]], out: Path) -> None:
    layers = [r[0] for r in rows]
    metrics = [
        ("delta_ate_vs_stage1", "ATE"),
        ("delta_are_vs_stage1", "ARE"),
        ("delta_rpe_rot_vs_stage1", "RPE Rot"),
        ("delta_rpe_trans_vs_stage1", "RPE Trans"),
        ("delta_chamfer_distance_vs_stage1", "Chamfer"),
    ]
    plt.figure(figsize=(12, 6))
    for key, label in metrics:
        ys = [r[1].get(key, float("nan")) for r in rows]
        plt.plot(layers, ys, marker="o", linewidth=1.6, markersize=3.5, label=label)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Skipped global layer index")
    plt.ylabel("Delta vs Stage1 (error metric, + means worse)")
    plt.title("Stage3 baseline1_skip: GT metric delta vs skipped layer")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot baseline1_skip layer-wise GT metric deltas from compare_vs_stage1.json.")
    p.add_argument("--baseline-dir", type=Path, default=Path("analysis_outputs/scannet_baseline"))
    p.add_argument("--out-prefix", type=str, default="baseline1_skip")
    args = p.parse_args()

    rows = _collect(args.baseline_dir)
    if not rows:
        raise SystemExit(f"No baseline1_skip_lXX compare results found under {args.baseline_dir}")

    out1 = args.baseline_dir / f"{args.out_prefix}_delta_metrics_vs_layer.png"
    _plot_delta(rows, out1)
    print(f"Wrote: {out1}")


if __name__ == "__main__":
    main()


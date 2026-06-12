"""Visualize CKA heatmap with greedy dense-triangle corner markers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.redundancy.metrics import upper_triangle_pack, upper_triangle_unpack_for_heatmap
from analysis.redundancy.triangle_prune import TrianglePick, greedy_dense_triangles


def _sim_to_heatmap_grid(sim: np.ndarray, fill_value: float = float("nan")) -> np.ndarray:
    n = int(sim.shape[0])
    packed = upper_triangle_pack(torch.from_numpy(sim.astype(np.float64)))
    grid = upper_triangle_unpack_for_heatmap(packed, n, fill_value)
    return np.asarray(grid, dtype=np.float64)


def _apply_similarity_cutoff_matrix(sim: np.ndarray, cutoff: float) -> np.ndarray:
    """Heatmap display only: match Heatmaps UI ``_similarity_packed_cutoff_01`` (values < cutoff -> 0)."""
    cutoff = float(np.clip(cutoff, 0.0, 1.0))
    if cutoff <= 0.0:
        return np.asarray(sim, dtype=np.float64)
    out = np.array(sim, dtype=np.float64, copy=True)
    mask = np.isfinite(out) & (out < cutoff)
    out[mask] = 0.0
    return out


def render_triangle_prune_heatmap(
    sim: np.ndarray,
    unit_ids: Sequence[str],
    picks: Sequence[TrianglePick],
    *,
    out_dir: Path,
    threshold: float,
    tag: str = "triangle_prune",
    title: Optional[str] = None,
    lora_unit_indices: Optional[Sequence[int]] = None,
    cache_unit_indices: Optional[Sequence[int]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(sim.shape[0])
    arr = _sim_to_heatmap_grid(_apply_similarity_cutoff_matrix(sim, threshold))
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title or f"CKA + dense triangles (tau={threshold}, n={len(picks)})")
    ax.set_xlabel("unit index j")
    ax.set_ylabel("unit index i")
    ticks = list(range(0, n, max(1, n // 12)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k, pick in enumerate(picks):
        ax.plot(
            pick.j,
            pick.i,
            "o",
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.5,
            markersize=12,
            label=f"T{k} ({pick.i},{pick.j})" if k < 6 else None,
        )
    drop_idxs = sorted({idx for p in picks for idx in range(int(p.i) + 1, int(p.j) + 1) if idx >= 0})
    for k, idx in enumerate(drop_idxs):
        ax.plot(
            idx,
            idx,
            "_",
            color="#ff6b6b",
            markersize=13,
            markeredgewidth=2.0,
            label="Dropped unit" if k == 0 else None,
        )
    cache_idxs = sorted({int(x) for x in (cache_unit_indices or []) if int(x) >= 0})
    for k, idx in enumerate(cache_idxs):
        ax.plot(
            idx,
            idx,
            "s",
            markerfacecolor="none",
            markeredgecolor="#00e5ff",
            markeredgewidth=1.2,
            markersize=9,
            label="Cache unit" if k == 0 else None,
        )
    lora_idxs = sorted({int(x) for x in (lora_unit_indices or []) if int(x) >= 0})
    for k, idx in enumerate(lora_idxs):
        ax.plot(
            idx,
            idx,
            "+",
            color="#ffa500",
            markeredgewidth=1.8,
            markersize=10,
            label="LoRA unit" if k == 0 else None,
        )
    if picks or lora_idxs or drop_idxs or cache_idxs:
        ax.legend(loc="upper right", fontsize=9)

    safe_tag = tag.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{safe_tag}_triangles.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _synthetic_unit_ids(depth: int = 24) -> List[str]:
    ids: List[str] = []
    for i in range(depth):
        ids.append(f"frame_{i}")
        ids.append(f"global_{i}")
    return ids


def _synthetic_sim(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0.5, 1.0, size=(n, n)).astype(np.float64)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    # Dense block in upper triangle around units 10..20
    for i in range(8, 18):
        for j in range(i + 1, 22):
            sim[i, j] = sim[j, i] = rng.uniform(0.92, 0.99)
    return sim


def _find_npz_under(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob("redundancy_graph_v3.npz"))
    return candidates[0] if candidates else None


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Visualize triangle greedy picks on CKA heatmap.")
    p.add_argument(
        "--redundancy-root",
        type=Path,
        default=Path("analysis_outputs_v3/first"),
        help="Search for redundancy_graph_v3.npz under this tree.",
    )
    p.add_argument("--npz", type=Path, default=None, help="Explicit redundancy npz path.")
    p.add_argument("--out-dir", type=Path, default=Path("tmp/triangle_prune_viz"))
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--max-triangles", type=int, default=2)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic 48x48 CKA if no npz.")
    args = p.parse_args(list(argv) if argv is not None else None)

    npz_path = args.npz
    if npz_path is None:
        npz_path = _find_npz_under(args.redundancy_root)

    if npz_path is not None and npz_path.is_file():
        z = np.load(npz_path, allow_pickle=True)
        unit_ids = [str(u) for u in list(z["unit_ids"])]
        sim = np.asarray(z["cka_mean"], dtype=np.float64)
        tag = npz_path.parent.parent.name + "_" + npz_path.parent.name
    elif args.synthetic or npz_path is None:
        unit_ids = _synthetic_unit_ids(24)
        sim = _synthetic_sim(len(unit_ids))
        tag = "synthetic_vggt48"
        print("No redundancy_graph_v3.npz found; using synthetic CKA matrix.", flush=True)
    else:
        raise FileNotFoundError(f"Missing npz: {npz_path}")

    picks, _ = greedy_dense_triangles(
        sim, unit_ids, threshold=float(args.threshold), max_triangles=int(args.max_triangles)
    )
    out = render_triangle_prune_heatmap(
        sim,
        unit_ids,
        picks,
        out_dir=args.out_dir,
        threshold=float(args.threshold),
        tag=tag,
    )
    print(f"Wrote {out} ({len(picks)} triangles)", flush=True)
    for k, pick in enumerate(picks):
        print(f"  [{k}] (i,j)=({pick.i},{pick.j}) span={pick.span} anchor={pick.unit_i}", flush=True)


if __name__ == "__main__":
    main()

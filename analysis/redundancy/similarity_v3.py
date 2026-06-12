"""Shared v3 redundancy similarity helpers (in-memory unit activations)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from analysis.redundancy.metrics import pairwise_similarity_matrices
from analysis.schema.v3 import SCHEMA_VERSION, stage_dir


def parse_token_regions(spec: str) -> List[str]:
    """Parse ``special,patch,all``. Empty spec defaults to ``all`` (backward compatible)."""
    allowed = {"special", "patch", "all"}
    regions = [r.strip().lower() for r in spec.split(",") if r.strip()]
    if not regions:
        return ["all"]
    bad = [r for r in regions if r not in allowed]
    if bad:
        raise ValueError(f"Unknown token region(s): {bad}; allowed: {sorted(allowed)}")
    out: List[str] = []
    for r in regions:
        if r not in out:
            out.append(r)
    return out


def flatten_unit_token_regions(
    t: torch.Tensor,
    *,
    patch_start_idx: int,
    regions: Sequence[str],
) -> List[Tuple[str, torch.Tensor]]:
    """Split a unit activation into flat ``[N,C]`` tensors per requested region."""
    want = set(regions)
    if t.dim() == 2:
        if "all" in want:
            return [("all", t.float())]
        return []
    if t.dim() == 3:
        flat = t.unsqueeze(0).reshape(-1, t.shape[-1]).float()
        if "all" in want:
            return [("all", flat)]
        return []
    if t.dim() != 4:
        raise ValueError(f"Unexpected activation shape {tuple(t.shape)} for redundancy")

    t = t.float()
    _b, _s, p, c = (int(x) for x in t.shape)
    psi = max(0, min(int(patch_start_idx), p))
    out: List[Tuple[str, torch.Tensor]] = []
    if "special" in want and psi > 0:
        out.append(("special", t[:, :, :psi, :].reshape(-1, c).float()))
    if "patch" in want and psi < p:
        out.append(("patch", t[:, :, psi:, :].reshape(-1, c).float()))
    if "all" in want:
        out.append(("all", t.reshape(-1, c).float()))
    return out


def should_compute_scene_redundancy(
    *,
    skip_redundancy: bool,
    metric_names: Sequence[str],
    regions: Sequence[str],
) -> bool:
    """True when per-scene flatten + pairwise similarity should run."""
    if skip_redundancy:
        return False
    if not metric_names:
        return False
    if not regions:
        return False
    return True


def metric_device(device: str) -> Optional[str]:
    if device == "cpu":
        return None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else None


def subsample_token_rows(t: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    n = int(t.shape[0])
    if max_tokens <= 0 or n <= max_tokens:
        return t
    g = torch.Generator(device=t.device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    idx = torch.randperm(n, generator=g, device=t.device)[:max_tokens]
    return t.index_select(0, idx)


def unit_map_to_flat_layers(
    unit_map: Dict[str, torch.Tensor],
    unit_ids: Sequence[str],
    *,
    patch_start_idx: int,
    regions: Sequence[str],
    max_tokens: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[str]]:
    flat_layers: List[torch.Tensor] = []
    scene_unit_ids: List[str] = []
    for uid in unit_ids:
        if uid not in unit_map:
            continue
        t = unit_map[uid]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"unit_map[{uid!r}] must be a torch.Tensor")
        for region_name, flat in flatten_unit_token_regions(
            t, patch_start_idx=patch_start_idx, regions=regions
        ):
            flat_layers.append(subsample_token_rows(flat, max_tokens, seed))
            scene_unit_ids.append(
                uid if region_name == "all" and len(regions) == 1 else f"{uid}_{region_name}"
            )
    return flat_layers, scene_unit_ids


def compute_scene_similarity(
    flat_layers: Sequence[torch.Tensor],
    *,
    metric_names: Sequence[str],
    metric_device: Optional[str],
    progress: bool,
    progress_desc: str,
) -> Dict[str, np.ndarray]:
    if len(flat_layers) < 2:
        raise ValueError("Need at least two layer tensors for pairwise similarity.")
    sims = pairwise_similarity_matrices(
        tuple(flat_layers),
        device=metric_device,
        progress=progress,
        progress_desc=progress_desc,
    )
    out: Dict[str, np.ndarray] = {}
    for m in metric_names:
        if m not in sims:
            continue
        mat = sims[m]
        out[m] = mat.numpy() if hasattr(mat, "numpy") else np.asarray(mat)
    return out


class RedundancyAccumulator:
    """Cross-scene redundancy graph builder (mean + per-scene stacks)."""

    def __init__(self, metric_names: Sequence[str], token_regions: Sequence[str]) -> None:
        self.metric_names = list(metric_names)
        self.token_regions = list(token_regions)
        self.accum: Dict[str, List[np.ndarray]] = {m: [] for m in self.metric_names}
        self.scene_stack: Dict[str, List[np.ndarray]] = {m: [] for m in self.metric_names}
        self.scene_ids_processed: List[str] = []
        self.graph_unit_ids: List[str] = []

    def add_scene(
        self,
        scene_id: str,
        flat_layers: Sequence[torch.Tensor],
        scene_unit_ids: Sequence[str],
        sims: Dict[str, np.ndarray],
    ) -> None:
        if len(flat_layers) < 2:
            return
        if not self.graph_unit_ids:
            self.graph_unit_ids = list(scene_unit_ids)
        elif list(scene_unit_ids) != self.graph_unit_ids:
            raise ValueError(
                f"Scene {scene_id!r} unit layout {scene_unit_ids} != prior {self.graph_unit_ids}; "
                "check token_regions and cache consistency across scenes."
            )
        self.scene_ids_processed.append(str(scene_id))
        for m, arr in sims.items():
            if m in self.metric_names:
                self.accum[m].append(arr)
                self.scene_stack[m].append(arr)

    def finalize_npz(
        self,
        *,
        fallback_unit_ids: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        unit_ids = self.graph_unit_ids or list(fallback_unit_ids)
        out: Dict[str, np.ndarray] = {
            "unit_ids": np.array(unit_ids, dtype=object),
            "schema_version": np.array([SCHEMA_VERSION]),
            "token_regions": np.array([",".join(self.token_regions)]),
            "scene_ids": np.array(self.scene_ids_processed, dtype=object),
        }
        for m, mats in self.accum.items():
            if mats:
                out[f"{m}_mean"] = np.mean(np.stack(mats, axis=0), axis=0)
        for m, mats in self.scene_stack.items():
            if mats:
                out[f"{m}_scenes"] = np.stack(mats, axis=0)
        return out


def write_redundancy_outputs(
    *,
    experiment: str,
    dataset: str,
    backbone: str,
    unit_ids: Sequence[str],
    accumulator: RedundancyAccumulator,
    max_span: int,
    max_tokens: int,
    metric_device: Optional[str],
    source_cache: str,
    timing_total_s: float,
) -> Path:
    red_dir = stage_dir(experiment, dataset, "redundancy")
    red_dir.mkdir(parents=True, exist_ok=True)
    out_npz = accumulator.finalize_npz(fallback_unit_ids=unit_ids)
    out_npz_path = red_dir / "redundancy_graph_v3.npz"
    np.savez(out_npz_path, **out_npz)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "backbone": backbone,
        "unit_ids": list(unit_ids),
        "metrics": accumulator.metric_names,
        "num_scenes": len(accumulator.scene_ids_processed),
        "scene_ids": accumulator.scene_ids_processed,
        "max_span": int(max_span),
        "max_tokens": int(max_tokens),
        "token_regions": accumulator.token_regions,
        "metric_device": metric_device or "cpu",
        "timing_seconds": {
            "total": float(timing_total_s),
            "source_cache": source_cache,
        },
    }
    (red_dir / "redundancy_summary_v3.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_npz_path} ({timing_total_s:.1f}s)", file=sys.stderr)
    return out_npz_path

"""Shared v3 redundancy similarity helpers (in-memory unit activations)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from analysis.redundancy.metrics import (
    SUPPORTED_SIMILARITY_METRICS,
    pairwise_similarity_matrices,
)
from analysis.schema.v3 import SCHEMA_VERSION, stage_dir

REDUNDANCY_GRAPH_REGIONS = ("special", "camera", "patch", "all")
LEGACY_REDUNDANCY_GRAPH_V3 = "redundancy_graph_v3.npz"


def redundancy_graph_filename(region: str) -> str:
    return f"redundancy_graph_{region}.npz"


def resolve_redundancy_graph_path(redundancy_dir: Path, region: str) -> Optional[Path]:
    """Per-region redundancy npz; legacy ``redundancy_graph_v3.npz`` counts as ``all``."""
    path = Path(redundancy_dir) / redundancy_graph_filename(region)
    if path.is_file():
        return path
    legacy = Path(redundancy_dir) / LEGACY_REDUNDANCY_GRAPH_V3
    if region == "all" and legacy.is_file():
        return legacy
    return None


def list_redundancy_graph_regions(redundancy_dir: Path) -> List[str]:
    red = Path(redundancy_dir)
    if not red.is_dir():
        return []
    out: List[str] = []
    for region in REDUNDANCY_GRAPH_REGIONS:
        if resolve_redundancy_graph_path(red, region) is not None:
            out.append(region)
    return out


def redundancy_dir_has_graphs(redundancy_dir: Path) -> bool:
    return bool(list_redundancy_graph_regions(redundancy_dir))


def parse_token_regions(spec: str) -> List[str]:
    """Parse ``special,camera,patch,all``. Empty spec defaults to ``all`` (backward compatible)."""
    allowed = set(REDUNDANCY_GRAPH_REGIONS)
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
    camera_token_count: int = 0,
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
    cam_n = max(0, min(int(camera_token_count), psi))
    out: List[Tuple[str, torch.Tensor]] = []
    if "camera" in want and cam_n > 0:
        out.append(("camera", t[:, :, :cam_n, :].reshape(-1, c).float()))
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


def unit_map_to_flat_layers_for_region(
    unit_map: Dict[str, torch.Tensor],
    unit_ids: Sequence[str],
    *,
    region: str,
    patch_start_idx: int,
    camera_token_count: int = 0,
    max_tokens: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Flatten one token region; ``scene_unit_ids`` are bare catalog ids (no region suffix)."""
    flat_layers: List[torch.Tensor] = []
    scene_unit_ids: List[str] = []
    for uid in unit_ids:
        if uid not in unit_map:
            continue
        t = unit_map[uid]
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"unit_map[{uid!r}] must be a torch.Tensor")
        for region_name, flat in flatten_unit_token_regions(
            t,
            patch_start_idx=patch_start_idx,
            regions=[region],
            camera_token_count=camera_token_count,
        ):
            if region_name != region:
                continue
            flat_layers.append(subsample_token_rows(flat, max_tokens, seed))
            scene_unit_ids.append(str(uid))
    return flat_layers, scene_unit_ids


def unit_map_to_flat_layers_by_region(
    unit_map: Dict[str, torch.Tensor],
    unit_ids: Sequence[str],
    *,
    patch_start_idx: int,
    camera_token_count: int = 0,
    regions: Sequence[str],
    max_tokens: int,
    seed: int,
) -> Dict[str, Tuple[List[torch.Tensor], List[str]]]:
    out: Dict[str, Tuple[List[torch.Tensor], List[str]]] = {}
    for region in regions:
        flat_layers, scene_unit_ids = unit_map_to_flat_layers_for_region(
            unit_map,
            unit_ids,
            region=str(region),
            patch_start_idx=patch_start_idx,
            camera_token_count=camera_token_count,
            max_tokens=max_tokens,
            seed=seed,
        )
        if flat_layers:
            out[str(region)] = (flat_layers, scene_unit_ids)
    return out


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
        metric_names=metric_names,
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


def validate_metric_names(metric_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in metric_names:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_SIMILARITY_METRICS:
            raise ValueError(
                f"Unsupported similarity metric {name!r}; "
                f"allowed: {sorted(SUPPORTED_SIMILARITY_METRICS)}"
            )
        if name not in out:
            out.append(name)
    return out


class RedundancyAccumulator:
    """Cross-scene redundancy graph builder for one token region (mean + per-scene stacks)."""

    def __init__(self, metric_names: Sequence[str], token_region: str) -> None:
        self.metric_names = list(metric_names)
        self.token_region = str(token_region)
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
                f"check token region {self.token_region!r} and cache consistency across scenes."
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
            "token_region": np.array([self.token_region]),
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
    unit_ids: Sequence[str],
    accumulators: Dict[str, RedundancyAccumulator],
    token_regions: Sequence[str],
    timing_total_s: float,
) -> List[Path]:
    red_dir = stage_dir(experiment, dataset, "redundancy")
    red_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for region in token_regions:
        acc = accumulators.get(str(region))
        if acc is None or not acc.scene_ids_processed:
            continue
        out_npz = acc.finalize_npz(fallback_unit_ids=unit_ids)
        out_npz_path = red_dir / redundancy_graph_filename(str(region))
        merged_npz = _merge_existing_redundancy_npz(out_npz_path, out_npz)
        tmp_path = out_npz_path.with_suffix(out_npz_path.suffix + ".tmp.npz")
        np.savez(tmp_path, **merged_npz)
        tmp_path.replace(out_npz_path)
        written.append(out_npz_path)
        print(f"Wrote {out_npz_path} ({timing_total_s:.1f}s)", file=sys.stderr)

    if not written:
        raise RuntimeError("No redundancy graphs were written (no scenes accumulated).")
    return written


def _object_array_to_str_list(value: Any) -> List[str]:
    arr = np.asarray(value, dtype=object).reshape(-1)
    return [str(item) for item in arr]


def _scalar_object_str(value: Any) -> str:
    arr = np.asarray(value, dtype=object).reshape(-1)
    return str(arr[0]) if arr.size else ""


def _merge_existing_redundancy_npz(
    path: Path,
    new_payload: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return dict(new_payload)

    with np.load(path, allow_pickle=True) as existing_npz:
        existing: Dict[str, np.ndarray] = {key: existing_npz[key] for key in existing_npz.files}

    for key in ("unit_ids", "scene_ids"):
        if key in existing and key in new_payload:
            old_values = _object_array_to_str_list(existing[key])
            new_values = _object_array_to_str_list(new_payload[key])
            if old_values and new_values and old_values != new_values:
                raise ValueError(
                    f"Refusing to merge redundancy metrics into {path}: {key} changed "
                    f"(old n={len(old_values)}, new n={len(new_values)})."
                )

    if "token_region" in existing and "token_region" in new_payload:
        old_region = _scalar_object_str(existing["token_region"])
        new_region = _scalar_object_str(new_payload["token_region"])
        if old_region and new_region and old_region != new_region:
            raise ValueError(
                f"Refusing to merge redundancy metrics into {path}: token_region changed "
                f"from {old_region!r} to {new_region!r}."
            )

    merged = dict(existing)
    merged.update(new_payload)
    return merged

"""Load v3 cache artifacts in the format expected by cache-first training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from analysis.schema.v3 import CacheManifestV3


def is_v3_cache_dir(cache_dir: Path) -> bool:
    return (cache_dir / "cache_manifest_v3.json").is_file()


def load_v3_manifest(cache_dir: Path) -> CacheManifestV3:
    return CacheManifestV3.load(cache_dir / "cache_manifest_v3.json")


def is_prune_plan_minimal_cache_manifest(manifest: CacheManifestV3) -> bool:
    policy = manifest.extra.get("cache_save_policy") or {}
    return isinstance(policy, dict) and policy.get("mode") == "prune_plan_minimal"


def is_prune_plan_minimal_cache_dir(cache_dir: Path) -> bool:
    if not is_v3_cache_dir(cache_dir):
        return False
    return is_prune_plan_minimal_cache_manifest(load_v3_manifest(cache_dir))


def is_prune_plan_minimal_scene_metadata(meta: Dict[str, Any]) -> bool:
    return str(meta.get("cache_save_mode", "")) == "prune_plan_minimal"


def require_minimal_cache_geometry(meta: Dict[str, Any], *, cache_dir: Path) -> None:
    """Raise if metadata from a minimal cache scene is missing fields needed without layer0."""
    if not is_prune_plan_minimal_scene_metadata(meta):
        return
    missing = []
    if meta.get("image_hw") is None:
        missing.append("image_hw")
    nf = meta.get("num_frames")
    if nf is None or int(nf) <= 0:
        missing.append("num_frames")
    if missing:
        raise RuntimeError(
            f"Minimal training cache under {cache_dir} is missing metadata {missing}. "
            "Re-run cache with PRUNE_PLAN_ROOT set, e.g. "
            "PRUNE_PLAN_ROOT=analysis/prune_plan SAVE_SCANNET=1 bash run_cache_redundancy.sh"
        )


def manifest_scenes_for_training(manifest: CacheManifestV3) -> List[Dict[str, Any]]:
    """Convert v3 scene entries to legacy manifest item dicts."""
    out: List[Dict[str, Any]] = []
    for row in manifest.scenes:
        sid = str(row.get("scene_id", "")).strip()
        if not sid:
            continue
        out.append(
            {
                "scene_id": sid,
                "dir": sid,
                "layer_files": dict(row.get("layer_files", {})),
                "auxiliary_files": dict(row.get("auxiliary_files", {})),
            }
        )
    return out


def load_scene_metadata_v3(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    scene_dir = cache_dir / str(item.get("dir", item.get("scene_id", "")))
    meta_path = scene_dir / "metadata_v3.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_scene_pred_v3(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    scene_dir = cache_dir / str(item.get("dir", item.get("scene_id", "")))
    pred_path = scene_dir / "pred_v3.pt"
    if not pred_path.is_file():
        pred_path = scene_dir / "pred.pt"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing prediction file under {scene_dir}")
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    if not isinstance(pred, dict):
        raise TypeError(f"Expected dict in {pred_path}")
    return pred


def _reshape_global_to_match_frame(frame: torch.Tensor, global_t: torch.Tensor) -> torch.Tensor:
    """Align global block activations to frame ``[B,S,P,C]`` for cache-first replay."""
    if frame.shape == global_t.shape:
        return global_t
    if frame.dim() != 4 or global_t.dim() != 4:
        raise ValueError(
            f"Expected 4D frame/global tensors, got frame={tuple(frame.shape)} global={tuple(global_t.shape)}"
        )
    b, s, p, c = (int(x) for x in frame.shape)
    if int(global_t.shape[-1]) != c:
        raise ValueError(
            f"Channel mismatch frame vs global: {tuple(frame.shape)} vs {tuple(global_t.shape)}"
        )
    g = global_t
    if int(g.shape[0]) == 1 and b > 1:
        g = g.expand(b, -1, -1, -1)
    elif int(g.shape[0]) != b:
        raise ValueError(
            f"Batch mismatch frame vs global: {tuple(frame.shape)} vs {tuple(global_t.shape)}"
        )
    if int(g.shape[1]) == s and int(g.shape[2]) == p:
        return g.contiguous()
    if int(g.shape[1]) == 1 and int(g.shape[2]) == s * p:
        return g.reshape(b, s, p, c).contiguous()
    raise ValueError(
        f"Cannot align global {tuple(global_t.shape)} to frame {tuple(frame.shape)}"
    )


def _load_unit_tensor(path: Path, *, map_location: Optional[str] = None) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing layer unit file {path}")
    load_kw: Dict[str, Any] = {"weights_only": False}
    if map_location is not None:
        load_kw["map_location"] = map_location
    t = torch.load(path, **load_kw)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() == 2:
        raise ValueError(
            f"Unit tensor at {path} is flattened [N,C]; re-run run_cache_redundancy.sh with SAVE_*=1 "
            "so training receives full [B,S,P,C] activations."
        )
    if t.dim() != 4:
        raise ValueError(f"Expected 4D unit tensor at {path}, got {tuple(t.shape)}")
    return t.contiguous()


def _unit_path_from_manifest(
    cache_dir: Path,
    scene_dir: Path,
    item: Dict[str, Any],
    unit_id: str,
) -> tuple[Path, bool]:
    layer_files = item.get("layer_files")
    if isinstance(layer_files, dict):
        rel = str(layer_files.get(unit_id, "")).strip()
        if rel:
            return cache_dir / rel, True
        return scene_dir / "layers" / f"{unit_id}.pt", False
    path = scene_dir / "layers" / f"{unit_id}.pt"
    return path, path.is_file()


def _missing_partner_like(ref: torch.Tensor) -> torch.Tensor:
    """Create zero tensor for missing frame/global partner with same shape."""
    return torch.zeros_like(ref).contiguous()


def load_scene_replay_layer_v3(
    cache_dir: Path,
    item: Dict[str, Any],
    layer_idx: int,
    *,
    num_layers: Optional[int] = None,
    map_location: Optional[str] = None,
) -> torch.Tensor:
    """Load one per-layer [B,S,P,2C] concat tensor from frame_i + global_i unit files."""
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be >= 0, got {layer_idx}")
    scene_dir = cache_dir / str(item.get("dir", item.get("scene_id", "")))
    frame_id, global_id = f"frame_{layer_idx}", f"global_{layer_idx}"
    frame_path, has_f = _unit_path_from_manifest(cache_dir, scene_dir, item, frame_id)
    global_path, has_g = _unit_path_from_manifest(cache_dir, scene_dir, item, global_id)
    if not has_f and not has_g:
        raise FileNotFoundError(f"Scene {scene_dir.name} missing both {frame_id} and {global_id} under layers/")
    if has_f:
        ft = _load_unit_tensor(frame_path, map_location=map_location)
        if has_g:
            gt = _reshape_global_to_match_frame(
                ft, _load_unit_tensor(global_path, map_location=map_location)
            )
        else:
            gt = _missing_partner_like(ft)
    else:
        gt_raw = _load_unit_tensor(global_path, map_location=map_location)
        gt = gt_raw.contiguous()
        ft = _missing_partner_like(gt_raw)
    return torch.cat([ft, gt], dim=-1).contiguous()


def load_scene_replay_layers_v3(
    cache_dir: Path,
    item: Dict[str, Any],
    *,
    num_layers: Optional[int] = None,
) -> List[torch.Tensor]:
    """Build per-layer [B,S,P,2C] concat tensors from frame_i + global_i unit files."""
    scene_dir = cache_dir / str(item.get("dir", item.get("scene_id", "")))
    manifest = load_v3_manifest(cache_dir)
    depth = num_layers
    if depth is None:
        depth = max(
            (u.index for u in manifest.layer_units if u.kind in ("frame", "global")),
            default=-1,
        ) + 1
    layers: List[torch.Tensor] = []
    for i in range(int(depth)):
        frame_id, global_id = f"frame_{i}", f"global_{i}"
        _, has_f = _unit_path_from_manifest(cache_dir, scene_dir, item, frame_id)
        _, has_g = _unit_path_from_manifest(cache_dir, scene_dir, item, global_id)
        if not has_f and not has_g:
            raise FileNotFoundError(
                f"Scene {scene_dir.name} missing both {frame_id} and {global_id} under layers/"
            )
        layers.append(load_scene_replay_layer_v3(cache_dir, item, i, num_layers=depth))
    return layers

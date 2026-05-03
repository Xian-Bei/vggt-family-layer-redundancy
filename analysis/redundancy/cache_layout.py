from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _scene_dir_name_from_item(item: Dict[str, Any]) -> str:
    d = str(item.get("dir", "")).strip()
    if d:
        return d
    f = str(item.get("file", "")).strip()
    if f:
        return Path(f).stem
    sid = str(item.get("scene_id", "")).strip()
    if sid:
        return sid.replace("/", "_")
    return ""


def scene_cache_dir(cache_dir: Path, item: Dict[str, Any]) -> Optional[Path]:
    name = _scene_dir_name_from_item(item)
    if not name:
        return None
    return cache_dir / name


def _replay_layer_files(layers_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in layers_dir.glob("layer_*.pt")
        if p.stem != "layer_-1"
    )


def count_scene_replay_layers_disk(cache_dir: Path, item: Dict[str, Any]) -> Optional[int]:
    """Return layer count from ``scene_dir/layers/layer_*.pt`` without loading tensors."""
    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is None:
        return None
    layers_dir = scene_dir / "layers"
    if not layers_dir.is_dir():
        return None
    files = _replay_layer_files(layers_dir)
    return len(files) if files else None


def load_scene_metadata(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is not None:
        meta_path = scene_dir / "metadata.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise SystemExit(f"Invalid metadata JSON in {meta_path}: expected dict.")
            return payload

    file_name = str(item.get("file", "")).strip()
    if not file_name:
        raise SystemExit(f"Manifest item missing both 'dir' and 'file': {item}")
    legacy_path = cache_dir / file_name
    if not legacy_path.exists():
        raise SystemExit(f"Missing legacy scene cache file: {legacy_path}")
    row = _torch_load_cpu(legacy_path)
    if not isinstance(row, dict):
        raise SystemExit(f"Invalid legacy cache row in {legacy_path}: expected dict.")
    return row


def load_scene_pred(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is not None:
        pred_path = scene_dir / "pred.pt"
        if pred_path.exists():
            pred = _torch_load_cpu(pred_path)
            if not isinstance(pred, dict):
                raise SystemExit(f"Invalid pred payload in {pred_path}: expected dict.")
            return pred

    row = load_scene_metadata(cache_dir, item)
    pred = row.get("pred")
    if not isinstance(pred, dict):
        raise SystemExit("Invalid cache row: missing dict key 'pred'.")
    return pred


def load_scene_replay_layers(
    cache_dir: Path,
    item: Dict[str, Any],
    *,
    expected_num_layers: Optional[int] = None,
) -> List[torch.Tensor]:
    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is not None:
        layers_dir = scene_dir / "layers"
        if layers_dir.exists():
            layer_files = _replay_layer_files(layers_dir)
            if expected_num_layers is not None and len(layer_files) != int(expected_num_layers):
                raise SystemExit(
                    f"Expected {expected_num_layers} replay layers under {layers_dir}, found {len(layer_files)}."
                )
            out: List[torch.Tensor] = []
            for idx, path in enumerate(layer_files):
                t = _torch_load_cpu(path)
                if not isinstance(t, torch.Tensor) or t.dim() != 4:
                    raise SystemExit(f"Invalid replay layer tensor in {path}: expected 4D torch.Tensor.")
                out.append(t.contiguous())
            if out:
                return out

    row = load_scene_metadata(cache_dir, item)
    replay_layers = row.get("aggregated_concat_per_layer")
    if not isinstance(replay_layers, list) or not replay_layers:
        raise SystemExit("Cache row lacks aggregated_concat_per_layer. Please rerun Stage1.")
    if expected_num_layers is not None and len(replay_layers) != int(expected_num_layers):
        raise SystemExit(
            f"Expected aggregated_concat_per_layer len={expected_num_layers}, got {len(replay_layers)}."
        )
    out: List[torch.Tensor] = []
    for idx, t in enumerate(replay_layers):
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            raise SystemExit(f"Invalid replay cache tensor at layer {idx}: expected 4D torch.Tensor.")
        out.append(t.contiguous())
    return out


def load_scene_initial_tokens(cache_dir: Path, item: Dict[str, Any]) -> torch.Tensor:
    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is not None:
        initial_path = scene_dir / "layers" / "layer_-1.pt"
        if initial_path.exists():
            t = _torch_load_cpu(initial_path)
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                raise SystemExit(f"Invalid initial token tensor in {initial_path}: expected 4D torch.Tensor.")
            return t.contiguous()

    row = load_scene_metadata(cache_dir, item)
    t = row.get("initial_tokens")
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        raise SystemExit(
            "Cache row lacks initial_tokens/layers/layer_-1.pt. Please rerun Stage1 cache_scannet "
            "with the updated cache format."
        )
    return t.contiguous()


def load_scene_replay_layer(
    cache_dir: Path,
    item: Dict[str, Any],
    layer_idx: int,
    *,
    expected_num_layers: Optional[int] = None,
) -> torch.Tensor:
    """
    Load a single replay layer tensor for one scene.

    This is the lazy/on-demand counterpart of ``load_scene_replay_layers`` and avoids
    deserializing all layers when only a subset is needed.
    """
    if layer_idx < 0:
        raise SystemExit(f"layer_idx must be >= 0, got {layer_idx}")

    scene_dir = scene_cache_dir(cache_dir, item)
    if scene_dir is not None:
        layers_dir = scene_dir / "layers"
        if layers_dir.exists():
            layer_files = _replay_layer_files(layers_dir)
            if not layer_files:
                raise SystemExit(f"No layer_*.pt files found under {layers_dir}")
            if expected_num_layers is not None and len(layer_files) != int(expected_num_layers):
                raise SystemExit(
                    f"Expected {expected_num_layers} replay layers under {layers_dir}, found {len(layer_files)}."
                )
            if layer_idx >= len(layer_files):
                raise SystemExit(
                    f"Requested layer_idx={layer_idx}, but only {len(layer_files)} layers exist under {layers_dir}."
                )
            t = _torch_load_cpu(layer_files[layer_idx])
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                raise SystemExit(f"Invalid replay layer tensor in {layer_files[layer_idx]}: expected 4D torch.Tensor.")
            return t.contiguous()

    row = load_scene_metadata(cache_dir, item)
    replay_layers = row.get("aggregated_concat_per_layer")
    if not isinstance(replay_layers, list) or not replay_layers:
        raise SystemExit("Cache row lacks aggregated_concat_per_layer. Please rerun Stage1.")
    if expected_num_layers is not None and len(replay_layers) != int(expected_num_layers):
        raise SystemExit(
            f"Expected aggregated_concat_per_layer len={expected_num_layers}, got {len(replay_layers)}."
        )
    if layer_idx >= len(replay_layers):
        raise SystemExit(f"Requested layer_idx={layer_idx}, but only {len(replay_layers)} replay layers exist.")
    t = replay_layers[layer_idx]
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        raise SystemExit(f"Invalid replay cache tensor at layer {layer_idx}: expected 4D torch.Tensor.")
    return t.contiguous()


def build_scene_row_compat(
    cache_dir: Path,
    item: Dict[str, Any],
    *,
    expected_num_layers: Optional[int] = None,
    include_pred: bool = True,
) -> Dict[str, Any]:
    row = dict(load_scene_metadata(cache_dir, item))
    row["aggregated_concat_per_layer"] = load_scene_replay_layers(
        cache_dir,
        item,
        expected_num_layers=expected_num_layers,
    )
    if include_pred:
        row["pred"] = load_scene_pred(cache_dir, item)
    return row


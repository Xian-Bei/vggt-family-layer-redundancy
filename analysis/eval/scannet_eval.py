from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vggt.utils.load_fn import load_and_preprocess_images


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SceneData:
    scene_id: str
    scene_dir: Path
    image_paths: List[Path]


def discover_scenes(data_root: Path) -> List[SceneData]:
    scenes: List[SceneData] = []
    for scene_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        image_dir = scene_dir / "images"
        if not image_dir.exists():
            image_dir = scene_dir
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if image_paths:
            scenes.append(SceneData(scene_id=scene_dir.name, scene_dir=scene_dir, image_paths=image_paths))
    return scenes


def _find_image_subdir(scene_dir: Path) -> Optional[Path]:
    """ScanNet-style color/ or generic images/."""
    for name in ("color", "rgb", "images", "image"):
        p = scene_dir / name
        if p.is_dir():
            return p
    return None


def discover_scannet_scenes(data_root: Path) -> List[SceneData]:
    """
    Discover scenes for ScanNet v2 or similar layouts under data_root.

    Supported layouts:
    - ``data_root/scans/sceneXXXX_XX/color/*.jpg`` (official ScanNet extraction)
    - ``data_root/sceneXXXX_XX/color/*.jpg`` (scans unpacked one level up)
    - Fallback: same as ``discover_scenes`` (``scene/images`` or scene root)
    """
    scenes: List[SceneData] = []
    scans = data_root / "scans"
    if scans.is_dir():
        for scene_dir in sorted(p for p in scans.iterdir() if p.is_dir()):
            img_root = _find_image_subdir(scene_dir)
            if img_root is None:
                continue
            image_paths = sorted(p for p in img_root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if image_paths:
                scenes.append(SceneData(scene_id=scene_dir.name, scene_dir=scene_dir, image_paths=image_paths))
        if scenes:
            return scenes

    for scene_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if scene_dir.name == "scans":
            continue
        img_root = _find_image_subdir(scene_dir)
        if img_root is None:
            continue
        image_paths = sorted(p for p in img_root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if image_paths:
            scenes.append(SceneData(scene_id=scene_dir.name, scene_dir=scene_dir, image_paths=image_paths))
    if scenes:
        return scenes

    return discover_scenes(data_root)


def load_scene_images(scene: SceneData, mode: str = "crop") -> torch.Tensor:
    image_list = [str(p) for p in scene.image_paths]
    return load_and_preprocess_images(image_list, mode=mode)


def build_frame_selection(
    image_paths: List[Path],
    available_pose_frame_ids: np.ndarray,
    input_frame: int,
) -> Tuple[List[int], List[Path], List[int]]:
    """
    Same logic as FastVGGT ``analysis.eval.eval_utils.build_frame_selection``:
    intersect image stems with valid pose frame ids, then subsample (first frame kept,
    then approximately uniform step on the rest) when more than ``input_frame`` frames.
    """
    all_image_frame_ids = [int(path.stem) for path in image_paths]
    pose_set = set(int(x) for x in np.asarray(available_pose_frame_ids).ravel().tolist())
    valid_frame_ids = sorted(list(set(all_image_frame_ids) & pose_set))

    if input_frame <= 0:
        selected_frame_ids = valid_frame_ids
    elif len(valid_frame_ids) > input_frame:
        if input_frame == 1:
            selected_frame_ids = [valid_frame_ids[0]]
        else:
            first_frame = valid_frame_ids[0]
            remaining_frames = valid_frame_ids[1:]
            step = max(1, len(remaining_frames) // (input_frame - 1))
            selected_remaining = remaining_frames[::step][: input_frame - 1]
            selected_frame_ids = [first_frame] + selected_remaining
    else:
        selected_frame_ids = valid_frame_ids

    frame_id_to_path = {int(path.stem): path for path in image_paths}
    selected_image_paths = [
        frame_id_to_path[fid] for fid in selected_frame_ids if fid in frame_id_to_path
    ]

    pose_list = [int(x) for x in np.asarray(available_pose_frame_ids).ravel().tolist()]
    pose_frame_to_idx = {fid: idx for idx, fid in enumerate(pose_list)}
    selected_pose_indices = [
        pose_frame_to_idx[fid] for fid in selected_frame_ids if fid in pose_frame_to_idx
    ]

    return selected_frame_ids, selected_image_paths, selected_pose_indices


def select_scene_image_paths_fastvggt(scene: SceneData, max_frames: int) -> SceneData:
    """
    Pick RGB frames the same way as FastVGGT ``eval_scannet.py``: require ``scene_dir/pose/``,
    ``load_poses`` (skip inf/nan), then :func:`build_frame_selection`.
    """
    pose_path = scene.scene_dir / "pose"
    if not pose_path.is_dir():
        raise FileNotFoundError(
            f"Scene {scene.scene_id} is missing required pose directory: {pose_path}"
        )

    from analysis.eval.eval_utils import load_poses

    _, _, available = load_poses(pose_path)
    _ids, selected_paths, _ = build_frame_selection(scene.image_paths, available, max_frames)
    if not selected_paths:
        raise ValueError(
            f"Scene {scene.scene_id} has no frames after pose/image intersection."
        )
    return SceneData(scene_id=scene.scene_id, scene_dir=scene.scene_dir, image_paths=selected_paths)


def estimate_global_flops(model, seq_len: int, image_size: int) -> float:
    agg = model.aggregator
    c = agg.frame_blocks[0].attn.qkv.in_features
    h = agg.frame_blocks[0].attn.num_heads
    mlp_hidden = agg.frame_blocks[0].mlp.fc1.weight.shape[0]
    patch_tokens = (image_size // agg.patch_size) * (image_size // agg.patch_size)
    l = seq_len * (patch_tokens + agg.patch_start_idx)
    per_block = (3 * l * c * c) + (2 * h * l * l * (c // h)) + (l * c * c) + (2 * l * c * mlp_hidden)
    return float(per_block * len(agg.global_blocks))


def count_parameters(model, global_only: bool = False) -> int:
    if not global_only:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.aggregator.global_blocks.parameters())


def aggregate_metric_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = float(sum(vals) / max(len(vals), 1))
    return out


def load_schemes_json(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "schemes" in payload:
        payload = payload["schemes"]
    if not isinstance(payload, list):
        raise ValueError("schemes json must be a list or {\"schemes\": [...]}")
    return payload


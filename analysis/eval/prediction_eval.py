"""FastVGGT-aligned ScanNet pose and geometry evaluation helpers.

This module lives under ``analysis.eval``; :class:`SceneData` and aggregation helpers come from
``analysis.eval.scannet_eval`` (dataset layout / scene discovery).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from analysis.eval.scannet_eval import SceneData, aggregate_metric_dict

from .metrics import (
    evaluate_scene_metrics_fastvggt,
    merge_numeric_metrics_into,
)

SCANNET_DEPTH_MAX = 10.0


def _squeeze_batch_to_ndim(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    a = arr
    while a.ndim > target_ndim and a.shape[0] == 1:
        a = a[0]
    return a


def _depth_or_conf_to_shw(x: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize depth or depth_conf to ``(S, H, W)`` (VGGT may store ``(1,S,1,H,W)``, ``(S,1,H,W)``, ``(S,H,W,1)``, etc.).
    """
    d = _squeeze_batch_to_ndim(x.astype(np.float64), 5)
    # Drop leading batch dim even if already 4D (e.g. (1,S,H,W)).
    while d.ndim >= 4 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 5:
        raise ValueError(f"{name}: unexpected 5D shape {d.shape} after removing batch")
    if d.ndim == 4:
        if d.shape[1] == 1:
            d = d[:, 0, :, :]
        elif d.shape[-1] == 1:
            d = d[..., 0]
        else:
            raise ValueError(f"{name}: expected channel-last or (S,1,H,W), got {d.shape}")
    if d.ndim != 3:
        raise ValueError(f"{name}: expected (S,H,W), got shape {d.shape}")
    return d


def prediction_payload_to_fastvggt_reconstruction(
    payload: Dict[str, Any],
    depth_conf_thresh: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    from .eval_utils import to_homogeneous

    if "extrinsic" not in payload or "depth" not in payload or "intrinsic" not in payload:
        raise ValueError("FastVGGT eval requires extrinsic, depth, intrinsic in prediction file")

    extr = _squeeze_batch_to_ndim(payload["extrinsic"].float().numpy(), 3)
    intr = _squeeze_batch_to_ndim(payload["intrinsic"].float().numpy(), 3)
    depth_t = payload["depth"]
    if not torch.is_tensor(depth_t):
        depth_t = torch.as_tensor(depth_t)
    depth_np = _depth_or_conf_to_shw(depth_t.float().numpy(), "depth").copy()

    dc = payload.get("depth_conf")
    if dc is not None:
        dcn = _depth_or_conf_to_shw(
            dc.float().numpy() if torch.is_tensor(dc) else dc,
            "depth_conf",
        )
        if dcn.shape != depth_np.shape:
            raise ValueError(f"depth_conf shape {dcn.shape} != depth shape {depth_np.shape}")
        mask = dcn >= depth_conf_thresh
        depth_np[~mask] = np.nan

    from analysis.backbones.pose_geometry import unproject_depth_payload

    world_points = unproject_depth_payload(
        depth_t,
        torch.as_tensor(extr),
        torch.as_tensor(intr),
        backbone=payload.get("backbone") or "vggt",
    )
    all_points: List[np.ndarray] = []
    for frame_idx in range(world_points.shape[0]):
        pts = world_points[frame_idx].reshape(-1, 3)
        valid = ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)
        all_points.append(pts[valid])

    cam_4 = to_homogeneous(extr.astype(np.float64))
    cam_mats = [cam_4[i] for i in range(cam_4.shape[0])]
    return extr, cam_mats, all_points


def resolve_gt_ply_dir(data_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    scans = data_root
    return scans if scans.is_dir() else None


@dataclass
class FastVggtSceneEvalContext:
    """Per-scene GT context reused across repeated metric calls.

    Includes pose alignment inputs and the GT mesh point cloud.
    """

    c2ws: np.ndarray
    first_gt_pose: np.ndarray
    frame_ids: List[int]
    gt_ply_dir: Path
    gt_point_cloud: Optional[np.ndarray]


def _resize_depth_nearest(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    pil = Image.fromarray(depth.astype(np.float32), mode="F")
    pil = pil.resize((out_w, out_h), Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.float32)


def _match_depth_hw(depth: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = depth.shape
    if h > target_h:
        start_y = (h - target_h) // 2
        depth = depth[start_y : start_y + target_h, :]
        h = depth.shape[0]
    if w > target_w:
        start_x = (w - target_w) // 2
        depth = depth[:, start_x : start_x + target_w]
        w = depth.shape[1]
    if h < target_h or w < target_w:
        pad_top = max(0, (target_h - h) // 2)
        pad_bottom = max(0, target_h - h - pad_top)
        pad_left = max(0, (target_w - w) // 2)
        pad_right = max(0, target_w - w - pad_left)
        depth = np.pad(depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0)
    return depth.astype(np.float32, copy=False)


def _preprocess_depth_like_vggt(depth: np.ndarray, *, target_width: int = 518, mode: str = "crop") -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    h, w = depth.shape
    if mode not in ("crop", "pad"):
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    if mode == "pad":
        if w >= h:
            new_w = target_width
            new_h = round(h * (new_w / w) / 14) * 14
        else:
            new_h = target_width
            new_w = round(w * (new_h / h) / 14) * 14
    else:
        new_w = target_width
        new_h = round(h * (new_w / w) / 14) * 14
    depth = _resize_depth_nearest(depth, (new_h, new_w))
    if mode == "crop" and new_h > target_width:
        start_y = (new_h - target_width) // 2
        depth = depth[start_y : start_y + target_width, :]
    elif mode == "pad":
        pad_top = max(0, (target_width - depth.shape[0]) // 2)
        pad_bottom = max(0, target_width - depth.shape[0] - pad_top)
        pad_left = max(0, (target_width - depth.shape[1]) // 2)
        pad_right = max(0, target_width - depth.shape[1] - pad_left)
        depth = np.pad(depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0.0)
    return depth.astype(np.float32, copy=False)


def _preprocess_depth_like_omega(depth: np.ndarray, *, output_hw: Tuple[int, int]) -> np.ndarray:
    """Align dense ScanNet depth to the same geometry as vggt_omega RGB preprocessing."""
    from analysis.datasets.driving_depth import _aspect_crop_dims

    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    crop_top, crop_left, crop_h, crop_w = _aspect_crop_dims(depth.shape[0], depth.shape[1])
    depth = depth[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
    return _resize_depth_nearest(depth, (int(output_hw[0]), int(output_hw[1])))


def _read_scannet_depth_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    depth = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth == 65535.0] = 0.0
    depth /= 1000.0
    depth[(depth > SCANNET_DEPTH_MAX) | (depth < 1e-3)] = 0.0
    return depth


def _find_gt_depth_path(scene_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        scene_dir / "depth" / f"{stem}.png",
        scene_dir / "depth_90" / f"{stem}.png",
        scene_dir / f"frame-{stem}.depth.proj.png",
        scene_dir / f"frame-{stem}.depth.png",
    ]
    try:
        stem_int = int(stem)
    except Exception:
        stem_int = None
    if stem_int is not None:
        candidates.extend(
            [
                scene_dir / "depth" / f"{stem_int}.png",
                scene_dir / "depth" / f"{stem_int:06d}.png",
                scene_dir / "depth" / f"depth{stem_int}.png",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _7andn_depth_path_from_color(color_path: Path) -> Path:
    name = color_path.name
    if ".color." not in name:
        raise ValueError(f"Unexpected 7Scenes/NRGBD color path: {color_path}")
    return color_path.parent / name.replace(".color.", ".depth.proj.")


def _read_7andn_depth_meters(path: Path) -> np.ndarray:
    import cv2

    depthmap = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depthmap is None:
        raise FileNotFoundError(f"Could not read 7andn depth: {path}")
    depthmap = np.asarray(depthmap)
    if depthmap.ndim == 3:
        depthmap = depthmap[..., 0]
    depthmap[depthmap == 65535] = 0
    depth = np.nan_to_num(depthmap.astype(np.float32), nan=0.0) / 1000.0
    depth[(depth > 10.0) | (depth < 1e-3)] = 0.0
    return depth


def _load_7andn_gt_depth_vggt(view: Dict[str, Any], image_hw: Tuple[int, int]) -> np.ndarray:
    color_path = Path(str(view["instance"]))
    depth_path = _7andn_depth_path_from_color(color_path)
    depth = _read_7andn_depth_meters(depth_path)
    import cv2

    rgb = cv2.imread(str(color_path))
    if rgb is None:
        raise FileNotFoundError(f"Could not read 7andn color image: {color_path}")
    rgb_h, rgb_w = int(rgb.shape[0]), int(rgb.shape[1])
    if depth.shape != (rgb_h, rgb_w):
        depth = _resize_depth_nearest(depth, (rgb_h, rgb_w))
    depth = _preprocess_depth_like_vggt(depth, target_width=int(image_hw[1]), mode="crop")
    return _match_depth_hw(depth, image_hw)


def _load_7andn_gt_depth_omega(view: Dict[str, Any], image_hw: Tuple[int, int]) -> np.ndarray:
    color_path = Path(str(view["instance"]))
    depth_path = _7andn_depth_path_from_color(color_path)
    depth = _read_7andn_depth_meters(depth_path)
    import cv2

    rgb = cv2.imread(str(color_path))
    if rgb is None:
        raise FileNotFoundError(f"Could not read 7andn color image: {color_path}")
    rgb_h, rgb_w = int(rgb.shape[0]), int(rgb.shape[1])
    if depth.shape != (rgb_h, rgb_w):
        depth = _resize_depth_nearest(depth, (rgb_h, rgb_w))
    depth = _preprocess_depth_like_omega(depth, output_hw=image_hw)
    return _match_depth_hw(depth, image_hw)


def _load_gt_depth_for_scene(scene: SceneData, image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load ScanNet GT depth and align it to the same crop-mode preprocessing as RGB input."""
    depth_maps: List[np.ndarray] = []
    shapes: set[Tuple[int, int]] = set()
    for image_path in scene.image_paths:
        depth_path = _find_gt_depth_path(scene.scene_dir, image_path.stem)
        if depth_path is None:
            raise RuntimeError(
                f"Missing GT depth for frame {image_path.stem} in {scene.scene_id}. "
                "Expected depth/*.png or frame-XXXXXX.depth(.proj).png."
            )
        depth = _read_scannet_depth_png(depth_path)
        with Image.open(image_path) as image:
            rgb_w, rgb_h = image.size
        if depth.shape != (rgb_h, rgb_w):
            depth = _resize_depth_nearest(depth, (rgb_h, rgb_w))
        depth = _preprocess_depth_like_vggt(depth, target_width=int(image_hw[1]), mode="crop")
        depth = _match_depth_hw(depth, image_hw)
        depth_maps.append(depth)
        shapes.add(tuple(depth.shape))
    if not depth_maps:
        raise RuntimeError(f"No GT depth maps loaded for {scene.scene_id}")
    if len(shapes) > 1:
        target_h = max(h for h, _ in shapes)
        target_w = max(w for _, w in shapes)
        depth_maps = [_match_depth_hw(d, (target_h, target_w)) for d in depth_maps]
    return np.stack(depth_maps, axis=0).astype(np.float32, copy=False)


def _load_gt_depth_for_scene_omega(scene: SceneData, image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load ScanNet GT depth aligned to vggt_omega balanced preprocessing."""
    depth_maps: List[np.ndarray] = []
    shapes: set[Tuple[int, int]] = set()
    for image_path in scene.image_paths:
        depth_path = _find_gt_depth_path(scene.scene_dir, image_path.stem)
        if depth_path is None:
            raise RuntimeError(
                f"Missing GT depth for frame {image_path.stem} in {scene.scene_id}. "
                "Expected depth/*.png or frame-XXXXXX.depth(.proj).png."
            )
        depth = _read_scannet_depth_png(depth_path)
        with Image.open(image_path) as image:
            rgb_w, rgb_h = image.size
        if depth.shape != (rgb_h, rgb_w):
            depth = _resize_depth_nearest(depth, (rgb_h, rgb_w))
        depth = _preprocess_depth_like_omega(depth, output_hw=image_hw)
        depth = _match_depth_hw(depth, image_hw)
        depth_maps.append(depth)
        shapes.add(tuple(depth.shape))
    if not depth_maps:
        raise RuntimeError(f"No GT depth maps loaded for {scene.scene_id}")
    if len(shapes) > 1:
        target_h = max(h for h, _ in shapes)
        target_w = max(w for _, w in shapes)
        depth_maps = [_match_depth_hw(d, (target_h, target_w)) for d in depth_maps]
    return np.stack(depth_maps, axis=0).astype(np.float32, copy=False)


def build_fastvggt_scene_eval_context(
    scene: SceneData,
    data_root: Path,
    gt_ply_dir: Optional[Path],
    *,
    image_hw: Optional[Tuple[int, int]] = None,
) -> FastVggtSceneEvalContext:
    from .eval_utils import dependencies_available, load_gt_pointcloud, load_poses

    if not dependencies_available():
        raise RuntimeError("ScanNet eval requires: pip install evo open3d scipy")

    resolved_ply = resolve_gt_ply_dir(data_root, gt_ply_dir)
    if resolved_ply is None:
        raise RuntimeError(
            "No ScanNet ply root: pass --gt-ply-dir (parent of scene folders with *_vh_clean_2.ply) "
            f"or use layout with {data_root / 'scans'}"
        )

    pose_dir = scene.scene_dir / "pose"
    if not pose_dir.is_dir():
        raise RuntimeError(f"No pose/ directory under {scene.scene_dir}")

    loaded = load_poses(pose_dir)
    if loaded[0] is None:
        raise RuntimeError(f"No valid pose txt under {pose_dir}")
    poses_gt_full, first_gt_pose, available_pose_frame_ids = loaded

    frame_ids = [int(p.stem) for p in scene.image_paths]
    pose_frame_to_idx = {int(fid): idx for idx, fid in enumerate(available_pose_frame_ids.tolist())}
    indices: List[int] = []
    for fid in frame_ids:
        if fid not in pose_frame_to_idx:
            raise RuntimeError(f"Missing pose txt for frame {fid} in {scene.scene_id}")
        indices.append(pose_frame_to_idx[fid])
    c2ws = poses_gt_full[np.array(indices, dtype=int)]

    gt_point_cloud, _ = load_gt_pointcloud(scene.scene_id, resolved_ply)
    if gt_point_cloud is None:
        raise RuntimeError(
            f"Missing GT mesh for {scene.scene_id}: expected "
            f"{resolved_ply / scene.scene_id / (scene.scene_id + '_vh_clean_2.ply')}"
        )
    del image_hw
    return FastVggtSceneEvalContext(
        c2ws=c2ws,
        first_gt_pose=first_gt_pose,
        frame_ids=frame_ids,
        gt_ply_dir=resolved_ply,
        gt_point_cloud=gt_point_cloud,
    )


def build_scannet_eval_row_from_payload_with_context(
    payload: Dict[str, Any],
    scene: SceneData,
    ctx: FastVggtSceneEvalContext,
    *,
    depth_conf_thresh: float = 1.0,
    chamfer_max_dist: float = 0.5,
    inference_time_ms: float = 0.0,
) -> Dict[str, Any]:
    """Score one in-memory prediction payload against ScanNet GT.

    Returns pose and geometry metrics. Missing GT is treated as an error.
    """
    row: Dict[str, Any] = {"scene_id": scene.scene_id}
    try:
        extr, cam_mats, pts_list = prediction_payload_to_fastvggt_reconstruction(payload, depth_conf_thresh)
    except Exception as e:
        raise RuntimeError(f"FastVGGT reconstruction failed for {scene.scene_id}: {e}") from e

    if extr.shape[0] != len(ctx.frame_ids):
        raise RuntimeError(
            f"Frame count mismatch: pred S={extr.shape[0]} vs images={len(ctx.frame_ids)} ({scene.scene_id})"
        )

    m = evaluate_scene_metrics_fastvggt(
        scene.scene_id,
        ctx.c2ws,
        ctx.first_gt_pose,
        ctx.frame_ids,
        cam_mats,
        pts_list,
        ctx.gt_ply_dir,
        chamfer_max_dist,
        inference_time_ms=inference_time_ms,
        plot_trajectory=False,
        plot_path=None,
        gt_point_cloud=ctx.gt_point_cloud,
    )
    if m is None:
        raise RuntimeError(f"FastVGGT evaluate_scene_metrics_fastvggt returned None for {scene.scene_id}")
    merge_numeric_metrics_into(row, m)
    row["protocol"] = "fastvggt_scannet"
    return row


def metrics_float_dict(row: Dict[str, Any]) -> Dict[str, float]:
    skip = {"scene_id", "protocol", "infer_time_ms"}
    out: Dict[str, float] = {}
    merge_numeric_metrics_into(out, {k: v for k, v in row.items() if k not in skip})
    return out


def delta_gt_metrics(new_m: Dict[str, float], ref_m: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in new_m.items():
        if k in ref_m:
            out[f"delta_{k}_vs_stage1"] = float(v - ref_m[k])
    return out


def read_eval_scannet_rows_by_scene(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", [])
    if not scenes:
        raise RuntimeError(f"{path} has empty scenes list.")
    out: Dict[str, Dict[str, Any]] = {}
    for row in scenes:
        sid = str(row.get("scene_id", "")).strip()
        if sid:
            out[sid] = row
    if not out:
        raise RuntimeError(f"{path} contains no valid scene_id entries.")
    return out


def write_eval_scannet_json(
    output_dir: Path,
    scene_rows: List[Dict[str, Any]],
    extra_top_level: Optional[Dict[str, Any]] = None,
) -> Path:
    numeric_for_agg = [
        {k: v for k, v in r.items() if k != "scene_id" and isinstance(v, (int, float))} for r in scene_rows
    ]
    payload: Dict[str, Any] = {
        "scenes": scene_rows,
        "aggregate": aggregate_metric_dict(numeric_for_agg),
    }
    if extra_top_level:
        payload.update(extra_top_level)
    path = output_dir / "eval_scannet.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

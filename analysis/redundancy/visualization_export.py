from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .prediction_eval import _depth_or_conf_to_shw, _squeeze_batch_to_ndim


def _safe_scene_id(scene_id: str) -> str:
    return scene_id.replace("/", "_")


def _to_numpy_rgb(images: torch.Tensor) -> np.ndarray:
    """
    Convert preprocessed scene images to numpy RGB in [0,1], shape [S,H,W,3].
    """
    if images.dim() == 5:
        images = images[0]
    if images.dim() != 4:
        raise ValueError(f"Expected images tensor [S,3,H,W] or [1,S,3,H,W], got {tuple(images.shape)}")
    rgb = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(rgb, 0.0, 1.0)


def _reconstruct_points_and_colors_from_payload(
    payload: Dict[str, Any],
    images_rgb: np.ndarray,
    depth_conf_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FastVGGT-aligned depth unprojection + confidence masking, with RGB sampling.
    Returns merged points/colors arrays of shape [N,3], [N,3].
    """
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if "extrinsic" not in payload or "depth" not in payload or "intrinsic" not in payload:
        raise ValueError("Visualization export requires extrinsic, depth, intrinsic in prediction payload")

    extr = _squeeze_batch_to_ndim(payload["extrinsic"].float().cpu().numpy(), 3)
    intr = _squeeze_batch_to_ndim(payload["intrinsic"].float().cpu().numpy(), 3)
    depth_np = _depth_or_conf_to_shw(payload["depth"].float().cpu().numpy(), "depth").copy()
    dc = payload.get("depth_conf")
    if dc is not None:
        dcn = _depth_or_conf_to_shw(dc.float().cpu().numpy(), "depth_conf")
        if dcn.shape != depth_np.shape:
            raise ValueError(f"depth_conf shape {dcn.shape} != depth shape {depth_np.shape}")
        depth_np[dcn < depth_conf_thresh] = np.nan

    world_points = unproject_depth_map_to_point_map(depth_np.astype(np.float64), extr, intr)  # [S,H,W,3]
    if world_points.shape[0] != images_rgb.shape[0]:
        raise ValueError(
            f"Frame mismatch between predictions ({world_points.shape[0]}) and images ({images_rgb.shape[0]})"
        )

    merged_pts: List[np.ndarray] = []
    merged_cols: List[np.ndarray] = []
    for fi in range(world_points.shape[0]):
        pts = world_points[fi].reshape(-1, 3)
        cols = images_rgb[fi].reshape(-1, 3)
        valid = ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)
        if np.any(valid):
            merged_pts.append(pts[valid])
            merged_cols.append(cols[valid])

    if not merged_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.vstack(merged_pts), np.vstack(merged_cols)


def _reconstruct_points_colors_conf_from_payload(
    payload: Dict[str, Any],
    images_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FastVGGT-aligned unprojection without confidence thresholding.
    Returns merged points/colors/conf/frame_idx arrays.
    """
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if "extrinsic" not in payload or "depth" not in payload or "intrinsic" not in payload:
        raise ValueError("Visualization export requires extrinsic, depth, intrinsic in prediction payload")

    extr = _squeeze_batch_to_ndim(payload["extrinsic"].float().cpu().numpy(), 3)
    intr = _squeeze_batch_to_ndim(payload["intrinsic"].float().cpu().numpy(), 3)
    depth_np = _depth_or_conf_to_shw(payload["depth"].float().cpu().numpy(), "depth").copy()
    dc = payload.get("depth_conf")
    if dc is not None:
        conf_np = _depth_or_conf_to_shw(dc.float().cpu().numpy(), "depth_conf")
        if conf_np.shape != depth_np.shape:
            raise ValueError(f"depth_conf shape {conf_np.shape} != depth shape {depth_np.shape}")
    else:
        conf_np = np.ones_like(depth_np, dtype=np.float64)

    world_points = unproject_depth_map_to_point_map(depth_np.astype(np.float64), extr, intr)  # [S,H,W,3]
    if world_points.shape[0] != images_rgb.shape[0]:
        raise ValueError(
            f"Frame mismatch between predictions ({world_points.shape[0]}) and images ({images_rgb.shape[0]})"
        )

    merged_pts: List[np.ndarray] = []
    merged_cols: List[np.ndarray] = []
    merged_conf: List[np.ndarray] = []
    merged_frame: List[np.ndarray] = []
    for fi in range(world_points.shape[0]):
        pts = world_points[fi].reshape(-1, 3)
        cols = images_rgb[fi].reshape(-1, 3)
        cf = conf_np[fi].reshape(-1)
        valid = ~np.isnan(pts).any(axis=1) & ~np.isinf(pts).any(axis=1)
        if np.any(valid):
            merged_pts.append(pts[valid])
            merged_cols.append(cols[valid])
            merged_conf.append(cf[valid])
            merged_frame.append(np.full((int(np.sum(valid)),), fi, dtype=np.int32))

    if not merged_pts:
        z3 = np.zeros((0, 3), dtype=np.float64)
        z1 = np.zeros((0,), dtype=np.float64)
        zf = np.zeros((0,), dtype=np.int32)
        return z3, z3, z1, zf
    return (
        np.vstack(merged_pts),
        np.vstack(merged_cols),
        np.concatenate(merged_conf),
        np.concatenate(merged_frame),
    )


def _camera_centers_from_extrinsics(extrinsic: np.ndarray) -> np.ndarray:
    """
    extrinsic is camera-from-world (OpenCV convention); return camera centers in world coordinates.
    """
    if extrinsic.ndim != 3 or extrinsic.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Unexpected extrinsic shape {extrinsic.shape}")
    if extrinsic.shape[1:] == (3, 4):
        E = np.repeat(np.eye(4)[None, :, :], extrinsic.shape[0], axis=0)
        E[:, :3, :4] = extrinsic
    else:
        E = extrinsic
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    centers = -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)
    return centers


def _build_cameras_json(
    scene_id: str,
    image_paths: List[Path],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    extr = _squeeze_batch_to_ndim(payload["extrinsic"].float().cpu().numpy(), 3)
    intr = _squeeze_batch_to_ndim(payload["intrinsic"].float().cpu().numpy(), 3)
    centers = _camera_centers_from_extrinsics(extr)

    image_hw = payload.get("image_hw", None)
    frames: List[Dict[str, Any]] = []
    n = min(extr.shape[0], intr.shape[0], len(image_paths), centers.shape[0])
    for i in range(n):
        frames.append(
            {
                "frame_index": i,
                "frame_id": image_paths[i].stem,
                "image_path": str(image_paths[i]),
                "extrinsic": extr[i].tolist(),
                "intrinsic": intr[i].tolist(),
                "camera_center_world": centers[i].tolist(),
            }
        )
    return {
        "scene_id": scene_id,
        "image_hw": image_hw,
        "frames": frames,
    }


def _open3d_write_visualization_assets(
    out_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
) -> None:
    import open3d as o3d

    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(out_dir / "pointcloud_rgb.ply"), pcd, write_ascii=False, compressed=False)

    if camera_centers.shape[0] >= 2:
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(camera_centers.astype(np.float64))
        lines = [[i, i + 1] for i in range(camera_centers.shape[0] - 1)]
        traj.lines = o3d.utility.Vector2iVector(lines)
        traj.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.4, 0.1]]), (len(lines), 1)))
        o3d.io.write_line_set(str(out_dir / "trajectory_line.ply"), traj, write_ascii=False, compressed=False)


def export_scene_visualization_assets(
    *,
    output_root: Path,
    scene_id: str,
    image_paths: List[Path],
    images_tensor: torch.Tensor,
    prediction_payload: Dict[str, Any],
    depth_conf_thresh: float,
) -> Optional[Path]:
    """
    Export per-scene visualization assets:
    - pointcloud_rgb.ply (FastVGGT-aligned depth unprojection)
    - trajectory_line.ply
    - cameras.json
    Returns output directory, or None if export is skipped.
    """
    safe_id = _safe_scene_id(scene_id)
    out_dir = output_root / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)

    images_rgb = _to_numpy_rgb(images_tensor)
    points_all, colors_all, conf_all, frame_idx_all = _reconstruct_points_colors_conf_from_payload(
        prediction_payload,
        images_rgb=images_rgb,
    )
    points, colors = _reconstruct_points_and_colors_from_payload(
        prediction_payload,
        images_rgb=images_rgb,
        depth_conf_thresh=depth_conf_thresh,
    )
    cams = _build_cameras_json(scene_id=scene_id, image_paths=image_paths, payload=prediction_payload)
    centers = np.asarray([f["camera_center_world"] for f in cams["frames"]], dtype=np.float64) if cams["frames"] else np.zeros((0, 3))

    try:
        _open3d_write_visualization_assets(out_dir, points, colors, centers)
    except ImportError:
        print("[vis][warn] open3d not installed; skip .ply export and keep browser assets only.", flush=True)
    (out_dir / "cameras.json").write_text(json.dumps(cams, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "viser_points.npz",
        points=points_all.astype(np.float32),
        colors=(np.clip(colors_all, 0.0, 1.0) * 255.0).astype(np.uint8),
        confidence=conf_all.astype(np.float32),
        frame_indices=frame_idx_all.astype(np.int32),
    )
    return out_dir


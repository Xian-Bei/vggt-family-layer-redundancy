"""ScanNet scene-level metrics vs GT — single entry under ``analysis.eval``.

Pose / Chamfer / timing use primitives from :mod:`analysis.eval.eval_utils`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .eval_utils import (
    compute_chamfer_distance,
    eval_trajectory,
    load_gt_pointcloud,
)

STREAMVGGT_DEPTH_MAX = 10.0


def evaluate_depth_metrics_streamvggt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    *,
    max_depth: float = STREAMVGGT_DEPTH_MAX,
) -> Dict[str, float]:
    """Compute depth_abs_rel / depth_delta1 with StreamVGGT-style scale alignment.

    The valid mask is derived directly from GT depth values in the current aligned depth map.
    """
    import torch

    pred = torch.from_numpy(np.asarray(pred_depth, dtype=np.float32))
    gt = torch.from_numpy(np.asarray(gt_depth, dtype=np.float32))
    if pred.shape != gt.shape:
        raise ValueError(f"Pred depth shape {pred.shape} != GT depth shape {gt.shape}")
    if pred.dim() == 3:
        _, _, w = pred.shape
        pred = pred.view(-1, w)
        gt = gt.view(-1, w)
    mask = (gt > 0) & (gt < float(max_depth))
    valid_pixels = int(mask.sum().item())
    if valid_pixels <= 0:
        raise RuntimeError(
            "depth metrics: no valid GT pixels in range (0, max_depth); "
            f"max_depth={max_depth}, pred shape={tuple(pred.shape)}, gt shape={tuple(gt.shape)}"
        )
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    # StreamVGGT video depth defaults to scale alignment.
    s = torch.nanmean(gt_valid) / torch.clamp(torch.nanmean(pred_valid), min=1e-8)
    for _ in range(10):
        residuals = s * pred_valid - gt_valid
        weights = 1.0 / (residuals.abs() + 1e-8)
        numerator = torch.sum(weights * pred_valid * gt_valid)
        denominator = torch.sum(weights * pred_valid.square()).clamp(min=1e-8)
        s = (numerator / denominator).clamp(min=1e-3).detach()
    pred_valid = pred_valid * s

    abs_rel = torch.mean(torch.abs(pred_valid - gt_valid) / gt_valid).item()
    max_ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid.clamp(min=1e-5))
    delta1 = torch.mean((max_ratio < 1.25).float()).item()
    return {
        "depth_abs_rel": float(abs_rel),
        "depth_delta1": float(delta1),
        "depth_valid_pixels": float(valid_pixels),
    }


def merge_numeric_metrics_into(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Copy numeric fields from ``src`` into ``dst`` as Python ``float`` (skips ``bool``).

    When ``dst`` is the same object as ``src``, this coerces numeric values in place.
    """
    for k, v in src.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            dst[k] = float(v)


def evaluate_scene_metrics(
    scene: str,
    c2ws: np.ndarray,
    first_gt_pose: np.ndarray,
    frame_ids: List[int],
    all_cam_to_world_mat: List[np.ndarray],
    all_world_points: List[np.ndarray],
    gt_ply_dir: Path,
    chamfer_max_dist: float,
    inference_time_ms: float,
    *,
    gt_point_cloud: Optional[np.ndarray] = None,
    pred_depth: Optional[np.ndarray] = None,
    gt_depth: Optional[np.ndarray] = None,
    depth_max: float = STREAMVGGT_DEPTH_MAX,
    plot_trajectory: bool = False,
) -> Tuple[Optional[Dict[str, float]], Optional[Image.Image]]:
    """Trajectory (ATE/ARE/RPE) + Chamfer vs ScanNet mesh GT (+ optional depth metrics)."""
    if not all_cam_to_world_mat or not all_world_points:
        raise RuntimeError(
            f"{scene}: cannot evaluate metrics — empty camera poses or point clouds"
        )

    # `c2ws` returned by load_poses() is normalized to the first-frame coordinate system.
    # For 2D plot output (plot.png), we keep this relative coordinate system because it yields
    # a stable, room-local top-down view. For Chamfer vs GT mesh, we must operate in the
    # original ScanNet world coordinate system (mesh ply coordinates).
    c2ws_rel = np.asarray(c2ws, dtype=np.float64)
    w2cs_rel = np.linalg.inv(c2ws_rel)
    traj_est_poses = np.array(all_cam_to_world_mat)
    n = min(len(traj_est_poses), len(w2cs_rel))
    timestamps = frame_ids[:n]
    # Plot + pose stats in relative coordinates (visual convenience).
    stats_aligned, traj_plot, _sim3_rel = eval_trajectory(
        traj_est_poses[:n], w2cs_rel[:n], timestamps, align=True, plot_trajectory=plot_trajectory
    )
    # Sim(3) in original ScanNet world for point cloud alignment against GT mesh.
    c2ws_world = np.asarray(first_gt_pose, dtype=np.float64) @ c2ws_rel
    w2cs_world = np.linalg.inv(c2ws_world)
    _, _, sim3_T = eval_trajectory(
        traj_est_poses[:n], w2cs_world[:n], timestamps, align=True, plot_trajectory=False
    )

    merged_point_cloud = np.vstack(all_world_points)
    if gt_point_cloud is None:
        gt_point_cloud, _ = load_gt_pointcloud(scene, gt_ply_dir)
    if gt_point_cloud is None:
        raise RuntimeError(
            f"{scene}: Chamfer / alignment require GT mesh under gt_ply_dir={gt_ply_dir!s}"
        )
    # Align predicted point cloud into GT mesh world using the same Sim(3) as trajectory alignment.
    P = np.asarray(merged_point_cloud, dtype=np.float64)
    R = np.asarray(sim3_T[:3, :3], dtype=np.float64)
    t = np.asarray(sim3_T[:3, 3], dtype=np.float64)
    world_points_scaled = (P @ R.T) + t[None, :]

    cd_value = compute_chamfer_distance(
        world_points_scaled, gt_point_cloud, max_dist=chamfer_max_dist
    )
    stats_aligned["chamfer_distance"] = float(cd_value)

    all_metrics: Dict[str, float] = deepcopy(stats_aligned)
    all_metrics["inference_time_ms"] = float(inference_time_ms)
    if pred_depth is not None and gt_depth is not None:
        depth_metrics = evaluate_depth_metrics_streamvggt(
            pred_depth, gt_depth, max_depth=depth_max
        )
        for k, v in depth_metrics.items():
            all_metrics[k] = float(v)
    merge_numeric_metrics_into(all_metrics, all_metrics)
    return all_metrics, traj_plot


def evaluate_scene_and_save(
    scene: str,
    c2ws: np.ndarray,
    first_gt_pose: np.ndarray,
    frame_ids: List[int],
    all_cam_to_world_mat: List[np.ndarray],
    all_world_points: List[np.ndarray],
    output_scene_dir: Path,
    gt_ply_dir: Path,
    chamfer_max_dist: float,
    inference_time_ms: float,
    plot_flag: bool,
    *,
    pred_depth: Optional[np.ndarray] = None,
    gt_depth: Optional[np.ndarray] = None,
    depth_max: float = STREAMVGGT_DEPTH_MAX,
) -> Optional[Dict[str, float]]:
    all_metrics, traj_plot = evaluate_scene_metrics(
        scene,
        c2ws,
        first_gt_pose,
        frame_ids,
        all_cam_to_world_mat,
        all_world_points,
        gt_ply_dir,
        chamfer_max_dist,
        inference_time_ms,
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        depth_max=depth_max,
        plot_trajectory=plot_flag,
    )
    if all_metrics is None:
        return None

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    with open(output_scene_dir / "metrics.json", "w") as f:
        import json

        json.dump(all_metrics, f, indent=4)
    if plot_flag and traj_plot is not None:
        traj_plot.save(output_scene_dir / "plot.png")

    return all_metrics


def compute_average_metrics_and_save(
    all_scenes_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    input_frame: Optional[int],
) -> Dict[str, float]:
    metric_names = [
        "chamfer_distance",
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "inference_time_ms",
        "depth_abs_rel",
        "depth_delta1",
    ]
    scenes_payload = all_scenes_metrics.get("scenes") or {}
    if not scenes_payload:
        raise RuntimeError("compute_average_metrics_and_save: empty scenes dict; nothing to average")

    average_metrics_list: Dict[str, List[float]] = {
        metric: [] for metric in metric_names
    }
    for _, metrics in scenes_payload.items():
        for metric_name, metric_value in metrics.items():
            if metric_name in average_metrics_list:
                average_metrics_list[metric_name].append(float(metric_value))

    average_metrics: Dict[str, float] = {}
    for metric_name, values in average_metrics_list.items():
        average_metrics[metric_name] = float(np.mean(values)) if values else 0.0

    output_path.mkdir(parents=True, exist_ok=True)

    out_dir = output_path if input_frame is None else (output_path / f"input_frame_{int(input_frame)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "average_metrics.json", "w") as f:
        import json

        json.dump(average_metrics, f, indent=4)

    return average_metrics


FASTVGGT_CORE_METRICS = (
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
)


def evaluate_scene_metrics_fastvggt(
    scene_id: str,
    c2ws: np.ndarray,
    first_gt_pose: np.ndarray,
    frame_ids: List[int],
    all_cam_to_world_mat: List[np.ndarray],
    all_world_points: List[np.ndarray],
    gt_ply_dir: Path,
    chamfer_max_dist: float,
    inference_time_ms: float = 0.0,
    plot_trajectory: bool = False,
    plot_path: Optional[Path] = None,
    gt_point_cloud: Optional[np.ndarray] = None,
    pred_depth: Optional[np.ndarray] = None,
    gt_depth: Optional[np.ndarray] = None,
    depth_max: float = STREAMVGGT_DEPTH_MAX,
) -> Optional[Dict[str, float]]:
    """Optional trajectory plot export; otherwise same as :func:`evaluate_scene_metrics`."""
    all_metrics, traj_plot = evaluate_scene_metrics(
        scene_id,
        c2ws,
        first_gt_pose,
        frame_ids,
        all_cam_to_world_mat,
        all_world_points,
        gt_ply_dir,
        chamfer_max_dist,
        inference_time_ms,
        gt_point_cloud=gt_point_cloud,
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        depth_max=depth_max,
        plot_trajectory=plot_trajectory,
    )
    if all_metrics is None:
        return None
    if plot_path is not None and traj_plot is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        traj_plot.save(plot_path)
    return all_metrics

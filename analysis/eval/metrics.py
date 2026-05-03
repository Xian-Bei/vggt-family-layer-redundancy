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
    compute_chamfer_and_normal_consistency,
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


def _seven_andn_gt_c2w_valid(c2w: np.ndarray) -> bool:
    """True when GT cam-to-world is usable for trajectory metrics.

    Neural RGB-D uses an identity matrix placeholder for frames whose ``poses.txt``
    line contained ``nan``; those must be excluded.
    """
    T = np.asarray(c2w, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    R = T[:3, :3]
    t = T[:3, 3]
    if np.allclose(R, np.eye(3), atol=1e-4) and np.allclose(t, 0.0, atol=1e-4):
        return False
    return True


def evaluate_7andn_trajectory_metrics(
    est_w2c: np.ndarray,
    gt_c2w: np.ndarray,
    frame_ids: List[int],
    *,
    plot_trajectory: bool = False,
) -> Tuple[Dict[str, float], Optional[Image.Image]]:
    """ATE / ARE / RPE vs GT for 7-Scenes / NRGBD, aligned with ScanNet trajectory eval.

    ``est_w2c`` must be **camera-from-world** (same convention as
    :func:`analysis.eval.eval_utils.infer_vggt_and_reconstruct` / ``pose_encoding_to_extri_intri``).
    ``gt_c2w`` are dataset **camera-to-world** matrices (``data.SevenScenes`` / ``data.NRGBD``).

    GT poses are normalized to the first **valid** frame (same idea as
    :func:`analysis.eval.eval_utils.load_poses`), converted to w2c, then compared to
    ``est_w2c`` via :func:`analysis.eval.eval_utils.eval_trajectory` with ``align=True``
    (Sim(3) on camera centers), matching :func:`evaluate_scene_metrics`.
    """
    est_w2c = np.asarray(est_w2c, dtype=np.float64)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    if est_w2c.ndim != 3 or est_w2c.shape[1:] != (4, 4):
        raise ValueError(f"est_w2c must be [N,4,4], got {est_w2c.shape}")
    if gt_c2w.ndim != 3 or gt_c2w.shape[1:] != (4, 4):
        raise ValueError(f"gt_c2w must be [N,4,4], got {gt_c2w.shape}")
    n_in = min(int(est_w2c.shape[0]), int(gt_c2w.shape[0]))
    if n_in < 2:
        return (
            {
                "ate": float("nan"),
                "are": float("nan"),
                "rpe_rot": float("nan"),
                "rpe_trans": float("nan"),
            },
            None,
        )

    valid_mask = np.array(
        [_seven_andn_gt_c2w_valid(gt_c2w[i]) for i in range(n_in)],
        dtype=bool,
    )
    if int(np.count_nonzero(valid_mask)) < 2:
        return (
            {
                "ate": float("nan"),
                "are": float("nan"),
                "rpe_rot": float("nan"),
                "rpe_trans": float("nan"),
            },
            None,
        )

    est_w2c = est_w2c[:n_in][valid_mask]
    gt_c2w = gt_c2w[:n_in][valid_mask]
    n = int(est_w2c.shape[0])
    if n < 2:
        return (
            {
                "ate": float("nan"),
                "are": float("nan"),
                "rpe_rot": float("nan"),
                "rpe_trans": float("nan"),
            },
            None,
        )

    t0_inv = np.linalg.inv(gt_c2w[0])
    c2ws_rel = np.stack([t0_inv @ gt_c2w[i] for i in range(n)], axis=0)
    w2cs_gt = np.linalg.inv(c2ws_rel)

    try:
        timestamps = [int(frame_ids[i]) for i in range(n_in) if bool(valid_mask[i])]
        if len(timestamps) != n:
            timestamps = list(range(n))
    except Exception:
        timestamps = list(range(n))
    try:
        stats_aligned, traj_plot, _ = eval_trajectory(
            est_w2c[:n],
            w2cs_gt[:n],
            timestamps,
            align=True,
            plot_trajectory=plot_trajectory,
        )
        out = {
            "ate": float(stats_aligned["ate"]),
            "are": float(stats_aligned["are"]),
            "rpe_rot": float(stats_aligned["rpe_rot"]),
            "rpe_trans": float(stats_aligned["rpe_trans"]),
        }
        return out, traj_plot
    except Exception as e:
        msg = str(e)
        if "Degenerate covariance rank" not in msg and "Umeyama alignment" not in msg:
            raise
        print(f"[warn] 7andn trajectory alignment failed ({msg}); writing NaN pose metrics.", flush=True)
        return (
            {
                "ate": float("nan"),
                "are": float("nan"),
                "rpe_rot": float("nan"),
                "rpe_trans": float("nan"),
            },
            None,
        )


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
    infer_peak_vram_gb: float = 0.0,
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
    traj_plot: Optional[Image.Image] = None
    try:
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

        geom = compute_chamfer_and_normal_consistency(
            world_points_scaled, gt_point_cloud, max_dist=chamfer_max_dist
        )
        stats_aligned["chamfer_distance"] = float(geom["chamfer_distance"])
        stats_aligned["nc1"] = float(geom["nc1"])
        stats_aligned["nc2"] = float(geom["nc2"])
        stats_aligned["nc"] = float(geom["nc"])
        stats_aligned["nc1_med"] = float(geom["nc1_med"])
        stats_aligned["nc2_med"] = float(geom["nc2_med"])
        stats_aligned["nc_med"] = float(geom["nc_med"])
    except Exception as e:
        msg = str(e)
        if "Degenerate covariance rank" not in msg and "Umeyama alignment" not in msg:
            raise
        print(f"[warn] {scene}: trajectory alignment failed ({msg}); writing NaN pose/chamfer metrics.")
        stats_aligned = {
            "ate": float("nan"),
            "are": float("nan"),
            "rpe_rot": float("nan"),
            "rpe_trans": float("nan"),
            "chamfer_distance": float("nan"),
            "nc1": float("nan"),
            "nc2": float("nan"),
            "nc": float("nan"),
            "nc1_med": float("nan"),
            "nc2_med": float("nan"),
            "nc_med": float("nan"),
        }

    all_metrics: Dict[str, float] = deepcopy(stats_aligned)
    all_metrics["inference_time_ms"] = float(inference_time_ms)
    all_metrics["infer_peak_vram_gb"] = float(infer_peak_vram_gb)
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
    infer_peak_vram_gb: float = 0.0,
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
        infer_peak_vram_gb=infer_peak_vram_gb,
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
        "nc1",
        "nc2",
        "nc",
        "nc1_med",
        "nc2_med",
        "nc_med",
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "inference_time_ms",
        "infer_peak_vram_gb",
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
        finite_vals = [float(v) for v in values if np.isfinite(float(v))]
        average_metrics[metric_name] = float(np.mean(finite_vals)) if finite_vals else float("nan")

    output_path.mkdir(parents=True, exist_ok=True)

    out_dir = output_path if input_frame is None else (output_path / f"input_frame_{int(input_frame)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "average_metrics.json", "w") as f:
        import json

        json.dump(average_metrics, f, indent=4)

    return average_metrics


FASTVGGT_CORE_METRICS = (
    "chamfer_distance",
    "nc1",
    "nc2",
    "nc",
    "nc1_med",
    "nc2_med",
    "nc_med",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "infer_peak_vram_gb",
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
    infer_peak_vram_gb: float = 0.0,
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
        infer_peak_vram_gb=infer_peak_vram_gb,
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

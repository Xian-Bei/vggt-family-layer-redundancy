# Vendored from FastVGGT (vggt/utils/eval_utils.py) for ScanNet metrics aligned with
# FastVGGT eval_scannet.py: evo ATE/ARE/RPE + Open3D Chamfer + scale alignment.
# SPDX: keep in sync manually when upstream changes.

from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.spatial.transform import Rotation


def dependencies_available() -> bool:
    try:
        import evo.main_ape as _  # noqa: F401
        import open3d as _  # noqa: F401

        return True
    except ImportError:
        return False


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True):
    assert src.shape == dst.shape, f"Input shapes don't match: src {src.shape}, dst {dst.shape}"
    assert src.shape[0] == 3, f"Expected point cloud dimension (3,N), got {src.shape}"

    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered @ src_centered.T

    try:
        U, D, Vt = svd(cov)
        V = Vt.T
        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1
        R = U @ S @ V.T
        if estimate_scale:
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                scale = 1.0
            else:
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()
        return scale, R, t
    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, t


def compute_chamfer_distance(points_pred: np.ndarray, points_gt: np.ndarray, max_dist: float = 1.0) -> float:
    import open3d as o3d

    MAX_POINTS = 100000
    if points_pred.shape[0] > MAX_POINTS:
        np.random.seed(33)
        indices = np.random.choice(points_pred.shape[0], MAX_POINTS, replace=False)
        points_pred = points_pred[indices]
    if points_gt.shape[0] > MAX_POINTS:
        np.random.seed(33)
        indices = np.random.choice(points_gt.shape[0], MAX_POINTS, replace=False)
        points_gt = points_gt[indices]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)

    voxel_size = 0.05
    pcd_pred = pcd_pred.voxel_down_sample(voxel_size)
    pcd_gt = pcd_gt.voxel_down_sample(voxel_size)

    distances1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    distances2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    distances1 = np.clip(distances1, 0, max_dist)
    distances2 = np.clip(distances2, 0, max_dist)
    return float(np.mean(distances1) + np.mean(distances2))


def load_gt_pointcloud(scene_id: str, gt_ply_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    import open3d as o3d

    scene_dir = gt_ply_dir / scene_id
    ply_path = scene_dir / (scene_id + "_vh_clean_2.ply")
    if not ply_path.is_file():
        return None, None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def eval_trajectory(
    poses_est: np.ndarray,
    poses_gt: np.ndarray,
    frame_ids: List[int],
    align: bool = False,
    plot_trajectory: bool = False,
) -> Tuple[Dict[str, float], Optional["Image.Image"], np.ndarray]:
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids,
    )
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids,
    )

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    ate = float(ate_result.stats["rmse"])

    transform = np.eye(4)
    if align:
        try:
            aligned_xyz = ate_result.trajectories["traj"].positions_xyz
            original_xyz = traj_est.positions_xyz
            if len(aligned_xyz) >= 3 and len(original_xyz) >= 3:
                min_points = min(len(aligned_xyz), len(original_xyz))
                aligned_xyz = aligned_xyz[:min_points]
                original_xyz = original_xyz[:min_points]
                try:
                    s, R, t = umeyama_alignment(original_xyz.T, aligned_xyz.T, True)
                    transform = np.eye(4)
                    transform[:3, :3] = s * R
                    transform[:3, 3] = t
                except Exception as e:
                    print(f"umeyama_alignment failed: {e}")
        except Exception as e:
            print(f"Error computing transformation matrix: {e}")

        if np.array_equal(transform, np.eye(4)) and hasattr(ate_result, "trajectories"):
            try:
                orig_pos = traj_est.positions_xyz[0]
                aligned_pos = ate_result.trajectories["traj"].positions_xyz[0]
                transform[:3, 3] = aligned_pos - orig_pos
            except Exception:
                pass

    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    are = float(are_result.stats["rmse"])

    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_rot = float(rpe_rots_result.stats["rmse"])

    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_trans = float(rpe_transs_result.stats["rmse"])

    stats = {"ate": ate, "are": are, "rpe_rot": rpe_rot, "rpe_trans": rpe_trans}
    pillow_image = None
    if plot_trajectory:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import evo.tools.plot as plot
        from PIL import Image

        plot_mode = plot.PlotMode.xz
        fig = plt.figure()
        ax = plot.prepare_axis(fig, plot_mode)
        ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")
        plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
        if align:
            traj_est_aligned = ate_result.trajectories["traj"]
            plot.traj_colormap(
                ax,
                traj_est_aligned,
                ate_result.np_arrays["error_array"],
                plot_mode,
                min_map=ate_result.stats["min"],
                max_map=ate_result.stats["max"],
            )
        else:
            plot.traj_colormap(
                ax,
                traj_est,
                ate_result.np_arrays["error_array"],
                plot_mode,
                min_map=ate_result.stats["min"],
                max_map=ate_result.stats["max"],
            )
        ax.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=90)
        buffer.seek(0)
        pillow_image = Image.open(buffer)
        pillow_image.load()
        buffer.close()
        plt.close(fig)

    return stats, pillow_image, transform


def load_poses(path: Path):
    pose_files = sorted(path.glob("*.txt"), key=lambda x: int(x.stem))
    if len(pose_files) == 0:
        print(f"Warning: No pose files (.txt) found in directory {path}")
        return None, None, None

    c2ws = []
    available_frame_ids = []
    for pose_file in pose_files:
        try:
            nums = [float(x) for x in pose_file.read_text(encoding="utf-8").strip().split()]
            pose = np.array(nums).reshape(4, 4)
            if not (np.isinf(pose).any() or np.isnan(pose).any()):
                c2ws.append(pose)
                available_frame_ids.append(int(pose_file.stem))
        except Exception as e:
            print(f"Error reading pose file {pose_file}: {e}")
            continue

    if len(c2ws) == 0:
        return None, None, None

    c2ws = np.stack(c2ws)
    available_frame_ids = np.array(available_frame_ids)
    first_gt_pose = c2ws[0].copy()
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    return c2ws, first_gt_pose, available_frame_ids


def to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    n = extrinsics.shape[0]
    homogeneous_extrinsics = np.eye(4)[None, :, :].repeat(n, axis=0)
    homogeneous_extrinsics[:, :3, :4] = extrinsics
    return homogeneous_extrinsics


def align_point_clouds_scale(source_pc: np.ndarray, target_pc: np.ndarray):
    source_min = np.min(source_pc, axis=0)
    source_max = np.max(source_pc, axis=0)
    target_min = np.min(target_pc, axis=0)
    target_max = np.max(target_pc, axis=0)
    source_size = source_max - source_min
    target_size = target_max - target_min
    source_center = (source_max + source_min) / 2
    target_center = (target_max + target_min) / 2
    source_diag = np.sqrt(np.sum(source_size**2))
    target_diag = np.sqrt(np.sum(target_size**2))
    if source_diag < 1e-8:
        scale_factor = 1.0
    else:
        scale_factor = target_diag / source_diag
    centered_source = source_pc - source_center
    scaled_centered = centered_source * scale_factor
    scaled_aligned_source = scaled_centered + target_center
    return scaled_aligned_source, float(scale_factor)


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
) -> Optional[Dict[str, float]]:
    """Same logic as FastVGGT ``evaluate_scene_and_save`` but returns metrics only (no JSON)."""
    if not all_cam_to_world_mat or not all_world_points:
        print(f"Skipping {scene_id}: failed to obtain valid camera poses or point clouds")
        return None

    poses_gt = c2ws
    w2cs = np.linalg.inv(poses_gt)
    traj_est_poses = np.array(all_cam_to_world_mat)
    n = min(len(traj_est_poses), len(w2cs))
    timestamps = frame_ids[:n]
    stats_aligned, traj_plot, _ = eval_trajectory(
        traj_est_poses[:n], w2cs[:n], timestamps, align=True, plot_trajectory=plot_trajectory
    )
    if plot_path is not None and traj_plot is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            traj_plot.save(plot_path)
        except Exception as e:
            print(f"Failed to save trajectory plot to {plot_path}: {e}")

    try:
        merged_point_cloud = np.vstack(all_world_points)
        if gt_point_cloud is None:
            gt_point_cloud, _ = load_gt_pointcloud(scene_id, gt_ply_dir)
        if gt_point_cloud is not None:
            homogeneous_points = np.hstack([merged_point_cloud, np.ones((merged_point_cloud.shape[0], 1))])
            world_points_raw = np.dot(homogeneous_points, first_gt_pose.T)[:, :3]
            world_points_scaled, scale_factor = align_point_clouds_scale(world_points_raw, gt_point_cloud)
            cd_value = compute_chamfer_distance(world_points_scaled, gt_point_cloud, max_dist=chamfer_max_dist)
            stats_aligned["chamfer_distance"] = float(cd_value)
            stats_aligned["scale_factor"] = float(scale_factor)
    except Exception as e:
        print(f"Error computing Chamfer Distance for {scene_id}: {e}")

    all_metrics: Dict[str, float] = deepcopy(stats_aligned)  # type: ignore[assignment]
    for metric_name, metric_value in list(stats_aligned.items()):
        all_metrics[f"aligned_{metric_name}"] = float(metric_value)
    for k in list(all_metrics.keys()):
        all_metrics[k] = float(all_metrics[k])
    all_metrics["inference_time_ms"] = float(inference_time_ms)
    return all_metrics


FASTVGGT_CORE_METRICS = (
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "scale_factor",
)

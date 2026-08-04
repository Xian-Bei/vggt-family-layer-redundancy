"""Unified evaluation metrics (v3): single implementation for all datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np

from analysis.schema.v3 import GeometryEvalProfile, UNIFIED_METRIC_KEYS

_POINT_METRIC_KEYS = ("acc", "comp", "cd")


def _nan_point_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in _POINT_METRIC_KEYS}


def compute_pose_metrics(
    est_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    frame_ids: Optional[list] = None,
) -> Dict[str, float]:
    """APE, ARE, RPE-Trans, RPE-Rot via eval_trajectory + Sim(3) alignment."""
    from analysis.eval.eval_utils import eval_trajectory

    n = min(len(est_w2c), len(gt_w2c))
    if n < 2:
        return {k: float("nan") for k in ("ape", "are", "rpe_trans", "rpe_rot")}
    timestamps = frame_ids if frame_ids is not None else list(range(n))
    stats, _, _ = eval_trajectory(
        est_w2c[:n], gt_w2c[:n], timestamps[:n], align=True, plot_trajectory=False
    )
    return {
        "ape": float(stats["ate"]),
        "are": float(stats["are"]),
        "rpe_trans": float(stats["rpe_trans"]),
        "rpe_rot": float(stats["rpe_rot"]),
    }


def _sort_points_xyz(points: np.ndarray) -> np.ndarray:
    """Lexicographic sort for deterministic downstream geometry ops."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    return pts[order]


def _umeyama_rigid_init_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Deterministic 4x4 rigid warm-start for ICP (equal-count prefix after sorting)."""
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 3)
    n = min(src.shape[0], dst.shape[0])
    if n < 3:
        return np.eye(4, dtype=np.float64)

    src = src[:n]
    dst = dst[:n]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    sigma = dst_c.T @ src_c / float(n)
    u, _, vt = np.linalg.svd(sigma)
    s_mat = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    rot = u @ s_mat @ vt
    trans = mu_dst - rot @ mu_src
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = trans
    return tf


def compute_point_metrics(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    profile: GeometryEvalProfile,
    *,
    use_icp: Optional[bool] = None,
) -> Dict[str, float]:
    """Acc, Comp, and CD (=(acc+comp)/2) using the shared geometry profile."""
    try:
        return _compute_point_metrics_impl(
            pred_pts,
            gt_pts,
            profile,
            use_icp=use_icp,
        )
    except Exception as exc:
        warnings.warn(
            f"Point metric computation failed; returning NaN point metrics: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _nan_point_metrics()


def _compute_point_metrics_impl(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    profile: GeometryEvalProfile,
    *,
    use_icp: Optional[bool] = None,
) -> Dict[str, float]:
    import open3d as o3d
    from scipy.spatial import cKDTree as KDTree

    pred = np.asarray(pred_pts, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_pts, dtype=np.float64).reshape(-1, 3)
    pred = pred[np.isfinite(pred).all(axis=-1)]
    gt = gt[np.isfinite(gt).all(axis=-1)]
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return _nan_point_metrics()

    mp = int(profile.max_points)
    if pred.shape[0] > mp:
        rng = np.random.default_rng(33)
        pred = pred[rng.choice(pred.shape[0], mp, replace=False)]
    if gt.shape[0] > mp:
        rng = np.random.default_rng(33)
        gt = gt[rng.choice(gt.shape[0], mp, replace=False)]

    pred = _sort_points_xyz(pred)
    gt = _sort_points_xyz(gt)

    pcd_p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    pcd_g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    if profile.voxel_size > 0:
        pcd_p = pcd_p.voxel_down_sample(float(profile.voxel_size))
        pcd_g = pcd_g.voxel_down_sample(float(profile.voxel_size))

    do_icp = bool(profile.use_icp if use_icp is None else use_icp)
    if do_icp:
        init_tf = _umeyama_rigid_init_transform(
            np.asarray(pcd_p.points, dtype=np.float64),
            np.asarray(pcd_g.points, dtype=np.float64),
        )
        pcd_p = pcd_p.transform(init_tf)
        reg = o3d.pipelines.registration.registration_icp(
            pcd_p,
            pcd_g,
            float(profile.icp_max_correspondence),
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd_p = pcd_p.transform(reg.transformation)

    gp = np.asarray(pcd_g.points)
    pp = np.asarray(pcd_p.points)

    clip = float(profile.dist_clip)
    gt_tree = KDTree(gp)
    acc_dist, _ = gt_tree.query(pp, workers=1)
    if clip > 0:
        acc_dist = np.clip(acc_dist, 0.0, clip)
    acc = float(np.mean(acc_dist))
    pred_tree = KDTree(pp)
    comp_dist, _ = pred_tree.query(gp, workers=1)
    if clip > 0:
        comp_dist = np.clip(comp_dist, 0.0, clip)
    comp = float(np.mean(comp_dist))
    cd = (float(acc) + float(comp)) / 2.0
    return {"acc": float(acc), "comp": float(comp), "cd": float(cd)}


def merge_scene_metrics(*parts: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in parts:
        for k, v in p.items():
            if k in UNIFIED_METRIC_KEYS or k.startswith("infer_"):
                out[k] = float(v)
    return out


def average_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in UNIFIED_METRIC_KEYS if any(k in r for r in rows)]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["num_scenes"] = float(len(rows))
    return out

"""Backbone-specific eval scene implementations (called from adapters only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.contracts import EvalSceneContext
from analysis.eval.metrics_core import (
    compute_point_metrics,
    merge_scene_metrics,
)
from analysis.eval.vis_export_v3 import (
    export_7andn_points_3d,
    save_scene_plot_2d,
    scene_vis_dir,
)


def _collate_7andn_views(batch) -> Any:
    from torch.utils.data._utils.collate import default_collate

    from analysis.eval.eval_7andn_utils import _uncollate_7andn_views

    views = batch.images
    if isinstance(views, list) and views and isinstance(views[0], dict):
        return _uncollate_7andn_views(default_collate(views))
    return views


def _nan_pose_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in ("ape", "are", "rpe_trans", "rpe_rot")}


def _trajectory_alignment(
    est_w2c: np.ndarray,
    gt_w2c: np.ndarray,
    frame_ids: List[int],
) -> tuple[Dict[str, float], np.ndarray | None]:
    """Compute pose metrics and the Sim(3) used for point alignment in one pass."""
    from analysis.eval.eval_utils import eval_trajectory

    n = min(int(len(est_w2c)), int(len(gt_w2c)), int(len(frame_ids)))
    if n < 2:
        return _nan_pose_metrics(), None
    try:
        stats, _, sim3 = eval_trajectory(
            np.asarray(est_w2c, dtype=np.float64)[:n],
            np.asarray(gt_w2c, dtype=np.float64)[:n],
            list(frame_ids)[:n],
            align=True,
            plot_trajectory=False,
        )
    except Exception:
        return _nan_pose_metrics(), None
    if not np.isfinite(sim3).all():
        sim3 = None
    return (
        {
            "ape": float(stats["ate"]),
            "are": float(stats["are"]),
            "rpe_trans": float(stats["rpe_trans"]),
            "rpe_rot": float(stats["rpe_rot"]),
        },
        sim3,
    )


def _transform_points_sim3(pred_points: np.ndarray, sim3: np.ndarray) -> np.ndarray | None:
    pred = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    if pred.shape[0] == 0 or sim3 is None or not np.isfinite(sim3).all():
        return None
    r = np.asarray(sim3[:3, :3], dtype=np.float64)
    t = np.asarray(sim3[:3, 3], dtype=np.float64)
    return (pred @ r.T) + t[None, :]


def _valid_7andn_gt_c2w(c2w: np.ndarray) -> bool:
    t = np.asarray(c2w, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return False
    r = t[:3, :3]
    trans = t[:3, 3]
    if np.allclose(r, np.eye(3), atol=1e-4) and np.allclose(trans, 0.0, atol=1e-4):
        return False
    return True


def _as_view_points_np(points: torch.Tensor, n_views: int) -> list[np.ndarray] | None:
    arr = points.detach().float().cpu().numpy()
    while arr.ndim >= 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim < 4 or arr.shape[-1] != 3:
        return None
    n = min(int(n_views), int(arr.shape[0]))
    return [arr[i].reshape(-1, 3) for i in range(n)]


def _as_view_conf_np(conf: torch.Tensor | None, n_views: int) -> list[np.ndarray] | None:
    if not isinstance(conf, torch.Tensor):
        return None
    arr = conf.detach().float().cpu().numpy()
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim < 3:
        return None
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    n = min(int(n_views), int(arr.shape[0]))
    return [arr[i].reshape(-1) for i in range(n)]


def _nan_point_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in ("acc", "comp", "cd", "nc")}


def eval_scannet_points_adapter(
    adapter, model, batch, device, profile, ctx: EvalSceneContext
) -> Dict[str, float]:
    from analysis.eval.eval_utils import load_gt_pointcloud
    from analysis.eval.metrics import evaluate_depth_metrics_streamvggt
    from analysis.eval.prediction_eval import _load_gt_depth_for_scene
    from analysis.eval.scannet_eval import SceneData, load_scene_images

    scene_dir = Path(batch.meta["scene_dir"])
    sel_paths = [Path(p) for p in batch.meta["image_paths"]]
    selected = SceneData(scene_id=batch.scene_id, scene_dir=scene_dir, image_paths=sel_paths)
    if isinstance(batch.images, torch.Tensor):
        images_b = batch.images.unsqueeze(0).to(device)
    else:
        images_b = torch.from_numpy(np.stack(load_scene_images(selected, mode="crop"))).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    t0 = time.perf_counter()
    ac_kw = autocast_device_dtype(device)
    with torch.no_grad(), torch.autocast(**ac_kw):
        bundle = adapter.forward_predictions(model, images_b)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated(device.index)) / (1024.0**3)

    metrics: Dict[str, float] = {
        "inference_time_ms": float(infer_ms),
        "infer_peak_vram_gb": float(peak_gb),
    }
    pts = bundle.points
    depth = bundle.depth
    est_w2c = None
    est_w2c_eval = None
    w2cs = None
    sim3_eval = None
    frame_ids_eval: List[int] = []
    frame_ids = [int(p.stem) for p in sel_paths]
    merged_points_raw: np.ndarray | None = None
    gt_pts: np.ndarray | None = None

    if pts is not None:
        merged_points_raw = pts.detach().float().cpu().numpy().reshape(-1, 3)
        gt_pts, _ = load_gt_pointcloud(batch.scene_id, scene_dir.parent)

        if ctx.export_vis_3d and ctx.vis_save_root is not None:
            try:
                from analysis.vis.vis_assets import write_viser_points_npz

                pred_dir = scene_vis_dir(ctx.vis_save_root, batch.scene_id) / "vis" / "pred"
                pred_dir.mkdir(parents=True, exist_ok=True)
                cols = np.tile(np.array([[200, 200, 200]], dtype=np.uint8), (merged_points_raw.shape[0], 1))
                write_viser_points_npz(
                    pred_dir / "viser_points.npz",
                    points=merged_points_raw.astype(np.float32),
                    colors_uint8=cols,
                    max_points=int(ctx.vis_max_points),
                )
            except Exception as e:
                print(f"[warn] ScanNet Pi3 3D vis failed for {batch.scene_id}: {e}", flush=True)

    # Depth metrics use all finite predictions on valid GT pixels; prediction confidence is not a metric mask.
    if isinstance(depth, torch.Tensor):
        try:
            d = depth.detach().float()
            if d.ndim == 5 and d.shape[-1] == 1:
                d = d.squeeze(-1)
            if d.ndim != 4:
                raise ValueError(f"Unexpected ScanNet depth shape: {tuple(d.shape)}")
            pred_stack = d[0].cpu().numpy()
            gt_depth = _load_gt_depth_for_scene(selected, image_hw=(int(pred_stack.shape[1]), int(pred_stack.shape[2])))
            if gt_depth is not None and gt_depth.shape == pred_stack.shape:
                depth_stats = evaluate_depth_metrics_streamvggt(
                    pred_stack.astype(np.float32, copy=False),
                    gt_depth.astype(np.float32, copy=False),
                    max_depth=float(profile.depth.max_depth),
                )
                metrics.update(depth_stats)
        except Exception:
            pass

    poses = bundle.camera_poses
    if poses is not None:
        est = poses.detach().float().cpu().numpy()
        if est.ndim == 4 and est.shape[0] == 1:
            est = est[0]
        est_w2c = np.linalg.inv(est)
        from analysis.eval.eval_utils import load_poses

        poses_gt, first_gt_pose, available = load_poses(scene_dir / "pose")
        pose_frame_to_idx = {int(fid): idx for idx, fid in enumerate(available.tolist())}
        matched = [
            (i, pose_frame_to_idx[int(fid)], int(fid))
            for i, fid in enumerate(frame_ids)
            if int(fid) in pose_frame_to_idx and i < len(est_w2c)
        ]
        if matched:
            est_idx = [m[0] for m in matched]
            pose_idx = [m[1] for m in matched]
            frame_ids_eval = [m[2] for m in matched]
            c2ws = poses_gt[pose_idx]
            w2cs = np.linalg.inv(np.asarray(first_gt_pose) @ c2ws)
            est_w2c_eval = est_w2c[est_idx]
            pose_metrics, sim3_eval = _trajectory_alignment(est_w2c_eval, w2cs, frame_ids_eval)
            metrics.update(pose_metrics)

    if merged_points_raw is not None and gt_pts is not None and sim3_eval is not None and len(frame_ids_eval) >= 3:
        aligned = _transform_points_sim3(merged_points_raw, sim3_eval)
        if aligned is not None:
            metrics.update(compute_point_metrics(aligned, gt_pts, profile.geometry, use_icp=False))
        else:
            metrics.update(_nan_point_metrics())
    elif merged_points_raw is not None and gt_pts is not None:
        metrics.update(_nan_point_metrics())

    if ctx.export_vis_2d and est_w2c_eval is not None and w2cs is not None and ctx.vis_save_root is not None:
        save_scene_plot_2d(
            scene_vis_dir(ctx.vis_save_root, batch.scene_id),
            dataset="scannet",
            est_w2c=est_w2c_eval,
            gt_poses=w2cs,
            frame_ids=frame_ids_eval,
        )

    return metrics


def eval_7andn_points_adapter(adapter, model, batch, device, profile, ctx: EvalSceneContext) -> Dict[str, float]:
    from analysis.eval.eval_7andn_utils import _move_batch_to_device, _center_crop_224_hw, _view_depthmap_to_hw

    scene_batch = _collate_7andn_views(batch)
    _move_batch_to_device(scene_batch, str(device))
    for view in scene_batch:
        view["img"] = (view["img"] + 1.0) / 2.0
    imgs = torch.stack([v["img"].squeeze(0) for v in scene_batch], dim=0).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    t0 = time.perf_counter()
    ac_kw = autocast_device_dtype(device)
    with torch.no_grad(), torch.autocast(**ac_kw):
        bundle = adapter.forward_predictions(model, imgs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated(device.index)) / (1024.0**3)

    metrics: Dict[str, float] = {
        "inference_time_ms": float(infer_ms),
        "infer_peak_vram_gb": float(peak_gb),
    }
    pts = bundle.points
    depth = bundle.depth
    conf = bundle.conf
    if pts is None:
        return metrics

    est_w2c = None
    gt_c2w = None
    est_w2c_point = None
    gt_w2c_point = None
    sim3_point = None
    frame_ids_point: List[int] = []
    pose_valid_mask = None
    poses = bundle.camera_poses
    if poses is not None:
        est = poses.detach().float().cpu().numpy()
        if est.ndim == 4 and est.shape[0] == 1:
            est = est[0]
        gt_poses = np.stack(
            [v["camera_pose"].detach().float().cpu().numpy() for v in scene_batch],
            axis=0,
        )
        while gt_poses.ndim > 3 and gt_poses.shape[1] == 1:
            gt_poses = gt_poses[:, 0]
        est_w2c = np.linalg.inv(est)
        gt_c2w = gt_poses
        n_pose = min(int(est_w2c.shape[0]), int(gt_c2w.shape[0]))
        pose_valid_mask = np.array(
            [_valid_7andn_gt_c2w(gt_c2w[i]) for i in range(n_pose)],
            dtype=bool,
        )
        if int(np.count_nonzero(pose_valid_mask)) >= 2:
            est_w2c_point = est_w2c[:n_pose][pose_valid_mask]
            gt_w2c_point = np.linalg.inv(gt_c2w[:n_pose][pose_valid_mask])
            frame_ids_point = [int(i) for i in range(n_pose) if bool(pose_valid_mask[i])]
            pose_metrics, sim3_point = _trajectory_alignment(
                est_w2c_point,
                gt_w2c_point,
                frame_ids_point,
            )
            metrics.update(pose_metrics)
        else:
            metrics.update(_nan_pose_metrics())

    pred_view_chunks = _as_view_points_np(pts, len(scene_batch))
    conf_view_chunks = _as_view_conf_np(conf, len(scene_batch))
    pred_chunks = []
    conf_chunks = []
    gt_chunks = []
    if pred_view_chunks is not None:
        for j, view in enumerate(scene_batch[: len(pred_view_chunks)]):
            if pose_valid_mask is not None and (j >= len(pose_valid_mask) or not bool(pose_valid_mask[j])):
                continue
            p_np = pred_view_chunks[j]
            c_np = conf_view_chunks[j] if conf_view_chunks is not None and j < len(conf_view_chunks) else None
            if c_np is not None and c_np.shape[0] == p_np.shape[0]:
                keep = c_np >= float(profile.depth.conf_thresh)
                p_np = p_np[keep]
                c_np = c_np[keep]
            g = view.get("pts3d")
            m = view.get("valid_mask")
            if g is None:
                continue
            g_np = g.detach().float().cpu().numpy().reshape(-1, 3)
            if m is not None:
                m_np = m.detach().cpu().numpy().astype(bool).reshape(-1)
                g_np = g_np[m_np]
            pred_chunks.append(p_np)
            if c_np is not None:
                conf_chunks.append(c_np)
            gt_chunks.append(g_np)
    pred = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, 3), dtype=np.float32)
    conf_flat = np.concatenate(conf_chunks, axis=0) if conf_chunks else None
    gt_all = np.concatenate(gt_chunks, axis=0) if gt_chunks else np.zeros((0, 3), dtype=np.float32)

    if (
        pred.shape[0] > 0
        and gt_all.shape[0] > 0
        and sim3_point is not None
        and len(frame_ids_point) >= 3
    ):
        aligned = _transform_points_sim3(pred, sim3_point)
        if aligned is not None:
            pred = aligned.astype(np.float32, copy=False)
            metrics.update(compute_point_metrics(pred, gt_all, profile.geometry, use_icp=False))
        else:
            metrics.update(_nan_point_metrics())
    elif pred.shape[0] > 0 and gt_all.shape[0] > 0:
        metrics.update(_nan_point_metrics())

    # Depth metrics use all finite predictions on valid GT pixels; prediction confidence is not a metric mask.
    if isinstance(depth, torch.Tensor):
        try:
            from analysis.eval.metrics import evaluate_depth_metrics_streamvggt

            d = depth.detach().float()
            if d.ndim == 5 and d.shape[-1] == 1:
                d = d.squeeze(-1)
            if d.ndim == 5 and d.shape[2] == 1:
                d = d[:, :, 0]
            if d.ndim != 4:
                raise ValueError(f"Unexpected depth shape: {tuple(d.shape)}")
            d_np = d[0].cpu().numpy()
            pred_crops = []
            gt_crops = []
            n_views = min(len(scene_batch), int(d_np.shape[0]))
            for j in range(n_views):
                pred_hw = d_np[j].astype(np.float32, copy=False)
                pred_crops.append(_center_crop_224_hw(pred_hw))
                gt_crops.append(_center_crop_224_hw(_view_depthmap_to_hw(scene_batch[j])))
            if pred_crops and gt_crops:
                pred_stack = np.stack(pred_crops, axis=0)
                gt_stack = np.stack(gt_crops, axis=0)
                depth_stats = evaluate_depth_metrics_streamvggt(
                    pred_stack.astype(np.float32, copy=False),
                    gt_stack.astype(np.float32, copy=False),
                    max_depth=float(profile.depth.max_depth),
                )
                metrics.update(depth_stats)
        except Exception:
            # Keep eval robust; depth metrics are auxiliary for points-adapter path.
            pass

    if ctx.export_vis_3d and ctx.vis_save_root is not None and pred.shape[0] > 0:
        try:
            image_paths = []
            for view in scene_batch:
                inst = view.get("instance", "")
                if inst:
                    image_paths.append(Path(str(inst)))
            extr = poses.detach().float().cpu().numpy() if poses is not None else np.zeros((0, 4, 4))
            if extr.ndim == 4 and extr.shape[0] == 1:
                extr = extr[0]
            export_7andn_points_3d(
                scene_dir=scene_vis_dir(ctx.vis_save_root, batch.scene_id),
                pred_points=pred,
                pred_colors_u8=np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (pred.shape[0], 1)),
                pred_conf=conf_flat,
                gt_points=gt_all,
                image_paths=image_paths,
                extrinsic_cw=extr,
                intrinsic=None,
                vis_max_points=int(ctx.vis_max_points),
            )
        except Exception as e:
            print(f"[warn] 7andn Pi3 3D vis failed for {batch.scene_id}: {e}", flush=True)

    if ctx.export_vis_2d and est_w2c is not None and gt_c2w is not None and ctx.vis_save_root is not None:
        ds_name = ctx.dataset_name if ctx.dataset_name in ("7scenes", "nrgbd") else "7scenes"
        save_scene_plot_2d(
            scene_vis_dir(ctx.vis_save_root, batch.scene_id),
            dataset=ds_name,
            est_w2c=est_w2c,
            gt_poses=gt_c2w,
            frame_ids=list(range(int(gt_c2w.shape[0]))),
        )

    return merge_scene_metrics(metrics)

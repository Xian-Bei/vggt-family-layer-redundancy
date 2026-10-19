"""Dataset-specific eval scene helpers used by dataset adaptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.contracts import EvalSceneContext, PredictionBundle
from analysis.datasets.contracts import PreparedEvalScene
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


def _transform_w2c_sim3(est_w2c: np.ndarray, sim3: np.ndarray) -> np.ndarray | None:
    """Express predicted world-to-camera poses in the Sim(3)-aligned world.

    ``sim3`` maps predicted world points/centers into the GT world.  Camera
    orientation uses only its rotational component; scale affects centers.
    """
    poses = np.asarray(est_w2c, dtype=np.float64)
    transform = np.asarray(sim3, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return None
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    scaled_rotation = transform[:3, :3]
    scale = float(np.cbrt(np.linalg.det(scaled_rotation)))
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return None
    world_rotation = scaled_rotation / scale
    translation = transform[:3, 3]
    aligned = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    for index, pose in enumerate(poses):
        rotation_cw = pose[:3, :3]
        center = -(rotation_cw.T @ pose[:3, 3])
        aligned_center = scaled_rotation @ center + translation
        aligned_rotation_cw = rotation_cw @ world_rotation.T
        aligned[index, :3, :3] = aligned_rotation_cw
        aligned[index, :3, 3] = -(aligned_rotation_cw @ aligned_center)
    return aligned


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


def _as_view_point_maps_np(points: torch.Tensor, n_views: int) -> list[np.ndarray] | None:
    arr = points.detach().float().cpu().numpy()
    while arr.ndim >= 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        return None
    n = min(int(n_views), int(arr.shape[0]))
    return [arr[i] for i in range(n)]


def _resize_valid_mask(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    while valid.ndim > 2 and valid.shape[0] == 1:
        valid = valid[0]
    if valid.ndim != 2:
        raise ValueError(f"Expected a 2D valid mask, got {valid.shape}")
    target = (int(output_hw[0]), int(output_hw[1]))
    if valid.shape == target:
        return valid.reshape(-1)
    import cv2

    resized = cv2.resize(
        valid.astype(np.uint8),
        (target[1], target[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool).reshape(-1)


def _prediction_valid_mask(
    adapter: Any,
    view: Dict[str, Any],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Map GT depth validity into a backbone's prediction raster geometry."""
    align = getattr(adapter, "align_7andn_gt_depth", None)
    if callable(align):
        depth = np.asarray(align(view, output_hw=output_hw), dtype=np.float32).squeeze()
        if depth.shape != tuple(output_hw):
            raise ValueError(
                f"Aligned GT depth shape {depth.shape} != prediction shape {output_hw}"
            )
        return (np.isfinite(depth) & (depth > 0.0)).reshape(-1)
    mask = view.get("valid_mask")
    if mask is None:
        return np.ones(int(output_hw[0]) * int(output_hw[1]), dtype=bool)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return _resize_valid_mask(np.asarray(mask), output_hw)


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


def _scannet_visual_attributes(
    points: torch.Tensor,
    images: torch.Tensor | None,
    confidence: torch.Tensor | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Map ScanNet prediction raster order to RGB, confidence, and frame ids."""
    point_arr = points.detach()
    while point_arr.ndim >= 5 and point_arr.shape[0] == 1:
        point_arr = point_arr[0]
    if point_arr.ndim != 4 or point_arr.shape[-1] != 3:
        return None, None, None
    views, height, width = map(int, point_arr.shape[:3])
    frame_indices = np.repeat(
        np.arange(views, dtype=np.int32), height * width
    )

    colors = None
    if isinstance(images, torch.Tensor):
        image_tensor = images.detach().float().cpu()
        while image_tensor.ndim >= 5 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        if image_tensor.ndim == 4 and image_tensor.shape[0] >= views:
            image_tensor = image_tensor[:views]
            if tuple(image_tensor.shape[-2:]) != (height, width):
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            colors = (
                image_tensor.permute(0, 2, 3, 1)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .numpy()
                .reshape(-1, 3)
            )

    conf_flat = None
    if isinstance(confidence, torch.Tensor):
        conf_arr = confidence.detach().float().cpu().numpy()
        while conf_arr.ndim >= 4 and conf_arr.shape[0] == 1:
            conf_arr = conf_arr[0]
        conf_flat = conf_arr.reshape(-1)
        if conf_flat.shape[0] != frame_indices.shape[0]:
            conf_flat = None
    return colors, conf_flat, frame_indices


def _nan_point_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in ("acc", "comp", "cd")}


def _prepare_7andn_eval_batch(adapter, batch, device: torch.device):
    scene_batch = _collate_7andn_views(batch)
    image_paths = [str(v["instance"]) for v in scene_batch]
    if not image_paths:
        raise ValueError(f"7andn {batch.scene_id}: missing view instance paths")
    imgs = adapter.load_eval_images(image_paths)
    images_b = imgs.unsqueeze(0).float().to(device)
    return scene_batch, images_b


def prepare_scannet_eval_scene_cpu(adapter, batch, profile) -> PreparedEvalScene:
    del profile
    from analysis.eval.eval_utils import load_gt_pointcloud
    from analysis.eval.scannet_eval import SceneData

    scene_dir = Path(batch.meta["scene_dir"])
    sel_paths = [Path(p) for p in batch.meta["image_paths"]]
    if not sel_paths:
        raise ValueError(f"ScanNet {batch.scene_id}: missing image_paths in batch meta")
    selected = SceneData(scene_id=batch.scene_id, scene_dir=scene_dir, image_paths=sel_paths)
    images = adapter.load_eval_images(sel_paths)
    images_b = images.unsqueeze(0).float().contiguous()

    gt_pts, gt_colors = load_gt_pointcloud(batch.scene_id, scene_dir.parent)
    from analysis.eval.eval_utils import load_poses

    poses_gt, first_gt_pose, available = load_poses(scene_dir / "pose")
    frame_ids = [int(p.stem) for p in sel_paths]
    pose_frame_to_idx = {int(fid): idx for idx, fid in enumerate(available.tolist())}
    matched_gt = [
        (i, pose_frame_to_idx[int(fid)], int(fid))
        for i, fid in enumerate(frame_ids)
        if int(fid) in pose_frame_to_idx
    ]
    return PreparedEvalScene(
        scene_id=str(batch.scene_id),
        images=images_b,
        state={
            "gt_points": gt_pts,
            "gt_colors": gt_colors,
            "poses_gt": poses_gt,
            "first_gt_pose": first_gt_pose,
            "matched_gt": matched_gt,
            "image_paths": sel_paths,
        },
    )


def prepare_scannet_skip_scene(adapter, batch, device, profile) -> PreparedEvalScene:
    prepared = prepare_scannet_eval_scene_cpu(adapter, batch, profile)
    prepared.images = prepared.images.to(device)
    return prepared


def score_scannet_skip_prediction(
    bundle: PredictionBundle,
    prepared: PreparedEvalScene,
    profile,
    ctx: EvalSceneContext,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    gt_pts = prepared.state.get("gt_points")

    pts = bundle.points
    est_w2c = None
    est_w2c_eval = None
    w2cs = None
    sim3_eval = None
    frame_ids_eval: List[int] = []
    merged_points_raw: np.ndarray | None = None
    pred_colors, pred_conf, pred_frame_indices = None, None, None

    if pts is not None:
        merged_points_raw = pts.detach().float().cpu().numpy().reshape(-1, 3)
        pred_colors, pred_conf, pred_frame_indices = _scannet_visual_attributes(
            pts, prepared.images, bundle.conf
        )

    poses = bundle.camera_poses
    if poses is not None:
        est = poses.detach().float().cpu().numpy()
        if est.ndim == 4 and est.shape[0] == 1:
            est = est[0]
        est_w2c = np.linalg.inv(est)
        poses_gt = prepared.state["poses_gt"]
        first_gt_pose = prepared.state["first_gt_pose"]
        matched = [m for m in prepared.state["matched_gt"] if m[0] < len(est_w2c)]
        if matched:
            est_idx = [m[0] for m in matched]
            pose_idx = [m[1] for m in matched]
            frame_ids_eval = [m[2] for m in matched]
            c2ws = poses_gt[pose_idx]
            w2cs = np.linalg.inv(np.asarray(first_gt_pose) @ c2ws)
            est_w2c_eval = est_w2c[est_idx]
            pose_metrics, sim3_eval = _trajectory_alignment(est_w2c_eval, w2cs, frame_ids_eval)
            metrics.update(pose_metrics)

    aligned_for_vis = None
    if merged_points_raw is not None and gt_pts is not None and sim3_eval is not None and len(frame_ids_eval) >= 3:
        aligned = _transform_points_sim3(merged_points_raw, sim3_eval)
        if aligned is not None:
            aligned_for_vis = aligned.astype(np.float32, copy=False)
            metrics.update(compute_point_metrics(aligned, gt_pts, profile.geometry, use_icp=False))
        else:
            metrics.update(_nan_point_metrics())
    elif merged_points_raw is not None and gt_pts is not None:
        metrics.update(_nan_point_metrics())

    if ctx.export_vis_3d and ctx.vis_save_root is not None and merged_points_raw is not None:
        try:
            from analysis.vis.vis_assets import write_cameras_json, write_viser_points_npz

            scene_dir = scene_vis_dir(ctx.vis_save_root, prepared.scene_id)
            pred_dir = scene_dir / "vis" / "pred"
            gt_dir = scene_dir / "vis" / "gt"
            pred_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            pred_points = aligned_for_vis if aligned_for_vis is not None else merged_points_raw
            if pred_colors is None or pred_colors.shape[0] != pred_points.shape[0]:
                pred_colors = np.tile(
                    np.array([[200, 200, 200]], dtype=np.uint8),
                    (pred_points.shape[0], 1),
                )
            write_viser_points_npz(
                pred_dir / "viser_points.npz",
                points=np.asarray(pred_points, dtype=np.float32),
                colors_uint8=pred_colors,
                confidence=pred_conf,
                frame_indices=pred_frame_indices,
                max_points=int(ctx.vis_max_points),
            )
            if gt_pts is not None and len(gt_pts) > 0:
                gt_colors = prepared.state.get("gt_colors")
                if gt_colors is None or len(gt_colors) != len(gt_pts):
                    gt_colors_u8 = np.tile(
                        np.array([[160, 160, 160]], dtype=np.uint8),
                        (len(gt_pts), 1),
                    )
                else:
                    gt_colors_arr = np.asarray(gt_colors)
                    if np.issubdtype(gt_colors_arr.dtype, np.floating):
                        gt_colors_arr = gt_colors_arr * 255.0
                    gt_colors_u8 = np.clip(gt_colors_arr, 0, 255).astype(np.uint8)
                write_viser_points_npz(
                    gt_dir / "viser_points_gt.npz",
                    points=np.asarray(gt_pts, dtype=np.float32),
                    colors_uint8=gt_colors_u8,
                    max_points=int(ctx.vis_max_points),
                )
            if est_w2c is not None:
                camera_w2c = _transform_w2c_sim3(est_w2c, sim3_eval)
                if camera_w2c is None:
                    camera_w2c = est_w2c
                write_cameras_json(
                    pred_dir / "cameras.json",
                    scene_id=prepared.scene_id,
                    image_paths=prepared.state.get("image_paths", []),
                    extrinsic_cw=np.asarray(camera_w2c, dtype=np.float64),
                    intrinsic=None,
                )
        except Exception as e:
            print(f"[warn] ScanNet 3D vis failed for {prepared.scene_id}: {e}", flush=True)

    if ctx.export_vis_2d and est_w2c_eval is not None and w2cs is not None and ctx.vis_save_root is not None:
        save_scene_plot_2d(
            scene_vis_dir(ctx.vis_save_root, prepared.scene_id),
            dataset="scannet",
            est_w2c=est_w2c_eval,
            gt_poses=w2cs,
            frame_ids=frame_ids_eval,
        )

    return metrics


def eval_scannet_points_adapter(
    adapter, model, batch, device, profile, ctx: EvalSceneContext
) -> Dict[str, float]:
    prepared = prepare_scannet_skip_scene(adapter, batch, device, profile)
    ac_kw = autocast_device_dtype(device)
    with torch.no_grad(), torch.autocast(**ac_kw):
        bundle = adapter.forward_predictions(model, prepared.images)
    return score_scannet_skip_prediction(bundle, prepared, profile, ctx)


def eval_7andn_points_adapter(adapter, model, batch, device, profile, ctx: EvalSceneContext) -> Dict[str, float]:
    scene_batch, imgs = _prepare_7andn_eval_batch(adapter, batch, device)

    ac_kw = autocast_device_dtype(device)
    with torch.no_grad(), torch.autocast(**ac_kw):
        bundle = adapter.forward_predictions(model, imgs)

    metrics: Dict[str, float] = {}
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

    pred_view_maps = _as_view_point_maps_np(pts, len(scene_batch))
    conf_view_chunks = _as_view_conf_np(conf, len(scene_batch))
    pred_chunks = []
    conf_chunks = []
    gt_chunks = []
    if pred_view_maps is not None:
        for j, view in enumerate(scene_batch[: len(pred_view_maps)]):
            if pose_valid_mask is not None and (j >= len(pose_valid_mask) or not bool(pose_valid_mask[j])):
                continue
            point_map = pred_view_maps[j]
            p_np = point_map.reshape(-1, 3)
            c_np = conf_view_chunks[j] if conf_view_chunks is not None and j < len(conf_view_chunks) else None
            # Confidence is retained only for optional visualization and never
            # gates metrics. GT depth validity is applied independently below.
            if c_np is not None and c_np.shape[0] != p_np.shape[0]:
                c_np = None
            pred_valid = _prediction_valid_mask(
                adapter,
                view,
                (int(point_map.shape[0]), int(point_map.shape[1])),
            )
            p_np = p_np[pred_valid]
            if c_np is not None:
                c_np = c_np[pred_valid]
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


def prepare_7andn_inference_scene_cpu(adapter, batch) -> PreparedEvalScene:
    scene_batch = batch.images
    if (
        not isinstance(scene_batch, list)
        or not scene_batch
        or not isinstance(scene_batch[0], dict)
    ):
        scene_batch = _collate_7andn_views(batch)
    image_paths = [str(v["instance"]) for v in scene_batch]
    if not image_paths:
        raise ValueError(f"7andn {batch.scene_id}: missing view instance paths")
    images = adapter.load_eval_images(image_paths).unsqueeze(0).float().contiguous()
    return PreparedEvalScene(scene_id=str(batch.scene_id), images=images)


def prepare_7andn_eval_scene_cpu(adapter, batch, profile) -> PreparedEvalScene:
    del profile
    scene_batch = batch.images
    if (
        not isinstance(scene_batch, list)
        or not scene_batch
        or not isinstance(scene_batch[0], dict)
    ):
        scene_batch = _collate_7andn_views(batch)
    prepared = prepare_7andn_inference_scene_cpu(adapter, batch)
    images = prepared.images

    def as_float32_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def as_bool_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(bool, copy=False)
        return np.asarray(value).astype(bool, copy=False)

    gt_poses = np.stack(
        [as_float32_numpy(v["camera_pose"]) for v in scene_batch], axis=0
    )
    while gt_poses.ndim > 3 and gt_poses.shape[1] == 1:
        gt_poses = gt_poses[:, 0]
    pose_valid = np.array([_valid_7andn_gt_c2w(p) for p in gt_poses], dtype=bool)
    gt_points_by_view: List[np.ndarray] = []
    pred_valid_masks_by_view: List[np.ndarray] = []
    pred_hw = (int(images.shape[-2]), int(images.shape[-1]))
    for view in scene_batch:
        gt = view.get("pts3d")
        mask = view.get("valid_mask")
        if gt is None:
            gt_points_by_view.append(np.zeros((0, 3), dtype=np.float32))
        else:
            gt_np = as_float32_numpy(gt).reshape(-1, 3)
            if mask is not None:
                gt_np = gt_np[as_bool_numpy(mask).reshape(-1)]
            gt_points_by_view.append(gt_np)
        pred_valid_masks_by_view.append(
            _prediction_valid_mask(adapter, view, pred_hw).reshape(pred_hw)
        )
    return PreparedEvalScene(
        scene_id=str(batch.scene_id),
        images=images,
        state={
            "num_views": int(len(scene_batch)),
            "gt_poses": gt_poses,
            "pose_valid": pose_valid,
            "gt_points_by_view": gt_points_by_view,
            "pred_valid_masks_by_view": pred_valid_masks_by_view,
        },
    )


def prepare_7andn_skip_scene(adapter, batch, device, profile) -> PreparedEvalScene:
    prepared = prepare_7andn_eval_scene_cpu(adapter, batch, profile)
    prepared.images = prepared.images.to(device)
    return prepared


def score_7andn_skip_prediction(
    bundle: PredictionBundle,
    prepared: PreparedEvalScene,
    profile,
    ctx: EvalSceneContext,
) -> Dict[str, float]:
    del ctx

    metrics: Dict[str, float] = {}
    scene_batch = prepared.state.get("scene_batch")
    num_views = int(
        prepared.state.get(
            "num_views",
            len(scene_batch) if scene_batch is not None else 0,
        )
    )
    gt_poses = prepared.state["gt_poses"]
    pose_valid = prepared.state["pose_valid"]
    sim3_point = None
    frame_ids_point: List[int] = []
    poses = bundle.camera_poses
    if poses is not None:
        est = poses.detach().float().cpu().numpy()
        if est.ndim == 4 and est.shape[0] == 1:
            est = est[0]
        est_w2c = np.linalg.inv(est)
        n_pose = min(int(est_w2c.shape[0]), int(gt_poses.shape[0]))
        valid = pose_valid[:n_pose]
        if int(np.count_nonzero(valid)) >= 2:
            est_eval = est_w2c[:n_pose][valid]
            gt_eval = np.linalg.inv(gt_poses[:n_pose][valid])
            frame_ids_point = [i for i in range(n_pose) if bool(valid[i])]
            pose_metrics, sim3_point = _trajectory_alignment(est_eval, gt_eval, frame_ids_point)
            metrics.update(pose_metrics)
        else:
            metrics.update(_nan_pose_metrics())

    pred_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    pred_maps = _as_view_point_maps_np(bundle.points, num_views) if bundle.points is not None else None
    if pred_maps is not None:
        stored_masks = prepared.state.get("pred_valid_masks_by_view", [])
        for j, pred_map in enumerate(pred_maps):
            if j >= len(pose_valid) or not bool(pose_valid[j]):
                continue
            pred_view = pred_map.reshape(-1, 3)
            if j < len(stored_masks):
                pred_valid = _resize_valid_mask(
                    stored_masks[j],
                    (int(pred_map.shape[0]), int(pred_map.shape[1])),
                )
                pred_view = pred_view[pred_valid]
            pred_chunks.append(pred_view)
            gt_chunks.append(prepared.state["gt_points_by_view"][j])
    pred_all = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, 3), dtype=np.float32)
    gt_all = np.concatenate(gt_chunks, axis=0) if gt_chunks else np.zeros((0, 3), dtype=np.float32)
    if pred_all.shape[0] and gt_all.shape[0] and sim3_point is not None and len(frame_ids_point) >= 3:
        aligned = _transform_points_sim3(pred_all, sim3_point)
        if aligned is not None:
            metrics.update(compute_point_metrics(aligned, gt_all, profile.geometry, use_icp=False))
        else:
            metrics.update(_nan_point_metrics())
    elif pred_all.shape[0] and gt_all.shape[0]:
        metrics.update(_nan_point_metrics())

    return merge_scene_metrics(metrics)

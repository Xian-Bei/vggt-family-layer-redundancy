"""Shared 7Scenes / NRGBD evaluation logic used by both eval_7andN.py and the
training-time periodic eval in train_lora_skip_baseline.py.

Why this module exists: previously eval_7andN.py inlined the per-scene forward,
metric, and aggregation pipeline. The training script could not reuse it, so a
parallel `collect_batch`-based path was used in dev eval which broke under
fastvggt's `forward_from_aggregated_tokens(autocast=False)`. Centralizing the
logic here lets both entry points share the same model forward + metric path
and unifies output schemas (logs.txt / logs_all.txt / average_metrics.json,
per-scene ``<scene>/metrics.json`` with geometry + trajectory + timing + peak VRAM (GB)
+ dense depth (``depth_*``, ScanNet / StreamVGGT protocol).
"""

from __future__ import annotations

import gc
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# data.py / criterion.py / utils.py / base.py import each other using both
# `analysis/eval/` and `analysis/`-relative paths (see eval_7andN.py top).
# Mirror that resolution here so this helper works whether invoked as a script
# or imported from the training script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ANALYSIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


SEVEN_ANDN_METRIC_KEYS = (
    "acc",
    "comp",
    "cd",
    "nc1",
    "nc2",
    "nc",
    "acc_med",
    "comp_med",
    "cd_med",
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
    "depth_valid_pixels",
)

# Output schema for ``average_metrics.json`` (ordered for stable diffs).
AVERAGE_METRICS_SCHEMA_KEYS = (
    "acc",
    "comp",
    "cd",
    "nc1",
    "nc2",
    "nc",
    "acc_med",
    "comp_med",
    "cd_med",
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
    "depth_valid_pixels",
    "num_scenes",
)


def _center_crop_224_hw(arr: np.ndarray) -> np.ndarray:
    """Match 7andn point-eval crop: 224×224 centered on VGGT-resolution frame."""
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW depth slice, got {arr.shape}")
    H, W = int(arr.shape[0]), int(arr.shape[1])
    cx_, cy_ = W // 2, H // 2
    l, t, r, b = cx_ - 112, cy_ - 112, cx_ + 112, cy_ + 112
    return arr[t:b, l:r]


def _view_depthmap_to_hw(view: Dict[str, Any]) -> np.ndarray:
    """GT depth map in meters (CPU); ``depthmap`` is not moved to CUDA by ``_move_batch_to_device``."""
    dm = view["depthmap"]
    if isinstance(dm, torch.Tensor):
        x = dm.detach().float().cpu().numpy()
    else:
        x = np.asarray(dm, dtype=np.float32)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected depthmap HxW after squeeze, got {x.shape}")
    return x.astype(np.float32, copy=False)


def _depth_to_bshw(depth: torch.Tensor) -> torch.Tensor:
    """Squeeze trailing channel from VGGT depth head: [B,S,H,W,1] -> [B,S,H,W]."""
    if depth.ndim == 5 and depth.size(-1) == 1:
        return depth.squeeze(-1)
    return depth


def _conf_to_bshw(conf: torch.Tensor) -> torch.Tensor:
    """Normalize confidence layout to [B,S,H,W] for shape-consistent masking/cropping."""
    if conf.ndim == 5 and conf.size(-1) == 1:
        return conf.squeeze(-1)
    return conf


def _depth_to_world_points_torch(
    depth_bshw: torch.Tensor,
    extrinsics_bsx34: torch.Tensor,
    intrinsics_bsx33: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unproject depth to world points (OpenCV extrinsic: cam_from_world)."""
    if depth_bshw.dim() != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {tuple(depth_bshw.shape)}")
    if extrinsics_bsx34.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics [B,S,3,4], got {tuple(extrinsics_bsx34.shape)}")
    if intrinsics_bsx33.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [B,S,3,3], got {tuple(intrinsics_bsx33.shape)}")

    B, S, H, W = depth_bshw.shape
    device = depth_bshw.device

    depth = depth_bshw.float()
    fx = intrinsics_bsx33[..., 0, 0].float().unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics_bsx33[..., 1, 1].float().unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics_bsx33[..., 0, 2].float().unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics_bsx33[..., 1, 2].float().unsqueeze(-1).unsqueeze(-1)

    u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(B, S, H, W)
    v = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(B, S, H, W)

    valid = torch.isfinite(depth) & (depth > eps)
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    R_cw = extrinsics_bsx34[..., :3].float()
    t_cw = extrinsics_bsx34[..., 3].float()
    R_wc = R_cw.transpose(-1, -2)
    t_wc = -(R_wc @ t_cw.unsqueeze(-1)).squeeze(-1)

    world = (
        torch.matmul(R_wc[:, :, None, None, :, :], cam_points.unsqueeze(-1)).squeeze(-1)
        + t_wc[:, :, None, None, :]
    )
    world = torch.where(valid.unsqueeze(-1), world, torch.full_like(world, float("nan")))
    return world


def compute_7andn_average_metrics(per_scene_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average per-scene metrics into the canonical average_metrics.json payload.

    Both eval_7andn (live) and the backfill script feed this exact function.
    Keys outside ``AVERAGE_METRICS_SCHEMA_KEYS`` are ignored when building the average.
    """
    if not per_scene_rows:
        raise ValueError("compute_7andn_average_metrics: empty per_scene_rows")
    out: Dict[str, Any] = {}
    for k in SEVEN_ANDN_METRIC_KEYS:
        vals = [float(row[k]) for row in per_scene_rows if k in row]
        if not vals:
            continue
        out[k] = float(sum(vals) / len(vals))
    out["num_scenes"] = int(len(per_scene_rows))
    return {k: out[k] for k in AVERAGE_METRICS_SCHEMA_KEYS if k in out}


def _move_batch_to_device(batch: List[Dict[str, Any]], device: str) -> None:
    """Mirror eval_7andN.py's per-batch device transfer (in place).

    ``img`` is left on CPU so :func:`_evaluate_one_scene_7andn` can ``torch.cat`` then
    ``.to(device)`` once. Moving every view's ``img`` to CUDA *and* catting duplicated
    ~400MB+ of pixels on GPU for long NRGBD clips and pushed the next scene over OOM.
    """
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "img"}
    for view in batch:
        for name in list(view.keys()):
            if name in ignore_keys:
                continue
            v = view[name]
            if isinstance(v, (tuple, list)):
                view[name] = [x.to(device, non_blocking=True) for x in v]
            else:
                view[name] = v.to(device, non_blocking=True)


def _evaluate_one_scene_7andn(
    *,
    model: torch.nn.Module,
    batch: List[Dict[str, Any]],
    device: str,
    eval_amp_dtype: torch.dtype,
    eval_conf_thresh: float,
    criterion: Any,
    revisit: int,
    use_proj: bool,
    rng_np: np.random.Generator,
    save_path: Optional[str],
    export_visualization: bool,
    vis_max_points: int,
    log_file: Optional[str],
    scene_infer_times: defaultdict,
    scene_peak_vram_gb: defaultdict,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run one 7andn scene through model + ICP + accuracy/completion + trajectory.

    Returns per-scene metrics dict (geometry, ``ate``/``are``/``rpe_*``, ``inference_time_ms``,
    ``infer_peak_vram_gb``)
    or None if the scene was skipped for lack of valid points.
    """
    import open3d as o3d  # local import: not all eval consumers need open3d
    from utils import accuracy, completion  # type: ignore  # analysis/eval/utils.py
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if isinstance(batch, dict) and "img" in batch:
        batch["img"] = (batch["img"] + 1.0) / 2.0
        imgs_tensor = batch["img"].to(device, non_blocking=True)
    elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
        for view in batch:
            view["img"] = (view["img"] + 1.0) / 2.0
        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0).to(device, non_blocking=True)
    else:
        raise RuntimeError("Unexpected 7andn batch layout (expected list[dict] with 'img').")

    # Defragment before the next forward: long runs leave the allocator unable to satisfy
    # a new large contiguous block (e.g. last NRGBD scene with more views than prior ones).
    if torch.cuda.is_available() and imgs_tensor.is_cuda:
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if imgs_tensor.is_cuda:
        torch.cuda.reset_peak_memory_stats(imgs_tensor.device.index)
    start = time.time()
    # Confine autocast to the model forward only, matching eval_7andN.py and eval_scannet.py.
    # FastVGGT's ``forward_from_aggregated_tokens`` opens its own
    # ``autocast(enabled=False)`` but does not upcast the bf16 aggregator outputs
    # before feeding them into ``camera_head.token_norm`` — that mismatch causes
    # the "expected scalar type BFloat16 but found Float" LayerNorm error. Detect
    # the vendor backend (no ``return_layers`` on aggregator), run the aggregator
    # inside autocast, then upcast tokens to fp32 for the heads.
    from analysis.aggregator_compat import aggregator_supports_return_layers

    if hasattr(model, "forward_from_aggregated_tokens") and not aggregator_supports_return_layers(
        getattr(model, "aggregator", None)
    ):
        # Match VGGT.forward's auto-unsqueeze ([S,3,H,W] -> [1,S,3,H,W]) so the
        # aggregator sees the same [B,S,3,H,W] layout the heads expect.
        agg_input = imgs_tensor if imgs_tensor.dim() == 5 else imgs_tensor.unsqueeze(0)
        with torch.cuda.amp.autocast(dtype=eval_amp_dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(agg_input)
        aggregated_tokens_list = [
            t.float() if isinstance(t, torch.Tensor) else t for t in aggregated_tokens_list
        ]
        preds = model.forward_from_aggregated_tokens(
            aggregated_tokens_list,
            int(patch_start_idx),
            agg_input,
        )
        del aggregated_tokens_list
    else:
        with torch.cuda.amp.autocast(dtype=eval_amp_dtype):
            preds = model(imgs_tensor)
    torch.cuda.synchronize()
    inference_time_ms = (time.time() - start) * 1000.0
    infer_peak_vram_gb = 0.0
    if imgs_tensor.is_cuda:
        infer_peak_vram_gb = float(
            torch.cuda.max_memory_allocated(imgs_tensor.device.index)
        ) / (1024.0**3)

    # ``torch.cat`` buffer: no longer needed after forward; keeping it through ICP/trajectory
    # inflated peak memory across scenes once extra post-forward work was added.
    img_hw = (int(imgs_tensor.shape[2]), int(imgs_tensor.shape[3]))
    del imgs_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = preds
    views = batch
    if "pose_enc" in predictions:
        B, S = predictions["pose_enc"].shape[:2]
    elif "world_points" in predictions:
        B, S = predictions["world_points"].shape[:2]
    else:
        raise KeyError("predictions is missing a key to infer sequence length")

    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], img_hw)
    depth = _depth_to_bshw(predictions["depth"])
    depth_conf = predictions.get("depth_conf", None)
    if depth_conf is None:
        raise KeyError("predictions missing required key `depth_conf` for depth->points.")
    depth_conf = _conf_to_bshw(depth_conf)
    depth_f = depth.float()
    if depth_f.shape != depth_conf.shape:
        raise ValueError(
            f"Depth/conf shape mismatch after layout normalize: "
            f"depth={tuple(depth_f.shape)} conf={tuple(depth_conf.shape)}"
        )
    depth_f = torch.where(
        depth_conf.float() >= eval_conf_thresh,
        depth_f,
        torch.full_like(depth_f, float("nan")),
    )
    world_points_from_depth = _depth_to_world_points_torch(depth_f, extrinsics, intrinsics)

    ress: List[Dict[str, Any]] = []
    for s in range(S):
        res = {
            "pts3d_in_other_view": world_points_from_depth[:, s],
            "conf": depth_conf[:, s],
            "depth": depth[:, s],
            "depth_conf": depth_conf[:, s],
            "camera_pose": predictions["pose_enc"][:, s, :],
        }
        if isinstance(views, list) and s < len(views) and "valid_mask" in views[s]:
            res["valid_mask"] = views[s]["valid_mask"]
        if "track" in predictions:
            res.update(
                {
                    "track": predictions["track"][:, s],
                    "vis": predictions.get("vis", None)[:, s] if "vis" in predictions else None,
                    "track_conf": predictions.get("conf", None)[:, s] if "conf" in predictions else None,
                }
            )
        ress.append(res)
    preds = ress

    valid_length = len(preds) // max(1, int(revisit))
    if int(revisit) > 1:
        preds = preds[-valid_length:]
        batch = batch[-valid_length:]

    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(batch, preds)

    dv = len(batch)
    depth_sl = depth[:, -dv:, :, :]
    depth_conf_sl = depth_conf[:, -dv:, :, :]
    dep_s_hw = depth_sl[0].detach().float().cpu().numpy()
    conf_s_hw = depth_conf_sl[0].detach().float().cpu().numpy()
    pred_m_hw = np.where(conf_s_hw >= float(eval_conf_thresh), dep_s_hw, np.nan).astype(np.float32)

    pts_all: List[np.ndarray] = []
    pts_gt_all: List[np.ndarray] = []
    images_all: List[np.ndarray] = []
    masks_all: List[np.ndarray] = []
    conf_all: List[np.ndarray] = []
    pred_depth_crops: List[np.ndarray] = []
    gt_depth_crops: List[np.ndarray] = []
    for j, view in enumerate(batch):
        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view["valid_mask"].cpu().numpy()[0]
        pts = pred_pts[j].cpu().numpy()[0]
        conf_np = preds[j]["conf"].cpu().data.numpy()[0]
        pts_gt = gt_pts[j].detach().cpu().numpy()[0]
        H, W = image.shape[:2]
        cx_, cy_ = W // 2, H // 2
        l, t, r, b = cx_ - 112, cy_ - 112, cx_ + 112, cy_ + 112
        image = image[t:b, l:r]
        mask = mask[t:b, l:r]
        pts = pts[t:b, l:r]
        pts_gt = pts_gt[t:b, l:r]
        conf_np = conf_np[t:b, l:r]
        mask = mask & (conf_np >= eval_conf_thresh)
        images_all.append(image[None, ...])
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_all.append(conf_np[None, ...])
        pred_depth_crops.append(_center_crop_224_hw(pred_m_hw[j]))
        gt_depth_crops.append(_center_crop_224_hw(_view_depthmap_to_hw(view)))

    images_arr = np.concatenate(images_all, axis=0)
    pts_arr = np.concatenate(pts_all, axis=0)
    pts_gt_arr = np.concatenate(pts_gt_all, axis=0)
    masks_arr = np.concatenate(masks_all, axis=0)
    conf_arr = np.concatenate(conf_all, axis=0) if conf_all else None

    # ``preds = ress`` replaced the list name but ``predictions`` still referenced the full
    # VGGT output dict (depth / pose_enc / …) until function end — extra trajectory + depth
    # metrics extended that lifetime and caused OOM on the *next* scene. Drop GPU tensors
    # once NumPy copies exist; keep ``extrinsics`` / ``intrinsics`` for vis + pose metrics.
    del preds, pred_pts, gt_pts, gt_factor, pr_factor, masks, monitoring
    del world_points_from_depth, depth_f, depth_sl, depth_conf_sl, depth, depth_conf, predictions
    for _v in batch:
        if isinstance(_v.get("img"), torch.Tensor) and _v["img"].is_cuda:
            _v["img"] = _v["img"].detach().cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scene_id = view["label"][0].rsplit("/", 1)[0]
    scene_num_images = len(batch)
    try:
        scene_infer_times[scene_id].append(float(inference_time_ms))
        scene_peak_vram_gb[scene_id].append(float(infer_peak_vram_gb))
    except Exception:
        pass

    if not quiet:
        print(
            f"{scene_id} — Inference time: {inference_time_ms:.2f}ms, "
            f"infer peak VRAM: {infer_peak_vram_gb:.4f} GB",
            flush=True,
        )

    pts_masked = pts_arr[masks_arr > 0]
    pts_gt_masked = pts_gt_arr[masks_arr > 0]
    images_masked = images_arr[masks_arr > 0]
    conf_masked = conf_arr[masks_arr > 0] if conf_arr is not None else None

    pred_valid = np.isfinite(pts_masked).all(axis=-1)
    gt_valid = np.isfinite(pts_gt_masked).all(axis=-1)
    point_valid = pred_valid & gt_valid
    pts_masked = pts_masked[point_valid]
    pts_gt_masked = pts_gt_masked[point_valid]
    images_masked = images_masked[point_valid]
    if conf_masked is not None:
        conf_masked = conf_masked[point_valid]

    pts_masked = pts_masked.reshape(-1, 3)
    pts_gt_masked = pts_gt_masked.reshape(-1, 3)
    images_masked = images_masked.reshape(-1, 3)
    if conf_masked is not None:
        conf_masked = conf_masked.reshape(-1)

    if pts_masked.shape[0] == 0 or pts_gt_masked.shape[0] == 0:
        if not quiet:
            print(f"[warn] skip {scene_id}: no valid finite points after masking", flush=True)
        return None

    if pts_masked.shape[0] > 999999:
        idx = rng_np.choice(pts_masked.shape[0], 999999, replace=False)
        pts_masked = pts_masked[idx]
        images_masked = images_masked[idx]
        if conf_masked is not None:
            conf_masked = conf_masked[idx]
    if pts_gt_masked.shape[0] > 999999:
        idx_gt = rng_np.choice(pts_gt_masked.shape[0], 999999, replace=False)
        pts_gt_masked = pts_gt_masked[idx_gt]

    if export_visualization and save_path is not None:
        try:
            from analysis.vis.vis_assets import write_cameras_json, write_viser_points_npz

            scene_out = Path(save_path) / str(scene_id)
            (scene_out / "vis" / "pred").mkdir(parents=True, exist_ok=True)
            (scene_out / "vis" / "gt").mkdir(parents=True, exist_ok=True)
            cols_u8 = (np.asarray(images_masked, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
            write_viser_points_npz(
                scene_out / "vis" / "pred" / "viser_points.npz",
                points=np.asarray(pts_masked, dtype=np.float32),
                colors_uint8=cols_u8,
                confidence=None if conf_masked is None else np.asarray(conf_masked, dtype=np.float32),
                frame_indices=None,
                max_points=int(vis_max_points),
            )
            write_viser_points_npz(
                scene_out / "vis" / "gt" / "viser_points_gt.npz",
                points=np.asarray(pts_gt_masked, dtype=np.float32),
                colors_uint8=np.tile(np.array([[160, 160, 160]], dtype=np.uint8), (int(pts_gt_masked.shape[0]), 1)),
                confidence=None,
                frame_indices=None,
                max_points=int(vis_max_points),
            )
            try:
                extr = extrinsics.detach().float().cpu().numpy()
                intr = intrinsics.detach().float().cpu().numpy()
                if extr.ndim == 4:
                    extr = extr[0]
                if intr.ndim == 4:
                    intr = intr[0]
                image_paths = (
                    [Path(v["path"][0]) for v in batch]
                    if isinstance(batch, list) and len(batch) > 0 and "path" in batch[0]
                    else []
                )
                write_cameras_json(
                    scene_out / "vis" / "pred" / "cameras.json",
                    scene_id=str(scene_id),
                    image_paths=image_paths,
                    extrinsic_cw=extr,
                    intrinsic=intr,
                    image_hw=None,
                )
            except Exception:
                pass
        except Exception as e:
            if not quiet:
                print(f"[warn] 7andN visualization export failed for {scene_id}: {e}", flush=True)

    if use_proj:
        def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
            assert src.shape == dst.shape
            N, dim = src.shape
            mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
            src_c, dst_c = src - mu_src, dst - mu_dst
            Sigma = dst_c.T @ src_c / N
            U, D, Vt = np.linalg.svd(Sigma)
            S_ = np.eye(dim)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S_[-1, -1] = -1
            R = U @ S_ @ Vt
            var_src = (src_c**2).sum() / N
            s = (D * S_.diagonal()).sum() / var_src
            t = mu_dst - s * R @ mu_src
            return (s * (R @ src.T)).T + t

        pts_masked = _umeyama(pts_masked.reshape(-1, 3), pts_gt_masked.reshape(-1, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_masked)
    pcd.colors = o3d.utility.Vector3dVector(images_masked)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_masked)
    pcd_gt.colors = o3d.utility.Vector3dVector(images_masked)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd = pcd.transform(reg_p2p.transformation)
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    gt_normal = np.asarray(pcd_gt.normals)
    pred_normal = np.asarray(pcd.normals)
    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

    pred_depth_stack = np.stack(pred_depth_crops, axis=0)
    gt_depth_stack = np.stack(gt_depth_crops, axis=0)
    depth_nan = {
        "depth_abs_rel": float("nan"),
        "depth_delta1": float("nan"),
        "depth_valid_pixels": float("nan"),
    }
    depth_stats: Dict[str, float] = dict(depth_nan)
    try:
        from .metrics import STREAMVGGT_DEPTH_MAX, evaluate_depth_metrics_streamvggt

        if pred_depth_stack.shape != gt_depth_stack.shape:
            raise ValueError(
                f"pred/gt depth shape mismatch: {pred_depth_stack.shape} vs {gt_depth_stack.shape}"
            )
        depth_stats = evaluate_depth_metrics_streamvggt(
            pred_depth_stack.astype(np.float32, copy=False),
            gt_depth_stack.astype(np.float32, copy=False),
            max_depth=float(STREAMVGGT_DEPTH_MAX),
        )
    except Exception as e:
        if not quiet:
            print(f"[warn] 7andn depth metrics skipped for {scene_id}: {e}", flush=True)

    extr_np = extrinsics.detach().float().cpu().numpy()
    if extr_np.ndim == 4:
        extr_np = extr_np[0]
    if int(extr_np.shape[0]) > dv:
        extr_np = extr_np[-dv:, :, :]
    gt_pose_rows: List[np.ndarray] = []
    for view in batch:
        cp = view["camera_pose"]
        if isinstance(cp, torch.Tensor):
            t_pose = cp.detach().float().cpu().numpy()
        else:
            t_pose = np.asarray(cp, dtype=np.float64)
        while t_pose.ndim > 2 and t_pose.shape[0] == 1:
            t_pose = t_pose[0]
        if t_pose.shape != (4, 4):
            raise ValueError(f"Unexpected camera_pose shape {t_pose.shape} for scene {scene_id}")
        gt_pose_rows.append(t_pose.astype(np.float64, copy=False))
    gt_c2w = np.stack(gt_pose_rows, axis=0)

    pose_nan = {
        "ate": float("nan"),
        "are": float("nan"),
        "rpe_rot": float("nan"),
        "rpe_trans": float("nan"),
    }
    pose_stats = dict(pose_nan)
    try:
        from .eval_utils import to_homogeneous
        from .metrics import evaluate_7andn_trajectory_metrics

        est_w2c = to_homogeneous(extr_np.astype(np.float32, copy=False))
        frame_ids = list(range(int(gt_c2w.shape[0])))
        pose_stats, _traj_plot = evaluate_7andn_trajectory_metrics(
            est_w2c, gt_c2w, frame_ids, plot_trajectory=False
        )
    except ImportError as e:
        if not quiet:
            print(f"[warn] 7andn trajectory eval skipped (missing deps): {e}", flush=True)

    # Pose path only needs ``extr_np`` (CPU); drop GPU pose tensors before writing logs / row.
    del extrinsics, intrinsics
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    cd = float(acc) + float(comp)
    cd_med = float(acc_med) + float(comp_med)
    line = (
        f"Idx: {scene_id}, NumImgs: {scene_num_images}, "
        f"Acc: {acc}, Comp: {comp}, Cd: {cd}, NC1: {nc1}, NC2: {nc2} - "
        f"Acc_med: {acc_med}, Compc_med: {comp_med}, Cd_med: {cd_med}, "
        f"NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
    )
    if not quiet:
        print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")
    torch.cuda.empty_cache()

    row: Dict[str, float] = {
        "scene_id": scene_id,
        "num_imgs": float(scene_num_images),
        "acc": float(acc),
        "comp": float(comp),
        "cd": cd,
        "nc1": float(nc1),
        "nc2": float(nc2),
        "nc": (float(nc1) + float(nc2)) / 2.0,
        "acc_med": float(acc_med),
        "comp_med": float(comp_med),
        "cd_med": cd_med,
        "nc1_med": float(nc1_med),
        "nc2_med": float(nc2_med),
        "nc_med": (float(nc1_med) + float(nc2_med)) / 2.0,
        "inference_time_ms": float(inference_time_ms),
        "infer_peak_vram_gb": float(infer_peak_vram_gb),
        "ate": float(pose_stats["ate"]),
        "are": float(pose_stats["are"]),
        "rpe_rot": float(pose_stats["rpe_rot"]),
        "rpe_trans": float(pose_stats["rpe_trans"]),
        "depth_abs_rel": float(depth_stats["depth_abs_rel"]),
        "depth_delta1": float(depth_stats["depth_delta1"]),
        "depth_valid_pixels": float(depth_stats["depth_valid_pixels"]),
    }

    if save_path is not None:
        scene_out = Path(save_path) / str(scene_id)
        scene_out.mkdir(parents=True, exist_ok=True)
        metrics_payload = {k: v for k, v in row.items() if k != "scene_id"}
        metrics_payload["scene_id"] = str(scene_id)
        with open(scene_out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)

    return row


def _write_logs_all(save_path: str, scene_infer_times: defaultdict, scene_peak_vram_gb: defaultdict) -> None:
    """Write logs_all.txt = logs.txt + per-scene Time_avg_ms + PeakVRAM_avg_GB blocks."""
    to_write = ""
    if os.path.exists(osp.join(save_path, "logs.txt")):
        with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
            to_write += f_sub.read()
    time_lines = [
        f"Idx: {sid}, Time_avg_ms: {np.mean(times):.2f}"
        for sid, times in scene_infer_times.items()
        if len(times) > 0
    ]
    time_block = "\n".join(time_lines) + ("\n" if time_lines else "")
    vram_lines = [
        f"Idx: {sid}, PeakVRAM_avg_GB: {np.mean(vals):.4f}"
        for sid, vals in scene_peak_vram_gb.items()
        if len(vals) > 0
    ]
    vram_block = "\n".join(vram_lines) + ("\n" if vram_lines else "")
    with open(osp.join(save_path, "logs_all.txt"), "w") as f:
        f.write(to_write + time_block + vram_block)


def evaluate_7andn_dataset(
    *,
    model: torch.nn.Module,
    dataset: Any,
    device: str = "cuda",
    eval_amp_dtype: torch.dtype = torch.bfloat16,
    save_path: Optional[str] = None,
    eval_conf_thresh: float = 1.0,
    max_samples: int = 0,
    revisit: int = 1,
    use_proj: bool = False,
    export_visualization: bool = False,
    vis_max_points: int = 0,
    rng_np: Optional[np.random.Generator] = None,
    write_average_metrics_json: bool = True,
    quiet: bool = False,
) -> Dict[str, float]:
    """Run 7Scenes/NRGBD eval for one dataset and return mean metrics dict.

    When ``save_path`` is provided, writes ``logs.txt``, ``logs_all.txt`` and
    ``average_metrics.json`` under it (mirroring eval_7andN.py's layout).

    When ``quiet`` is True, suppresses per-scene stdout and the inner tqdm bar
    (e.g. when called from training next to another progress bar).
    """
    from criterion import L21, Regr3D_t_ScaleShiftInv  # type: ignore  # analysis/eval/criterion.py

    if rng_np is None:
        rng_np = np.random.default_rng(33)

    log_file: Optional[str] = None
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        log_file = osp.join(save_path, "logs.txt")
        # Truncate logs.txt so re-runs do not accumulate stale lines.
        open(log_file, "w").close()

    scene_infer_times: defaultdict = defaultdict(list)
    scene_peak_vram_gb: defaultdict = defaultdict(list)
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    total_items = len(dataset)
    if max_samples > 0:
        total_items = min(total_items, max_samples)

    per_scene_rows: List[Dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for data_idx in tqdm(
                range(total_items),
                desc="eval 7andn",
                leave=False,
                disable=bool(quiet),
            ):
                batch = default_collate([dataset[data_idx]])
                _move_batch_to_device(batch, device)
                row = _evaluate_one_scene_7andn(
                    model=model,
                    batch=batch,
                    device=device,
                    eval_amp_dtype=eval_amp_dtype,
                    eval_conf_thresh=eval_conf_thresh,
                    criterion=criterion,
                    revisit=int(revisit),
                    use_proj=bool(use_proj),
                    rng_np=rng_np,
                    save_path=save_path,
                    export_visualization=bool(export_visualization),
                    vis_max_points=int(vis_max_points),
                    log_file=log_file,
                    scene_infer_times=scene_infer_times,
                    scene_peak_vram_gb=scene_peak_vram_gb,
                    quiet=bool(quiet),
                )
                del batch
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                if row is not None:
                    per_scene_rows.append(row)
    finally:
        if was_training:
            model.train()

    if save_path is not None:
        _write_logs_all(save_path, scene_infer_times, scene_peak_vram_gb)

    if not per_scene_rows:
        # Empty mean is preferable to crashing the surrounding training loop.
        # Caller can detect by missing keys in the returned dict.
        return {}

    average = compute_7andn_average_metrics(per_scene_rows)
    if save_path is not None and write_average_metrics_json:
        with open(osp.join(save_path, "average_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(average, f, indent=4)

    return {k: float(v) for k, v in average.items() if k != "num_scenes"}

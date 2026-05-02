"""Shared 7Scenes / NRGBD evaluation logic used by both eval_7andN.py and the
training-time periodic eval in train_lora_skip_baseline.py.

Why this module exists: previously eval_7andN.py inlined the per-scene forward,
metric, and aggregation pipeline. The training script could not reuse it, so a
parallel `collect_batch`-based path was used in dev eval which broke under
fastvggt's `forward_from_aggregated_tokens(autocast=False)`. Centralizing the
logic here lets both entry points share the same model forward + metric path
and unifies output schemas (logs.txt / logs_all.txt / average_metrics.json).
"""

from __future__ import annotations

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
    "nc1",
    "nc2",
    "nc",
    "acc_med",
    "comp_med",
    "nc1_med",
    "nc2_med",
    "nc_med",
)

# Output schema that must remain byte-equivalent across (a) live eval_7andN.py
# runs and (b) the one-shot backfill from logs_all.txt. Keys ordered to ease diffing.
AVERAGE_METRICS_SCHEMA_KEYS = (
    "acc",
    "comp",
    "nc1",
    "nc2",
    "nc",
    "acc_med",
    "comp_med",
    "nc1_med",
    "nc2_med",
    "nc_med",
    "num_scenes",
)


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


def compute_7andn_average_metrics(per_scene_rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Average per-scene metrics into the canonical average_metrics.json payload.

    Both eval_7andn (live) and the backfill script feed this exact function so
    the JSON they produce is byte-equivalent. Keys outside AVERAGE_METRICS_SCHEMA_KEYS
    are ignored to keep the schema stable.
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
    """Mirror eval_7andN.py's per-batch device transfer (in place)."""
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
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
) -> Optional[Dict[str, float]]:
    """Run one 7andn scene through model + ICP + accuracy/completion.

    Returns per-scene metrics dict ({acc, comp, nc1, nc2, nc, *_med, num_imgs})
    or None if the scene was skipped for lack of valid points.
    """
    import open3d as o3d  # local import: not all eval consumers need open3d
    from utils import accuracy, completion  # type: ignore  # analysis/eval/utils.py
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if isinstance(batch, dict) and "img" in batch:
        batch["img"] = (batch["img"] + 1.0) / 2.0
    elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
        for view in batch:
            view["img"] = (view["img"] + 1.0) / 2.0
        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)
    else:
        raise RuntimeError("Unexpected 7andn batch layout (expected list[dict] with 'img').")

    torch.cuda.synchronize()
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
    else:
        with torch.cuda.amp.autocast(dtype=eval_amp_dtype):
            preds = model(imgs_tensor)
    torch.cuda.synchronize()
    inference_time_ms = (time.time() - start) * 1000.0

    predictions = preds
    views = batch
    if "pose_enc" in predictions:
        B, S = predictions["pose_enc"].shape[:2]
    elif "world_points" in predictions:
        B, S = predictions["world_points"].shape[:2]
    else:
        raise KeyError("predictions is missing a key to infer sequence length")

    img_hw = (int(imgs_tensor.shape[2]), int(imgs_tensor.shape[3]))
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

    pts_all: List[np.ndarray] = []
    pts_gt_all: List[np.ndarray] = []
    images_all: List[np.ndarray] = []
    masks_all: List[np.ndarray] = []
    conf_all: List[np.ndarray] = []
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

    images_arr = np.concatenate(images_all, axis=0)
    pts_arr = np.concatenate(pts_all, axis=0)
    pts_gt_arr = np.concatenate(pts_gt_all, axis=0)
    masks_arr = np.concatenate(masks_all, axis=0)
    conf_arr = np.concatenate(conf_all, axis=0) if conf_all else None

    scene_id = view["label"][0].rsplit("/", 1)[0]
    scene_num_images = len(batch)
    try:
        scene_infer_times[scene_id].append(float(inference_time_ms))
    except Exception:
        pass

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

    line = (
        f"Idx: {scene_id}, NumImgs: {scene_num_images}, "
        f"Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
        f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
    )
    print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")
    torch.cuda.empty_cache()

    return {
        "scene_id": scene_id,
        "num_imgs": float(scene_num_images),
        "acc": float(acc),
        "comp": float(comp),
        "nc1": float(nc1),
        "nc2": float(nc2),
        "nc": (float(nc1) + float(nc2)) / 2.0,
        "acc_med": float(acc_med),
        "comp_med": float(comp_med),
        "nc1_med": float(nc1_med),
        "nc2_med": float(nc2_med),
        "nc_med": (float(nc1_med) + float(nc2_med)) / 2.0,
    }


def _write_logs_all(save_path: str, scene_infer_times: defaultdict) -> None:
    """Write logs_all.txt = logs.txt + per-scene Time_avg_ms block (kept for backward compat)."""
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
    with open(osp.join(save_path, "logs_all.txt"), "w") as f:
        f.write(to_write + time_block)


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
) -> Dict[str, float]:
    """Run 7Scenes/NRGBD eval for one dataset and return mean metrics dict.

    When ``save_path`` is provided, writes ``logs.txt``, ``logs_all.txt`` and
    ``average_metrics.json`` under it (mirroring eval_7andN.py's layout).
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
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    total_items = len(dataset)
    if max_samples > 0:
        total_items = min(total_items, max_samples)

    per_scene_rows: List[Dict[str, float]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for data_idx in tqdm(range(total_items), desc="eval 7andn", leave=False):
                batch = default_collate([dataset[data_idx]])
                _move_batch_to_device(batch, device)
                row = _evaluate_one_scene_7andn(
                    model=model,
                    batch=batch,
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
                )
                if row is not None:
                    per_scene_rows.append(row)
    finally:
        if was_training:
            model.train()

    if save_path is not None:
        _write_logs_all(save_path, scene_infer_times)

    if not per_scene_rows:
        # Empty mean is preferable to crashing the surrounding training loop.
        # Caller can detect by missing keys in the returned dict.
        return {}

    average = compute_7andn_average_metrics(per_scene_rows)
    if save_path is not None and write_average_metrics_json:
        import json

        with open(osp.join(save_path, "average_metrics.json"), "w") as f:
            json.dump(average, f, indent=4)

    return {k: float(v) for k, v in average.items() if k != "num_scenes"}

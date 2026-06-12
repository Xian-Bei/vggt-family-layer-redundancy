"""Eval-stage 2D/3D visualization export for v3 (view.py layout)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np


def scene_vis_dir(vis_save_root: Path, scene_id: str) -> Path:
    return Path(vis_save_root) / str(scene_id)


def save_scene_plot_2d(
    scene_dir: Path,
    *,
    dataset: str,
    est_w2c: np.ndarray,
    gt_poses: np.ndarray,
    frame_ids: Optional[List[int]] = None,
    first_gt_pose: Optional[np.ndarray] = None,
) -> bool:
    """Write ``plot.png`` trajectory comparison. Returns True if saved."""
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    plot_path = scene_dir / "plot.png"

    ds = str(dataset).lower()
    try:
        if ds == "scannet":
            from analysis.eval.eval_utils import eval_trajectory

            est = np.asarray(est_w2c, dtype=np.float64)
            gt_w2c = np.asarray(gt_poses, dtype=np.float64)
            n = min(len(est), len(gt_w2c))
            if n < 2:
                print(f"[warn] 2D plot skipped (<2 frames): {scene_dir.name}", flush=True)
                return False
            timestamps = frame_ids if frame_ids is not None else list(range(n))
            _, traj_plot, _ = eval_trajectory(
                est[:n], gt_w2c[:n], timestamps[:n], align=True, plot_trajectory=True
            )
        elif ds in ("7scenes", "nrgbd"):
            from analysis.eval.metrics import evaluate_7andn_trajectory_metrics

            fids = frame_ids if frame_ids is not None else list(range(int(gt_poses.shape[0])))
            _, traj_plot = evaluate_7andn_trajectory_metrics(
                est_w2c, gt_poses, fids, plot_trajectory=True
            )
        else:
            print(f"[warn] 2D plot unsupported dataset: {dataset}", flush=True)
            return False

        if traj_plot is None:
            print(f"[warn] 2D plot empty for {scene_dir.name}", flush=True)
            return False
        traj_plot.save(plot_path)
        return True
    except Exception as e:
        print(f"[warn] 2D plot export failed for {scene_dir.name}: {e}", flush=True)
        return False


def export_7andn_points_3d(
    *,
    scene_dir: Path,
    pred_points: np.ndarray,
    pred_colors_u8: np.ndarray,
    pred_conf: Optional[np.ndarray],
    gt_points: np.ndarray,
    image_paths: List[Path],
    extrinsic_cw: np.ndarray,
    intrinsic: Optional[np.ndarray],
    vis_max_points: int = 0,
) -> None:
    """Write 7andn-style vis/pred and vis/gt NPZ (+ cameras.json when poses available)."""
    from analysis.vis.vis_assets import write_cameras_json, write_viser_points_npz

    scene_dir = Path(scene_dir)
    pred_dir = scene_dir / "vis" / "pred"
    gt_dir = scene_dir / "vis" / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    write_viser_points_npz(
        pred_dir / "viser_points.npz",
        points=np.asarray(pred_points, dtype=np.float32),
        colors_uint8=np.asarray(pred_colors_u8, dtype=np.uint8),
        confidence=None if pred_conf is None else np.asarray(pred_conf, dtype=np.float32),
        frame_indices=None,
        max_points=int(vis_max_points),
    )
    write_viser_points_npz(
        gt_dir / "viser_points_gt.npz",
        points=np.asarray(gt_points, dtype=np.float32),
        colors_uint8=np.tile(
            np.array([[160, 160, 160]], dtype=np.uint8),
            (int(np.asarray(gt_points).shape[0]), 1),
        ),
        confidence=None,
        frame_indices=None,
        max_points=int(vis_max_points),
    )
    try:
        write_cameras_json(
            pred_dir / "cameras.json",
            scene_id=scene_dir.name,
            image_paths=image_paths,
            extrinsic_cw=np.asarray(extrinsic_cw, dtype=np.float64),
            intrinsic=intrinsic,
            image_hw=None,
        )
    except Exception as e:
        print(f"[warn] cameras.json export failed for {scene_dir.name}: {e}", flush=True)

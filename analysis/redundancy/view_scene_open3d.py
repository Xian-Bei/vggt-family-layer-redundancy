from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open3D viewer for per-scene visualization assets.")
    p.add_argument("--vis-root", type=Path, required=True, help="Root containing <scene_id>/ assets.")
    p.add_argument("--scene-id", type=str, required=True, help="Scene id (folder name, usually sceneXXXX_XX).")
    p.add_argument("--show-frustum-every", type=int, default=5, help="Show one camera frustum every N frames.")
    p.add_argument("--frustum-scale", type=float, default=0.2, help="Frustum size scale.")
    p.add_argument("--show-trajectory", action="store_true", help="Force draw trajectory from cameras.json.")
    return p


def _load_cameras_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_4x4(extrinsic: List[List[float]]) -> np.ndarray:
    e = np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = e
        return out
    if e.shape == (4, 4):
        return e
    raise ValueError(f"Unexpected extrinsic shape {e.shape}")


def _trajectory_from_centers(centers: np.ndarray) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
    segs = [[i, i + 1] for i in range(max(0, centers.shape[0] - 1))]
    line.lines = o3d.utility.Vector2iVector(segs)
    line.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.4, 0.1]]), (len(segs), 1)))
    return line


def main() -> None:
    args = _build_parser().parse_args()
    scene_dir = args.vis_root / args.scene_id
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene visualization dir: {scene_dir}")

    pcd_path = scene_dir / "pointcloud_rgb.ply"
    cam_path = scene_dir / "cameras.json"
    traj_path = scene_dir / "trajectory_line.ply"
    if not pcd_path.exists() or not cam_path.exists():
        raise SystemExit(f"Expected {pcd_path} and {cam_path}")

    geoms: List[o3d.geometry.Geometry] = []
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    geoms.append(pcd)

    cams = _load_cameras_json(cam_path)
    frames = cams.get("frames", [])
    image_hw = cams.get("image_hw", None)
    if image_hw is None or len(image_hw) != 2:
        image_hw = [518, 518]
    h, w = int(image_hw[0]), int(image_hw[1])

    if traj_path.exists() and not args.show_trajectory:
        geoms.append(o3d.io.read_line_set(str(traj_path)))
    else:
        centers = np.asarray([f["camera_center_world"] for f in frames], dtype=np.float64) if frames else np.zeros((0, 3))
        if centers.shape[0] >= 2:
            geoms.append(_trajectory_from_centers(centers))

    step = max(1, int(args.show_frustum_every))
    for i in range(0, len(frames), step):
        f = frames[i]
        intr = np.asarray(f["intrinsic"], dtype=np.float64)
        ext = _as_4x4(f["extrinsic"])
        cam = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=w,
            view_height_px=h,
            intrinsic=intr,
            extrinsic=ext,
            scale=float(args.frustum_scale),
        )
        cam.paint_uniform_color([0.2, 0.9, 0.2])
        geoms.append(cam)

    print(f"Loaded scene {args.scene_id}: points={len(pcd.points)} cameras={len(frames)}")
    o3d.visualization.draw_geometries(geoms, window_name=f"scene {args.scene_id}")


if __name__ == "__main__":
    main()


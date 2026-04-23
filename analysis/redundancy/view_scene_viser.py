from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import viser
import viser.transforms as viser_tf
from vggt.utils.geometry import closed_form_inverse_se3


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Browser viewer (viser) for Stage1/2 scene visualization assets.")
    p.add_argument("--vis-root", type=Path, required=True, help="Root containing <scene_id>/ assets.")
    p.add_argument("--scene-id", type=str, required=True, help="Scene id folder under --vis-root.")
    p.add_argument("--port", type=int, default=8080, help="Viser server port.")
    p.add_argument("--conf-percent", type=float, default=50.0, help="Initial percentile threshold for confidence.")
    p.add_argument("--show-frustum-every", type=int, default=1, help="Show one frustum every N frames.")
    p.add_argument("--frustum-scale", type=float, default=0.05, help="Frustum scale.")
    p.add_argument("--point-size", type=float, default=0.001, help="Rendered point size.")
    return p


def _load_assets(scene_dir: Path) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_path = scene_dir / "cameras.json"
    npz_path = scene_dir / "viser_points.npz"
    if not cam_path.exists() or not npz_path.exists():
        raise SystemExit(f"Missing {cam_path} or {npz_path}")
    cams = json.loads(cam_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)
    return cams, z["points"], z["colors"], z["confidence"], z["frame_indices"]


def _as_4x4(extrinsic: List[List[float]]) -> np.ndarray:
    e = np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = e
        return out
    if e.shape == (4, 4):
        return e
    raise ValueError(f"Unexpected extrinsic shape {e.shape}")


def _camera_from_world_to_world_from_camera(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert camera-from-world (world->camera) to world-from-camera (camera->world).
    Accepts 3x4 or 4x4, returns 4x4.
    """
    ext4 = _as_4x4(extrinsic.tolist() if isinstance(extrinsic, np.ndarray) else extrinsic)
    c2w = closed_form_inverse_se3(ext4[None, :3, :4])[0]
    return c2w


def _fov_from_intrinsics(intr: np.ndarray, h: int) -> float:
    fy = float(intr[1, 1]) if intr.shape == (3, 3) else 1.0 * h
    fy = max(fy, 1e-6)
    return float(2 * np.arctan2(h / 2.0, fy))


def main() -> None:
    args = _build_parser().parse_args()
    scene_dir = args.vis_root / args.scene_id
    if not scene_dir.exists():
        raise SystemExit(f"Missing scene directory: {scene_dir}")

    cams, points, colors, conf, frame_idx = _load_assets(scene_dir)
    frames = cams.get("frames", [])
    image_hw = cams.get("image_hw", [518, 518])
    h, w = int(image_hw[0]), int(image_hw[1])

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    gui_show_cams = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_conf = server.gui.add_slider("Confidence Percent", min=0.0, max=100.0, step=0.1, initial_value=args.conf_percent)
    frame_options = ["All"] + [str(i) for i in sorted(set(frame_idx.tolist()))]
    gui_frame = server.gui.add_dropdown("Show Points from Frames", options=frame_options, initial_value="All")
    gui_point_size = server.gui.add_slider("Point Size", min=0.0002, max=0.01, step=0.0002, initial_value=args.point_size)

    if points.shape[0] == 0:
        raise SystemExit(f"No points found in {scene_dir / 'viser_points.npz'}")
    center = points.mean(axis=0, keepdims=True)
    points_centered = points - center
    cam_centers = np.asarray([f["camera_center_world"] for f in frames], dtype=np.float64) if frames else np.zeros((0, 3))
    cam_centers = cam_centers - center

    thr = np.percentile(conf, gui_conf.value)
    mask0 = (conf >= thr) & (conf > 1e-6)
    pcd = server.scene.add_point_cloud(
        "pcd",
        points=points_centered[mask0],
        colors=colors[mask0],
        point_size=float(gui_point_size.value),
        point_shape="circle",
    )

    handles: List[Any] = []
    step = max(1, int(args.show_frustum_every))
    for i in range(0, len(frames), step):
        f = frames[i]
        ext_c2w = _camera_from_world_to_world_from_camera(np.asarray(f["extrinsic"], dtype=np.float64))
        ext_c2w = ext_c2w.copy()
        ext_c2w[:3, 3] = ext_c2w[:3, 3] - center.reshape(3)
        intr = np.asarray(f["intrinsic"], dtype=np.float64)
        fov = _fov_from_intrinsics(intr, h)

        frame_pose = viser_tf.SE3.from_matrix(ext_c2w[:3, :4])
        frame_h = server.scene.add_frame(
            f"cam_{i}",
            wxyz=frame_pose.rotation().wxyz,
            position=frame_pose.translation(),
            axes_length=0.04,
            axes_radius=0.0015,
            origin_radius=0.0015,
        )
        frustum = server.scene.add_camera_frustum(
            f"cam_{i}/frustum",
            fov=fov,
            aspect=w / max(h, 1),
            scale=float(args.frustum_scale),
            line_width=1.0,
        )
        handles.extend([frame_h, frustum])

    traj = None
    if cam_centers.shape[0] >= 2:
        segs = np.array([[i, i + 1] for i in range(cam_centers.shape[0] - 1)], dtype=np.int32)
        traj = server.scene.add_line_segments(
            "trajectory",
            points=np.stack([cam_centers[segs[:, 0]], cam_centers[segs[:, 1]]], axis=1),
            colors=np.full((segs.shape[0], 2, 3), np.array([1.0, 0.4, 0.1], dtype=np.float32)),
            line_width=2.0,
        )

    def _update_points() -> None:
        threshold = np.percentile(conf, gui_conf.value)
        conf_mask = (conf >= threshold) & (conf > 1e-6)
        if gui_frame.value == "All":
            fmask = np.ones_like(conf_mask, dtype=bool)
        else:
            fmask = frame_idx == int(gui_frame.value)
        mask = conf_mask & fmask
        pcd.points = points_centered[mask]
        pcd.colors = colors[mask]
        pcd.point_size = float(gui_point_size.value)

    @gui_conf.on_update
    def _(_evt) -> None:
        _update_points()

    @gui_frame.on_update
    def _(_evt) -> None:
        _update_points()

    @gui_point_size.on_update
    def _(_evt) -> None:
        _update_points()

    @gui_show_cams.on_update
    def _(_evt) -> None:
        for hdl in handles:
            hdl.visible = bool(gui_show_cams.value)
        if traj is not None:
            traj.visible = bool(gui_show_cams.value)

    print(f"viser running on http://0.0.0.0:{args.port} for scene={args.scene_id}")
    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()


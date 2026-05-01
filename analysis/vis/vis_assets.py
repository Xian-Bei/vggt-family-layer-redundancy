from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_float_list(x: np.ndarray) -> list:
    return [float(v) for v in np.asarray(x).reshape(-1).tolist()]


def camera_centers_from_extrinsics(extrinsic: np.ndarray) -> np.ndarray:
    """extrinsic: camera-from-world (OpenCV), shape [S,3,4] or [S,4,4]."""
    E = np.asarray(extrinsic, dtype=np.float64)
    if E.ndim != 3 or E.shape[0] <= 0:
        raise ValueError(f"extrinsic must be [S,3,4] or [S,4,4], got {E.shape}")
    if E.shape[1:] == (3, 4):
        R = E[:, :3, :3]
        t = E[:, :3, 3]
    elif E.shape[1:] == (4, 4):
        R = E[:, :3, :3]
        t = E[:, :3, 3]
    else:
        raise ValueError(f"extrinsic must be [S,3,4] or [S,4,4], got {E.shape}")
    centers = -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)
    return centers


def write_cameras_json(
    out_path: Path,
    *,
    scene_id: str,
    image_paths: Sequence[Path],
    extrinsic_cw: np.ndarray,
    intrinsic: Optional[np.ndarray],
    image_hw: Optional[Tuple[int, int]] = None,
) -> None:
    extr = np.asarray(extrinsic_cw, dtype=np.float64)
    intr = None if intrinsic is None else np.asarray(intrinsic, dtype=np.float64)
    centers = camera_centers_from_extrinsics(extr)
    n = min(len(image_paths), int(extr.shape[0]), int(centers.shape[0]))
    frames = []
    for i in range(n):
        frames.append(
            {
                "frame_index": int(i),
                "frame_id": str(image_paths[i].stem),
                "image_path": str(image_paths[i]),
                "extrinsic": extr[i, :3, :4].tolist() if extr.shape[1:] == (4, 4) else extr[i].tolist(),
                "intrinsic": None if intr is None else intr[i].tolist() if intr.ndim == 3 else intr.tolist(),
                "camera_center_world": centers[i].tolist(),
            }
        )
    payload: Dict[str, Any] = {"scene_id": scene_id, "frames": frames}
    if image_hw is not None:
        payload["image_hw"] = [int(image_hw[0]), int(image_hw[1])]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_downsample_indices(n: int, *, max_points: int, seed: int = 33) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n, int(max_points), replace=False).astype(np.int64)


def write_viser_points_npz(
    out_path: Path,
    *,
    points: np.ndarray,
    colors_uint8: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    frame_indices: Optional[np.ndarray] = None,
    max_points: int = 0,
    seed: int = 33,
) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors_uint8, dtype=np.uint8).reshape(-1, 3)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"points N={pts.shape[0]} != colors N={cols.shape[0]}")
    conf = (
        np.ones((pts.shape[0],), dtype=np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32).reshape(-1)
    )
    if conf.shape[0] != pts.shape[0]:
        raise ValueError(f"confidence N={conf.shape[0]} != points N={pts.shape[0]}")
    frames = (
        np.zeros((pts.shape[0],), dtype=np.int32)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    )
    if frames.shape[0] != pts.shape[0]:
        raise ValueError(f"frame_indices N={frames.shape[0]} != points N={pts.shape[0]}")

    keep = _maybe_downsample_indices(int(pts.shape[0]), max_points=int(max_points), seed=int(seed))
    np.savez_compressed(
        out_path,
        points=pts[keep],
        colors=cols[keep],
        confidence=conf[keep],
        frame_indices=frames[keep],
    )


"""Compact, versioned storage for sparse camera Z-depth samples.

The driving-dataset extractors store projected LiDAR samples in the original
camera pixel coordinate system.  Keeping floating-point ``(u, v, z)`` samples
instead of a mostly-empty dense image makes the files small and lets evaluation
apply exactly the same resize/crop transform as the RGB input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


SPARSE_DEPTH_FORMAT_VERSION = 1
SPARSE_DEPTH_TYPE = "camera_z"


@dataclass(frozen=True)
class SparseDepth:
    u: np.ndarray
    v: np.ndarray
    depth: np.ndarray
    image_width: int
    image_height: int
    source: str
    format_version: int = SPARSE_DEPTH_FORMAT_VERSION
    depth_type: str = SPARSE_DEPTH_TYPE

    @property
    def count(self) -> int:
        return int(self.depth.size)


def _clean_samples(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    if not (u.shape == v.shape == depth.shape):
        raise ValueError(f"Sparse depth arrays must have equal shape: {u.shape}, {v.shape}, {depth.shape}")
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(depth)
        & (depth > float(min_depth))
        & (depth < float(max_depth))
        & (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )
    return u[valid], v[valid], depth[valid]


def save_sparse_depth(
    path: Path | str,
    *,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    image_width: int,
    image_height: int,
    source: str,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> SparseDepth:
    """Validate and save compressed original-image ``(u, v, camera_z)`` samples."""
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    u, v, depth = _clean_samples(
        u,
        v,
        depth,
        image_width=width,
        image_height=height,
        min_depth=float(min_depth),
        max_depth=float(max_depth),
    )
    item = SparseDepth(
        u=u,
        v=v,
        depth=depth,
        image_width=width,
        image_height=height,
        source=str(source),
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        u=item.u,
        v=item.v,
        depth=item.depth,
        image_width=np.int32(item.image_width),
        image_height=np.int32(item.image_height),
        source=np.asarray(item.source),
        format_version=np.int32(item.format_version),
        depth_type=np.asarray(item.depth_type),
    )
    return item


def load_sparse_depth(path: Path | str) -> SparseDepth:
    src = Path(path)
    with np.load(src, allow_pickle=False) as raw:
        required = {"u", "v", "depth", "image_width", "image_height", "source", "format_version", "depth_type"}
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(f"{src}: missing sparse-depth fields: {missing}")
        version = int(raw["format_version"])
        depth_type = str(raw["depth_type"].item())
        if version != SPARSE_DEPTH_FORMAT_VERSION:
            raise ValueError(f"{src}: unsupported format_version={version}")
        if depth_type != SPARSE_DEPTH_TYPE:
            raise ValueError(f"{src}: expected depth_type={SPARSE_DEPTH_TYPE!r}, got {depth_type!r}")
        item = SparseDepth(
            u=np.asarray(raw["u"], dtype=np.float32).reshape(-1),
            v=np.asarray(raw["v"], dtype=np.float32).reshape(-1),
            depth=np.asarray(raw["depth"], dtype=np.float32).reshape(-1),
            image_width=int(raw["image_width"]),
            image_height=int(raw["image_height"]),
            source=str(raw["source"].item()),
            format_version=version,
            depth_type=depth_type,
        )
    _validate_loaded(item, src)
    return item


def _validate_loaded(item: SparseDepth, path: Path | None = None) -> None:
    label = str(path) if path is not None else "sparse depth"
    if not (item.u.shape == item.v.shape == item.depth.shape):
        raise ValueError(f"{label}: array shape mismatch")
    if item.image_width <= 0 or item.image_height <= 0:
        raise ValueError(f"{label}: invalid image size")
    if item.count == 0:
        raise ValueError(f"{label}: no valid projected samples")
    if not (np.isfinite(item.u).all() and np.isfinite(item.v).all() and np.isfinite(item.depth).all()):
        raise ValueError(f"{label}: non-finite samples")
    if not (
        (item.u >= 0).all()
        and (item.u < item.image_width).all()
        and (item.v >= 0).all()
        and (item.v < item.image_height).all()
        and (item.depth > 0).all()
    ):
        raise ValueError(f"{label}: samples outside image/depth bounds")


def rasterize_sparse_depth(item: SparseDepth, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Rasterize with a nearest-pixel z-buffer; intended for validation previews."""
    out_w = int(width or item.image_width)
    out_h = int(height or item.image_height)
    x = np.rint(item.u * (out_w / item.image_width)).astype(np.int64)
    y = np.rint(item.v * (out_h / item.image_height)).astype(np.int64)
    valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    flat = np.full(out_h * out_w, np.inf, dtype=np.float32)
    np.minimum.at(flat, y[valid] * out_w + x[valid], item.depth[valid])
    depth = flat.reshape(out_h, out_w)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def sparse_depth_stats(item: SparseDepth) -> Dict[str, Any]:
    unique_pixels = np.unique(
        np.rint(item.v).astype(np.int64) * item.image_width + np.rint(item.u).astype(np.int64)
    ).size
    return {
        "samples": item.count,
        "unique_pixels": int(unique_pixels),
        "coverage": float(unique_pixels) / float(item.image_width * item.image_height),
        "min_depth": float(item.depth.min()),
        "max_depth": float(item.depth.max()),
        "mean_depth": float(item.depth.mean()),
        "image_width": item.image_width,
        "image_height": item.image_height,
        "source": item.source,
    }

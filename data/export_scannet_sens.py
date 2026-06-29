#!/usr/bin/env python3
"""
Export ScanNet raw ``*.sens`` streams to ``color/``, ``depth/``, ``pose/``, and
``intrinsic/`` directories.

This script is intended to be run from the repository root and matches the
current repository layout:

- raw ScanNet scenes live under ``data/scans/<scene_id>/``
- each scene contains ``<scene_id>.sens``
- exported outputs are written into a scene-based layout compatible with the
  current VGGT / FastVGGT evaluation pipeline

Exported files:

- ``<scene>/color/*.jpg``
- ``<scene>/depth/*.png``
- ``<scene>/pose/*.txt``
- ``<scene>/intrinsic/*.txt``

Usage examples::

    # Default behavior: read from data/scans and export to data/process_scannet
    python data/export_scannet_sens.py

    # Export directly into data/scans
    python data/export_scannet_sens.py \
        --scans-root data/scans \
        --output-root data/scans \
        --depth --intrinsics

    # Restrict export to a subset of scenes
    python data/export_scannet_sens.py --scenes scene0000_00,scene0013_02

    # Use a scene list file (one scene id per line)
    python data/export_scannet_sens.py --scene-list /path/to/scannet_subset.txt

Notes:

- ``--frame-skip 1`` exports every frame; larger values subsample frames.
- ``--skip-existing`` only checks for ``color/0.jpg`` and ``pose/0.txt``.
- Depth is exported as 16-bit PNG in the raw ScanNet depth units stored in the
  ``.sens`` stream.
- This file is a Python 3 port of ScanNet ``SensReader/python/SensorData.py``
  plus ``reader.py``.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
from PIL import Image

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        if compression_type == "raw_ushort":
            return self.depth_data
        raise NotImplementedError(f"Unsupported depth compression: {compression_type!r}")

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type == "jpeg":
            return self._decompress_color_jpeg()
        if compression_type == "png":
            return self._decompress_color_png()
        if compression_type == "raw":
            raise NotImplementedError(
                "Raw color export needs width/height; use official C++ toolkit or extend this script."
            )
        raise NotImplementedError(f"Unsupported color compression: {compression_type!r}")

    def _decompress_color_jpeg(self) -> np.ndarray:
        img = Image.open(BytesIO(self.color_data)).convert("RGB")
        return np.asarray(img)

    def _decompress_color_png(self) -> np.ndarray:
        img = Image.open(BytesIO(self.color_data)).convert("RGB")
        return np.asarray(img)


class SensorData:
    def __init__(self, filename: Path) -> None:
        self.version = 4
        self.load(filename)

    def load(self, filename: Path) -> None:
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            if version != self.version:
                raise ValueError(f"Expected sens file version {self.version}, got {version}")
            strlen = struct.unpack("Q", f.read(8))[0]
            raw_name = f.read(strlen)
            self.sensor_name = raw_name.decode("utf-8", errors="replace")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames: List[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    @staticmethod
    def save_mat_to_file(matrix: np.ndarray, filename: Path) -> None:
        with open(filename, "w", encoding="utf-8") as fh:
            for line in matrix:
                np.savetxt(fh, line[np.newaxis], fmt="%f")

    def export_depth_images(self, output_path: Path, image_size=None, frame_skip: int = 1) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        n = (len(self.frames) + frame_skip - 1) // frame_skip
        print(f"  exporting {n} depth frames -> {output_path}")
        for fi in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[fi].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                import cv2

                depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            out_file = output_path / f"{fi}.png"
            Image.fromarray(depth, mode="I;16").save(out_file)

    def export_color_images(self, output_path: Path, image_size=None, frame_skip: int = 1) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        n = (len(self.frames) + frame_skip - 1) // frame_skip
        print(f"  exporting {n} color frames -> {output_path}")
        for fi in range(0, len(self.frames), frame_skip):
            color = self.frames[fi].decompress_color(self.color_compression_type)
            if image_size is not None:
                import cv2

                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            out_file = output_path / f"{fi}.jpg"
            Image.fromarray(color).save(out_file, "JPEG", quality=95)

    def export_poses(self, output_path: Path, frame_skip: int = 1) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        n = (len(self.frames) + frame_skip - 1) // frame_skip
        print(f"  exporting {n} camera poses -> {output_path}")
        for fi in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(self.frames[fi].camera_to_world, output_path / f"{fi}.txt")

    def export_intrinsics(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  exporting intrinsics -> {output_path}")
        self.save_mat_to_file(self.intrinsic_color, output_path / "intrinsic_color.txt")
        self.save_mat_to_file(self.extrinsic_color, output_path / "extrinsic_color.txt")
        self.save_mat_to_file(self.intrinsic_depth, output_path / "intrinsic_depth.txt")
        self.save_mat_to_file(self.extrinsic_depth, output_path / "extrinsic_depth.txt")


def _load_scene_ids(path: Path) -> Set[str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def _find_sens_files(scans_root: Path) -> List[tuple[Path, str]]:
    """Return list of (path_to_sens, scene_id)."""
    out: List[tuple[Path, str]] = []
    if not scans_root.is_dir():
        return out
    for scene_dir in sorted(p for p in scans_root.iterdir() if p.is_dir()):
        sid = scene_dir.name
        sens = scene_dir / f"{sid}.sens"
        if sens.is_file():
            out.append((sens, sid))
    return out


def _parse_scene_filter(scene_list: Optional[Path], scenes: Optional[str]) -> Optional[Set[str]]:
    if scene_list is not None:
        return _load_scene_ids(scene_list)
    if scenes:
        return {s.strip() for s in scenes.split(",") if s.strip()}
    return None


def main(argv: Optional[Iterable[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_scans = repo_root / "data" / "scans"
    default_out = repo_root / "data" / "process_scannet"

    p = argparse.ArgumentParser(description="Export ScanNet .sens to color/pose (FastVGGT-compatible layout).")
    p.add_argument("--scans-root", type=Path, default=default_scans, help="Directory containing sceneXXXX_XX folders.")
    p.add_argument("--output-root", type=Path, default=default_out, help="Parent dir for exported scene folders.")
    p.add_argument(
        "--scene-list",
        type=Path,
        default=None,
        help="Text file with one scene id per line (e.g. scannet_50.yaml content as .txt).",
    )
    p.add_argument("--scenes", type=str, default=None, help="Comma-separated scene ids (overrides discovery subset).")
    p.add_argument("--frame-skip", type=int, default=1, help="Export every k-th frame (1 = all).")
    p.add_argument("--no-color", action="store_true", help="Skip RGB export (default: export color).")
    p.add_argument("--no-pose", action="store_true", help="Skip pose export (default: export pose).")
    p.add_argument("--depth", action="store_true", help="Export 16-bit depth PNGs (adds opencv-python if resize used).")
    p.add_argument("--intrinsics", action="store_true", help="Export intrinsic/*.txt matrices.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scene if output scene dir already has color/0.jpg and pose/0.txt.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    scene_filter = _parse_scene_filter(args.scene_list, args.scenes)
    pairs = _find_sens_files(args.scans_root)
    if scene_filter is not None:
        pairs = [(s, sid) for s, sid in pairs if sid in scene_filter]

    if not pairs:
        print(f"No .sens files found under {args.scans_root}", file=sys.stderr)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Scans root:    {args.scans_root}")
    print(f"Output root:   {args.output_root}")
    print(f"Scenes to run: {len(pairs)}")

    ok = 0
    for sens_path, sid in pairs:
        scene_out = args.output_root / sid
        if args.skip_existing:
            c0 = scene_out / "color" / "0.jpg"
            p0 = scene_out / "pose" / "0.txt"
            if c0.is_file() and p0.is_file():
                print(f"[skip] {sid} (already has color/0.jpg and pose/0.txt)")
                ok += 1
                continue

        print(f"== {sid} <- {sens_path.name}")
        try:
            sd = SensorData(sens_path)
        except Exception as e:
            print(f"  ERROR loading sens: {e}", file=sys.stderr)
            continue

        try:
            if not args.no_color:
                sd.export_color_images(scene_out / "color", frame_skip=args.frame_skip)
            if not args.no_pose:
                sd.export_poses(scene_out / "pose", frame_skip=args.frame_skip)
            if args.depth:
                sd.export_depth_images(scene_out / "depth", frame_skip=args.frame_skip)
            if args.intrinsics:
                sd.export_intrinsics(scene_out / "intrinsic")
        except NotImplementedError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  ERROR export: {e}", file=sys.stderr)
            continue

        ok += 1

    print(f"Done. Successfully processed {ok}/{len(pairs)} scenes.")
    return 0 if ok == len(pairs) else 2


if __name__ == "__main__":
    raise SystemExit(main())

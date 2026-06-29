from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import torch

from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.contracts import EvalSceneContext
from analysis.datasets.contracts import DatasetAdaptor, EvalSampleBatch, PreparedSkipEvalScene
from analysis.datasets.frame_selection import cap_sequence
from analysis.eval.metrics_core import (
    compute_depth_metrics,
    compute_point_metrics,
    merge_scene_metrics,
    require_depth_stack_shapes,
)
from analysis.eval.vis_export_v3 import export_7andn_points_3d, save_scene_plot_2d, scene_vis_dir
from analysis.schema.v3 import DatasetEvalProfile, DepthEvalProfile, GeometryEvalProfile

_VALID_SPLITS = ("train", "val")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class NuScenesAdaptor(DatasetAdaptor):
    name = "nuscenes"
    camera_name = "CAM_FRONT"
    valid_splits = _VALID_SPLITS
    split_aliases: Dict[str, str] = {}

    def validate_split(self, split: str | None = None) -> str:
        if not split:
            raise ValueError(f"{self.name} requires split in {self.valid_splits}")
        split_norm = str(split).strip().lower()
        split_norm = self.split_aliases.get(split_norm, split_norm)
        if split_norm not in self.valid_splits:
            expected = " or ".join(self.valid_splits)
            raise ValueError(f"Invalid {self.name} split: {split!r} (expected {expected})")
        return split_norm

    def output_tag(self, split: str | None = None) -> str:
        return f"{self.name}_{self.validate_split(split)}"

    def count_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        split: str | None = None,
    ) -> int:
        n = len(self._scene_names_for_split(Path(data_root), self.validate_split(split)))
        return n if int(max_scenes) <= 0 else min(n, int(max_scenes))

    def eval_profile(self) -> DatasetEvalProfile:
        return DatasetEvalProfile(
            dataset=self.name,
            geometry=GeometryEvalProfile(dist_clip=1.0, use_icp=True, max_points=100_000),
            depth=DepthEvalProfile(max_depth=80.0, conf_thresh=1.0, crop_mode="none"),
            kf_every=1,
            size=518,
        )

    def iter_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        max_frames: int,
        split: str | None = None,
    ) -> Iterator[EvalSampleBatch]:
        split_norm = self.validate_split(split)
        names = self._scene_names_for_split(Path(data_root), split_norm)
        if int(max_scenes) > 0:
            names = names[: int(max_scenes)]
        for scene_id in names:
            yield self.load_scene_for_cache(data_root, scene_id, max_frames, split=split_norm)

    def load_scene_for_cache(
        self,
        data_root: Path,
        scene_id: str,
        max_frames: int,
        *,
        split: str | None = None,
    ) -> EvalSampleBatch:
        split_norm = self.validate_split(split)
        scene_dir = self._scene_dir(Path(data_root), str(scene_id), split_norm)
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"{self.name} scene not found: {scene_dir}")

        image_paths = _selected_front_image_paths(scene_dir, camera_name=self.camera_name, max_frames=max_frames)
        if not image_paths:
            raise FileNotFoundError(f"No {self.camera_name} images found under {scene_dir}")
        frame_ids = [int(p.stem) for p in image_paths]
        pose_paths = [scene_dir / "cam2world" / self.camera_name / f"{fid:03d}.txt" for fid in frame_ids]
        lidar_paths = [scene_dir / "lidar" / f"{fid:03d}.bin" for fid in frame_ids]
        intrinsic_path = scene_dir / "intrinsic" / f"{self.camera_name}.txt"
        depth_paths = [scene_dir / "depth_points" / self.camera_name / f"{fid:03d}.npz" for fid in frame_ids]
        missing_depth = [p for p in depth_paths if not p.is_file()]
        if missing_depth:
            raise FileNotFoundError(
                f"Missing sparse camera-Z depth for {self.name} scene {scene_id}: {missing_depth[0]}. "
                "Re-run the official dataset extractor."
            )

        images = None
        return EvalSampleBatch(
            scene_id=str(scene_id),
            images=images,
            frame_ids=frame_ids,
            meta={
                "scene_dir": str(scene_dir),
                "split": split_norm,
                "image_paths": [str(p.resolve()) for p in image_paths],
                "frame_ids": [f"{fid:03d}" for fid in frame_ids],
                "pose_paths": [str(p) for p in pose_paths],
                "lidar_paths": [str(p) for p in lidar_paths],
                "intrinsic_path": str(intrinsic_path),
                "depth_paths": [str(p) for p in depth_paths],
                "max_frames": int(max_frames),
                "camera_name": self.camera_name,
            },
        )

    def _scene_names_for_split(self, data_root: Path, split: str) -> List[str]:
        return _scene_names(data_root, split)

    def _scene_dir(self, data_root: Path, scene_id: str, split: str) -> Path:
        del split
        return data_root / str(scene_id)

    def evaluate_scene(
        self,
        backbone_adaptor: Any,
        model: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        from analysis.datasets import eval_impl

        device = ctx.device
        image_paths = [Path(p) for p in batch.meta.get("image_paths", [])]
        if not image_paths:
            raise ValueError(f"{self.name} {batch.scene_id}: missing image_paths in batch meta")
        images = backbone_adaptor.load_eval_images(image_paths)
        images_b = images.unsqueeze(0).float().to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device.index)
        t0 = time.perf_counter()
        ac_kw = autocast_device_dtype(device)
        with torch.no_grad(), torch.autocast(**ac_kw):
            bundle = backbone_adaptor.forward_predictions(model, images_b)
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

        if not ctx.skip_depth and isinstance(bundle.depth, torch.Tensor):
            pred_depth = bundle.depth.detach().float()
            if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth.squeeze(-1)
            if pred_depth.ndim != 4 or pred_depth.shape[0] != 1:
                raise ValueError(f"Unexpected {self.name} predicted depth shape: {tuple(pred_depth.shape)}")
            pred_stack = pred_depth[0].cpu().numpy().astype(np.float32, copy=False)
            depth_paths = [Path(p) for p in batch.meta.get("depth_paths", [])]
            if len(depth_paths) != int(pred_stack.shape[0]):
                raise ValueError(
                    f"{self.name} {batch.scene_id}: prediction/GT frame count mismatch "
                    f"{pred_stack.shape[0]} vs {len(depth_paths)}"
                )
            gt_stack, valid_frames = backbone_adaptor.load_driving_depth_stack(
                depth_paths,
                output_hw=(int(pred_stack.shape[1]), int(pred_stack.shape[2])),
            )
            require_depth_stack_shapes(
                pred_stack,
                gt_stack,
                context=f"{self.name} {batch.scene_id}",
            )
            metrics.update(compute_depth_metrics(pred_stack, gt_stack, ctx.profile.depth))
            metrics["depth_valid_frames"] = float(valid_frames)

        gt_c2w = _load_pose_stack(batch.meta.get("pose_paths", []))
        gt_w2c = np.linalg.inv(gt_c2w) if gt_c2w.shape[0] else np.zeros((0, 4, 4), dtype=np.float64)
        frame_ids = [int(x) for x in batch.frame_ids]
        est_w2c = None
        est_w2c_eval = None
        gt_w2c_eval = None
        gt_c2w_eval = None
        frame_ids_eval: List[int] = []
        sim3_eval = None

        poses = bundle.camera_poses
        if poses is not None and gt_w2c.shape[0] > 0:
            est = poses.detach().float().cpu().numpy()
            if est.ndim == 4 and est.shape[0] == 1:
                est = est[0]
            est_w2c = np.linalg.inv(est)
            n = min(int(est_w2c.shape[0]), int(gt_w2c.shape[0]), len(frame_ids))
            if n >= 2:
                est_w2c_eval = est_w2c[:n]
                gt_w2c_eval = gt_w2c[:n]
                gt_c2w_eval = gt_c2w[:n]
                frame_ids_eval = frame_ids[:n]
                pose_metrics, sim3_eval = eval_impl._trajectory_alignment(
                    est_w2c_eval,
                    gt_w2c_eval,
                    frame_ids_eval,
                )
                metrics.update(pose_metrics)
            else:
                metrics.update(eval_impl._nan_pose_metrics())

        pred_points = None
        if bundle.points is not None:
            pred_points = bundle.points.detach().float().cpu().numpy().reshape(-1, 3)
        gt_points = _load_lidar_points(batch.meta.get("lidar_paths", []))
        if pred_points is not None and gt_points.shape[0] > 0 and sim3_eval is not None and len(frame_ids_eval) >= 3:
            aligned = eval_impl._transform_points_sim3(pred_points, sim3_eval)
            if aligned is not None:
                metrics.update(compute_point_metrics(aligned, gt_points, ctx.profile.geometry, use_icp=False))
            else:
                metrics.update(eval_impl._nan_point_metrics())
        elif pred_points is not None and gt_points.shape[0] > 0:
            metrics.update(eval_impl._nan_point_metrics())

        if (
            ctx.export_vis_2d
            and est_w2c_eval is not None
            and gt_c2w_eval is not None
            and ctx.vis_save_root is not None
        ):
            save_scene_plot_2d(
                scene_vis_dir(ctx.vis_save_root, batch.scene_id),
                dataset=self.name,
                est_w2c=est_w2c_eval,
                gt_poses=gt_c2w_eval,
                frame_ids=frame_ids_eval,
            )

        if ctx.export_vis_3d and ctx.vis_save_root is not None and pred_points is not None and gt_points.shape[0] > 0:
            try:
                intrinsic = _load_intrinsic(batch.meta.get("intrinsic_path"))
                export_7andn_points_3d(
                    scene_dir=scene_vis_dir(ctx.vis_save_root, batch.scene_id),
                    pred_points=pred_points,
                    pred_colors_u8=np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (pred_points.shape[0], 1)),
                    pred_conf=None,
                    gt_points=gt_points,
                    image_paths=[Path(p) for p in batch.meta.get("image_paths", [])],
                    extrinsic_cw=np.linalg.inv(est_w2c) if est_w2c is not None else np.zeros((0, 4, 4)),
                    intrinsic=intrinsic,
                    vis_max_points=int(ctx.vis_max_points),
                )
            except Exception as e:
                print(f"[warn] {self.name} 3D vis failed for {batch.scene_id}: {e}", flush=True)

        return merge_scene_metrics(metrics)

    def prepare_skip_eval_scene(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> PreparedSkipEvalScene:
        image_paths = [Path(p) for p in batch.meta.get("image_paths", [])]
        if not image_paths:
            raise ValueError(f"{self.name} {batch.scene_id}: missing image_paths in batch meta")
        images = backbone_adaptor.load_eval_images(image_paths)
        images_b = images.unsqueeze(0).float().to(ctx.device)
        depth_paths = [Path(p) for p in batch.meta.get("depth_paths", [])]
        gt_depth = None
        valid_depth_frames = 0
        if not ctx.skip_depth and depth_paths:
            gt_depth, valid_depth_frames = backbone_adaptor.load_driving_depth_stack(
                depth_paths,
                output_hw=(int(images_b.shape[-2]), int(images_b.shape[-1])),
            )
        gt_c2w = _load_pose_stack(batch.meta.get("pose_paths", []))
        gt_w2c = np.linalg.inv(gt_c2w) if gt_c2w.shape[0] else np.zeros((0, 4, 4), dtype=np.float64)
        return PreparedSkipEvalScene(
            scene_id=str(batch.scene_id),
            images=images_b,
            state={
                "gt_depth": gt_depth,
                "valid_depth_frames": int(valid_depth_frames),
                "gt_c2w": gt_c2w,
                "gt_w2c": gt_w2c,
                "gt_points": _load_lidar_points(batch.meta.get("lidar_paths", [])),
                "frame_ids": [int(x) for x in batch.frame_ids],
            },
        )

    def score_skip_eval_prediction(
        self,
        backbone_adaptor: Any,
        prediction: Any,
        prepared: PreparedSkipEvalScene,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        from analysis.datasets import eval_impl

        del backbone_adaptor
        metrics: Dict[str, float] = {}
        if not ctx.skip_depth and isinstance(prediction.depth, torch.Tensor):
            pred_depth = prediction.depth.detach().float()
            if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth.squeeze(-1)
            if pred_depth.ndim != 4 or pred_depth.shape[0] != 1:
                raise ValueError(f"Unexpected {self.name} predicted depth shape: {tuple(pred_depth.shape)}")
            pred_stack = pred_depth[0].cpu().numpy().astype(np.float32, copy=False)
            gt_stack = prepared.state.get("gt_depth")
            if gt_stack is not None:
                if tuple(gt_stack.shape) != tuple(pred_stack.shape):
                    raise ValueError(
                        f"{self.name} {prepared.scene_id}: prediction/GT depth shape mismatch "
                        f"{pred_stack.shape} vs {gt_stack.shape}"
                    )
                metrics.update(compute_depth_metrics(pred_stack, gt_stack, ctx.profile.depth))
                metrics["depth_valid_frames"] = float(prepared.state["valid_depth_frames"])

        gt_w2c = prepared.state["gt_w2c"]
        frame_ids = prepared.state["frame_ids"]
        sim3_eval = None
        frame_ids_eval: List[int] = []
        poses = prediction.camera_poses
        if poses is not None and gt_w2c.shape[0] > 0:
            est = poses.detach().float().cpu().numpy()
            if est.ndim == 4 and est.shape[0] == 1:
                est = est[0]
            est_w2c = np.linalg.inv(est)
            n = min(int(est_w2c.shape[0]), int(gt_w2c.shape[0]), len(frame_ids))
            if n >= 2:
                frame_ids_eval = frame_ids[:n]
                pose_metrics, sim3_eval = eval_impl._trajectory_alignment(
                    est_w2c[:n], gt_w2c[:n], frame_ids_eval
                )
                metrics.update(pose_metrics)
            else:
                metrics.update(eval_impl._nan_pose_metrics())

        pred_points = None
        if prediction.points is not None:
            pred_points = prediction.points.detach().float().cpu().numpy().reshape(-1, 3)
        gt_points = prepared.state["gt_points"]
        if pred_points is not None and gt_points.shape[0] > 0 and sim3_eval is not None and len(frame_ids_eval) >= 3:
            aligned = eval_impl._transform_points_sim3(pred_points, sim3_eval)
            if aligned is not None:
                metrics.update(compute_point_metrics(aligned, gt_points, ctx.profile.geometry, use_icp=False))
            else:
                metrics.update(eval_impl._nan_point_metrics())
        elif pred_points is not None and gt_points.shape[0] > 0:
            metrics.update(eval_impl._nan_point_metrics())
        return merge_scene_metrics(metrics)


def _scene_names(data_root: Path, split: str) -> List[str]:
    manifest = Path(data_root) / "selected_scenes.json"
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        names = raw.get("splits", {}).get(split, [])
        if not isinstance(names, list):
            raise ValueError(f"Invalid nuScenes selected_scenes.json split entry: {split}")
        return [str(x) for x in names if (Path(data_root) / str(x)).is_dir()]
    return sorted(p.name for p in Path(data_root).iterdir() if p.is_dir() and p.name.startswith("scene-"))


def _selected_front_image_paths(scene_dir: Path, *, camera_name: str, max_frames: int) -> List[Path]:
    image_dir = Path(scene_dir) / "image" / camera_name
    if not image_dir.is_dir():
        return []
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return cap_sequence(paths, max_frames)


def _load_pose_stack(paths: Any) -> np.ndarray:
    mats = []
    for p_raw in paths or []:
        p = Path(str(p_raw))
        if p.is_file():
            mat = np.loadtxt(p, dtype=np.float64)
            if mat.shape == (4, 4) and np.isfinite(mat).all():
                mats.append(mat)
    return np.stack(mats, axis=0) if mats else np.zeros((0, 4, 4), dtype=np.float64)


def _load_lidar_points(paths: Any) -> np.ndarray:
    chunks = []
    for p_raw in paths or []:
        p = Path(str(p_raw))
        if not p.is_file():
            continue
        arr = np.fromfile(p, dtype=np.float32)
        if arr.size >= 3:
            chunks.append(arr.reshape(-1, 3))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float32)


def _load_intrinsic(path_raw: Any) -> np.ndarray | None:
    if not path_raw:
        return None
    p = Path(str(path_raw))
    if not p.is_file():
        return None
    arr = np.loadtxt(p, dtype=np.float64)
    return arr if arr.shape == (3, 3) else None

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import torch

from analysis.backbones.contracts import EvalSceneContext
from analysis.datasets.contracts import DatasetAdaptor, EvalSampleBatch, PreparedSkipEvalScene
from analysis.eval.scannet_eval import (
    discover_scannet_scenes,
    load_scene_images,
    select_scene_image_paths_fastvggt,
)
from analysis.schema.v3 import DatasetEvalProfile, GeometryEvalProfile


class ScanNetAdaptor(DatasetAdaptor):
    name = "scannet"

    def count_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        split: str | None = None,
    ) -> int:
        self.validate_split(split)
        n = len(discover_scannet_scenes(data_root))
        return n if max_scenes <= 0 else min(n, int(max_scenes))

    def eval_profile(self) -> DatasetEvalProfile:
        return DatasetEvalProfile(
            dataset=self.name,
            geometry=GeometryEvalProfile(dist_clip=0.5, use_icp=True, max_points=100_000),
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
        self.validate_split(split)
        scenes = discover_scannet_scenes(data_root)
        if max_scenes > 0:
            scenes = scenes[:max_scenes]
        for scene in scenes:
            yield self.load_scene_for_cache(data_root, scene.scene_id, max_frames, split=split)

    def load_scene_for_cache(
        self,
        data_root: Path,
        scene_id: str,
        max_frames: int,
        *,
        split: str | None = None,
    ) -> EvalSampleBatch:
        self.validate_split(split)
        scenes = {s.scene_id: s for s in discover_scannet_scenes(data_root)}
        if scene_id not in scenes:
            raise FileNotFoundError(scene_id)
        scene = select_scene_image_paths_fastvggt(scenes[scene_id], max_frames)
        image_paths = [str(p.resolve()) for p in scene.image_paths]
        frame_ids = [p.stem for p in scene.image_paths]
        return EvalSampleBatch(
            scene_id=scene_id,
            images=None,
            frame_ids=[int(x) for x in frame_ids if str(x).isdigit()],
            meta={
                "scene_dir": str(scene.scene_dir),
                "image_paths": image_paths,
                "frame_ids": frame_ids,
                "max_frames": int(max_frames),
            },
        )

    def prepare_prediction_images(
        self,
        batch: EvalSampleBatch,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(batch.images, torch.Tensor):
            return batch.images.unsqueeze(0).to(device)
        images_np = np.stack(batch.images)
        return torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    def evaluate_scene(
        self,
        backbone_adaptor: Any,
        model: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        from analysis.datasets import eval_impl

        return eval_impl.eval_scannet_points_adapter(
            backbone_adaptor,
            model,
            batch,
            ctx.device,
            ctx.profile,
            ctx,
        )

    def prepare_skip_eval_scene(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> PreparedSkipEvalScene:
        from analysis.datasets import eval_impl

        return eval_impl.prepare_scannet_skip_scene(
            backbone_adaptor, batch, ctx.device, ctx.profile
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
        return eval_impl.score_scannet_skip_prediction(
            prediction, prepared, ctx.profile, ctx
        )

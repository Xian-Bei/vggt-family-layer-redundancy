from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np

from analysis.datasets.contracts import DatasetPlugin, EvalSampleBatch
from analysis.datasets.registry import register
from analysis.schema.v3 import DatasetEvalProfile, DepthEvalProfile, GeometryEvalProfile
from analysis.eval.scannet_eval import SceneData, discover_scannet_scenes, load_scene_images, select_scene_image_paths_fastvggt


class ScanNetPlugin(DatasetPlugin):
    name = "scannet"

    def eval_profile(self) -> DatasetEvalProfile:
        return DatasetEvalProfile(
            dataset=self.name,
            geometry=GeometryEvalProfile(dist_clip=0.5, use_icp=True, max_points=100_000),
            depth=DepthEvalProfile(max_depth=10.0, conf_thresh=1.0, crop_mode="none"),
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
        scenes = {s.scene_id: s for s in discover_scannet_scenes(data_root)}
        if scene_id not in scenes:
            raise FileNotFoundError(scene_id)
        scene = select_scene_image_paths_fastvggt(scenes[scene_id], max_frames)
        images = load_scene_images(scene, mode="crop")
        image_paths = [str(p.resolve()) for p in scene.image_paths]
        frame_ids = [p.stem for p in scene.image_paths]
        return EvalSampleBatch(
            scene_id=scene_id,
            images=images,
            frame_ids=[int(x) for x in frame_ids if str(x).isdigit()],
            meta={
                "scene_dir": str(scene.scene_dir),
                "image_paths": image_paths,
                "frame_ids": frame_ids,
                "max_frames": int(max_frames),
            },
        )


register(ScanNetPlugin())

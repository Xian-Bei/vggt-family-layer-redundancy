from __future__ import annotations

from pathlib import Path
from typing import Iterator

from analysis.datasets.contracts import DatasetPlugin, EvalSampleBatch
from analysis.datasets.plugins._7andn_common import iter_7andn_scenes, load_7andn_scene
from analysis.datasets.registry import register
from analysis.schema.v3 import DatasetEvalProfile, DepthEvalProfile, GeometryEvalProfile


class NRGBDPlugin(DatasetPlugin):
    name = "nrgbd"

    def eval_profile(self) -> DatasetEvalProfile:
        return DatasetEvalProfile(
            dataset=self.name,
            geometry=GeometryEvalProfile(dist_clip=0.1, use_icp=True, max_points=999_999),
            depth=DepthEvalProfile(max_depth=10.0, conf_thresh=1.0, crop_mode="center224"),
            kf_every=10,
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
        yield from iter_7andn_scenes(
            dataset=self.name,
            data_root=data_root,
            prof=self.eval_profile(),
            max_scenes=max_scenes,
            max_frames=max_frames,
            split=split,
        )

    def load_scene_for_cache(
        self,
        data_root: Path,
        scene_id: str,
        max_frames: int,
        *,
        split: str | None = None,
    ) -> EvalSampleBatch:
        return load_7andn_scene(
            dataset=self.name,
            data_root=data_root,
            scene_id=scene_id,
            prof=self.eval_profile(),
            max_frames=max_frames,
            split=split,
        )


register(NRGBDPlugin())

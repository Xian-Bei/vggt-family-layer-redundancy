from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from analysis.backbones.contracts import EvalSceneContext
from analysis.datasets.contracts import DatasetAdaptor, EvalSampleBatch, PreparedSkipEvalScene
from analysis.datasets.seven_and_n_common import count_7andn_scenes, iter_7andn_scenes, load_7andn_scene
from analysis.schema.v3 import DatasetEvalProfile, DepthEvalProfile, GeometryEvalProfile

_VALID_SPLITS = ("train", "test")


class SevenScenesAdaptor(DatasetAdaptor):
    name = "7scenes"

    def validate_split(self, split: str | None = None) -> str:
        if not split:
            raise ValueError("7scenes requires split='train' or 'test'")
        split_norm = str(split).strip().lower()
        if split_norm not in _VALID_SPLITS:
            raise ValueError(f"Invalid 7scenes split: {split!r} (expected train or test)")
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
        return count_7andn_scenes(
            dataset=self.name,
            data_root=data_root,
            prof=self.eval_profile(),
            max_scenes=max_scenes,
            split=self.validate_split(split),
        )

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
            split=self.validate_split(split),
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
            split=self.validate_split(split),
        )

    def evaluate_scene(
        self,
        backbone_adaptor: Any,
        model: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        from analysis.datasets import eval_impl

        return eval_impl.eval_7andn_points_adapter(
            backbone_adaptor,
            model,
            batch,
            ctx.device,
            ctx.profile,
            ctx,
        )

    def prepare_skip_eval_scene(self, backbone_adaptor: Any, batch: EvalSampleBatch, ctx: EvalSceneContext) -> PreparedSkipEvalScene:
        from analysis.datasets import eval_impl
        return eval_impl.prepare_7andn_skip_scene(backbone_adaptor, batch, ctx.device, ctx.profile)

    def score_skip_eval_prediction(self, backbone_adaptor: Any, prediction: Any, prepared: PreparedSkipEvalScene, ctx: EvalSceneContext) -> Dict[str, float]:
        from analysis.datasets import eval_impl
        del backbone_adaptor
        return eval_impl.score_7andn_skip_prediction(prediction, prepared, ctx.profile, ctx)

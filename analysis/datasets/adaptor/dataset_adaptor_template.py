"""Template for adding a new LaRe dataset adaptor.

Copy this file to ``analysis/datasets/adaptor/<dataset_name>.py``, rename
``NewDatasetAdaptor``, fill every TODO, then register it in
``analysis/datasets/registry.py``. This module is intentionally not imported
or registered by default.

Dataset adaptors own dataset-specific details:
- dataset/split validation and output naming;
- scene discovery and frame selection;
- parsing raw files into ``EvalSampleBatch``;
- evaluating a normalized backbone ``PredictionBundle`` against this
  dataset's GT conventions.

Public pipeline code should only obtain adaptors through
``analysis.datasets.registry.get_dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from analysis.backbones.contracts import EvalSceneContext
from analysis.datasets.contracts import DatasetAdaptor, EvalSampleBatch
from analysis.schema.v3 import DatasetEvalProfile, GeometryEvalProfile


class NewDatasetAdaptor(DatasetAdaptor):
    """Fill-in template for a new dataset.

    Required methods:
    - ``eval_profile``: metric thresholds and frame stride/size.
    - ``iter_scenes``: discover scenes and yield unified ``EvalSampleBatch``.
    - ``load_scene_for_cache``: load one scene by stable ``scene_id``.
    - ``evaluate_scene``: run backbone prediction and compute dataset metrics.

    Optional methods:
    - ``validate_split``: enforce dataset split names, e.g. train/test.
    - ``output_tag``: customize output directory tag, e.g. dataset_test.
    - ``count_scenes``: provide fast tqdm totals without loading images.
    """

    # Lowercase registry key used by CLI ``--dataset`` and ``get_dataset``.
    name = "new_dataset"

    def validate_split(self, split: str | None = None) -> str | None:
        """Validate and normalize an optional split.

        Return ``None`` for datasets without splits. Raise ``ValueError`` for
        invalid split names. Public scripts call this before loading data.
        """
        if split is None:
            return None
        split_norm = str(split).strip().lower()
        # TODO: validate split_norm against this dataset's supported splits.
        return split_norm

    def output_tag(self, split: str | None = None) -> str:
        """Return the on-disk dataset tag under ``OUTPUT_ROOT/<backbone>/``.

        Examples:
        - ``scannet`` -> ``scannet``
        - ``7scenes`` + ``test`` -> ``7scenes_test``
        """
        split_norm = self.validate_split(split)
        if split_norm:
            return f"{self.name}_{split_norm}"
        return self.name

    def count_scenes(
        self,
        data_root: Path,
        *,
        max_scenes: int = 0,
        split: str | None = None,
    ) -> int | None:
        """Fast scene count for progress bars.

        Prefer filesystem metadata or dataset indices here. Return ``None`` if
        counting would require loading images/GT.
        """
        split = self.validate_split(split)
        del data_root, split
        # TODO: return total scene count, capped by max_scenes when > 0.
        del max_scenes
        return None

    def eval_profile(self) -> DatasetEvalProfile:
        """Return dataset-specific metric/eval settings.

        ``GeometryEvalProfile`` controls point metrics such as Acc/Comp/CD.
        ``kf_every`` and ``size`` describe the dataset's default evaluation
        sampling/resolution conventions.
        """
        return DatasetEvalProfile(
            dataset=self.name,
            geometry=GeometryEvalProfile(
                dist_clip=0.1,
                use_icp=True,
                max_points=999_999,
            ),
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
        """Yield scenes in deterministic evaluation order.

        Each yielded ``EvalSampleBatch`` should contain:
        - ``scene_id``: stable id used in output paths and comparisons.
        - ``images``: raw/preprocessed images or view dicts understood by this
          adaptor's ``evaluate_scene`` and cache path.
        - ``frame_ids`` and/or ``meta``: selected frame ids, paths, scene dir,
          and any dataset-specific GT lookup info.

        Apply ``max_scenes`` and ``max_frames`` here so all public stages share
        the same dataset sampling.
        """
        split = self.validate_split(split)
        del data_root, max_scenes, max_frames, split
        # TODO: discover scenes and yield ``EvalSampleBatch`` objects.
        raise NotImplementedError
        yield  # pragma: no cover

    def load_scene_for_cache(
        self,
        data_root: Path,
        scene_id: str,
        max_frames: int,
        *,
        split: str | None = None,
    ) -> EvalSampleBatch:
        """Load a single scene by ``scene_id`` for cache/replay workflows.

        This must use the same frame-selection policy as ``iter_scenes`` so a
        scene has identical inputs across eval, cache, and skip-eval stages.
        """
        split = self.validate_split(split)
        del data_root, scene_id, max_frames, split
        # TODO: locate scene_id and return the same batch shape as iter_scenes.
        raise NotImplementedError

    def evaluate_scene(
        self,
        backbone_adaptor: Any,
        model: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        """Run model prediction and compute unified metrics for one scene.

        The backbone adaptor should only provide normalized predictions through
        ``backbone_adaptor.forward_predictions(model, images_batched)``. Dataset
        adaptors own how to:
        - prepare dataset images/views for that backbone call;
        - load and align GT poses, depth, or point clouds;
        - compute unified metric keys used by ``EvalRecordV3``.

        Return only JSON-serializable metric values. Common keys include:
        ``ape``, ``are``, ``rpe_trans``, ``rpe_rot``, ``acc``, ``comp``, ``cd``,
        ``inference_time_ms``, and ``infer_peak_vram_gb``.
        """
        del backbone_adaptor, model, batch, ctx
        # TODO: implement directly or call helpers in ``analysis.datasets``.
        raise NotImplementedError

    def prepare_eval_scene_cpu(
        self,
        backbone_adaptor: Any,
        batch: EvalSampleBatch,
        ctx: EvalSceneContext,
    ) -> Any:
        """Decode/preprocess one scene and load GT without moving tensors to CUDA.

        Return a ``PreparedEvalScene`` whose ``images`` tensor is CPU FP32 and
        contiguous. The shared preload pipeline pins that tensor before H2D.
        Keep GT, poses, frame metadata, and scoring state on CPU.
        """
        del backbone_adaptor, batch, ctx
        raise NotImplementedError

    def score_eval_prediction(
        self,
        backbone_adaptor: Any,
        prediction: Any,
        prepared: Any,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        """Score a CPU ``PredictionBundle`` against the prepared CPU state."""
        del backbone_adaptor, prediction, prepared, ctx
        raise NotImplementedError

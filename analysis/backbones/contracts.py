"""Backbone adapter contracts for LaRe v3 pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.prune_plan import UnitPrunePlan, prune_actions_from_plan
from analysis.schema.v3 import LayerUnitSpec, PruneAction


@dataclass
class BackboneCapabilities:
    name: str
    supports_cache: bool = True
    supports_prune: bool = True
    supports_lora: bool = True
    supports_eval: bool = True
    num_frame_blocks: int = 0
    num_global_blocks: int = 0


@dataclass
class PredictionBundle:
    """Normalized prediction fields for eval/training."""

    pose_enc: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    depth_conf: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    local_points: Optional[torch.Tensor] = None
    conf: Optional[torch.Tensor] = None
    camera_poses: Optional[torch.Tensor] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheForwardResult:
    unit_map: Dict[str, torch.Tensor]
    meta: Dict[str, Any]
    pred: PredictionBundle
    initial_tokens: Optional[torch.Tensor] = None


@dataclass
class PredictionLossWeights:
    w_pred: float = 1.0


@dataclass(frozen=True)
class TrainingHyperparams:
    optimizer_cls: Any = torch.optim.AdamW
    lr: float = 5e-5
    weight_decay: float = 0.05
    min_lr: float = 1e-8
    warmup_ratio: float = 0.05
    grad_clip_max_norm: float = 1.0
    w_align: float = 1.0
    w_align_mse: float = 1.0
    w_align_cos: float = 1.0
    w_align_cka: float = 0.0
    w_pred: float = 1.0
    w_reg: float = 0.0


@dataclass
class TrainFromCacheArgs:
    experiment: str
    dataset: str
    cache_dir: Path
    prune_plan_path: Path
    output_dir: Path
    hf_model: str
    device: str
    dtype: str = "bfloat16"
    epochs: int = 1
    max_scenes: int = 0
    max_steps: int = 0
    data_root: Optional[Path] = None
    max_align_tokens: int = 4096
    eval_epoch: int = 0
    save_every_epoch: int = 10
    eval_max_scenes: int = 0
    max_eval_views: int = 50
    eval_at_start: bool = False
    eval_bounds_root: Path = Path("analysis_outputs_v3")
    eval_scannet_dev50: bool = True
    eval_scannet_test50: bool = True
    eval_7scenes: bool = True
    eval_7scenes_split: str = "test"
    eval_nrgbd: bool = True
    eval_nuscenes: bool = False
    eval_nuscenes_split: str = "val"
    eval_waymo: bool = False
    eval_waymo_split: str = "validation"
    root_7scenes: Path = Path("data/7scenes")
    root_nrgbd: Path = Path("data/neural_rgbd_data")
    root_nuscenes: Path = Path("data/nuscenes")
    root_waymo: Path = Path("data/waymo")
    prune_plan_name: str = ""
    recovery_plan_root: Path = Path("")
    recovery_plan_name: str = ""
    cache_dirs: Sequence[Path] = field(default_factory=tuple)


@dataclass
class EvalSceneContext:
    device: torch.device
    profile: Any
    export_vis_2d: bool = False
    export_vis_3d: bool = False
    vis_save_root: Optional[Path] = None
    vis_max_points: int = 0
    backbone_name: str = ""
    dataset_name: str = ""


class BackboneAdapter(ABC):
    """Adapter around vendored backbone code under ``backbones/``."""

    name: str

    @abstractmethod
    def capabilities(self) -> BackboneCapabilities:
        ...

    @abstractmethod
    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        ...

    @abstractmethod
    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        ...

    @abstractmethod
    def extract_layer_units(
        self,
        model: nn.Module,
        images: torch.Tensor,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Any] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Return {unit_id: activation tensor} and metadata (e.g. patch_start_idx)."""

    @abstractmethod
    def forward_predictions(self, model: nn.Module, images: torch.Tensor) -> PredictionBundle:
        ...

    def supports_incremental_skip_eval(self) -> bool:
        """Whether this adaptor implements scene-local incremental skip replay."""
        return False

    def iter_skip_span_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int, PredictionBundle]]:
        """Yield one normalized prediction for every requested closed unit span."""
        del model, images, spans
        raise NotImplementedError(
            f"Backbone {self.name!r} does not implement incremental skip evaluation"
        )

    @abstractmethod
    def apply_prune_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> nn.Module:
        ...

    @abstractmethod
    def inject_lora_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> List[nn.Parameter]:
        ...

    def default_hf_model(self) -> str:
        return "facebook/VGGT-1B"

    def training_config(self) -> TrainingHyperparams:
        raise NotImplementedError(f"{self.name} does not define training hyperparameters")

    def build_training_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        config: Optional[TrainingHyperparams] = None,
    ) -> torch.optim.Optimizer:
        del trainable, config
        raise NotImplementedError(f"{self.name} does not define a training optimizer")

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        return [repo_root]

    def load_eval_images(self, image_paths: Sequence[Path | str]) -> torch.Tensor:
        """Load and preprocess eval/cache images -> ``[S, 3, H, W]`` in ``[0, 1]``."""
        raise NotImplementedError(f"{self.name} does not implement load_eval_images")

    def load_driving_depth_stack(
        self,
        depth_paths: Sequence[Path | str],
        *,
        output_hw: Tuple[int, int],
    ) -> Tuple[Any, int]:
        """Rasterize sparse driving-depth GT to match model RGB preprocessing."""
        raise NotImplementedError(f"{self.name} does not implement load_driving_depth_stack")

    def load_scannet_gt_depth(
        self,
        scene: Any,
        *,
        output_hw: Tuple[int, int],
    ) -> Any:
        """Load ScanNet dense GT depth aligned to ``load_eval_images`` output size."""
        from analysis.eval.prediction_eval import _load_gt_depth_for_scene

        return _load_gt_depth_for_scene(scene, image_hw=output_hw)

    def align_7andn_gt_depth(self, view: Any, *, output_hw: Tuple[int, int]) -> Any:
        from analysis.eval.prediction_eval import _load_7andn_gt_depth_vggt

        return _load_7andn_gt_depth_vggt(view, image_hw=output_hw)

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        """Return the concrete module for a public unit id."""
        raise NotImplementedError(f"{self.name} does not map unit ids to blocks")

    def normalize_unit_activation(
        self,
        unit_id: str,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Normalize a block-internal activation to public ``[B,S,P,C]`` layout."""
        kind, _ = str(unit_id).split("_", 1)
        t = tensor
        if t.dim() == 4:
            if int(t.shape[0]) == int(batch_size) and int(t.shape[1]) == int(num_frames):
                return t.contiguous()
            if (
                kind == "global"
                and int(t.shape[0]) == int(batch_size)
                and int(t.shape[1]) == 1
                and int(t.shape[2]) % int(num_frames) == 0
            ):
                return t.reshape(
                    batch_size,
                    num_frames,
                    int(t.shape[2]) // int(num_frames),
                    int(t.shape[3]),
                ).contiguous()
            if int(t.shape[0]) == 1 and int(t.shape[1]) == int(batch_size) * int(num_frames):
                return t.reshape(batch_size, num_frames, int(t.shape[2]), int(t.shape[3])).contiguous()
        if t.dim() == 3 and kind == "frame" and int(t.shape[0]) == int(batch_size) * int(num_frames):
            return t.reshape(batch_size, num_frames, int(t.shape[1]), int(t.shape[2])).contiguous()
        if t.dim() == 3 and kind == "global" and int(t.shape[0]) == int(batch_size):
            if int(t.shape[1]) % int(num_frames) == 0:
                return t.reshape(
                    batch_size,
                    num_frames,
                    int(t.shape[1]) // int(num_frames),
                    int(t.shape[2]),
                ).contiguous()
        raise ValueError(
            f"Cannot normalize {self.name} {unit_id} activation {tuple(t.shape)} "
            f"to [B={batch_size},S={num_frames},P,C]"
        )

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Apply a named operation to a public unit id.

        Public pipeline code should call this instead of reaching into
        aggregator/decoder internals. Adapters may return operation-specific
        values, e.g. LoRA trainable params.
        """
        raise NotImplementedError(f"{self.name} does not implement unit operation {operation!r}")

    def forward_cache_batch(
        self,
        model: nn.Module,
        images: torch.Tensor,
        device: torch.device,
    ) -> CacheForwardResult:
        from analysis.amp_policy import amp_dtype_for_device, autocast_device_dtype

        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device)
        ac_kw = autocast_device_dtype(device)
        with torch.no_grad(), torch.autocast(**ac_kw):
            fused = self.extract_layer_units_and_predictions(model, images)
            if fused is not None:
                return fused
            unit_map, meta = self.extract_layer_units(model, images)
            pred = self.forward_predictions(model, images)
        return CacheForwardResult(unit_map=unit_map, meta=meta, pred=pred, initial_tokens=None)

    def forward_units_and_prediction(
        self,
        model: nn.Module,
        images: torch.Tensor,
        device: torch.device,
        unit_ids: Optional[Set[str]] = None,
    ) -> CacheForwardResult:
        """Forward once for selected unit activations plus normalized predictions."""
        if unit_ids is None:
            return self.forward_cache_batch(model, images, device)
        from analysis.amp_policy import autocast_device_dtype

        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(device)
        ac_kw = autocast_device_dtype(device)
        with torch.no_grad(), torch.autocast(**ac_kw):
            unit_map, meta = self.extract_layer_units(model, images, unit_ids=unit_ids)
            pred = self.forward_predictions(model, images)
        return CacheForwardResult(
            unit_map=unit_map,
            meta=meta,
            pred=pred,
            initial_tokens=self.build_initial_tokens(model, images),
        )

    def extract_layer_units_and_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Optional[CacheForwardResult]:
        """Optional fused cache path: return layer units + predictions from one forward."""
        del model, images
        return None

    def prediction_bundle_from_raw(
        self,
        raw: Dict[str, Any],
        images: torch.Tensor,
    ) -> PredictionBundle:
        """Normalize a backbone raw prediction dict into the public bundle."""
        del images
        return PredictionBundle(raw=dict(raw))

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Return layer_-1 replay seed tokens if this backbone supports them."""
        del model, images
        return None

    def cache_manifest_extra(self, model: nn.Module) -> Dict[str, Any]:
        return {}

    def minimal_prune_actions(self, *, cache_ref: str, dataset: str) -> List[Dict[str, Any]]:
        return [
            {"action": "drop_unit", "unit_id": "global_7"},
            {"action": "drop_unit", "unit_id": "global_8"},
            {"action": "lora_unit", "unit_id": "frame_6", "rank": 8, "alpha": 16.0},
        ]

    def train_from_cache(self, args: TrainFromCacheArgs) -> None:
        from analysis.backbones.train_from_cache import launch_cache_replay_training

        launch_cache_replay_training(self.name, args)

    def infer_train_cache_geometry(
        self,
        row: Dict[str, Any],
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        *,
        cache_dir: Optional[Path] = None,
    ) -> Tuple[int, int, int, int]:
        """Return (height, width, seq_len, patch_start_idx) for cache-first training."""
        from analysis.backbones.train_cache_replay import infer_train_cache_geometry

        return infer_train_cache_geometry(row, model, get_layer_cpu, cache_dir=cache_dir)

    def build_train_dummy_images(
        self,
        row: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        get_layer_cpu: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        h, w, seq_len, _ = self.infer_train_cache_geometry(row, model, get_layer_cpu)
        from analysis.backbones.train_cache_replay import build_vggt_family_dummy_images

        return build_vggt_family_dummy_images(seq_len=seq_len, h=h, w=w, device=device)

    def train_replay_extra_kwargs(self, *, h: int, w: int) -> Dict[str, Any]:
        return {}

    def required_head_layers_for_cache(self, model: nn.Module, num_layers: int) -> List[int]:
        """Layers whose concat activations are needed before replay start."""
        del model, num_layers
        return []

    def validate_cached_prediction(self, pred: Dict[str, Any]) -> None:
        del pred

    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        del teacher_pred, images_batched, device, dataset_name
        raise NotImplementedError(f"{self.name} does not implement teacher-target conversion")

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, Any],
        *,
        weights: PredictionLossWeights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del student_pred, target_batch, weights
        raise NotImplementedError(f"{self.name} does not implement prediction loss")

    def replay_student_cache_first(
        self,
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        num_layers: int,
        **kwargs: Any,
    ) -> Any:
        del model, get_layer_cpu, num_layers, kwargs
        raise NotImplementedError(f"{self.name} does not implement cache replay")

    def cache_replay_engine(self) -> Any:
        raise NotImplementedError(f"{self.name} does not expose a cache replay engine")

    def after_train_replay_release_cpu(
        self,
        layer_cpu_cache: Dict[int, torch.Tensor],
        *,
        rules: Sequence[Any],
        prev_layers: Sequence[int],
        num_layers: int,
        seed_layer: int = 0,
    ) -> None:
        del layer_cpu_cache, rules, prev_layers, num_layers, seed_layer

    def apply_checkpoint_structure(
        self,
        model: nn.Module,
        payload: Dict[str, Any],
        *,
        prune_plan: Optional[Any] = None,
    ) -> None:
        ckpt_plan = payload.get("prune_plan")
        if isinstance(ckpt_plan, dict):
            up = UnitPrunePlan.from_dict(ckpt_plan)
            drop_actions, lora_actions = prune_actions_from_plan(up)
            self.apply_prune_plan(model, drop_actions)
            self.inject_lora_plan(model, lora_actions)
        elif isinstance(prune_plan, UnitPrunePlan):
            drop_actions, lora_actions = prune_actions_from_plan(prune_plan)
            self.apply_prune_plan(model, drop_actions)
            self.inject_lora_plan(model, lora_actions)
        elif prune_plan is not None:
            raise ValueError("Legacy prune plan formats are no longer supported")
        recovery_specs = payload.get("recovery_specs")
        recovery_weights = payload.get("recovery_weights")
        if recovery_weights is not None and not (isinstance(recovery_specs, list) and recovery_specs):
            raise ValueError("Compact recovery checkpoint is missing recovery_specs")
        if isinstance(recovery_specs, list) and recovery_specs:
            from analysis.recovery.operators import (
                apply_recovery_weights,
                compact_recovery_operators,
                install_recovery_modules,
            )

            device = next(model.parameters()).device
            recovery_specs, recovery_weights = compact_recovery_operators(
                recovery_specs, recovery_weights
            )
            install_recovery_modules(model, self, recovery_specs, device=device)
            if recovery_weights is not None:
                apply_recovery_weights(model, self, recovery_specs, recovery_weights, device=device)
        if "trainable_state_dict" in payload:
            state = payload["trainable_state_dict"]
            current = dict(model.named_parameters())
            unexpected = sorted(set(state) - set(current))
            if unexpected:
                raise RuntimeError(
                    f"Delta checkpoint contains unknown parameters: {unexpected[:8]}"
                )
            model.load_state_dict(state, strict=False)
            return
        if "model_state_dict" not in payload and recovery_weights is not None:
            return
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=False)

    def unit_id(self, kind: str, index: int) -> str:
        return f"{kind}_{index}"

    def parse_unit_id(self, unit_id: str) -> Tuple[str, int]:
        kind, idx = unit_id.split("_", 1)
        return kind, int(idx)

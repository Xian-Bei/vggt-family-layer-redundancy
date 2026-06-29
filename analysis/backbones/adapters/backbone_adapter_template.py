"""Template for adding a new LaRe backbone adapter.

Copy this file to ``analysis/backbones/adapters/<name>_adapter.py``, rename
``NewBackboneAdapter``, fill every TODO, then register the adapter in
``analysis/backbones/registry.py``. This module is intentionally not imported
or registered by default.

The public pipeline talks only in unit ids (``frame_i`` / ``global_i``) and
normalized ``PredictionBundle`` objects. All mapping from unit ids to vendor
modules belongs inside the adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    PredictionBundle,
    PredictionLossWeights,
)
from analysis.schema.v3 import LayerUnitSpec, PruneAction


class NewBackboneAdapter(BackboneAdapter):
    """Fill-in template for a new backbone.

    Required public abilities:

    1. load vendor model
    2. expose frame/global unit catalog
    3. return selected unit activations
    4. convert raw model outputs to PredictionBundle
    5. apply drop_unit and lora_unit by unit id
    6. provide cache-first replay and prediction loss for LoRA training
    7. evaluate one scene through the shared eval contract
    """

    name = "new_backbone"

    # Fill this with the adapter-private mapping from public unit id to module.
    # Example:
    # UNIT_MODULE_PATHS = {
    #     "frame_0": "aggregator.frame_blocks.0",
    #     "global_0": "aggregator.global_blocks.0",
    # }
    UNIT_MODULE_PATHS: Dict[str, str] = {}

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            name=self.name,
            supports_cache=True,
            supports_prune=True,
            supports_lora=True,
            supports_eval=True,
            # TODO: fill real counts. They should match layer_unit_catalog().
            num_frame_blocks=0,
            num_global_blocks=0,
        )

    def default_hf_model(self) -> str:
        # TODO: return a Hugging Face id or local default checkpoint path.
        return "vendor-or-local/default-checkpoint"

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        # TODO: return vendor package paths if this backbone lives outside the
        # normal repo import path. Keep repo_root in the list unless unneeded.
        return [repo_root]

    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        """Load the official/vendor model.

        Keep weights in fp32 unless the vendor model explicitly requires
        another storage dtype. The public paths use autocast for forward speed.
        Do not silently fall back to CPU when a CUDA device was requested.
        """
        del hf_model, device, dtype, eval_mode
        raise NotImplementedError("load the vendor model and move it to device")

    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        """Return every public unit the shared pipeline may operate on."""
        del model
        units: List[LayerUnitSpec] = []
        for unit_id, module_path in self.UNIT_MODULE_PATHS.items():
            kind, index = self.parse_unit_id(unit_id)
            units.append(
                LayerUnitSpec(
                    unit_id=unit_id,
                    kind=kind,  # type: ignore[arg-type]
                    index=index,
                    module_path=module_path,
                    droppable=True,
                    lora_target=True,
                )
            )
        kind_order = {"frame": 0, "global": 1}
        return sorted(units, key=lambda unit: (unit.index, kind_order.get(unit.kind, 99)))

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        """Map public unit id to the concrete vendor module."""
        module_path = self.UNIT_MODULE_PATHS.get(unit_id)
        if not module_path:
            raise KeyError(f"Unknown unit id for {self.name}: {unit_id}")
        return self._get_submodule(model, module_path)

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Apply one operation to one public unit id.

        Supported operations used by public code:

        - ``drop_unit``: replace the concrete block with an identity/skip block
          that preserves input and output shape.
        - ``lora_unit``: inject LoRA into the concrete block and return the
          trainable parameters.
        """
        block = self.block_for_unit(model, unit_id)
        if operation == "drop_unit":
            del block, kwargs
            # TODO: replace the module at UNIT_MODULE_PATHS[unit_id] with the
            # backbone-specific identity/skip implementation.
            raise NotImplementedError("replace unit with a shape-preserving identity")
        if operation == "lora_unit":
            del block
            # TODO: inject LoRA according to kwargs: rank, alpha, dropout, and
            # target_submodules. Return List[nn.Parameter].
            raise NotImplementedError("inject LoRA into this unit")
        raise ValueError(f"Unsupported unit operation for {self.name}: {operation!r}")

    def apply_prune_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> nn.Module:
        for action in actions:
            if action.action != "drop_unit":
                continue
            self.apply_unit_operation(model, action.unit_id, "drop_unit")
        return model

    def inject_lora_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for action in actions:
            if action.action != "lora_unit":
                continue
            injected = self.apply_unit_operation(
                model,
                action.unit_id,
                "lora_unit",
                rank=action.rank,
                alpha=action.alpha,
                dropout=action.dropout,
                target_submodules=tuple(action.target_submodules),
            )
            params.extend(list(injected or []))
        return params

    def extract_layer_units(
        self,
        model: nn.Module,
        images: torch.Tensor,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Any] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Run the aggregator and return requested unit activations.

        ``images`` is usually ``[B, S, 3, H, W]``. If the caller passes
        ``[S, 3, H, W]``, normalize it here or delegate to a common helper.

        Return:
        - unit_map[unit_id]: tensor shaped ``[B, S, P, C]``
        - meta["patch_start_idx"]: index where patch tokens start
        - optional meta["aggregated_concat"]: per-layer ``[B, S, P, 2C]``
        """
        del model, images, unit_ids, layer_callback
        raise NotImplementedError("collect selected frame/global unit activations")

    def extract_layer_units_and_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Optional[CacheForwardResult]:
        """Optional fused path for cache/redundancy.

        Implement this when one vendor forward can return activations and raw
        prediction together. Returning None lets the base class call
        extract_layer_units() and forward_predictions() separately.
        """
        del model, images
        return None

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Return layer_-1 seed tokens for cache-first replay, if supported."""
        del model, images
        # TODO: return ``[B, S, P, C]`` tokens before the first public unit.
        return None

    def forward_predictions(self, model: nn.Module, images: torch.Tensor) -> PredictionBundle:
        """Run complete inference and normalize vendor raw prediction."""
        del model, images
        # TODO: call the official full forward, then:
        # return self.prediction_bundle_from_raw(raw_prediction, images)
        raise NotImplementedError("run full inference")

    def prediction_bundle_from_raw(
        self,
        raw: Dict[str, Any],
        images: torch.Tensor,
    ) -> PredictionBundle:
        """Convert vendor output to the common prediction contract.

        Common fields:
        - pose_enc: raw pose encoding if available
        - camera_poses: c2w-like poses, ``[B, S, 4, 4]`` or ``[S, 4, 4]``
        - depth/depth_conf: dense depth fields, or None if unsupported
        - points/local_points/conf: point prediction fields
        - raw: keep the vendor output for debugging or backbone-specific loss
        """
        del images
        # TODO: fill every supported field. Unsupported fields must stay None.
        return PredictionBundle(raw=dict(raw))

    def cache_manifest_extra(self, model: nn.Module) -> Dict[str, Any]:
        """Optional metadata saved into cache_manifest_v3.json."""
        del model
        return {}

    def minimal_prune_actions(self, *, cache_ref: str, dataset: str) -> List[Dict[str, Any]]:
        """Fallback plan used when auto planning needs a minimal action set."""
        del cache_ref, dataset
        # TODO: choose safe default units for this backbone.
        return []

    def required_head_layers_for_cache(self, model: nn.Module, num_layers: int) -> List[int]:
        """Layers whose cached activations are needed to replay prediction heads."""
        del model, num_layers
        return []

    def validate_cached_prediction(self, pred: Dict[str, Any]) -> None:
        """Validate teacher prediction saved in cache before training."""
        del pred
        # TODO: check keys and tensor ranks required by build_target_batch_from_teacher().

    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """Convert cached teacher prediction to the target batch for loss."""
        del teacher_pred, images_batched, device, dataset_name
        raise NotImplementedError("build backbone-specific prediction targets")

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, Any],
        *,
        weights: PredictionLossWeights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Backbone-specific prediction loss.

        The shared training path owns align loss. This method owns only the
        prediction loss needed to distill/evaluate the current backbone.
        """
        del student_pred, target_batch, weights
        raise NotImplementedError("compute prediction loss")

    def train_replay_extra_kwargs(self, *, h: int, w: int) -> Dict[str, Any]:
        """Optional geometry passed into cache replay."""
        del h, w
        return {}

    def replay_student_cache_first(
        self,
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        num_layers: int,
        **kwargs: Any,
    ) -> Any:
        """Replay the trainable student from cached seed activations."""
        del model, get_layer_cpu, num_layers, kwargs
        raise NotImplementedError("replay from cache seed through trainable units and heads")

    def cache_replay_engine(self) -> Any:
        """Return the backbone-specific replay engine module/object."""
        raise NotImplementedError("return cache replay engine")

    def after_train_replay_release_cpu(
        self,
        layer_cpu_cache: Dict[int, torch.Tensor],
        *,
        rules: Sequence[Any],
        prev_layers: Sequence[int],
        num_layers: int,
        seed_layer: int = 0,
    ) -> None:
        """Optional memory cleanup hook after one replay step."""
        del layer_cpu_cache, rules, prev_layers, num_layers, seed_layer

    def periodic_eval_skip_depth(self) -> bool:
        """Return True when eval should omit dense-depth metrics."""
        return False

    @staticmethod
    def _get_submodule(model: nn.Module, module_path: str) -> nn.Module:
        module: nn.Module = model
        for part in module_path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module


# After copying this file, register the real adapter by editing
# analysis/backbones/registry.py:
#
#     from .adapters.<name>_adapter import <Name>Adapter
#     register(<Name>Adapter())

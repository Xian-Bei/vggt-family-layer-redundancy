"""MapAnything and MapAnything V1 adapters.

Both checkpoints expose the same public model API.  Their information-sharing
depth is discovered from the loaded checkpoint (16 blocks for the current
model, 24 for V1), so all execution logic lives in this shared base class.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    PredictionBundle,
    TrainingHyperparams,
)
from analysis.backbones.registry import register
from analysis.recovery.train.model import inject_lora_attn
from analysis.schema.v3 import LayerUnitSpec, PruneAction


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[3] / "backbones" / "map-anything"


def _ensure_vendor_path() -> None:
    value = str(_vendor_root())
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


class _MapAnythingSkipBlock(nn.Module):
    def forward(self, x: torch.Tensor, xpos=None, **kwargs: Any) -> torch.Tensor:
        del xpos, kwargs
        return x


class MapAnythingAdapterBase(BackboneAdapter):
    """Shared implementation for the current and V1 MapAnything checkpoints."""

    hf_model_id = "facebook/map-anything"

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            name=self.name,
            supports_cache=True,
            supports_prune=True,
            supports_lora=True,
            supports_eval=True,
        )

    def default_hf_model(self) -> str:
        return self.hf_model_id

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        return [repo_root / "backbones" / "map-anything", repo_root]

    def training_config(self) -> TrainingHyperparams:
        return TrainingHyperparams()

    def build_training_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        config: Optional[TrainingHyperparams] = None,
    ) -> torch.optim.Optimizer:
        cfg = config or self.training_config()
        return cfg.optimizer_cls(
            trainable,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        _ensure_vendor_path()
        from mapanything.models import MapAnything

        model = MapAnything.from_pretrained(str(hf_model or self.default_hf_model()))
        # The official inference path controls autocast; retain fp32 master weights.
        if dtype not in (None, torch.float32):
            raise ValueError("MapAnything expects fp32 weights with autocast; pass dtype=None")
        model = model.to(device=device)
        if eval_mode:
            model.eval()
        return model

    @staticmethod
    def _blocks(model: nn.Module) -> nn.ModuleList:
        blocks = model.info_sharing.self_attention_blocks
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("MapAnything info_sharing.self_attention_blocks is not a ModuleList")
        return blocks

    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        units: List[LayerUnitSpec] = []
        for physical_index in range(len(self._blocks(model))):
            # UniCeption alternating attention executes global first, then frame.
            kind = "global" if physical_index % 2 == 0 else "frame"
            logical_index = physical_index // 2
            units.append(
                LayerUnitSpec(
                    unit_id=f"{kind}_{logical_index}",
                    kind=kind,
                    index=logical_index,
                    module_path=f"info_sharing.self_attention_blocks.{physical_index}",
                    droppable=True,
                    lora_target=True,
                )
            )
        return units

    def _physical_index(self, unit_id: str) -> int:
        kind, layer = self.parse_unit_id(unit_id)
        if kind not in {"global", "frame"}:
            raise KeyError(unit_id)
        return 2 * int(layer) + (0 if kind == "global" else 1)

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        index = self._physical_index(unit_id)
        blocks = self._blocks(model)
        if index < 0 or index >= len(blocks):
            raise KeyError(unit_id)
        return blocks[index]

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        index = self._physical_index(unit_id)
        blocks = self._blocks(model)
        if index < 0 or index >= len(blocks):
            raise KeyError(unit_id)
        if operation == "drop_unit":
            blocks[index] = _MapAnythingSkipBlock()
            return None
        if operation == "lora_unit":
            params = inject_lora_attn(
                blocks[index],
                rank=int(kwargs["rank"]),
                alpha=float(kwargs["alpha"]),
                dropout=float(kwargs.get("dropout", 0.0)),
            )
            if not params:
                raise RuntimeError(f"No attention qkv/proj projections found in {unit_id}")
            return params
        raise ValueError(f"Unsupported unit operation {operation!r}")

    def apply_prune_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> nn.Module:
        for action in actions:
            if action.action == "drop_unit":
                self.apply_unit_operation(model, action.unit_id, "drop_unit")
        return model

    def inject_lora_plan(
        self, model: nn.Module, actions: Sequence[PruneAction]
    ) -> List[nn.Parameter]:
        trainable: List[nn.Parameter] = []
        for action in actions:
            if action.action == "lora_unit":
                trainable.extend(
                    self.apply_unit_operation(
                        model,
                        action.unit_id,
                        "lora_unit",
                        rank=action.rank,
                        alpha=action.alpha,
                        dropout=action.dropout,
                    )
                )
        return trainable

    def load_eval_images(self, image_paths: Sequence[Path | str]) -> torch.Tensor:
        _ensure_vendor_path()
        from mapanything.utils.image import load_images

        views = load_images(
            [str(path) for path in image_paths],
            resize_mode="fixed_mapping",
            norm_type="dinov2",
            patch_size=14,
            resolution_set=518,
            verbose=False,
        )
        return torch.cat([view["img"] for view in views], dim=0)

    def load_driving_depth_stack(
        self,
        depth_paths: Sequence[Path | str],
        *,
        output_hw: Tuple[int, int],
    ) -> Tuple[Any, int]:
        from analysis.datasets.driving_depth import load_driving_depth_stack_vggt

        return load_driving_depth_stack_vggt(depth_paths, output_hw=output_hw)

    def align_7andn_gt_depth(self, view: Any, *, output_hw: Tuple[int, int]) -> Any:
        from analysis.eval.prediction_eval import _load_7andn_gt_depth_mapanything

        return _load_7andn_gt_depth_mapanything(view, image_hw=output_hw)

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _images_to_views(self, model: nn.Module, images: torch.Tensor) -> List[Dict[str, Any]]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        images = images.to(self._model_device(model), dtype=torch.float32)
        bsz, seq_len, _c, height, width = (int(x) for x in images.shape)
        norm_type = str(model.encoder.data_norm_type)
        return [
            {
                "img": images[:, index],
                "true_shape": np.repeat(
                    np.asarray([[height, width]], dtype=np.int32), bsz, axis=0
                ),
                "idx": [index] * bsz,
                "instance": [str(index)] * bsz,
                "data_norm_type": [norm_type] * bsz,
            }
            for index in range(seq_len)
        ]

    def _infer(self, model: nn.Module, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # Isolate the caller's AMP context so official infer() can enable AMP
        # only around forward(), leaving its post-processing (including the
        # intrinsics linear solve) in fp32.
        device = self._model_device(model)
        with torch.autocast(device_type=device.type, enabled=False):
            return model.infer(
                self._images_to_views(model, images),
                memory_efficient_inference=False,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=False,
                mask_edges=False,
                apply_confidence_mask=False,
                ignore_calibration_inputs=True,
                ignore_depth_inputs=True,
                ignore_pose_inputs=True,
                ignore_depth_scale_inputs=True,
                ignore_pose_scale_inputs=True,
            )

    @staticmethod
    def _stack(preds: Sequence[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
        values = [pred.get(key) for pred in preds]
        if not values or any(not torch.is_tensor(value) for value in values):
            return None
        return torch.stack(values, dim=1)

    def prediction_bundle_from_raw(
        self, raw: Dict[str, Any], images: torch.Tensor
    ) -> PredictionBundle:
        del images
        preds = raw.get("views")
        if not isinstance(preds, list):
            raise ValueError("MapAnything raw prediction must contain a views list")
        points = self._stack(preds, "pts3d")
        local_points = self._stack(preds, "pts3d_cam")
        depth = self._stack(preds, "depth_z")
        conf = self._stack(preds, "conf")
        camera_poses = self._stack(preds, "camera_poses")
        tensor_raw: Dict[str, Any] = {}
        for key in (
            "pts3d",
            "pts3d_cam",
            "depth_z",
            "conf",
            "camera_poses",
            "metric_scaling_factor",
        ):
            value = self._stack(preds, key)
            if value is not None:
                tensor_raw[key] = value
        return PredictionBundle(
            depth=depth,
            depth_conf=conf,
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            raw=tensor_raw,
        )

    def forward_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        defer_cpu_geometry: bool = False,
    ) -> PredictionBundle:
        del defer_cpu_geometry
        return self.prediction_bundle_from_raw({"views": self._infer(model, images)}, images)

    @staticmethod
    def _hook_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Unsupported MapAnything block output type: {type(output)!r}")

    def _normalize_capture(
        self,
        unit_id: str,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        patch_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        kind, _ = self.parse_unit_id(unit_id)
        if kind == "global":
            if tensor.dim() != 3 or int(tensor.shape[0]) != batch_size:
                raise ValueError(f"Unexpected {unit_id} shape: {tuple(tensor.shape)}")
            # The single scene-scale token is appended after all view tokens.
            if (int(tensor.shape[1]) - 1) % num_frames == 0:
                tensor = tensor[:, :-1]
            if int(tensor.shape[1]) % num_frames != 0:
                raise ValueError(f"Cannot split {unit_id} over {num_frames} views")
            tensor = tensor.reshape(batch_size, num_frames, -1, int(tensor.shape[-1]))
        else:
            if tensor.dim() != 3 or int(tensor.shape[0]) != batch_size * num_frames:
                raise ValueError(f"Unexpected {unit_id} shape: {tuple(tensor.shape)}")
            tensor = tensor.reshape(batch_size, num_frames, -1, int(tensor.shape[-1]))
        register_count = int(tensor.shape[2]) - int(patch_tokens)
        if register_count < 0:
            raise ValueError(
                f"{unit_id} has {tensor.shape[2]} tokens, fewer than {patch_tokens} patches"
            )
        # UniCeption stores patches before registers. Public caches place specials first.
        if register_count:
            tensor = torch.cat(
                [tensor[:, :, patch_tokens:], tensor[:, :, :patch_tokens]], dim=2
            )
        return tensor.contiguous(), register_count

    def normalize_unit_activation(
        self,
        unit_id: str,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Normalize nested-module captures used by recovery auxiliaries.

        Unlike block hooks, these callbacks do not carry image geometry.  The
        only MapAnything-specific ambiguity is the global scale token, which is
        always the final token and must not enter per-view recovery features.
        """
        kind, _ = self.parse_unit_id(unit_id)
        if tensor.dim() != 3:
            return super().normalize_unit_activation(
                unit_id,
                tensor,
                batch_size=batch_size,
                num_frames=num_frames,
            )
        if kind == "global" and int(tensor.shape[0]) == int(batch_size):
            value = tensor
            if (int(value.shape[1]) - 1) % int(num_frames) == 0:
                value = value[:, :-1]
            if int(value.shape[1]) % int(num_frames) == 0:
                return value.reshape(
                    batch_size,
                    num_frames,
                    int(value.shape[1]) // num_frames,
                    int(value.shape[2]),
                ).contiguous()
        if kind == "frame" and int(tensor.shape[0]) == int(batch_size) * int(num_frames):
            return tensor.reshape(
                batch_size, num_frames, int(tensor.shape[1]), int(tensor.shape[2])
            ).contiguous()
        return super().normalize_unit_activation(
            unit_id,
            tensor,
            batch_size=batch_size,
            num_frames=num_frames,
        )

    def _capture_and_predict(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        unit_ids: Optional[Set[str]],
        layer_callback: Optional[Callable[[str, torch.Tensor], None]],
        with_prediction: bool,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Any],
        Optional[PredictionBundle],
        torch.Tensor,
    ]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        bsz, seq_len, _c, height, width = (int(x) for x in images.shape)
        patch_size = int(getattr(model.encoder, "patch_size", 14))
        patch_tokens = (height // patch_size) * (width // patch_size)
        catalog = self.layer_unit_catalog(model)
        wanted = unit_ids if unit_ids is not None else {unit.unit_id for unit in catalog}
        captures: Dict[str, torch.Tensor] = {}
        register_counts: Dict[str, int] = {}
        initial_tokens: Optional[torch.Tensor] = None
        handles = []

        def initial_hook(_module: nn.Module, inputs: Any) -> None:
            nonlocal initial_tokens
            value, _register_count = self._normalize_capture(
                "global_0",
                inputs[0],
                batch_size=bsz,
                num_frames=seq_len,
                patch_tokens=patch_tokens,
            )
            initial_tokens = value.detach()

        def make_hook(unit_id: str):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                value, register_count = self._normalize_capture(
                    unit_id,
                    self._hook_tensor(output),
                    batch_size=bsz,
                    num_frames=seq_len,
                    patch_tokens=patch_tokens,
                )
                value = value.detach()
                captures[unit_id] = value
                register_counts[unit_id] = register_count
                if layer_callback is not None:
                    layer_callback(unit_id, value)

            return hook

        for unit in catalog:
            if unit.unit_id in wanted:
                handles.append(self.block_for_unit(model, unit.unit_id).register_forward_hook(make_hook(unit.unit_id)))
        if not catalog:
            raise RuntimeError("MapAnything information-sharing transformer has no blocks")
        handles.append(self.block_for_unit(model, catalog[0].unit_id).register_forward_pre_hook(initial_hook))
        try:
            preds = self._infer(model, images)
        finally:
            for handle in handles:
                handle.remove()
        counts = set(register_counts.values())
        if len(counts) > 1:
            raise RuntimeError(f"Inconsistent MapAnything register counts: {sorted(counts)}")
        register_count = next(iter(counts), 0)
        meta = {
            "patch_start_idx": register_count,
            "camera_token_count": 0,
            "register_token_count": register_count,
            "patch_token_count": patch_tokens,
            "scale_token_excluded": True,
            "unit_ids": [unit.unit_id for unit in catalog],
        }
        bundle = (
            self.prediction_bundle_from_raw({"views": preds}, images)
            if with_prediction
            else None
        )
        if initial_tokens is None:
            raise RuntimeError("MapAnything did not expose the layer_-1 input to global_0")
        return captures, meta, bundle, initial_tokens

    def _capture_layer_units_only(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        unit_ids: Optional[Set[str]],
        layer_callback: Optional[Callable[[str, torch.Tensor], None]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor]:
        """Capture information-sharing activations without prediction heads."""
        from analysis.amp_policy import autocast_device_dtype

        if images.dim() == 4:
            images = images.unsqueeze(0)
        catalog = self.layer_unit_catalog(model)
        if not catalog:
            raise RuntimeError("MapAnything information-sharing transformer has no blocks")
        wanted = unit_ids if unit_ids is not None else {unit.unit_id for unit in catalog}
        wanted_indices = [
            self._physical_index(unit.unit_id)
            for unit in catalog
            if unit.unit_id in wanted
        ]
        last_needed = max(wanted_indices, default=-1)
        captures: Dict[str, torch.Tensor] = {}
        register_counts: Set[int] = set()
        device = self._model_device(model)
        images = images.to(device=device, dtype=torch.float32)

        try:
            with torch.inference_mode(), torch.autocast(**autocast_device_dtype(device)):
                context = self._prepare_replay_context(model, images)
                tokens = context["tokens"]
                bsz = int(context["batch_size"])
                seq_len = int(context["num_views"])
                patch_tokens = int(context["patch_tokens"])
                tokens_per_view = int(context["tokens_per_view"])
                initial_tokens, initial_register_count = self._normalize_capture(
                    "global_0",
                    tokens,
                    batch_size=bsz,
                    num_frames=seq_len,
                    patch_tokens=patch_tokens,
                )
                register_counts.add(initial_register_count)

                for physical_index in range(last_needed + 1):
                    tokens = self._run_replay_block(
                        model,
                        tokens,
                        physical_index,
                        batch_size=bsz,
                        num_views=seq_len,
                        tokens_per_view=tokens_per_view,
                    )
                    unit = catalog[physical_index]
                    if unit.unit_id not in wanted:
                        continue
                    kind, _ = self.parse_unit_id(unit.unit_id)
                    raw = tokens
                    if kind == "frame":
                        view_count = seq_len * tokens_per_view
                        raw = tokens[:, :view_count].reshape(
                            bsz * seq_len, tokens_per_view, int(tokens.shape[-1])
                        )
                    value, register_count = self._normalize_capture(
                        unit.unit_id,
                        raw,
                        batch_size=bsz,
                        num_frames=seq_len,
                        patch_tokens=patch_tokens,
                    )
                    value = value.detach()
                    captures[unit.unit_id] = value
                    register_counts.add(register_count)
                    if layer_callback is not None:
                        layer_callback(unit.unit_id, value)
        finally:
            model._restore_original_geometric_input_config()

        if len(register_counts) != 1:
            raise RuntimeError(
                f"Inconsistent MapAnything register counts: {sorted(register_counts)}"
            )
        register_count = next(iter(register_counts))
        meta = {
            "patch_start_idx": register_count,
            "camera_token_count": 0,
            "register_token_count": register_count,
            "patch_token_count": patch_tokens,
            "scale_token_excluded": True,
            "unit_ids": [unit.unit_id for unit in catalog],
        }
        return captures, meta, initial_tokens.detach()

    def extract_layer_units(
        self,
        model: nn.Module,
        images: torch.Tensor,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        units, meta, _initial = self._capture_layer_units_only(
            model,
            images,
            unit_ids=unit_ids,
            layer_callback=layer_callback,
        )
        return units, meta

    def extract_layer_units_and_predictions(
        self, model: nn.Module, images: torch.Tensor
    ) -> Optional[CacheForwardResult]:
        units, meta, prediction, initial_tokens = self._capture_and_predict(
            model,
            images,
            unit_ids=None,
            layer_callback=None,
            with_prediction=True,
        )
        assert prediction is not None
        return CacheForwardResult(
            unit_map=units,
            meta=meta,
            pred=prediction,
            initial_tokens=initial_tokens,
        )

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        _units, _meta, initial_tokens = self._capture_layer_units_only(
            model,
            images,
            unit_ids=set(),
            layer_callback=None,
        )
        return initial_tokens

    def cache_manifest_extra(self, model: nn.Module) -> Dict[str, Any]:
        catalog = self.layer_unit_catalog(model)
        return {
            "unit_ids": [unit.unit_id for unit in catalog],
            "physical_depth": len(catalog),
            "metric_scale_applied_by_backbone": True,
        }

    def supports_incremental_skip_eval(self) -> bool:
        return True

    def _prepare_replay_context(
        self, model: nn.Module, images: torch.Tensor
    ) -> Dict[str, Any]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        views = self._images_to_views(model, images)
        model._configure_geometric_input_config(False, False, False, False, False)
        encoder_features, encoder_registers = model._encode_n_views(views)
        # Match MapAnything.forward(): image encoding and information sharing
        # use AMP, but geometric-input fusion is explicitly fp32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            encoder_features = model._encode_and_fuse_optional_geometric_inputs(
                views, encoder_features
            )
        info = model.info_sharing
        bsz = int(images.shape[0])
        seq_len = int(images.shape[1])
        _b, channels, patch_h, patch_w = encoder_features[0].shape
        patch_tokens = patch_h * patch_w
        register_count = 0
        per_view = []
        for index, feature in enumerate(encoder_features):
            flat = feature.reshape(bsz, channels, patch_tokens)
            if encoder_registers is not None:
                registers = encoder_registers[index]
                register_count = int(registers.shape[2])
                flat = torch.cat([flat, registers], dim=2)
            per_view.append(flat)
        tokens_per_view = patch_tokens + register_count
        view_tokens = torch.stack(per_view, dim=1).permute(0, 1, 3, 2)
        view_tokens = view_tokens.reshape(bsz, seq_len * tokens_per_view, channels)
        scale = model.scale_token.reshape(1, 1, -1).expand(bsz, 1, -1)
        tokens = info.proj_embed(torch.cat([view_tokens, scale], dim=1))

        if bool(getattr(info, "distinguish_ref_and_non_ref_views", False)):
            ref_pe = info.view_pos_table[0].detach().reshape(1, 1, -1)
            tokens[:, :tokens_per_view] = tokens[:, :tokens_per_view] + ref_pe
        if bool(getattr(info, "distinguish_ref_and_non_ref_views", False)) and bool(
            getattr(info, "use_pe_for_non_reference_views", False)
        ):
            if bool(getattr(info, "use_rand_idx_pe_for_non_reference_views", False)):
                raise RuntimeError(
                    "Incremental skip replay requires deterministic non-reference view positions"
                )
            pe = info.view_pos_table[torch.arange(1, seq_len, device=tokens.device)]
            pe = pe.reshape(1, seq_len - 1, 1, -1).expand(
                bsz, -1, tokens_per_view, -1
            )
            tokens[:, tokens_per_view : seq_len * tokens_per_view] += pe.reshape(
                bsz, (seq_len - 1) * tokens_per_view, -1
            )
        return {
            "views": views,
            "encoder_features": encoder_features,
            "tokens": tokens,
            "batch_size": bsz,
            "num_views": seq_len,
            "patch_hw": (patch_h, patch_w),
            "image_hw": (int(images.shape[-2]), int(images.shape[-1])),
            "patch_tokens": patch_tokens,
            "tokens_per_view": tokens_per_view,
        }

    @staticmethod
    def _run_replay_block(
        model: nn.Module,
        tokens: torch.Tensor,
        physical_index: int,
        *,
        batch_size: int,
        num_views: int,
        tokens_per_view: int,
    ) -> torch.Tensor:
        block = model.info_sharing.self_attention_blocks[int(physical_index)]
        positions = [None] * num_views
        if int(physical_index) % 2 == 0:
            return block(tokens, positions)
        view_count = num_views * tokens_per_view
        scale = tokens[:, view_count:]
        frame = tokens[:, :view_count].reshape(
            batch_size * num_views, tokens_per_view, -1
        )
        frame = block(frame, positions)
        return torch.cat(
            [frame.reshape(batch_size, view_count, -1), scale], dim=1
        )

    def _tokens_to_dense_feature(
        self, tokens: torch.Tensor, context: Dict[str, Any]
    ) -> torch.Tensor:
        bsz = int(context["batch_size"])
        seq_len = int(context["num_views"])
        tokens_per_view = int(context["tokens_per_view"])
        patch_tokens = int(context["patch_tokens"])
        patch_h, patch_w = context["patch_hw"]
        views = tokens[:, : seq_len * tokens_per_view].reshape(
            bsz, seq_len, tokens_per_view, -1
        )[:, :, :patch_tokens]
        views = views.reshape(bsz, seq_len, patch_h, patch_w, -1)
        views = views.permute(1, 0, 4, 2, 3).contiguous()
        return views.reshape(bsz * seq_len, int(views.shape[2]), patch_h, patch_w)

    def _prediction_from_replay(
        self,
        model: nn.Module,
        context: Dict[str, Any],
        final_tokens: torch.Tensor,
        intermediate_tokens: Dict[int, torch.Tensor],
    ) -> PredictionBundle:
        info = model.info_sharing
        take_indices = [int(x) for x in info.indices]
        missing = [index for index in take_indices if index not in intermediate_tokens]
        if missing:
            raise RuntimeError(f"Replay missed DPT intermediate layers: {missing}")
        normalized_final = info.norm(final_tokens)
        layered = [torch.cat(context["encoder_features"], dim=0)]
        layered.extend(
            self._tokens_to_dense_feature(intermediate_tokens[index], context)
            for index in take_indices
        )
        layered.append(self._tokens_to_dense_feature(normalized_final, context))
        image_hw = context["image_hw"]
        seq_len = int(context["num_views"])
        view_count = seq_len * int(context["tokens_per_view"])
        scale_features = normalized_final[:, view_count:].permute(0, 2, 1).contiguous()
        # Match MapAnything.forward(): downstream heads and all geometry/output
        # post-processing run outside AMP.
        with torch.autocast(device_type=final_tokens.device.type, enabled=False):
            return self._prediction_from_replay_fp32(
                model,
                context,
                layered=layered,
                scale_features=scale_features,
                image_hw=image_hw,
            )

    def _prediction_from_replay_fp32(
        self,
        model: nn.Module,
        context: Dict[str, Any],
        *,
        layered: List[torch.Tensor],
        scale_features: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> PredictionBundle:
        seq_len = int(context["num_views"])
        dense, pose, scale = model.downstream_head(
            dense_head_inputs=layered,
            scale_head_inputs=scale_features,
            img_shape=image_hw,
            memory_efficient_inference=False,
        )
        if not str(model.scene_rep_type).startswith("raydirs+depth+pose"):
            raise NotImplementedError(
                f"Replay head packing is not implemented for {model.scene_rep_type!r}"
            )
        from mapanything.utils.geometry import (
            convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
        )
        from mapanything.utils.inference import postprocess_model_outputs_for_inference

        dense_value = dense.value.permute(0, 2, 3, 1).contiguous()
        rays, depth = dense_value.split([3, 1], dim=-1)
        trans, quats = pose.value.split([3, 4], dim=-1)
        points = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            rays, depth, trans, quats
        )
        local_points = rays * depth
        conf = None
        if "confidence" in str(model.scene_rep_type):
            conf = dense.confidence.permute(0, 2, 3, 1).squeeze(-1).contiguous()
        masks = logits = None
        if "mask" in str(model.scene_rep_type):
            masks = dense.mask.permute(0, 2, 3, 1).squeeze(-1).contiguous() > 0.5
            logits = dense.logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
        raw_views: List[Dict[str, torch.Tensor]] = []
        for index in range(seq_len):
            sl = slice(index * int(context["batch_size"]), (index + 1) * int(context["batch_size"]))
            scaling = scale
            item: Dict[str, torch.Tensor] = {
                "pts3d": points[sl] * scaling.unsqueeze(-1).unsqueeze(-1),
                "pts3d_cam": local_points[sl] * scaling.unsqueeze(-1).unsqueeze(-1),
                "ray_directions": rays[sl],
                "depth_along_ray": depth[sl] * scaling.unsqueeze(-1).unsqueeze(-1),
                "cam_trans": trans[sl] * scaling,
                "cam_quats": quats[sl],
                "metric_scaling_factor": scaling,
            }
            if conf is not None:
                item["conf"] = conf[sl]
            if masks is not None and logits is not None:
                item["non_ambiguous_mask"] = masks[sl]
                item["non_ambiguous_mask_logits"] = logits[sl]
            raw_views.append(item)
        processed = postprocess_model_outputs_for_inference(
            raw_views,
            context["views"],
            apply_mask=False,
            mask_edges=False,
            apply_confidence_mask=False,
        )
        return self.prediction_bundle_from_raw({"views": processed}, torch.empty(0))

    def iter_skip_span_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
        *,
        defer_cpu_geometry: bool = False,
    ) -> Iterable[Tuple[int, int, PredictionBundle]]:
        del defer_cpu_geometry
        from analysis.amp_policy import autocast_device_dtype

        total = len(self._blocks(model))
        requested: Dict[int, List[int]] = {}
        for start_raw, end_raw in spans:
            start, end = int(start_raw), int(end_raw)
            if start < 0 or end < start or end >= total:
                raise ValueError(f"Invalid skip span ({start},{end}) for {total} units")
            requested.setdefault(start, []).append(end)
        for ends in requested.values():
            ends.sort()

        device = self._model_device(model)
        ac_kw = autocast_device_dtype(device)
        try:
            with torch.inference_mode(), torch.autocast(**ac_kw):
                context = self._prepare_replay_context(model, images)
                prefix = context["tokens"]
            prefix_intermediates: Dict[int, torch.Tensor] = {}
            take_indices = set(int(x) for x in model.info_sharing.indices)
            for start in range(total):
                for end in requested.get(start, []):
                    with torch.inference_mode(), torch.autocast(**ac_kw):
                        candidate = prefix
                        intermediates = dict(prefix_intermediates)
                        for physical_index in range(start, total):
                            if physical_index > end:
                                candidate = self._run_replay_block(
                                    model,
                                    candidate,
                                    physical_index,
                                    batch_size=context["batch_size"],
                                    num_views=context["num_views"],
                                    tokens_per_view=context["tokens_per_view"],
                                )
                            if physical_index in take_indices:
                                intermediates[physical_index] = (
                                    model.info_sharing.norm(candidate)
                                    if bool(model.info_sharing.norm_intermediate)
                                    else candidate
                                )
                        prediction = self._prediction_from_replay(
                            model, context, candidate, intermediates
                        )
                    yield start, end, prediction
                with torch.inference_mode(), torch.autocast(**ac_kw):
                    prefix = self._run_replay_block(
                        model,
                        prefix,
                        start,
                        batch_size=context["batch_size"],
                        num_views=context["num_views"],
                        tokens_per_view=context["tokens_per_view"],
                    )
                    if start in take_indices:
                        prefix_intermediates[start] = (
                            model.info_sharing.norm(prefix)
                            if bool(model.info_sharing.norm_intermediate)
                            else prefix
                        )
        finally:
            model._restore_original_geometric_input_config()


class MapAnythingAdapter(MapAnythingAdapterBase):
    name = "mapanything"
    hf_model_id = "facebook/map-anything"


class MapAnythingV1Adapter(MapAnythingAdapterBase):
    name = "mapanything_v1"
    hf_model_id = "facebook/map-anything-v1"


register(MapAnythingAdapter())
register(MapAnythingV1Adapter())

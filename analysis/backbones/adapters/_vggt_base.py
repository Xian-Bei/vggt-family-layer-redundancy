"""VGGT backbone adapter logic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.prune_plan import UnitPrunePlan, prune_actions_from_plan
from analysis.schema.v3 import LayerUnitSpec, PruneAction
from analysis.recovery.train.model import (
    SkipIdentityBlock,
    inject_lora_attn,
)
from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    PredictionLossWeights,
    PredictionBundle,
    TrainingHyperparams,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def prepend_sys_path(path: Path) -> None:
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def import_vggt_class():
    from vggt.models.vggt import VGGT

    return VGGT


def vggt_family_training_config() -> TrainingHyperparams:
    return TrainingHyperparams(
        optimizer_cls=torch.optim.AdamW,
        lr=5e-5,
        weight_decay=0.05,
        min_lr=1e-8,
        warmup_ratio=0.05,
        grad_clip_max_norm=1.0,
        w_align=1.0,
        w_align_mse=1.0,
        w_align_cos=1.0,
        w_align_cka=0.0,
        w_pred=1.0,
        w_reg=0.0,
    )


def build_vggt_family_training_optimizer(
    trainable: Sequence[nn.Parameter],
    *,
    config: Optional[TrainingHyperparams] = None,
) -> torch.optim.Optimizer:
    cfg = config or vggt_family_training_config()
    return cfg.optimizer_cls(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )


class VGGTFamilyAdapterMixin(BackboneAdapter):
    """Shared implementation for VGGT's frame/global alternating blocks."""

    name: str = "vggt"

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            name=self.name,
            supports_cache=True,
            supports_prune=True,
            supports_lora=True,
            supports_eval=True,
        )

    def training_config(self) -> TrainingHyperparams:
        return vggt_family_training_config()

    def build_training_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        config: Optional[TrainingHyperparams] = None,
    ) -> torch.optim.Optimizer:
        return build_vggt_family_training_optimizer(trainable, config=config or self.training_config())

    def load_eval_images(self, image_paths: Sequence[Path | str]) -> torch.Tensor:
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images([str(p) for p in image_paths], mode="crop")

    def load_driving_depth_stack(
        self,
        depth_paths: Sequence[Path | str],
        *,
        output_hw: Tuple[int, int],
    ) -> Tuple[Any, int]:
        from analysis.datasets.driving_depth import load_driving_depth_stack_vggt

        return load_driving_depth_stack_vggt(depth_paths, output_hw=output_hw)

    def align_7andn_gt_depth(self, view: Any, *, output_hw: Tuple[int, int]) -> Any:
        from analysis.eval.prediction_eval import _load_7andn_gt_depth_vggt

        return _load_7andn_gt_depth_vggt(view, image_hw=output_hw)

    def load_scannet_gt_depth(
        self,
        scene: Any,
        *,
        output_hw: Tuple[int, int],
    ) -> Any:
        from analysis.eval.prediction_eval import _load_gt_depth_for_scene

        return _load_gt_depth_for_scene(scene, image_hw=output_hw)

    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        self._assert_vendor_origin()
        VGGT = import_vggt_class()
        model = VGGT.from_pretrained(str(hf_model))
        model = model.to(device=device)
        # Official VGGT inference keeps fp32 weights and uses autocast for mixed precision.
        if dtype is not None and dtype != torch.float32:
            raise ValueError(
                f"{self.name} expects fp32 weights + autocast; got load dtype={dtype}. Pass dtype=None."
            )
        if eval_mode:
            model.eval()
        return model

    def _expected_vendor_root(self) -> Path:
        return _repo_root()

    def _assert_vendor_origin(self) -> None:
        import importlib
        import importlib.util

        expected = self._expected_vendor_root().resolve()
        last_origin: Optional[Path] = None
        for attempt in range(2):
            spec = importlib.util.find_spec("vggt.models.vggt")
            if spec is None or spec.origin is None:
                raise ImportError(f"Could not resolve vggt.models.vggt for backbone {self.name!r}")
            origin = Path(spec.origin).resolve()
            last_origin = origin
            try:
                origin.relative_to(expected)
                return
            except ValueError:
                if attempt == 0:
                    for key in list(sys.modules):
                        if key == "vggt" or key.startswith("vggt."):
                            sys.modules.pop(key, None)
                    importlib.invalidate_caches()
                    continue
                raise ImportError(
                    f"Backbone {self.name!r} resolved vggt from {origin}, expected it under {expected}. "
                    "Call analysis.backbones.bootstrap.apply_pythonpath() before loading the model."
                )
        raise ImportError(f"Backbone {self.name!r} could not resolve expected VGGT vendor under {expected}; last={last_origin}")

    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        agg = model.aggregator
        depth = len(agg.frame_blocks)
        units: List[LayerUnitSpec] = []
        for i in range(depth):
            units.append(
                LayerUnitSpec(
                    unit_id=f"frame_{i}",
                    kind="frame",
                    index=i,
                    module_path=f"aggregator.frame_blocks.{i}",
                    droppable=True,
                    lora_target=True,
                )
            )
            units.append(
                LayerUnitSpec(
                    unit_id=f"global_{i}",
                    kind="global",
                    index=i,
                    module_path=f"aggregator.global_blocks.{i}",
                    droppable=True,
                    lora_target=True,
                )
            )
        cap = self.capabilities()
        cap.num_frame_blocks = depth
        cap.num_global_blocks = depth
        return units

    def _block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        kind, idx = unit_id.split("_", 1)
        i = int(idx)
        agg = model.aggregator
        if kind == "frame":
            return agg.frame_blocks[i]
        if kind == "global":
            return agg.global_blocks[i]
        raise KeyError(unit_id)

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        return self._block_for_unit(model, unit_id)

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        kind, idx = self.parse_unit_id(unit_id)
        if operation == "drop_unit":
            if kind == "frame":
                model.aggregator.frame_blocks[idx] = SkipIdentityBlock()
            elif kind == "global":
                model.aggregator.global_blocks[idx] = SkipIdentityBlock()
            else:
                raise KeyError(unit_id)
            return None
        if operation == "lora_unit":
            return inject_lora_attn(
                self.block_for_unit(model, unit_id),
                rank=int(kwargs["rank"]),
                alpha=float(kwargs["alpha"]),
                dropout=float(kwargs.get("dropout", 0.0)),
            )
        raise ValueError(f"Unsupported unit operation {operation!r}")

    def extract_layer_units(
        self,
        model: nn.Module,
        images: torch.Tensor,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        return self._extract_layer_units_hooks(
            model, images, unit_ids=unit_ids, layer_callback=layer_callback
        )

    def _extract_layer_units_hooks(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Capture VGGT frame/global blocks with forward hooks."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        bsz, seq_len = int(images.shape[0]), int(images.shape[1])
        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = unit_ids or set(catalog.keys())
        captures: Dict[str, torch.Tensor] = {}
        handles = []

        def _make_hook(uid: str):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t = self.normalize_unit_activation(
                    uid,
                    t,
                    batch_size=bsz,
                    num_frames=seq_len,
                )
                captures[uid] = t.detach()
                if layer_callback is not None:
                    layer_callback(uid, t.detach())

            return hook

        for uid in want:
            if uid not in catalog:
                continue
            handles.append(self._block_for_unit(model, uid).register_forward_hook(_make_hook(uid)))

        try:
            with torch.no_grad():
                _tokens_list, patch_start_idx = model.aggregator(images)
        finally:
            for h in handles:
                h.remove()

        out = self._normalize_unit_map(model, captures)
        concat_list = self._concat_layers_from_unit_map(model, out)
        meta = {
            "patch_start_idx": int(patch_start_idx),
            "camera_token_count": 1,
            "aggregated_concat": concat_list,
        }
        return out, meta

    def _normalize_unit_map(
        self,
        model: nn.Module,
        captures: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = dict(captures)
        for i in range(len(getattr(model.aggregator, "global_blocks", []))):
            fid, gid = f"frame_{i}", f"global_{i}"
            if fid not in out or gid not in out:
                continue
            ft, gt = out[fid], out[gid]
            if ft.dim() == 3:
                ft = ft.unsqueeze(0)
            if gt.dim() == 3:
                gt = gt.unsqueeze(0)
            if ft.dim() == 4 and gt.dim() == 4 and ft.shape != gt.shape:
                b, s, p, c = (int(x) for x in ft.shape)
                if int(gt.shape[-1]) == c and int(gt.shape[1]) == 1 and int(gt.shape[2]) == s * p:
                    gt = gt.reshape(b, s, p, c)
                elif int(gt.shape[-1]) == c and gt.numel() == ft.numel():
                    gt = gt.reshape_as(ft)
            out[fid] = ft.contiguous()
            out[gid] = gt.contiguous()
        return out

    def _concat_layers_from_unit_map(
        self,
        model: nn.Module,
        unit_map: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        concat_list: List[torch.Tensor] = []
        for i in range(len(getattr(model.aggregator, "global_blocks", []))):
            fid, gid = f"frame_{i}", f"global_{i}"
            if fid not in unit_map or gid not in unit_map:
                continue
            concat_list.append(torch.cat([unit_map[fid], unit_map[gid]], dim=-1).detach().cpu().contiguous())
        return concat_list

    def run_heads_from_aggregated_tokens(
        self,
        model: nn.Module,
        aggregated_tokens_list: Sequence[Any],
        patch_start_idx: int,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        predictions: Dict[str, Any] = {}

        # Match the official VGGT precision boundary: the aggregator inherits
        # the caller's AMP context, while camera/depth/point heads run in fp32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            if getattr(model, "camera_head", None) is not None:
                pose_enc_list = model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if getattr(model, "depth_head", None) is not None:
                depth, depth_conf = model.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=int(patch_start_idx)
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if getattr(model, "point_head", None) is not None:
                pts3d, pts3d_conf = model.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=int(patch_start_idx)
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if getattr(model, "track_head", None) is not None and query_points is not None:
            track_list, vis, conf = model.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=int(patch_start_idx),
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not model.training:
            predictions["images"] = images

        return predictions

    def cache_manifest_extra(self, model: nn.Module) -> Dict[str, Any]:
        if hasattr(model, "aggregator") and hasattr(model.aggregator.global_blocks[0], "attn"):
            return {
                "num_heads": int(getattr(model.aggregator.global_blocks[0].attn, "num_heads", 0)),
            }
        return {}

    def minimal_prune_actions(self, *, cache_ref: str, dataset: str) -> List[Dict[str, Any]]:
        return [
            {"action": "drop_unit", "unit_id": "global_7"},
            {"action": "drop_unit", "unit_id": "global_8"},
            {"action": "lora_unit", "unit_id": "frame_6", "rank": 8, "alpha": 16.0},
        ]

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

    def apply_checkpoint_structure(
        self,
        model: nn.Module,
        payload: Dict[str, Any],
        *,
        prune_plan: Optional[Any] = None,
    ) -> None:
        ckpt_plan_raw = payload.get("prune_plan")
        if isinstance(ckpt_plan_raw, dict):
            tmp = UnitPrunePlan.from_dict(ckpt_plan_raw)
            dacts, lacts = prune_actions_from_plan(tmp)
            self.apply_prune_plan(model, dacts)
            self.inject_lora_plan(model, lacts)
        elif isinstance(prune_plan, UnitPrunePlan):
            dacts, lacts = prune_actions_from_plan(prune_plan)
            self.apply_prune_plan(model, dacts)
            self.inject_lora_plan(model, lacts)
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

    def forward_predictions(self, model: nn.Module, images: torch.Tensor) -> PredictionBundle:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)
        preds = self.run_heads_from_aggregated_tokens(
            model, aggregated_tokens_list, int(patch_start_idx), images
        )
        return self.prediction_bundle_from_raw(dict(preds) if isinstance(preds, dict) else {}, images)

    def supports_incremental_skip_eval(self) -> bool:
        return True

    @staticmethod
    def _run_incremental_skip_unit(
        model: nn.Module,
        tokens_bspc: torch.Tensor,
        unit_idx: int,
        *,
        pos_frame: Optional[torch.Tensor],
        pos_global: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one public frame/global unit while preserving the aggregator layouts."""
        from analysis.backbones.cache_replay.vggt_family import run_block_with_checkpoint

        bsz, seq_len, num_tokens, channels = (int(x) for x in tokens_bspc.shape)
        layer_idx, half = divmod(int(unit_idx), 2)
        if half == 0:
            frame_in = tokens_bspc.reshape(bsz * seq_len, num_tokens, channels)
            frame_out = run_block_with_checkpoint(
                model.aggregator.frame_blocks[layer_idx],
                frame_in,
                pos=pos_frame,
                training=False,
                use_checkpoint=False,
                use_reentrant=False,
            )
            return frame_out.reshape(bsz, seq_len, num_tokens, channels)
        global_in = tokens_bspc.reshape(bsz, seq_len * num_tokens, channels)
        global_out = run_block_with_checkpoint(
            model.aggregator.global_blocks[layer_idx],
            global_in,
            pos=pos_global,
            training=False,
            use_checkpoint=False,
            use_reentrant=False,
        )
        return global_out.reshape(bsz, seq_len, num_tokens, channels)

    def _predict_incremental_skip_span(
        self,
        model: nn.Module,
        images: torch.Tensor,
        *,
        prefix_tokens: torch.Tensor,
        prefix_layers: Dict[int, torch.Tensor],
        start_unit: int,
        end_unit: int,
        total_units: int,
        required_layers: Set[int],
        pos_frame: Optional[torch.Tensor],
        pos_global: Optional[torch.Tensor],
    ) -> PredictionBundle:
        tokens = prefix_tokens
        selected = dict(prefix_layers)
        frame_at_layer: Optional[torch.Tensor] = tokens if int(start_unit) % 2 == 1 else None
        for unit_idx in range(int(start_unit), int(total_units)):
            layer_idx, half = divmod(unit_idx, 2)
            if unit_idx > int(end_unit):
                tokens = self._run_incremental_skip_unit(
                    model,
                    tokens,
                    unit_idx,
                    pos_frame=pos_frame,
                    pos_global=pos_global,
                )
            if half == 0:
                frame_at_layer = tokens
            else:
                if frame_at_layer is None:
                    raise RuntimeError(f"Missing frame state before global unit {unit_idx}")
                if layer_idx in required_layers:
                    selected[layer_idx] = torch.cat([frame_at_layer, tokens], dim=-1)
                frame_at_layer = None

        num_layers = int(total_units) // 2
        missing = sorted(required_layers.difference(selected))
        if missing:
            raise RuntimeError(f"Incremental skip replay missed head layers: {missing}")
        aggregated: List[Any] = [None] * num_layers
        for layer_idx, tensor in selected.items():
            aggregated[int(layer_idx)] = tensor
        raw = self.run_heads_from_aggregated_tokens(
            model,
            aggregated,
            int(model.aggregator.patch_start_idx),
            images,
        )
        return self.prediction_bundle_from_raw(raw, images)

    def iter_skip_span_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int, PredictionBundle]]:
        """Stream exact VGGT span predictions while computing baseline prefixes once."""
        from analysis.amp_policy import autocast_device_dtype
        from analysis.backbones.cache_replay.vggt_family import build_pos

        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected skip-eval images [B,S,3,H,W], got {tuple(images.shape)}")

        unit_ids = [u.unit_id for u in self.layer_unit_catalog(model)]
        total_units = len(unit_ids)
        if total_units == 0 or total_units % 2 != 0:
            raise RuntimeError(f"VGGT incremental replay requires paired units, got {total_units}")
        aa_order = list(getattr(model.aggregator, "aa_order", ["frame", "global"]))
        aa_block_size = int(getattr(model.aggregator, "aa_block_size", 1))
        if aa_order != ["frame", "global"] or aa_block_size != 1:
            raise RuntimeError(
                "VGGT incremental skip replay currently requires "
                f"aa_order=['frame','global'] and aa_block_size=1; got {aa_order}, {aa_block_size}"
            )
        requested: Dict[int, List[int]] = {}
        for i_raw, j_raw in spans:
            i, j = int(i_raw), int(j_raw)
            if i < 0 or j < i or j >= total_units:
                raise ValueError(f"Invalid skip span ({i},{j}) for {total_units} units")
            requested.setdefault(i, []).append(j)
        for ends in requested.values():
            ends.sort()

        device = images.device
        ac_kw = autocast_device_dtype(device)
        with torch.inference_mode(), torch.autocast(**ac_kw):
            pos_frame, pos_global = build_pos(model, images)
            prefix_tokens = self.build_initial_tokens(model, images)
        if prefix_tokens is None:
            raise RuntimeError("VGGT adaptor did not produce initial replay tokens")

        required_layers = set(
            int(x) for x in self.required_head_layers_for_cache(model, total_units // 2)
        )
        prefix_layers: Dict[int, torch.Tensor] = {}
        baseline_frame: Optional[torch.Tensor] = None

        for start_unit in range(total_units):
            for end_unit in requested.get(start_unit, []):
                with torch.inference_mode(), torch.autocast(**ac_kw):
                    prediction = self._predict_incremental_skip_span(
                        model,
                        images,
                        prefix_tokens=prefix_tokens,
                        prefix_layers=prefix_layers,
                        start_unit=start_unit,
                        end_unit=end_unit,
                        total_units=total_units,
                        required_layers=required_layers,
                        pos_frame=pos_frame,
                        pos_global=pos_global,
                    )
                yield start_unit, end_unit, prediction

            with torch.inference_mode(), torch.autocast(**ac_kw):
                prefix_tokens = self._run_incremental_skip_unit(
                    model,
                    prefix_tokens,
                    start_unit,
                    pos_frame=pos_frame,
                    pos_global=pos_global,
                )
                layer_idx, half = divmod(start_unit, 2)
                if half == 0:
                    baseline_frame = prefix_tokens
                else:
                    if baseline_frame is None:
                        raise RuntimeError(f"Missing baseline frame state at layer {layer_idx}")
                    if layer_idx in required_layers:
                        prefix_layers[layer_idx] = torch.cat(
                            [baseline_frame, prefix_tokens], dim=-1
                        )
                    baseline_frame = None

    def _prediction_bundle_from_preds(self, preds: Dict[str, Any], images: torch.Tensor) -> PredictionBundle:
        return self.prediction_bundle_from_raw(dict(preds) if isinstance(preds, dict) else {}, images)

    def prediction_bundle_from_raw(
        self,
        raw: Dict[str, Any],
        images: torch.Tensor,
    ) -> PredictionBundle:
        from analysis.backbones.prediction import bundle_from_pose_depth

        return bundle_from_pose_depth(raw=raw, images=images, backbone_name=self.name)

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        from analysis.backbones.adapters._vggt_collect import _aggregator_initial_tokens

        return _aggregator_initial_tokens(model, images).detach().contiguous()

    def extract_layer_units_and_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Optional[CacheForwardResult]:
        agg = getattr(model, "aggregator", None)
        if agg is None:
            return None
        if images.dim() == 4:
            images = images.unsqueeze(0)
        bsz, seq_len = int(images.shape[0]), int(images.shape[1])
        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = set(catalog.keys())
        captures: Dict[str, torch.Tensor] = {}
        initial_tokens: Optional[torch.Tensor] = None
        handles = []

        def _capture_initial(_mod, args):
            nonlocal initial_tokens
            value = args[0] if args else None
            while isinstance(value, (tuple, list)) and value:
                value = value[0]
            if isinstance(value, torch.Tensor):
                initial_tokens = self.normalize_unit_activation(
                    "frame_0",
                    value,
                    batch_size=bsz,
                    num_frames=seq_len,
                ).detach().contiguous()

        def _make_hook(uid: str):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t = self.normalize_unit_activation(
                    uid,
                    t,
                    batch_size=bsz,
                    num_frames=seq_len,
                )
                captures[uid] = t.detach()

            return hook

        for uid in want:
            if uid not in catalog:
                continue
            handles.append(self._block_for_unit(model, uid).register_forward_hook(_make_hook(uid)))
        handles.append(self._block_for_unit(model, "frame_0").register_forward_pre_hook(_capture_initial))

        try:
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
        finally:
            for h in handles:
                h.remove()

        unit_map = self._normalize_unit_map(model, captures)
        concat_list = self._concat_layers_from_unit_map(model, unit_map)

        preds = self.run_heads_from_aggregated_tokens(
            model, aggregated_tokens_list, int(patch_start_idx), images
        )
        meta = {
            "patch_start_idx": int(patch_start_idx),
            "camera_token_count": 1,
            "aggregated_concat": concat_list,
        }
        pred = self._prediction_bundle_from_preds(preds, images)
        if initial_tokens is None:
            raise RuntimeError("VGGT forward did not expose the layer_-1 input to frame_0")
        return CacheForwardResult(
            unit_map=unit_map,
            meta=meta,
            pred=pred,
            initial_tokens=initial_tokens,
        )

    def apply_prune_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> nn.Module:
        for act in actions:
            if act.action != "drop_unit":
                continue
            self.apply_unit_operation(model, act.unit_id, "drop_unit")
        return model

    def inject_lora_plan(self, model: nn.Module, actions: Sequence[PruneAction]) -> List[nn.Parameter]:
        trainable: List[nn.Parameter] = []
        for act in actions:
            if act.action != "lora_unit":
                continue
            trainable.extend(
                self.apply_unit_operation(
                    model,
                    act.unit_id,
                    "lora_unit",
                    rank=int(act.rank),
                    alpha=float(act.alpha),
                    dropout=float(act.dropout),
                )
            )
        return trainable

    def required_head_layers_for_cache(self, model: nn.Module, num_layers: int) -> List[int]:
        from analysis.backbones.cache_replay.vggt_family import required_head_layers

        return list(required_head_layers(model, num_layers))

    def validate_cached_prediction(self, pred: Dict[str, Any]) -> None:
        required = ("pose_enc", "depth", "depth_conf")
        missing = [k for k in required if k not in pred]
        if missing:
            raise SystemExit(f"Invalid cache row pred: missing keys {missing}")

    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
        dataset_name: str = "",
    ) -> Dict[str, torch.Tensor]:
        from analysis.backbones.cache_replay.vggt_family import build_target_batch_from_teacher

        return build_target_batch_from_teacher(
            teacher_pred,
            images_batched=images_batched,
            device=device,
            backbone_name=self.name,
            dataset_name=dataset_name,
        )

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, Any],
        *,
        weights: PredictionLossWeights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        from analysis.backbones.cache_replay.vggt_family import prediction_official_camera_depth

        return prediction_official_camera_depth(
            student_pred,
            target_batch,  # type: ignore[arg-type]
            w_pred=float(weights.w_pred),
        )

    def replay_student_cache_first(
        self,
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        num_layers: int,
        **kwargs: Any,
    ) -> Any:
        from analysis.backbones.cache_replay.vggt_family import _replay_model_cache_first

        return _replay_model_cache_first(
            model,
            get_layer_cpu,
            num_layers,
            backbone_name=self.name,
            **kwargs,
        )

    def cache_replay_engine(self) -> Any:
        from analysis.backbones.cache_replay.vggt_family import VGGTCacheReplayEngine

        return VGGTCacheReplayEngine(backbone_name=self.name)

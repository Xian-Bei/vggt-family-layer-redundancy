"""Shared VGGT-family adapter logic (repo vggt or FastVGGT vendor)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.prune_plan import UnitPrunePlan, prune_actions_from_plan
from analysis.schema.v3 import LayerUnitSpec, PruneAction
from analysis.train.model import (
    SkipIdentityBlock,
    inject_lora_attn,
)
from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    EvalSceneContext,
    PredictionLossWeights,
    PredictionBundle,
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


class VGGTFamilyAdapterMixin(BackboneAdapter):
    """Mixin for VGGT / FastVGGT backends with frame_blocks + global_blocks."""

    name: str = "vggt"

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            name=self.name,
            supports_cache=True,
            supports_prune=True,
            supports_lora=True,
            supports_eval=True,
        )

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
        """FastVGGT-style aggregators without ``return_layers``: hook frame/global blocks."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = unit_ids or set(catalog.keys())
        captures: Dict[str, torch.Tensor] = {}
        handles = []

        def _make_hook(uid: str):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.dim() == 3:
                    t = t.unsqueeze(0)
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

    def eval_scene(
        self,
        model: nn.Module,
        *,
        dataset_name: str,
        batch: Any,
        ctx: EvalSceneContext,
    ) -> Dict[str, float]:
        from analysis.backbones import eval_impl

        if dataset_name == "scannet":
            return eval_impl.eval_scannet_points_adapter(self, model, batch, ctx.device, ctx.profile, ctx)
        if dataset_name in ("7scenes", "nrgbd"):
            return eval_impl.eval_7andn_points_adapter(self, model, batch, ctx.device, ctx.profile, ctx)
        raise ValueError(f"Unsupported dataset for VGGT-family eval: {dataset_name}")

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
        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = set(catalog.keys())
        captures: Dict[str, torch.Tensor] = {}
        handles = []

        def _make_hook(uid: str):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                captures[uid] = t.detach()

            return hook

        for uid in want:
            if uid not in catalog:
                continue
            handles.append(self._block_for_unit(model, uid).register_forward_hook(_make_hook(uid)))

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
            "aggregated_concat": concat_list,
        }
        pred = self._prediction_bundle_from_preds(preds, images)
        return CacheForwardResult(unit_map=unit_map, meta=meta, pred=pred, initial_tokens=None)

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
    ) -> Dict[str, torch.Tensor]:
        from analysis.backbones.cache_replay.vggt_family import build_target_batch_from_teacher

        return build_target_batch_from_teacher(
            teacher_pred,
            images_batched=images_batched,
            device=device,
            backbone_name=self.name,
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
            w_camera=float(weights.w_camera),
            w_depth=float(weights.w_depth),
            camera_loss_type=str(weights.camera_loss_type),
            depth_valid_range=float(weights.depth_valid_range),
            depth_gradient_loss_fn=str(weights.depth_gradient_loss_fn),
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

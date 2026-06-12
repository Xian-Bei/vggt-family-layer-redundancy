"""Pi3 backbone adapter: maps decoder blocks to frame_i/global_i semantics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    EvalSceneContext,
    PredictionLossWeights,
    PredictionBundle,
)
from analysis.backbones.registry import register
from analysis.schema.v3 import LayerUnitSpec, PruneAction
from analysis.train.model import inject_lora_attn


def _pi3_root() -> Path:
    return Path(__file__).resolve().parents[3] / "backbones" / "Pi3"


def _pi3_curope_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor_extensions" / "pi3_curope"


def _ensure_pi3_path() -> None:
    for path in (_pi3_root(), _pi3_curope_root()):
        s = str(path.resolve())
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


class _Pi3SkipBlock(nn.Module):
    """Identity residual block matching BlockRope call signature."""

    def forward(self, x: torch.Tensor, xpos=None, **kwargs) -> torch.Tensor:
        return x


class Pi3Adapter(BackboneAdapter):
    name = "pi3"

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            name=self.name,
            supports_cache=True,
            supports_prune=True,
            supports_lora=True,
            supports_eval=True,
        )

    def default_hf_model(self) -> str:
        return "yyfz233/Pi3"

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        return [
            repo_root / "analysis" / "backbones" / "vendor_extensions" / "pi3_curope",
            repo_root / "backbones" / "Pi3",
            repo_root,
        ]

    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        _ensure_pi3_path()
        from pi3.models.pi3 import Pi3

        model = Pi3.from_pretrained(str(hf_model)) if hasattr(Pi3, "from_pretrained") else Pi3()
        model = model.to(device=device)
        # Official Pi3 keeps fp32 weights; heads run fp32 under autocast(enabled=False) inside forward.
        if dtype is not None and dtype != torch.float32:
            raise ValueError(
                "pi3 expects fp32 weights + autocast; got load dtype="
                f"{dtype}. Pass dtype=None."
            )
        if eval_mode:
            model.eval()
        return model

    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        depth = len(model.decoder)
        units: List[LayerUnitSpec] = []
        n_pairs = (depth + 1) // 2
        for p in range(n_pairs):
            fi, gi = 2 * p, 2 * p + 1
            if fi < depth:
                units.append(
                    LayerUnitSpec(
                        unit_id=f"frame_{p}",
                        kind="frame",
                        index=p,
                        module_path=f"decoder.{fi}",
                        droppable=True,
                        lora_target=True,
                    )
                )
            if gi < depth:
                units.append(
                    LayerUnitSpec(
                        unit_id=f"global_{p}",
                        kind="global",
                        index=p,
                        module_path=f"decoder.{gi}",
                        droppable=True,
                        lora_target=True,
                    )
                )
        return units

    def _decoder_index(self, unit_id: str) -> int:
        kind, idx_s = unit_id.split("_", 1)
        p = int(idx_s)
        return 2 * p if kind == "frame" else 2 * p + 1

    def _forward_device(self, model: nn.Module, images: torch.Tensor) -> torch.device:
        if torch.is_tensor(images) and images.device.type != "cpu":
            return images.device
        for p in model.parameters():
            if p.device.type != "cpu":
                return p.device
        return next(model.parameters()).device

    def _prepare_full_forward(self, model: nn.Module, device: torch.device) -> bool:
        encoder = getattr(model, "encoder", None)
        restore_encoder_cpu = bool(
            encoder is not None and any(p.device.type == "cpu" for p in encoder.parameters())
        )
        model.to(device=device)
        return restore_encoder_cpu

    def _restore_full_forward(self, model: nn.Module, restore_encoder_cpu: bool) -> None:
        if not restore_encoder_cpu:
            return
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        idx = self._decoder_index(unit_id)
        if idx >= len(model.decoder):
            raise KeyError(unit_id)
        return model.decoder[idx]

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        idx = self._decoder_index(unit_id)
        if idx >= len(model.decoder):
            raise KeyError(unit_id)
        if operation == "drop_unit":
            model.decoder[idx] = _Pi3SkipBlock()
            return None
        if operation == "lora_unit":
            return inject_lora_attn(
                model.decoder[idx],
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
        captures: Dict[str, torch.Tensor] = {}
        handles = []

        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = unit_ids or set(catalog.keys())

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
            idx = self._decoder_index(uid)
            if idx < len(model.decoder):
                handles.append(model.decoder[idx].register_forward_hook(_make_hook(uid)))

        if images.dim() == 4:
            images = images.unsqueeze(0)
        device = self._forward_device(model, images)
        restore_encoder_cpu = self._prepare_full_forward(model, device)
        images = images.to(device=device, dtype=torch.float32)
        try:
            from analysis.amp_policy import autocast_device_dtype

            with torch.no_grad(), torch.autocast(**autocast_device_dtype(device)):
                _ = model(images)
        finally:
            self._restore_full_forward(model, restore_encoder_cpu)

        for h in handles:
            h.remove()

        return captures, {"patch_start_idx": int(getattr(model, "patch_start_idx", 0))}

    def forward_predictions(self, model: nn.Module, images: torch.Tensor) -> PredictionBundle:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        device = self._forward_device(model, images)
        restore_encoder_cpu = self._prepare_full_forward(model, device)
        images = images.to(device=device, dtype=torch.float32)
        try:
            from analysis.amp_policy import autocast_device_dtype

            with torch.no_grad(), torch.autocast(**autocast_device_dtype(device)):
                out = model(images)
        finally:
            self._restore_full_forward(model, restore_encoder_cpu)
        return self.prediction_bundle_from_raw(dict(out), images)

    def prediction_bundle_from_raw(
        self,
        raw: Dict[str, Any],
        images: torch.Tensor,
    ) -> PredictionBundle:
        del images
        from analysis.backbones.prediction import bundle_from_pi3_raw

        return bundle_from_pi3_raw(raw)

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        device = self._forward_device(model, images)
        restore_encoder_cpu = self._prepare_full_forward(model, device)
        images = images.to(device=device, dtype=torch.float32)
        bsz, seq_len, channels, height, width = (int(x) for x in images.shape)
        imgs = (images - model.image_mean) / model.image_std
        x = imgs.reshape(bsz * seq_len, channels, height, width)
        try:
            from analysis.amp_policy import autocast_device_dtype

            with torch.no_grad(), torch.autocast(**autocast_device_dtype(device)):
                hidden = model.encoder(x, is_training=True)
        finally:
            self._restore_full_forward(model, restore_encoder_cpu)
        if isinstance(hidden, dict):
            hidden = hidden.get("x_norm_patchtokens", hidden.get("x"))
        if not isinstance(hidden, torch.Tensor):
            return None
        register = model.register_token.repeat(bsz, seq_len, 1, 1)
        register = register.reshape(bsz * seq_len, *model.register_token.shape[-2:])
        tokens = torch.cat([register, hidden], dim=1)
        return tokens.view(bsz, seq_len, int(tokens.shape[1]), int(tokens.shape[2])).detach().contiguous()

    def extract_layer_units_and_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Optional[CacheForwardResult]:
        captures: Dict[str, torch.Tensor] = {}
        handles = []

        catalog = {u.unit_id: u for u in self.layer_unit_catalog(model)}
        want = set(catalog.keys())

        def _make_hook(uid: str):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                captures[uid] = t.detach()

            return hook

        for uid in want:
            idx = self._decoder_index(uid)
            if idx < len(model.decoder):
                handles.append(model.decoder[idx].register_forward_hook(_make_hook(uid)))

        if images.dim() == 4:
            images = images.unsqueeze(0)
        device = self._forward_device(model, images)
        restore_encoder_cpu = self._prepare_full_forward(model, device)
        images = images.to(device=device, dtype=torch.float32)
        try:
            from analysis.amp_policy import autocast_device_dtype

            with torch.no_grad(), torch.autocast(**autocast_device_dtype(device)):
                out = model(images)
        finally:
            self._restore_full_forward(model, restore_encoder_cpu)

        for h in handles:
            h.remove()

        pred = self.prediction_bundle_from_raw(dict(out), images)
        init_tokens = self.build_initial_tokens(model, images)
        meta = {"patch_start_idx": int(getattr(model, "patch_start_idx", 0))}
        return CacheForwardResult(unit_map=captures, meta=meta, pred=pred, initial_tokens=init_tokens)

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

    def minimal_prune_actions(self, *, cache_ref: str, dataset: str) -> List[Dict[str, Any]]:
        return [
            {"action": "drop_unit", "unit_id": "global_1"},
            {"action": "lora_unit", "unit_id": "frame_0", "rank": 8, "alpha": 16.0},
        ]

    def build_train_dummy_images(
        self,
        row: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        get_layer_cpu: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        h, w, _, _ = self.infer_train_cache_geometry(row, model, get_layer_cpu)
        from analysis.backbones.train_cache_replay import build_pi3_dummy_images

        return build_pi3_dummy_images(h=h, w=w, device=device)

    def train_replay_extra_kwargs(self, *, h: int, w: int) -> Dict[str, Any]:
        return {"image_hw": (int(h), int(w))}

    def after_train_replay_release_cpu(
        self,
        layer_cpu_cache: Dict[int, torch.Tensor],
        *,
        rules: Sequence[Any],
        prev_layers: Sequence[int],
        num_layers: int,
        seed_layer: int = 0,
    ) -> None:
        from analysis.backbones.cache_replay.pi3 import Pi3CacheReplayEngine

        del prev_layers, num_layers
        keep = Pi3CacheReplayEngine.layers_to_keep_in_cpu_cache(
            seed_layer=int(seed_layer),
            alignment_layers=[int(x) for x in rules],  # type: ignore[arg-type]
        )
        Pi3CacheReplayEngine.release_layer_cpu_cache(layer_cpu_cache, keep)

    def periodic_eval_skip_depth(self) -> bool:
        return True

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
        raise ValueError(f"Unsupported dataset for Pi3 eval: {dataset_name}")

    def validate_cached_prediction(self, pred: Dict[str, Any]) -> None:
        required = ("points", "camera_poses")
        missing = [k for k in required if k not in pred]
        if missing:
            raise SystemExit(f"Invalid Pi3 cache row pred: missing keys {missing}")

    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        from analysis.backbones.cache_replay.pi3 import build_target_batch_from_teacher_pi3

        return build_target_batch_from_teacher_pi3(
            teacher_pred,
            images_batched=images_batched,
            device=device,
        )

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, Any],
        *,
        weights: PredictionLossWeights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        from analysis.backbones.cache_replay.pi3 import prediction_points_conf

        return prediction_points_conf(
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
        from analysis.backbones.cache_replay.pi3 import _replay_pi3_cache_first

        return _replay_pi3_cache_first(model, get_layer_cpu, num_layers, **kwargs)

    def cache_replay_engine(self) -> Any:
        from analysis.backbones.cache_replay.pi3 import Pi3CacheReplayEngine

        return Pi3CacheReplayEngine()


register(Pi3Adapter())

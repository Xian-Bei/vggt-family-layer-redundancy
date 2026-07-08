"""Pi3 backbone adapter: maps decoder blocks to frame_i/global_i semantics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from analysis.backbones.contracts import (
    BackboneAdapter,
    BackboneCapabilities,
    CacheForwardResult,
    PredictionLossWeights,
    PredictionBundle,
    TrainingHyperparams,
)
from analysis.backbones.registry import register
from analysis.schema.v3 import LayerUnitSpec, PruneAction
from analysis.recover.train.model import inject_lora_attn


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


def pi3_official_training_config() -> TrainingHyperparams:
    """Pi3 official training defaults adapted to cache-replay LoRA training.

    The official code uses AdamW(lr=5e-5, wd=5e-2), grad clip 1.0, bf16
    autocast, and OneCycleLR with pct_start=0.0. Pi3 prediction loss is the
    official Pi3Loss, whose camera term is already weighted by 0.1 internally.
    """

    return TrainingHyperparams(
        optimizer_cls=torch.optim.AdamW,
        lr=5e-5,
        weight_decay=5e-2,
        min_lr=5e-10,
        warmup_ratio=0.0,
        grad_clip_max_norm=1.0,
        w_align=1.0,
        w_align_mse=1.0,
        w_align_cos=1.0,
        w_align_cka=0.0,
        w_pred=1.0,
        w_reg=0.0,
    )


def build_pi3_official_training_optimizer(
    trainable: Sequence[nn.Parameter],
    *,
    config: Optional[TrainingHyperparams] = None,
) -> torch.optim.Optimizer:
    cfg = config or pi3_official_training_config()
    return cfg.optimizer_cls(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )


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

    def training_config(self) -> TrainingHyperparams:
        return pi3_official_training_config()

    def build_training_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        config: Optional[TrainingHyperparams] = None,
    ) -> torch.optim.Optimizer:
        return build_pi3_official_training_optimizer(trainable, config=config or self.training_config())

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        return [
            repo_root / "analysis" / "backbones" / "vendor_extensions" / "pi3_curope",
            repo_root / "backbones" / "Pi3",
            repo_root,
        ]

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

    def _normalize_decoder_capture(
        self,
        unit_id: str,
        tensor: torch.Tensor,
        *,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Return decoder block activations as [B, N, tokens_per_frame, C]."""
        kind, _ = unit_id.split("_", 1)
        if tensor.dim() == 4:
            if int(tensor.shape[0]) == bsz and int(tensor.shape[1]) == seq_len:
                return tensor.contiguous()
            if int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == bsz * seq_len:
                return tensor.reshape(bsz, seq_len, int(tensor.shape[2]), int(tensor.shape[3])).contiguous()
            if int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == bsz and int(tensor.shape[2]) % seq_len == 0:
                tokens_per_frame = int(tensor.shape[2]) // seq_len
                return tensor.reshape(bsz, seq_len, tokens_per_frame, int(tensor.shape[3])).contiguous()
            raise ValueError(f"Unexpected Pi3 {unit_id} activation shape {tuple(tensor.shape)}")
        if tensor.dim() != 3:
            raise ValueError(f"Unexpected Pi3 {unit_id} activation shape {tuple(tensor.shape)}")

        if kind == "frame":
            if int(tensor.shape[0]) != bsz * seq_len:
                raise ValueError(
                    f"Unexpected Pi3 frame activation batch for {unit_id}: "
                    f"{tuple(tensor.shape)} with B={bsz}, N={seq_len}"
                )
            return tensor.reshape(bsz, seq_len, int(tensor.shape[1]), int(tensor.shape[2])).contiguous()

        if kind == "global":
            if int(tensor.shape[0]) != bsz or int(tensor.shape[1]) % seq_len != 0:
                raise ValueError(
                    f"Unexpected Pi3 global activation layout for {unit_id}: "
                    f"{tuple(tensor.shape)} with B={bsz}, N={seq_len}"
                )
            tokens_per_frame = int(tensor.shape[1]) // seq_len
            return tensor.reshape(bsz, seq_len, tokens_per_frame, int(tensor.shape[2])).contiguous()

        raise KeyError(unit_id)

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
                t = self._normalize_decoder_capture(uid, t, bsz=bsz, seq_len=seq_len).detach()
                captures[uid] = t
                if layer_callback is not None:
                    layer_callback(uid, t)

            return hook

        if images.dim() == 4:
            images = images.unsqueeze(0)
        bsz, seq_len = int(images.shape[0]), int(images.shape[1])
        for uid in want:
            if uid not in catalog:
                continue
            idx = self._decoder_index(uid)
            if idx < len(model.decoder):
                handles.append(model.decoder[idx].register_forward_hook(_make_hook(uid)))

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

        return captures, {
            "patch_start_idx": int(getattr(model, "patch_start_idx", 0)),
            "camera_token_count": 0,
        }

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

    def supports_incremental_skip_eval(self) -> bool:
        return True

    @staticmethod
    def _run_incremental_skip_unit(
        model: nn.Module,
        tokens_bspc: torch.Tensor,
        unit_idx: int,
        *,
        pos_bnpc: torch.Tensor,
    ) -> torch.Tensor:
        from analysis.backbones.cache_replay.pi3 import _run_pi3_block

        bsz, seq_len, num_tokens, channels = (int(x) for x in tokens_bspc.shape)
        idx = int(unit_idx)
        if idx < 0 or idx >= len(model.decoder):
            raise IndexError(f"Pi3 decoder unit index out of range: {idx}")
        if idx % 2 == 0:
            hidden = tokens_bspc.reshape(bsz * seq_len, num_tokens, channels)
            pos = pos_bnpc.reshape(bsz * seq_len, num_tokens, -1)
            out = _run_pi3_block(
                model.decoder[idx],
                hidden,
                pos,
                training=False,
                use_checkpoint=False,
            )
            return out.reshape(bsz, seq_len, num_tokens, channels)
        hidden = tokens_bspc.reshape(bsz, seq_len * num_tokens, channels)
        pos = pos_bnpc.reshape(bsz, seq_len * num_tokens, -1)
        out = _run_pi3_block(
            model.decoder[idx],
            hidden,
            pos,
            training=False,
            use_checkpoint=False,
        )
        return out.reshape(bsz, seq_len, num_tokens, channels)

    @staticmethod
    def _pi3_head_unit_indices(model: nn.Module) -> Tuple[int, int]:
        dec_depth = int(len(model.decoder))
        if dec_depth < 2:
            raise RuntimeError(f"Pi3 skip eval needs at least two decoder blocks, got {dec_depth}")
        return dec_depth - 2, dec_depth - 1

    def _forward_pi3_heads_from_decoder_outputs(
        self,
        model: nn.Module,
        *,
        head_outputs: Dict[int, torch.Tensor],
        pos_bnpc: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        from analysis.backbones.cache_replay.pi3 import _set_pi3_head_checkpoint
        from analysis.backbones.cache_replay.pi3_official.geometry import homogenize_points

        penultimate_idx, last_idx = self._pi3_head_unit_indices(model)
        missing = [idx for idx in (penultimate_idx, last_idx) if idx not in head_outputs]
        if missing:
            raise RuntimeError(f"Pi3 incremental skip replay missed head units: {missing}")
        penultimate = head_outputs[penultimate_idx]
        final = head_outputs[last_idx]
        if penultimate.shape != final.shape:
            raise RuntimeError(
                "Pi3 head decoder outputs have inconsistent shapes: "
                f"{tuple(penultimate.shape)} vs {tuple(final.shape)}"
            )
        bsz, seq_len, num_tokens, _channels = (int(x) for x in penultimate.shape)
        h, w = int(image_hw[0]), int(image_hw[1])
        patch_h, patch_w = h // int(model.patch_size), w // int(model.patch_size)
        patch_start = int(model.patch_start_idx)
        head_in = torch.cat(
            [
                penultimate.reshape(bsz * seq_len, num_tokens, -1),
                final.reshape(bsz * seq_len, num_tokens, -1),
            ],
            dim=-1,
        )
        pos_bn = pos_bnpc.reshape(bsz * seq_len, num_tokens, -1)
        _set_pi3_head_checkpoint(model, False)

        point_hidden = model.point_decoder(head_in, xpos=pos_bn)
        conf_hidden = model.conf_decoder(head_in, xpos=pos_bn)
        camera_hidden = model.camera_decoder(head_in, xpos=pos_bn)

        with torch.autocast(device_type=head_in.device.type, enabled=False):
            point_hidden = point_hidden.float()
            ret = model.point_head([point_hidden[:, patch_start:]], (h, w)).reshape(
                bsz, seq_len, h, w, -1
            )
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = model.conf_head([conf_hidden[:, patch_start:]], (h, w)).reshape(
                bsz, seq_len, h, w, -1
            )

            camera_hidden = camera_hidden.float()
            camera_poses = model.camera_head(
                camera_hidden[:, patch_start:],
                patch_h,
                patch_w,
            ).reshape(bsz, seq_len, 4, 4)

            points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                camera_poses,
                homogenize_points(local_points),
            )[..., :3]

        return {
            "points": points,
            "local_points": local_points,
            "conf": conf,
            "camera_poses": camera_poses,
        }

    def _predict_incremental_skip_span(
        self,
        model: nn.Module,
        *,
        prefix_tokens: torch.Tensor,
        prefix_head_outputs: Dict[int, torch.Tensor],
        start_unit: int,
        end_unit: int,
        total_units: int,
        pos_bnpc: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> PredictionBundle:
        tokens = prefix_tokens
        head_outputs = dict(prefix_head_outputs)
        head_unit_set = set(self._pi3_head_unit_indices(model))
        for unit_idx in range(int(start_unit), int(total_units)):
            if unit_idx > int(end_unit):
                tokens = self._run_incremental_skip_unit(
                    model,
                    tokens,
                    unit_idx,
                    pos_bnpc=pos_bnpc,
                )
            if unit_idx in head_unit_set:
                head_outputs[unit_idx] = tokens
        raw = self._forward_pi3_heads_from_decoder_outputs(
            model,
            head_outputs=head_outputs,
            pos_bnpc=pos_bnpc,
            image_hw=image_hw,
        )
        return self.prediction_bundle_from_raw(raw, prefix_tokens)

    def iter_skip_span_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int, PredictionBundle]]:
        from analysis.amp_policy import autocast_device_dtype
        from analysis.backbones.cache_replay.pi3 import _build_pi3_pos

        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected skip-eval images [B,S,3,H,W], got {tuple(images.shape)}")
        device = self._forward_device(model, images)
        restore_encoder_cpu = self._prepare_full_forward(model, device)
        x = images.to(device=device, dtype=torch.float32)

        unit_ids = [u.unit_id for u in self.layer_unit_catalog(model)]
        total_units = len(unit_ids)
        if total_units != len(model.decoder):
            raise RuntimeError(
                f"Pi3 unit catalog/decoder mismatch: units={total_units} decoder={len(model.decoder)}"
            )
        requested: Dict[int, List[int]] = {}
        for i_raw, j_raw in spans:
            i, j = int(i_raw), int(j_raw)
            if i < 0 or j < i or j >= total_units:
                raise ValueError(f"Invalid skip span ({i},{j}) for {total_units} units")
            requested.setdefault(i, []).append(j)
        for ends in requested.values():
            ends.sort()

        bsz, seq_len, _channels, height, width = (int(v) for v in x.shape)
        ac_kw = autocast_device_dtype(device)
        try:
            with torch.inference_mode(), torch.autocast(**ac_kw):
                prefix_tokens = self.build_initial_tokens(model, x)
                pos = _build_pi3_pos(
                    model,
                    B=bsz,
                    N=seq_len,
                    H=height,
                    W=width,
                    device=device,
                )
            if prefix_tokens is None:
                raise RuntimeError("Pi3 adaptor did not produce initial replay tokens")
            pos_bnpc = pos.reshape(bsz, seq_len, int(prefix_tokens.shape[2]), -1)
            prefix_head_outputs: Dict[int, torch.Tensor] = {}
            head_unit_set = set(self._pi3_head_unit_indices(model))

            for start_unit in range(total_units):
                for end_unit in requested.get(start_unit, []):
                    with torch.inference_mode(), torch.autocast(**ac_kw):
                        prediction = self._predict_incremental_skip_span(
                            model,
                            prefix_tokens=prefix_tokens,
                            prefix_head_outputs=prefix_head_outputs,
                            start_unit=start_unit,
                            end_unit=end_unit,
                            total_units=total_units,
                            pos_bnpc=pos_bnpc,
                            image_hw=(height, width),
                        )
                    yield start_unit, end_unit, prediction

                with torch.inference_mode(), torch.autocast(**ac_kw):
                    prefix_tokens = self._run_incremental_skip_unit(
                        model,
                        prefix_tokens,
                        start_unit,
                        pos_bnpc=pos_bnpc,
                    )
                    if start_unit in head_unit_set:
                        prefix_head_outputs[start_unit] = prefix_tokens
        finally:
            self._restore_full_forward(model, restore_encoder_cpu)

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
                captures[uid] = self._normalize_decoder_capture(
                    uid, t, bsz=bsz, seq_len=seq_len
                ).detach()

            return hook

        if images.dim() == 4:
            images = images.unsqueeze(0)
        bsz, seq_len = int(images.shape[0]), int(images.shape[1])
        for uid in want:
            idx = self._decoder_index(uid)
            if idx < len(model.decoder):
                handles.append(model.decoder[idx].register_forward_hook(_make_hook(uid)))

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
        meta = {
            "patch_start_idx": int(getattr(model, "patch_start_idx", 0)),
            "camera_token_count": 0,
        }
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
        dataset_name: str = "",
    ) -> Dict[str, torch.Tensor]:
        from analysis.backbones.cache_replay.pi3 import build_target_batch_from_teacher_pi3

        return build_target_batch_from_teacher_pi3(
            teacher_pred,
            images_batched=images_batched,
            device=device,
            dataset_name=dataset_name,
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

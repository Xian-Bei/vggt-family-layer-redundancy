from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.backbones.adapters._vggt_base import (
    build_vggt_family_training_optimizer,
    vggt_family_training_config,
)
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
from analysis.recovery.train.model import inject_lora_attn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _omega_root() -> Path:
    return _repo_root() / "backbones" / "vggt-omega"


def _ensure_omega_path() -> None:
    root = str(_omega_root().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class _OmegaSkipBlock(nn.Module):
    """Identity block compatible with Omega attention block signature."""

    def forward(self, x: torch.Tensor, rope=None, pos=None, **kwargs: Any) -> torch.Tensor:
        del rope, pos, kwargs
        return x


class VGGTOmegaAdapter(BackboneAdapter):
    name = "vggt_omega"

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

    def default_hf_model(self) -> str:
        local = _repo_root() / "backbones" / "vggt-omega" / "vggt_omega_1b_512.pt"
        return str(local) if local.is_file() else "vggt_omega_1b_512.pt"

    def pythonpath_entries(self, repo_root: Path) -> List[Path]:
        return [repo_root / "backbones" / "vggt-omega", repo_root]

    def load_pretrained(
        self,
        *,
        hf_model: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
    ) -> nn.Module:
        _ensure_omega_path()
        from vggt_omega.models import VGGTOmega

        model = VGGTOmega()
        ckpt_path = Path(hf_model)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"vggt_omega expects a local checkpoint path; missing: {ckpt_path}"
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint format at {ckpt_path}: {type(state)}")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model = model.to(device=device)
        if dtype is not None and dtype != torch.float32:
            raise ValueError(
                "vggt_omega expects fp32 weights + autocast; got load dtype="
                f"{dtype}. Pass dtype=None."
            )
        if eval_mode:
            model.eval()
        return model

    def _inter_frame_blocks(self, model: nn.Module) -> nn.ModuleList:
        agg = model.aggregator
        blocks = getattr(agg, "inter_frame_blocks", None)
        if blocks is None:
            raise AttributeError("Omega aggregator missing inter_frame_blocks")
        return blocks

    def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module:
        kind, idx_s = unit_id.split("_", 1)
        idx = int(idx_s)
        if kind == "frame":
            return model.aggregator.frame_blocks[idx]
        if kind == "global":
            return self._inter_frame_blocks(model)[idx]
        raise KeyError(unit_id)

    def apply_unit_operation(
        self,
        model: nn.Module,
        unit_id: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        kind, idx = self.parse_unit_id(unit_id)
        if operation == "drop_unit":
            if kind == "frame" and 0 <= idx < len(model.aggregator.frame_blocks):
                model.aggregator.frame_blocks[idx] = _OmegaSkipBlock()
            elif kind == "global" and 0 <= idx < len(self._inter_frame_blocks(model)):
                self._inter_frame_blocks(model)[idx] = _OmegaSkipBlock()
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

    def layer_unit_catalog(self, model: nn.Module) -> List[LayerUnitSpec]:
        depth = len(model.aggregator.frame_blocks)
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
                    module_path=f"aggregator.inter_frame_blocks.{i}",
                    droppable=True,
                    lora_target=True,
                )
            )
        cap = self.capabilities()
        cap.num_frame_blocks = depth
        cap.num_global_blocks = depth
        return units

    def _crop_to_supported_aspect_ratio(self, images: torch.Tensor) -> torch.Tensor:
        # Omega preprocess: crop extreme aspect ratios into [0.5, 2.0].
        b, s, c, h, w = (int(x) for x in images.shape)
        ratio = h / max(w, 1)
        if ratio < 0.5:
            crop_w = min(w, max(1, int(round(h / 0.5))))
            left = max((w - crop_w) // 2, 0)
            images = images[..., :, left : left + crop_w]
        elif ratio > 2.0:
            crop_h = min(h, max(1, int(round(w * 2.0))))
            top = max((h - crop_h) // 2, 0)
            images = images[..., top : top + crop_h, :]
        return images

    def _balanced_target_shape(
        self,
        h: int,
        w: int,
        *,
        image_resolution: int = 512,
        patch_size: int = 16,
    ) -> Tuple[int, int]:
        # Match Omega load_fn mode="balanced".
        ratio = float(h) / max(float(w), 1.0)
        token_number = (image_resolution // patch_size) ** 2
        w_patches = max(1, int(round(math.sqrt(token_number / max(ratio, 1e-8)))))
        h_patches = max(1, int(round(token_number / max(float(w_patches), 1.0))))
        return h_patches * patch_size, w_patches * patch_size

    def _prepare_images_like_omega(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        x = images.float().clamp(0.0, 1.0)
        x = self._crop_to_supported_aspect_ratio(x)
        _, _, _, h, w = (int(v) for v in x.shape)
        th, tw = self._balanced_target_shape(h, w, image_resolution=512, patch_size=16)
        if h == th and w == tw:
            return x
        bs = int(x.shape[0]) * int(x.shape[1])
        x2 = x.reshape(bs, int(x.shape[2]), h, w)
        x2 = F.interpolate(x2, size=(th, tw), mode="bicubic", align_corners=False)
        return x2.reshape(int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), th, tw)

    def load_eval_images(self, image_paths: Sequence[Path | str]) -> torch.Tensor:
        from analysis.datasets.driving_depth import load_raw_eval_images

        raw = load_raw_eval_images(image_paths)
        return self._prepare_images_like_omega(raw.unsqueeze(0)).squeeze(0)

    def load_driving_depth_stack(
        self,
        depth_paths: Sequence[Path | str],
        *,
        output_hw: Tuple[int, int],
    ) -> Tuple[Any, int]:
        from analysis.datasets.driving_depth import load_driving_depth_stack_omega

        return load_driving_depth_stack_omega(depth_paths, output_hw=output_hw)

    def _ensure_eval_batch(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        return images

    def load_scannet_gt_depth(
        self,
        scene: Any,
        *,
        output_hw: Tuple[int, int],
    ) -> Any:
        from analysis.eval.prediction_eval import _load_gt_depth_for_scene_omega

        return _load_gt_depth_for_scene_omega(scene, image_hw=output_hw)

    def align_7andn_gt_depth(self, view: Any, *, output_hw: Tuple[int, int]) -> Any:
        from analysis.eval.prediction_eval import _load_7andn_gt_depth_omega

        return _load_7andn_gt_depth_omega(view, image_hw=output_hw)

    def _omega_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        agg = model.aggregator
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.dim() != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        bsz, seq_len, channels, height, width = (int(x) for x in images.shape)
        if channels != 3:
            raise ValueError(f"Expected 3 image channels, got {channels}")
        x = (images - agg._resnet_mean) / agg._resnet_std
        x = x.view(bsz * seq_len, channels, height, width)
        camera_token = agg.camera_token[:, 0:1].expand(bsz, 1, *agg.camera_token.shape[2:])
        if seq_len > 1:
            camera_other = agg.camera_token[:, 1:].expand(bsz, seq_len - 1, *agg.camera_token.shape[2:])
            camera_token = torch.cat([camera_token, camera_other], dim=1)
        camera_token = camera_token.view(bsz * seq_len, *camera_token.shape[2:])
        register_token = agg.register_token[:, 0:1].expand(bsz, 1, *agg.register_token.shape[2:])
        if seq_len > 1:
            register_other = agg.register_token[:, 1:].expand(bsz, seq_len - 1, *agg.register_token.shape[2:])
            register_token = torch.cat([register_token, register_other], dim=1)
        register_token = register_token.view(bsz * seq_len, *register_token.shape[2:])
        patch_tokens = agg.patch_embed(x)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        return tokens.view(bsz, seq_len, int(tokens.shape[1]), int(tokens.shape[2])).detach().contiguous()

    def build_initial_tokens(self, model: nn.Module, images: torch.Tensor) -> Optional[torch.Tensor]:
        return self._omega_initial_tokens(model, images)

    def cache_manifest_extra(self, model: nn.Module) -> Dict[str, Any]:
        try:
            attn = model.aggregator.inter_frame_blocks[0].attn
            return {"num_heads": int(getattr(attn, "num_heads", 0))}
        except Exception:
            return {}

    def extract_layer_units(
        self,
        model: nn.Module,
        images: torch.Tensor,
        unit_ids: Optional[Set[str]] = None,
        layer_callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        x = self._ensure_eval_batch(images)
        if x.device.type != "cuda":
            raise RuntimeError("vggt_omega forward requires CUDA device")

        agg = model.aggregator
        depth = len(agg.frame_blocks)
        want = unit_ids or {u.unit_id for u in self.layer_unit_catalog(model)}
        prev_cached = set(getattr(agg, "cached_layer_indices", set()))
        try:
            agg.cached_layer_indices = set(range(depth))
            with torch.no_grad():
                tokens_list, patch_start = agg(x)
        finally:
            agg.cached_layer_indices = prev_cached

        out: Dict[str, torch.Tensor] = {}
        concat_list: List[torch.Tensor] = []
        for i, concat_t in enumerate(tokens_list):
            if not isinstance(concat_t, torch.Tensor):
                continue
            if concat_t.dim() != 4:
                continue
            concat_list.append(concat_t.detach().contiguous())
            two_c = int(concat_t.shape[-1])
            if two_c % 2 != 0:
                continue
            c = two_c // 2
            fid, gid = f"frame_{i}", f"global_{i}"
            if fid in want:
                ft = concat_t[..., :c].contiguous()
                out[fid] = ft
                if layer_callback is not None:
                    layer_callback(fid, ft)
            if gid in want:
                gt = concat_t[..., c:].contiguous()
                out[gid] = gt
                if layer_callback is not None:
                    layer_callback(gid, gt)
        return out, {"patch_start_idx": int(patch_start), "camera_token_count": 1, "aggregated_concat": concat_list}

    def forward_predictions(self, model: nn.Module, images: torch.Tensor) -> PredictionBundle:
        x = self._ensure_eval_batch(images)
        if x.device.type != "cuda":
            raise RuntimeError("vggt_omega forward requires CUDA device")

        with torch.no_grad():
            preds = model(x)
        return self.prediction_bundle_from_raw(dict(preds) if isinstance(preds, dict) else {}, x)

    def _prediction_bundle_from_preds(self, preds: Dict[str, Any], x: torch.Tensor) -> PredictionBundle:
        return self.prediction_bundle_from_raw(dict(preds) if isinstance(preds, dict) else {}, x)

    def supports_incremental_skip_eval(self) -> bool:
        return True

    @staticmethod
    def _run_incremental_skip_unit(
        model: nn.Module,
        tokens_bspc: torch.Tensor,
        unit_idx: int,
        *,
        frame_rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        from analysis.backbones.cache_replay.vggt_family import (
            _run_omega_frame_block,
            _run_omega_inter_frame_block,
        )

        bsz, seq_len, num_tokens, embed_dim = (int(x) for x in tokens_bspc.shape)
        layer_idx, half = divmod(int(unit_idx), 2)
        agg = model.aggregator
        if half == 0:
            frame_in = tokens_bspc.reshape(bsz * seq_len, num_tokens, embed_dim)
            frame_out = _run_omega_frame_block(
                agg.frame_blocks[layer_idx],
                frame_in,
                frame_rope,
                training=False,
                use_checkpoint=False,
            )
            return frame_out.reshape(bsz, seq_len, num_tokens, embed_dim)
        return _run_omega_inter_frame_block(
            agg,
            layer_idx,
            tokens_bspc,
            bsz=bsz,
            seq_len=seq_len,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            block=agg.inter_frame_blocks[layer_idx],
            training=False,
            use_checkpoint=False,
        )

    def _forward_heads_from_incremental_layers(
        self,
        model: nn.Module,
        selected: Dict[int, torch.Tensor],
        *,
        num_layers: int,
        patch_start: int,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        aggregated_tokens_list: List[Optional[torch.Tensor]] = [None] * int(num_layers)
        for idx, tensor in selected.items():
            if 0 <= int(idx) < int(num_layers):
                aggregated_tokens_list[int(idx)] = tensor
        final_tokens = aggregated_tokens_list[-1] if aggregated_tokens_list else None
        preds: Dict[str, torch.Tensor] = {}
        if isinstance(final_tokens, torch.Tensor):
            preds["camera_and_register_tokens"] = final_tokens[:, :, : int(patch_start)].contiguous()
        with torch.autocast(device_type=images.device.type, enabled=False):
            if getattr(model, "camera_head", None) is not None:
                preds["pose_enc"] = model.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=int(patch_start),
                )
            if getattr(model, "dense_head", None) is not None:
                depth_t, depth_conf_t = model.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=int(patch_start),
                )
                preds["depth"] = depth_t
                preds["depth_conf"] = depth_conf_t
            if getattr(model, "text_alignment_head", None) is not None:
                preds.update(
                    model.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=int(patch_start),
                    )
                )
        if not model.training:
            preds["images"] = images
        return preds

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
        frame_rope: Tuple[torch.Tensor, torch.Tensor],
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
                    frame_rope=frame_rope,
                )
            if half == 0:
                frame_at_layer = tokens
            else:
                if frame_at_layer is None:
                    raise RuntimeError(f"Missing Omega frame state before global unit {unit_idx}")
                if layer_idx in required_layers:
                    selected[layer_idx] = torch.cat([frame_at_layer, tokens], dim=-1)
                frame_at_layer = None

        missing = sorted(required_layers.difference(selected))
        if missing:
            raise RuntimeError(f"Omega incremental skip replay missed head layers: {missing}")
        num_layers = int(total_units) // 2
        raw = self._forward_heads_from_incremental_layers(
            model,
            selected,
            num_layers=num_layers,
            patch_start=int(model.aggregator.patch_token_start),
            images=images,
        )
        return self.prediction_bundle_from_raw(raw, images)

    def iter_skip_span_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int, PredictionBundle]]:
        from analysis.amp_policy import autocast_device_dtype
        from analysis.backbones.cache_replay.vggt_family import _build_omega_frame_rope

        x = self._ensure_eval_batch(images)
        if x.device.type != "cuda":
            raise RuntimeError("vggt_omega incremental skip eval requires CUDA device")

        unit_ids = [u.unit_id for u in self.layer_unit_catalog(model)]
        total_units = len(unit_ids)
        if total_units == 0 or total_units % 2 != 0:
            raise RuntimeError(f"Omega incremental replay requires paired units, got {total_units}")
        requested: Dict[int, List[int]] = {}
        for i_raw, j_raw in spans:
            i, j = int(i_raw), int(j_raw)
            if i < 0 or j < i or j >= total_units:
                raise ValueError(f"Invalid skip span ({i},{j}) for {total_units} units")
            requested.setdefault(i, []).append(j)
        for ends in requested.values():
            ends.sort()

        device = x.device
        ac_kw = autocast_device_dtype(device)
        with torch.inference_mode(), torch.autocast(**ac_kw):
            frame_rope = _build_omega_frame_rope(model.aggregator, x, device)
            prefix_tokens = self.build_initial_tokens(model, x)
        if prefix_tokens is None:
            raise RuntimeError("Omega adaptor did not produce initial replay tokens")

        num_layers = int(total_units) // 2
        required_layers = set(int(v) for v in self.required_head_layers_for_cache(model, num_layers))
        if getattr(model, "text_alignment_head", None) is not None and num_layers > 0:
            required_layers.add(num_layers - 1)
        prefix_layers: Dict[int, torch.Tensor] = {}
        baseline_frame: Optional[torch.Tensor] = None

        for start_unit in range(total_units):
            for end_unit in requested.get(start_unit, []):
                with torch.inference_mode(), torch.autocast(**ac_kw):
                    prediction = self._predict_incremental_skip_span(
                        model,
                        x,
                        prefix_tokens=prefix_tokens,
                        prefix_layers=prefix_layers,
                        start_unit=start_unit,
                        end_unit=end_unit,
                        total_units=total_units,
                        required_layers=required_layers,
                        frame_rope=frame_rope,
                    )
                yield start_unit, end_unit, prediction

            with torch.inference_mode(), torch.autocast(**ac_kw):
                prefix_tokens = self._run_incremental_skip_unit(
                    model,
                    prefix_tokens,
                    start_unit,
                    frame_rope=frame_rope,
                )
                layer_idx, half = divmod(start_unit, 2)
                if half == 0:
                    baseline_frame = prefix_tokens
                else:
                    if baseline_frame is None:
                        raise RuntimeError(f"Missing Omega baseline frame state at layer {layer_idx}")
                    if layer_idx in required_layers:
                        prefix_layers[layer_idx] = torch.cat(
                            [baseline_frame, prefix_tokens], dim=-1
                        )
                    baseline_frame = None

    def prediction_bundle_from_raw(
        self,
        raw: Dict[str, Any],
        images: torch.Tensor,
    ) -> PredictionBundle:
        from analysis.backbones.prediction import bundle_from_pose_depth

        return bundle_from_pose_depth(raw=raw, images=images, backbone_name=self.name)

    def extract_layer_units_and_predictions(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Optional[CacheForwardResult]:
        x = self._ensure_eval_batch(images)
        if x.device.type != "cuda":
            raise RuntimeError("vggt_omega forward requires CUDA device")

        agg = model.aggregator
        depth = len(agg.frame_blocks)
        bsz, seq_len = int(x.shape[0]), int(x.shape[1])
        initial_tokens: Optional[torch.Tensor] = None

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

        initial_handle = agg.frame_blocks[0].register_forward_pre_hook(_capture_initial)
        prev_cached = set(getattr(agg, "cached_layer_indices", set()))
        try:
            agg.cached_layer_indices = set(range(depth))
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                tokens_list, patch_start = agg(x)
        finally:
            agg.cached_layer_indices = prev_cached
            initial_handle.remove()

        out: Dict[str, torch.Tensor] = {}
        concat_list: List[torch.Tensor] = []
        for i, concat_t in enumerate(tokens_list):
            if not isinstance(concat_t, torch.Tensor):
                continue
            if concat_t.dim() != 4:
                continue
            concat_list.append(concat_t.detach().contiguous())
            two_c = int(concat_t.shape[-1])
            if two_c % 2 != 0:
                continue
            c = two_c // 2
            out[f"frame_{i}"] = concat_t[..., :c].contiguous()
            out[f"global_{i}"] = concat_t[..., c:].contiguous()

        final_tokens = tokens_list[-1] if tokens_list else None
        if not isinstance(final_tokens, torch.Tensor):
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        preds: Dict[str, torch.Tensor] = {
            "camera_and_register_tokens": final_tokens[:, :, : int(patch_start)].contiguous(),
        }
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            if getattr(model, "camera_head", None) is not None:
                preds["pose_enc"] = model.camera_head(
                    tokens_list,
                    patch_token_start=int(patch_start),
                )
            if getattr(model, "dense_head", None) is not None:
                depth_t, depth_conf_t = model.dense_head(
                    tokens_list,
                    images=x,
                    patch_token_start=int(patch_start),
                )
                preds["depth"] = depth_t
                preds["depth_conf"] = depth_conf_t
            if getattr(model, "text_alignment_head", None) is not None:
                preds.update(
                    model.text_alignment_head(
                        tokens_list,
                        patch_token_start=int(patch_start),
                    )
                )
        if not model.training:
            preds["images"] = x

        pred = self._prediction_bundle_from_preds(preds, x)
        meta = {
            "patch_start_idx": int(patch_start),
            "camera_token_count": 1,
            "aggregated_concat": concat_list,
        }
        if initial_tokens is None:
            raise RuntimeError("VGGT-Omega forward did not expose the layer_-1 input to frame_0")
        return CacheForwardResult(
            unit_map=out,
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

        return _replay_model_cache_first(model, get_layer_cpu, num_layers, **kwargs)

    def cache_replay_engine(self) -> Any:
        from analysis.backbones.cache_replay.vggt_family import VGGTCacheReplayEngine

        return VGGTCacheReplayEngine(backbone_name=self.name)

register(VGGTOmegaAdapter())

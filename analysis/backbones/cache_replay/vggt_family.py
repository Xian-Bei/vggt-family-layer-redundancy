"""VGGT cache-first replay training."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
    LinearParamScheduler,
)
from torch.utils.checkpoint import checkpoint

from analysis.backbones.cache_replay.base import CacheReplayEngine
from analysis.backbones.cache_replay.common import ReplayResult, is_skipped, split_concat
from analysis.backbones.contracts import PredictionLossWeights, TrainingHyperparams
from analysis.backbones.registry import get_backbone
from analysis.schema.v3 import PruneAction
from analysis.recovery.train.model import (
    count_unique_params,
)
VGGT_OFFICIAL_CAMERA_WEIGHT = 5.0
VGGT_OFFICIAL_DEPTH_WEIGHT = 1.0
VGGT_OFFICIAL_CAMERA_LOSS_TYPE = "l1"
VGGT_OFFICIAL_DEPTH_VALID_RANGE = 0.98
VGGT_OFFICIAL_DEPTH_GRADIENT_LOSS_FN = "grad"


def _build_vggt_official_lr_scheduler(cfg: TrainingHyperparams) -> CompositeParamScheduler:
    return CompositeParamScheduler(
        schedulers=[
            LinearParamScheduler(float(cfg.min_lr), float(cfg.lr)),
            CosineParamScheduler(float(cfg.lr), float(cfg.min_lr)),
        ],
        lengths=[float(cfg.warmup_ratio), 1.0 - float(cfg.warmup_ratio)],
        interval_scaling=["rescaled", "rescaled"],
    )


def _build_vggt_official_weight_decay_scheduler(cfg: TrainingHyperparams) -> ConstantParamScheduler:
    return ConstantParamScheduler(float(cfg.weight_decay))


def _is_omega_model(model: nn.Module) -> bool:
    agg = getattr(model, "aggregator", None)
    return agg is not None and hasattr(agg, "inter_frame_blocks") and not hasattr(agg, "global_blocks")


def _global_blocks(model: nn.Module) -> nn.ModuleList:
    agg = model.aggregator
    if hasattr(agg, "global_blocks"):
        return agg.global_blocks
    if hasattr(agg, "inter_frame_blocks"):
        return agg.inter_frame_blocks
    raise AttributeError("Model aggregator has neither global_blocks nor inter_frame_blocks")


def _prune_drop_actions(
    frame_skip_intervals: Sequence[Tuple[int, int]],
    global_skip_intervals: Sequence[Tuple[int, int]],
) -> List[PruneAction]:
    actions: List[PruneAction] = []
    for l, r in frame_skip_intervals:
        for idx in range(int(l), int(r) + 1):
            actions.append(PruneAction(action="drop_unit", unit_id=f"frame_{idx}"))
    for l, r in global_skip_intervals:
        for idx in range(int(l), int(r) + 1):
            actions.append(PruneAction(action="drop_unit", unit_id=f"global_{idx}"))
    return actions


def _decode_pose_to_extrinsics_intrinsics(
    pose_enc: torch.Tensor,
    image_size_hw: Tuple[int, int],
    *,
    backbone_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from analysis.backbones.pose_geometry import decode_pose_enc_to_extrinsics_intrinsics

    extrinsic, intrinsics = decode_pose_enc_to_extrinsics_intrinsics(
        pose_enc, image_size_hw, backbone=backbone_name
    )
    if intrinsics is None:
        raise RuntimeError(f"{backbone_name} pose decode returned no intrinsics")
    return extrinsic, intrinsics


def build_pos(model: nn.Module, images: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    agg = model.aggregator
    if agg.rope is None:
        return None, None
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if images.dim() != 5:
        raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
    bsz, seq_len, _, h, w = images.shape
    ph = h // int(agg.patch_size)
    pw = w // int(agg.patch_size)
    pos = agg.position_getter(bsz * seq_len, ph, pw, device=images.device)
    if int(agg.patch_start_idx) > 0:
        pos = pos + 1
        pos_special = torch.zeros(bsz * seq_len, int(agg.patch_start_idx), 2, device=images.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)
    p = int(pos.shape[1])
    pos_frame = pos.view(bsz * seq_len, p, 2)
    pos_global = pos_frame.view(bsz, seq_len, p, 2).view(bsz, seq_len * p, 2)
    return pos_frame, pos_global


def run_block_with_checkpoint(
    block: nn.Module,
    x: torch.Tensor,
    *,
    pos: Optional[torch.Tensor],
    training: bool,
    use_checkpoint: bool,
    use_reentrant: bool,
) -> torch.Tensor:
    if (not training) or (not use_checkpoint):
        return block(x, pos=pos)
    if pos is None:
        return checkpoint(lambda _x: block(_x, pos=None), x, use_reentrant=use_reentrant)
    return checkpoint(lambda _x, _p: block(_x, pos=_p), x, pos, use_reentrant=use_reentrant)


def required_head_layers(model: nn.Module, num_layers: int) -> List[int]:
    keep: set[int] = set()
    dense = getattr(model, "dense_head", None)
    if dense is not None:
        keep.update(int(i) for i in getattr(dense, "intermediate_layer_idx", []))
        if model.camera_head is not None and num_layers > 0:
            keep.add(num_layers - 1)
        return sorted(i for i in keep if 0 <= i < num_layers)
    if model.camera_head is not None and num_layers > 0:
        keep.add(num_layers - 1)
    if model.depth_head is not None:
        keep.update(int(i) for i in getattr(model.depth_head, "intermediate_layer_idx", []))
    return sorted(i for i in keep if 0 <= i < num_layers)


def _load_replay_seed(
    get_layer_cpu: Callable[[int], torch.Tensor],
    *,
    seed_layer: int,
    seed_kind: str,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int, int, int]:
    """Load cached half-unit seed and return (seed_tensor, bsz, seq_len, p, c)."""
    kind = str(seed_kind).strip().lower()
    unit_loader = getattr(get_layer_cpu, "unit_loader", None)
    if callable(unit_loader) and kind in {"frame", "global"}:
        seed = unit_loader(f"{kind}_{int(seed_layer)}").to(
            device=device, non_blocking=(device.type == "cuda")
        )
    else:
        seed_concat = get_layer_cpu(int(seed_layer)).to(
            device=device, non_blocking=(device.type == "cuda")
        )
        seed_frame, seed_global = split_concat(seed_concat)
        if kind == "frame":
            seed = seed_frame
        elif kind == "global":
            seed = seed_global
        else:
            raise ValueError(f"seed_kind must be frame or global, got {seed_kind!r}")
    bsz, seq_len, p, c = (int(x) for x in seed.shape)
    return seed, bsz, seq_len, p, c


def _replay_vggt_cache_first(
    model: nn.Module,
    get_layer_cpu: Callable[[int], torch.Tensor],
    num_layers: int,
    *,
    frame_skip_intervals: Sequence[Tuple[int, int]],
    global_skip_intervals: Sequence[Tuple[int, int]],
    lora_anchor_layers: Sequence[int],
    alignment_capture_layers: Sequence[int] = (),
    seed_layer: int,
    seed_kind: str,
    initial_tokens: Optional[torch.Tensor],
    use_checkpoint: bool,
    device: torch.device,
    dummy_images: torch.Tensor,
    backbone_name: str = "vggt",
) -> ReplayResult:
    if num_layers <= 0:
        raise ValueError("Replay cache is empty.")
    start_layer = min(int(x) for x in lora_anchor_layers) if lora_anchor_layers else 0
    if start_layer < 0:
        raise ValueError(f"Expected start_layer>=0, got {start_layer}")
    if start_layer >= num_layers:
        raise ValueError(f"start_layer={start_layer} out of range for num_layers={num_layers}")

    seed_kind_norm = str(seed_kind).strip().lower()
    if seed_kind_norm == "none":
        if int(start_layer) != 0:
            raise ValueError(
                f"seed_kind=none only supported when replay starts at layer 0, got start_layer={start_layer}"
            )
        if initial_tokens is None:
            raise RuntimeError("initial_tokens are required when seed_kind=none")
        init_tokens = initial_tokens.to(device=device, non_blocking=(device.type == "cuda"))
        if init_tokens.dim() != 4:
            raise RuntimeError(f"Expected layer_-1 tokens [B,S,P,C], got {tuple(init_tokens.shape)}")
        seed_tensor = init_tokens
        use_frame_seed = True
    else:
        seed_tensor, *_shape = _load_replay_seed(
            get_layer_cpu, seed_layer=int(seed_layer), seed_kind=str(seed_kind), device=device
        )
        use_frame_seed = seed_kind_norm == "frame"
    bsz, seq_len, p, c = (int(x) for x in seed_tensor.shape)

    pos_frame, pos_global = build_pos(model, dummy_images)
    if pos_frame is not None and pos_frame.shape[0] != bsz * seq_len:
        raise RuntimeError(
            f"Position mismatch: pos_frame={tuple(pos_frame.shape)}, replay b*s={bsz * seq_len}"
        )

    required = set(required_head_layers(model, num_layers))
    required.update(int(x) for x in lora_anchor_layers)
    required.update(int(x) for x in alignment_capture_layers)
    concat_out: Dict[int, torch.Tensor] = {}

    global_prev = seed_tensor.view(bsz, seq_len * p, c)

    use_reentrant = bool(getattr(model.aggregator, "use_reentrant", False))
    for layer_idx in range(start_layer, num_layers):
        if use_frame_seed and layer_idx == start_layer:
            frame_view = seed_tensor.view(bsz, seq_len, p, c)
        else:
            frame_in = global_prev.view(bsz * seq_len, p, c)
            if is_skipped(layer_idx, frame_skip_intervals):
                frame_out = frame_in
            else:
                frame_out = run_block_with_checkpoint(
                    model.aggregator.frame_blocks[layer_idx],
                    frame_in,
                    pos=pos_frame,
                    training=bool(model.training),
                    use_checkpoint=use_checkpoint,
                    use_reentrant=use_reentrant,
                )
            frame_view = frame_out.view(bsz, seq_len, p, c)

        global_in = frame_view.view(bsz, seq_len * p, c)
        if is_skipped(layer_idx, global_skip_intervals):
            global_out = global_in
        else:
            global_out = run_block_with_checkpoint(
                model.aggregator.global_blocks[layer_idx],
                global_in,
                pos=pos_global,
                training=bool(model.training),
                use_checkpoint=use_checkpoint,
                use_reentrant=use_reentrant,
            )
        if layer_idx in required:
            global_view = global_out.view(bsz, seq_len, p, c)
            concat_out[layer_idx] = torch.cat([frame_view, global_view], dim=-1)
        global_prev = global_out

    selected: Dict[int, torch.Tensor] = {}
    for idx in required:
        if idx < start_layer:
            selected[idx] = get_layer_cpu(idx).to(device=device, non_blocking=(device.type == "cuda"))
        else:
            if idx not in concat_out:
                raise RuntimeError(f"Missing replay output at required layer {idx}")
            selected[idx] = concat_out[idx]
    agg_list: List[torch.Tensor] = [None] * num_layers  # type: ignore[list-item]
    for idx, t in selected.items():
        agg_list[idx] = t
    pred = get_backbone(backbone_name).run_heads_from_aggregated_tokens(
        model,
        agg_list,
        int(model.aggregator.patch_start_idx),
        dummy_images,
    )
    return ReplayResult(concat_by_layer=concat_out, pred=pred)


def _build_omega_frame_rope(
    agg: nn.Module,
    dummy_images: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dummy_images.dim() == 4:
        dummy_images = dummy_images.unsqueeze(0)
    _, _, _, h, w = (int(x) for x in dummy_images.shape)
    patch_grid = (h // int(agg.patch_size), w // int(agg.patch_size))
    with torch.no_grad():
        rope_sin, rope_cos = agg.rope_embed(H=patch_grid[0], W=patch_grid[1])
    return (
        rope_sin.to(device=device, dtype=torch.float32),
        rope_cos.to(device=device, dtype=torch.float32),
    )


def _run_omega_frame_block(
    block: nn.Module,
    x: torch.Tensor,
    rope: Tuple[torch.Tensor, torch.Tensor],
    *,
    training: bool,
    use_checkpoint: bool,
) -> torch.Tensor:
    if (not training) or (not use_checkpoint):
        return block(x, rope)
    return checkpoint(lambda _x: block(_x, rope), x, use_reentrant=False)


def _run_omega_inter_frame_block(
    agg: nn.Module,
    block_idx: int,
    tokens: torch.Tensor,
    *,
    bsz: int,
    seq_len: int,
    num_tokens: int,
    embed_dim: int,
    block: nn.Module,
    training: bool,
    use_checkpoint: bool,
) -> torch.Tensor:
    attention_type = agg.inter_frame_attention_types[block_idx]
    if attention_type == "global":
        g_in = tokens.view(bsz, seq_len * num_tokens, embed_dim)
        if (not training) or (not use_checkpoint):
            g_out = block(g_in, None)
        else:
            g_out = checkpoint(lambda h: block(h, None), g_in, use_reentrant=False)
        return g_out.view(bsz, seq_len, num_tokens, embed_dim)

    patch_token_start = int(agg.patch_token_start)
    cam_reg = tokens[:, :, :patch_token_start].reshape(bsz, seq_len * patch_token_start, embed_dim)
    patch_tokens = tokens[:, :, patch_token_start:].reshape(
        bsz, seq_len * (num_tokens - patch_token_start), embed_dim
    )
    if (not training) or (not use_checkpoint):
        cam_reg = block(cam_reg, None)
    else:
        cam_reg = checkpoint(lambda h: block(h, None), cam_reg, use_reentrant=False)
    merged = torch.cat([cam_reg, patch_tokens], dim=1)
    cam_reg = merged[:, : seq_len * patch_token_start].view(bsz, seq_len, patch_token_start, embed_dim)
    patch_tokens = merged[:, seq_len * patch_token_start :].view(
        bsz, seq_len, num_tokens - patch_token_start, embed_dim
    )
    return torch.cat([cam_reg, patch_tokens], dim=2)


def _forward_omega_heads(
    model: nn.Module,
    selected: Dict[int, torch.Tensor],
    num_layers: int,
    patch_start: int,
    dummy_images: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    agg_list: List[Optional[torch.Tensor]] = [None] * num_layers
    for idx, t in selected.items():
        agg_list[idx] = t
    with torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(agg_list, patch_token_start=patch_start)
        depth, depth_conf = model.dense_head(
            agg_list,
            images=dummy_images,
            patch_token_start=patch_start,
        )
    return {"pose_enc": pose_enc, "depth": depth, "depth_conf": depth_conf}


def _replay_omega_cache_first(
    model: nn.Module,
    get_layer_cpu: Callable[[int], torch.Tensor],
    num_layers: int,
    *,
    frame_skip_intervals: Sequence[Tuple[int, int]],
    global_skip_intervals: Sequence[Tuple[int, int]],
    lora_anchor_layers: Sequence[int],
    alignment_capture_layers: Sequence[int] = (),
    seed_layer: int,
    seed_kind: str,
    initial_tokens: Optional[torch.Tensor],
    use_checkpoint: bool,
    device: torch.device,
    dummy_images: torch.Tensor,
) -> ReplayResult:
    if num_layers <= 0:
        raise ValueError("Replay cache is empty.")
    start_layer = min(int(x) for x in lora_anchor_layers) if lora_anchor_layers else 0
    if start_layer < 0:
        raise ValueError(f"Expected start_layer>=0, got {start_layer}")
    if start_layer >= num_layers:
        raise ValueError(f"start_layer={start_layer} out of range for num_layers={num_layers}")

    seed_kind_norm = str(seed_kind).strip().lower()
    if seed_kind_norm == "none":
        if int(start_layer) != 0:
            raise ValueError(
                f"seed_kind=none only supported when replay starts at layer 0, got start_layer={start_layer}"
            )
        if initial_tokens is None:
            raise RuntimeError("initial_tokens are required when seed_kind=none")
        init_tokens = initial_tokens.to(device=device, non_blocking=(device.type == "cuda"))
        if init_tokens.dim() != 4:
            raise RuntimeError(f"Expected layer_-1 tokens [B,S,P,C], got {tuple(init_tokens.shape)}")
        seed_tensor = init_tokens
        use_frame_seed = True
    else:
        seed_tensor, *_shape = _load_replay_seed(
            get_layer_cpu, seed_layer=int(seed_layer), seed_kind=str(seed_kind), device=device
        )
        use_frame_seed = seed_kind_norm == "frame"
    bsz, seq_len, num_tokens, embed_dim = (int(x) for x in seed_tensor.shape)

    agg = model.aggregator
    frame_rope = _build_omega_frame_rope(agg, dummy_images, device)
    required = set(required_head_layers(model, num_layers))
    required.update(int(x) for x in lora_anchor_layers)
    required.update(int(x) for x in alignment_capture_layers)
    concat_out: Dict[int, torch.Tensor] = {}

    tokens = seed_tensor.view(bsz, seq_len, num_tokens, embed_dim)
    global_blocks = _global_blocks(model)
    for layer_idx in range(start_layer, num_layers):
        if use_frame_seed and layer_idx == start_layer:
            frame_view = tokens
        else:
            frame_in = tokens.view(bsz * seq_len, num_tokens, embed_dim)
            if is_skipped(layer_idx, frame_skip_intervals):
                frame_out = frame_in
            else:
                frame_out = _run_omega_frame_block(
                    agg.frame_blocks[layer_idx],
                    frame_in,
                    frame_rope,
                    training=bool(model.training),
                    use_checkpoint=use_checkpoint,
                )
            frame_view = frame_out.view(bsz, seq_len, num_tokens, embed_dim)

        if is_skipped(layer_idx, global_skip_intervals):
            tokens = frame_view
        else:
            tokens = _run_omega_inter_frame_block(
                agg,
                layer_idx,
                frame_view,
                bsz=bsz,
                seq_len=seq_len,
                num_tokens=num_tokens,
                embed_dim=embed_dim,
                block=global_blocks[layer_idx],
                training=bool(model.training),
                use_checkpoint=use_checkpoint,
            )

        if layer_idx in required:
            concat_out[layer_idx] = torch.cat([frame_view, tokens], dim=-1)

    selected: Dict[int, torch.Tensor] = {}
    for idx in required:
        if idx < start_layer:
            selected[idx] = get_layer_cpu(idx).to(device=device, non_blocking=(device.type == "cuda"))
        else:
            if idx not in concat_out:
                raise RuntimeError(f"Missing replay output at required layer {idx}")
            selected[idx] = concat_out[idx]

    patch_start = int(getattr(agg, "patch_token_start", getattr(model, "patch_start_idx", 0)))
    pred = _forward_omega_heads(model, selected, num_layers, patch_start, dummy_images)
    return ReplayResult(concat_by_layer=concat_out, pred=pred)


def _replay_model_cache_first(
    model: nn.Module,
    get_layer_cpu: Callable[[int], torch.Tensor],
    num_layers: int,
    *,
    backbone_name: str = "vggt",
    **kwargs: Any,
) -> ReplayResult:
    if _is_omega_model(model):
        return _replay_omega_cache_first(model, get_layer_cpu, num_layers, **kwargs)
    return _replay_vggt_cache_first(
        model,
        get_layer_cpu,
        num_layers,
        backbone_name=backbone_name,
        **kwargs,
    )


def depth_to_bshw(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.dim() == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.dim() == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def mask_from_depth(depth_bshw: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(depth_bshw) & (depth_bshw > 1e-6)


def build_target_batch_from_teacher(
    teacher_pred: Dict[str, torch.Tensor],
    *,
    images_batched: torch.Tensor,
    device: torch.device,
    backbone_name: str = "vggt",
    dataset_name: str = "",
) -> Dict[str, torch.Tensor]:
    del dataset_name
    if "pose_enc" not in teacher_pred or "depth" not in teacher_pred:
        raise RuntimeError("teacher target mode requires teacher keys: pose_enc and depth")
    h = int(images_batched.shape[-2])
    w = int(images_batched.shape[-1])
    t_pose = teacher_pred["pose_enc"].float().to(device)
    extrinsics, intrinsics = _decode_pose_to_extrinsics_intrinsics(
        t_pose, (h, w), backbone_name=backbone_name
    )
    t_depth = depth_to_bshw(teacher_pred["depth"].float().to(device))
    point_masks = mask_from_depth(t_depth)
    return {
        "images": images_batched.float(),
        "extrinsics": extrinsics.float(),
        "intrinsics": intrinsics.float(),
        "depths": t_depth.float(),
        "point_masks": point_masks.bool(),
    }


def prediction_official_camera_depth(
    student_pred: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    *,
    w_pred: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    from analysis.backbones.adapters.vggt_loss import compute_camera_loss, compute_depth_loss

    device = next(iter(student_pred.values())).device
    total = torch.zeros((), device=device)
    logs: Dict[str, float] = {}
    wp = float(w_pred)

    pred_for_camera = student_pred
    if "pose_enc_list" not in student_pred and "pose_enc" in student_pred:
        pred_for_camera = {**student_pred, "pose_enc_list": [student_pred["pose_enc"]]}
    if "pose_enc_list" in pred_for_camera:
        camera_loss_dict = compute_camera_loss(
            {"pose_enc_list": pred_for_camera["pose_enc_list"]},
            target_batch,
            loss_type=VGGT_OFFICIAL_CAMERA_LOSS_TYPE,
        )
        l_camera = camera_loss_dict["loss_camera"]
        total = total + VGGT_OFFICIAL_CAMERA_WEIGHT * l_camera
        logs["loss_camera"] = float((wp * VGGT_OFFICIAL_CAMERA_WEIGHT * l_camera).detach().item())

    if "depth" in student_pred and "depth_conf" in student_pred:
        depth_loss_dict = compute_depth_loss(
            {"depth": student_pred["depth"], "depth_conf": student_pred["depth_conf"]},
            target_batch,
            gradient_loss_fn=VGGT_OFFICIAL_DEPTH_GRADIENT_LOSS_FN,
            valid_range=VGGT_OFFICIAL_DEPTH_VALID_RANGE,
        )
        l_conf = depth_loss_dict["loss_conf_depth"]
        l_reg = depth_loss_dict["loss_reg_depth"]
        l_grad = depth_loss_dict["loss_grad_depth"]
        l_depth = l_conf + l_reg + l_grad
        total = total + VGGT_OFFICIAL_DEPTH_WEIGHT * l_depth
        wd = VGGT_OFFICIAL_DEPTH_WEIGHT
        logs["loss_depth"] = float((wp * wd * l_depth).detach().item())
        logs["loss_depth_conf"] = float((wp * wd * l_conf).detach().item())
        logs["loss_depth_reg"] = float((wp * wd * l_reg).detach().item())
        logs["loss_depth_grad"] = float((wp * wd * l_grad).detach().item())

    return total, logs


class VGGTCacheReplayEngine(CacheReplayEngine):
    def __init__(self, backbone_name: str = "vggt") -> None:
        self.backbone_name = backbone_name.strip().lower()
        self._lr_scheduler: Optional[CompositeParamScheduler] = None
        self._weight_decay_scheduler: Optional[ConstantParamScheduler] = None
        self._scheduler_key: Optional[Tuple[float, float, float, float]] = None

    def training_config(self) -> TrainingHyperparams:
        return get_backbone(self.backbone_name).training_config()

    def validate_cache_row(self, row: Dict[str, Any], *, num_layers: int) -> None:
        del num_layers
        pred = row.get("pred")
        if not isinstance(pred, dict):
            raise SystemExit("Invalid cache row: missing dict key 'pred'.")
        get_backbone(self.backbone_name).validate_cached_prediction(pred)

    def build_train_model(
        self,
        *,
        hf_model: str,
        device: torch.device,
        drop_actions: Sequence[PruneAction],
        lora_actions: Sequence[PruneAction],
        frame_skip_intervals: Sequence[Tuple[int, int]],
        global_skip_intervals: Sequence[Tuple[int, int]],
        skipped_frame_idxs: Sequence[int],
        skipped_global_idxs: Sequence[int],
        prune: bool,
        inject_lora: bool,
    ) -> Tuple[nn.Module, List[nn.Parameter], int, int]:
        adapter = get_backbone(self.backbone_name)
        m = adapter.load_pretrained(
            hf_model=hf_model,
            device=str(device),
            dtype=None,
            eval_mode=False,
        )
        if hasattr(m, "point_head"):
            m.point_head = None
        for pp in m.parameters():
            pp.requires_grad_(False)
        skipped_modules_before: List[nn.Module] = []
        for idx in skipped_frame_idxs:
            skipped_modules_before.append(adapter.block_for_unit(m, f"frame_{int(idx)}"))
        for idx in skipped_global_idxs:
            skipped_modules_before.append(adapter.block_for_unit(m, f"global_{int(idx)}"))
        pre_total = sum(p.numel() for p in self.parameter_accounting_parameters(m))
        removed = count_unique_params(skipped_modules_before) if prune else 0
        if prune:
            adapter.apply_prune_plan(m, list(drop_actions))
        trainable_local: List[nn.Parameter] = []
        if inject_lora:
            trainable_local = adapter.inject_lora_plan(m, list(lora_actions))
        return m, trainable_local, pre_total, removed

    def replay_student_cache_first(
        self,
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        num_layers: int,
        *,
        frame_skip_intervals: Sequence[Tuple[int, int]],
        global_skip_intervals: Sequence[Tuple[int, int]],
        lora_anchor_layers: Sequence[int],
        alignment_capture_layers: Sequence[int],
        seed_layer: int,
        seed_kind: str,
        initial_tokens: Optional[torch.Tensor],
        use_checkpoint: bool,
        device: torch.device,
        dummy_images: torch.Tensor,
    ) -> ReplayResult:
        return _replay_model_cache_first(
            model,
            get_layer_cpu,
            num_layers,
            backbone_name=self.backbone_name,
            frame_skip_intervals=frame_skip_intervals,
            global_skip_intervals=global_skip_intervals,
            lora_anchor_layers=lora_anchor_layers,
            alignment_capture_layers=alignment_capture_layers,
            seed_layer=int(seed_layer),
            seed_kind=str(seed_kind),
            initial_tokens=initial_tokens,
            use_checkpoint=use_checkpoint,
            device=device,
            dummy_images=dummy_images,
        )

    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
        dataset_name: str = "",
    ) -> Dict[str, torch.Tensor]:
        return get_backbone(self.backbone_name).build_target_batch_from_teacher(
            teacher_pred,
            images_batched=images_batched,
            device=device,
            dataset_name=dataset_name,
        )

    def build_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        args: Any,
    ) -> torch.optim.Optimizer:
        del args
        cfg = self.training_config()
        return get_backbone(self.backbone_name).build_training_optimizer(trainable, config=cfg)

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        *,
        args: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del args
        cfg = self.training_config()
        weights = PredictionLossWeights(
            w_pred=float(cfg.w_pred),
        )
        return get_backbone(self.backbone_name).prediction_loss(
            student_pred,
            target_batch,
            weights=weights,
        )

    def regularization_loss(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        args: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del args
        cfg = self.training_config()
        loss_reg = torch.zeros((), device=device)
        if float(cfg.w_reg) > 0:
            for p in trainable:
                loss_reg = loss_reg + p.float().pow(2).mean()
        return loss_reg

    def update_optimizer_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        args: Any,
        global_step: int,
        total_steps: int,
    ) -> float:
        del args
        cfg = self.training_config()
        key = (
            float(cfg.lr),
            float(cfg.min_lr),
            float(cfg.warmup_ratio),
            float(cfg.weight_decay),
        )
        if self._lr_scheduler is None or self._weight_decay_scheduler is None or self._scheduler_key != key:
            self._lr_scheduler = _build_vggt_official_lr_scheduler(cfg)
            self._weight_decay_scheduler = _build_vggt_official_weight_decay_scheduler(cfg)
            self._scheduler_key = key
        where = float(global_step) / max(float(total_steps), 1.0)
        where = max(0.0, min(1.0, where))
        if where >= 1.0:
            return float(optimizer.param_groups[0]["lr"])
        lr_now = float(self._lr_scheduler(where))
        weight_decay_now = float(self._weight_decay_scheduler(where))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
            pg["weight_decay"] = weight_decay_now
        return lr_now

    def clip_gradients(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        args: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del args
        cfg = self.training_config()
        grad_norm = torch.tensor(0.0, device=device)
        if float(cfg.grad_clip_max_norm) > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=float(cfg.grad_clip_max_norm),
            )
        return grad_norm

    def supports_dataset_targets(self) -> bool:
        return True

    def supports_periodic_eval(self) -> bool:
        return True

    def trainable_param_module(self, model: nn.Module) -> nn.Module:
        return model.aggregator  # type: ignore[no-any-return]

    def parameter_accounting_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        """Count only prunable frame/global blocks, excluding patch embedding and tokens."""
        yield from model.aggregator.frame_blocks.parameters()
        yield from _global_blocks(model).parameters()

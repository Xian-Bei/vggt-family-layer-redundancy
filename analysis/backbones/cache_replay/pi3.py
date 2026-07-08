"""Pi3 cache-first replay training (shared trainer entry)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from analysis.backbones.cache_replay.base import CacheReplayEngine
from analysis.backbones.cache_replay.common import ReplayResult, is_skipped, split_concat
from analysis.backbones.contracts import PredictionLossWeights, TrainingHyperparams
from analysis.backbones.registry import get_backbone
from analysis.schema.v3 import PruneAction, parse_dataset_plugin_args
from analysis.recover.train.model import count_unique_params


def pi3_official_dataset_name(dataset_name: str) -> str:
    """Map LaRe dataset tags to the names Pi3 official loss uses for dataset gates."""
    name = str(dataset_name or "").strip()
    if name:
        name, _split = parse_dataset_plugin_args(name)
    key = name.lower()
    if key in {"scannet", "scannet_dev50", "scannet_test50"}:
        return "ScanNet"
    if key in {"scannetpp", "scannet++", "scannet_plus_plus"}:
        return "Scannetpp"
    if key == "taskonomy":
        return "Taskonomy"
    if key == "hypersim":
        return "Hypersim"
    return name


def _decoder_index(unit_id: str) -> int:
    kind, idx_s = unit_id.split("_", 1)
    p = int(idx_s)
    return 2 * p if kind == "frame" else 2 * p + 1


def _build_pi3_pos(model: nn.Module, *, B: int, N: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    pos = model.position_getter(B * N, H // int(model.patch_size), W // int(model.patch_size), device)
    if int(model.patch_start_idx) > 0:
        pos = pos + 1
        pos_special = torch.zeros(B * N, int(model.patch_start_idx), 2, device=device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)
    return pos


def _resolve_pi3_image_hw(
    dummy_images: Optional[torch.Tensor],
    image_hw: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    if image_hw is not None:
        return int(image_hw[0]), int(image_hw[1])
    if dummy_images is None:
        raise ValueError("Pi3 cache replay needs image_hw=(H, W) or dummy_images for RoPE grid.")
    t = dummy_images.unsqueeze(0) if dummy_images.dim() == 4 else dummy_images
    if t.dim() != 5:
        raise ValueError(f"Expected dummy_images [B,N,3,H,W], got {tuple(t.shape)}")
    return int(t.shape[-2]), int(t.shape[-1])


def _canonicalize_pi3_replay_seed(
    seed_tensor: torch.Tensor,
    *,
    model: nn.Module,
    h: int,
    w: int,
) -> torch.Tensor:
    if seed_tensor.dim() != 4:
        raise RuntimeError(f"Expected Pi3 seed tokens [B,S,P,C], got {tuple(seed_tensor.shape)}")
    patch_tokens = (int(h) // int(model.patch_size)) * (int(w) // int(model.patch_size))
    frame_tokens = int(patch_tokens) + int(model.patch_start_idx)
    if frame_tokens <= 0:
        raise RuntimeError(f"Invalid Pi3 frame token count: {frame_tokens}")
    b, s, p, c = (int(x) for x in seed_tensor.shape)
    if p == frame_tokens:
        return seed_tensor
    if s == 1 and p % frame_tokens == 0:
        n = p // frame_tokens
        return seed_tensor.reshape(b, n, frame_tokens, c).contiguous()
    raise RuntimeError(
        "Cannot reshape Pi3 replay seed to [B,N,P,C]: "
        f"seed={tuple(seed_tensor.shape)} frame_tokens={frame_tokens} image_hw={(int(h), int(w))}"
    )


def _alignment_concat_layers(
    lora_anchor_layers: Sequence[int],
    alignment_capture_layers: Sequence[int],
) -> set[int]:
    """Layers whose concat activations must be kept for alignment (not for heads)."""
    layers = {int(x) for x in lora_anchor_layers}
    layers.update(int(x) for x in alignment_capture_layers)
    return layers


def _prepare_pi3_cache_replay_model(model: nn.Module) -> None:
    """Cache replay starts from cached decoder activations — never run DINO encoder on GPU."""
    enc = getattr(model, "encoder", None)
    if enc is not None:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name in ("conf_decoder", "conf_head"):
        mod = getattr(model, name, None)
        if mod is None:
            continue
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)


def _set_pi3_head_checkpoint(model: nn.Module, use_checkpoint: bool) -> None:
    for name in ("point_decoder", "camera_decoder"):
        dec = getattr(model, name, None)
        if dec is not None and hasattr(dec, "use_checkpoint"):
            dec.use_checkpoint = bool(use_checkpoint)


def _run_pi3_block(
    block: nn.Module,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    *,
    training: bool,
    use_checkpoint: bool,
) -> torch.Tensor:
    if (not training) or (not use_checkpoint):
        return block(hidden, xpos=pos)
    return checkpoint(lambda h, p: block(h, xpos=p), hidden, pos, use_reentrant=False)


def _replay_pi3_cache_first(
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
    dummy_images: Optional[torch.Tensor] = None,
    image_hw: Optional[Tuple[int, int]] = None,
) -> ReplayResult:
    if num_layers <= 0:
        raise ValueError("Replay cache is empty.")
    start_layer = min(int(x) for x in lora_anchor_layers) if lora_anchor_layers else 0
    if start_layer < 0:
        raise ValueError(f"Expected start_layer>=0, got {start_layer}")
    if start_layer >= num_layers:
        raise ValueError(f"start_layer={start_layer} out of range for num_layers={num_layers}")

    kind = str(seed_kind).strip().lower()
    if kind == "none":
        if int(start_layer) != 0:
            raise ValueError(
                f"seed_kind=none only supported when replay starts at layer 0, got start_layer={start_layer}"
            )
        if initial_tokens is None:
            raise RuntimeError("initial_tokens are required when seed_kind=none")
        seed_tensor = initial_tokens.to(device=device, non_blocking=(device.type == "cuda"))
        if seed_tensor.dim() != 4:
            raise RuntimeError(f"Expected layer_-1 tokens [B,S,P,C], got {tuple(seed_tensor.shape)}")
        use_frame_seed = True
    else:
        seed_concat = get_layer_cpu(int(seed_layer)).to(device=device, non_blocking=(device.type == "cuda"))
        seed_frame, seed_global = split_concat(seed_concat)
        if kind == "frame":
            seed_tensor = seed_frame
        elif kind == "global":
            seed_tensor = seed_global
        else:
            raise ValueError(f"seed_kind must be frame, global, or none, got {seed_kind!r}")
        use_frame_seed = kind == "frame"
    H, W = _resolve_pi3_image_hw(dummy_images, image_hw)
    seed_tensor = _canonicalize_pi3_replay_seed(seed_tensor, model=model, h=H, w=W)
    bsz, seq_len, p, c = (int(x) for x in seed_tensor.shape)
    B, N = bsz, seq_len
    pos = _build_pi3_pos(model, B=B, N=N, H=H, W=W, device=device)

    align_layers = _alignment_concat_layers(lora_anchor_layers, alignment_capture_layers)
    concat_out: Dict[int, torch.Tensor] = {}

    hidden = seed_tensor.reshape(B, N * p, c)
    dec_depth = len(model.decoder)
    penultimate_frame: Optional[torch.Tensor] = None
    last_global: Optional[torch.Tensor] = None

    for layer_idx in range(start_layer, num_layers):
        fi, gi = 2 * layer_idx, 2 * layer_idx + 1
        # frame block
        pos_f = pos.reshape(B * N, p, -1)
        if use_frame_seed and layer_idx == start_layer:
            h_out = seed_tensor.reshape(B * N, p, c)
        else:
            h_in = hidden.reshape(B * N, p, c)
            if is_skipped(layer_idx, frame_skip_intervals):
                h_out = h_in
            else:
                h_out = _run_pi3_block(
                    model.decoder[fi],
                    h_in,
                    pos_f,
                    training=bool(model.training),
                    use_checkpoint=use_checkpoint,
                )

        # global block
        pos_g = pos.reshape(B, N * p, -1)
        g_in = h_out.reshape(B, N * p, c)
        if is_skipped(layer_idx, global_skip_intervals):
            g_out = g_in
        else:
            g_out = _run_pi3_block(
                model.decoder[gi],
                g_in,
                pos_g,
                training=bool(model.training),
                use_checkpoint=use_checkpoint,
            )
        hidden = g_out

        if layer_idx in align_layers:
            frame_view = h_out.reshape(B, N, p, c)
            global_view = g_out.reshape(B, N, p, c)
            concat_out[layer_idx] = torch.cat([frame_view, global_view], dim=-1)

        # Heads only need the last two decoder block outputs (same as Pi3.decode).
        if (fi + 1) == dec_depth - 1:
            penultimate_frame = h_out.reshape(B * N, p, -1)
        if (gi + 1) == dec_depth:
            last_global = g_out.reshape(B * N, p, -1)

    if penultimate_frame is None or last_global is None:
        raise RuntimeError("Pi3 replay did not reach final decoder layers for head forward.")
    head_in = torch.cat([penultimate_frame, last_global], dim=-1)
    del penultimate_frame, last_global
    pos_bn = pos.reshape(B * N, p, -1)
    _set_pi3_head_checkpoint(model, use_checkpoint)

    patch_h, patch_w = int(H) // 14, int(W) // 14
    patch_start = int(model.patch_start_idx)
    point_hidden = model.point_decoder(head_in, xpos=pos_bn)
    camera_hidden = model.camera_decoder(head_in, xpos=pos_bn)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        point_hidden = point_hidden.float()
        ret = model.point_head([point_hidden[:, patch_start:]], (H, W)).reshape(B, N, H, W, -1)
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1)
        del point_hidden, ret

        camera_hidden = camera_hidden.float()
        camera_poses = model.camera_head(camera_hidden[:, patch_start:], patch_h, patch_w).reshape(
            B, N, 4, 4
        )
        del camera_hidden, head_in

        from analysis.backbones.cache_replay.pi3_official.geometry import homogenize_points

        points = torch.einsum("bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points))[..., :3]
    pred = {
        "points": points,
        "local_points": local_points,
        "camera_poses": camera_poses,
        "global_points": None,
    }
    return ReplayResult(concat_by_layer=concat_out, pred=pred)


def build_target_batch_from_teacher_pi3(
    teacher_pred: Dict[str, torch.Tensor],
    *,
    images_batched: torch.Tensor,
    device: torch.device,
    dataset_name: str = "",
) -> Dict[str, torch.Tensor]:
    if "points" not in teacher_pred or "camera_poses" not in teacher_pred:
        raise RuntimeError("Pi3 teacher target mode requires cached keys: points and camera_poses")
    pts = teacher_pred["points"].float().to(device)
    poses = teacher_pred["camera_poses"].float().to(device)
    if pts.dim() == 4:  # [N,H,W,3] -> [1,N,H,W,3]
        pts = pts.unsqueeze(0)
    if poses.dim() == 3:  # [N,4,4] -> [1,N,4,4]
        poses = poses.unsqueeze(0)
    if images_batched.dim() == 4:
        images_batched = images_batched.unsqueeze(0)
    imgs = images_batched.float()

    # valid_mask: teacher targets do not provide GT mask; use finite + positive depth.
    valid = torch.isfinite(pts).all(dim=-1) & (pts[..., 2] > 1e-6)
    if valid.dim() == 3:  # [N,H,W] -> [1,N,H,W]
        valid = valid.unsqueeze(0)

    # Build the exact gt_raw structure expected by the official Pi3Loss.
    B, N, H, W, _ = (int(x) for x in pts.shape)
    if imgs.shape[0] != B or imgs.shape[1] != N or imgs.shape[-2] != H or imgs.shape[-1] != W:
        # Cache replay uses a lightweight dummy image tensor; loss only needs shape-matched placeholders.
        imgs = torch.zeros((B, N, 3, H, W), device=device, dtype=torch.float32)
    gt_raw: List[Dict[str, torch.Tensor | str | List[str]]] = []
    pi3_dataset_name = pi3_official_dataset_name(dataset_name)
    dataset_names = [pi3_dataset_name] * B
    for i in range(B):
        views: List[Dict[str, Any]] = []
        for j in range(N):
            views.append(
                {
                    "img": imgs[i, j].unsqueeze(0),
                    "pts3d": pts[i, j].unsqueeze(0),
                    "valid_mask": valid[i, j].unsqueeze(0),
                    "camera_pose": poses[i, j].unsqueeze(0),
                    "dataset": dataset_names,
                }
            )
        gt_raw.append({"views": views})
    # Flatten into list-of-views (B is effectively always 1 in analysis pipeline).
    views0 = gt_raw[0]["views"]  # type: ignore[index]
    return {"gt_raw": views0, "images": imgs, "pi3_dataset_names": dataset_names}


def prediction_points_conf(
    student_pred: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    *,
    w_pred: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    del w_pred
    from analysis.backbones.cache_replay.pi3_official.loss import Pi3Loss

    gt_raw = target_batch.get("gt_raw")
    if not isinstance(gt_raw, list) or not gt_raw:
        raise RuntimeError("Pi3 target_batch must contain non-empty list gt_raw")
    # Official loss expects local_points and camera_poses. global_points is optional
    # and disabled for the official default Pi3 configuration.
    device = next(iter(student_pred.values())).device
    student_pred_fp32: Dict[str, torch.Tensor] = {
        k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
        for k, v in student_pred.items()
    }
    gt_raw_fp32: List[Dict[str, Any]] = []
    for view in gt_raw:
        gt_raw_fp32.append(
            {
                k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
                for k, v in view.items()
            }
        )
    loss_fn = Pi3Loss(train_conf=False).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        total_loss, details = loss_fn(student_pred_fp32, gt_raw_fp32)
    logs: Dict[str, float] = {f"pi3_{k}": float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in details.items()}
    logs["loss_pi3_total"] = float(total_loss.detach().item())
    return total_loss, logs


class Pi3CacheReplayEngine(CacheReplayEngine):
    backbone_name = "pi3"

    def __init__(self) -> None:
        self._lr_scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR] = None
        self._lr_scheduler_key: Optional[Tuple[int, int, float]] = None
        self._lr_scheduler_steps = 0

    @staticmethod
    def layers_to_keep_in_cpu_cache(
        *,
        seed_layer: int,
        alignment_layers: Sequence[int],
    ) -> set[int]:
        """CPU layer tensors still needed after one training step (alignment + replay seed)."""
        keep = {int(seed_layer)}
        for r in alignment_layers:
            keep.add(int(r))
        return keep

    @staticmethod
    def release_layer_cpu_cache(layer_cpu_cache: Dict[int, torch.Tensor], keep: set[int]) -> None:
        for idx in list(layer_cpu_cache.keys()):
            if int(idx) not in keep:
                del layer_cpu_cache[int(idx)]

    def validate_cache_row(self, row: Dict[str, Any], *, num_layers: int) -> None:
        del num_layers
        pred = row.get("pred")
        if not isinstance(pred, dict):
            raise SystemExit("Invalid cache row: missing dict key 'pred'.")
        get_backbone(self.backbone_name).validate_cached_prediction(pred)

    def training_config(self) -> TrainingHyperparams:
        return get_backbone(self.backbone_name).training_config()

    def normalize_alignment_term(self, term: torch.Tensor) -> torch.Tensor:
        if term.dim() != 4:
            return term
        b, s, p, c = (int(x) for x in term.shape)
        return term.reshape(b, 1, s * p, c)

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
        m = adapter.load_pretrained(hf_model=hf_model, device=str(device), dtype=None, eval_mode=False)
        skipped_modules: List[nn.Module] = []
        if prune:
            for idx in skipped_frame_idxs:
                skipped_modules.append(adapter.block_for_unit(m, f"frame_{int(idx)}"))
            for idx in skipped_global_idxs:
                skipped_modules.append(adapter.block_for_unit(m, f"global_{int(idx)}"))
        pre_total = sum(p.numel() for p in m.decoder.parameters())
        removed = count_unique_params(skipped_modules) if prune else 0
        if prune:
            adapter.apply_prune_plan(m, list(drop_actions))
        trainable: List[nn.Parameter] = []
        if inject_lora:
            trainable = adapter.inject_lora_plan(m, list(lora_actions))
        m.to(device=device)
        _prepare_pi3_cache_replay_model(m)
        return m, trainable, pre_total, removed

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
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> ReplayResult:
        return _replay_pi3_cache_first(
            model,
            get_layer_cpu,
            num_layers,
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
            image_hw=image_hw,
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
        total = max(int(total_steps), 1)
        key = (id(optimizer), total, float(cfg.lr))
        if self._lr_scheduler is None or self._lr_scheduler_key != key:
            self._lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(cfg.lr),
                total_steps=total,
                pct_start=0.0,
                anneal_strategy="cos",
                div_factor=1000.0,
                final_div_factor=100.0,
            )
            self._lr_scheduler_key = key
            self._lr_scheduler_steps = 0

        # The shared trainer asks for the LR before optimizer.step(), while Pi3
        # official calls scheduler.step() after optimizer.step(). Advance here
        # only for optimizer steps already completed by previous iterations.
        completed_steps = max(0, min(int(global_step), total))
        while self._lr_scheduler_steps < completed_steps:
            self._lr_scheduler.step()
            self._lr_scheduler_steps += 1

        return float(optimizer.param_groups[0]["lr"])

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

    def trainable_param_module(self, model: nn.Module) -> nn.Module:
        return model.decoder

    def supports_periodic_eval(self) -> bool:
        return True

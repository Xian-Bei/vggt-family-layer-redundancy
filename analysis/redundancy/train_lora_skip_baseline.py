from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

# Keep OpenCV/Matplotlib in headless mode for eval_utils imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

_TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))

from vggt.models.vggt import VGGT
from analysis.eval.eval_utils import (
    build_frame_selection,
    get_all_scenes,
    get_sorted_image_paths,
    get_vgg_input_imgs,
    infer_vggt_and_reconstruct,
    load_images_rgb,
    load_poses,
)
from analysis.eval import evaluate_scene_and_save
from .collect import collect_batch
from .cache_layout import (
    load_scene_metadata,
    load_scene_pred,
    load_scene_replay_layer,
)
from analysis.eval.prediction_eval import (
    _depth_or_conf_to_shw,
    _load_gt_depth_for_scene,
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
    write_eval_scannet_json,
)
from .prediction_save import build_scene_prediction_payload
from .scannet_eval import (
    SceneData,
    aggregate_metric_dict,
    discover_scannet_scenes,
    load_scene_images,
    select_scene_image_paths_fastvggt,
)
from training.loss import compute_camera_loss, compute_depth_loss
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@dataclass(frozen=True)
class IntervalRule:
    l: int
    r: int
    skip_frame_in_interval: bool
    inject_prev: Literal["global", "frame_global"]
    inject_frame_inside_interval: bool

    @property
    def prev_layer(self) -> int:
        return int(self.l) - 1

    @property
    def has_frame_lora(self) -> bool:
        return bool(self.inject_prev == "frame_global" or self.inject_frame_inside_interval)


def _load_interval_rules(path: Path) -> List[IntervalRule]:
    if not path.exists():
        raise SystemExit(f"Missing --interval-config-file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules_raw: Any = payload.get("intervals") if isinstance(payload, dict) else payload
    if not isinstance(rules_raw, list) or not rules_raw:
        raise SystemExit("Interval config must be a non-empty list (or {\"intervals\": [...]}).")

    parsed: List[IntervalRule] = []
    for i, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise SystemExit(f"intervals[{i}] must be an object.")
        try:
            l = int(item["l"])
            r = int(item["r"])
        except Exception as e:
            raise SystemExit(f"intervals[{i}] must provide integer l/r: {e}") from e
        if l < 0 or r < 0 or l > r:
            raise SystemExit(f"intervals[{i}] invalid range [{l},{r}]")
        inject_prev = str(item.get("inject_prev", "global"))
        if inject_prev not in ("global", "frame_global"):
            raise SystemExit(f"intervals[{i}].inject_prev must be 'global' or 'frame_global'.")
        skip_frame = bool(item.get("skip_frame_in_interval", True))
        inject_inside = bool(item.get("inject_frame_inside_interval", False))
        if skip_frame and inject_inside:
            raise SystemExit(
                f"intervals[{i}] has skip_frame_in_interval=true and inject_frame_inside_interval=true; incompatible."
            )
        parsed.append(
            IntervalRule(
                l=l,
                r=r,
                skip_frame_in_interval=skip_frame,
                inject_prev=inject_prev,  # type: ignore[arg-type]
                inject_frame_inside_interval=inject_inside,
            )
        )
    parsed = sorted(parsed, key=lambda x: (int(x.l), int(x.r)))
    for i in range(1, len(parsed)):
        pl, pr = int(parsed[i - 1].l), int(parsed[i - 1].r)
        l, r = int(parsed[i].l), int(parsed[i].r)
        if l <= pr:
            raise SystemExit(f"Overlapped intervals in config: [{pl},{pr}] and [{l},{r}]")
    return parsed


def _intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return [(int(x.l), int(x.r)) for x in rules]


def _global_skip_intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return _intervals_from_rules(rules)


def _frame_skip_intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return [(int(x.l), int(x.r)) for x in rules if bool(x.skip_frame_in_interval)]


def _lora_injection_plan_from_rules(
    rules: Sequence[IntervalRule],
) -> Tuple[List[int], List[int]]:
    frame_idxs: set[int] = set()
    global_idxs: set[int] = set()
    for rule in rules:
        pidx = int(rule.prev_layer)
        global_idxs.add(pidx)
        if rule.inject_prev == "frame_global":
            frame_idxs.add(pidx)
        if (not rule.skip_frame_in_interval) and bool(rule.inject_frame_inside_interval):
            frame_idxs.update(range(int(rule.l), int(rule.r) + 1))
    return sorted(frame_idxs), sorted(global_idxs)


def _load_manifest(cache_dir: Path) -> List[Dict[str, Any]]:
    p = cache_dir / "manifest.json"
    if not p.exists():
        raise SystemExit(f"Missing cache manifest: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise SystemExit(f"Manifest has no scenes: {p}")
    return scenes


def _load_scene_cache_row(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(load_scene_metadata(cache_dir, item))
    row["pred"] = load_scene_pred(cache_dir, item)
    return row


def _split_concat(layer_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B,S,P,2C] -> [B,S,P,C] frame/global
    if layer_t.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(layer_t.shape)}")
    two_c = int(layer_t.shape[-1])
    if two_c % 2 != 0:
        raise ValueError(f"Expected even channel dimension, got {two_c}")
    c = two_c // 2
    return layer_t[..., :c], layer_t[..., c:]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        in_f = int(base.in_features)
        out_f = int(base.out_features)
        self.lora_A = nn.Linear(in_f, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Keep LoRA params on the same device/dtype as wrapped base layer.
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class SkipIdentityBlock(nn.Module):
    """
    Identity block that is API-compatible with transformer blocks (accepts pos kwarg).
    Used to physically replace skipped layers while keeping module list indices unchanged.
    """

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


def _inject_lora_attn(block: nn.Module, rank: int, alpha: float, dropout: float) -> List[nn.Parameter]:
    trainable: List[nn.Parameter] = []
    attn = getattr(block, "attn", None)
    if attn is None:
        return trainable
    for name in ("qkv", "proj"):
        mod = getattr(attn, name, None)
        if isinstance(mod, nn.Linear):
            lora_mod = LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout)
            setattr(attn, name, lora_mod)
            trainable.extend([p for p in lora_mod.parameters() if p.requires_grad])
    return trainable


def _is_skipped(layer_idx: int, intervals: Sequence[Tuple[int, int]]) -> bool:
    for l, r in intervals:
        if l <= layer_idx <= r:
            return True
    return False


def _replace_skipped_layers_with_identity(
    model: VGGT,
    *,
    frame_intervals: Sequence[Tuple[int, int]],
    global_intervals: Sequence[Tuple[int, int]],
) -> None:
    depth = len(model.aggregator.frame_blocks)
    for l, r in frame_intervals:
        for idx in range(int(l), int(r) + 1):
            if idx < 0 or idx >= depth:
                raise SystemExit(f"Skip layer index {idx} out of range for model depth {depth}.")
            model.aggregator.frame_blocks[idx] = SkipIdentityBlock()
    for l, r in global_intervals:
        for idx in range(int(l), int(r) + 1):
            if idx < 0 or idx >= depth:
                raise SystemExit(f"Skip layer index {idx} out of range for model depth {depth}.")
            model.aggregator.global_blocks[idx] = SkipIdentityBlock()


def _build_pos(model: VGGT, images: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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


def _run_block_with_checkpoint(
    block: nn.Module,
    x: torch.Tensor,
    *,
    pos: Optional[torch.Tensor],
    training: bool,
    use_checkpoint: bool,
    use_reentrant: bool,
) -> torch.Tensor:
    """
    Match official aggregator behavior: training-time checkpointing with use_reentrant=False.
    """
    if (not training) or (not use_checkpoint):
        return block(x, pos=pos)
    if pos is None:
        return checkpoint(lambda _x: block(_x, pos=None), x, use_reentrant=use_reentrant)
    return checkpoint(lambda _x, _p: block(_x, pos=_p), x, pos, use_reentrant=use_reentrant)


def _required_head_layers(model: VGGT, num_layers: int) -> List[int]:
    keep: set[int] = set()
    if model.camera_head is not None and num_layers > 0:
        keep.add(num_layers - 1)
    if model.depth_head is not None:
        keep.update(int(i) for i in getattr(model.depth_head, "intermediate_layer_idx", []))
    return sorted(i for i in keep if 0 <= i < num_layers)


@dataclass
class ReplayResult:
    concat_by_layer: Dict[int, torch.Tensor]
    pred: Dict[str, torch.Tensor]


def _replay_student_cache_first(
    model: VGGT,
    get_layer_cpu: Callable[[int], torch.Tensor],
    num_layers: int,
    *,
    frame_skip_intervals: Sequence[Tuple[int, int]],
    global_skip_intervals: Sequence[Tuple[int, int]],
    lora_anchor_prev_layers: Sequence[int],
    use_checkpoint: bool,
    device: torch.device,
    dummy_images: torch.Tensor,
) -> ReplayResult:
    _get_layer_cpu = get_layer_cpu
    if num_layers <= 0:
        raise ValueError("Replay cache is empty.")
    start_layer = min(int(x) for x in lora_anchor_prev_layers)
    if start_layer <= 0:
        raise ValueError(f"Expected start_layer>0, got {start_layer}")
    if start_layer >= num_layers:
        raise ValueError(f"start_layer={start_layer} out of range for num_layers={num_layers}")

    # Seed from cached global of layer start-1
    seed_concat = _get_layer_cpu(start_layer - 1).to(device=device, non_blocking=(device.type == "cuda"))
    _, seed_global = _split_concat(seed_concat)
    bsz, seq_len, p, c = seed_global.shape

    pos_frame, pos_global = _build_pos(model, dummy_images)
    if pos_frame is not None and pos_frame.shape[0] != bsz * seq_len:
        raise RuntimeError(
            f"Position mismatch: pos_frame={tuple(pos_frame.shape)}, replay b*s={bsz*seq_len}"
        )

    required = set(_required_head_layers(model, num_layers))
    required.update(int(x) for x in lora_anchor_prev_layers)
    concat_out: Dict[int, torch.Tensor] = {}

    global_prev = seed_global.view(bsz, seq_len * p, c)
    use_reentrant = bool(getattr(model.aggregator, "use_reentrant", False))
    for layer_idx in range(start_layer, num_layers):
        frame_in = global_prev.view(bsz * seq_len, p, c)
        if _is_skipped(layer_idx, frame_skip_intervals):
            frame_out = frame_in
        else:
            frame_out = _run_block_with_checkpoint(
                model.aggregator.frame_blocks[layer_idx],
                frame_in,
                pos=pos_frame,
                training=bool(model.training),
                use_checkpoint=use_checkpoint,
                use_reentrant=use_reentrant,
            )
        frame_view = frame_out.view(bsz, seq_len, p, c)

        global_in = frame_view.view(bsz, seq_len * p, c)
        if _is_skipped(layer_idx, global_skip_intervals):
            global_out = global_in
        else:
            global_out = _run_block_with_checkpoint(
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
            selected[idx] = _get_layer_cpu(idx).to(device=device, non_blocking=(device.type == "cuda"))
        else:
            if idx not in concat_out:
                raise RuntimeError(f"Missing replay output at required layer {idx}")
            selected[idx] = concat_out[idx]
    agg_list: List[torch.Tensor] = [None] * num_layers  # type: ignore[list-item]
    for idx, t in selected.items():
        agg_list[idx] = t
    pred = model.forward_from_aggregated_tokens(
        agg_list,
        int(model.aggregator.patch_start_idx),
        dummy_images,
    )
    return ReplayResult(concat_by_layer=concat_out, pred=pred)


def _loss_mse_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(a, b)
    af = a.reshape(-1, a.shape[-1]).float()
    bf = b.reshape(-1, b.shape[-1]).float()
    af = af / (af.norm(dim=-1, keepdim=True) + eps)
    bf = bf / (bf.norm(dim=-1, keepdim=True) + eps)
    cos = 1.0 - (af * bf).sum(dim=-1).mean()
    return mse, cos


def _spatial_grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Supports [B,S,H,W,C] or [B,S,H,W]
    if pred.dim() == 5:
        p = pred.permute(0, 1, 4, 2, 3).contiguous()
        t = target.permute(0, 1, 4, 2, 3).contiguous()
    elif pred.dim() == 4:
        p = pred.unsqueeze(2)
        t = target.unsqueeze(2)
    else:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if p.shape[-2] < 2 or p.shape[-1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    px = p[..., 1:, :] - p[..., :-1, :]
    tx = t[..., 1:, :] - t[..., :-1, :]
    py = p[..., :, 1:] - p[..., :, :-1]
    ty = t[..., :, 1:] - t[..., :, :-1]
    return 0.5 * (F.l1_loss(px, tx) + F.l1_loss(py, ty))


def _teacher_pred_to_map(pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            # Keep teacher targets on CPU to reduce persistent GPU memory.
            t = v.contiguous()
            if t.dim() >= 1 and t.shape[0] != 1:
                t = t.unsqueeze(0)
            out[k] = t
    return out


def _depth_to_bshw(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.dim() == 5 and depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.dim() == 4:
        return depth
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _mask_from_depth(depth_bshw: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(depth_bshw) & (depth_bshw > 1e-6)


def _load_scannet_intrinsics(scene_dir: Path) -> np.ndarray:
    intrinsic_dir = scene_dir / "intrinsic"
    candidates = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
        scene_dir / "intrinsic.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        vals: List[List[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals.append([float(x) for x in line.split()])
        arr = np.array(vals, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr[:3, :3].copy()
        if arr.shape == (3, 3):
            return arr.copy()
    raise FileNotFoundError(f"Cannot find ScanNet intrinsics under {scene_dir}")


def _resize_intrinsics_like_vgg_input(
    k: np.ndarray,
    *,
    rgb_width: int,
    rgb_height: int,
    new_width: int = 518,
) -> np.ndarray:
    k = k.astype(np.float32, copy=True)
    scale = float(new_width) / float(rgb_width)
    new_height = round(rgb_height * scale / 14) * 14
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] *= scale
    k[1, 2] *= scale
    if new_height > 518:
        start_y = (new_height - 518) // 2
        k[1, 2] -= float(start_y)
    return k


def _build_target_batch_from_teacher(
    teacher_pred: Dict[str, torch.Tensor],
    *,
    images_batched: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if "pose_enc" not in teacher_pred or "depth" not in teacher_pred:
        raise RuntimeError("teacher target mode requires teacher keys: pose_enc and depth")
    h = int(images_batched.shape[-2])
    w = int(images_batched.shape[-1])
    t_pose = teacher_pred["pose_enc"].float().to(device)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(t_pose, (h, w))
    t_depth = _depth_to_bshw(teacher_pred["depth"].float().to(device))
    point_masks = _mask_from_depth(t_depth)
    return {
        "images": images_batched.float(),
        "extrinsics": extrinsics.float(),
        "intrinsics": intrinsics.float(),
        "depths": t_depth.float(),
        "point_masks": point_masks.bool(),
    }


def _select_scannet_scene_with_frames(
    scene: SceneData,
    *,
    max_frames: int,
) -> SceneData:
    selected = select_scene_image_paths_fastvggt(scene, max_frames=max_frames)
    return selected


def _build_target_batch_from_dataset(
    *,
    scene: SceneData,
    expected_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    selected = _select_scannet_scene_with_frames(scene, max_frames=expected_seq_len)
    if len(selected.image_paths) != int(expected_seq_len):
        raise RuntimeError(
            f"Scene {scene.scene_id} selected {len(selected.image_paths)} frames, expected {expected_seq_len}"
        )

    images_np = load_images_rgb(selected.image_paths)
    vgg_input, _pw, _ph = get_vgg_input_imgs(np.array(images_np))
    image_hw = (int(vgg_input.shape[-2]), int(vgg_input.shape[-1]))
    gt_depth_np = _load_gt_depth_for_scene(selected, image_hw=image_hw)
    if gt_depth_np is None:
        raise RuntimeError(f"Failed to load GT depth for scene {scene.scene_id}")

    pose_dir = selected.scene_dir / "pose"
    if not pose_dir.is_dir():
        raise RuntimeError(f"Missing pose dir for scene {scene.scene_id}: {pose_dir}")
    c2ws_all, _first_gt_pose, available_frame_ids = load_poses(pose_dir)
    pose_frame_to_idx = {int(fid): idx for idx, fid in enumerate(available_frame_ids.tolist())}
    frame_ids = [int(p.stem) for p in selected.image_paths]
    indices = [pose_frame_to_idx[int(fid)] for fid in frame_ids]
    c2ws = c2ws_all[np.array(indices, dtype=np.int64)]
    w2cs = np.linalg.inv(c2ws)[:, :3, :4].astype(np.float32, copy=False)

    k_src = _load_scannet_intrinsics(selected.scene_dir)
    k_list: List[np.ndarray] = []
    for p in selected.image_paths:
        with Image.open(p) as im:
            rgb_w, rgb_h = im.size
        k_list.append(
            _resize_intrinsics_like_vgg_input(
                k_src,
                rgb_width=int(rgb_w),
                rgb_height=int(rgb_h),
                new_width=int(image_hw[1]),
            )
        )
    intrinsics = np.stack(k_list, axis=0).astype(np.float32, copy=False)

    depths = torch.from_numpy(gt_depth_np).float().unsqueeze(0).to(device)
    images = vgg_input.float().unsqueeze(0).to(device)
    point_masks = _mask_from_depth(depths)
    return {
        "images": images,
        "extrinsics": torch.from_numpy(w2cs).float().unsqueeze(0).to(device),
        "intrinsics": torch.from_numpy(intrinsics).float().unsqueeze(0).to(device),
        "depths": depths,
        "point_masks": point_masks.bool(),
    }


def _prediction_official_camera_depth(
    student_pred: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    *,
    w_camera: float,
    w_depth: float,
    w_depth_grad: float,
    camera_loss_type: str,
    depth_valid_range: float,
    depth_gradient_loss_fn: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(iter(student_pred.values())).device
    total = torch.zeros((), device=device)
    logs: Dict[str, float] = {}

    if "pose_enc_list" in student_pred:
        camera_loss_dict = compute_camera_loss(
            {"pose_enc_list": student_pred["pose_enc_list"]},
            target_batch,
            loss_type=camera_loss_type,
        )
        l_camera = camera_loss_dict["loss_camera"]
        total = total + float(w_camera) * l_camera
        logs["loss_camera"] = float(l_camera.detach().item())
        logs["loss_T"] = float(camera_loss_dict["loss_T"].detach().item())
        logs["loss_R"] = float(camera_loss_dict["loss_R"].detach().item())
        logs["loss_FL"] = float(camera_loss_dict["loss_FL"].detach().item())
        logs["loss_camera_term"] = float((float(w_camera) * l_camera).detach().item())

    if "depth" in student_pred and "depth_conf" in student_pred:
        depth_loss_dict = compute_depth_loss(
            {"depth": student_pred["depth"], "depth_conf": student_pred["depth_conf"]},
            target_batch,
            gradient_loss_fn=depth_gradient_loss_fn,
            valid_range=depth_valid_range,
        )
        l_conf = depth_loss_dict["loss_conf_depth"]
        l_reg = depth_loss_dict["loss_reg_depth"]
        l_grad = depth_loss_dict["loss_grad_depth"]
        l_depth = l_conf + l_reg + float(w_depth_grad) * l_grad
        total = total + float(w_depth) * l_depth
        logs["loss_conf_depth"] = float(l_conf.detach().item())
        logs["loss_reg_depth"] = float(l_reg.detach().item())
        logs["loss_grad_depth"] = float(l_grad.detach().item())
        logs["loss_depth_term"] = float((float(w_depth) * l_depth).detach().item())
        logs["loss_depth_conf_term"] = float((float(w_depth) * l_conf).detach().item())
        logs["loss_depth_reg_term"] = float((float(w_depth) * l_reg).detach().item())
        logs["loss_depth_grad_term"] = float(
            (float(w_depth) * float(w_depth_grad) * l_grad).detach().item()
        )

    return total, logs


def _lr_with_warmup_cosine(
    *,
    where: float,
    base_lr: float,
    min_lr: float,
    warmup_ratio: float,
) -> float:
    where = float(max(0.0, min(1.0, where)))
    warmup_ratio = float(max(0.0, min(1.0, warmup_ratio)))
    if warmup_ratio > 0.0 and where < warmup_ratio:
        t = where / max(warmup_ratio, 1e-12)
        return float(min_lr + t * (base_lr - min_lr))
    if where >= 1.0:
        return float(min_lr)
    if warmup_ratio >= 1.0:
        return float(base_lr)
    t = (where - warmup_ratio) / max(1.0 - warmup_ratio, 1e-12)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(min_lr + (base_lr - min_lr) * cos_decay)


def _compute_total_steps(num_scenes: int, epochs: int, max_steps: int) -> int:
    natural = max(0, int(num_scenes) * int(epochs))
    if max_steps > 0:
        return min(natural, int(max_steps))
    return natural


def _build_skipped_layer_indices(intervals: Sequence[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    for l, r in intervals:
        out.extend(range(int(l), int(r) + 1))
    return sorted(set(out))


def _count_unique_params(modules: Sequence[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for m in modules:
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += int(p.numel())
    return total


def _format_param_count(n: int) -> str:
    v = float(n)
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return str(int(v))


def _validate_schema(row: Dict[str, Any], num_layers: int) -> None:
    pred = row.get("pred")
    if not isinstance(pred, dict):
        raise SystemExit("Invalid cache row: missing dict key 'pred'.")
    required = ("pose_enc", "depth", "depth_conf")
    missing = [k for k in required if k not in pred]
    if missing:
        raise SystemExit(f"Invalid cache row pred: missing keys {missing}")


def _infer_num_layers(cache_dir: Path, scenes: Sequence[Dict[str, Any]]) -> int:
    for item in scenes:
        scene_dir_name = str(item.get("dir", "")).strip()
        if not scene_dir_name:
            sid = str(item.get("scene_id", "")).strip()
            scene_dir_name = sid.replace("/", "_") if sid else ""
        if not scene_dir_name:
            continue
        layers_dir = cache_dir / scene_dir_name / "layers"
        if not layers_dir.exists():
            continue
        layer_files = sorted(layers_dir.glob("layer_*.pt"))
        if layer_files:
            return len(layer_files)
    raise SystemExit("Could not infer num_layers from cache.")


def _write_curves(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "train_curves.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    numeric_keys = sorted(
        {
            str(k)
            for r in rows
            for k, v in r.items()
            if k != "epoch" and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    )
    epoch_buckets: Dict[int, Dict[str, float]] = {}
    epoch_counts: Dict[int, int] = {}
    for r in rows:
        ep = int(r.get("epoch", 0))
        if ep not in epoch_buckets:
            epoch_buckets[ep] = {k: 0.0 for k in numeric_keys}
            epoch_counts[ep] = 0
        for k in numeric_keys:
            epoch_buckets[ep][k] += float(r.get(k, 0.0))
        epoch_counts[ep] += 1

    epoch_rows: List[Dict[str, Any]] = []
    for ep in sorted(epoch_buckets.keys()):
        c = max(1, epoch_counts[ep])
        row: Dict[str, Any] = {"epoch": ep, "steps_in_epoch": c}
        for k in numeric_keys:
            row[f"mean_{k}"] = epoch_buckets[ep][k] / c
        epoch_rows.append(row)
    with (out_dir / "train_curves_epoch_avg.csv").open("w", encoding="utf-8", newline="") as f:
        epoch_fields = sorted({k for r in epoch_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=epoch_fields)
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        epochs = [int(r.get("epoch", 0)) for r in epoch_rows]
        total = [float(r.get("mean_loss_total", float("nan"))) for r in epoch_rows]
        align = [float(r.get("mean_loss_align", float("nan"))) for r in epoch_rows]
        pred = [float(r.get("mean_loss_pred", float("nan"))) for r in epoch_rows]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, total, label="mean_loss_total")
        plt.plot(epochs, align, label="mean_loss_align")
        plt.plot(epochs, pred, label="mean_loss_pred")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train Curves (epoch average, scans_dev50)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "train_curves.png", dpi=150)
        plt.close()

        # Single figure: only terms with independent weights.
        weighted_cols = [
            # Alignment weighted parts
            ("mean_loss_align_mse_term", "align_mse"),
            ("mean_loss_align_cos_term", "align_cos"),
            # Prediction weighted parts (outer w_pred already applied in *_contrib)
            ("mean_loss_camera_term_contrib", "pred_camera"),
            ("mean_loss_depth_reg_term_contrib", "pred_depth_reg"),
            ("mean_loss_depth_conf_term_contrib", "pred_depth_conf"),
            ("mean_loss_depth_grad_term_contrib", "pred_depth_grad"),
        ]
        plt.figure(figsize=(11, 6))
        any_plotted = False
        for col, label in weighted_cols:
            ys = [float(r.get(col, float("nan"))) for r in epoch_rows]
            if not any(np.isfinite(y) for y in ys):
                continue
            plt.plot(epochs, ys, label=label)
            any_plotted = True
        if any_plotted:
            plt.xlabel("epoch")
            plt.ylabel("weighted_loss_component")
            plt.title("Weighted loss components")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "train_loss_weighted_components.png", dpi=150)
        plt.close()
    except Exception:
        pass


def _float_metrics_from_aggregate(agg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in agg.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def _evaluate_dev_stage1_style(
    *,
    model: VGGT,
    data_root: Path,
    out_dir: Path,
    max_scenes: int,
    max_frames: int,
    depth_conf_thresh: float,
    chamfer_max_dist: float,
    eval_amp_dtype: torch.dtype,
    write_json: bool = True,
) -> Dict[str, float]:
    scenes = discover_scannet_scenes(data_root)
    if max_scenes > 0:
        scenes = scenes[:max_scenes]
    if not scenes:
        raise RuntimeError(f"No scenes found under {data_root}")
    eval_rows: List[Dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    t_eval0 = time.perf_counter()
    try:
        for scene in tqdm(scenes, desc="eval dev", leave=False):
            scene_eff = select_scene_image_paths_fastvggt(scene, max_frames)
            if not scene_eff.image_paths:
                continue
            images = load_scene_images(scene_eff, mode="crop")
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.to(next(model.parameters()).device)
            use_amp = images.device.type == "cuda"
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=eval_amp_dtype, enabled=use_amp):
                batch = collect_batch(model, images, keep_activations_on_gpu=False, collect_full_cache=False)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            pred_payload = build_scene_prediction_payload(
                batch.predictions,
                (int(images.shape[-2]), int(images.shape[-1])),
            )
            ctx = build_fastvggt_scene_eval_context(
                scene_eff,
                data_root,
                data_root,
                image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            )
            eval_rows.append(
                build_scannet_eval_row_from_payload_with_context(
                    pred_payload,
                    scene_eff,
                    ctx,
                    depth_conf_thresh=depth_conf_thresh,
                    chamfer_max_dist=chamfer_max_dist,
                    inference_time_ms=infer_ms,
                )
            )
    finally:
        if was_training:
            model.train()
    if not eval_rows:
        raise RuntimeError("No dev evaluation rows produced.")
    aggregate = aggregate_metric_dict(
        [{k: v for k, v in row.items() if k != "scene_id" and isinstance(v, (int, float))} for row in eval_rows]
    )
    if write_json:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_eval_scannet_json(
            out_dir,
            eval_rows,
            extra_top_level={
                "stage": "lora_skip_baseline_eval_dev",
                "elapsed_seconds": float(time.perf_counter() - t_eval0),
            },
        )
    return _float_metrics_from_aggregate(aggregate)


def _evaluate_test_run_eval_style(
    *,
    model: VGGT,
    data_dir: Path,
    gt_ply_dir: Path,
    out_dir: Path,
    input_frame: int,
    num_scenes: int,
    depth_conf_thresh: float,
    chamfer_max_dist: float,
    eval_amp_dtype: torch.dtype,
) -> Dict[str, float]:
    scene_names = get_all_scenes(data_dir, num_scenes) if num_scenes > 0 else [
        line.strip()
        for line in (Path(__file__).resolve().parents[1] / "eval" / "scannet_50.yaml").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_scenes_metrics: Dict[str, Dict[str, float]] = {}
    was_training = bool(model.training)
    model.eval()
    try:
        for scene in tqdm(scene_names, desc="eval test", leave=False):
            scene_dir = data_dir / f"{scene}"
            images_dir = scene_dir / "color"
            pose_path = scene_dir / "pose"
            image_paths = get_sorted_image_paths(images_dir)
            poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
            selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
                image_paths,
                available_pose_frame_ids,
                input_frame,
            )
            c2ws = poses_gt[selected_pose_indices]
            if len(selected_image_paths) == 0:
                continue
            images = load_images_rgb(selected_image_paths)
            if not images or len(images) < 3:
                continue
            images_array = np.stack(images)
            vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
            if hasattr(model, "update_patch_dimensions"):
                model.update_patch_dimensions(patch_width, patch_height)
            # Reuse run_eval/eval_scannet.py reconstruction + dense-depth GT path (for parity with average_metrics.json).
            _, _, all_world_points, _, all_cam_to_world_mat, inference_time_ms, depth_raw_np = (
                infer_vggt_and_reconstruct(
                    model,
                    vgg_input,
                    eval_amp_dtype,
                    depth_conf_thresh,
                    selected_image_paths,
                )
            )
            gt_depth = None
            pred_depth_for_metrics = None
            try:
                pred_depth_for_metrics = _depth_or_conf_to_shw(depth_raw_np, "depth")
            except Exception as e:
                print(f"[warn] pred depth layout normalize failed for {scene}: {e}")
                pred_depth_for_metrics = np.asarray(depth_raw_np, dtype=np.float32)
            try:
                scene_data = SceneData(
                    scene_id=scene,
                    scene_dir=scene_dir,
                    image_paths=list(selected_image_paths),
                )
                ctx = build_fastvggt_scene_eval_context(
                    scene_data,
                    data_dir,
                    gt_ply_dir,
                    image_hw=(int(vgg_input.shape[2]), int(vgg_input.shape[3])),
                )
                gt_depth = ctx.gt_depth
            except Exception as e:
                print(
                    f"[warn] depth GT load skipped for {scene} (align with eval_scannet when data is complete): {e}"
                )
            with tempfile.TemporaryDirectory(prefix="lora_eval_scannet_") as tmpd:
                metrics = evaluate_scene_and_save(
                    scene,
                    c2ws,
                    first_gt_pose,
                    selected_frame_ids,
                    all_cam_to_world_mat,
                    all_world_points,
                    Path(tmpd),
                    gt_ply_dir,
                    chamfer_max_dist,
                    inference_time_ms,
                    False,
                    pred_depth=pred_depth_for_metrics,
                    gt_depth=gt_depth,
                )
            if metrics is not None:
                all_scenes_metrics[scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key
                    in (
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                        "depth_abs_rel",
                        "depth_delta1",
                    )
                }
    finally:
        if was_training:
            model.train()
    if not all_scenes_metrics:
        raise RuntimeError("No test evaluation rows produced.")
    keys = sorted({k for row in all_scenes_metrics.values() for k in row.keys()})
    avg: Dict[str, float] = {}
    for k in keys:
        vals = [float(row[k]) for row in all_scenes_metrics.values() if k in row]
        if vals:
            avg[k] = float(sum(vals) / len(vals))
    return avg


def _run_stage1_upper_sanity_check(
    *,
    dev_upper_metrics: Dict[str, float],
    cache_dir: Path,
    rtol: float,
    atol: float,
) -> None:
    path = cache_dir / "eval_scannet.json"
    if not path.exists():
        raise SystemExit(f"Stage1 sanity check failed: missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = _float_metrics_from_aggregate(payload.get("aggregate", {}))
    keys = ("ate", "are", "rpe_rot", "rpe_trans", "chamfer_distance", "depth_abs_rel", "depth_delta1")
    missing = [k for k in keys if k not in dev_upper_metrics or k not in ref]
    if missing:
        raise SystemExit(f"Stage1 sanity check failed: missing metrics {missing}")
    mismatch: List[str] = []
    for k in keys:
        a = float(dev_upper_metrics[k])
        b = float(ref[k])
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            mismatch.append(f"{k}: upper={a:.8f} stage1={b:.8f}")
    if mismatch:
        msg = "\n".join(mismatch)
        raise SystemExit(f"Stage1 sanity check failed (upper(dev) != stage1 eval):\n{msg}")


def _run_test_upper_sanity_check(
    *,
    test_upper_metrics: Dict[str, float],
    ref_average_metrics_path: Path,
    rtol: float,
    atol: float,
) -> None:
    if not ref_average_metrics_path.exists():
        raise SystemExit(f"Test sanity check failed: missing {ref_average_metrics_path}")
    ref_payload = json.loads(ref_average_metrics_path.read_text(encoding="utf-8"))
    ref = _float_metrics_from_aggregate(ref_payload)
    # Keep sanity focused on quality metrics; inference latency is environment-sensitive.
    keys = ("ate", "are", "rpe_rot", "rpe_trans", "chamfer_distance", "depth_abs_rel", "depth_delta1")
    missing = [k for k in keys if k not in test_upper_metrics or k not in ref]
    if missing:
        raise SystemExit(f"Test sanity check failed: missing metrics {missing}")
    mismatch: List[str] = []
    for k in keys:
        a = float(test_upper_metrics[k])
        b = float(ref[k])
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            mismatch.append(f"{k}: upper={a:.8f} run_eval={b:.8f}")
    if mismatch:
        msg = "\n".join(mismatch)
        raise SystemExit(f"Test sanity check failed (upper(test) != run_eval average):\n{msg}")


def _append_eval_history(
    history_rows: List[Dict[str, Any]],
    *,
    epoch: int,
    split: str,
    tag: str,
    metrics: Dict[str, float],
) -> None:
    for k, v in metrics.items():
        history_rows.append({"epoch": int(epoch), "split": str(split), "tag": str(tag), "metric": str(k), "value": float(v)})


def _eval_epochs_metrics_payload(history_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One nested summary for periodic eval: epoch → dev/test metric dicts (for eval_epochs_metrics.json)."""
    ep_dev: Dict[int, Dict[str, float]] = {}
    ep_test: Dict[int, Dict[str, float]] = {}
    for r in history_rows:
        if str(r.get("tag", "")) != "train":
            continue
        split = str(r.get("split", ""))
        ep = int(r["epoch"])
        key = str(r["metric"])
        val = float(r["value"])
        if split == "dev":
            ep_dev.setdefault(ep, {})[key] = val
        elif split == "test":
            ep_test.setdefault(ep, {})[key] = val
    epochs_sorted = sorted(set(ep_dev.keys()) | set(ep_test.keys()))
    epochs_out: List[Dict[str, Any]] = []
    for ep in epochs_sorted:
        row: Dict[str, Any] = {"epoch": ep}
        if ep in ep_dev:
            row["dev"] = {k: ep_dev[ep][k] for k in sorted(ep_dev[ep].keys())}
        if ep in ep_test:
            row["test"] = {k: ep_test[ep][k] for k in sorted(ep_test[ep].keys())}
        epochs_out.append(row)
    return {"epochs": epochs_out}


def _write_eval_artifacts(out_dir: Path, bounds: Dict[str, Dict[str, float]], history_rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    (out_dir / "eval_epochs_metrics.json").write_text(
        json.dumps(_eval_epochs_metrics_payload(history_rows), indent=2),
        encoding="utf-8",
    )
    with (out_dir / "eval_metrics_history.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["epoch", "split", "tag", "metric", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in history_rows:
            w.writerow(r)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        train_rows = [r for r in history_rows if str(r.get("tag", "")) == "train"]
        metrics_with_dev = {str(r["metric"]) for r in train_rows if str(r.get("split", "")) == "dev"}
        metrics_with_test = {str(r["metric"]) for r in train_rows if str(r.get("split", "")) == "test"}
        # Only metrics that appear in periodic eval for both splits.
        candidate_metrics = sorted(metrics_with_dev & metrics_with_test)

        dev_color = "#c62828"  # red
        test_color = "#1565c0"  # blue

        for metric in candidate_metrics:
            dev_rows = [r for r in train_rows if str(r.get("split", "")) == "dev" and str(r.get("metric", "")) == metric]
            test_rows = [r for r in train_rows if str(r.get("split", "")) == "test" and str(r.get("metric", "")) == metric]
            dev_ub = bounds.get("dev", {}).get("upper", {}).get(metric)
            dev_lb = bounds.get("dev", {}).get("lower", {}).get(metric)
            test_ub = bounds.get("test", {}).get("upper", {}).get(metric)
            test_lb = bounds.get("test", {}).get("lower", {}).get(metric)
            # One figure = dev upper/lower/plot + test upper/lower/plot (six series); skip otherwise.
            if not dev_rows or not test_rows:
                continue
            if dev_ub is None or dev_lb is None or test_ub is None or test_lb is None:
                continue

            plt.figure(figsize=(10, 6))
            z_band = 1
            z_curve = 3

            plt.axhline(
                float(dev_ub),
                linestyle="-.",
                color=dev_color,
                linewidth=1.6,
                label="dev_upper",
                zorder=z_band,
                alpha=0.95,
            )
            plt.axhline(
                float(dev_lb),
                linestyle=":",
                color=dev_color,
                linewidth=1.6,
                label="dev_lower",
                zorder=z_band,
                alpha=0.95,
            )
            plt.axhline(
                float(test_ub),
                linestyle="-.",
                color=test_color,
                linewidth=1.6,
                label="test_upper",
                zorder=z_band,
                alpha=0.95,
            )
            plt.axhline(
                float(test_lb),
                linestyle=":",
                color=test_color,
                linewidth=1.6,
                label="test_lower",
                zorder=z_band,
                alpha=0.95,
            )

            dev_xs = [int(r["epoch"]) for r in dev_rows]
            dev_ys = [float(r["value"]) for r in dev_rows]
            test_xs = [int(r["epoch"]) for r in test_rows]
            test_ys = [float(r["value"]) for r in test_rows]
            plt.plot(
                dev_xs,
                dev_ys,
                marker="o",
                linestyle="-",
                color=dev_color,
                linewidth=2.0,
                label="dev",
                zorder=z_curve,
            )
            plt.plot(
                test_xs,
                test_ys,
                marker="s",
                linestyle="--",
                color=test_color,
                linewidth=2.0,
                label="test",
                zorder=z_curve,
            )

            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} (dev red vs test blue; upper/lower/plot per split)")
            plt.legend(loc="best", fontsize=9)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"eval_plot_{metric}.png", dpi=150)
            plt.close()
    except Exception:
        pass


def _fmt_metric_brief(metrics: Dict[str, float], keys: Sequence[str]) -> str:
    parts: List[str] = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.6f}")
    return " ".join(parts)


def _resolve_bounds_cache_paths(bounds_cache_file: Optional[Path], out_dir: Path) -> Tuple[Path, Path, Path]:
    combined_path = Path(bounds_cache_file) if bounds_cache_file is not None else (out_dir / "bounds_summary.json")
    suffix = combined_path.suffix if combined_path.suffix else ".json"
    upper_path = combined_path.with_name(f"{combined_path.stem}_upper{suffix}")
    lower_path = combined_path.with_name(f"{combined_path.stem}_lower{suffix}")
    return combined_path, upper_path, lower_path


def _extract_bound_payload(bounds: Dict[str, Dict[str, Dict[str, float]]], tag: Literal["upper", "lower"]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for split in ("dev", "test"):
        out[split] = dict(bounds.get(split, {}).get(tag, {}))
    return out


def _load_bound_payload(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid bounds payload (expect object): {path}")
    out: Dict[str, Dict[str, float]] = {}
    for split in ("dev", "test"):
        split_payload = payload.get(split, {})
        if split_payload is None:
            split_payload = {}
        if not isinstance(split_payload, dict):
            raise SystemExit(f"Invalid bounds payload split '{split}' (expect object): {path}")
        casted: Dict[str, float] = {}
        for k, v in split_payload.items():
            try:
                casted[str(k)] = float(v)
            except Exception:
                continue
        out[split] = casted
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train cache-first multi-interval skip+LoRA baseline.")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--interval-config-file", type=Path, required=True, help="JSON file path for interval rules.")
    p.add_argument(
        "--variant",
        type=str,
        default="multi_interval_config",
        choices=("multi_interval_config",),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes.")
    p.add_argument("--max-steps", type=int, default=0, help="0 means no early stop.")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--min-lr", type=float, default=1e-8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip-max-norm", type=float, default=1.0)
    p.add_argument("--save-every-epoch", type=int, default=10)
    p.add_argument("--resume-from", type=Path, default=None, help="Path to checkpoint to resume from.")
    p.add_argument(
        "--bounds-cache-file",
        type=Path,
        default=None,
        help=(
            "Base cache JSON path. Script now reads/writes split files "
            "('<stem>_upper.json' and '<stem>_lower.json'); "
            "legacy combined file is still used as fallback."
        ),
    )
    p.add_argument("--eval-epoch", type=int, default=0, help="Run dev/test eval every N epochs. 0 disables periodic eval.")
    p.add_argument("--dev-data-root", type=Path, default=Path("data/scans_dev50"))
    p.add_argument("--test-data-dir", type=Path, default=Path("data/scans_test50"))
    p.add_argument("--test-gt-ply-dir", type=Path, default=Path("data/scans_test50"))
    p.add_argument(
        "--test-sanity-average-metrics-file",
        type=Path,
        default=None,
        help="Path to run_eval ScanNet baseline average_metrics.json used for upper(test) sanity check.",
    )
    p.add_argument("--eval-max-scenes", type=int, default=0, help="0 means full split.")
    p.add_argument("--eval-input-frame", type=int, default=30)
    p.add_argument("--stage1-sanity-atol", type=float, default=1e-6)
    p.add_argument("--stage1-sanity-rtol", type=float, default=1e-6)
    p.add_argument(
        "--test-sanity-atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for upper(test) vs run_eval average sanity check.",
    )
    p.add_argument(
        "--test-sanity-rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for upper(test) vs run_eval average sanity check.",
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--w-align", type=float, default=1.0)
    p.add_argument("--w-align-mse", type=float, default=1.0)
    p.add_argument("--w-align-cos", type=float, default=1.0)
    p.add_argument("--w-pred", type=float, default=1.0)
    p.add_argument("--w-camera", type=float, default=1.0)
    p.add_argument("--w-depth", type=float, default=1.0)
    p.add_argument("--w-depth-grad", type=float, default=1.0)
    p.add_argument("--depth-valid-range", type=float, default=0.98)
    p.add_argument("--depth-gradient-loss-fn", type=str, default="grad")
    p.add_argument("--camera-loss-type", type=str, default="l1", choices=("l1", "l2"))
    p.add_argument("--target-source", type=str, default="teacher", choices=("teacher", "dataset"))
    p.add_argument(
        "--gt-data-root",
        type=Path,
        default=Path("data/scans_dev50"),
        help="ScanNet-style GT root for --target-source=dataset.",
    )
    p.add_argument("--w-reg", type=float, default=0.0)
    p.add_argument(
        "--use-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for replayed frame/global blocks.",
    )
    p.add_argument(
        "--print-step-timing",
        action="store_true",
        help="Print per-step timing breakdown to help identify bottlenecks.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[str(args.dtype)]
    use_amp = device.type == "cuda" and str(args.dtype) in ("float16", "bfloat16")
    test_sanity_avg_path = (
        Path(args.test_sanity_average_metrics_file)
        if args.test_sanity_average_metrics_file is not None
        else Path("analysis_outputs/evaluation/scans_test50/baseline") / f"input_frame_{int(args.eval_input_frame)}" / "average_metrics.json"
    )

    rules = _load_interval_rules(args.interval_config_file)
    intervals = _intervals_from_rules(rules)
    global_skip_intervals = _global_skip_intervals_from_rules(rules)
    frame_skip_intervals = _frame_skip_intervals_from_rules(rules)
    prev_layers = [int(x.prev_layer) for x in rules]
    if any(p < 0 for p in prev_layers):
        raise SystemExit(f"Interval starts must be >0: {intervals}")

    scenes = _load_manifest(args.cache_dir)
    if int(args.max_scenes) > 0:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise SystemExit("No scenes selected from cache.")

    num_layers = _infer_num_layers(args.cache_dir, scenes)
    if any(r >= num_layers for (_l, r) in intervals):
        raise SystemExit(f"Interval out of range for num_layers={num_layers}: {intervals}")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "cache_dir": str(args.cache_dir),
                "hf_model": args.hf_model,
                "intervals": intervals,
                "interval_rules": [vars(x) for x in rules],
                "variant": args.variant,
                "epochs": int(args.epochs),
                "max_scenes": int(args.max_scenes),
                "max_steps": int(args.max_steps),
                "save_every_epoch": int(args.save_every_epoch),
                "eval_epoch": int(args.eval_epoch),
                "eval_max_scenes": int(args.eval_max_scenes),
                "eval_input_frame": int(args.eval_input_frame),
                "dev_data_root": str(args.dev_data_root),
                "test_data_dir": str(args.test_data_dir),
                "test_gt_ply_dir": str(args.test_gt_ply_dir),
                "bounds_cache_file": str(args.bounds_cache_file) if args.bounds_cache_file else None,
                "test_sanity_average_metrics_file": (
                    str(args.test_sanity_average_metrics_file) if args.test_sanity_average_metrics_file else None
                ),
                "test_sanity_atol": float(args.test_sanity_atol),
                "test_sanity_rtol": float(args.test_sanity_rtol),
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "interval_config_file": str(args.interval_config_file),
                "device": str(device),
                "dtype": str(args.dtype),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_file = (out_dir / "train.log").open("a", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    skipped_global_idxs = _build_skipped_layer_indices(global_skip_intervals)
    skipped_frame_idxs = _build_skipped_layer_indices(frame_skip_intervals)
    skipped_idxs = sorted(set(skipped_global_idxs).union(skipped_frame_idxs))
    eval_dir = out_dir
    bounds_path, bounds_upper_path, bounds_lower_path = _resolve_bounds_cache_paths(args.bounds_cache_file, out_dir)
    eval_history_rows: List[Dict[str, Any]] = []
    bounds: Dict[str, Dict[str, Dict[str, float]]] = {"dev": {"upper": {}, "lower": {}}, "test": {"upper": {}, "lower": {}}}

    def _build_model_variant(*, prune: bool, inject_lora: bool) -> Tuple[VGGT, List[nn.Parameter], int, int]:
        m = VGGT.from_pretrained(args.hf_model).to(device).train()
        # We never train/evaluate point in this script path.
        m.point_head = None
        for pp in m.parameters():
            pp.requires_grad_(False)
        skipped_modules_before: List[nn.Module] = []
        for idx in skipped_frame_idxs:
            if idx < 0 or idx >= len(m.aggregator.frame_blocks):
                raise SystemExit(f"Invalid skipped frame index {idx} for model depth {len(m.aggregator.frame_blocks)}")
            skipped_modules_before.append(m.aggregator.frame_blocks[idx])
        for idx in skipped_global_idxs:
            if idx < 0 or idx >= len(m.aggregator.global_blocks):
                raise SystemExit(f"Invalid skipped global index {idx} for model depth {len(m.aggregator.global_blocks)}")
            skipped_modules_before.append(m.aggregator.global_blocks[idx])
        pre_total = sum(p.numel() for p in m.parameters())
        removed = _count_unique_params(skipped_modules_before) if prune else 0
        if prune:
            _replace_skipped_layers_with_identity(
                m,
                frame_intervals=frame_skip_intervals,
                global_intervals=global_skip_intervals,
            )
        trainable_local: List[nn.Parameter] = []
        if inject_lora:
            frame_lora_idxs, global_lora_idxs = _lora_injection_plan_from_rules(rules)
            for pidx in frame_lora_idxs:
                trainable_local.extend(
                    _inject_lora_attn(
                        m.aggregator.frame_blocks[pidx],
                        rank=int(args.lora_rank),
                        alpha=float(args.lora_alpha),
                        dropout=float(args.lora_dropout),
                    )
                )
            for pidx in global_lora_idxs:
                trainable_local.extend(
                    _inject_lora_attn(
                        m.aggregator.global_blocks[pidx],
                        rank=int(args.lora_rank),
                        alpha=float(args.lora_alpha),
                        dropout=float(args.lora_dropout),
                    )
                )
        return m, trainable_local, pre_total, removed

    has_upper = False
    has_lower = False
    if bounds_upper_path.exists():
        upper_payload = _load_bound_payload(bounds_upper_path)
        bounds["dev"]["upper"] = upper_payload.get("dev", {})
        bounds["test"]["upper"] = upper_payload.get("test", {})
        has_upper = True
        _log(f"[eval] reuse existing upper bound from {bounds_upper_path}")
    if bounds_lower_path.exists():
        lower_payload = _load_bound_payload(bounds_lower_path)
        bounds["dev"]["lower"] = lower_payload.get("dev", {})
        bounds["test"]["lower"] = lower_payload.get("test", {})
        has_lower = True
        _log(f"[eval] reuse existing lower bound from {bounds_lower_path}")
    if bounds_path.exists():
        legacy_bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
        for split in ("dev", "test"):
            split_payload = legacy_bounds.get(split, {}) if isinstance(legacy_bounds, dict) else {}
            if not isinstance(split_payload, dict):
                continue
            if not has_upper and isinstance(split_payload.get("upper"), dict):
                bounds["dev" if split == "dev" else "test"]["upper"] = {
                    str(k): float(v) for k, v in split_payload["upper"].items()
                }
            if not has_lower and isinstance(split_payload.get("lower"), dict):
                bounds["dev" if split == "dev" else "test"]["lower"] = {
                    str(k): float(v) for k, v in split_payload["lower"].items()
                }
        if not has_upper and bounds["dev"]["upper"] and bounds["test"]["upper"]:
            has_upper = True
            bounds_upper_path.parent.mkdir(parents=True, exist_ok=True)
            bounds_upper_path.write_text(json.dumps(_extract_bound_payload(bounds, "upper"), indent=2), encoding="utf-8")
            _log(f"[eval] hydrated upper bound cache from legacy file -> {bounds_upper_path}")
        if not has_lower and bounds["dev"]["lower"] and bounds["test"]["lower"]:
            has_lower = True
            bounds_lower_path.parent.mkdir(parents=True, exist_ok=True)
            bounds_lower_path.write_text(json.dumps(_extract_bound_payload(bounds, "lower"), indent=2), encoding="utf-8")
            _log(f"[eval] hydrated lower bound cache from legacy file -> {bounds_lower_path}")

    # Baseline bounds: upper=original model, lower=pruned model without LoRA.
    need_upper = not has_upper
    need_lower = not has_lower
    if need_upper or need_lower:
        missing = []
        if need_upper:
            missing.append("upper")
        if need_lower:
            missing.append("lower")
        _log(f"[eval] computing missing bounds on dev/test: {','.join(missing)}")
    else:
        _log(f"[eval] reuse existing split bounds from {bounds_upper_path} and {bounds_lower_path}")

    if need_upper:
        upper_model, _, _, _ = _build_model_variant(prune=False, inject_lora=False)
        try:
            t_upper_dev0 = time.perf_counter()
            upper_dev = _evaluate_dev_stage1_style(
                model=upper_model,
                data_root=args.dev_data_root,
                out_dir=eval_dir / "eval_upper_bound_dev",
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.eval_input_frame),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
                write_json=False,
            )
            _log(f"[eval][upper][scans_dev50] elapsed={time.perf_counter() - t_upper_dev0:.2f}s")
            _log(
                "[eval][upper][scans_dev50] "
                + _fmt_metric_brief(
                    upper_dev,
                    ("chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "depth_abs_rel", "depth_delta1"),
                )
            )
            _run_stage1_upper_sanity_check(
                dev_upper_metrics=upper_dev,
                cache_dir=args.cache_dir,
                rtol=float(args.stage1_sanity_rtol),
                atol=float(args.stage1_sanity_atol),
            )
            _log("[eval][upper][scans_dev50] stage1 sanity check: PASSED")
            t_upper_test0 = time.perf_counter()
            upper_test = _evaluate_test_run_eval_style(
                model=upper_model,
                data_dir=args.test_data_dir,
                gt_ply_dir=args.test_gt_ply_dir,
                out_dir=eval_dir / "eval_upper_bound_test",
                input_frame=int(args.eval_input_frame),
                num_scenes=int(args.eval_max_scenes),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
            )
            _log(f"[eval][upper][scans_test50] elapsed={time.perf_counter() - t_upper_test0:.2f}s")
            _run_test_upper_sanity_check(
                test_upper_metrics=upper_test,
                ref_average_metrics_path=test_sanity_avg_path,
                rtol=float(args.test_sanity_rtol),
                atol=float(args.test_sanity_atol),
            )
            _log(
                f"[eval][upper][scans_test50] run_eval sanity check: PASSED "
                f"(ref={test_sanity_avg_path})"
            )
            _log(
                "[eval][upper][scans_test50] "
                + _fmt_metric_brief(upper_test, ("chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"))
            )
        finally:
            del upper_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        bounds["dev"]["upper"] = upper_dev
        bounds["test"]["upper"] = upper_test
        bounds_upper_path.parent.mkdir(parents=True, exist_ok=True)
        bounds_upper_path.write_text(json.dumps(_extract_bound_payload(bounds, "upper"), indent=2), encoding="utf-8")
        _log(f"[eval] wrote upper bound cache to {bounds_upper_path}")

    if need_lower:
        lower_model, _, _, _ = _build_model_variant(prune=True, inject_lora=False)
        try:
            t_lower_dev0 = time.perf_counter()
            lower_dev = _evaluate_dev_stage1_style(
                model=lower_model,
                data_root=args.dev_data_root,
                out_dir=eval_dir / "eval_lower_bound_dev",
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.eval_input_frame),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
                write_json=False,
            )
            _log(f"[eval][lower][scans_dev50] elapsed={time.perf_counter() - t_lower_dev0:.2f}s")
            _log(
                "[eval][lower][scans_dev50] "
                + _fmt_metric_brief(
                    lower_dev,
                    ("chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "depth_abs_rel", "depth_delta1"),
                )
            )
            t_lower_test0 = time.perf_counter()
            lower_test = _evaluate_test_run_eval_style(
                model=lower_model,
                data_dir=args.test_data_dir,
                gt_ply_dir=args.test_gt_ply_dir,
                out_dir=eval_dir / "eval_lower_bound_test",
                input_frame=int(args.eval_input_frame),
                num_scenes=int(args.eval_max_scenes),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
            )
            _log(f"[eval][lower][scans_test50] elapsed={time.perf_counter() - t_lower_test0:.2f}s")
            _log(
                "[eval][lower][scans_test50] "
                + _fmt_metric_brief(lower_test, ("chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"))
            )
        finally:
            del lower_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        bounds["dev"]["lower"] = lower_dev
        bounds["test"]["lower"] = lower_test
        bounds_lower_path.parent.mkdir(parents=True, exist_ok=True)
        bounds_lower_path.write_text(json.dumps(_extract_bound_payload(bounds, "lower"), indent=2), encoding="utf-8")
        _log(f"[eval] wrote lower bound cache to {bounds_lower_path}")

    if need_upper or need_lower:
        bounds_path.parent.mkdir(parents=True, exist_ok=True)
        bounds_path.write_text(json.dumps(bounds, indent=2), encoding="utf-8")
        _write_eval_artifacts(out_dir, bounds, eval_history_rows)

    model, trainable, pre_prune_total_params, removed_params = _build_model_variant(prune=True, inject_lora=True)
    if not trainable:
        raise SystemExit("No LoRA parameters were injected; nothing to train.")
    gt_scene_map: Dict[str, SceneData] = {}
    if str(args.target_source) == "dataset":
        gt_scenes = discover_scannet_scenes(Path(args.gt_data_root))
        gt_scene_map = {str(s.scene_id): s for s in gt_scenes}
        if not gt_scene_map:
            raise SystemExit(f"No GT scenes discovered under --gt-data-root={args.gt_data_root}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    post_vs_pre_ratio = (100.0 * total_params / pre_prune_total_params) if pre_prune_total_params > 0 else 0.0
    frozen_ratio = (100.0 * frozen_params / total_params) if total_params > 0 else 0.0
    train_ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    _log(
        f"[params] pre_prune={_format_param_count(pre_prune_total_params)} "
        f"post_prune={_format_param_count(total_params)} ({post_vs_pre_ratio:.2f}% of pre-prune) "
        f"frozen={_format_param_count(frozen_params)} ({frozen_ratio:.2f}% of post-prune) "
        f"trainable={_format_param_count(trainable_params)} ({train_ratio:.2f}% of post-prune) "
        f"skipped {len(skipped_idxs)} layers"
    )

    optim = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rows: List[Dict[str, Any]] = []
    global_step = 0
    start_epoch = 0

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise SystemExit(f"--resume-from not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scaler_state = ckpt.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        _log(f"[resume] from={ckpt_path} start_epoch={start_epoch} global_step={global_step}")
    t0_train = time.perf_counter()
    remaining_epochs = max(0, int(args.epochs) - int(start_epoch))
    total_steps = _compute_total_steps(len(scenes), remaining_epochs, int(args.max_steps))
    pbar = tqdm(total=total_steps if total_steps > 0 else None, desc="train", dynamic_ncols=True)
    def _save_checkpoint(ckpt_path: Path, epoch_idx: int) -> None:
        torch.save(
            {
                "global_step": global_step,
                "epoch": int(epoch_idx),
                "epochs": int(args.epochs),
                "intervals": intervals,
                "interval_rules": [vars(x) for x in rules],
                "variant": args.variant,
                "save_every_epoch": int(args.save_every_epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

    for epoch in range(start_epoch, int(args.epochs)):
        epoch_scenes = list(scenes)
        random.Random(int(args.seed) + epoch).shuffle(epoch_scenes)
        for si, item in enumerate(epoch_scenes):
            t_step0 = time.perf_counter()
            sid = str(item.get("scene_id", ""))
            t_load0 = time.perf_counter()
            row = _load_scene_cache_row(args.cache_dir, item)
            _validate_schema(row, num_layers=num_layers)
            teacher_pred = _teacher_pred_to_map(row["pred"])
            layer_cpu_cache: Dict[int, torch.Tensor] = {}

            def _get_layer_cpu(idx: int) -> torch.Tensor:
                cached = layer_cpu_cache.get(idx)
                if cached is not None:
                    return cached
                t = load_scene_replay_layer(
                    args.cache_dir,
                    item,
                    int(idx),
                    expected_num_layers=num_layers,
                )
                layer_cpu_cache[idx] = t
                return t

            hw = row.get("image_hw", [518, 518])
            h = int(hw[0])
            w = int(hw[1])
            seq_len = int(row.get("num_frames", _get_layer_cpu(0).shape[1]))
            dummy_images = torch.zeros((1, seq_len, 3, h, w), device=device, dtype=torch.float32)
            t_load = time.perf_counter() - t_load0

            t_replay = 0.0
            t_align = 0.0
            t_pred = 0.0
            t_backward = 0.0
            t_optim = 0.0
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                t_replay0 = time.perf_counter()
                replay = _replay_student_cache_first(
                    model,
                    get_layer_cpu=_get_layer_cpu,
                    num_layers=int(num_layers),
                    frame_skip_intervals=frame_skip_intervals,
                    global_skip_intervals=global_skip_intervals,
                    lora_anchor_prev_layers=prev_layers,
                    use_checkpoint=bool(args.use_checkpoint),
                    device=device,
                    dummy_images=dummy_images,
                )
                t_replay = time.perf_counter() - t_replay0

                loss_align = torch.zeros((), device=device)
                align_mse = torch.zeros((), device=device)
                align_cos = torch.zeros((), device=device)
                n_terms = 0
                t_align0 = time.perf_counter()
                for rule in rules:
                    pidx = int(rule.prev_layer)
                    s_concat = replay.concat_by_layer[pidx]
                    t_concat = _get_layer_cpu(int(rule.r)).to(device=device, non_blocking=(device.type == "cuda"))
                    sf, sg = _split_concat(s_concat)
                    tf, tg = _split_concat(t_concat)
                    mg, cg = _loss_mse_cos(sg, tg)
                    mse_i = mg
                    cos_i = cg
                    if bool(rule.has_frame_lora):
                        mf, cf = _loss_mse_cos(sf, tf)
                        mse_i = mse_i + mf
                        cos_i = cos_i + cf
                    loss_align = loss_align + float(args.w_align_mse) * mse_i + float(args.w_align_cos) * cos_i
                    align_mse = align_mse + mse_i
                    align_cos = align_cos + cos_i
                    n_terms += 1
                if n_terms > 0:
                    loss_align = loss_align / n_terms
                    align_mse = align_mse / n_terms
                    align_cos = align_cos / n_terms
                t_align = time.perf_counter() - t_align0
                loss_align_mse_term = float(args.w_align) * float(args.w_align_mse) * align_mse
                loss_align_cos_term = float(args.w_align) * float(args.w_align_cos) * align_cos

                t_pred0 = time.perf_counter()
                if str(args.target_source) == "teacher":
                    target_batch = _build_target_batch_from_teacher(
                        teacher_pred,
                        images_batched=dummy_images,
                        device=device,
                    )
                else:
                    if sid not in gt_scene_map:
                        raise RuntimeError(
                            f"Scene {sid} not found under --gt-data-root={args.gt_data_root} "
                            f"for --target-source=dataset"
                        )
                    target_batch = _build_target_batch_from_dataset(
                        scene=gt_scene_map[sid],
                        expected_seq_len=int(seq_len),
                        device=device,
                    )
                loss_pred, pred_logs = _prediction_official_camera_depth(
                    replay.pred,
                    target_batch,
                    w_camera=float(args.w_camera),
                    w_depth=float(args.w_depth),
                    w_depth_grad=float(args.w_depth_grad),
                    camera_loss_type=str(args.camera_loss_type),
                    depth_valid_range=float(args.depth_valid_range),
                    depth_gradient_loss_fn=str(args.depth_gradient_loss_fn),
                )
                t_pred = time.perf_counter() - t_pred0

                loss_reg = torch.zeros((), device=device)
                if float(args.w_reg) > 0:
                    for p in trainable:
                        loss_reg = loss_reg + p.float().pow(2).mean()

                loss_total = (
                    float(args.w_align) * loss_align
                    + float(args.w_pred) * loss_pred
                    + float(args.w_reg) * loss_reg
                )

            optim.zero_grad(set_to_none=True)
            t_bw0 = time.perf_counter()
            scaler.scale(loss_total).backward()
            t_backward = time.perf_counter() - t_bw0
            t_opt0 = time.perf_counter()
            where = float(global_step) / max(float(total_steps), 1.0)
            lr_now = _lr_with_warmup_cosine(
                where=where,
                base_lr=float(args.lr),
                min_lr=float(args.min_lr),
                warmup_ratio=float(args.warmup_ratio),
            )
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            grad_norm = torch.tensor(0.0, device=device)
            if float(args.grad_clip_max_norm) > 0:
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.grad_clip_max_norm))
            scaler.step(optim)
            scaler.update()
            t_optim = time.perf_counter() - t_opt0

            global_step += 1
            step_s = time.perf_counter() - t_step0
            log_row: Dict[str, Any] = {
                "epoch": epoch,
                "scene_idx": si,
                "scene_id": sid,
                "global_step": global_step,
                "loss_total": float(loss_total.detach().item()),
                "loss_align": float(loss_align.detach().item()),
                "loss_align_mse": float(align_mse.detach().item()),
                "loss_align_cos": float(align_cos.detach().item()),
                "loss_align_mse_term": float(loss_align_mse_term.detach().item()),
                "loss_align_cos_term": float(loss_align_cos_term.detach().item()),
                "loss_pred": float(loss_pred.detach().item()),
                "loss_pred_term": float((float(args.w_pred) * loss_pred).detach().item()),
                "loss_reg": float(loss_reg.detach().item()),
                "loss_reg_term": float((float(args.w_reg) * loss_reg).detach().item()),
                "step_seconds": step_s,
                "time_load_s": t_load,
                "time_replay_s": t_replay,
                "time_align_s": t_align,
                "time_pred_s": t_pred,
                "time_backward_s": t_backward,
                "time_optim_s": t_optim,
                "optim_lr": float(lr_now),
                "grad_norm": float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            }
            log_row.update(pred_logs)
            for k, v in pred_logs.items():
                if k.endswith("_term"):
                    log_row[f"{k}_contrib"] = float(args.w_pred) * float(v)
            rows.append(log_row)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "epoch": f"{epoch+1}/{int(args.epochs)}",
                    "scene": sid,
                    "loss": f"{log_row['loss_total']:.2f}",
                    "align_loss": f"{log_row['loss_align']:.2f}",
                    "prd_loss": f"{log_row['loss_pred']:.2f}",
                    # "step_s": f"{step_s:.1f}",
                    "rep": f"{t_replay:.1f}s",
                    "align": f"{t_align:.1f}s",
                    "prd": f"{t_pred:.1f}s",
                    "bw": f"{t_backward:.1f}s",
                    "opt": f"{t_optim:.1f}s",
                },
                refresh=False,
            )
            if bool(args.print_step_timing):
                _log(
                    f"[timing] step={global_step} load={t_load:.2f}s replay={t_replay:.2f}s "
                    f"align={t_align:.2f}s pred={t_pred:.2f}s backward={t_backward:.2f}s "
                    f"optim={t_optim:.2f}s total={step_s:.2f}s"
                )

            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                break
        if int(args.save_every_epoch) > 0 and (((epoch + 1) % int(args.save_every_epoch)) == 0):
            _save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch+1}.pt", epoch)
        _save_checkpoint(out_dir / "checkpoints" / "latest.pt", epoch)
        _write_curves(out_dir, rows)
        if int(args.eval_epoch) > 0 and (((epoch + 1) % int(args.eval_epoch)) == 0):
            _log(f"[eval] epoch={epoch+1}: evaluating current LoRA model on dev/test")
            t_eval_dev0 = time.perf_counter()
            dev_metrics = _evaluate_dev_stage1_style(
                model=model,
                data_root=args.dev_data_root,
                out_dir=out_dir,
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.eval_input_frame),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
                write_json=False,
            )
            _log(f"[eval][epoch={epoch+1}][scans_dev50] elapsed={time.perf_counter() - t_eval_dev0:.2f}s")
            t_eval_test0 = time.perf_counter()
            test_metrics = _evaluate_test_run_eval_style(
                model=model,
                data_dir=args.test_data_dir,
                gt_ply_dir=args.test_gt_ply_dir,
                out_dir=out_dir,
                input_frame=int(args.eval_input_frame),
                num_scenes=int(args.eval_max_scenes),
                depth_conf_thresh=1.0,
                chamfer_max_dist=0.5,
                eval_amp_dtype=amp_dtype,
            )
            _log(f"[eval][epoch={epoch+1}][scans_test50] elapsed={time.perf_counter() - t_eval_test0:.2f}s")
            _append_eval_history(eval_history_rows, epoch=epoch + 1, split="dev", tag="train", metrics=dev_metrics)
            _append_eval_history(eval_history_rows, epoch=epoch + 1, split="test", tag="train", metrics=test_metrics)
            _write_eval_artifacts(out_dir, bounds, eval_history_rows)
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break
    pbar.close()

    if int(args.epochs) == 0 or start_epoch >= int(args.epochs):
        _save_checkpoint(out_dir / "checkpoints" / "latest.pt", max(0, start_epoch - 1))
    best = min(rows, key=lambda x: float(x.get("loss_total", float("inf")))) if rows else {}
    summary = {
        "global_steps": global_step,
        "elapsed_seconds": time.perf_counter() - t0_train,
        "best_train_loss_total": best.get("loss_total"),
        "best_train_step": best.get("global_step"),
        "num_scenes": len(scenes),
        "intervals": intervals,
        "interval_rules": [vars(x) for x in rules],
        "variant": args.variant,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_curves(out_dir, rows)
    _log(f"[done] out_dir={out_dir} steps={global_step}")
    log_file.close()


if __name__ == "__main__":
    main()


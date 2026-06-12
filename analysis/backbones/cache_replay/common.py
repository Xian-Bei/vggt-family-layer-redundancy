"""Shared cache-replay training utilities (backbone-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from analysis.redundancy.metrics import linear_cka_value_tensor, sample_cosine_similarity_tensor


@dataclass
class ReplayResult:
    concat_by_layer: Dict[int, torch.Tensor]
    pred: Dict[str, torch.Tensor]


def split_concat(layer_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer_t.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(layer_t.shape)}")
    two_c = int(layer_t.shape[-1])
    if two_c % 2 != 0:
        raise ValueError(f"Expected even channel dimension, got {two_c}")
    c = two_c // 2
    return layer_t[..., :c], layer_t[..., c:]


def is_skipped(layer_idx: int, intervals: Sequence[Tuple[int, int]]) -> bool:
    for l, r in intervals:
        if l <= layer_idx <= r:
            return True
    return False


def loss_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b)


def loss_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - sample_cosine_similarity_tensor(a, b, eps=eps)


def loss_cka(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - linear_cka_value_tensor(a, b, eps=eps)


def teacher_pred_to_map(pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            t = v.contiguous()
            if t.dim() >= 1 and t.shape[0] != 1:
                t = t.unsqueeze(0)
            out[k] = t
    return out


def infer_num_layers(cache_dir: Path, scenes: Sequence[Dict[str, Any]]) -> int:
    from analysis.redundancy.cache_layout import _replay_layer_files
    from analysis.redundancy.cache_layout_v3 import is_v3_cache_dir, load_v3_manifest

    if is_v3_cache_dir(cache_dir):
        manifest = load_v3_manifest(cache_dir)
        if not manifest.layer_units:
            raise SystemExit("v3 manifest has no layer_units")
        return max(u.index for u in manifest.layer_units if u.kind in ("frame", "global")) + 1
    for item in scenes:
        scene_dir_name = str(item.get("dir", "")).strip() or str(item.get("scene_id", "")).strip()
        if not scene_dir_name:
            continue
        layers_dir = cache_dir / scene_dir_name / "layers"
        if not layers_dir.exists():
            continue
        layer_files = _replay_layer_files(layers_dir)
        if layer_files:
            return len(layer_files)
    raise SystemExit(f"Cannot infer num_layers from cache at {cache_dir}")


def align_activation_term(
    student_term: torch.Tensor,
    teacher_term: torch.Tensor,
    *,
    w_mse: float,
    w_cos: float,
    w_cka: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_i = torch.zeros((), device=student_term.device)
    cos_i = torch.zeros((), device=student_term.device)
    cka_i = torch.zeros((), device=student_term.device)
    if w_mse != 0.0:
        mse_i = mse_i + loss_mse(student_term, teacher_term)
    if w_cos != 0.0:
        cos_i = cos_i + loss_cos(student_term, teacher_term)
    if w_cka != 0.0:
        cka_i = cka_i + loss_cka(student_term, teacher_term)
    return mse_i, cos_i, cka_i, w_mse * mse_i + w_cos * cos_i + w_cka * cka_i

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from .metrics import pairwise_similarity_matrices, spectral_stats


def flatten_global_block_activations_for_cache(
    t: torch.Tensor,
    max_tokens: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Flatten ``[B, S, P, C]`` activations to ``[N, C]`` float16 **CPU** for disk cache.

    Works if ``t`` is on CUDA (subsample on device, one transfer) or CPU.

    If ``max_tokens <= 0``, **all** token rows are kept (no subsampling).
    If ``max_tokens > 0`` and ``N > max_tokens``, randomly keep ``max_tokens`` rows
    (reproducible for a given ``seed``).
    """
    flat = t.reshape(-1, t.shape[-1]).float()
    n = flat.shape[0]
    if max_tokens <= 0 or n <= max_tokens:
        return flat.half().cpu()
    flat_cpu = flat.cpu()
    g = torch.Generator()
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    idx = torch.randperm(n, generator=g)[:max_tokens]
    return flat_cpu[idx].half()


def infer_bs_and_seq_len(images: torch.Tensor) -> tuple[int, int]:
    """
    Match ``VGGT.forward`` batch handling so hooks reshape like the aggregator.

    - ``[S, 3, H, W]`` (no batch dim) → ``B=1``, ``S=S`` (forward prepends a batch dim).
    - ``[B, S, 3, H, W]`` → ``B``, ``S``.
    """
    if images.dim() == 4:
        return 1, int(images.shape[0])
    if images.dim() == 5:
        return int(images.shape[0]), int(images.shape[1])
    raise ValueError(f"Expected images with 4 or 5 dimensions, got shape {tuple(images.shape)}")


@dataclass
class RedundancyBatchResult:
    frame_tokens: List[torch.Tensor]
    global_tokens: List[torch.Tensor]
    aggregated_concat_tokens: List[torch.Tensor]
    global_patch_tokens: List[torch.Tensor]
    global_special_tokens: List[torch.Tensor]
    patch_start_idx: int
    scene_scores: Dict[str, float]
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)


def _required_prediction_layers(model: VGGT, num_layers: int) -> List[int]:
    required: set[int] = set()
    if model.camera_head is not None and num_layers > 0:
        required.add(num_layers - 1)
    if model.depth_head is not None:
        required.update(int(i) for i in getattr(model.depth_head, "intermediate_layer_idx", []))
    if model.point_head is not None:
        required.update(int(i) for i in getattr(model.point_head, "intermediate_layer_idx", []))
    return sorted(i for i in required if 0 <= i < num_layers)


def _sparse_layer_list(num_layers: int, selected: Dict[int, torch.Tensor], *, require_last: bool) -> List[torch.Tensor]:
    if num_layers <= 0:
        raise RuntimeError(f"Invalid num_layers={num_layers}")
    out: List[torch.Tensor] = [None] * num_layers  # type: ignore[list-item]
    for idx, tensor in selected.items():
        out[idx] = tensor
    if require_last and out[-1] is None:
        raise RuntimeError("Sparse layer list is missing final layer needed by CameraHead.")
    return out


def _attention_weight_stats(attn_module: torch.nn.Module) -> Dict[str, float]:
    with torch.no_grad():
        qkv_w = attn_module.qkv.weight.detach().cpu()
        dim = attn_module.qkv.in_features
        heads = attn_module.num_heads
        head_dim = dim // heads
        # Keep only Q projection for head redundancy.
        q_w = qkv_w[:dim].reshape(heads, head_dim, dim)
        pairwise = []
        for i in range(heads):
            for j in range(i + 1, heads):
                v1 = q_w[i].reshape(-1)
                v2 = q_w[j].reshape(-1)
                pairwise.append(float(F.cosine_similarity(v1, v2, dim=0).item()))
        mean_head_cosine = float(sum(pairwise) / max(len(pairwise), 1))
    return {"mean_head_cosine": mean_head_cosine}


def compute_global_weight_stats(model: VGGT) -> Dict[str, Any]:
    """Spectral / multi-head statistics over ``model.aggregator.global_blocks`` (checkpoint only, no data)."""
    global_attn: List[Dict[str, Any]] = []
    spectral: List[Dict[str, Any]] = []
    for idx, g_blk in enumerate(model.aggregator.global_blocks):
        global_attn.append({"layer": idx, **_attention_weight_stats(g_blk.attn)})
        spectral.append(
            {"layer": idx, "module": "global_attn_qkv", **spectral_stats(g_blk.attn.qkv.weight.detach().cpu())}
        )
        spectral.append(
            {"layer": idx, "module": "global_mlp_fc1", **spectral_stats(g_blk.mlp.fc1.weight.detach().cpu())}
        )
    return {"attention_stats": {"global": global_attn}, "spectral_stats": spectral}


def predictions_to_cpu_float(preds: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in preds.items():
        if k == "pose_enc_list":
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().float().cpu()
    return out


def collect_batch(
    model: VGGT,
    images: torch.Tensor,
    *,
    keep_activations_on_gpu: bool = False,
    collect_full_cache: bool = True,
) -> RedundancyBatchResult:
    if images.dim() == 4:
        images = images.unsqueeze(0)
    elif images.dim() != 5:
        raise ValueError(f"Expected images with shape [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")

    num_layers = len(model.aggregator.global_blocks)
    required_layers = _required_prediction_layers(model, num_layers)
    selected_layer_set = set(required_layers)
    selected_layer_tensors: Dict[int, torch.Tensor] = {}
    aggregated_concat_tokens_cpu: Optional[List[Optional[torch.Tensor]]] = (
        [None] * num_layers if collect_full_cache else None
    )

    def _layer_callback(layer_idx: int, concat_tokens: torch.Tensor) -> None:
        if not collect_full_cache:
            return
        t = concat_tokens.detach()
        assert aggregated_concat_tokens_cpu is not None
        aggregated_concat_tokens_cpu[layer_idx] = t.cpu().contiguous()
        if layer_idx in selected_layer_set:
            selected_layer_tensors[layer_idx] = t

    with torch.no_grad():
        selected_outputs, patch_start_idx = model.aggregator(
            images,
            return_layers=selected_layer_set,
            layer_callback=_layer_callback if collect_full_cache else None,
        )
        if len(selected_outputs) != len(required_layers):
            raise RuntimeError(
                f"Aggregator returned {len(selected_outputs)} selected layers, expected {len(required_layers)}."
            )
        for layer_idx, tensor in zip(required_layers, selected_outputs):
            selected_layer_tensors[layer_idx] = tensor
        sparse_layers = _sparse_layer_list(num_layers, selected_layer_tensors, require_last=model.camera_head is not None)
        preds = model.forward_from_aggregated_tokens(
            sparse_layers,
            patch_start_idx,
            images,
        )

    frame_tokens: List[torch.Tensor] = []
    global_tokens: List[torch.Tensor] = []
    aggregated_concat_tokens: List[torch.Tensor] = []
    global_patch_tokens: List[torch.Tensor] = []
    global_special_tokens: List[torch.Tensor] = []
    if collect_full_cache:
        assert aggregated_concat_tokens_cpu is not None
        for concat_t in aggregated_concat_tokens_cpu:
            if concat_t is None:
                raise RuntimeError("Missing cached aggregated layer from aggregator callback.")
            if concat_t.dim() != 4:
                raise RuntimeError(f"Expected cached aggregated tensor [B,S,P,2C], got shape {tuple(concat_t.shape)}")
            two_c = int(concat_t.shape[-1])
            if two_c % 2 != 0:
                raise RuntimeError(f"Invalid channel count in cached aggregated tensor: {two_c}")
            c = two_c // 2
            frame_view = concat_t[..., :c]
            global_view = concat_t[..., c:]
            frame_tokens.append(frame_view)
            global_tokens.append(global_view)
            aggregated_concat_tokens.append(concat_t)
            global_patch_tokens.append(global_view[:, :, patch_start_idx:, :])
            global_special_tokens.append(global_view[:, :, :patch_start_idx, :])

    scene_scores: Dict[str, float] = {}
    if "depth_conf" in preds:
        scene_scores["depth_conf_mean"] = float(preds["depth_conf"].mean().item())
    if "world_points_conf" in preds:
        scene_scores["point_conf_mean"] = float(preds["world_points_conf"].mean().item())
    if "pose_enc" in preds:
        scene_scores["pose_norm"] = float(preds["pose_enc"].norm(dim=-1).mean().item())

    preds_cpu = predictions_to_cpu_float(preds)

    return RedundancyBatchResult(
        frame_tokens=frame_tokens,
        global_tokens=global_tokens,
        aggregated_concat_tokens=aggregated_concat_tokens,
        global_patch_tokens=global_patch_tokens,
        global_special_tokens=global_special_tokens,
        patch_start_idx=patch_start_idx,
        scene_scores=scene_scores,
        predictions=preds_cpu,
    )


def summarize_redundancy(model: VGGT, batch_result: RedundancyBatchResult) -> Dict[str, object]:
    dev = next(model.parameters()).device
    patch_tuple = tuple(batch_result.global_patch_tokens)
    special_tuple = tuple(batch_result.global_special_tokens)
    patch_similarity = pairwise_similarity_matrices(patch_tuple, compute_pwcca=True, device=dev)
    special_similarity = pairwise_similarity_matrices(special_tuple, compute_pwcca=True, device=dev)

    ws = compute_global_weight_stats(model)
    return {
        "global_patch_similarity": {k: v.tolist() for k, v in patch_similarity.items()},
        "global_special_similarity": {k: v.tolist() for k, v in special_similarity.items()},
        **ws,
        "scene_scores": batch_result.scene_scores,
    }


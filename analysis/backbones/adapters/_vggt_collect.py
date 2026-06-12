from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from analysis.backbones.registry import get_backbone
from analysis.redundancy.metrics import pairwise_similarity_matrices, spectral_stats


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
    initial_tokens: Optional[torch.Tensor]
    global_patch_tokens: List[torch.Tensor]
    global_special_tokens: List[torch.Tensor]
    patch_start_idx: int
    scene_scores: Dict[str, float]
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)


def _required_prediction_layers(model: Any, num_layers: int) -> List[int]:
    required: set[int] = set()
    if model.camera_head is not None and num_layers > 0:
        required.add(num_layers - 1)
    if model.depth_head is not None:
        required.update(int(i) for i in getattr(model.depth_head, "intermediate_layer_idx", []))
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


def _slice_expand_and_flatten(token_tensor: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
    query = token_tensor[:, 0:1, ...].expand(bsz, 1, *token_tensor.shape[2:])
    if seq_len <= 1:
        return query.reshape(bsz * seq_len, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(bsz, seq_len - 1, *token_tensor.shape[2:])
    return torch.cat([query, others], dim=1).reshape(bsz * seq_len, *token_tensor.shape[2:])


def _aggregator_initial_tokens(model: Any, images: torch.Tensor) -> torch.Tensor:
    agg = model.aggregator
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if images.dim() != 5:
        raise ValueError(f"Expected images with shape [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")
    bsz, seq_len, channels, height, width = images.shape
    if channels != 3:
        raise ValueError(f"Expected 3 image channels, got {channels}.")

    norm_images = (images - agg._resnet_mean) / agg._resnet_std
    patch_tokens = agg.patch_embed(norm_images.view(bsz * seq_len, channels, height, width))
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    camera_token = _slice_expand_and_flatten(agg.camera_token, bsz, seq_len)
    register_token = _slice_expand_and_flatten(agg.register_token, bsz, seq_len)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
    return tokens.view(bsz, seq_len, tokens.shape[1], tokens.shape[2])


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


def compute_global_weight_stats(model: Any) -> Dict[str, Any]:
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


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_batch(
    model: Any,
    images: torch.Tensor,
    *,
    backbone_name: str = "vggt",
    keep_activations_on_gpu: bool = False,
    collect_full_cache: bool = True,
    timing_ms: Optional[Dict[str, float]] = None,
) -> RedundancyBatchResult:
    del keep_activations_on_gpu
    if images.dim() == 4:
        images = images.unsqueeze(0)
    elif images.dim() != 5:
        raise ValueError(f"Expected images with shape [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")

    dev = next(model.parameters()).device
    if timing_ms is not None:
        timing_ms.clear()

    with torch.no_grad():
        initial_tokens = (
            _aggregator_initial_tokens(model, images).detach().contiguous() if collect_full_cache else None
        )
        adapter = get_backbone(backbone_name)
        fwd = adapter.extract_layer_units_and_predictions(model, images)
        if fwd is None:
            raise RuntimeError(f"Backbone {backbone_name!r} did not return cache-forward results")

    frame_tokens: List[torch.Tensor] = []
    global_tokens: List[torch.Tensor] = []
    aggregated_concat_out: List[torch.Tensor] = []
    global_patch_tokens: List[torch.Tensor] = []
    global_special_tokens: List[torch.Tensor] = []
    if collect_full_cache:
        patch_start_idx = int(fwd.meta.get("patch_start_idx", 0))
        unit_map = fwd.unit_map
        depth = len(getattr(model.aggregator, "global_blocks", []))
        for i in range(depth):
            concat_t = torch.cat([unit_map[f"frame_{i}"], unit_map[f"global_{i}"]], dim=-1)
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
            aggregated_concat_out.append(concat_t)
            global_patch_tokens.append(global_view[:, :, patch_start_idx:, :])
            global_special_tokens.append(global_view[:, :, :patch_start_idx, :])
    else:
        patch_start_idx = int(fwd.meta.get("patch_start_idx", 0))

    scene_scores: Dict[str, float] = {}
    preds = fwd.pred.raw
    if "depth_conf" in preds:
        scene_scores["depth_conf_mean"] = float(preds["depth_conf"].mean().item())
    if "pose_enc" in preds:
        scene_scores["pose_norm"] = float(preds["pose_enc"].norm(dim=-1).mean().item())

    preds_cpu = predictions_to_cpu_float(preds)

    return RedundancyBatchResult(
        frame_tokens=frame_tokens,
        global_tokens=global_tokens,
        aggregated_concat_tokens=aggregated_concat_out,
        initial_tokens=initial_tokens,
        global_patch_tokens=global_patch_tokens,
        global_special_tokens=global_special_tokens,
        patch_start_idx=patch_start_idx,
        scene_scores=scene_scores,
        predictions=preds_cpu,
    )


def summarize_redundancy(model: Any, batch_result: RedundancyBatchResult) -> Dict[str, object]:
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

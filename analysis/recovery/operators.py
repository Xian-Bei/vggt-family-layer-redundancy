from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from analysis.backbones.contracts import BackboneAdapter


class BoundaryRecovery(nn.Module):
    """Single boundary operator used by LinearPatch-style and Ghost recovery."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if weight.dim() != 2 or int(weight.shape[0]) != int(weight.shape[1]):
            raise ValueError(f"recovery weight must be square [C,C], got {tuple(weight.shape)}")
        self.weight = nn.Parameter(weight.detach().to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x.to(torch.float32), self.weight).to(dtype=x.dtype)


def recovery_module_name(source_unit_id: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(source_unit_id))
    return f"recovery_patch_{safe}"


_REPLACEME_TYPES = {
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_replaceme_cos_more",
    "recovery_replaceme_cos_scale",
}
_FUSED_TYPES = _REPLACEME_TYPES | {"recovery_prunecomp"}


def _scale_affine_output(module: nn.Module, scale: float) -> None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected affine output module with weight, got {type(module).__name__}")
    weight.mul_(float(scale))
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        bias.mul_(float(scale))


def _scale_initial_token_sources(model: nn.Module, scale: float) -> None:
    agg = getattr(model, "aggregator", None)
    token_owner = agg if isinstance(agg, nn.Module) else model
    for name in ("camera_token", "register_token"):
        value = getattr(token_owner, name, None)
        if isinstance(value, torch.Tensor):
            value.mul_(float(scale))

    patch_source = getattr(agg, "patch_embed", None) if agg is not None else getattr(model, "encoder", None)
    if not isinstance(patch_source, nn.Module):
        raise AttributeError("Model has no recognized patch-token source for Prune&Comp fusion")
    norm = getattr(patch_source, "norm", None)
    if isinstance(norm, nn.Module) and isinstance(getattr(norm, "weight", None), torch.Tensor):
        _scale_affine_output(norm, scale)
        return
    proj = getattr(patch_source, "proj", None)
    if isinstance(proj, nn.Module) and isinstance(getattr(proj, "weight", None), torch.Tensor):
        _scale_affine_output(proj, scale)
        return
    raise AttributeError(
        f"Cannot fuse Prune&Comp into patch-token source {type(patch_source).__name__}: "
        "missing affine norm/proj"
    )


def fuse_replaceme_transform(
    model: nn.Module,
    adapter: BackboneAdapter,
    source_unit_id: str,
    transform: torch.Tensor,
) -> None:
    block = adapter.block_for_unit(model, source_unit_id)
    mlp = getattr(block, "mlp", None)
    fc2 = getattr(mlp, "fc2", None)
    if not isinstance(fc2, nn.Linear):
        raise TypeError(f"{adapter.name} {source_unit_id} does not expose MLP fc2 as nn.Linear")
    channels = int(fc2.out_features)
    if transform.dim() != 2 or tuple(transform.shape) != (channels, channels):
        raise ValueError(
            f"ReplaceMe transform for {source_unit_id} must be [{channels},{channels}], "
            f"got {tuple(transform.shape)}"
        )
    a = transform.to(device=fc2.weight.device, dtype=torch.float32)
    ls2 = getattr(block, "ls2", None)
    gamma = getattr(ls2, "gamma", None)
    if isinstance(gamma, torch.Tensor):
        g = gamma.detach().to(device=a.device, dtype=torch.float32).reshape(-1)
        if int(g.numel()) != channels:
            raise ValueError(f"LayerScale size mismatch for {source_unit_id}: {int(g.numel())} != {channels}")
        if bool((g.abs() < 1e-12).any()):
            raise ValueError(f"LayerScale for {source_unit_id} contains values too close to zero")
        raw_transform = g[:, None] * a / g[None, :]
    else:
        raw_transform = a
    with torch.no_grad():
        weight = fc2.weight.detach().to(torch.float32)
        fc2.weight.copy_((raw_transform.T @ weight).to(dtype=fc2.weight.dtype))
        if fc2.bias is not None:
            bias = fc2.bias.detach().to(torch.float32)
            fc2.bias.copy_((bias @ raw_transform).to(dtype=fc2.bias.dtype))


def fuse_prunecomp_scale(
    model: nn.Module,
    adapter: BackboneAdapter,
    source_unit_id: str,
    scale: float,
) -> None:
    alpha = float(scale)
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0.0:
        raise ValueError(f"Prune&Comp scale must be finite and positive, got {alpha}")
    from analysis.prune_plan.build_prune_plan_common import unit_id_to_index

    source_index = unit_id_to_index(source_unit_id)
    with torch.no_grad():
        _scale_initial_token_sources(model, alpha)
        if source_index < 0:
            return
        catalog = adapter.layer_unit_catalog(model)
        for unit_index, spec in enumerate(catalog):
            if unit_index > source_index:
                break
            block = adapter.block_for_unit(model, spec.unit_id)
            attn_proj = getattr(getattr(block, "attn", None), "proj", None)
            mlp_fc2 = getattr(getattr(block, "mlp", None), "fc2", None)
            if not isinstance(attn_proj, nn.Module) or not isinstance(mlp_fc2, nn.Module):
                raise TypeError(f"{adapter.name} {spec.unit_id} lacks attn.proj/mlp.fc2 for Prune&Comp")
            _scale_affine_output(attn_proj, alpha)
            _scale_affine_output(mlp_fc2, alpha)


def _patch_output(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
    patch = getattr(module, "_lare_recovery_patch")
    if torch.is_tensor(output):
        return patch(output)
    if isinstance(output, tuple):
        if not output:
            return output
        first = output[0]
        if torch.is_tensor(first):
            return (patch(first),) + output[1:]
        if isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
            patched_first = type(first)((patch(first[0]), *first[1:]))
            return (patched_first,) + output[1:]
    if isinstance(output, list):
        if not output:
            return output
        out = list(output)
        if torch.is_tensor(out[0]):
            out[0] = patch(out[0])
            return out
    return output


def install_recovery_modules(
    model: nn.Module,
    adapter: BackboneAdapter,
    recovery_specs: Sequence[Dict[str, Any]],
    *,
    device: torch.device | str,
) -> List[Any]:
    """Attach recovery modules and hooks described by checkpoint specs."""
    handles: List[Any] = []
    dev = torch.device(device)
    for raw in recovery_specs:
        spec = dict(raw)
        recovery_type = str(spec.get("type", ""))
        if recovery_type in _FUSED_TYPES:
            continue
        source_unit_id = str(spec.get("source_unit_id", "")).strip()
        if not source_unit_id:
            raise ValueError(f"recovery spec missing source_unit_id: {spec}")
        shape = spec.get("weight_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"recovery spec missing weight_shape: {spec}")
        module_name = str(spec.get("module_name") or recovery_module_name(source_unit_id))
        weight = torch.eye(int(shape[0]), int(shape[1]), dtype=torch.float32)
        patch = BoundaryRecovery(weight).to(device=dev)
        block = adapter.block_for_unit(model, source_unit_id)
        setattr(block, module_name, patch)
        setattr(block, "_lare_recovery_patch", patch)
        handles.append(block.register_forward_hook(_patch_output))
    return handles


def apply_recovery_weights(
    model: nn.Module,
    adapter: BackboneAdapter,
    recovery_specs: Sequence[Dict[str, Any]],
    recovery_weights: Any,
    *,
    device: torch.device | str,
) -> None:
    """Copy compact recovery checkpoint weights into installed modules."""
    dev = torch.device(device)
    entries: List[Dict[str, Any]] = []
    if isinstance(recovery_weights, dict):
        for key, weight in recovery_weights.items():
            entries.append({"module_name": str(key), "weight": weight})
    elif isinstance(recovery_weights, Sequence) and not torch.is_tensor(recovery_weights):
        for item in recovery_weights:
            if isinstance(item, dict):
                entries.append(dict(item))
            elif torch.is_tensor(item):
                entries.append({"weight": item})
            else:
                raise TypeError(f"Unsupported recovery weight entry: {type(item)}")
    else:
        raise TypeError(f"Unsupported recovery_weights payload: {type(recovery_weights)}")

    used: set[int] = set()
    for idx, raw in enumerate(recovery_specs):
        spec = dict(raw)
        source_unit_id = str(spec.get("source_unit_id", "")).strip()
        module_name = str(spec.get("module_name") or recovery_module_name(source_unit_id))
        match_idx = -1
        for entry_idx, entry in enumerate(entries):
            if entry_idx in used:
                continue
            entry_module = str(entry.get("module_name", module_name))
            entry_source = str(entry.get("source_unit_id", source_unit_id))
            if entry_module == module_name and entry_source == source_unit_id:
                match_idx = entry_idx
                break
        if match_idx < 0 and idx < len(entries) and idx not in used:
            match_idx = idx
        if match_idx < 0:
            raise KeyError(f"Missing recovery weight for {source_unit_id}/{module_name}")

        entry = entries[match_idx]
        used.add(match_idx)
        weight = entry.get("weight")
        if not torch.is_tensor(weight):
            raise TypeError(f"Recovery weight for {source_unit_id}/{module_name} is not a tensor")

        recovery_type = str(spec.get("type", ""))
        if recovery_type in _REPLACEME_TYPES:
            fuse_replaceme_transform(model, adapter, source_unit_id, weight)
            continue
        if recovery_type == "recovery_prunecomp":
            if int(weight.numel()) != 1:
                raise ValueError(f"Prune&Comp checkpoint must contain one scalar, got {tuple(weight.shape)}")
            fuse_prunecomp_scale(model, adapter, source_unit_id, float(weight.reshape(-1)[0].item()))
            continue

        block = adapter.block_for_unit(model, source_unit_id)
        patch = getattr(block, module_name)
        patch.weight.data.copy_(weight.to(device=dev, dtype=torch.float32))

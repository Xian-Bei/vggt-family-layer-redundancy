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
        # Fitted recovery matrices are inference constants.  A buffer keeps
        # them in state_dict/device moves without exposing them to optimizers.
        self.register_buffer("weight", weight.detach().to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recovery matrices are fitted and stored in fp32.  Keep their
        # application in fp32 as well: the surrounding backbone forward runs
        # under autocast, where an unguarded matmul would otherwise produce a
        # bf16/fp16 boundary activation and change the residual-stream dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = torch.matmul(x.to(torch.float32), self.weight.to(torch.float32))
        return output.to(dtype=x.dtype)


def recovery_module_name(source_unit_id: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(source_unit_id))
    return f"recovery_patch_{safe}"


_REPLACEME_TYPES = {
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
}
_FUSED_TYPES = _REPLACEME_TYPES
_COMPACTABLE_BOUNDARY_TYPES = {
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}


def compact_recovery_operators(
    recovery_specs: Sequence[Dict[str, Any]], recovery_weights: Any
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge consecutive boundary matrices into one minimal operator."""
    specs = [dict(spec) for spec in recovery_specs]
    if not isinstance(recovery_weights, Sequence) or torch.is_tensor(recovery_weights):
        return specs, recovery_weights
    entries = [dict(item) for item in recovery_weights if isinstance(item, dict)]
    if len(entries) != len(specs):
        return specs, entries

    compact_specs: List[Dict[str, Any]] = []
    compact_entries: List[Dict[str, Any]] = []
    for spec, entry in zip(specs, entries):
        weight = entry.get("weight")
        if not torch.is_tensor(weight):
            return specs, entries
        can_merge = bool(compact_specs) and (
            str(spec.get("type", "")) in _COMPACTABLE_BOUNDARY_TYPES
            and str(compact_specs[-1].get("type", "")) == str(spec.get("type", ""))
            and int(compact_specs[-1].get("target_unit_index", -2))
            == int(spec.get("source_unit_index", -1))
        )
        if not can_merge:
            compact_specs.append(spec)
            compact_entries.append(entry)
            continue
        previous_spec = compact_specs[-1]
        previous_entry = compact_entries[-1]
        previous_weight = previous_entry["weight"]
        combined = previous_weight.to(torch.float32) @ weight.to(torch.float32)
        previous_spec["target_unit_id"] = spec.get("target_unit_id")
        previous_spec["target_unit_index"] = spec.get("target_unit_index")
        previous_spec["weight_shape"] = [int(x) for x in combined.shape]
        previous_entry["weight"] = combined
    return compact_specs, compact_entries


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


def _make_patch_input(patch: BoundaryRecovery):
    """Apply a recovery transform at a full-unit boundary before its consumer."""

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not args or not torch.is_tensor(args[0]):
            return None
        return (patch(args[0]),) + args[1:]

    return hook


def _omega_register_boundary_consumer(
    model: nn.Module,
    adapter: BackboneAdapter,
    source_unit_id: str,
) -> nn.Module | None:
    """Return the next full-token consumer for an Omega register-only global unit."""
    if str(getattr(adapter, "name", "")).strip().lower() != "vggt_omega":
        return None
    kind, index_s = str(source_unit_id).split("_", 1)
    if kind != "global":
        return None
    index = int(index_s)
    agg = getattr(model, "aggregator", None)
    attention_types = getattr(agg, "inter_frame_attention_types", None)
    if not isinstance(attention_types, (list, tuple)):
        return None
    if index >= len(attention_types) or str(attention_types[index]) != "register":
        return None
    # Omega reassembles register and bypassed patch tokens after global_i.  The
    # next frame block is the first module that receives that complete state.
    return adapter.block_for_unit(model, f"frame_{index + 1}")


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
        model_dtype = next(
            (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
            torch.float32,
        )
        patch = BoundaryRecovery(weight).to(device=dev, dtype=model_dtype)
        block = adapter.block_for_unit(model, source_unit_id)
        setattr(block, module_name, patch)
        setattr(block, "_lare_recovery_patch", patch)
        boundary_consumer = _omega_register_boundary_consumer(
            model, adapter, source_unit_id
        )
        if boundary_consumer is not None:
            handles.append(
                boundary_consumer.register_forward_pre_hook(_make_patch_input(patch))
            )
        else:
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
        block = adapter.block_for_unit(model, source_unit_id)
        patch = getattr(block, module_name)
        patch.weight.data.copy_(weight.to(device=dev, dtype=patch.weight.dtype))

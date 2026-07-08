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
        source_unit_id = str(spec.get("source_unit_id", "")).strip()
        if not source_unit_id:
            raise ValueError(f"recovery spec missing source_unit_id: {spec}")
        shape = spec.get("weight_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"recovery spec missing weight_shape: {spec}")
        module_name = str(spec.get("module_name") or recovery_module_name(source_unit_id))
        weight = torch.eye(int(shape[0]), int(shape[1]), dtype=torch.float32)
        patch = BoundaryRecovery(weight).to(device=dev)
        if source_unit_id == "layer_-1":
            first = adapter.block_for_unit(model, "frame_0")
            setattr(first, module_name, patch)

            def make_pre_hook(attr_name: str):
                def pre_hook(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
                    if not args:
                        return args
                    x = args[0]
                    local_patch = getattr(module, attr_name)
                    if torch.is_tensor(x):
                        return (local_patch(x),) + args[1:]
                    return args

                return pre_hook

            handles.append(first.register_forward_pre_hook(make_pre_hook(module_name)))
            continue

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

        block = adapter.block_for_unit(model, "frame_0" if source_unit_id == "layer_-1" else source_unit_id)
        patch = getattr(block, module_name)
        patch.weight.data.copy_(weight.to(device=dev, dtype=torch.float32))

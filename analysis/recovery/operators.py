from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from analysis.backbones.contracts import BackboneAdapter


class BoundaryRecovery(nn.Module):
    """Single boundary operator used by LinearPatch-style and Ghost recovery."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        token_groups: Sequence[str] = ("all",),
        token_group_starts: Sequence[int] = (0,),
        runtime_token_groups: Sequence[str] | None = None,
        runtime_trailing_token_count: int = 0,
        unit_kind: str = "",
        context_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        groups = tuple(str(name).strip() for name in token_groups)
        starts = tuple(int(start) for start in token_group_starts)
        if weight.dim() == 2:
            if int(weight.shape[0]) != int(weight.shape[1]):
                raise ValueError(
                    f"recovery weight must be square [C,C], got {tuple(weight.shape)}"
                )
        elif weight.dim() == 3:
            if int(weight.shape[1]) != int(weight.shape[2]):
                raise ValueError(
                    "grouped recovery weight must be square [G,C,C], got "
                    f"{tuple(weight.shape)}"
                )
            if int(weight.shape[0]) != len(groups):
                raise ValueError(
                    f"grouped recovery weight has {weight.shape[0]} matrices for {len(groups)} groups"
                )
        else:
            raise ValueError(
                f"recovery weight must be [C,C] or [G,C,C], got {tuple(weight.shape)}"
            )
        if not groups or any(not name for name in groups) or len(groups) != len(starts):
            raise ValueError(f"invalid recovery token layout: groups={groups}, starts={starts}")
        if starts[0] != 0 or any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError(f"token-group starts must begin at 0 and increase: {starts}")
        runtime_groups = tuple(
            str(name).strip()
            for name in (runtime_token_groups if runtime_token_groups is not None else groups)
        )
        if len(runtime_groups) != len(groups) or set(runtime_groups) != set(groups):
            raise ValueError(
                f"runtime token groups must be a permutation of {groups}, got {runtime_groups}"
            )
        if int(runtime_trailing_token_count) < 0:
            raise ValueError("runtime trailing token count must be non-negative")
        # Fitted recovery matrices are inference constants.  A buffer keeps
        # them in state_dict/device moves without exposing them to optimizers.
        self.register_buffer("weight", weight.detach().to(torch.float32))
        if context_weight is None:
            self.context_weight = None
        else:
            if context_weight.dim() != 2 or context_weight.shape[0] != context_weight.shape[1]:
                raise ValueError("context weight must be square [C,C]")
            self.register_buffer("context_weight", context_weight.detach().to(torch.float32))
        self.token_groups = groups
        self.token_group_starts = starts
        self.runtime_token_groups = runtime_groups
        self.runtime_trailing_token_count = int(runtime_trailing_token_count)
        self.unit_kind = str(unit_kind).strip().lower()
        self._batch_size: int | None = None
        self._num_frames: int | None = None

    @property
    def is_grouped(self) -> bool:
        return self.weight.dim() == 3

    def set_token_layout(self, *, batch_size: int, num_frames: int) -> None:
        if int(batch_size) <= 0 or int(num_frames) <= 0:
            raise ValueError(
                f"runtime token layout must be positive, got B={batch_size}, S={num_frames}"
            )
        self._batch_size = int(batch_size)
        self._num_frames = int(num_frames)

    def _frame_tokens(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, int | None]:
        """View a block activation as ``[B,S,P,C]`` plus excluded trailing tokens."""
        batch_size = self._batch_size
        num_frames = self._num_frames
        if x.dim() == 4:
            if batch_size is None or num_frames is None:
                return x, None, None
            if int(x.shape[0]) == batch_size and int(x.shape[1]) == num_frames:
                return x, None, None
            if (
                int(x.shape[0]) == batch_size
                and int(x.shape[1]) == 1
            ):
                main_tokens = int(x.shape[2]) - self.runtime_trailing_token_count
                if main_tokens >= 0 and main_tokens % num_frames == 0:
                    trailing = (
                        x[:, :, main_tokens:]
                        if self.runtime_trailing_token_count
                        else None
                    )
                    return x[:, :, :main_tokens].reshape(
                        batch_size,
                        num_frames,
                        main_tokens // num_frames,
                        int(x.shape[3]),
                    ), trailing, 2
            if int(x.shape[0]) == 1 and int(x.shape[1]) == batch_size * num_frames:
                return (
                    x.reshape(
                        batch_size,
                        num_frames,
                        int(x.shape[2]),
                        int(x.shape[3]),
                    ),
                    None,
                    None,
                )
        if x.dim() == 3 and batch_size is not None and num_frames is not None:
            if self.unit_kind == "frame" and int(x.shape[0]) == batch_size * num_frames:
                return (
                    x.reshape(batch_size, num_frames, int(x.shape[1]), int(x.shape[2])),
                    None,
                    None,
                )
            if int(x.shape[0]) == batch_size:
                main_tokens = int(x.shape[1]) - self.runtime_trailing_token_count
                if main_tokens >= 0 and main_tokens % num_frames == 0:
                    trailing = (
                        x[:, main_tokens:]
                        if self.runtime_trailing_token_count
                        else None
                    )
                    return x[:, :main_tokens].reshape(
                        batch_size,
                        num_frames,
                        main_tokens // num_frames,
                        int(x.shape[2]),
                    ), trailing, 1
            if int(x.shape[0]) == batch_size * num_frames:
                return (
                    x.reshape(batch_size, num_frames, int(x.shape[1]), int(x.shape[2])),
                    None,
                    None,
                )
        raise RuntimeError(
            "Grouped Ghost could not recover per-frame token layout for activation "
            f"{tuple(x.shape)} with B={batch_size}, S={num_frames}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recovery matrices are fitted and stored in fp32.  Keep their
        # application in fp32 as well: the surrounding backbone forward runs
        # under autocast, where an unguarded matmul would otherwise produce a
        # bf16/fp16 boundary activation and change the residual-stream dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if not self.is_grouped:
                output = torch.matmul(x.to(torch.float32), self.weight.to(torch.float32))
            else:
                frame_tokens, trailing_tokens, trailing_dim = self._frame_tokens(x)
                tokens_per_frame = int(frame_tokens.shape[2])
                if self.token_group_starts[-1] >= tokens_per_frame:
                    raise RuntimeError(
                        f"Ghost token-group start {self.token_group_starts[-1]} is outside "
                        f"a {tokens_per_frame}-token frame"
                    )
                output_frames = torch.empty_like(frame_tokens, dtype=torch.float32)
                group_sizes = {
                    name: (
                        self.token_group_starts[group_index + 1] - start
                        if group_index + 1 < len(self.token_group_starts)
                        else tokens_per_frame - start
                    )
                    for group_index, (name, start) in enumerate(
                        zip(self.token_groups, self.token_group_starts)
                    )
                }
                runtime_start = 0
                runtime_slices = {}
                cursor = 0
                for name in self.runtime_token_groups:
                    runtime_slices[name] = (cursor, cursor + group_sizes[name])
                    cursor += group_sizes[name]
                context = None
                if self.context_weight is not None:
                    if "special" not in runtime_slices or "patch" not in runtime_slices:
                        raise RuntimeError("Patch-mean conditioning requires special/patch groups")
                    patch_left, patch_right = runtime_slices["patch"]
                    patch_mean = frame_tokens[:, :, patch_left:patch_right].to(torch.float32).mean(dim=2)
                    context = torch.matmul(patch_mean, self.context_weight.to(torch.float32)).unsqueeze(2)
                for name in self.runtime_token_groups:
                    group_index = self.token_groups.index(name)
                    end = runtime_start + group_sizes[name]
                    output_frames[:, :, runtime_start:end] = torch.matmul(
                        frame_tokens[:, :, runtime_start:end].to(torch.float32),
                        self.weight[group_index].to(torch.float32),
                    )
                    if name == "special" and context is not None:
                        output_frames[:, :, runtime_start:end] += context
                    runtime_start = end
                if runtime_start != tokens_per_frame:
                    raise RuntimeError(
                        f"Ghost runtime groups cover {runtime_start}/{tokens_per_frame} tokens"
                    )
                if trailing_tokens is None:
                    output = output_frames.reshape(x.shape)
                else:
                    main = output_frames.reshape(
                        int(output_frames.shape[0]), -1, int(output_frames.shape[-1])
                    )
                    if trailing_dim == 1:
                        output = torch.cat([main, trailing_tokens.to(torch.float32)], dim=1)
                    elif trailing_dim == 2:
                        output = torch.cat(
                            [main.unsqueeze(1), trailing_tokens.to(torch.float32)], dim=2
                        )
                    else:
                        raise RuntimeError("Grouped Ghost lost its trailing-token layout")
        return output.to(dtype=x.dtype)


class StreamlineFFN(nn.Module):
    """LLM-Streamline's point-wise FFN replacement module.

    The official implementation uses two biased linear layers with a 4x hidden
    expansion and ReLU, without an outer residual connection.  Keeping the
    module independent of any backbone block makes the same cached boundary
    objective usable for frame and global units.
    """

    def __init__(self, hidden_size: int, *, expansion_factor: int = 4) -> None:
        super().__init__()
        channels = int(hidden_size)
        expansion = int(expansion_factor)
        if channels <= 0 or expansion <= 0:
            raise ValueError(
                "Streamline FFN hidden size and expansion factor must be positive, "
                f"got hidden_size={channels}, expansion_factor={expansion}"
            )
        self.hidden_size = channels
        self.expansion_factor = expansion
        intermediate = channels * expansion
        self.fc1 = nn.Linear(channels, intermediate, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(intermediate, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        compute_dtype = self.fc1.weight.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.fc2(self.activation(self.fc1(x.to(compute_dtype))))
        return output.to(dtype=input_dtype)


def pack_streamline_ffn_parameters(module: StreamlineFFN) -> torch.Tensor:
    """Serialize a Streamline FFN into the compact recovery-weight slot."""
    return torch.cat(
        [
            module.fc1.weight.detach().reshape(-1),
            module.fc1.bias.detach().reshape(-1),
            module.fc2.weight.detach().reshape(-1),
            module.fc2.bias.detach().reshape(-1),
        ]
    ).to(device="cpu", dtype=torch.float32)


def load_streamline_ffn_parameters(
    module: StreamlineFFN, packed: torch.Tensor
) -> None:
    """Load parameters produced by :func:`pack_streamline_ffn_parameters`."""
    if packed.dim() != 1:
        raise ValueError(
            f"Streamline FFN packed parameters must be 1D, got {tuple(packed.shape)}"
        )
    parameters = (
        module.fc1.weight,
        module.fc1.bias,
        module.fc2.weight,
        module.fc2.bias,
    )
    expected = sum(int(parameter.numel()) for parameter in parameters)
    if int(packed.numel()) != expected:
        raise ValueError(
            f"Streamline FFN packed parameter size mismatch: {int(packed.numel())} != {expected}"
        )
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            end = offset + int(parameter.numel())
            value = packed[offset:end].reshape(parameter.shape)
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            offset = end


def set_recovery_token_layout(
    model: nn.Module, *, batch_size: int, num_frames: int
) -> None:
    """Set frame geometry for recovery patches used by direct block replay paths."""
    for module in model.modules():
        if isinstance(module, BoundaryRecovery) and module.is_grouped:
            module.set_token_layout(batch_size=batch_size, num_frames=num_frames)


def recovery_module_name(source_unit_id: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(source_unit_id))
    return f"recovery_patch_{safe}"


_REPLACEME_TYPES = {
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
}
_FUSED_TYPES = _REPLACEME_TYPES
# Unit-wise methods intentionally remain one operator per removed physical unit
# so prediction heads can observe progressively recovered intermediate states.
_COMPACTABLE_BOUNDARY_TYPES = {
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_interval_sequential_grouped",
    "recovery_ghost_interval_sequential_autogroup",
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
            and str(spec.get("ghost_coupling_profile", "none")) == "none"
            and str(compact_specs[-1].get("ghost_coupling_profile", "none")) == "none"
            and str(compact_specs[-1].get("type", "")) == str(spec.get("type", ""))
            and int(compact_specs[-1].get("target_unit_index", -2))
            == int(spec.get("source_unit_index", -1))
            and tuple(compact_specs[-1].get("token_groups") or ("all",))
            == tuple(spec.get("token_groups") or ("all",))
            and tuple(compact_specs[-1].get("token_group_starts") or (0,))
            == tuple(spec.get("token_group_starts") or (0,))
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


def _make_patch_input(patch: nn.Module):
    """Apply a recovery transform at a full-unit boundary before its consumer."""

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not args or not torch.is_tensor(args[0]):
            return None
        return (patch(args[0]),) + args[1:]

    return hook


def _batch_frames_from_forward_value(value: Any) -> Tuple[int, int] | None:
    if torch.is_tensor(value):
        if value.dim() == 5:
            return int(value.shape[0]), int(value.shape[1])
        return None
    if isinstance(value, dict):
        image = value.get("img")
        if torch.is_tensor(image) and image.dim() == 4:
            return int(image.shape[0]), 1
        for nested in value.values():
            found = _batch_frames_from_forward_value(nested)
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        if value and all(isinstance(item, dict) for item in value):
            first_image = value[0].get("img")
            if torch.is_tensor(first_image) and first_image.dim() == 4:
                return int(first_image.shape[0]), len(value)
        for nested in value:
            found = _batch_frames_from_forward_value(nested)
            if found is not None:
                return found
    return None


def _make_token_layout_context(patch: BoundaryRecovery):
    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        found = _batch_frames_from_forward_value(args)
        if found is not None:
            patch.set_token_layout(batch_size=found[0], num_frames=found[1])

    return hook


def _token_layout_context_modules(model: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    for candidate in (
        model,
        getattr(model, "aggregator", None),
        getattr(model, "info_sharing", None),
    ):
        if isinstance(candidate, nn.Module) and all(candidate is not item for item in modules):
            modules.append(candidate)
    return modules


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
        module_name = str(spec.get("module_name") or recovery_module_name(source_unit_id))
        if recovery_type == "recovery_streamline_ffn":
            hidden_size = int(spec.get("hidden_size", 0))
            expansion_factor = int(spec.get("expansion_factor", 4))
            if hidden_size <= 0:
                raise ValueError(f"Streamline FFN spec missing hidden_size: {spec}")
            patch = StreamlineFFN(
                hidden_size, expansion_factor=expansion_factor
            ).to(device=dev)
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
            continue
        shape = spec.get("weight_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
            raise ValueError(f"recovery spec missing weight_shape: {spec}")
        coupling_profile = str(spec.get("ghost_coupling_profile", "none"))
        is_coupled = coupling_profile != "none"
        if len(shape) == 3 and not recovery_type.startswith("recovery_ghost"):
            raise ValueError(
                f"Grouped recovery matrices are only supported by Ghost, got {recovery_type!r}"
            )
        if len(shape) == 2:
            weight = torch.eye(int(shape[0]), int(shape[1]), dtype=torch.float32)
        elif is_coupled:
            if int(shape[0]) != 7 or int(shape[1]) != int(shape[2]):
                raise ValueError(f"coupled Ghost weight must be packed [7,C,C], got {shape}")
            weight = torch.eye(int(shape[1]), dtype=torch.float32).repeat(2, 1, 1)
        else:
            if int(shape[1]) != int(shape[2]):
                raise ValueError(f"grouped recovery weight must be [G,C,C], got {shape}")
            weight = torch.eye(int(shape[1]), dtype=torch.float32).repeat(int(shape[0]), 1, 1)
        model_dtype = next(
            (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
            torch.float32,
        )
        token_groups = tuple(spec.get("token_groups") or ("all",))
        runtime_order_fn = getattr(adapter, "ghost_runtime_token_group_order", None)
        runtime_token_groups = (
            tuple(runtime_order_fn(token_groups))
            if callable(runtime_order_fn)
            else token_groups
        )
        trailing_count_fn = getattr(adapter, "ghost_runtime_trailing_token_count", None)
        runtime_trailing_token_count = (
            int(trailing_count_fn()) if callable(trailing_count_fn) else 0
        )
        patch = BoundaryRecovery(
            weight,
            token_groups=token_groups,
            token_group_starts=tuple(spec.get("token_group_starts") or (0,)),
            runtime_token_groups=runtime_token_groups,
            runtime_trailing_token_count=runtime_trailing_token_count,
            unit_kind=source_unit_id.split("_", 1)[0],
            context_weight=(torch.zeros(int(shape[-1]), int(shape[-1])) if is_coupled else None),
        ).to(device=dev, dtype=model_dtype)
        block = adapter.block_for_unit(model, source_unit_id)
        setattr(block, module_name, patch)
        setattr(block, "_lare_recovery_patch", patch)
        if patch.is_grouped:
            for context_module in _token_layout_context_modules(model):
                handles.append(
                    context_module.register_forward_pre_hook(
                        _make_token_layout_context(patch)
                    )
                )
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
        if recovery_type == "recovery_streamline_ffn":
            if not isinstance(patch, StreamlineFFN):
                raise TypeError(
                    f"Installed recovery module {source_unit_id}/{module_name} is not StreamlineFFN"
                )
            load_streamline_ffn_parameters(patch, weight)
            continue
        if str(spec.get("ghost_coupling_profile", "none")) != "none":
            if weight.dim() != 3 or int(weight.shape[0]) != 7:
                raise ValueError(f"coupled Ghost checkpoint must contain [7,C,C], got {tuple(weight.shape)}")
            patch.weight.data.copy_(weight[:2].to(device=dev, dtype=patch.weight.dtype))
            if patch.context_weight is None:
                raise TypeError("coupled Ghost module lacks context buffer")
            patch.context_weight.data.copy_(weight[2].to(device=dev, dtype=patch.context_weight.dtype))
        else:
            patch.weight.data.copy_(weight.to(device=dev, dtype=patch.weight.dtype))

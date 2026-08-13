"""Training-free boundary recovery for pruned LaRe models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import copy
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.amp_policy import autocast_device_dtype
from analysis.datasets.image_batch import prepare_cache_batch
from analysis.prune_plan import (
    PruneAction,
    UnitPrunePlan,
    load_named_prune_plan,
    prune_plan_bundle_path,
    unit_index_to_kind_layer,
    unit_index_to_unit_id,
)
from analysis.recovery.operators import (
    apply_recovery_weights,
    compact_recovery_operators,
    fuse_replaceme_transform,
    install_recovery_modules,
    recovery_module_name,
)
from analysis.recovery.activation_store import (
    RecoveryActivationStore,
    RecoverySession,
    RowShard,
    ShardedRowStore,
    rows_view,
)
from analysis.recovery.train.checkpoint_v3 import atomic_save_checkpoint
from analysis.recovery.shared_results import recovery_result_complete
from analysis.redundancy.cache_layout import load_scene_unit
from analysis.redundancy.cache_layout_v3 import (
    _torch_load_tensor,
    load_scene_metadata_v3,
    load_v3_manifest,
    manifest_scenes_for_training,
)
from analysis.redundancy.cache_redundancy_v3 import (
    forward_selected_units_with_auxiliaries,
)
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir
from analysis.prune_plan.recovery_plan import (
    TRAINING_FREE_RECOVERY_METHODS,
    compose_recovery_unit_prune_plan,
    load_recovery_plan_bundle,
)


DEFAULT_RECOVERY_CHUNK_ROWS = 16_384
_DERIVED_GHOST_METHODS = {
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fit training-free recovery operators for a pruned model.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--prune-plan-root", type=Path, required=True)
    p.add_argument("--prune-plan-name", type=str, required=True)
    p.add_argument("--recovery-plan-root", type=Path, default=None)
    p.add_argument("--recovery-plan-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument(
        "--recovery-chunk-rows",
        type=int,
        default=DEFAULT_RECOVERY_CHUNK_ROWS,
        help="Maximum activation rows staged on the GPU per recovery chunk.",
    )
    p.add_argument("--replaceme-ls-alpha", type=float, default=0.0)
    p.add_argument("--replaceme-cos-epochs", type=int, default=10)
    p.add_argument("--replaceme-cos-batch-size", type=int, default=1024)
    p.add_argument("--replaceme-cos-lr", type=float, default=1e-4)
    p.add_argument("--replaceme-seed", type=int, default=0)
    p.add_argument(
        "--jobs-manifest",
        type=Path,
        default=None,
        help="JSON list of prune/recovery/output requests handled in one process.",
    )
    return p


def _parse_name_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").replace(",", " ").split() if x.strip()]


_SESSION_SCENES: Dict[Tuple[Tuple[str, ...], int], List[Dict[str, Any]]] = {}


def _load_training_scenes(cache_dirs: Sequence[Path], max_scenes: int) -> List[Dict[str, Any]]:
    session_key = (tuple(str(Path(path).resolve()) for path in cache_dirs), int(max_scenes))
    cached = _SESSION_SCENES.get(session_key)
    if cached is not None:
        return cached
    scenes: List[Dict[str, Any]] = []
    for cache_dir in cache_dirs:
        manifest = load_v3_manifest(cache_dir)
        for item in manifest_scenes_for_training(manifest):
            item = dict(item)
            # The legacy training view intentionally contains only tensor-file
            # references. Progressive recovery additionally needs the exact
            # frame ids/paths and geometry saved in per-scene metadata.
            item.update(load_scene_metadata_v3(cache_dir, item))
            item["_cache_dir"] = str(cache_dir)
            scenes.append(item)
            if int(max_scenes) > 0 and len(scenes) >= int(max_scenes):
                _SESSION_SCENES[session_key] = scenes
                return scenes
    _SESSION_SCENES[session_key] = scenes
    return scenes


def _load_unit_tensor(
    cache_dir: Path,
    item: Dict[str, Any],
    unit_id: str,
    *,
    num_layers: int,
    map_location: Optional[str] = None,
) -> torch.Tensor:
    del num_layers  # retained for call-site compatibility
    if _ACTIVE_RECOVERY_SESSION is not None:
        return _ACTIVE_RECOVERY_SESSION.unit(cache_dir, item, str(unit_id))
    return load_scene_unit(
        cache_dir,
        item,
        str(unit_id),
        mmap=_is_cpu_location_for_recovery(map_location),
        map_location=map_location,
    )


def _is_cpu_location_for_recovery(map_location: Optional[str]) -> bool:
    if map_location is None:
        return True
    try:
        return torch.device(map_location).type == "cpu"
    except (TypeError, RuntimeError):
        return False


_ACTIVE_RECOVERY_SESSION: Optional[RecoverySession] = None


def _flatten(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 4:
        raise ValueError(f"expected 4D unit tensor [B,S,P,C], got {tuple(t.shape)}")
    return t.reshape(-1, int(t.shape[-1])).to(torch.float32)


def _stage_row_chunk(
    rows: torch.Tensor,
    start: int,
    end: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cpu = rows[int(start) : int(end)].to(device="cpu", dtype=dtype)
    if device.type == "cuda":
        if not cpu.is_pinned():
            cpu = cpu.pin_memory()
        return cpu.to(device=device, non_blocking=True)
    return cpu


def _align_replaceme_activations(
    h: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    *,
    scene_id: str,
    source_unit_id: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align boundary activations with the tokens updated by the source MLP.

    Some backbones run an inter-frame block on only the leading camera/register
    tokens while carrying patch tokens through unchanged.  The cached source and
    target activations retain all tokens, but the captured MLP residual naturally
    contains only that leading subset.
    """
    if h.dim() != 4 or y.dim() != 4 or r.dim() != 4:
        raise RuntimeError(
            f"ReplaceMe expects [B,S,P,C] activations for {scene_id}: "
            f"H={tuple(h.shape)} Y={tuple(y.shape)} R={tuple(r.shape)}"
        )
    if h.shape != y.shape:
        raise RuntimeError(
            f"ReplaceMe boundary mismatch for {scene_id}: H={tuple(h.shape)} Y={tuple(y.shape)}"
        )
    if h.shape == r.shape:
        return h, y, r
    same_layout = (
        int(h.shape[0]) == int(r.shape[0])
        and int(h.shape[1]) == int(r.shape[1])
        and int(h.shape[3]) == int(r.shape[3])
    )
    is_prefix_residual = (
        str(source_unit_id).startswith("global_")
        and same_layout
        and 0 < int(r.shape[2]) < int(h.shape[2])
    )
    if is_prefix_residual:
        token_count = int(r.shape[2])
        return h[:, :, :token_count, :], y[:, :, :token_count, :], r
    raise RuntimeError(
        f"ReplaceMe activation mismatch for {scene_id}: "
        f"H={tuple(h.shape)} Y={tuple(y.shape)} R={tuple(r.shape)}"
    )


def _load_auxiliary_tensor(
    cache_dir: Path,
    item: Dict[str, Any],
    auxiliary_id: str,
    *,
    map_location: Optional[str] = None,
) -> torch.Tensor:
    files = item.get("auxiliary_files")
    rel = str(files.get(auxiliary_id, "")).strip() if isinstance(files, dict) else ""
    if not rel:
        raise FileNotFoundError(
            f"Scene {item.get('scene_id')} lacks {auxiliary_id!r}; rerun stage 8 cache "
            "with the ReplaceMe recovery plan selected"
        )
    path = cache_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing recovery auxiliary tensor {path}")
    tensor = _torch_load_tensor(
        path,
        mmap=_is_cpu_location_for_recovery(map_location),
        map_location=map_location,
    )
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        raise ValueError(f"Expected [B,S,P,C] tensor at {path}, got {type(tensor)}")
    return tensor


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard(size: int, *, device: torch.device) -> torch.Tensor:
    requested = int(size)
    if requested <= 0:
        raise ValueError(f"LinearPatch channels must be positive, got {size}")
    # Sylvester Hadamard matrices exist at powers of two.  For model widths
    # such as MapAnything's 768, construct the next matrix and orthogonalize
    # its leading square submatrix.  Power-of-two widths retain the exact
    # historical transform.
    expanded = 1 << (requested - 1).bit_length()
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    while int(h.shape[0]) < expanded:
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    if expanded == requested:
        return h / float(requested) ** 0.5
    q, _r = torch.linalg.qr(h[:requested, :requested])
    return q


def _pair_records(
    bundle: Dict[str, Any], prune_plan: UnitPrunePlan, recovery_plan_name: str
) -> Tuple[str, List[Dict[str, Any]]]:
    method = str(bundle.get("method", "")).strip().lower()
    if method not in TRAINING_FREE_RECOVERY_METHODS:
        raise ValueError(f"{recovery_plan_name} is not training-free recovery: method={method!r}")
    extra = prune_plan.extra
    boundaries = extra.get("recovery_boundaries")
    if isinstance(boundaries, list) and boundaries:
        records = [dict(item) for item in boundaries if isinstance(item, dict)]
        if len(records) != len(boundaries):
            raise ValueError("Recovery boundaries must contain only objects")
        return method, records
    # Backward compatibility for checkpoints/plans written before multi-boundary
    # recovery was introduced.
    return method, [
        {
            "source_unit_index": int(extra["recovery_source_unit_index"]),
            "source_unit_id": str(extra["recovery_source_unit_id"]),
            "target_unit_index": int(extra["recovery_target_unit_index"]),
            "target_unit_id": str(extra["recovery_target_unit_id"]),
        }
    ]


_CLOSED_FORM_SESSION_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def _fit_boundary_weight(
    *,
    method: str,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    ridge: float,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
    boundary_index: int = 0,
    boundary_total: int = 1,
) -> torch.Tensor:
    cache_key: Tuple[Any, ...] = (
        str(method), str(source_unit_id), str(target_unit_id), int(num_layers),
        str(device), float(ridge), int(chunk_rows),
        tuple(
            (str(item.get("_cache_dir", "")), str(item.get("scene_id", "")))
            for item in scenes
        ),
    )
    cached_weight = _CLOSED_FORM_SESSION_CACHE.get(cache_key)
    if cached_weight is not None:
        return cached_weight.clone()
    xtx: Optional[torch.Tensor] = None
    xtd: Optional[torch.Tensor] = None
    sum_x_abs: Optional[torch.Tensor] = None
    sum_y_abs: Optional[torch.Tensor] = None
    count = 0
    hmat: Optional[torch.Tensor] = None
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        cache_dir = Path(str(item["_cache_dir"]))
        x_cpu = rows_view(
            _load_unit_tensor(
                cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu"
            )
        )
        y_cpu = rows_view(
            _load_unit_tensor(
                cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu"
            )
        )
        if x_cpu.shape != y_cpu.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(x_cpu.shape)} {target_unit_id}={tuple(y_cpu.shape)}"
            )
        for start in range(0, int(x_cpu.shape[0]), int(chunk_rows)):
            end = min(start + int(chunk_rows), int(x_cpu.shape[0]))
            x = _stage_row_chunk(x_cpu, start, end, device=device)
            y = _stage_row_chunk(y_cpu, start, end, device=device)
            if method == "recovery_ghost":
                x64 = x.to(torch.float64)
                delta64 = (y - x).to(torch.float64)
                xx = x64.T @ x64
                xd = x64.T @ delta64
                xtx = xx if xtx is None else xtx + xx
                xtd = xd if xtd is None else xtd + xd
                del x64, delta64, xx, xd
            else:
                if hmat is None:
                    hmat = _hadamard(int(x.shape[-1]), device=device)
                xx = x @ hmat if method == "recovery_linearpatch" else x
                yy = y @ hmat if method == "recovery_linearpatch" else y
                sx = xx.abs().sum(dim=0)
                sy = yy.abs().sum(dim=0)
                sum_x_abs = sx if sum_x_abs is None else sum_x_abs + sx
                sum_y_abs = sy if sum_y_abs is None else sum_y_abs + sy
                count += int(xx.shape[0])
                del xx, yy, sx, sy
            del x, y
        del x_cpu, y_cpu
        if scene_index == total_scenes or scene_index % 10 == 0:
            print(
                "[recovery-fit]"
                f"[boundary={boundary_index + 1}/{boundary_total}]"
                f"[scenes={scene_index}/{total_scenes}] "
                f"method={method.removeprefix('recovery_')}",
                file=sys.stderr,
                flush=True,
            )

    if _ACTIVE_RECOVERY_SESSION is not None:
        _ACTIVE_RECOVERY_SESSION.clear_units()
    if method == "recovery_ghost":
        if xtx is None or xtd is None:
            raise RuntimeError("No activation rows were available for ghost recovery")
        result = _solve_ghost_from_statistics(
            xtx, xtd, ridge=float(ridge)
        ).cpu()
        _CLOSED_FORM_SESSION_CACHE[cache_key] = result.clone()
        return result

    if sum_x_abs is None or sum_y_abs is None or count <= 0:
        raise RuntimeError(f"No activation rows were available for {method}")
    scale = sum_y_abs / (sum_x_abs + 1e-8)
    assert hmat is not None
    result = (hmat @ torch.diag(scale.to(torch.float32)) @ hmat.T).to(torch.float32).cpu()
    _CLOSED_FORM_SESSION_CACHE[cache_key] = result.clone()
    return result


def _solve_ghost_from_statistics(
    xtx: torch.Tensor,
    xtd: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    channels = int(xtx.shape[0])
    reg = float(ridge) * torch.eye(channels, device=xtx.device, dtype=torch.float64)
    delta = torch.linalg.solve(xtx + reg, xtd)
    return (
        torch.eye(channels, device=xtx.device, dtype=torch.float64) + delta
    ).to(torch.float32)


def _fit_replaceme_ls(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    alpha: float,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
    boundary_index: int = 0,
    boundary_total: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    rtr: Optional[torch.Tensor] = None
    rtd: Optional[torch.Tensor] = None
    rows = 0
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        cache_dir = Path(str(item["_cache_dir"]))
        h_raw = _load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu")
        y_raw = _load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu")
        r_raw = _load_auxiliary_tensor(cache_dir, item, auxiliary_id, map_location="cpu")
        h_raw, y_raw, r_raw = _align_replaceme_activations(
            h_raw,
            y_raw,
            r_raw,
            scene_id=str(item.get("scene_id", "<unknown>")),
            source_unit_id=source_unit_id,
        )
        h_cpu, y_cpu, r_cpu = rows_view(h_raw), rows_view(y_raw), rows_view(r_raw)
        for start in range(0, int(r_cpu.shape[0]), int(chunk_rows)):
            end = min(start + int(chunk_rows), int(r_cpu.shape[0]))
            h = _stage_row_chunk(h_cpu, start, end, device=device)
            y = _stage_row_chunk(y_cpu, start, end, device=device)
            r = _stage_row_chunk(r_cpu, start, end, device=device)
            r64 = r.to(torch.float64)
            d64 = (y + r - h).to(torch.float64)
            rr = r64.T @ r64
            rd = r64.T @ d64
            rtr = rr if rtr is None else rtr + rr
            rtd = rd if rtd is None else rtd + rd
            rows += int(r.shape[0])
            del h, y, r, r64, d64, rr, rd
        del h_raw, y_raw, r_raw, h_cpu, y_cpu, r_cpu
        if scene_index == total_scenes or scene_index % 10 == 0:
            print(
                "[recovery-fit]"
                f"[boundary={boundary_index + 1}/{boundary_total}]"
                f"[scenes={scene_index}/{total_scenes}] "
                "method=replaceme_ls",
                file=sys.stderr,
                flush=True,
            )
    if _ACTIVE_RECOVERY_SESSION is not None:
        _ACTIVE_RECOVERY_SESSION.clear_units()
    if rtr is None or rtd is None or rows <= 0:
        raise RuntimeError("No activation rows were available for ReplaceMe LS")
    channels = int(rtr.shape[0])
    reg = float(alpha) * torch.eye(channels, device=device, dtype=torch.float64)
    transform = torch.linalg.solve(rtr + reg, rtd).to(torch.float32).cpu()
    return transform, {
        "num_rows": rows,
        "ls_alpha": float(alpha),
        "chunk_rows": int(chunk_rows),
    }


def _available_cpu_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _materialize_cosine_rows(
    row_store: ShardedRowStore,
    *,
    device: torch.device,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """Build contiguous FP32 activation matrices without changing fit inputs."""
    num_rows = int(row_store.num_rows)
    channels = int(row_store.channels)
    required_bytes = num_rows * channels * 2 * torch.float32.itemsize
    candidates: List[Tuple[torch.device, str]] = []
    if device.type == "cuda":
        indexed_device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(indexed_device)
        reserve_bytes = max(4 * 1024**3, int(total_bytes * 0.10))
        if required_bytes + reserve_bytes <= free_bytes:
            candidates.append((indexed_device, "contiguous_gpu_float32"))
        else:
            print(
                "[recovery-warning] method=replaceme_cos "
                "fallback=contiguous_cpu reason=insufficient_gpu_memory "
                f"required={required_bytes} free={free_bytes} reserve={reserve_bytes}",
                file=sys.stderr,
                flush=True,
            )
    cpu_available = _available_cpu_memory_bytes()
    if cpu_available <= 0 or required_bytes + 8 * 1024**3 <= cpu_available:
        candidates.append((torch.device("cpu"), "contiguous_cpu_float32"))
    elif device.type != "cuda" or not candidates:
        print(
            "[recovery-warning] method=replaceme_cos "
            "fallback=scene_sharded reason=insufficient_cpu_memory "
            f"required={required_bytes} available={cpu_available}",
            file=sys.stderr,
            flush=True,
        )

    for storage_device, storage_mode in candidates:
        inputs: Optional[torch.Tensor] = None
        desired: Optional[torch.Tensor] = None
        try:
            inputs = torch.empty(
                (num_rows, channels), device=storage_device, dtype=torch.float32
            )
            desired = torch.empty_like(inputs)
            cursor = 0
            for _scene_id, scene_inputs, scene_desired in row_store.iter_scenes():
                scene_rows = int(scene_inputs.shape[0])
                if scene_desired.shape != scene_inputs.shape:
                    raise RuntimeError(
                        "ReplaceMe contiguous materialization received mismatched rows: "
                        f"{tuple(scene_inputs.shape)} vs {tuple(scene_desired.shape)}"
                    )
                for start in range(0, scene_rows, int(chunk_rows)):
                    end = min(start + int(chunk_rows), scene_rows)
                    out_start = cursor + start
                    out_end = cursor + end
                    inputs[out_start:out_end].copy_(
                        scene_inputs[start:end].to(
                            device=storage_device, dtype=torch.float32
                        )
                    )
                    desired[out_start:out_end].copy_(
                        scene_desired[start:end].to(
                            device=storage_device, dtype=torch.float32
                        )
                    )
                cursor += scene_rows
            if cursor != num_rows:
                raise RuntimeError(
                    f"ReplaceMe materialized {cursor} rows, expected {num_rows}"
                )
            row_store.close()
            print(
                "[recovery-fit] "
                f"method=replaceme_cos storage={storage_mode} "
                f"rows={num_rows} channels={channels} "
                f"bytes={required_bytes}",
                file=sys.stderr,
                flush=True,
            )
            return inputs, desired, storage_mode
        except torch.OutOfMemoryError:
            del inputs, desired
            if storage_device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                "[recovery-warning] method=replaceme_cos "
                f"storage={storage_mode} fallback=next_available reason=oom",
                file=sys.stderr,
                flush=True,
            )
    print(
        "[recovery-warning] method=replaceme_cos "
        "fallback=scene_sharded reason=no_contiguous_storage_available",
        file=sys.stderr,
        flush=True,
    )
    return None, None, "scene_sharded"


def _optimize_cosine_transform(
    *,
    input_rows: Optional[Sequence[torch.Tensor]] = None,
    desired_rows: Optional[Sequence[torch.Tensor]] = None,
    row_store: Optional[ShardedRowStore] = None,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    method_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit the ReplaceMe cosine objective from contiguous or sharded rows."""
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError(
            f"{method_name} epochs, batch size, and learning rate must be positive"
        )
    owns_store = row_store is None
    if row_store is None:
        if not input_rows or desired_rows is None or len(input_rows) != len(desired_rows):
            raise RuntimeError(f"No paired activation rows were available for {method_name}")
        row_store = ShardedRowStore(
            [
                RowShard(
                    scene_id=str(index),
                    inputs=rows_view(x) if x.dim() == 4 else x,
                    desired=rows_view(y) if y.dim() == 4 else y,
                )
                for index, (x, y) in enumerate(zip(input_rows, desired_rows))
            ]
        )

    channels = int(row_store.channels)
    num_rows = int(row_store.num_rows)
    all_inputs, all_desired, row_storage = _materialize_cosine_rows(
        row_store, device=device
    )
    transform = torch.eye(
        channels, device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([transform], lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows_seen = 0
    optimizer_steps = 0
    final_loss = float("nan")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(num_rows, generator=generator)
        order_device = order.to(device=device) if all_inputs is not None and all_inputs.device.type == "cuda" else order
        epoch_loss = torch.zeros((), device=device, dtype=torch.float32)
        last_loss: Optional[torch.Tensor] = None
        epoch_steps = 0
        for start in range(0, num_rows, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            if all_inputs is not None and all_desired is not None:
                storage_indices = (
                    order_device[start : start + int(batch_size)]
                    if all_inputs.device.type == "cuda"
                    else indices
                )
                xb = all_inputs[storage_indices].to(device=device, dtype=torch.float32)
                yb = all_desired[storage_indices].to(device=device, dtype=torch.float32)
            else:
                xb_cpu, yb_cpu = row_store.gather(
                    indices, pin_memory=device.type == "cuda"
                )
                xb = xb_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
                )
                yb = yb_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
                )
                row_store.record_transfer()
            optimizer.zero_grad(set_to_none=True)
            prediction = xb @ transform
            prediction_norm = prediction / prediction.norm(dim=1, keepdim=True)
            desired_norm = yb / yb.norm(dim=1, keepdim=True)
            loss = 1.0 - (prediction_norm * desired_norm).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
            epoch_loss.add_(last_loss)
            epoch_steps += 1
            rows_seen += int(xb.shape[0])
            optimizer_steps += 1
        mean_loss_tensor = epoch_loss / float(epoch_steps)
        if not bool(torch.isfinite(mean_loss_tensor)):
            raise FloatingPointError(f"{method_name} produced a non-finite loss")
        mean_loss = float(mean_loss_tensor.item())
        assert last_loss is not None
        final_loss = float(last_loss.item())
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
            }
        )
        print(
            "[recovery-fit] "
            f"method={method_name.lower().replace(' ', '_')} "
            f"epochs={epoch}/{epochs} mean_loss={history[-1]['mean_loss']:.6g}",
            file=sys.stderr,
            flush=True,
        )

    result = transform.detach().cpu()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{method_name} produced a non-finite transform")
    diagnostics = {
        "rows_seen": rows_seen,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "loss_components": ["cosine"],
        "final_loss": final_loss,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "row_storage": row_storage,
    }
    del all_inputs, all_desired
    if owns_store:
        row_store.close()
    return result, diagnostics


def _fit_replaceme_cos(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    boundary_index: int = 0,
    boundary_total: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    shards: List[RowShard] = []
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        cache_dir = Path(str(item["_cache_dir"]))
        h_raw = _load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu")
        y_raw = _load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu")
        r_raw = _load_auxiliary_tensor(cache_dir, item, auxiliary_id, map_location="cpu")
        h_raw, y_raw, r_raw = _align_replaceme_activations(
            h_raw,
            y_raw,
            r_raw,
            scene_id=str(item.get("scene_id", "<unknown>")),
            source_unit_id=source_unit_id,
        )
        h = rows_view(h_raw)
        y = rows_view(y_raw)
        r = rows_view(r_raw)
        # Capture mmap-backed operands; only selected batch rows are combined.
        def desired_builder(
            indices: torch.Tensor,
            *,
            h_rows: torch.Tensor = h,
            y_rows: torch.Tensor = y,
            r_rows: torch.Tensor = r,
        ) -> torch.Tensor:
            return y_rows[indices] + r_rows[indices] - h_rows[indices]

        shards.append(
            RowShard(
                scene_id=str(item.get("scene_id", "")),
                inputs=r,
                desired_builder=desired_builder,
            )
        )
        if scene_index == total_scenes or scene_index % 10 == 0:
            print(
                "[recovery-fit]"
                f"[boundary={boundary_index + 1}/{boundary_total}]"
                f"[scenes={scene_index}/{total_scenes}] "
                "method=replaceme_cos phase=collection",
                file=sys.stderr,
                flush=True,
            )
    store = RecoveryActivationStore(shards)
    try:
        return _optimize_cosine_transform(
        row_store=store,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        method_name="ReplaceMe cosine",
        )
    finally:
        store.close()
        if _ACTIVE_RECOVERY_SESSION is not None:
            _ACTIVE_RECOVERY_SESSION.clear_units()


def _forward_current_ghost_boundary(
    *,
    model: Any,
    adapter: Any,
    source_unit_id: str,
    target_unit_id: str,
    item: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Forward one scene while retaining only the requested Ghost boundary."""
    image_paths = [str(path) for path in item.get("image_paths", []) if str(path)]
    if not image_paths:
        raise FileNotFoundError(
            f"Progressive recovery requires image_paths in cache metadata for "
            f"scene {item.get('scene_id', '<unknown>')}"
        )
    prepared = prepare_cache_batch(
        None,
        max_frames=0,
        meta={
            "image_paths": image_paths,
            "frame_ids": list(item.get("frame_ids") or []),
        },
        backbone_adapter=adapter,
    )
    images = prepared.images.to(device=device, non_blocking=False)
    requested = {source_unit_id, target_unit_id}
    with torch.inference_mode(), torch.autocast(**autocast_device_dtype(device)):
        unit_map, _meta = adapter.extract_layer_units(
            model,
            images,
            unit_ids=requested,
        )
    missing = sorted(requested - set(unit_map))
    if missing:
        raise KeyError(
            f"Sequential Ghost forward missed {source_unit_id}/{target_unit_id}: {missing}"
        )
    return unit_map, images


def _fit_current_ghost_boundary(
    *,
    model: Any,
    adapter: Any,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
    ridge: float,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
    boundary_index: int = 0,
    boundary_total: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit one progressive Ghost boundary with O(C^2) persistent memory."""
    xtx: Optional[torch.Tensor] = None
    xtd: Optional[torch.Tensor] = None
    num_rows = 0
    num_scenes = 0
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        unit_map, images = _forward_current_ghost_boundary(
            model=model,
            adapter=adapter,
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            item=item,
            device=device,
        )
        source = unit_map[source_unit_id]
        target = unit_map[target_unit_id]
        if source.shape != target.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(source.shape)} "
                f"{target_unit_id}={tuple(target.shape)}"
            )
        channels = int(source.shape[-1])
        source_flat = source.reshape(-1, channels)
        target_flat = target.reshape(-1, channels)
        # Bound FP64 workspace while retaining the existing FP64 normal equations.
        for row_start in range(0, int(source_flat.shape[0]), int(chunk_rows)):
            row_end = min(row_start + int(chunk_rows), int(source_flat.shape[0]))
            x = source_flat[row_start:row_end].to(device=device, dtype=torch.float64)
            delta = target_flat[row_start:row_end].to(
                device=device, dtype=torch.float64
            )
            delta.sub_(x)
            xx = x.T @ x
            xd = x.T @ delta
            if xtx is None:
                xtx, xtd = xx, xd
            else:
                xtx.add_(xx)
                assert xtd is not None
                xtd.add_(xd)
            del x, delta, xx, xd
        num_rows += int(source_flat.shape[0])
        num_scenes += 1
        del unit_map, images, source, target, source_flat, target_flat
        if scene_index == total_scenes or scene_index % 10 == 0:
            print(
                "[recovery-fit]"
                f"[boundary={boundary_index + 1}/{boundary_total}]"
                f"[scenes={scene_index}/{total_scenes}] "
                "method=ghost_interval_sequential "
                f"source={source_unit_id} target={target_unit_id}",
                file=sys.stderr,
                flush=True,
            )
    if xtx is None or xtd is None:
        raise RuntimeError("Ghost recovery requires at least one activation row")
    weight = _solve_ghost_from_statistics(xtx, xtd, ridge=float(ridge))
    return weight.detach().cpu(), {
        "fit_mode": "sequential_current_model",
        "num_scenes": num_scenes,
        "num_rows": num_rows,
        "ridge": float(ridge),
        "chunk_rows": int(chunk_rows),
    }


_SEQUENTIAL_GHOST_PREFIX_CACHE: Dict[
    Tuple[Any, ...], Dict[Tuple[Tuple[Any, ...], ...], List[Tuple[torch.Tensor, Dict[str, Any]]]]
] = {}
_SEQUENTIAL_REPLACEME_PREFIX_CACHE: Dict[
    Tuple[Any, ...], Dict[Tuple[Tuple[Any, ...], ...], List[Tuple[torch.Tensor, Dict[str, Any]]]]
] = {}


def _calibration_manifest_hash(scenes: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    seen: Set[str] = set()
    for item in scenes:
        cache_dir = str(item.get("_cache_dir", ""))
        if cache_dir not in seen:
            manifest_path = Path(cache_dir) / "cache_manifest_v3.json"
            if manifest_path.is_file():
                digest.update(manifest_path.read_bytes())
            seen.add(cache_dir)
        digest.update(str(item.get("scene_id", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


_BASE_CHECKPOINT_HASH_CACHE: Dict[str, str] = {}


def _base_checkpoint_hash(hf_model: str) -> str:
    """Hash local checkpoints exactly; hash immutable model identifiers otherwise."""
    value = str(hf_model)
    if value in _BASE_CHECKPOINT_HASH_CACHE:
        return _BASE_CHECKPOINT_HASH_CACHE[value]
    path = Path(value)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(value.encode("utf-8"))
    result = digest.hexdigest()
    _BASE_CHECKPOINT_HASH_CACHE[value] = result
    return result


def _boundary_signature(raw: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(raw.get("source_unit_id", "")),
        str(raw.get("target_unit_id", "")),
        int(raw.get("drop_start_index", -1)),
        int(raw.get("drop_end_index", -1)),
    )


def _has_single_drop_interval(prune_plan: UnitPrunePlan) -> bool:
    dropped = sorted(set(int(index) for index in prune_plan.drop))
    return bool(dropped) and dropped == list(range(dropped[0], dropped[-1] + 1))


def _replace_with_relative_symlink(target: Path, source: Path) -> None:
    target = Path(target)
    source = Path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    relative_source = os.path.relpath(source, start=target.parent)
    target.symlink_to(relative_source, target_is_directory=source.is_dir())


def _try_link_derived_ghost_recovery(
    *,
    method: str,
    prune_plan: UnitPrunePlan,
    recovery_plan_name: str,
    out_dir: Path,
    plan_root: Path,
    backbone: str,
    selector: str,
) -> bool:
    """Reuse ordinary Ghost artifacts when a derived method is identical."""
    if method not in _DERIVED_GHOST_METHODS or not _has_single_drop_interval(prune_plan):
        return False
    source_dir = Path(out_dir).parent / "recovery_ghost"
    if not recovery_result_complete(
        source_dir,
        plan_root=Path(plan_root),
        backbone=backbone,
        selector=selector,
        recovery_plan_name="recovery_ghost",
    ):
        return False
    _replace_with_relative_symlink(Path(out_dir), source_dir)
    print(
        "\n  recovery reuse\n"
        f"    method: {recovery_plan_name.removeprefix('recovery_')}\n"
        "    source: ghost\n"
        "    reason: single contiguous drop interval\n"
        f"    link: {out_dir} -> {source_dir}\n",
        file=sys.stderr,
        flush=True,
    )
    return True


def _scenes_have_cached_units(
    scenes: Sequence[Dict[str, Any]], unit_ids: Sequence[str]
) -> bool:
    """Return whether every calibration scene has all requested unit files."""
    if not scenes:
        return False
    requested = [str(unit_id) for unit_id in unit_ids]
    for item in scenes:
        layer_files = item.get("layer_files")
        if not isinstance(layer_files, dict):
            return False
        cache_dir = Path(str(item.get("_cache_dir", "")))
        for unit_id in requested:
            relative = str(layer_files.get(unit_id, "")).strip()
            if not relative or not (cache_dir / relative).is_file():
                return False
    return True


def _collect_current_replaceme_rows(
    *,
    model: Any,
    adapter: Any,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
) -> ShardedRowStore:
    """Capture one boundary into a temporary file-backed row store."""
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    scratch_root = Path(tempfile.mkdtemp(prefix="lare_replaceme_"))
    # Existing cache file sizes provide a conservative, model-independent
    # estimate for three dynamic activation streams plus serialization headers.
    estimated = 0
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        layer_files = item.get("layer_files") or {}
        auxiliary_files = item.get("auxiliary_files") or {}
        for key in (source_unit_id, target_unit_id):
            rel = str(layer_files.get(key, ""))
            path = cache_dir / rel if rel else Path()
            if rel and path.is_file():
                estimated += path.stat().st_size
        rel = str(auxiliary_files.get(auxiliary_id, ""))
        path = cache_dir / rel if rel else Path()
        if rel and path.is_file():
            estimated += path.stat().st_size
    estimated = max(estimated + 16 * 1024 * 1024, 64 * 1024 * 1024)
    free = shutil.disk_usage(scratch_root).free
    if free < estimated:
        shutil.rmtree(scratch_root)
        raise OSError(
            f"Sequential ReplaceMe needs about {estimated} scratch bytes, only {free} are free"
        )

    shards: List[RowShard] = []
    try:
        for scene_index, item in enumerate(scenes):
            image_paths = [str(path) for path in item.get("image_paths", []) if str(path)]
            if not image_paths:
                raise FileNotFoundError(
                    f"Progressive recovery requires image_paths for {item.get('scene_id')}"
                )
            prepared = prepare_cache_batch(
                None,
                max_frames=0,
                meta={
                    "image_paths": image_paths,
                    "frame_ids": list(item.get("frame_ids") or []),
                },
                backbone_adapter=adapter,
            )
            images = prepared.images.to(device=device, non_blocking=False)
            unit_map, _selected_meta, auxiliary_map = forward_selected_units_with_auxiliaries(
                adapter=adapter,
                model=model,
                images=images,
                device=device,
                unit_ids={source_unit_id, target_unit_id},
                auxiliary_ids={auxiliary_id},
            )
            h_raw, y_raw, r_raw = _align_replaceme_activations(
                unit_map[source_unit_id],
                unit_map[target_unit_id],
                auxiliary_map[auxiliary_id],
                scene_id=str(item.get("scene_id", "<unknown>")),
                source_unit_id=source_unit_id,
            )
            paths = []
            for name, tensor in (("h", h_raw), ("y", y_raw), ("r", r_raw)):
                path = scratch_root / f"{scene_index:06d}_{name}.pt"
                torch.save(tensor.detach().cpu().contiguous(), path)
                paths.append(path)
            h_disk, y_disk, r_disk = (
                torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                for path in paths
            )
            h, y, r = rows_view(h_disk), rows_view(y_disk), rows_view(r_disk)

            def desired_builder(
                indices: torch.Tensor,
                *,
                h_rows: torch.Tensor = h,
                y_rows: torch.Tensor = y,
                r_rows: torch.Tensor = r,
            ) -> torch.Tensor:
                return y_rows[indices] + r_rows[indices] - h_rows[indices]

            shards.append(
                RowShard(
                    scene_id=str(item.get("scene_id", scene_index)),
                    inputs=r,
                    desired_builder=desired_builder,
                )
            )
            del images, unit_map, auxiliary_map, h_raw, y_raw, r_raw
        store = RecoveryActivationStore(shards)
        setattr(store, "_scratch_root", scratch_root)
        return store
    except Exception:
        shards.clear()
        shutil.rmtree(scratch_root, ignore_errors=True)
        raise


def _close_dynamic_row_store(store: ShardedRowStore) -> None:
    scratch_root = getattr(store, "_scratch_root", None)
    store.close()
    if scratch_root is not None:
        shutil.rmtree(Path(scratch_root), ignore_errors=True)


def _fit_ghost_interval_sequential(
    *,
    adapter: Any,
    hf_model: str,
    records: Sequence[Dict[str, Any]],
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
    ridge: float,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
    method: str = "recovery_ghost_interval_sequential",
    num_layers: int = 0,
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Fit Ghost operators front-to-back on the evolving model."""
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(device),
        dtype=None,
        eval_mode=True,
    )
    results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    handles: List[Any] = []
    ordered_signatures = tuple(_boundary_signature(dict(raw)) for raw in records)
    prefix_base_key = (
        str(getattr(adapter, "name", type(adapter).__name__)),
        _base_checkpoint_hash(str(hf_model or adapter.default_hf_model())),
        _calibration_manifest_hash(scenes), "default", str(method), float(ridge),
        int(chunk_rows),
    )
    prefix_entries = _SEQUENTIAL_GHOST_PREFIX_CACHE.setdefault(prefix_base_key, {})
    reusable: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    for signature, candidate in prefix_entries.items():
        if (
            len(signature) > len(reusable)
            and len(signature) <= len(ordered_signatures)
            and signature == ordered_signatures[: len(signature)]
        ):
            reusable = candidate
    for boundary_index, raw in enumerate(records):
        record = dict(raw)
        source_unit_id = str(record["source_unit_id"])
        target_unit_id = str(record["target_unit_id"])
        print(
            "[recovery-fit]"
            f"[boundary={boundary_index + 1}/{len(records)}] "
            f"method={method.removeprefix('recovery_')} "
            f"source={source_unit_id} target={target_unit_id} scenes={len(scenes)}",
            file=sys.stderr,
            flush=True,
        )
        if boundary_index < len(reusable):
            weight = reusable[boundary_index][0].clone()
            diagnostics = dict(reusable[boundary_index][1])
        elif boundary_index == 0 and _scenes_have_cached_units(
            scenes, (source_unit_id, target_unit_id)
        ):
            print(
                "[recovery-fit]"
                f"[boundary=1/{len(records)}] source=disk_cache",
                file=sys.stderr,
                flush=True,
            )
            weight = _fit_boundary_weight(
                method="recovery_ghost",
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=int(num_layers),
                device=device,
                ridge=ridge,
                chunk_rows=chunk_rows,
                boundary_index=boundary_index,
                boundary_total=len(records),
            )
            diagnostics = {
                "fit_mode": "disk_cache",
                "num_scenes": len(scenes),
                "ridge": float(ridge),
                "chunk_rows": int(chunk_rows),
            }
        else:
            weight, diagnostics = _fit_current_ghost_boundary(
                model=model,
                adapter=adapter,
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                device=device,
                ridge=ridge,
                chunk_rows=chunk_rows,
                boundary_index=boundary_index,
                boundary_total=len(records),
            )
        diagnostics["boundary_index"] = boundary_index
        module_name = recovery_module_name(source_unit_id)
        spec = {
            "type": str(method),
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "module_name": module_name,
            "weight_shape": [int(x) for x in weight.shape],
        }
        actions = [
            PruneAction(action="drop_unit", unit_id=unit_index_to_unit_id(index))
            for index in range(
                int(record["drop_start_index"]), int(record["drop_end_index"]) + 1
            )
        ]
        model = adapter.apply_prune_plan(model, actions)
        # Install after pruning.  This is equivalent for retained source units
        # and is required for Omega register-global boundaries, whose full-token
        # hook lives on the newly installed identity consumer.
        handles.extend(
            install_recovery_modules(model, adapter, [spec], device=device)
        )
        apply_recovery_weights(
            model,
            adapter,
            [spec],
            [
                {
                    "source_unit_id": source_unit_id,
                    "module_name": module_name,
                    "weight": weight,
                }
            ],
            device=device,
        )
        results.append((weight, diagnostics))
        prefix_entries[ordered_signatures[: boundary_index + 1]] = [
            (saved_weight.clone(), dict(saved_diagnostics))
            for saved_weight, saved_diagnostics in results
        ]
    for handle in handles:
        handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _fit_replaceme_cos_sequential(
    *,
    adapter: Any,
    hf_model: str,
    records: Sequence[Dict[str, Any]],
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Fit Multi-LT cosine transforms front-to-back on the evolving model."""
    if _ACTIVE_RECOVERY_SESSION is not None:
        _ACTIVE_RECOVERY_SESSION.clear_units()
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(device),
        dtype=None,
        eval_mode=True,
    )
    results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    ordered_signatures = tuple(_boundary_signature(dict(raw)) for raw in records)
    prefix_base_key = (
        str(getattr(adapter, "name", type(adapter).__name__)),
        _base_checkpoint_hash(str(hf_model or adapter.default_hf_model())),
        _calibration_manifest_hash(scenes),
        "default",
        "recovery_replaceme_cos_sequential",
        int(epochs), int(batch_size), float(lr), int(seed),
    )
    prefix_entries = _SEQUENTIAL_REPLACEME_PREFIX_CACHE.setdefault(prefix_base_key, {})
    reusable: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    for signature, candidate in prefix_entries.items():
        if (
            len(signature) > len(reusable)
            and len(signature) <= len(ordered_signatures)
            and signature == ordered_signatures[: len(signature)]
        ):
            reusable = candidate
    for boundary_index, raw in enumerate(records):
        record = dict(raw)
        source_unit_id = str(record["source_unit_id"])
        target_unit_id = str(record["target_unit_id"])
        print(
            "[recovery-fit]"
            f"[boundary={boundary_index + 1}/{len(records)}] "
            "method=replaceme_cos_sequential "
            f"source={source_unit_id} target={target_unit_id} scenes={len(scenes)}",
            file=sys.stderr,
            flush=True,
        )
        if boundary_index < len(reusable):
            weight = reusable[boundary_index][0].clone()
            diagnostics = dict(reusable[boundary_index][1])
        else:
            collected_rows = _collect_current_replaceme_rows(
                model=model,
                adapter=adapter,
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                device=device,
            )
            # Tuple support keeps the internal test/fake-adapter contract compatible
            # with the pre-file-backed collector.
            row_store = collected_rows if isinstance(collected_rows, ShardedRowStore) else None
            legacy_input, legacy_desired = (
                collected_rows if row_store is None else (None, None)
            )
            try:
                weight, diagnostics = _optimize_cosine_transform(
                    row_store=row_store,
                    input_rows=legacy_input,
                    desired_rows=legacy_desired,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=int(seed),
                    method_name="ReplaceMe cosine sequential",
                )
            finally:
                if row_store is not None:
                    _close_dynamic_row_store(row_store)
        diagnostics.update(
            {
                "fit_mode": "sequential_current_model",
                "boundary_index": boundary_index,
            }
        )
        fuse_replaceme_transform(model, adapter, source_unit_id, weight.to(device))
        actions = [
            PruneAction(action="drop_unit", unit_id=unit_index_to_unit_id(index))
            for index in range(
                int(record["drop_start_index"]), int(record["drop_end_index"]) + 1
            )
        ]
        model = adapter.apply_prune_plan(model, actions)
        results.append((weight.cpu(), diagnostics))
        prefix_entries[ordered_signatures[: boundary_index + 1]] = [
            (saved_weight.clone(), dict(saved_diagnostics))
            for saved_weight, saved_diagnostics in results
        ]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _resource_snapshot(device: torch.device) -> Dict[str, int]:
    values = {"rss_bytes": 0, "anonymous_bytes": 0, "mapped_bytes": 0}
    try:
        status = Path("/proc/self/smaps_rollup").read_text(encoding="utf-8")
        parsed: Dict[str, int] = {}
        for line in status.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].rstrip(":") in {"Rss", "Anonymous"}:
                parsed[fields[0].rstrip(":")] = int(fields[1]) * 1024
        values["rss_bytes"] = parsed.get("Rss", 0)
        values["anonymous_bytes"] = parsed.get("Anonymous", 0)
        values["mapped_bytes"] = max(
            0, values["rss_bytes"] - values["anonymous_bytes"]
        )
    except OSError:
        pass
    if device.type == "cuda":
        values["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        values["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
    return values


def _compact_resource_log(
    *, method: str, boundary: int, total: int, device: torch.device
) -> None:
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
        return
    stats = _resource_snapshot(device)
    print(
        "[recovery-resource] "
        f"method={method} boundary={boundary + 1}/{total} "
        f"mapped={stats.get('mapped_bytes', 0)} anon={stats.get('anonymous_bytes', 0)} "
        f"cuda_alloc={stats.get('cuda_allocated_bytes', 0)} "
        f"cuda_reserved={stats.get('cuda_reserved_bytes', 0)}",
        file=sys.stderr,
        flush=True,
    )


def _run_recovery_request(args: argparse.Namespace) -> None:
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    chunk_rows = int(args.recovery_chunk_rows)
    if chunk_rows <= 0:
        raise SystemExit("--recovery-chunk-rows must be greater than zero")
    plan_root = Path(args.prune_plan_root)
    recovery_root = Path(args.recovery_plan_root or args.prune_plan_root)
    recovery_plan_name = str(args.recovery_plan_name).strip()
    prune_plan_name = str(args.prune_plan_name).strip()
    prune_plan = compose_recovery_unit_prune_plan(
        prune_plan_root=plan_root,
        recovery_plan_root=recovery_root,
        prune_selector=prune_plan_name,
        recovery_plan=recovery_plan_name,
    )
    bundle = load_recovery_plan_bundle(recovery_root, recovery_plan_name)
    method, pair_records = _pair_records(bundle, prune_plan, recovery_plan_name)
    default_output_root = Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT_DEFAULT)))
    out_dir = (
        args.output_dir
        or default_output_root
        / adapter.name
        / "recovery"
        / prune_plan_name
        / recovery_plan_name
    )
    if _try_link_derived_ghost_recovery(
        method=method,
        prune_plan=prune_plan,
        recovery_plan_name=recovery_plan_name,
        out_dir=out_dir,
        plan_root=plan_root,
        backbone=adapter.name,
        selector=prune_plan_name,
    ):
        return
    dataset_names = _parse_name_list(args.datasets or args.dataset)
    cache_dirs = [stage_dir(args.experiment, name, "cache") for name in dataset_names]
    scenes = _load_training_scenes(cache_dirs, int(args.max_scenes))
    if not scenes:
        raise SystemExit("No scenes selected from cache")
    first_manifest = load_v3_manifest(cache_dirs[0])
    num_layers = max((int(u.index) for u in first_manifest.layer_units if u.kind in ("frame", "global")), default=-1) + 1
    if int(prune_plan.num_units) != int(num_layers * 2):
        raise SystemExit(f"prune plan num_units={prune_plan.num_units} != cache num_units={num_layers * 2}")

    job_index = int(getattr(args, "_job_index", 1))
    job_total = int(getattr(args, "_job_total", 1))
    print(
        "\n  recovery fit\n"
        f"    job: {job_index}/{job_total}\n"
        f"    model: {adapter.name}\n"
        f"    span: {prune_plan_name.removeprefix('prune_span')}\n"
        f"    method: {method.removeprefix('recovery_')}\n"
        f"    datasets: {','.join(dataset_names)}\n"
        f"    scenes: {len(scenes)}\n"
        f"    boundaries: {len(pair_records)}\n",
        file=sys.stderr,
        flush=True,
    )

    weights: List[torch.Tensor] = []
    recovery_specs: List[Dict[str, Any]] = []
    fit_diagnostics: List[Dict[str, Any]] = []
    sequential_replaceme_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    sequential_ghost_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    if method == "recovery_ghost_interval_sequential" and len(pair_records) > 1:
        sequential_ghost_results = _fit_ghost_interval_sequential(
            adapter=adapter,
            hf_model=str(args.hf_model),
            records=pair_records,
            scenes=scenes,
            device=device,
            ridge=float(args.ridge),
            chunk_rows=chunk_rows,
            method=method,
            num_layers=num_layers,
        )
    if method == "recovery_replaceme_cos" and len(pair_records) > 1:
        sequential_replaceme_results = _fit_replaceme_cos_sequential(
            adapter=adapter,
            hf_model=str(args.hf_model),
            records=pair_records,
            scenes=scenes,
            device=device,
            epochs=int(args.replaceme_cos_epochs),
            batch_size=int(args.replaceme_cos_batch_size),
            lr=float(args.replaceme_cos_lr),
            seed=int(args.replaceme_seed),
        )
    for record_index, record in enumerate(pair_records):
        _compact_resource_log(
            method=method,
            boundary=record_index,
            total=len(pair_records),
            device=device,
        )
        source_unit_id = str(record.get("source_unit_id", ""))
        target_unit_id = str(record.get("target_unit_id", ""))
        diagnostics: Dict[str, Any] = {}
        if sequential_ghost_results is None and sequential_replaceme_results is None:
            print(
                "[recovery-fit]"
                f"[boundary={record_index + 1}/{len(pair_records)}] "
                f"method={method.removeprefix('recovery_')} "
                f"source={source_unit_id} target={target_unit_id}",
                file=sys.stderr,
                flush=True,
            )
        if sequential_ghost_results is not None:
            weight, diagnostics = sequential_ghost_results[record_index]
        elif sequential_replaceme_results is not None:
            weight, diagnostics = sequential_replaceme_results[record_index]
        elif method in (
            "recovery_ghost",
            "recovery_ghost_interval",
            "recovery_ghost_interval_sequential",
            "recovery_linearpatch",
        ):
            weight = _fit_boundary_weight(
                method=(
                    "recovery_linearpatch"
                    if method == "recovery_linearpatch"
                    else "recovery_ghost"
                ),
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                ridge=float(args.ridge),
                chunk_rows=chunk_rows,
                boundary_index=record_index,
                boundary_total=len(pair_records),
            )
        elif method == "recovery_replaceme_ls":
            weight, diagnostics = _fit_replaceme_ls(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                alpha=float(args.replaceme_ls_alpha),
                chunk_rows=chunk_rows,
                boundary_index=record_index,
                boundary_total=len(pair_records),
            )
        elif method == "recovery_replaceme_cos":
            weight, diagnostics = _fit_replaceme_cos(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                epochs=int(args.replaceme_cos_epochs),
                batch_size=int(args.replaceme_cos_batch_size),
                lr=float(args.replaceme_cos_lr),
                seed=int(args.replaceme_seed),
                boundary_index=record_index,
                boundary_total=len(pair_records),
            )
        else:
            raise ValueError(f"Unsupported training-free recovery method {method!r}")
        weights.append(weight)
        fit_diagnostics.append(diagnostics)
        recovery_specs.append(
            {
                "type": method,
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "source_unit_index": int(record.get("source_unit_index")),
                "target_unit_index": int(record.get("target_unit_index")),
                "module_name": recovery_module_name(source_unit_id),
                "weight_shape": [int(x) for x in weight.shape],
            }
        )

    recovery_specs, compact_entries = compact_recovery_operators(
        recovery_specs,
        [
            {
                "source_unit_id": str(spec["source_unit_id"]),
                "module_name": str(spec["module_name"]),
                "weight": weight,
            }
            for spec, weight in zip(recovery_specs, weights)
        ],
    )
    weights = [entry["weight"] for entry in compact_entries]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    summary = {
        "backbone": adapter.name,
        "method": method,
        "prune_plan_name": prune_plan_name,
        "recovery_plan_name": recovery_plan_name,
        "num_scenes": len(scenes),
        "cache_dirs": [str(p) for p in cache_dirs],
        "recovery_specs": recovery_specs,
        "fit_diagnostics": fit_diagnostics,
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hf_model = str(args.hf_model or adapter.default_hf_model())
    def checkpoint_payload(
        checkpoint_weights: Sequence[torch.Tensor],
        *,
        epoch: int,
        variant: str,
    ) -> Dict[str, Any]:
        return {
            "global_step": 0,
            "epoch": int(epoch),
            "epochs": int(epoch),
            "drop_indices": sorted(set(int(x) for x in prune_plan.drop)),
            "lora_indices": [],
            "train_unit_indices": [],
            "training_method": method,
            "recovery_method": method,
            "recovery_specs": recovery_specs,
            "recovery_weights": [
                {
                    "source_unit_id": str(spec["source_unit_id"]),
                    "module_name": str(spec["module_name"]),
                    "weight": weight.detach().cpu().to(torch.float32),
                }
                for spec, weight in zip(recovery_specs, checkpoint_weights)
            ],
            "prune_plan": asdict(prune_plan),
            "variant": variant,
            "base_hf_model": hf_model,
            "optimizer_state_dict": {},
            "args": vars(args),
        }

    atomic_save_checkpoint(
        checkpoint_payload(
            weights,
            epoch=int(
                fit_diagnostics[0].get("selected_epoch", 0)
                if fit_diagnostics
                else 0
            ),
            variant="training_free_recovery_delta",
        ),
        out_dir / "checkpoints" / "latest.pt",
    )
    print(f"[recovery] wrote {out_dir / 'checkpoints' / 'latest.pt'}", flush=True)


def main() -> None:
    global _ACTIVE_RECOVERY_SESSION
    args = _build_parser().parse_args()
    if args.jobs_manifest is None:
        _run_recovery_request(args)
        return
    payload = json.loads(Path(args.jobs_manifest).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--jobs-manifest must contain a non-empty JSON list")
    _ACTIVE_RECOVERY_SESSION = RecoverySession(str(args.backbone))
    try:
        job_total = len(payload)
        for index, raw in enumerate(payload):
            if not isinstance(raw, dict):
                raise ValueError(f"Recovery job {index} is not an object")
            job_args = copy.copy(args)
            for field in ("prune_plan_name", "recovery_plan_name", "output_dir"):
                if field not in raw:
                    raise ValueError(f"Recovery job {index} is missing {field}")
                value = raw[field]
                if field == "output_dir":
                    value = Path(str(value))
                setattr(job_args, field, value)
            setattr(job_args, "_job_index", index + 1)
            setattr(job_args, "_job_total", job_total)
            _run_recovery_request(job_args)
    finally:
        _ACTIVE_RECOVERY_SESSION.close()
        _ACTIVE_RECOVERY_SESSION = None


if __name__ == "__main__":
    main()

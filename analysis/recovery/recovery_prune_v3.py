"""Cached-boundary recovery fitting for pruned LaRe models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import copy
import hashlib
import math
from dataclasses import asdict, dataclass
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
    StreamlineFFN,
    apply_recovery_weights,
    compact_recovery_operators,
    fuse_replaceme_transform,
    install_recovery_modules,
    pack_streamline_ffn_parameters,
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
    CACHE_FIT_RECOVERY_METHODS,
    TOKEN_GROUPED_RECOVERY_BASE_METHODS,
    compose_recovery_unit_prune_plan,
    load_recovery_plan_bundle,
    recovery_method_semantics_version,
)


DEFAULT_RECOVERY_CHUNK_ROWS = 16_384
DEFAULT_GHOST_AUTOGROUP_THRESHOLD = 20.0
_DERIVED_GHOST_METHODS = {
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}
_SPECIAL_PATCH_GROUPED_GHOST_METHODS = frozenset(
    {
        "recovery_ghost_interval_sequential_grouped",
        "recovery_ghost_interval_sequential_autogroup",
    }
)
_TOKEN_GROUPED_GHOST_METHODS = _SPECIAL_PATCH_GROUPED_GHOST_METHODS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fit cached-boundary recovery operators for a pruned model.")
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
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional shared output root containing <backbone>/<dataset>/cache.",
    )
    p.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Maximum calibration scenes per cache dataset; 0 means all scenes.",
    )
    p.add_argument(
        "--scene-limits",
        default="",
        help=(
            "Optional exact per-dataset quotas as tag=count pairs separated by commas. "
            "When set, every selected dataset must appear exactly once."
        ),
    )
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument(
        "--ghost-stability-profile",
        choices=("legacy", "hybrid", "trace_identity", "trace_all", "native_all"),
        default="legacy",
    )
    p.add_argument("--ghost-special-ridge", type=float, default=0.0)
    p.add_argument("--ghost-camera-ridge", type=float, default=0.0)
    p.add_argument("--ghost-register-ridge", type=float, default=0.0)
    p.add_argument(
        "--ghost-coupling-profile",
        choices=("none", "shared_residual", "patch_mean", "shared_residual_patch_mean"),
        default="none",
    )
    p.add_argument("--ghost-residual-ridge", type=float, default=0.0)
    p.add_argument("--ghost-context-ridge", type=float, default=0.0)
    p.add_argument(
        "--ghost-autogroup-threshold",
        type=float,
        default=DEFAULT_GHOST_AUTOGROUP_THRESHOLD,
        help=(
            "Use grouped special/patch matrices only when "
            "||Ws-Wall||F/||Wall-I||F is below this value. Used only by "
            "recovery_ghost_interval_sequential_autogroup."
        ),
    )
    p.add_argument(
        "--ghost-boundary-fit-mode",
        choices=(
            "sequential_current_target",
            "sequential_full_target",
            "independent_full_target",
        ),
        default="sequential_current_target",
        help=(
            "How later interval Ghost boundaries are fit. The default preserves "
            "the current sequential source/current target behavior."
        ),
    )
    p.add_argument(
        "--reuse-boundary-prefix-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional compatible recovery checkpoint whose leading boundary "
            "weights are reused instead of refit."
        ),
    )
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
    p.add_argument("--streamline-ffn-epochs", type=int, default=1)
    p.add_argument("--streamline-ffn-batch-size", type=int, default=16384)
    p.add_argument("--streamline-ffn-lr", type=float, default=2e-4)
    p.add_argument("--streamline-ffn-min-lr", type=float, default=5e-6)
    p.add_argument("--streamline-ffn-weight-decay", type=float, default=1e-3)
    p.add_argument("--streamline-ffn-expansion", type=int, default=4)
    p.add_argument("--streamline-ffn-seed", type=int, default=0)
    p.add_argument(
        "--jobs-manifest",
        type=Path,
        default=None,
        help="JSON list of prune/recovery/output requests handled in one process.",
    )
    return p


def _parse_name_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").replace(",", " ").split() if x.strip()]


@dataclass(frozen=True)
class GhostStabilityConfig:
    profile: str = "legacy"
    special_ridge: float = 0.0
    camera_ridge: float = 0.0
    register_ridge: float = 0.0
    coupling_profile: str = "none"
    residual_ridge: float = 0.0
    context_ridge: float = 0.0

    def validate(self) -> "GhostStabilityConfig":
        if self.profile not in {
            "legacy", "hybrid", "trace_identity", "trace_all", "native_all"
        }:
            raise ValueError(f"Unsupported Ghost stability profile: {self.profile!r}")
        if self.coupling_profile not in {
            "none", "shared_residual", "patch_mean", "shared_residual_patch_mean"
        }:
            raise ValueError(f"Unsupported Ghost coupling profile: {self.coupling_profile!r}")
        for name, value in (
            ("special_ridge", self.special_ridge),
            ("camera_ridge", self.camera_ridge),
            ("register_ridge", self.register_ridge),
            ("residual_ridge", self.residual_ridge),
            ("context_ridge", self.context_ridge),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        return self

    def signature_payload(self) -> Dict[str, Any]:
        return asdict(self.validate())


def _ghost_stability_config(args: argparse.Namespace) -> GhostStabilityConfig:
    return GhostStabilityConfig(
        profile=str(args.ghost_stability_profile),
        special_ridge=float(args.ghost_special_ridge),
        camera_ridge=float(args.ghost_camera_ridge),
        register_ridge=float(args.ghost_register_ridge),
        coupling_profile=str(args.ghost_coupling_profile),
        residual_ridge=float(args.ghost_residual_ridge),
        context_ridge=float(args.ghost_context_ridge),
    ).validate()


_SESSION_SCENES: Dict[
    Tuple[Tuple[str, ...], int, Tuple[int, ...]], List[Dict[str, Any]]
] = {}


def _parse_scene_limits(raw: str, dataset_names: Sequence[str]) -> Dict[str, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    limits: Dict[str, int] = {}
    for entry in text.replace(" ", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry.count("=") != 1:
            raise ValueError(
                f"Invalid scene-limit entry {entry!r}; expected dataset=count"
            )
        dataset, value = (part.strip() for part in entry.split("=", 1))
        if not dataset or dataset in limits:
            raise ValueError(f"Duplicate or empty scene-limit dataset: {dataset!r}")
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid scene limit for {dataset!r}: {value!r}") from exc
        if count <= 0:
            raise ValueError(
                f"Scene limit for {dataset!r} must be a positive integer, got {count}"
            )
        limits[dataset] = count
    expected = list(dict.fromkeys(str(name) for name in dataset_names))
    missing = [name for name in expected if name not in limits]
    extra = [name for name in limits if name not in expected]
    if missing or extra:
        raise ValueError(
            "Scene-limit datasets do not match selected datasets: "
            f"missing={missing}, extra={extra}"
        )
    return limits


def _load_training_scenes(
    cache_dirs: Sequence[Path],
    max_scenes: int,
    exact_limits: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    if exact_limits is not None and len(exact_limits) != len(cache_dirs):
        raise ValueError(
            f"exact_limits length {len(exact_limits)} != cache_dirs length {len(cache_dirs)}"
        )
    limit_key = tuple(int(value) for value in (exact_limits or ()))
    session_key = (
        tuple(str(Path(path).resolve()) for path in cache_dirs),
        int(max_scenes),
        limit_key,
    )
    cached = _SESSION_SCENES.get(session_key)
    if cached is not None:
        return cached
    scenes: List[Dict[str, Any]] = []
    for cache_index, cache_dir in enumerate(cache_dirs):
        manifest = load_v3_manifest(cache_dir)
        dataset_scenes = manifest_scenes_for_training(manifest)
        exact_limit = int(exact_limits[cache_index]) if exact_limits is not None else 0
        if exact_limit > 0:
            if len(dataset_scenes) < exact_limit:
                raise ValueError(
                    f"Cache {cache_dir} contains {len(dataset_scenes)} scenes, "
                    f"fewer than its requested exact quota {exact_limit}"
                )
            dataset_scenes = dataset_scenes[:exact_limit]
        elif int(max_scenes) > 0:
            dataset_scenes = dataset_scenes[: int(max_scenes)]
        for item in dataset_scenes:
            item = dict(item)
            # The legacy training view intentionally contains only tensor-file
            # references. Progressive recovery additionally needs the exact
            # frame ids/paths and geometry saved in per-scene metadata.
            item.update(load_scene_metadata_v3(cache_dir, item))
            item["_cache_dir"] = str(cache_dir)
            scenes.append(item)
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


def _validate_ghost_token_layout(
    token_group_names: Sequence[str], token_group_starts: Sequence[int]
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    names = tuple(str(name).strip() for name in token_group_names)
    starts = tuple(int(start) for start in token_group_starts)
    if not names or any(not name for name in names):
        raise ValueError(f"Ghost token-group names must be non-empty: {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"Ghost token-group names must be unique: {names}")
    if len(names) != len(starts):
        raise ValueError(
            f"Ghost token groups and starts disagree: {len(names)} != {len(starts)}"
        )
    if starts[0] != 0 or any(right <= left for left, right in zip(starts, starts[1:])):
        raise ValueError(f"Ghost token-group starts must begin at 0 and increase: {starts}")
    return names, starts


def _merge_special_ghost_token_groups(
    token_group_names: Sequence[str], token_group_starts: Sequence[int]
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Merge camera/register groups into one leading special-token group."""
    names, starts = _validate_ghost_token_layout(
        token_group_names, token_group_starts
    )
    merged_names: List[str] = []
    merged_starts: List[int] = []
    for name, start in zip(names, starts):
        merged_name = "special" if name in {"camera", "register", "special"} else name
        if merged_names and merged_names[-1] == merged_name:
            continue
        if merged_name in merged_names:
            raise ValueError(
                "Ghost special groups must be contiguous: "
                f"groups={names}, starts={starts}"
            )
        merged_names.append(merged_name)
        merged_starts.append(start)
    return _validate_ghost_token_layout(merged_names, merged_starts)


def _ghost_token_layout_for_method(
    *, adapter: Any, method: str, patch_start_idx: int
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    names, starts = _validate_ghost_token_layout(
        *adapter.ghost_token_group_layout(patch_start_idx=int(patch_start_idx))
    )
    if str(method) in _SPECIAL_PATCH_GROUPED_GHOST_METHODS:
        return _merge_special_ghost_token_groups(names, starts)
    return names, starts


def _ghost_token_group_ids(
    tensor: torch.Tensor,
    *,
    token_group_names: Sequence[str],
    token_group_starts: Sequence[int],
) -> Optional[torch.Tensor]:
    """Return flattened token-group ids aligned with a public ``[B,S,P,C]`` tensor."""
    names, starts = _validate_ghost_token_layout(token_group_names, token_group_starts)
    if len(names) == 1:
        return None
    if tensor.dim() != 4:
        raise ValueError(f"Token grouping expects [B,S,P,C], got {tuple(tensor.shape)}")
    bsz, frames, tokens, _channels = (int(value) for value in tensor.shape)
    if starts[-1] >= tokens:
        raise ValueError(
            f"Ghost token-group start {starts[-1]} is outside a {tokens}-token frame"
        )
    per_frame = torch.empty(tokens, dtype=torch.int16)
    for group_index, start in enumerate(starts):
        end = starts[group_index + 1] if group_index + 1 < len(starts) else tokens
        per_frame[start:end] = group_index
    return per_frame.repeat(bsz * frames).to(dtype=torch.long)


def _scene_patch_start(item: Dict[str, Any]) -> int:
    if "patch_start_idx" not in item:
        raise KeyError(
            f"Scene {item.get('scene_id', '<unknown>')} lacks patch_start_idx; "
            "rebuild its recovery cache before token-grouped fitting"
        )
    return int(item["patch_start_idx"])


def _common_patch_start(scenes: Sequence[Dict[str, Any]]) -> int:
    values = {_scene_patch_start(item) for item in scenes}
    if len(values) != 1:
        raise ValueError(f"Recovery scenes disagree on patch_start_idx: {sorted(values)}")
    return next(iter(values))


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
    if method not in CACHE_FIT_RECOVERY_METHODS:
        raise ValueError(f"{recovery_plan_name} is not cache-fit recovery: method={method!r}")
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
    ghost_token_group_names: Sequence[str] = ("all",),
    ghost_token_group_starts: Sequence[int] = (0,),
    ghost_stability: Optional[GhostStabilityConfig] = None,
    ghost_autogroup_threshold: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Any:
    stability = (ghost_stability or GhostStabilityConfig()).validate()
    token_group_names, token_group_starts = _validate_ghost_token_layout(
        ghost_token_group_names, ghost_token_group_starts
    )
    if method != "recovery_ghost" and len(token_group_names) != 1:
        raise ValueError(f"Token-group matrices are only supported by Ghost, not {method}")
    cache_key: Tuple[Any, ...] = (
        str(method), str(source_unit_id), str(target_unit_id), int(num_layers),
        str(device), float(ridge), int(chunk_rows), token_group_names, token_group_starts,
        tuple(sorted(stability.signature_payload().items())),
        ghost_autogroup_threshold,
        tuple(
            (str(item.get("_cache_dir", "")), str(item.get("scene_id", "")))
            for item in scenes
        ),
    )
    cached_weight = _CLOSED_FORM_SESSION_CACHE.get(cache_key)
    if cached_weight is not None:
        cached_result = cached_weight.clone()
        if return_diagnostics:
            return cached_result, {"cache_hit": True, "stability": stability.signature_payload()}
        return cached_result
    if method == "recovery_ghost" and stability.coupling_profile != "none":
        coupled_stats: Optional[Dict[str, torch.Tensor]] = None
        patch_start_idx = _common_patch_start(scenes)
        for scene_index, item in enumerate(scenes, start=1):
            cache_dir = Path(str(item["_cache_dir"]))
            source = _load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu")
            target = _load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu")
            coupled_stats = _accumulate_coupled_statistics(
                coupled_stats, source, target,
                patch_start_idx=patch_start_idx, device=device,
            )
            del source, target
            if scene_index == len(scenes) or scene_index % 10 == 0:
                print(
                    f"[recovery-fit][boundary={boundary_index + 1}/{boundary_total}]"
                    f"[scenes={scene_index}/{len(scenes)}] method=ghost_coupled",
                    file=sys.stderr, flush=True,
                )
        assert coupled_stats is not None
        result, diagnostics = _solve_coupled_ghost_statistics(
            coupled_stats, legacy_ridge=float(ridge), stability=stability
        )
        diagnostics.update({"token_groups": ["special", "patch"], "token_group_starts": [0, patch_start_idx]})
        _CLOSED_FORM_SESSION_CACHE[cache_key] = result.clone()
        return (result, diagnostics) if return_diagnostics else result
    xtx_by_group: List[Optional[torch.Tensor]] = [None] * len(token_group_names)
    xtd_by_group: List[Optional[torch.Tensor]] = [None] * len(token_group_names)
    delta_ss_by_group = [0.0] * len(token_group_names)
    rows_by_group = [0] * len(token_group_names)
    sum_x_abs: Optional[torch.Tensor] = None
    sum_y_abs: Optional[torch.Tensor] = None
    count = 0
    hmat: Optional[torch.Tensor] = None
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        cache_dir = Path(str(item["_cache_dir"]))
        x_tensor = _load_unit_tensor(
            cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu"
        )
        y_tensor = _load_unit_tensor(
            cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu"
        )
        x_cpu = rows_view(x_tensor)
        y_cpu = rows_view(y_tensor)
        if x_cpu.shape != y_cpu.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(x_cpu.shape)} {target_unit_id}={tuple(y_cpu.shape)}"
            )
        group_ids = _ghost_token_group_ids(
            x_tensor,
            token_group_names=token_group_names,
            token_group_starts=token_group_starts,
        )
        for start in range(0, int(x_cpu.shape[0]), int(chunk_rows)):
            end = min(start + int(chunk_rows), int(x_cpu.shape[0]))
            x = _stage_row_chunk(x_cpu, start, end, device=device)
            y = _stage_row_chunk(y_cpu, start, end, device=device)
            if method == "recovery_ghost":
                chunk_groups = (
                    group_ids[start:end].to(device=device)
                    if group_ids is not None
                    else None
                )
                for group_index in range(len(token_group_names)):
                    if chunk_groups is None:
                        x_group, y_group = x, y
                    else:
                        mask = chunk_groups == group_index
                        if not bool(mask.any()):
                            continue
                        x_group, y_group = x[mask], y[mask]
                    x64 = x_group.to(torch.float64)
                    delta64 = (y_group - x_group).to(torch.float64)
                    xx = x64.T @ x64
                    xd = x64.T @ delta64
                    delta_ss = float(torch.sum(delta64.square()).item())
                    old_xx = xtx_by_group[group_index]
                    old_xd = xtd_by_group[group_index]
                    xtx_by_group[group_index] = xx if old_xx is None else old_xx + xx
                    xtd_by_group[group_index] = xd if old_xd is None else old_xd + xd
                    delta_ss_by_group[group_index] += delta_ss
                    rows_by_group[group_index] += int(x_group.shape[0])
                    del x_group, y_group, x64, delta64, xx, xd
                del chunk_groups
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
        del x_tensor, y_tensor, x_cpu, y_cpu, group_ids
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
        result, diagnostics = _solve_grouped_ghost_statistics(
            xtx_by_group=xtx_by_group,
            xtd_by_group=xtd_by_group,
            delta_ss_by_group=delta_ss_by_group,
            rows_by_group=rows_by_group,
            token_group_names=token_group_names,
            legacy_ridge=float(ridge),
            stability=stability,
            autogroup_threshold=ghost_autogroup_threshold,
        )
        _CLOSED_FORM_SESSION_CACHE[cache_key] = result.clone()
        return (result, diagnostics) if return_diagnostics else result

    if sum_x_abs is None or sum_y_abs is None or count <= 0:
        raise RuntimeError(f"No activation rows were available for {method}")
    scale = sum_y_abs / (sum_x_abs + 1e-8)
    assert hmat is not None
    result = (hmat @ torch.diag(scale.to(torch.float32)) @ hmat.T).to(torch.float32).cpu()
    _CLOSED_FORM_SESSION_CACHE[cache_key] = result.clone()
    return (result, {}) if return_diagnostics else result


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


def _solve_ghost_with_prior(
    xtx: torch.Tensor,
    xtd: torch.Tensor,
    *,
    strength: float,
    prior_weight: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Solve a trace-scaled ridge objective centered on ``prior_weight``."""
    channels = int(xtx.shape[0])
    scale = float((torch.trace(xtx) / max(channels, 1)).item())
    regularization = float(strength) * scale
    eye = torch.eye(channels, device=xtx.device, dtype=torch.float64)
    prior_delta = prior_weight.to(device=xtx.device, dtype=torch.float64) - eye
    delta = torch.linalg.solve(
        xtx + regularization * eye,
        xtd + regularization * prior_delta,
    )
    return (eye + delta).to(torch.float32), regularization


def _matrix_diagnostics(weight: torch.Tensor) -> Dict[str, Any]:
    matrix = weight.to(dtype=torch.float64)
    channels = int(matrix.shape[0])
    singular = torch.linalg.svdvals(matrix)
    identity = torch.eye(channels, device=matrix.device, dtype=matrix.dtype)
    finite = bool(torch.isfinite(matrix).all().item())
    minimum = float(singular.min().item())
    maximum = float(singular.max().item())
    return {
        "finite": finite,
        "delta_rms": float(torch.linalg.vector_norm(matrix - identity).item() / math.sqrt(channels)),
        "singular_min": minimum,
        "singular_max": maximum,
        "matrix_condition": (maximum / minimum if minimum > 0.0 else None),
    }


def _zero_prior_matrix_diagnostics(weight: torch.Tensor) -> Dict[str, Any]:
    matrix = weight.to(dtype=torch.float64)
    channels = int(matrix.shape[0])
    singular = torch.linalg.svdvals(matrix)
    minimum = float(singular.min().item())
    maximum = float(singular.max().item())
    return {
        "finite": bool(torch.isfinite(matrix).all().item()),
        "delta_rms": float(torch.linalg.vector_norm(matrix).item() / math.sqrt(channels)),
        "singular_min": minimum,
        "singular_max": maximum,
        "matrix_condition": (maximum / minimum if minimum > 0.0 else None),
    }


def _gram_diagnostics(xtx: torch.Tensor, rows: int) -> Dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(xtx)
    maximum = max(float(eigenvalues.max().item()), 0.0)
    positive = eigenvalues[eigenvalues > maximum * 1e-10]
    minimum = float(positive.min().item()) if int(positive.numel()) else 0.0
    return {
        "rows": int(rows),
        "trace": float(torch.trace(xtx).item()),
        "effective_rank": int(positive.numel()),
        "gram_condition": (maximum / minimum if minimum > 0.0 else None),
    }


def _solve_grouped_ghost_statistics(
    *,
    xtx_by_group: Sequence[Optional[torch.Tensor]],
    xtd_by_group: Sequence[Optional[torch.Tensor]],
    rows_by_group: Sequence[int],
    token_group_names: Sequence[str],
    legacy_ridge: float,
    stability: GhostStabilityConfig,
    autogroup_threshold: Optional[float] = None,
    delta_ss_by_group: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    names = tuple(str(name) for name in token_group_names)
    populated = [value for value in xtx_by_group if value is not None]
    if not populated:
        raise RuntimeError("Ghost recovery requires at least one activation row")
    channels = int(populated[0].shape[0])
    all_xtx = sum(
        (value for value in xtx_by_group if value is not None),
        torch.zeros_like(populated[0]),
    )
    populated_d = [value for value in xtd_by_group if value is not None]
    all_xtd = sum(populated_d, torch.zeros_like(populated_d[0]))
    all_weight = _solve_ghost_from_statistics(
        all_xtx, all_xtd, ridge=float(legacy_ridge)
    )
    identity = torch.eye(channels, device=all_xtx.device, dtype=torch.float32)
    weights: List[torch.Tensor] = []
    group_diagnostics: Dict[str, Any] = {}
    for index, name in enumerate(names):
        xtx, xtd = xtx_by_group[index], xtd_by_group[index]
        regularization = float(legacy_ridge)
        prior_name = "identity"
        if xtx is None or xtd is None:
            weight = identity
        elif stability.profile == "hybrid" and name == "special":
            weight = all_weight
            regularization = 0.0
            prior_name = "all_exact"
        elif stability.profile == "trace_identity" and name == "special":
            weight, regularization = _solve_ghost_with_prior(
                xtx, xtd,
                strength=float(stability.special_ridge),
                prior_weight=identity,
            )
        elif stability.profile == "trace_all" and name == "special":
            weight, regularization = _solve_ghost_with_prior(
                xtx, xtd,
                strength=float(stability.special_ridge),
                prior_weight=all_weight,
            )
            prior_name = "all"
        elif stability.profile == "native_all" and name in {"camera", "register"}:
            strength = (
                stability.camera_ridge if name == "camera" else stability.register_ridge
            )
            weight, regularization = _solve_ghost_with_prior(
                xtx, xtd, strength=float(strength), prior_weight=all_weight
            )
            prior_name = "all"
        else:
            weight = _solve_ghost_from_statistics(
                xtx, xtd, ridge=float(legacy_ridge)
            )
        weights.append(weight.cpu())
        gram = (
            _gram_diagnostics(xtx, int(rows_by_group[index]))
            if xtx is not None
            else {"rows": 0}
        )
        group_diagnostics[name] = {
            **gram,
            **_matrix_diagnostics(weight),
            "prior": prior_name,
            "regularization": float(regularization),
        }
    result = weights[0] if len(weights) == 1 else torch.stack(weights)
    autogroup: Optional[Dict[str, Any]] = None
    if autogroup_threshold is not None:
        threshold = float(autogroup_threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                f"Ghost autogroup threshold must be finite and positive, got {threshold}"
            )
        if names == ("patch",) and result.dim() == 2:
            autogroup = {
                "metric": "special_to_all_over_all_delta",
                "value": None,
                "threshold": threshold,
                "comparison": "not_applicable",
                "decision": "un-grouped",
                "reason": "no_special_tokens",
                "candidate_grouped_weight_shape": None,
                "selected_weight_shape": [int(x) for x in all_weight.shape],
            }
            # A patch-only backbone has no second token group to choose from.
            # Select the shared solve explicitly so this path is numerically
            # identical to ordinary sequential Ghost recovery.
            result = all_weight.detach().cpu().to(torch.float32)
        elif names != ("special", "patch") or result.dim() != 3:
            raise ValueError(
                "Ghost autogroup requires special/patch or patch-only token layout, "
                f"got {names}"
            )
        else:
            special_weight = result[names.index("special")].to(torch.float64)
            all_weight_cpu = all_weight.detach().cpu().to(torch.float64)
            identity_cpu = torch.eye(channels, dtype=torch.float64)
            numerator = float(
                torch.linalg.vector_norm(special_weight - all_weight_cpu).item()
            )
            denominator = float(
                torch.linalg.vector_norm(all_weight_cpu - identity_cpu).item()
            )
            relative = (
                numerator / denominator
                if denominator > 0.0
                else (0.0 if numerator == 0.0 else math.inf)
            )
            use_grouped = bool(math.isfinite(relative) and relative < threshold)
            autogroup = {
                "metric": "special_to_all_over_all_delta",
                "value": relative,
                "threshold": threshold,
                "comparison": "lt",
                "decision": "grouped" if use_grouped else "un-grouped",
                "candidate_grouped_weight_shape": [int(x) for x in result.shape],
                "selected_weight_shape": [
                    int(x) for x in (
                        result.shape if use_grouped else all_weight_cpu.shape
                    )
                ],
            }
            if not use_grouped:
                result = all_weight_cpu.to(torch.float32)
    for index, name in enumerate(names):
        xtx = xtx_by_group[index]
        xtd = xtd_by_group[index]
        rows = int(rows_by_group[index])
        if xtx is None or xtd is None or rows <= 0:
            continue
        if delta_ss_by_group is None:
            continue
        selected_weight = result if result.dim() == 2 else result[index]
        delta_matrix = (
            selected_weight.to(device=xtx.device, dtype=torch.float64)
            - torch.eye(channels, device=xtx.device, dtype=torch.float64)
        )
        delta_ss = float(delta_ss_by_group[index])
        residual_ss = (
            delta_ss
            - 2.0 * float(torch.sum(delta_matrix * xtd).item())
            + float(torch.trace(delta_matrix.T @ xtx @ delta_matrix).item())
        )
        denominator = max(rows * channels, 1)
        residual_rms = math.sqrt(max(residual_ss, 0.0) / denominator)
        target_delta_rms = math.sqrt(max(delta_ss, 0.0) / denominator)
        group_diagnostics[name].update(
            {
                "activation_residual_rms": residual_rms,
                "activation_target_delta_rms": target_delta_rms,
                "activation_residual_ratio": (
                    residual_rms / target_delta_rms if target_delta_rms > 0.0 else None
                ),
            }
        )
    diagnostics = {
        "stability": stability.signature_payload(),
        "groups": group_diagnostics,
        "all_token_matrix": _matrix_diagnostics(all_weight),
    }
    if autogroup is not None:
        diagnostics["autogroup"] = autogroup
    return result, diagnostics


def _accumulate_coupled_statistics(
    stats: Optional[Dict[str, torch.Tensor]],
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_start_idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Accumulate scene/frame-aware normal equations without retaining activations."""
    if source.dim() != 4 or source.shape != target.shape:
        raise ValueError(f"coupled Ghost expects matching [B,S,P,C], got {source.shape}/{target.shape}")
    x = source.to(device=device, dtype=torch.float64)
    y = target.to(device=device, dtype=torch.float64)
    special_x, patch_x = x[:, :, :patch_start_idx], x[:, :, patch_start_idx:]
    special_y, patch_y = y[:, :, :patch_start_idx], y[:, :, patch_start_idx:]
    channels = int(x.shape[-1])
    xs = special_x.reshape(-1, channels)
    xp = patch_x.reshape(-1, channels)
    ys = special_y.reshape(-1, channels)
    yp = patch_y.reshape(-1, channels)
    mean_patch = patch_x.mean(dim=2, keepdim=True).expand(
        -1, -1, int(special_x.shape[2]), -1
    ).reshape(-1, channels)
    values = {
        "all_xx": x.reshape(-1, channels).T @ x.reshape(-1, channels),
        "all_xy": x.reshape(-1, channels).T @ y.reshape(-1, channels),
        "s_xx": xs.T @ xs, "s_xy": xs.T @ ys,
        "p_xx": xp.T @ xp, "p_xy": xp.T @ yp,
        "s_xm": xs.T @ mean_patch,
        "m_xx": mean_patch.T @ mean_patch,
        "m_xy": mean_patch.T @ ys,
    }
    if stats is None:
        stats = {key: value for key, value in values.items()}
        stats["rows_s"] = torch.tensor(int(xs.shape[0]), device=device, dtype=torch.int64)
        stats["rows_p"] = torch.tensor(int(xp.shape[0]), device=device, dtype=torch.int64)
    else:
        for key, value in values.items():
            stats[key] += value
        stats["rows_s"] += int(xs.shape[0])
        stats["rows_p"] += int(xp.shape[0])
    return stats


def _solve_coupled_ghost_statistics(
    stats: Dict[str, torch.Tensor], *, legacy_ridge: float, stability: GhostStabilityConfig
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    channels = int(stats["all_xx"].shape[0])
    eye = torch.eye(channels, device=stats["all_xx"].device, dtype=torch.float64)
    w0 = _solve_ghost_from_statistics(
        stats["all_xx"], stats["all_xy"] - stats["all_xx"], ridge=legacy_ridge
    ).to(device=eye.device, dtype=torch.float64)
    zszs = w0.T @ stats["s_xx"] @ w0
    zsys = w0.T @ stats["s_xy"]
    zpzp = w0.T @ stats["p_xx"] @ w0
    zpyp = w0.T @ stats["p_xy"]
    mz_mz = w0.T @ stats["m_xx"] @ w0
    zs_mz = w0.T @ stats["s_xm"] @ w0
    mz_ys = w0.T @ stats["m_xy"]

    def solve_identity(xx: torch.Tensor, xy: torch.Tensor, strength: float) -> Tuple[torch.Tensor, float]:
        scale = float((torch.trace(xx) / channels).item())
        reg = float(strength) * scale
        return torch.linalg.solve(xx + (reg + legacy_ridge) * eye, xy + reg * eye), reg

    profile = stability.coupling_profile
    rs = eye.clone()
    rp = eye.clone()
    bps = torch.zeros_like(eye)
    residual_reg_s = 0.0
    residual_reg_p = 0.0
    context_reg = 0.0
    if profile in {"shared_residual", "shared_residual_patch_mean"}:
        rp, residual_reg_p = solve_identity(zpzp, zpyp, stability.residual_ridge)
    if profile == "shared_residual":
        rs, residual_reg_s = solve_identity(zszs, zsys, stability.residual_ridge)
    elif profile == "patch_mean":
        scale = float((torch.trace(mz_mz) / channels).item())
        context_reg = float(stability.context_ridge) * scale
        bps = torch.linalg.solve(
            mz_mz + (context_reg + legacy_ridge) * eye,
            mz_ys - zs_mz.T,
        )
    elif profile == "shared_residual_patch_mean":
        scale_r = float((torch.trace(zszs) / channels).item())
        scale_c = float((torch.trace(mz_mz) / channels).item())
        residual_reg_s = float(stability.residual_ridge) * scale_r
        context_reg = float(stability.context_ridge) * scale_c
        normal = torch.cat(
            [torch.cat([zszs, zs_mz], dim=1), torch.cat([zs_mz.T, mz_mz], dim=1)], dim=0
        )
        prior = torch.cat([eye, torch.zeros_like(eye)], dim=0)
        regularizer = torch.diag(torch.cat([
            torch.full((channels,), residual_reg_s, device=eye.device, dtype=eye.dtype),
            torch.full((channels,), context_reg, device=eye.device, dtype=eye.dtype),
        ]))
        rhs = torch.cat([zsys, mz_ys], dim=0) + regularizer @ prior
        jitter = float(legacy_ridge) * torch.eye(
            2 * channels, device=eye.device, dtype=eye.dtype
        )
        solution = torch.linalg.solve(normal + regularizer + jitter, rhs)
        rs, bps = solution[:channels], solution[channels:]
    elif profile != "patch_mean":
        raise ValueError(f"Unsupported coupled profile {profile!r}")

    effective_s = w0 @ rs
    effective_p = w0 @ rp
    effective_context = w0 @ bps
    packed = torch.stack([effective_s, effective_p, effective_context, w0, rs, rp, bps]).to(torch.float32).cpu()
    if not bool(torch.isfinite(packed).all().item()):
        raise FloatingPointError(f"Coupled Ghost profile {profile!r} produced non-finite matrices")
    diagnostics = {
        "coupling": stability.signature_payload(),
        "rows_by_token_group": {"special": int(stats["rows_s"].item()), "patch": int(stats["rows_p"].item())},
        "all_token_matrix": {
            **_gram_diagnostics(
                stats["all_xx"], int(stats["rows_s"].item() + stats["rows_p"].item())
            ),
            **_matrix_diagnostics(w0),
        },
        "groups": {
            "special": {
                **_gram_diagnostics(zszs, int(stats["rows_s"].item())),
                **_matrix_diagnostics(effective_s),
                "prior": "all_then_identity",
                "regularization": residual_reg_s,
            },
            "patch": {
                **_gram_diagnostics(zpzp, int(stats["rows_p"].item())),
                **_matrix_diagnostics(effective_p),
                "prior": "all_then_identity",
                "regularization": residual_reg_p,
            },
        },
        "context": {
            **_gram_diagnostics(mz_mz, int(stats["rows_s"].item())),
            **_zero_prior_matrix_diagnostics(bps),
            "regularization": context_reg,
            "prior": "zero",
            "matrix": "Bps",
        },
        "effective_context": {
            **_zero_prior_matrix_diagnostics(effective_context),
            "matrix": "W0@Bps",
        },
        "components": {
            "packing": ["effective_special", "effective_patch", "effective_context", "W0", "Rs", "Rp", "Bps"],
            "W0": _matrix_diagnostics(w0),
            "Rs": _matrix_diagnostics(rs),
            "Rp": _matrix_diagnostics(rp),
            "Bps": _zero_prior_matrix_diagnostics(bps),
            "effective_context": _zero_prior_matrix_diagnostics(effective_context),
        },
    }
    return packed, diagnostics


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
        # Released tensors from a previous sequential boundary may still be
        # held by PyTorch's caching allocator. Return only those unused blocks
        # before asking the driver how much memory is actually available.
        torch.cuda.empty_cache()
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


def _streamline_lr_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay, matching Streamline's schedule."""
    current = int(step)
    if current < int(warmup_steps):
        return float(current) / float(max(1, int(warmup_steps)))
    progress = float(current - int(warmup_steps)) / float(
        max(1, int(total_steps) - int(warmup_steps))
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine


def _optimize_streamline_ffn(
    *,
    row_store: ShardedRowStore,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    expansion_factor: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit the official two-layer Streamline FFN with boundary-alignment MSE."""
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("Streamline FFN epochs, batch size, and learning rate must be positive")
    if min_lr < 0 or min_lr > lr:
        raise ValueError(f"Streamline FFN min_lr must be in [0, lr], got {min_lr}")
    if weight_decay < 0 or expansion_factor <= 0:
        raise ValueError("Streamline FFN weight decay must be non-negative and expansion positive")

    channels = int(row_store.channels)
    num_rows = int(row_store.num_rows)
    steps_per_epoch = math.ceil(num_rows / int(batch_size))
    total_steps = int(epochs) * steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.03))
    cuda_devices: List[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            int(device.index) if device.index is not None else torch.cuda.current_device()
        )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        module = StreamlineFFN(
            channels, expansion_factor=int(expansion_factor)
        ).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _streamline_lr_factor(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=float(min_lr) / float(lr),
            ),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        rows_seen = 0
        optimizer_steps = 0
        best_loss = float("inf")
        best_epoch = 0
        best_parameters: Optional[torch.Tensor] = None
        history: List[Dict[str, Any]] = []

        for epoch in range(1, int(epochs) + 1):
            order = torch.randperm(num_rows, generator=generator)
            epoch_squared_error = 0.0
            epoch_elements = 0
            for start in range(0, num_rows, int(batch_size)):
                indices = order[start : start + int(batch_size)]
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
                prediction = module(xb)
                loss = torch.nn.functional.mse_loss(prediction, yb)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Streamline FFN produced a non-finite loss")
                loss.backward()
                optimizer.step()
                scheduler.step()
                elements = int(yb.numel())
                epoch_squared_error += float(loss.detach().item()) * elements
                epoch_elements += elements
                rows_seen += int(xb.shape[0])
                optimizer_steps += 1

            mean_loss = epoch_squared_error / float(max(1, epoch_elements))
            current_lr = float(optimizer.param_groups[0]["lr"])
            history.append({"epoch": epoch, "mean_loss": mean_loss, "lr": current_lr})
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_epoch = epoch
                best_parameters = pack_streamline_ffn_parameters(module)
            print(
                "[recovery-fit] method=streamline_ffn "
                f"epochs={epoch}/{epochs} mean_loss={mean_loss:.6g} lr={current_lr:.6g}",
                file=sys.stderr,
                flush=True,
            )

        if best_parameters is None or not bool(torch.isfinite(best_parameters).all()):
            raise FloatingPointError("Streamline FFN produced non-finite parameters")
        diagnostics = {
            "num_rows": num_rows,
            "rows_seen": rows_seen,
            "hidden_size": channels,
            "expansion_factor": int(expansion_factor),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "min_lr": float(min_lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "warmup_steps": warmup_steps,
            "optimizer_steps": optimizer_steps,
            "selected_epoch": best_epoch,
            "best_loss": best_loss,
            "loss_components": ["mse"],
            "history": history,
            "row_storage": "scene_sharded_cpu_float32",
        }
        return best_parameters, diagnostics


def _fit_streamline_ffn(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    expansion_factor: int,
    seed: int,
    boundary_index: int = 0,
    boundary_total: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Collect cached component boundaries and train one Streamline FFN."""
    shards: List[RowShard] = []
    total_scenes = len(scenes)
    for scene_index, item in enumerate(scenes, start=1):
        cache_dir = Path(str(item["_cache_dir"]))
        source = _load_unit_tensor(
            cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu"
        )
        target = _load_unit_tensor(
            cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu"
        )
        source_rows = rows_view(source)
        target_rows = rows_view(target)
        if source_rows.shape != target_rows.shape:
            raise RuntimeError(
                f"Streamline FFN boundary mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(source_rows.shape)} "
                f"{target_unit_id}={tuple(target_rows.shape)}"
            )
        shards.append(
            RowShard(
                scene_id=str(item.get("scene_id", "")),
                inputs=source_rows,
                desired=target_rows,
            )
        )
        if scene_index == total_scenes or scene_index % 10 == 0:
            print(
                "[recovery-fit]"
                f"[boundary={boundary_index + 1}/{boundary_total}]"
                f"[scenes={scene_index}/{total_scenes}] "
                "method=streamline_ffn phase=collection",
                file=sys.stderr,
                flush=True,
            )
    store = RecoveryActivationStore(shards)
    try:
        return _optimize_streamline_ffn(
            row_store=store,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            expansion_factor=expansion_factor,
            seed=seed,
        )
    finally:
        store.close()
        if _ACTIVE_RECOVERY_SESSION is not None:
            _ACTIVE_RECOVERY_SESSION.clear_units()


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
    """Forward one scene while retaining the requested evolving boundaries."""
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
            f"Sequential Ghost forward missed source {source_unit_id}: {missing}"
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
    ghost_token_group_names: Sequence[str] = ("all",),
    ghost_token_group_starts: Sequence[int] = (0,),
    ghost_stability: Optional[GhostStabilityConfig] = None,
    ghost_autogroup_threshold: Optional[float] = None,
    num_layers: int = 0,
    target_mode: str = "sequential_current_model",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit an evolving Ghost source to a current- or full-model target."""
    if target_mode not in {"sequential_current_model", "full_model_disk_cache"}:
        raise ValueError(f"Unsupported current Ghost target mode: {target_mode!r}")
    token_group_names, token_group_starts = _validate_ghost_token_layout(
        ghost_token_group_names, ghost_token_group_starts
    )
    stability = (ghost_stability or GhostStabilityConfig()).validate()
    if stability.coupling_profile != "none":
        coupled_stats: Optional[Dict[str, torch.Tensor]] = None
        patch_start_idx = int(token_group_starts[1])
        for scene_index, item in enumerate(scenes, start=1):
            unit_map, images = _forward_current_ghost_boundary(
                model=model, adapter=adapter, source_unit_id=source_unit_id,
                target_unit_id=target_unit_id, item=item, device=device,
            )
            target = (
                unit_map[target_unit_id]
                if target_mode == "sequential_current_model"
                else _load_unit_tensor(
                    Path(str(item["_cache_dir"])), item, target_unit_id,
                    num_layers=num_layers, map_location="cpu",
                )
            )
            coupled_stats = _accumulate_coupled_statistics(
                coupled_stats, unit_map[source_unit_id], target,
                patch_start_idx=patch_start_idx, device=device,
            )
            del unit_map, images, target
            if scene_index == len(scenes) or scene_index % 10 == 0:
                print(
                    f"[recovery-fit][boundary={boundary_index + 1}/{boundary_total}]"
                    f"[scenes={scene_index}/{len(scenes)}] method=ghost_coupled_current",
                    file=sys.stderr, flush=True,
                )
        assert coupled_stats is not None
        weight, details = _solve_coupled_ghost_statistics(
            coupled_stats, legacy_ridge=float(ridge), stability=stability
        )
        return weight, {
            "fit_mode": "sequential_current_model",
            "source_mode": "sequential_current_model",
            "target_mode": target_mode,
            "num_scenes": len(scenes),
            "ridge": float(ridge),
            "chunk_rows": int(chunk_rows),
            "token_groups": ["special", "patch"],
            "token_group_starts": [0, patch_start_idx],
            **details,
        }
    xtx_by_group: List[Optional[torch.Tensor]] = [None] * len(token_group_names)
    xtd_by_group: List[Optional[torch.Tensor]] = [None] * len(token_group_names)
    delta_ss_by_group = [0.0] * len(token_group_names)
    rows_by_group = [0] * len(token_group_names)
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
        target = (
            unit_map[target_unit_id]
            if target_mode == "sequential_current_model"
            else _load_unit_tensor(
                Path(str(item["_cache_dir"])), item, target_unit_id,
                num_layers=num_layers, map_location="cpu",
            )
        )
        if source.shape != target.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(source.shape)} "
                f"{target_unit_id}={tuple(target.shape)}"
            )
        channels = int(source.shape[-1])
        source_flat = source.reshape(-1, channels)
        target_flat = target.reshape(-1, channels)
        group_ids = _ghost_token_group_ids(
            source,
            token_group_names=token_group_names,
            token_group_starts=token_group_starts,
        )
        # Bound FP64 workspace while retaining the existing FP64 normal equations.
        for row_start in range(0, int(source_flat.shape[0]), int(chunk_rows)):
            row_end = min(row_start + int(chunk_rows), int(source_flat.shape[0]))
            x_chunk = source_flat[row_start:row_end].to(device=device, dtype=torch.float64)
            target_chunk = target_flat[row_start:row_end].to(
                device=device, dtype=torch.float64
            )
            chunk_groups = (
                group_ids[row_start:row_end].to(device=device)
                if group_ids is not None
                else None
            )
            for group_index in range(len(token_group_names)):
                if chunk_groups is None:
                    x, target_group = x_chunk, target_chunk
                else:
                    mask = chunk_groups == group_index
                    if not bool(mask.any()):
                        continue
                    x, target_group = x_chunk[mask], target_chunk[mask]
                delta = target_group - x
                xx = x.T @ x
                xd = x.T @ delta
                delta_ss = float(torch.sum(delta.square()).item())
                old_xx = xtx_by_group[group_index]
                old_xd = xtd_by_group[group_index]
                xtx_by_group[group_index] = xx if old_xx is None else old_xx + xx
                xtd_by_group[group_index] = xd if old_xd is None else old_xd + xd
                delta_ss_by_group[group_index] += delta_ss
                rows_by_group[group_index] += int(x.shape[0])
                del x, target_group, delta, xx, xd
            del x_chunk, target_chunk, chunk_groups
        num_scenes += 1
        del unit_map, images, source, target, source_flat, target_flat, group_ids
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
    weight, stability_diagnostics = _solve_grouped_ghost_statistics(
        xtx_by_group=xtx_by_group,
        xtd_by_group=xtd_by_group,
        delta_ss_by_group=delta_ss_by_group,
        rows_by_group=rows_by_group,
        token_group_names=token_group_names,
        legacy_ridge=float(ridge),
        stability=stability,
        autogroup_threshold=ghost_autogroup_threshold,
    )
    diagnostics = {
        "fit_mode": "sequential_current_model",
        "source_mode": "sequential_current_model",
        "target_mode": target_mode,
        "num_scenes": num_scenes,
        "num_rows": sum(rows_by_group),
        "ridge": float(ridge),
        "chunk_rows": int(chunk_rows),
        **stability_diagnostics,
    }
    if len(token_group_names) > 1:
        diagnostics.update(
            {
                "rows_by_token_group": dict(zip(token_group_names, rows_by_group)),
                "token_groups": list(token_group_names),
                "token_group_starts": list(token_group_starts),
            }
        )
    return weight.detach().cpu(), diagnostics


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
    pi3_token_group_alias = (
        str(backbone).strip().lower() == "pi3"
        and method in TOKEN_GROUPED_RECOVERY_BASE_METHODS
    )
    if pi3_token_group_alias:
        source_method = TOKEN_GROUPED_RECOVERY_BASE_METHODS[method]
        reason = "Pi3 exposes one patch-token group"
    elif method in _DERIVED_GHOST_METHODS and _has_single_drop_interval(prune_plan):
        source_method = "recovery_ghost"
        reason = "single contiguous drop interval"
    else:
        return False
    source_dir = Path(out_dir).parent / source_method
    if not recovery_result_complete(
        source_dir,
        plan_root=Path(plan_root),
        backbone=backbone,
        selector=selector,
        recovery_plan_name=source_method,
    ):
        if pi3_token_group_alias:
            raise FileNotFoundError(
                f"Pi3 token-grouped recovery {method!r} must reuse the complete "
                f"non-grouped result {source_method!r}, but {source_dir} is incomplete"
            )
        return False
    _replace_with_relative_symlink(Path(out_dir), source_dir)
    print(
        "\n  recovery reuse\n"
        f"    method: {recovery_plan_name.removeprefix('recovery_')}\n"
        f"    source: {source_method.removeprefix('recovery_')}\n"
        f"    reason: {reason}\n"
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


def _scenes_have_cached_replaceme_boundary(
    scenes: Sequence[Dict[str, Any]], source_unit_id: str, target_unit_id: str
) -> bool:
    """Return whether disk cache can fit one complete ReplaceMe boundary."""
    if not _scenes_have_cached_units(scenes, (source_unit_id, target_unit_id)):
        return False
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    for item in scenes:
        auxiliary_files = item.get("auxiliary_files")
        if not isinstance(auxiliary_files, dict):
            return False
        relative = str(auxiliary_files.get(auxiliary_id, "")).strip()
        cache_dir = Path(str(item.get("_cache_dir", "")))
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
    """Capture current source/residual/target rows on the evolving model."""
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    scratch_root = Path(tempfile.mkdtemp(prefix="lare_replaceme_"))
    # Existing cache file sizes provide a conservative, model-independent
    # estimate for the three dynamic streams plus serialization headers.
    estimated = 0
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        layer_files = item.get("layer_files") or {}
        auxiliary_files = item.get("auxiliary_files") or {}
        for key in (source_unit_id,):
            rel = str(layer_files.get(key, ""))
            path = cache_dir / rel if rel else Path()
            if rel and path.is_file():
                estimated += path.stat().st_size
        rel = str(auxiliary_files.get(auxiliary_id, ""))
        path = cache_dir / rel if rel else Path()
        if rel and path.is_file():
            estimated += path.stat().st_size
        rel = str(layer_files.get(target_unit_id, ""))
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
            requested_units = {source_unit_id, target_unit_id}
            unit_map, _selected_meta, auxiliary_map = forward_selected_units_with_auxiliaries(
                adapter=adapter,
                model=model,
                images=images,
                device=device,
                unit_ids=requested_units,
                auxiliary_ids={auxiliary_id},
            )
            target = unit_map[target_unit_id]
            h_raw, y_raw, r_raw = _align_replaceme_activations(
                unit_map[source_unit_id],
                target,
                auxiliary_map[auxiliary_id],
                scene_id=str(item.get("scene_id", "<unknown>")),
                source_unit_id=source_unit_id,
            )
            tensors_to_stage = [("h", h_raw), ("y", y_raw), ("r", r_raw)]
            paths = []
            for name, tensor in tensors_to_stage:
                path = scratch_root / f"{scene_index:06d}_{name}.pt"
                torch.save(tensor.detach().cpu().contiguous(), path)
                paths.append(path)
            staged = [
                torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                for path in paths
            ]
            h_disk, y_disk, r_disk = staged
            y = rows_view(y_disk)
            h, r = rows_view(h_disk), rows_view(r_disk)

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
            del images, unit_map, auxiliary_map, target, h_raw, y_raw, r_raw
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


def _fit_replaceme_ls_from_row_store(
    *,
    row_store: ShardedRowStore,
    device: torch.device,
    alpha: float,
    chunk_rows: int = DEFAULT_RECOVERY_CHUNK_ROWS,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve ReplaceMe LS from activations captured on the evolving model."""
    if int(chunk_rows) <= 0:
        raise ValueError("Sequential ReplaceMe LS chunk_rows must be positive")
    rtr: Optional[torch.Tensor] = None
    rtd: Optional[torch.Tensor] = None
    rows = 0
    for start in range(0, int(row_store.num_rows), int(chunk_rows)):
        end = min(start + int(chunk_rows), int(row_store.num_rows))
        indices = torch.arange(start, end, dtype=torch.long)
        input_rows, desired_rows = row_store.gather(
            indices, pin_memory=device.type == "cuda"
        )
        if input_rows.shape != desired_rows.shape:
            raise RuntimeError(
                "Sequential ReplaceMe LS received mismatched rows: "
                f"{tuple(input_rows.shape)} vs {tuple(desired_rows.shape)}"
            )
        residual = input_rows.to(
            device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        desired = desired_rows.to(
            device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        row_store.record_transfer()
        residual64 = residual.to(torch.float64)
        desired64 = desired.to(torch.float64)
        rr = residual64.T @ residual64
        rd = residual64.T @ desired64
        rtr = rr if rtr is None else rtr + rr
        rtd = rd if rtd is None else rtd + rd
        rows += int(residual.shape[0])
        del residual, desired, residual64, desired64, rr, rd
    if rtr is None or rtd is None or rows <= 0:
        raise RuntimeError("No activation rows were available for sequential ReplaceMe LS")
    channels = int(rtr.shape[0])
    reg = float(alpha) * torch.eye(channels, device=device, dtype=torch.float64)
    transform = torch.linalg.solve(rtr + reg, rtd).to(torch.float32).cpu()
    return transform, {
        "num_scenes": len(row_store.shards),
        "num_rows": rows,
        "ls_alpha": float(alpha),
        "chunk_rows": int(chunk_rows),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_boundary_prefix_checkpoint(
    path: Path,
    records: Sequence[Dict[str, Any]],
    method: str,
    prune_plan_name: str,
) -> Tuple[List[Tuple[torch.Tensor, Dict[str, Any]]], str]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid prefix checkpoint payload: {checkpoint_path}")
    recorded_method = str(payload.get("recovery_method", ""))
    if recorded_method != str(method):
        raise ValueError(
            f"Prefix checkpoint method mismatch: {recorded_method!r} != {method!r}"
        )
    configuration = payload.get("fit_configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError(f"Prefix checkpoint has no fit configuration: {checkpoint_path}")
    recorded_selector = str(configuration.get("prune_plan_name", ""))
    if recorded_selector != str(prune_plan_name):
        raise ValueError(
            f"Prefix checkpoint selector mismatch: {recorded_selector!r} != {prune_plan_name!r}"
        )
    specs = payload.get("recovery_specs")
    weights = payload.get("recovery_weights")
    diagnostics = payload.get("fit_diagnostics", [])
    if (
        not isinstance(specs, list)
        or not isinstance(weights, list)
        or len(specs) != len(records)
        or len(weights) != len(records)
    ):
        raise ValueError(
            f"Prefix checkpoint boundary count mismatch: {checkpoint_path}"
        )
    for boundary_index, (raw_record, raw_spec) in enumerate(zip(records, specs)):
        record = dict(raw_record)
        spec = dict(raw_spec)
        for field in ("source_unit_id", "target_unit_id"):
            if str(spec.get(field, "")) != str(record.get(field, "")):
                raise ValueError(
                    f"Prefix checkpoint boundary {boundary_index} mismatch for {field}: "
                    f"{spec.get(field)!r} != {record.get(field)!r}"
                )
    spec = dict(specs[0])
    weight_entry = dict(weights[0])
    weight = weight_entry.get("weight")
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"Prefix checkpoint boundary weight is not a tensor: {checkpoint_path}")
    detail = dict(diagnostics[0]) if isinstance(diagnostics, list) and diagnostics else {}
    detail.update({
        "reused_prefix_checkpoint": str(checkpoint_path),
        "reused_prefix_checkpoint_sha256": _sha256_file(checkpoint_path),
    })
    return [(weight.detach().cpu().to(torch.float32), detail)], _sha256_file(checkpoint_path)


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
    ghost_token_group_names: Sequence[str] = ("all",),
    ghost_token_group_starts: Sequence[int] = (0,),
    ghost_stability: Optional[GhostStabilityConfig] = None,
    ghost_autogroup_threshold: Optional[float] = None,
    boundary_fit_mode: str = "sequential_current_target",
    prefix_results: Sequence[Tuple[torch.Tensor, Dict[str, Any]]] = (),
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Fit evolving Ghost interval boundaries front-to-back."""
    if boundary_fit_mode not in {
        "sequential_current_target",
        "sequential_full_target",
        "independent_full_target",
    }:
        raise ValueError(f"Unsupported Ghost boundary fit mode: {boundary_fit_mode!r}")
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(device),
        dtype=None,
        eval_mode=True,
    )
    token_group_names, token_group_starts = _validate_ghost_token_layout(
        ghost_token_group_names, ghost_token_group_starts
    )
    stability = (ghost_stability or GhostStabilityConfig()).validate()
    results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    handles: List[Any] = []
    ordered_signatures = tuple(_boundary_signature(dict(raw)) for raw in records)
    prefix_base_key = (
        str(getattr(adapter, "name", type(adapter).__name__)),
        _base_checkpoint_hash(str(hf_model or adapter.default_hf_model())),
        _calibration_manifest_hash(scenes), "default", str(method), float(ridge),
        recovery_method_semantics_version(str(method)),
        int(chunk_rows),
        token_group_names,
        token_group_starts,
        tuple(sorted(stability.signature_payload().items())),
        ghost_autogroup_threshold,
        boundary_fit_mode,
    )
    prefix_entries = _SEQUENTIAL_GHOST_PREFIX_CACHE.setdefault(prefix_base_key, {})
    reusable: List[Tuple[torch.Tensor, Dict[str, Any]]] = [
        (weight.clone(), dict(diagnostics)) for weight, diagnostics in prefix_results
    ]
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
            weight, fit_details = _fit_boundary_weight(
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
                ghost_token_group_names=token_group_names,
                ghost_token_group_starts=token_group_starts,
                ghost_stability=stability,
                ghost_autogroup_threshold=ghost_autogroup_threshold,
                return_diagnostics=True,
            )
            diagnostics = {
                "fit_mode": "disk_cache",
                "source_mode": "full_model_disk_cache",
                "target_mode": "current_model_equivalent_full_model_disk_cache",
                "num_scenes": len(scenes),
                "ridge": float(ridge),
                "chunk_rows": int(chunk_rows),
                **fit_details,
            }
            if len(token_group_names) > 1:
                diagnostics.update(
                    {
                        "token_groups": list(token_group_names),
                        "token_group_starts": list(token_group_starts),
                    }
                )
        elif boundary_fit_mode == "independent_full_target":
            weight, diagnostics = _fit_boundary_weight(
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
                ghost_token_group_names=token_group_names,
                ghost_token_group_starts=token_group_starts,
                ghost_stability=stability,
                ghost_autogroup_threshold=ghost_autogroup_threshold,
                return_diagnostics=True,
            )
            diagnostics.update({
                "fit_mode": "independent_full_target",
                "source_mode": "full_model_disk_cache",
                "target_mode": "full_model_disk_cache",
            })
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
                ghost_token_group_names=token_group_names,
                ghost_token_group_starts=token_group_starts,
                ghost_stability=stability,
                ghost_autogroup_threshold=ghost_autogroup_threshold,
                num_layers=int(num_layers),
                target_mode=(
                    "full_model_disk_cache"
                    if boundary_fit_mode == "sequential_full_target"
                    else "sequential_current_model"
                ),
            )
        diagnostics["boundary_fit_mode"] = boundary_fit_mode
        diagnostics["boundary_index"] = boundary_index
        module_name = recovery_module_name(source_unit_id)
        spec = {
            "type": str(method),
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "module_name": module_name,
            "weight_shape": [int(x) for x in weight.shape],
        }
        if weight.dim() == 3:
            spec["token_groups"] = list(token_group_names)
            spec["token_group_starts"] = list(token_group_starts)
        if stability.coupling_profile != "none":
            spec["ghost_coupling_profile"] = stability.coupling_profile
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
    num_layers: int = 0,
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Match official ReplaceMe Cosine: fit each interval on the current model."""
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
        recovery_method_semantics_version("recovery_replaceme_cos"),
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
                "source_mode": "sequential_current_model",
                "target_mode": "sequential_current_model",
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
    ghost_stability = _ghost_stability_config(args)
    prune_plan = compose_recovery_unit_prune_plan(
        prune_plan_root=plan_root,
        recovery_plan_root=recovery_root,
        prune_selector=prune_plan_name,
        recovery_plan=recovery_plan_name,
    )
    bundle = load_recovery_plan_bundle(recovery_root, recovery_plan_name)
    method, pair_records = _pair_records(bundle, prune_plan, recovery_plan_name)
    ghost_boundary_fit_mode = str(args.ghost_boundary_fit_mode)
    ghost_methods = {
        "recovery_ghost_interval_sequential",
        *_TOKEN_GROUPED_GHOST_METHODS,
    }
    if ghost_boundary_fit_mode != "sequential_current_target" and (
        method not in ghost_methods or len(pair_records) <= 1
    ):
        raise SystemExit(
            "Non-default --ghost-boundary-fit-mode requires a multi-interval "
            "sequential Ghost recovery plan"
        )
    if args.reuse_boundary_prefix_checkpoint is not None and (
        method not in ghost_methods or len(pair_records) <= 1
    ):
        raise SystemExit(
            "--reuse-boundary-prefix-checkpoint requires a multi-interval "
            "sequential Ghost recovery plan"
        )
    is_autogroup = method == "recovery_ghost_interval_sequential_autogroup"
    ghost_autogroup_threshold: Optional[float] = None
    if is_autogroup:
        ghost_autogroup_threshold = float(args.ghost_autogroup_threshold)
        if (
            not math.isfinite(ghost_autogroup_threshold)
            or ghost_autogroup_threshold <= 0.0
        ):
            raise SystemExit("--ghost-autogroup-threshold must be finite and positive")
        if ghost_stability != GhostStabilityConfig():
            raise SystemExit(
                "autogroup currently requires legacy Ghost fitting without coupling"
            )
    default_output_root = Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT_DEFAULT)))
    out_dir = (
        args.output_dir
        or default_output_root
        / adapter.name
        / "recovery"
        / prune_plan_name
        / recovery_plan_name
    )
    if ghost_stability.coupling_profile == "none" and _try_link_derived_ghost_recovery(
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
    scene_limits = _parse_scene_limits(args.scene_limits, dataset_names)
    cache_dirs = (
        [Path(args.cache_root) / adapter.name / name / "cache" for name in dataset_names]
        if args.cache_root is not None
        else [stage_dir(args.experiment, name, "cache") for name in dataset_names]
    )
    scenes = _load_training_scenes(
        cache_dirs,
        int(args.max_scenes),
        exact_limits=(
            [scene_limits[name] for name in dataset_names] if scene_limits else None
        ),
    )
    if not scenes:
        raise SystemExit("No scenes selected from cache")
    first_manifest = load_v3_manifest(cache_dirs[0])
    num_layers = max((int(u.index) for u in first_manifest.layer_units if u.kind in ("frame", "global")), default=-1) + 1
    if int(prune_plan.num_units) != int(num_layers * 2):
        raise SystemExit(f"prune plan num_units={prune_plan.num_units} != cache num_units={num_layers * 2}")

    ghost_token_group_names: Tuple[str, ...] = ("all",)
    ghost_token_group_starts: Tuple[int, ...] = (0,)
    if method in _TOKEN_GROUPED_GHOST_METHODS:
        patch_start_idx = _common_patch_start(scenes)
        if ghost_stability.profile == "native_all":
            ghost_token_group_names, ghost_token_group_starts = (
                _validate_ghost_token_layout(
                    *adapter.ghost_token_group_layout(
                        patch_start_idx=int(patch_start_idx)
                    )
                )
            )
        else:
            ghost_token_group_names, ghost_token_group_starts = (
                _ghost_token_layout_for_method(
                    adapter=adapter,
                    method=method,
                    patch_start_idx=patch_start_idx,
                )
            )

    prefix_results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    prefix_checkpoint_sha256 = ""
    if args.reuse_boundary_prefix_checkpoint is not None:
        prefix_results, prefix_checkpoint_sha256 = _load_boundary_prefix_checkpoint(
            Path(args.reuse_boundary_prefix_checkpoint),
            pair_records,
            method,
            prune_plan_name,
        )

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
        f"    boundaries: {len(pair_records)}\n"
        + (
            f"    ghost token groups: {','.join(ghost_token_group_names)}\n"
            if method in _TOKEN_GROUPED_GHOST_METHODS
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    weights: List[torch.Tensor] = []
    recovery_specs: List[Dict[str, Any]] = []
    fit_diagnostics: List[Dict[str, Any]] = []
    sequential_replaceme_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    sequential_ghost_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    if method in (
        "recovery_ghost_interval_sequential",
        *_TOKEN_GROUPED_GHOST_METHODS,
    ) and len(pair_records) > 1:
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
            ghost_token_group_names=ghost_token_group_names,
            ghost_token_group_starts=ghost_token_group_starts,
            ghost_stability=ghost_stability,
            ghost_autogroup_threshold=ghost_autogroup_threshold,
            boundary_fit_mode=ghost_boundary_fit_mode,
            prefix_results=prefix_results,
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
            num_layers=num_layers,
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
            *_TOKEN_GROUPED_GHOST_METHODS,
        ):
            weight, diagnostics = _fit_boundary_weight(
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
                ghost_token_group_names=(
                    ghost_token_group_names
                    if method in _TOKEN_GROUPED_GHOST_METHODS
                    else ("all",)
                ),
                ghost_token_group_starts=(
                    ghost_token_group_starts
                    if method in _TOKEN_GROUPED_GHOST_METHODS
                    else (0,)
                ),
                ghost_stability=(
                    ghost_stability
                    if method in _TOKEN_GROUPED_GHOST_METHODS
                    else GhostStabilityConfig()
                ),
                ghost_autogroup_threshold=ghost_autogroup_threshold,
                return_diagnostics=True,
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
        elif method == "recovery_streamline_ffn":
            weight, diagnostics = _fit_streamline_ffn(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                epochs=int(args.streamline_ffn_epochs),
                batch_size=int(args.streamline_ffn_batch_size),
                lr=float(args.streamline_ffn_lr),
                min_lr=float(args.streamline_ffn_min_lr),
                weight_decay=float(args.streamline_ffn_weight_decay),
                expansion_factor=int(args.streamline_ffn_expansion),
                seed=int(args.streamline_ffn_seed),
                boundary_index=record_index,
                boundary_total=len(pair_records),
            )
        else:
            raise ValueError(f"Unsupported cache-fit recovery method {method!r}")
        if method in _TOKEN_GROUPED_GHOST_METHODS:
            diagnostics.setdefault("token_groups", list(ghost_token_group_names))
            diagnostics.setdefault("token_group_starts", list(ghost_token_group_starts))
        weights.append(weight)
        fit_diagnostics.append(diagnostics)
        spec = {
            "type": method,
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "source_unit_index": int(record.get("source_unit_index")),
            "target_unit_index": int(record.get("target_unit_index")),
            "module_name": recovery_module_name(source_unit_id),
            "weight_shape": [int(x) for x in weight.shape],
        }
        if method in _TOKEN_GROUPED_GHOST_METHODS and weight.dim() == 3:
            spec["token_groups"] = list(ghost_token_group_names)
            spec["token_group_starts"] = list(ghost_token_group_starts)
            if ghost_stability.coupling_profile != "none":
                spec["ghost_coupling_profile"] = ghost_stability.coupling_profile
        if method == "recovery_streamline_ffn":
            spec.update(
                {
                    "hidden_size": int(diagnostics["hidden_size"]),
                    "expansion_factor": int(diagnostics["expansion_factor"]),
                    "activation": "relu",
                    "objective": "boundary_mse",
                }
            )
        recovery_specs.append(spec)

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
    num_scenes_by_cache = {
        str(cache_dir): sum(
            1 for scene in scenes if str(scene.get("_cache_dir", "")) == str(cache_dir)
        )
        for cache_dir in cache_dirs
    }
    cache_manifest_sha256_by_dataset = {
        str(dataset_name): _sha256_file(Path(cache_dir) / "cache_manifest_v3.json")
        for dataset_name, cache_dir in zip(dataset_names, cache_dirs)
    }
    hf_model = str(args.hf_model or adapter.default_hf_model())
    fit_configuration = {
        "backbone": adapter.name,
        "base_hf_model": hf_model,
        "base_checkpoint_hash": _base_checkpoint_hash(hf_model),
        "method": method,
        "recovery_semantics_version": recovery_method_semantics_version(method),
        "prune_plan_name": prune_plan_name,
        "recovery_plan_name": recovery_plan_name,
        "prune_drop_indices": sorted(set(int(index) for index in prune_plan.drop)),
        "boundaries": [dict(record) for record in pair_records],
        "recovery_specs": recovery_specs,
        "ghost_stability": ghost_stability.signature_payload(),
        "ghost_token_layout": {
            "groups": list(ghost_token_group_names),
            "starts": list(ghost_token_group_starts),
        },
        "ghost_autogroup_threshold": ghost_autogroup_threshold,
        "ghost_boundary_fit_mode": ghost_boundary_fit_mode,
        "reused_boundary_prefix_checkpoint": (
            str(Path(args.reuse_boundary_prefix_checkpoint))
            if args.reuse_boundary_prefix_checkpoint is not None else ""
        ),
        "reused_boundary_prefix_checkpoint_sha256": prefix_checkpoint_sha256,
        "legacy_ridge": float(args.ridge),
        "chunk_rows": chunk_rows,
        "cache_root": str(Path(args.cache_root)) if args.cache_root else "",
        "calibration_manifest_hash": _calibration_manifest_hash(scenes),
        "cache_manifest_sha256_by_dataset": cache_manifest_sha256_by_dataset,
        "datasets": list(dataset_names),
        "max_scenes_per_dataset": int(args.max_scenes),
        "scene_limits_by_dataset": dict(scene_limits),
    }
    fit_signature = hashlib.sha256(
        json.dumps(fit_configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "backbone": adapter.name,
        "method": method,
        "prune_plan_name": prune_plan_name,
        "recovery_plan_name": recovery_plan_name,
        "num_scenes": len(scenes),
        "max_scenes_per_dataset": int(args.max_scenes),
        "scene_limits_by_dataset": dict(scene_limits),
        "num_scenes_by_cache": num_scenes_by_cache,
        "cache_dirs": [str(p) for p in cache_dirs],
        "recovery_specs": recovery_specs,
        "fit_diagnostics": fit_diagnostics,
        "fit_configuration": fit_configuration,
        "fit_signature": fit_signature,
        "recovery_semantics_version": recovery_method_semantics_version(method),
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
            "recovery_semantics_version": recovery_method_semantics_version(method),
            "fit_configuration": fit_configuration,
            "fit_signature": fit_signature,
            "fit_diagnostics": fit_diagnostics,
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
            variant=(
                "local_fit_recovery_delta"
                if method == "recovery_streamline_ffn"
                else "training_free_recovery_delta"
            ),
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

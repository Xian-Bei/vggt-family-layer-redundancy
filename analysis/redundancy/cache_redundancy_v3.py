"""Merged v3 cache + redundancy: forward once per scene, similarity on GPU, optional disk cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from tqdm import tqdm

from analysis.amp_policy import amp_dtype_for_device
from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.image_batch import prepare_cache_batch
from analysis.datasets.registry import dataset_names, get_dataset
from analysis.redundancy.similarity_v3 import (
    RedundancyAccumulator,
    compute_scene_similarity,
    metric_device,
    parse_token_regions,
    should_compute_scene_redundancy,
    unit_map_to_flat_layers_by_region,
    validate_metric_names,
    write_redundancy_outputs,
)
from analysis.schema.v3 import SCHEMA_VERSION, CacheManifestV3, stage_dir


@dataclass
class CacheRequirements:
    unit_ids: Set[str] = field(default_factory=set)
    auxiliary_ids: Set[str] = field(default_factory=set)
    require_predictions: bool = False
    require_initial_tokens: bool = False
    metadata_only: bool = False

    def merge_plan(self, plan: Any, unit_ids_all: List[str], model: Any = None) -> None:
        method = str(getattr(plan, "extra", {}).get("recovery_method", ""))
        if method in {"recovery_ghost_sequential", "recovery_ghost_interval_sequential"}:
            # Progressive Ghost recomputes activations from image paths.
            self.metadata_only = not self.unit_ids and not self.require_predictions
            return
        ids = _cache_unit_ids_from_plan_extra(plan, unit_ids_all)
        if ids is None:
            from analysis.prune_plan import required_cache_unit_ids

            if model is None:
                raise ValueError("CacheRequirements.merge_plan needs model for legacy plans")
            ids = set(required_cache_unit_ids(plan, model))
        self.unit_ids.update(ids - {"layer_-1"})
        self.require_initial_tokens |= "layer_-1" in ids
        self.auxiliary_ids.update(_auxiliary_ids_from_plan_extra(plan))
        trainable = method in {"recovery_lora", "recovery_train"} or bool(
            getattr(plan, "lora", {})
        )
        self.require_predictions |= trainable
        self.metadata_only = False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache + redundancy (v3, single forward per scene).")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True, choices=dataset_names())
    p.add_argument(
        "--output-dataset",
        type=str,
        default="",
        help="Optional output dataset tag override (e.g. scannet_dev50/scannet_test50).",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split, validated by the dataset adaptor.",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-cache", action="store_true", help="Write layer units + manifest to disk after redundancy.")
    p.add_argument(
        "--skip-redundancy",
        action="store_true",
        help="Only forward (+ optional --save-cache); skip similarity matrices and redundancy npz.",
    )
    p.add_argument("--metrics", type=str, default="cka,cosine,angular,inv_mse,cca")
    p.add_argument("--max-span", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096, help="Per-layer token subsample for similarity (0=all).")
    p.add_argument("--token-regions", type=str, default="all")
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="If set with --save-cache, only write layer units required for LoRA training "
        "(reads {backbone}.json under this directory).",
    )
    p.add_argument(
        "--prune-plan-name",
        dest="prune_plan_name",
        type=str,
        default="",
        help="Optional prune selector list (e.g., 'prune_span5 prune_span10') from prune_plan_all.json.",
    )
    p.add_argument(
        "--recovery-plan-root",
        type=Path,
        default=None,
        help="Directory with recovery plan bundles. Defaults to --prune-plan-root.",
    )
    p.add_argument(
        "--recovery-plan-name",
        type=str,
        default="",
        help="Optional recovery method names. If omitted, use all entries in recovery_plan_all.json.",
    )
    return p


def _scene_out(cache_root: Path, scene_id: str) -> Path:
    return cache_root / scene_id


def _resolve_prune_plan_refs(prune_plan_root: Path, backbone: str, plan_name: str) -> List[Union[Path, str]]:
    from analysis.prune_plan import list_plan_groups, resolve_prune_plan_path

    root = Path(prune_plan_root)
    key = str(plan_name or "").strip()
    if key:
        return [part for part in re.split(r"[\s,]+", key) if part]

    named = list_plan_groups(root)
    if named:
        return named

    # Backward compatibility: legacy single-file format (<backbone>.json).
    return [resolve_prune_plan_path(root, backbone)]


def _cache_unit_ids_from_plan_extra(plan: Any, unit_ids_all: List[str]) -> Optional[Set[str]]:
    extra = getattr(plan, "extra", None)
    if not isinstance(extra, dict):
        return None
    ids = extra.get("cache_unit_ids")
    if not isinstance(ids, list):
        return None
    parsed = {str(x) for x in ids}
    # Upgrade older plan metadata in memory: always retain each LoRA unit's own
    # output even if its serialized cache list only contains the replay seed.
    from analysis.prune_plan import unit_index_to_unit_id

    parsed.update(unit_index_to_unit_id(int(index)) for index in plan.lora.keys())
    known = set(unit_ids_all) | {"layer_-1"}
    bad = sorted(parsed - known)
    if bad:
        raise ValueError(
            f"Plan contains unknown cache_unit_ids not present in model catalog: {bad[:8]}"
            + (" ..." if len(bad) > 8 else "")
        )
    return parsed


def _auxiliary_ids_from_plan_extra(plan: Any) -> Set[str]:
    extra = getattr(plan, "extra", None)
    if not isinstance(extra, dict):
        return set()
    raw = extra.get("recovery_auxiliary_ids", [])
    if not isinstance(raw, list):
        raise ValueError("Plan recovery_auxiliary_ids must be a list")
    out = {str(x) for x in raw}
    bad = sorted(x for x in out if not x.startswith("replaceme_residual:"))
    if bad:
        raise ValueError(f"Unsupported recovery auxiliary cache ids: {bad}")
    return out


def _install_auxiliary_capture_hooks(adapter, model, auxiliary_ids: Set[str], images: torch.Tensor):
    captures: Dict[str, torch.Tensor] = {}
    handles = []
    if images.dim() == 4:
        bsz, seq_len = 1, int(images.shape[0])
    elif images.dim() == 5:
        bsz, seq_len = int(images.shape[0]), int(images.shape[1])
    else:
        raise ValueError(f"Expected cache images with 4/5 dims, got {tuple(images.shape)}")

    for auxiliary_id in sorted(auxiliary_ids):
        prefix, unit_id = auxiliary_id.split(":", 1)
        if prefix != "replaceme_residual":
            raise ValueError(f"Unsupported auxiliary capture {auxiliary_id!r}")
        block = adapter.block_for_unit(model, unit_id)
        residual_module = getattr(block, "ls2", None)
        if residual_module is None:
            raise AttributeError(f"{adapter.name} {unit_id} has no ls2 module for ReplaceMe capture")

        def make_hook(key: str, uid: str):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(tensor):
                    raise TypeError(f"ReplaceMe residual hook {key} returned {type(tensor)}")
                captures[key] = adapter.normalize_unit_activation(
                    uid,
                    tensor,
                    batch_size=bsz,
                    num_frames=seq_len,
                ).detach()

            return hook

        handles.append(residual_module.register_forward_hook(make_hook(auxiliary_id, unit_id)))
    return captures, handles


def forward_cache_with_auxiliaries(
    *,
    adapter: Any,
    model: Any,
    images: torch.Tensor,
    device: torch.device,
    auxiliary_ids: Set[str],
) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """Run the common cache forward path and capture requested auxiliaries.

    Both the persistent full-model cache and progressive recovery cache use
    this function so hook lifetime, autocast, activation normalization, and
    missing-capture validation cannot drift apart.
    """
    auxiliary_map, handles = _install_auxiliary_capture_hooks(
        adapter, model, auxiliary_ids, images
    )
    try:
        forward = adapter.forward_cache_batch(model, images, device)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(auxiliary_ids) - set(auxiliary_map))
    if missing:
        raise RuntimeError(f"Cache forward missed recovery auxiliary activations {missing}")
    return forward, auxiliary_map


def forward_selected_units_with_auxiliaries(
    *,
    adapter: Any,
    model: Any,
    images: torch.Tensor,
    device: torch.device,
    unit_ids: Set[str],
    auxiliary_ids: Set[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, torch.Tensor]]:
    """Capture only selected units and residuals, without prediction heads."""
    auxiliary_map, handles = _install_auxiliary_capture_hooks(
        adapter, model, auxiliary_ids, images
    )
    try:
        with torch.inference_mode(), torch.autocast(**autocast_device_dtype(device)):
            unit_map, meta = adapter.extract_layer_units(
                model, images, unit_ids=set(unit_ids)
            )
    finally:
        for handle in handles:
            handle.remove()
    missing_units = sorted(set(unit_ids) - set(unit_map))
    missing_aux = sorted(set(auxiliary_ids) - set(auxiliary_map))
    if missing_units or missing_aux:
        raise RuntimeError(
            f"Selected cache forward missed units={missing_units}, auxiliaries={missing_aux}"
        )
    return unit_map, meta, auxiliary_map


def _save_scene_cache(
    *,
    cache_root: Path,
    scene_id: str,
    unit_map: Dict[str, torch.Tensor],
    pred,
    meta: Dict[str, Any],
    images: torch.Tensor,
    frame_ids: List[str],
    image_paths: List[str],
    scene_meta: Dict[str, Any],
    infer_s: float,
    adapter,
    model,
    initial_tokens: Optional[torch.Tensor] = None,
    unit_ids_filter: Optional[Set[str]] = None,
    auxiliary_map: Optional[Dict[str, torch.Tensor]] = None,
    require_predictions: bool = True,
) -> Dict[str, Any]:
    out = _scene_out(cache_root, scene_id)
    layers_dir = out / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    bsz, seq_len = int(images.shape[0]), int(images.shape[1])
    image_hw = (int(images.shape[-2]), int(images.shape[-1]))
    layer_files: Dict[str, str] = {}
    auxiliary_files: Dict[str, str] = {}

    items = unit_map.items()
    need_initial_tokens = unit_ids_filter is not None and "layer_-1" in unit_ids_filter
    if unit_ids_filter is not None:
        allowed = set(unit_map.keys())
        if need_initial_tokens:
            allowed.add("layer_-1")
        missing = sorted(unit_ids_filter - allowed)
        if missing:
            raise KeyError(
                f"Scene {scene_id}: forward missing unit activations required for training cache: {missing[:8]}"
                + (" ..." if len(missing) > 8 else "")
            )
        items = ((uid, unit_map[uid]) for uid in sorted(unit_ids_filter) if uid != "layer_-1")

    for uid, tensor in items:
        t = tensor.detach().contiguous()
        lp = layers_dir / f"{uid}.pt"
        torch.save(t, lp)
        layer_files[uid] = str(lp.relative_to(cache_root))
    if need_initial_tokens:
        if isinstance(initial_tokens, torch.Tensor):
            init_t = initial_tokens.detach().contiguous()
        else:
            init_t = adapter.build_initial_tokens(model, images)
        if init_t is None:
            raise RuntimeError(
                f"Backbone {adapter.name} does not support building initial tokens for layer_-1 cache."
            )
        lp = layers_dir / "layer_-1.pt"
        torch.save(init_t, lp)
        layer_files["layer_-1"] = str(lp.relative_to(cache_root))

    for auxiliary_id, tensor in sorted((auxiliary_map or {}).items()):
        prefix, unit_id = auxiliary_id.split(":", 1)
        aux_path = out / "aux" / prefix / f"{unit_id}.pt"
        aux_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().contiguous(), aux_path)
        auxiliary_files[auxiliary_id] = str(aux_path.relative_to(cache_root))

    pred_path = out / "pred_v3.pt"
    if require_predictions:
        pred_payload: Dict[str, Any] = {}
        for key in ("pose_enc", "depth", "depth_conf", "points", "camera_poses", "conf"):
            val = getattr(pred, key, None)
            if isinstance(val, torch.Tensor):
                pred_payload[key] = val.detach().contiguous()
        torch.save(pred_payload, pred_path)

    meta_payload = {
        "scene_id": scene_id,
        "layer_files": layer_files,
        "auxiliary_files": auxiliary_files,
        "num_frames": seq_len,
        "image_hw": list(image_hw),
        "patch_start_idx": int(meta.get("patch_start_idx", 0)),
        "frame_ids": frame_ids,
        "image_paths": image_paths,
        "inference_time_ms": float(infer_s * 1000.0),
        "pred_path": str(pred_path.relative_to(cache_root)) if require_predictions else None,
        "activations_device": str(next(iter(unit_map.values())).device) if unit_map else "cpu",
    }
    if scene_meta.get("scene_dir"):
        meta_payload["scene_dir"] = str(scene_meta["scene_dir"])
    if unit_ids_filter is not None:
        meta_payload["cache_save_mode"] = "prune_plan_minimal"
        if seq_len <= 0:
            raise ValueError(f"Scene {scene_id}: invalid num_frames={seq_len} for minimal cache metadata")
        if image_hw[0] <= 0 or image_hw[1] <= 0:
            raise ValueError(f"Scene {scene_id}: invalid image_hw={image_hw} for minimal cache metadata")
    (out / "metadata_v3.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return meta_payload


def run_cache_redundancy(
    *,
    experiment: str,
    dataset: str,
    data_root: Path,
    backbone: str,
    hf_model: str = "",
    device: str = "cuda",
    max_scenes: int = 0,
    max_frames: int = 50,
    seed: int = 0,
    save_cache: bool = False,
    metrics: str = "cka,cosine,angular,inv_mse,cca",
    max_span: int = 8,
    max_tokens: int = 4096,
    token_regions: str = "all",
    progress: bool = True,
    prune_plan_root: Optional[Path] = None,
    plan_name: str = "",
    recovery_plan_root: Optional[Path] = None,
    recovery_plan_name: str = "",
    skip_redundancy: bool = False,
    split: str | None = None,
    output_dataset: str = "",
) -> Path:
    if skip_redundancy and not save_cache:
        raise ValueError("--skip-redundancy requires --save-cache (otherwise there is no output stage).")
    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    ds_plugin = get_dataset(dataset)
    split = ds_plugin.validate_split(split)
    output_dataset = str(output_dataset).strip() or ds_plugin.output_tag(split)
    torch_device = torch.device(device)
    amp_dt = amp_dtype_for_device(torch_device)
    hf = str(hf_model or adapter.default_hf_model())
    metric_names = validate_metric_names([m.strip() for m in metrics.split(",") if m.strip()])
    regions = parse_token_regions(token_regions)
    compute_redundancy = should_compute_scene_redundancy(
        skip_redundancy=skip_redundancy,
        metric_names=metric_names,
        regions=regions,
    )
    metric_dev = metric_device(device) if compute_redundancy else None

    cache_root = stage_dir(experiment, output_dataset, "cache")
    if save_cache:
        cache_root.mkdir(parents=True, exist_ok=True)

    t_run0 = time.perf_counter()
    t_load0 = time.perf_counter()
    model = adapter.load_pretrained(hf_model=hf, device=str(torch_device))
    load_s = time.perf_counter() - t_load0

    units = adapter.layer_unit_catalog(model)
    extra = adapter.cache_manifest_extra(model)
    unit_ids = [u.unit_id for u in units]

    unit_ids_filter: Optional[Set[str]] = None
    auxiliary_ids_filter: Set[str] = set()
    requirements: Optional[CacheRequirements] = None
    cache_save_policy: Optional[Dict[str, Any]] = None
    if save_cache and prune_plan_root is not None:
        from analysis.prune_plan import (
            UnitPrunePlan,
            required_cache_unit_ids,
        )
        from analysis.prune_plan.recovery_plan import (
            compose_recovery_unit_prune_plan,
            list_recovery_plan_names,
        )

        plan_refs = _resolve_prune_plan_refs(Path(prune_plan_root), adapter.name, plan_name)
        missing = [str(p) for p in plan_refs if isinstance(p, Path) and not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing prune plan for minimal cache save: {', '.join(missing)}")
        recovery_root = Path(recovery_plan_root) if recovery_plan_root is not None else Path(prune_plan_root)
        requested_recovery = [x for x in str(recovery_plan_name or "").replace(",", " ").split() if x.strip()]
        recovery_names = requested_recovery or list_recovery_plan_names(recovery_root)
        if not recovery_names and any(not isinstance(p, Path) for p in plan_refs):
            raise FileNotFoundError(
                f"No recovery plan bundles found under {recovery_root}; "
                "cache minimal mode requires both prune and recovery plans."
            )
        requirements = CacheRequirements(metadata_only=True)
        unit_ids_filter = set()
        explicit_cache_units_used = 0
        plan_labels: List[str] = []
        for plan_ref in plan_refs:
            if isinstance(plan_ref, Path):
                plan = UnitPrunePlan.load(plan_ref)
                plan_labels.append(str(plan_ref.resolve()))
                explicit_ids = _cache_unit_ids_from_plan_extra(plan, unit_ids)
                method = str(getattr(plan, "extra", {}).get("recovery_method", ""))
                sequential_ghost = method in {
                    "recovery_ghost_sequential", "recovery_ghost_interval_sequential"
                }
                if explicit_ids is not None and not sequential_ghost:
                    unit_ids_filter.update(explicit_ids)
                    explicit_cache_units_used += 1
                elif not sequential_ghost:
                    unit_ids_filter.update(required_cache_unit_ids(plan, model))
                if not sequential_ghost:
                    auxiliary_ids_filter.update(_auxiliary_ids_from_plan_extra(plan))
                requirements.require_predictions |= (
                    method in {"recovery_lora", "recovery_train"}
                    or bool(getattr(plan, "lora", {}))
                )
                requirements.metadata_only &= sequential_ghost
            else:
                for recovery_name in recovery_names:
                    plan = compose_recovery_unit_prune_plan(
                        prune_plan_root=Path(prune_plan_root),
                        recovery_plan_root=recovery_root,
                        prune_selector=str(plan_ref),
                        recovery_plan=str(recovery_name),
                    )
                    plan_labels.append(f"{plan_ref}+{recovery_name}")
                    explicit_ids = _cache_unit_ids_from_plan_extra(plan, unit_ids)
                    method = str(getattr(plan, "extra", {}).get("recovery_method", ""))
                    sequential_ghost = method in {
                        "recovery_ghost_sequential", "recovery_ghost_interval_sequential"
                    }
                    if explicit_ids is not None and not sequential_ghost:
                        unit_ids_filter.update(explicit_ids)
                        explicit_cache_units_used += 1
                    elif not sequential_ghost:
                        unit_ids_filter.update(required_cache_unit_ids(plan, model))
                    if not sequential_ghost:
                        auxiliary_ids_filter.update(_auxiliary_ids_from_plan_extra(plan))
                    requirements.require_predictions |= (
                        method in {"recovery_lora", "recovery_train"}
                        or bool(getattr(plan, "lora", {}))
                    )
                    requirements.metadata_only &= sequential_ghost

        requirements.require_initial_tokens = "layer_-1" in unit_ids_filter
        requirements.unit_ids = set(unit_ids_filter) - {"layer_-1"}
        requirements.auxiliary_ids = set(auxiliary_ids_filter)

        cache_save_policy = {
            "mode": "prune_plan_minimal_union" if len(plan_refs) > 1 else "prune_plan_minimal",
            "prune_plan_root": str(Path(prune_plan_root).resolve()),
            "recovery_plan_root": str(recovery_root.resolve()),
            "prune_plan_selectors": plan_labels,
            "units_saved": len(unit_ids_filter),
            "units_total": len(unit_ids),
            "explicit_cache_unit_ids_plans": int(explicit_cache_units_used),
            "auxiliary_ids": sorted(auxiliary_ids_filter),
            "require_predictions": bool(requirements.require_predictions),
            "require_initial_tokens": bool(requirements.require_initial_tokens),
            "metadata_only": bool(requirements.metadata_only),
        }
        if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
            print(
                f"[cache+redundancy] minimal cache save: {len(unit_ids_filter)}/{len(unit_ids)} units "
                f"from {len(plan_refs)} plan(s)",
                file=sys.stderr,
            )

    manifest_extra: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_eval_profile": asdict(ds_plugin.eval_profile()),
        "replay_amp_dtype": str(amp_dt).replace("torch.", ""),
    }
    if cache_save_policy is not None:
        manifest_extra["cache_save_policy"] = cache_save_policy
    if skip_redundancy:
        manifest_extra["redundancy_skipped"] = True

    manifest = CacheManifestV3(
        backbone=adapter.name,
        hf_model=hf,
        dataset=output_dataset,
        layer_units=units,
        num_heads=int(extra.get("num_heads", 0)),
        extra=manifest_extra,
    )

    accumulators = (
        {r: RedundancyAccumulator(metric_names, r) for r in regions} if compute_redundancy else None
    )
    scene_rows: List[Dict[str, Any]] = []
    sum_infer_s = 0.0
    sum_disk_s = 0.0

    scene_list = list(
        ds_plugin.iter_scenes(
            data_root,
            max_scenes=max_scenes,
            max_frames=max_frames,
            split=split,
        )
    )
    if max_scenes > 0:
        scene_list = scene_list[: int(max_scenes)]

    iter_label = (
        f"cache ({output_dataset})" if skip_redundancy else f"cache+redundancy ({output_dataset})"
    )
    if os.environ.get("COMPACT_LOG", "").lower() in {"1", "true", "yes"}:
        iter_label = "    scenes"
    scene_iter = tqdm(scene_list, desc=iter_label, unit="scene", file=sys.stderr) if progress else scene_list
    for batch in scene_iter:
        scene_id = batch.scene_id
        prepared = prepare_cache_batch(batch.images, max_frames=0, meta=batch.meta, backbone_adapter=adapter)
        frame_ids = list(prepared.frame_ids)
        image_paths = list(prepared.image_paths)

        if progress and hasattr(scene_iter, "set_postfix_str"):
            scene_iter.set_postfix_str(f"{scene_id} | forward", refresh=True)

        metadata_only_scene = bool(
            requirements is not None
            and requirements.metadata_only
            and not compute_redundancy
        )
        images = (
            prepared.images
            if metadata_only_scene
            else prepared.images.to(torch_device, non_blocking=False)
        )
        if torch_device.type == "cuda" and not metadata_only_scene:
            torch.cuda.synchronize(torch_device)
        t_infer0 = time.perf_counter()
        selected_only = requirements is not None and not compute_redundancy
        if selected_only and requirements.metadata_only:
            fwd = None
            auxiliary_map = {}
            unit_map = {}
            meta = {"patch_start_idx": 0, "camera_token_count": 0}
            pred = None
            fwd_initial_tokens = None
        elif selected_only and not requirements.require_predictions:
            unit_map, meta, auxiliary_map = forward_selected_units_with_auxiliaries(
                adapter=adapter,
                model=model,
                images=images,
                device=torch_device,
                unit_ids=requirements.unit_ids,
                auxiliary_ids=requirements.auxiliary_ids,
            )
            pred = None
            fwd = None
            fwd_initial_tokens = (
                adapter.build_initial_tokens(model, images)
                if requirements.require_initial_tokens
                else None
            )
        else:
            fwd, auxiliary_map = forward_cache_with_auxiliaries(
                adapter=adapter,
                model=model,
                images=images,
                device=torch_device,
                auxiliary_ids=auxiliary_ids_filter,
            )
            unit_map, meta, pred = fwd.unit_map, fwd.meta, fwd.pred
            fwd_initial_tokens = getattr(fwd, "initial_tokens", None)
        infer_s = time.perf_counter() - t_infer0
        sum_infer_s += infer_s
        patch_start_idx = int(meta.get("patch_start_idx", 0))
        camera_token_count = int(meta.get("camera_token_count", 0))

        if compute_redundancy:
            assert accumulators is not None
            by_region = unit_map_to_flat_layers_by_region(
                unit_map,
                unit_ids,
                patch_start_idx=patch_start_idx,
                camera_token_count=camera_token_count,
                regions=regions,
                max_tokens=int(max_tokens),
                seed=int(seed),
            )
            for region, (flat_layers, scene_unit_ids) in by_region.items():
                if len(flat_layers) < 2:
                    continue
                n_pairs = len(flat_layers) * (len(flat_layers) + 1) // 2
                if progress and hasattr(scene_iter, "set_postfix_str"):
                    tok_s = "all" if max_tokens <= 0 else str(max_tokens)
                    scene_iter.set_postfix_str(
                        f"{scene_id} | sim {region} 0/{n_pairs} ({tok_s} tok, {metric_dev or 'cpu'})",
                        refresh=True,
                    )
                sims = compute_scene_similarity(
                    flat_layers,
                    metric_names=metric_names,
                    metric_device=metric_dev,
                    progress=progress,
                    progress_desc=f"similarity {scene_id} [{region}]",
                )
                accumulators[region].add_scene(scene_id, flat_layers, scene_unit_ids, sims)
            del by_region

        if save_cache:
            t_disk0 = time.perf_counter()
            row = _save_scene_cache(
                cache_root=cache_root,
                scene_id=scene_id,
                unit_map=unit_map,
                pred=pred,
                meta=meta,
                images=images,
                frame_ids=frame_ids,
                image_paths=image_paths,
                scene_meta=batch.meta,
                infer_s=infer_s,
                adapter=adapter,
                model=model,
                initial_tokens=fwd_initial_tokens,
                unit_ids_filter=unit_ids_filter,
                auxiliary_map=auxiliary_map,
                require_predictions=(
                    requirements is None or requirements.require_predictions
                ),
            )
            sum_disk_s += time.perf_counter() - t_disk0
            scene_rows.append(row)

        del unit_map, meta, pred, fwd, fwd_initial_tokens, auxiliary_map, images
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

        if progress and hasattr(scene_iter, "set_postfix_str"):
            scene_iter.set_postfix_str(
                f"{scene_id} | done" + (" +disk" if save_cache else ""),
                refresh=True,
            )

    total_s = time.perf_counter() - t_run0
    out_path: Path
    if compute_redundancy:
        assert accumulators is not None
        out_paths = write_redundancy_outputs(
            experiment=experiment,
            dataset=output_dataset,
            unit_ids=unit_ids,
            accumulators=accumulators,
            token_regions=regions,
            timing_total_s=total_s,
        )
        out_path = out_paths[0]
    else:
        out_path = cache_root

    if save_cache:
        manifest.scenes = scene_rows
        manifest.extra["timing_seconds"] = {
            "load": float(load_s),
            "infer": float(sum_infer_s),
            "disk": float(sum_disk_s),
            "total": float(total_s),
        }
        manifest.save(cache_root / "cache_manifest_v3.json")
        if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
            print(f"Wrote {cache_root / 'cache_manifest_v3.json'}", file=sys.stderr)

    mode = "cache-only" if skip_redundancy else "cache+redundancy"
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
        print(
            f"[{mode}] scenes={len(scene_list)} load={load_s:.1f}s infer={sum_infer_s:.1f}s "
            f"disk={sum_disk_s:.1f}s total={total_s:.1f}s save_cache={save_cache} redundancy={compute_redundancy}",
            file=sys.stderr,
        )
    return out_path


def main() -> None:
    args = _build_parser().parse_args()
    run_cache_redundancy(
        experiment=args.experiment,
        dataset=args.dataset,
        data_root=args.data_root,
        backbone=args.backbone,
        hf_model=str(args.hf_model),
        device=str(args.device),
        max_scenes=int(args.max_scenes),
        max_frames=int(args.max_frames),
        seed=int(args.seed),
        save_cache=bool(args.save_cache),
        metrics=str(args.metrics),
        max_span=int(args.max_span),
        max_tokens=int(args.max_tokens),
        token_regions=str(args.token_regions),
        progress=True,
        prune_plan_root=args.prune_plan_root,
        plan_name=str(args.prune_plan_name),
        recovery_plan_root=args.recovery_plan_root,
        recovery_plan_name=str(args.recovery_plan_name),
        skip_redundancy=bool(args.skip_redundancy),
        split=args.split,
        output_dataset=str(args.output_dataset),
    )


if __name__ == "__main__":
    main()

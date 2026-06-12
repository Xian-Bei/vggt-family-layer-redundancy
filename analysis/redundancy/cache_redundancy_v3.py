"""Merged v3 cache + redundancy: forward once per scene, similarity on GPU, optional disk cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from tqdm import tqdm

from analysis.amp_policy import amp_dtype_for_device
from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.image_batch import prepare_cache_batch
from analysis.datasets.registry import get_dataset
from analysis.redundancy.similarity_v3 import (
    RedundancyAccumulator,
    compute_scene_similarity,
    metric_device,
    parse_token_regions,
    should_compute_scene_redundancy,
    unit_map_to_flat_layers,
    write_redundancy_outputs,
)
from analysis.schema.v3 import SCHEMA_VERSION, CacheManifestV3, dataset_output_tag, stage_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache + redundancy (v3, single forward per scene).")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True, choices=["scannet", "7scenes", "nrgbd"])
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="7scenes split (required when --dataset=7scenes).",
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
    p.add_argument("--metrics", type=str, default="cka,cosine,neg_mse")
    p.add_argument("--max-span", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096, help="Per-layer token subsample for similarity (0=all).")
    p.add_argument("--token-regions", type=str, default="all")
    p.add_argument("--similarity-threshold", type=float, default=0.95)
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="If set with --save-cache, only write layer units required for LoRA training "
        "(reads {backbone}.json under this directory).",
    )
    p.add_argument(
        "--plan-name",
        type=str,
        default="",
        help="Optional named prune plan (e.g., plan_t1_l1). If set, resolve <prune-plan-root>/<plan-name>.json.",
    )
    return p


def _scene_out(cache_root: Path, scene_id: str) -> Path:
    return cache_root / scene_id


def _resolve_prune_plan_paths(prune_plan_root: Path, backbone: str, plan_name: str) -> List[Path]:
    from analysis.prune_plan import resolve_named_prune_plan_path, resolve_prune_plan_path

    root = Path(prune_plan_root)
    key = str(plan_name or "").strip()
    if key:
        return [resolve_named_prune_plan_path(root, key)]

    # New layout: per-backbone multi-plan files (plan_t*_l*.json).
    named = sorted(root.glob("plan_t*_l*.json"))
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
    known = set(unit_ids_all) | {"layer_-1"}
    bad = sorted(parsed - known)
    if bad:
        raise ValueError(
            f"Plan contains unknown cache_unit_ids not present in model catalog: {bad[:8]}"
            + (" ..." if len(bad) > 8 else "")
        )
    return parsed


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
) -> Dict[str, Any]:
    out = _scene_out(cache_root, scene_id)
    layers_dir = out / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    bsz, seq_len = int(images.shape[0]), int(images.shape[1])
    image_hw = (int(images.shape[-2]), int(images.shape[-1]))
    layer_files: Dict[str, str] = {}

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

    pred_path = out / "pred_v3.pt"
    pred_payload: Dict[str, Any] = {}
    for key in ("pose_enc", "depth", "depth_conf", "points", "camera_poses", "conf"):
        val = getattr(pred, key, None)
        if isinstance(val, torch.Tensor):
            pred_payload[key] = val.detach().contiguous()
    torch.save(pred_payload, pred_path)

    meta_payload = {
        "scene_id": scene_id,
        "layer_files": layer_files,
        "num_frames": seq_len,
        "image_hw": list(image_hw),
        "patch_start_idx": int(meta.get("patch_start_idx", 0)),
        "frame_ids": frame_ids,
        "image_paths": image_paths,
        "inference_time_ms": float(infer_s * 1000.0),
        "pred_path": str(pred_path.relative_to(cache_root)),
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
    metrics: str = "cka,cosine,neg_mse",
    max_span: int = 8,
    max_tokens: int = 4096,
    token_regions: str = "all",
    similarity_threshold: float = 0.95,
    progress: bool = True,
    prune_plan_root: Optional[Path] = None,
    plan_name: str = "",
    skip_redundancy: bool = False,
    split: str | None = None,
) -> Path:
    if skip_redundancy and not save_cache:
        raise ValueError("--skip-redundancy requires --save-cache (otherwise there is no output stage).")
    if dataset == "7scenes" and not split:
        raise ValueError("7scenes requires split='train' or 'test'")

    output_dataset = dataset_output_tag(dataset, split)
    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    ds_plugin = get_dataset(dataset)
    torch_device = torch.device(device)
    amp_dt = amp_dtype_for_device(torch_device)
    hf = str(hf_model or adapter.default_hf_model())
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
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
    cache_save_policy: Optional[Dict[str, Any]] = None
    if save_cache and prune_plan_root is not None:
        from analysis.prune_plan import (
            UnitPrunePlan,
            required_cache_unit_ids,
        )

        plan_paths = _resolve_prune_plan_paths(Path(prune_plan_root), adapter.name, plan_name)
        missing = [str(p) for p in plan_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing prune plan for minimal cache save: {', '.join(missing)}")
        unit_ids_filter = set()
        explicit_cache_units_used = 0
        for plan_path in plan_paths:
            plan = UnitPrunePlan.load(plan_path)
            explicit_ids = _cache_unit_ids_from_plan_extra(plan, unit_ids)
            if explicit_ids is not None:
                unit_ids_filter.update(explicit_ids)
                explicit_cache_units_used += 1
            else:
                unit_ids_filter.update(required_cache_unit_ids(plan, model))

        cache_save_policy = {
            "mode": "prune_plan_minimal_union" if len(plan_paths) > 1 else "prune_plan_minimal",
            "prune_plan_root": str(Path(prune_plan_root).resolve()),
            "prune_plan_files": [str(p.resolve()) for p in plan_paths],
            "units_saved": len(unit_ids_filter),
            "units_total": len(unit_ids),
            "explicit_cache_unit_ids_plans": int(explicit_cache_units_used),
        }
        print(
            f"[cache+redundancy] minimal cache save: {len(unit_ids_filter)}/{len(unit_ids)} units "
            f"from {len(plan_paths)} plan(s)",
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

    accumulator = RedundancyAccumulator(metric_names, regions) if compute_redundancy else None
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
    scene_iter = tqdm(scene_list, desc=iter_label, unit="scene", file=sys.stderr) if progress else scene_list
    for batch in scene_iter:
        scene_id = batch.scene_id
        prepared = prepare_cache_batch(batch.images, max_frames=0, meta=batch.meta)
        images = prepared.images
        frame_ids = list(prepared.frame_ids)
        image_paths = list(prepared.image_paths)

        if progress and hasattr(scene_iter, "set_postfix_str"):
            scene_iter.set_postfix_str(f"{scene_id} | forward", refresh=True)

        t_infer0 = time.perf_counter()
        fwd = adapter.forward_cache_batch(model, images, torch_device)
        infer_s = time.perf_counter() - t_infer0
        sum_infer_s += infer_s
        unit_map, meta, pred = fwd.unit_map, fwd.meta, fwd.pred
        fwd_initial_tokens = getattr(fwd, "initial_tokens", None)
        patch_start_idx = int(meta.get("patch_start_idx", 0))

        if compute_redundancy:
            assert accumulator is not None
            flat_layers, scene_unit_ids = unit_map_to_flat_layers(
                unit_map,
                unit_ids,
                patch_start_idx=patch_start_idx,
                regions=regions,
                max_tokens=int(max_tokens),
                seed=int(seed),
            )
            if len(flat_layers) >= 2:
                n_pairs = len(flat_layers) * (len(flat_layers) + 1) // 2
                if progress and hasattr(scene_iter, "set_postfix_str"):
                    tok_s = "all" if max_tokens <= 0 else str(max_tokens)
                    scene_iter.set_postfix_str(
                        f"{scene_id} | sim 0/{n_pairs} ({tok_s} tok, {metric_dev or 'cpu'})",
                        refresh=True,
                    )
                sims = compute_scene_similarity(
                    flat_layers,
                    metric_names=metric_names,
                    metric_device=metric_dev,
                    progress=progress,
                    progress_desc=f"similarity {scene_id}",
                )
                accumulator.add_scene(scene_id, flat_layers, scene_unit_ids, sims)
            del flat_layers

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
            )
            sum_disk_s += time.perf_counter() - t_disk0
            scene_rows.append(row)

        del unit_map, meta, pred, fwd, fwd_initial_tokens, images
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
        assert accumulator is not None
        source_cache = str(cache_root) if save_cache else "(memory-only)"
        out_path = write_redundancy_outputs(
            experiment=experiment,
            dataset=output_dataset,
            backbone=adapter.name,
            unit_ids=unit_ids,
            accumulator=accumulator,
            max_span=int(max_span),
            max_tokens=int(max_tokens),
            metric_device=metric_dev,
            source_cache=source_cache,
            timing_total_s=total_s,
        )
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
        print(f"Wrote {cache_root / 'cache_manifest_v3.json'}", file=sys.stderr)

    mode = "cache-only" if skip_redundancy else "cache+redundancy"
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
        similarity_threshold=float(args.similarity_threshold),
        progress=True,
        prune_plan_root=args.prune_plan_root,
        plan_name=str(args.plan_name),
        skip_redundancy=bool(args.skip_redundancy),
        split=args.split,
    )


if __name__ == "__main__":
    main()

"""CKA redundancy from fresh pruned-model inference without persisting activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.image_batch import prepare_cache_batch
from analysis.datasets.registry import dataset_names, get_dataset
from analysis.prune_plan import (
    UnitPrunePlan,
    interval_bounds_from_selector,
    is_interval_selector,
    load_named_prune_plan,
    prune_actions_from_plan,
)
from analysis.redundancy.similarity_v3 import (
    RedundancyAccumulator,
    _merge_existing_redundancy_npz,
    compute_scene_similarity,
    metric_device,
    parse_token_regions,
    redundancy_graph_filename,
    unit_map_to_flat_layers_by_region,
    validate_metric_names,
)
from analysis.schema.v3 import SCHEMA_VERSION, stage_dir

REMAINING_REDUNDANCY_STAGE = "remaining_redundancy"
MANIFEST_SCHEMA = "remaining_redundancy_v1"
_SPAN_SELECTOR_RE = re.compile(r"^prune_span(?P<count>\d+)$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one prune plan before inference and compute redundancy from each "
            "fresh pruned-model forward pass without writing activations to disk."
        )
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_names())
    parser.add_argument("--output-dataset", type=str, default="")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="cka")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--token-regions", type=str, default="special")
    parser.add_argument("--prune-plan-root", type=Path, required=True)
    parser.add_argument("--prune-plan-name", type=str, required=True)
    parser.add_argument("--expected-drop-count", type=int, required=True)
    parser.add_argument("--base-selection-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit output directory. Defaults to "
            "<OUTPUT_ROOT>/<experiment>/<dataset>/remaining_redundancy."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _validate_base_selector(selector: str, expected_drop_count: int) -> None:
    name = str(selector).strip()
    count = int(expected_drop_count)
    if is_interval_selector(name):
        start, end = interval_bounds_from_selector(name)
        actual = end - start + 1
        if actual != count:
            raise ValueError(
                f"Prune selector {name!r} drops {actual} units, expected {count}"
            )
        return
    match = _SPAN_SELECTOR_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "Remaining redundancy requires an exact interval selector such as "
            f"i8_j20 or a legacy prune_span{count} selector; got {selector!r}"
        )
    selector_count = int(match.group("count"))
    if selector_count != count:
        raise ValueError(
            f"Prune selector {name!r} encodes {selector_count} drops, expected {count}"
        )


def _load_base_prune_plan(
    *,
    root: Path,
    selector: str,
    backbone: str,
    num_units: int,
) -> UnitPrunePlan:
    name = str(selector).strip()
    if not is_interval_selector(name):
        return load_named_prune_plan(Path(root), name)
    start, end = interval_bounds_from_selector(name)
    plan = UnitPrunePlan(
        backbone=str(backbone),
        dataset="",
        num_units=int(num_units),
        drop=list(range(start, end + 1)),
        drop_length=end - start + 1,
        lora={},
        source_cache="",
        extra={
            "selector": name,
            "interval_selector": name,
            "closed_interval": True,
            "start": start,
            "end": end,
            "source": "first_skip_eval_gt_single_interval_knee",
        },
    )
    plan.validate()
    return plan


def _selection_provenance(
    path: Path | None,
    *,
    selector: str,
    expected_drop_count: int,
) -> Dict[str, Any]:
    if path is None:
        return {}
    selection_path = Path(path)
    raw_bytes = selection_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Base selection must be a JSON object: {selection_path}")
    if str(payload.get("selector", "")).strip() != str(selector).strip():
        raise ValueError(f"{selection_path}: selector does not match {selector!r}")
    if int(payload.get("drop_count", -1)) != int(expected_drop_count):
        raise ValueError(
            f"{selection_path}: drop_count does not match {expected_drop_count}"
        )
    return {
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_schema": str(payload.get("schema_version", "")),
        "selection_method": str(payload.get("method", "")),
        "selection_datasets": list(payload.get("datasets", [])),
        "selection_threshold": payload.get("threshold"),
        "selection_mean_normalized_degradation": payload.get(
            "mean_normalized_degradation"
        ),
    }


def _manifest_signature(manifest: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "schema",
        "schema_version",
        "backbone",
        "dataset",
        "expected_drop_count",
        "dropped_unit_ids",
        "remaining_unit_ids",
        "token_regions",
        "max_tokens",
        "seed",
        "max_frames",
        "scene_ids",
    )
    return {key: manifest.get(key) for key in keys}


def _prepare_output_dir(output_dir: Path, expected_manifest: Dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not manifest_path.is_file():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} already contains data but has no manifest.json; "
                "refusing to mix an incomplete or unrelated run."
            )
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _manifest_signature(existing) != _manifest_signature(expected_manifest):
        raise ValueError(
            f"Existing remaining-redundancy manifest at {manifest_path} is incompatible "
            "with this run. Use a clean output directory."
        )


def _write_graphs(
    *,
    output_dir: Path,
    remaining_unit_ids: Sequence[str],
    accumulators: Dict[str, RedundancyAccumulator],
    token_regions: Sequence[str],
    provenance: Dict[str, Any],
) -> List[Path]:
    written: List[Path] = []
    for region in token_regions:
        accumulator = accumulators.get(str(region))
        if accumulator is None or not accumulator.scene_ids_processed:
            raise RuntimeError(
                f"No {region!r} activations were accumulated from the pruned-model inference."
            )
        payload = accumulator.finalize_npz(fallback_unit_ids=remaining_unit_ids)
        payload.update(
            {
                "analysis_stage": np.array([REMAINING_REDUNDANCY_STAGE]),
                "prune_plan_name": np.array([str(provenance["prune_plan_name"])]),
                "dropped_unit_ids": np.array(provenance["dropped_unit_ids"], dtype=object),
                "remaining_unit_ids": np.array(remaining_unit_ids, dtype=object),
            }
        )
        path = output_dir / redundancy_graph_filename(str(region))
        payload = _merge_existing_redundancy_npz(path, payload)
        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez(temporary, **payload)
        temporary.replace(path)
        written.append(path)
    return written


def run_remaining_redundancy(
    *,
    experiment: str,
    dataset: str,
    data_root: Path,
    backbone: str,
    prune_plan_root: Path,
    prune_plan_name: str,
    expected_drop_count: int,
    base_selection_path: Path | None = None,
    hf_model: str = "",
    device: str = "cuda",
    max_scenes: int = 0,
    max_frames: int = 50,
    seed: int = 0,
    metrics: str = "cka",
    max_tokens: int = 0,
    token_regions: str = "special",
    split: str | None = None,
    output_dataset: str = "",
    output_dir: Path | None = None,
    progress: bool = True,
) -> Path:
    _validate_base_selector(prune_plan_name, expected_drop_count)
    if int(expected_drop_count) <= 0:
        raise ValueError("expected_drop_count must be positive")
    selection_provenance = _selection_provenance(
        base_selection_path,
        selector=str(prune_plan_name),
        expected_drop_count=int(expected_drop_count),
    )

    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    dataset_plugin = get_dataset(dataset)
    split = dataset_plugin.validate_split(split)
    output_dataset = str(output_dataset).strip() or dataset_plugin.output_tag(split)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("Remaining redundancy must run on a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    metric_names = validate_metric_names(
        [name.strip() for name in str(metrics).split(",") if name.strip()]
    )
    regions = parse_token_regions(token_regions)
    metric_dev = metric_device(str(device))

    # The prune plan is loaded and applied before dataset iteration and before
    # any scene activation hooks are installed.
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(torch_device),
    )
    full_catalog = adapter.layer_unit_catalog(model)
    full_unit_ids = [unit.unit_id for unit in full_catalog]
    plan = _load_base_prune_plan(
        root=Path(prune_plan_root),
        selector=str(prune_plan_name),
        backbone=adapter.name,
        num_units=len(full_unit_ids),
    )
    if str(plan.backbone).strip() and str(plan.backbone).strip() != adapter.name:
        raise ValueError(
            f"Plan {prune_plan_name!r} targets backbone {plan.backbone!r}, "
            f"but the loaded backbone is {adapter.name!r}"
        )
    if int(plan.num_units) > 0 and int(plan.num_units) != len(full_unit_ids):
        raise ValueError(
            f"Plan {prune_plan_name!r} expects {plan.num_units} units, "
            f"but the loaded model exposes {len(full_unit_ids)}"
        )
    dropped_indices = sorted({int(index) for index in plan.drop})
    if len(dropped_indices) != int(expected_drop_count):
        raise ValueError(
            f"Plan {prune_plan_name!r} drops {len(dropped_indices)} units, "
            f"expected {expected_drop_count}"
        )
    bad_indices = [index for index in dropped_indices if index < 0 or index >= len(full_unit_ids)]
    if bad_indices:
        raise IndexError(
            f"Plan {prune_plan_name!r} contains unit indices outside "
            f"[0, {len(full_unit_ids) - 1}]: {bad_indices}"
        )
    dropped_unit_ids = [full_unit_ids[index] for index in dropped_indices]
    dropped_set = set(dropped_unit_ids)
    remaining_unit_ids = [unit_id for unit_id in full_unit_ids if unit_id not in dropped_set]
    if len(remaining_unit_ids) < 2:
        raise ValueError("At least two units must remain to compute pairwise CKA")

    dropped_blocks_before = {
        unit_id: adapter.block_for_unit(model, unit_id) for unit_id in dropped_unit_ids
    }
    drop_actions, _ = prune_actions_from_plan(plan)
    adapter.apply_prune_plan(model, drop_actions)
    unchanged_dropped_blocks = [
        unit_id
        for unit_id, old_block in dropped_blocks_before.items()
        if adapter.block_for_unit(model, unit_id) is old_block
    ]
    if unchanged_dropped_blocks:
        raise RuntimeError(
            "Prune plan did not structurally replace the requested blocks before inference: "
            f"{unchanged_dropped_blocks}"
        )
    model.eval()

    scene_list = list(
        dataset_plugin.iter_scenes(
            Path(data_root),
            max_scenes=int(max_scenes),
            max_frames=int(max_frames),
            split=split,
        )
    )
    if int(max_scenes) > 0:
        scene_list = scene_list[: int(max_scenes)]
    scene_ids = [str(batch.scene_id) for batch in scene_list]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError(f"Dataset produced duplicate scene ids: {scene_ids}")

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else stage_dir(str(experiment), output_dataset, REMAINING_REDUNDANCY_STAGE)
    )
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backbone": adapter.name,
        "dataset": output_dataset,
        "prune_plan_root": str(Path(prune_plan_root).resolve()),
        "prune_plan_name": str(prune_plan_name),
        "expected_drop_count": int(expected_drop_count),
        "dropped_unit_ids": dropped_unit_ids,
        "remaining_unit_ids": remaining_unit_ids,
        "metrics": metric_names,
        "token_regions": regions,
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "max_frames": int(max_frames),
        "scene_ids": scene_ids,
        "activations_persisted": False,
        "scenes": [],
        **selection_provenance,
    }
    _prepare_output_dir(resolved_output_dir, manifest)
    existing_manifest_path = resolved_output_dir / "manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.is_file()
        else {}
    )

    accumulators = {
        region: RedundancyAccumulator(metric_names, region) for region in regions
    }
    scene_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    scene_iter = (
        tqdm(scene_list, desc=f"remaining redundancy ({output_dataset})", unit="scene")
        if progress
        else scene_list
    )
    for batch in scene_iter:
        prepared = prepare_cache_batch(
            batch.images,
            max_frames=0,
            meta=batch.meta,
            backbone_adapter=adapter,
        )
        images = prepared.images.to(torch_device, non_blocking=False)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        infer_started = time.perf_counter()

        # This forward runs after structural pruning. Only surviving units are
        # requested from the adapter, so deleted block activations are not saved.
        unit_map, meta = adapter.extract_layer_units(
            model,
            images,
            unit_ids=set(remaining_unit_ids),
        )
        infer_seconds = time.perf_counter() - infer_started
        unit_map = {
            unit_id: unit_map[unit_id]
            for unit_id in remaining_unit_ids
            if unit_id in unit_map
        }
        missing = [
            unit_id for unit_id in remaining_unit_ids if unit_id not in unit_map
        ]
        extra = sorted(set(unit_map) - set(remaining_unit_ids))
        if missing or extra:
            raise RuntimeError(
                f"Scene {batch.scene_id!r} activation layout mismatch: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )

        by_region = unit_map_to_flat_layers_by_region(
            unit_map,
            remaining_unit_ids,
            patch_start_idx=int(meta.get("patch_start_idx", 0)),
            camera_token_count=int(meta.get("camera_token_count", 0)),
            regions=regions,
            max_tokens=int(max_tokens),
            seed=int(seed),
        )
        for region in regions:
            if region not in by_region:
                raise RuntimeError(
                    f"Scene {batch.scene_id!r} produced no {region!r} tokens "
                    "from the pruned-model activations"
                )
            flat_layers, scene_unit_ids = by_region[region]
            similarities = compute_scene_similarity(
                flat_layers,
                metric_names=metric_names,
                metric_device=metric_dev,
                progress=progress,
                progress_desc=f"remaining similarity {batch.scene_id} [{region}]",
            )
            accumulators[region].add_scene(
                str(batch.scene_id),
                flat_layers,
                scene_unit_ids,
                similarities,
            )

        scene_rows.append(
            {
                "scene_id": str(batch.scene_id),
                "num_frames": int(images.shape[1]),
                "frame_ids": list(prepared.frame_ids),
                "image_paths": list(prepared.image_paths),
                "inference_seconds": float(infer_seconds),
            }
        )
        del unit_map, by_region, images
        torch.cuda.empty_cache()

    if not scene_rows:
        raise RuntimeError("Dataset produced no scenes; no activations or CKA were written")

    graph_paths = _write_graphs(
        output_dir=resolved_output_dir,
        remaining_unit_ids=remaining_unit_ids,
        accumulators=accumulators,
        token_regions=regions,
        provenance=manifest,
    )
    existing_metrics = [
        str(name) for name in existing_manifest.get("metrics", []) if str(name)
    ]
    manifest["metrics"] = list(dict.fromkeys([*existing_metrics, *metric_names]))
    manifest["scenes"] = scene_rows
    manifest["graph_files"] = [str(path.relative_to(resolved_output_dir)) for path in graph_paths]
    manifest["timing_seconds"] = {"total": float(time.perf_counter() - started)}
    manifest_path = resolved_output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
        print(f"Wrote {manifest_path}", file=sys.stderr)
    return resolved_output_dir


def main() -> None:
    args = _build_parser().parse_args()
    run_remaining_redundancy(
        experiment=str(args.experiment),
        dataset=str(args.dataset),
        data_root=Path(args.data_root),
        backbone=str(args.backbone),
        prune_plan_root=Path(args.prune_plan_root),
        prune_plan_name=str(args.prune_plan_name),
        expected_drop_count=int(args.expected_drop_count),
        base_selection_path=(
            Path(args.base_selection_path)
            if args.base_selection_path is not None
            else None
        ),
        hf_model=str(args.hf_model),
        device=str(args.device),
        max_scenes=int(args.max_scenes),
        max_frames=int(args.max_frames),
        seed=int(args.seed),
        metrics=str(args.metrics),
        max_tokens=int(args.max_tokens),
        token_regions=str(args.token_regions),
        split=args.split,
        output_dataset=str(args.output_dataset),
        output_dir=args.output_dir,
        progress=not bool(args.no_progress),
    )


if __name__ == "__main__":
    main()

"""Brute-force skip-GT evaluation on a structurally pruned backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import EvalSceneContext
from analysis.backbones.registry import get_backbone
from analysis.datasets.registry import dataset_names, get_dataset
from analysis.prune_plan import (
    UnitPrunePlan,
    interval_bounds_from_selector,
    is_interval_selector,
    load_named_prune_plan,
    prune_actions_from_plan,
)
from analysis.redundancy import skip_gt_eval_v3 as base
from analysis.redundancy.remaining_skip_layout import (
    REMAINING_INTERNAL_DIR,
    build_remaining_span_records,
    evaluated_logical_spans,
    load_reused_span_metrics,
    materialize_evaluated_intervals,
    update_remaining_index,
    upgraded_manifest_config,
)
from analysis.redundancy.remaining_skip_rebase import resolve_plan_interval
from analysis.schema.v3 import stage_dir


PRUNING_STAGE = "eval/pruning"
_SPAN_SELECTOR_RE = re.compile(r"^prune_span(?P<count>\d+)$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Brute-force skip-GT evaluation after applying one selected base interval."
        )
    )
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_names())
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="vggt")
    parser.add_argument("--output-dataset", type=str, default="")
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--max-span", type=int, default=0)
    parser.add_argument("--mode", type=str, default="mixed_units", choices=["mixed_units"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prune-plan-root", type=Path, required=True)
    parser.add_argument("--prune-plan-name", type=str, required=True)
    parser.add_argument("--expected-drop-count", type=int, required=True)
    parser.add_argument("--base-selection-path", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
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
            "Remaining skip evaluation requires an exact interval selector such as "
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
        raise ValueError(
            f"{selection_path}: selector does not match {selector!r}"
        )
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


def _prepare_manifest_allowing_selector_alias(
    out_root: Path,
    *,
    config: Dict[str, Any],
    resume: bool,
) -> None:
    manifest_path = Path(out_root) / "skip_eval_manifest_v3.json"
    if resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_config = (
            existing.get("config", {}) if isinstance(existing, dict) else {}
        )
        if isinstance(existing_config, dict):
            left = dict(existing_config)
            right = dict(config)
            same_base = (
                str(left.get("base_selector", ""))
                == str(right.get("base_selector", ""))
                and list(
                    left.get("base_prune_plan", {}).get("dropped_unit_ids", [])
                )
                == list(
                    right.get("base_prune_plan", {}).get("dropped_unit_ids", [])
                )
            )
            if same_base:
                left["base_prune_plan"] = right.get("base_prune_plan")
                if left == right:
                    existing["config"] = right
                    existing["fingerprint"] = base._manifest_fingerprint(right)
                    base._atomic_write_json(manifest_path, existing)
    base._prepare_checkpoint_manifest(out_root, config=config, resume=bool(resume))


def build_remaining_span_mapping(
    *,
    full_unit_ids: Sequence[str],
    dropped_indices: Sequence[int],
    max_span: int,
) -> Tuple[List[str], List[Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Map logical remaining-layer spans to physical full-catalog endpoints."""
    dropped = {int(index) for index in dropped_indices}
    remaining_physical = [
        index for index in range(len(full_unit_ids)) if index not in dropped
    ]
    remaining_ids = [str(full_unit_ids[index]) for index in remaining_physical]
    logical_spans = list(base.iter_unit_spans(len(remaining_ids), max_span=int(max_span)))
    physical_to_logical: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for logical_i, logical_j in logical_spans:
        physical_span = (
            int(remaining_physical[logical_i]),
            int(remaining_physical[logical_j]),
        )
        if physical_span in physical_to_logical:
            raise RuntimeError(f"Duplicate physical skip span mapping: {physical_span}")
        physical_to_logical[physical_span] = (int(logical_i), int(logical_j))
    return remaining_ids, logical_spans, physical_to_logical


def _iter_logical_predictions(
    adapter: Any,
    model: torch.nn.Module,
    images: torch.Tensor,
    physical_to_logical: Dict[Tuple[int, int], Tuple[int, int]],
) -> Iterable[Tuple[int, int, Any]]:
    physical_spans = list(physical_to_logical)
    for physical_i, physical_j, prediction in adapter.iter_skip_span_predictions(
        model, images, physical_spans
    ):
        physical_span = (int(physical_i), int(physical_j))
        if physical_span not in physical_to_logical:
            raise RuntimeError(
                f"Adaptor yielded unrequested physical skip span {physical_span}"
            )
        logical_i, logical_j = physical_to_logical[physical_span]
        yield logical_i, logical_j, prediction


def _append_provenance(path: Path, provenance: Dict[str, Any]) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["analysis_stage"] = "remaining_skip_eval"
        payload["base_prune_plan"] = provenance
        base._atomic_write_json(path, payload)


def run_remaining_skip_gt_eval(
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
    output_dataset: str = "",
    device: str = "cuda",
    max_scenes: int = 0,
    max_frames: int = 50,
    max_span: int = 0,
    mode: str = "mixed_units",
    output_dir: Optional[Path] = None,
    split: str | None = None,
    progress: bool = True,
    timing: Optional[bool] = None,
    resume: bool = True,
) -> Path:
    _validate_base_selector(prune_plan_name, expected_drop_count)
    if int(expected_drop_count) <= 0:
        raise ValueError("expected_drop_count must be positive")

    timing_on = base._timing_enabled(timing)
    run_t0 = time.perf_counter()
    adapter = get_backbone(backbone)
    apply_pythonpath(adapter.name)
    if not adapter.supports_incremental_skip_eval():
        raise RuntimeError(
            f"Backbone {adapter.name!r} does not implement incremental skip evaluation"
        )
    dataset_plugin = get_dataset(dataset)
    split = dataset_plugin.validate_split(split)
    tagged_dataset = str(output_dataset).strip() or dataset_plugin.output_tag(split)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("Remaining skip evaluation must run on a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    hf = str(hf_model or adapter.default_hf_model())
    pruning_root = stage_dir(experiment, tagged_dataset, PRUNING_STAGE)

    base._latency_log(
        f"[remaining_skip_eval] start dataset={tagged_dataset} backbone={adapter.name} "
        f"plan={prune_plan_name} max_scenes={max_scenes} max_frames={max_frames} "
        f"max_span={max_span}"
    )
    base._latency_log("[remaining_skip_eval] load_pretrained ...")
    load_t0 = time.perf_counter()
    model = adapter.load_pretrained(hf_model=hf, device=str(torch_device), eval_mode=True)
    full_catalog = adapter.layer_unit_catalog(model)
    full_unit_ids = [unit.unit_id for unit in full_catalog]
    plan = _load_base_prune_plan(
        root=Path(prune_plan_root),
        selector=str(prune_plan_name),
        backbone=str(adapter.name),
        num_units=len(full_unit_ids),
    )
    if str(plan.backbone).strip() and str(plan.backbone).strip() != adapter.name:
        raise ValueError(
            f"Plan {prune_plan_name!r} targets {plan.backbone!r}, loaded {adapter.name!r}"
        )
    if int(plan.num_units) > 0 and int(plan.num_units) != len(full_unit_ids):
        raise ValueError(
            f"Plan expects {plan.num_units} units, model exposes {len(full_unit_ids)}"
        )
    dropped_indices = sorted({int(index) for index in plan.drop})
    if len(dropped_indices) != int(expected_drop_count):
        raise ValueError(
            f"Plan {prune_plan_name!r} drops {len(dropped_indices)} units, "
            f"expected {expected_drop_count}"
        )
    invalid = [
        index for index in dropped_indices if index < 0 or index >= len(full_unit_ids)
    ]
    if invalid:
        raise IndexError(f"Prune plan indices outside the model catalog: {invalid}")
    dropped_unit_ids = [full_unit_ids[index] for index in dropped_indices]
    _, _, base_pruning_selector = resolve_plan_interval(
        plan,
        full_unit_ids=full_unit_ids,
        expected_dropped_unit_ids=dropped_unit_ids,
    )
    base_pruning_summary = (
        pruning_root
        / base_pruning_selector
        / "summary.json"
    )
    if not base_pruning_summary.is_file():
        raise FileNotFoundError(
            "The exact base pruning result required by the prune plan is missing: "
            f"{base_pruning_summary}"
        )
    out_root = (
        Path(output_dir)
        if output_dir is not None
        else pruning_root / REMAINING_INTERNAL_DIR / base_pruning_selector
    )
    out_root.mkdir(parents=True, exist_ok=True)
    old_dropped_blocks = {
        unit_id: adapter.block_for_unit(model, unit_id) for unit_id in dropped_unit_ids
    }
    drop_actions, _ = prune_actions_from_plan(plan)
    adapter.apply_prune_plan(model, drop_actions)
    unchanged = [
        unit_id
        for unit_id, old_block in old_dropped_blocks.items()
        if adapter.block_for_unit(model, unit_id) is old_block
    ]
    if unchanged:
        raise RuntimeError(
            "Prune plan did not structurally replace blocks before evaluation: "
            f"{unchanged}"
        )
    model.eval()
    remaining_unit_ids, all_spans, physical_to_logical = build_remaining_span_mapping(
        full_unit_ids=full_unit_ids,
        dropped_indices=dropped_indices,
        max_span=int(max_span),
    )
    span_records = build_remaining_span_records(
        full_unit_ids=full_unit_ids,
        remaining_unit_ids=remaining_unit_ids,
        dropped_indices=dropped_indices,
        logical_spans=all_spans,
        physical_to_logical=physical_to_logical,
    )
    spans = evaluated_logical_spans(span_records)
    n_units = len(remaining_unit_ids)
    if n_units < 1:
        raise RuntimeError("No units remain after applying the prune plan")
    base._latency_log(
        f"[remaining_skip_eval] pruned before dataset iteration: full={len(full_unit_ids)} "
        f"dropped={len(dropped_unit_ids)} remaining={n_units} "
        f"candidates={len(all_spans)} reused={len(all_spans) - len(spans)} "
        f"evaluated={len(spans)} "
        f"base_pruning={base_pruning_selector} "
        f"load+prune={time.perf_counter() - load_t0:.2f}s"
    )

    upper_metrics, upper_scene_metrics = base.load_upper_eval_or_fail(
        experiment=experiment,
        tagged_dataset=tagged_dataset,
        require_depth=dataset in {"nuscenes", "waymo"}
        and not adapter.periodic_eval_skip_depth(),
    )
    metrics = [metric for metric in base.SKIP_EVAL_METRICS if metric in upper_metrics]
    if not metrics:
        raise RuntimeError(f"No skip-eval metrics overlap upper eval: {upper_metrics}")

    provenance: Dict[str, Any] = {
        "prune_plan_root": str(Path(prune_plan_root).resolve()),
        "prune_plan_name": str(prune_plan_name),
        "expected_drop_count": int(expected_drop_count),
        "dropped_unit_ids": list(dropped_unit_ids),
        "remaining_unit_ids": list(remaining_unit_ids),
        **_selection_provenance(
            base_selection_path,
            selector=str(prune_plan_name),
            expected_drop_count=int(expected_drop_count),
        ),
    }
    manifest_config: Dict[str, Any] = {
        "analysis_stage": "remaining_skip_eval",
        "backbone": str(adapter.name),
        "dataset": str(tagged_dataset),
        "source_dataset": str(dataset),
        "data_root": str(Path(data_root).resolve()),
        "split": split,
        "hf_model": hf,
        "max_scenes": int(max_scenes),
        "max_frames": int(max_frames),
        "max_span": int(max_span),
        "mode": str(mode),
        "metrics": list(metrics),
        "full_unit_ids": list(full_unit_ids),
        "unit_ids": list(remaining_unit_ids),
        "spans": [[int(i), int(j)] for i, j in all_spans],
        "physical_spans": [
            [int(i), int(j)] for i, j in physical_to_logical
        ],
        "skip_depth": bool(adapter.periodic_eval_skip_depth()),
        "base_prune_plan": provenance,
    }
    manifest_config = upgraded_manifest_config(
        manifest_config,
        records=span_records,
    )
    direct_manifest_path = pruning_root / "skip_eval_manifest_v3.json"
    if not direct_manifest_path.is_file():
        raise FileNotFoundError(
            f"Single-interval skip-eval manifest is missing: {direct_manifest_path}"
        )
    direct_manifest = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
    direct_config = (
        direct_manifest.get("config", {}) if isinstance(direct_manifest, dict) else {}
    )
    compatibility_keys = (
        "backbone",
        "dataset",
        "source_dataset",
        "data_root",
        "split",
        "hf_model",
        "max_scenes",
        "max_frames",
        "mode",
        "metrics",
    )
    mismatches = {
        key: (manifest_config.get(key), direct_config.get(key))
        for key in compatibility_keys
        if manifest_config.get(key) != direct_config.get(key)
    }
    if list(direct_config.get("unit_ids", [])) != list(full_unit_ids):
        mismatches["unit_ids"] = (
            list(full_unit_ids),
            list(direct_config.get("unit_ids", [])),
        )
    if mismatches:
        raise ValueError(
            "Single-interval results are incompatible with remaining skip eval: "
            f"{mismatches}"
        )
    reused_span_metrics = load_reused_span_metrics(
        pruning_root=pruning_root,
        records=span_records,
    )
    expected_scene_ids = set(upper_scene_metrics)
    reused_scene_ids = set(reused_span_metrics)
    if reused_scene_ids != expected_scene_ids:
        raise ValueError(
            "Reusable single-interval scene coverage does not match upper evaluation: "
            f"missing={sorted(expected_scene_ids - reused_scene_ids)[:8]}, "
            f"extra={sorted(reused_scene_ids - expected_scene_ids)[:8]}"
        )
    expected_reused_count = len(all_spans) - len(spans)
    incomplete_reused = {
        scene_id: len(scene_spans)
        for scene_id, scene_spans in reused_span_metrics.items()
        if len(scene_spans) != expected_reused_count
    }
    if incomplete_reused:
        raise ValueError(
            "Reusable single-interval coverage is incomplete: "
            f"{dict(list(incomplete_reused.items())[:8])}, "
            f"expected={expected_reused_count}"
        )
    _prepare_manifest_allowing_selector_alias(
        out_root,
        config=manifest_config,
        resume=bool(resume),
    )

    profile = dataset_plugin.eval_profile()
    eval_context = EvalSceneContext(
        device=torch_device,
        profile=profile,
        skip_depth=bool(adapter.periodic_eval_skip_depth()),
        export_vis_2d=False,
        export_vis_3d=False,
        vis_save_root=None,
        vis_max_points=0,
        backbone_name=str(adapter.name),
        dataset_name=str(dataset_plugin.name),
    )
    # Dataset access begins only after the model has been structurally pruned.
    total_scenes = dataset_plugin.count_scenes(
        data_root, max_scenes=int(max_scenes), split=split
    )
    scene_source = dataset_plugin.iter_scenes(
        data_root,
        max_scenes=int(max_scenes),
        max_frames=int(max_frames),
        split=split,
    )
    scene_iter = tqdm(
        scene_source,
        total=total_scenes,
        desc=f"remaining_skip_eval {adapter.name} ({tagged_dataset})",
        unit="scene",
        file=sys.stderr,
        dynamic_ncols=True,
        disable=not progress,
    )
    span_set = set(spans)
    for scene_index, batch in enumerate(scene_iter):
        scene_id = str(batch.scene_id)
        completed = base._load_scene_checkpoint(out_root, scene_id) if resume else {}
        missing = [span for span in spans if span not in completed]
        if not missing:
            base._latency_log(
                f"[remaining_skip_eval][resume] scene={scene_id} reused all {len(spans)} spans"
            )
            continue
        missing_set = set(missing)
        requested_physical = {
            physical: logical
            for physical, logical in physical_to_logical.items()
            if logical in missing_set
        }
        prepared = None
        span_iter = None
        scene_t0 = time.perf_counter()
        try:
            prepared = dataset_plugin.prepare_skip_eval_scene(
                adapter, batch, eval_context
            )
            if not isinstance(prepared.images, torch.Tensor):
                raise TypeError(
                    f"Dataset {dataset_plugin.name!r} returned non-tensor images"
                )
            yielded: Set[Tuple[int, int]] = set()
            prediction_iter = _iter_logical_predictions(
                adapter, model, prepared.images, requested_physical
            )
            span_iter = tqdm(
                prediction_iter,
                total=len(missing),
                desc=f"  remaining spans {scene_id}",
                unit="span",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                disable=not progress,
            )
            for logical_i, logical_j, prediction in span_iter:
                logical_span = (int(logical_i), int(logical_j))
                if logical_span not in span_set or logical_span not in missing_set:
                    raise RuntimeError(
                        f"Adaptor yielded unrequested logical span {logical_span}"
                    )
                if logical_span in yielded:
                    raise RuntimeError(f"Adaptor yielded duplicate span {logical_span}")
                yielded.add(logical_span)
                try:
                    metric_row = base.normalize_metrics(
                        dataset_plugin.score_skip_eval_prediction(
                            adapter, prediction, prepared, eval_context
                        )
                    )
                    base._append_scene_checkpoint(
                        out_root,
                        scene_id=scene_id,
                        span=logical_span,
                        metrics=metric_row,
                    )
                finally:
                    del prediction
            span_iter.close()
            if yielded != missing_set:
                raise RuntimeError(
                    "Logical span coverage mismatch: "
                    f"missing={sorted(missing_set - yielded)[:8]} "
                    f"extra={sorted(yielded - missing_set)[:8]}"
                )
            base._latency_log(
                f"[remaining_skip_eval] scene {scene_index + 1}/{total_scenes or '?'} "
                f"{scene_id} spans={len(missing)} elapsed={time.perf_counter() - scene_t0:.2f}s"
            )
            if timing_on:
                base._timing_log(
                    f"[timing] remaining scene={scene_id} "
                    f"total={time.perf_counter() - scene_t0:.3f}s"
                )
        except Exception as exc:
            base._latency_log(
                f"[warn] scene={scene_id} remaining incremental skip eval failed: {exc}"
            )
        finally:
            if span_iter is not None:
                span_iter.close()
            if prepared is not None:
                del prepared
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    report_path = base._finalize_skip_outputs_from_disk(
        out_root=out_root,
        spans=all_spans,
        n_units=n_units,
        unit_ids=remaining_unit_ids,
        metrics=metrics,
        upper_metrics=upper_metrics,
        upper_scene_metrics=upper_scene_metrics,
        backbone=adapter.name,
        dataset=tagged_dataset,
        max_span=int(max_span),
        fusion_mode=str(mode),
        elapsed_seconds=time.perf_counter() - run_t0,
        ensure_scene_reports=True,
        supplemental_span_metrics_by_scene=reused_span_metrics,
        materialize_intervals=False,
    )
    for filename in (
        "skip_eval_report_v3.json",
        "skip_eval_scenes_v3.json",
        "skip_eval_summary_v3.json",
        "skip_eval_interval_index_v3.json",
    ):
        _append_provenance(out_root / filename, provenance)
    interval_metadata = materialize_evaluated_intervals(
        staging_root=out_root,
        result_root=pruning_root,
        public_pruning_root=pruning_root,
        records=span_records,
        provenance=provenance,
    )
    update_remaining_index(
        pruning_root=pruning_root,
        staging_root=out_root,
        records=span_records,
        provenance=provenance,
        interval_metadata=interval_metadata,
    )
    return report_path


def main() -> None:
    args = _build_parser().parse_args()
    run_remaining_skip_gt_eval(
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
        output_dataset=str(args.output_dataset),
        device=str(args.device),
        max_scenes=int(args.max_scenes),
        max_frames=int(args.max_frames),
        max_span=int(args.max_span),
        mode=str(args.mode),
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        split=args.split,
        progress=not bool(args.no_progress),
        timing=bool(args.timing) if args.timing else None,
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()

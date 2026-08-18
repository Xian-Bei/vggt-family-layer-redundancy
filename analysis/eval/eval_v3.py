"""Stage 4 (v3): unified evaluation across datasets and backbones."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import EvalSceneContext
from analysis.backbones.registry import get_backbone
from analysis.datasets.batch_frames import count_eval_frames
from analysis.datasets.registry import dataset_names, get_dataset
from analysis.amp_policy import autocast_device_dtype
from analysis.eval.async_pipeline import (
    DEFAULT_ASYNC_SCORE_MAX_PENDING,
    DEFAULT_ASYNC_SCORE_WORKERS,
    AsyncPredictionScorePipeline,
    AsyncScenePreloader,
    async_score_enabled,
    scene_preload_enabled,
    stage_preloaded_scene_to_cuda,
)
from analysis.eval.metrics_core import average_metrics
from analysis.eval.interval_results import materialize_eval_interval_result
from analysis.inference_compression.contracts import InferenceCompressionConfig
from analysis.inference_compression.registry import get_inference_compression
from analysis.prune_plan import (
    UnitPrunePlan,
    default_prune_plan_root,
    load_named_prune_plan,
    resolve_prune_plan_path,
)
from analysis.prune_plan.recovery_plan import compose_recovery_unit_prune_plan
from analysis.schema.v3 import EvalRecordV3, stage_dir
from analysis.recovery.train.checkpoint_v3 import EvalMode, apply_eval_model_structure
from analysis.recovery.shared_results import eval_result_complete

_EVAL_MODES = ("upper", "lower", "model")
_DERIVED_GHOST_RECOVERY_NAMES = {
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
}


@dataclass
class _EvalScoreKey:
    scene_id: str
    num_frames: int
    prepared: Any
    inference_start: Optional[torch.cuda.Event]
    inference_end: Optional[torch.cuda.Event]
    infer_peak_vram_gb: float


def _replace_with_relative_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    target.symlink_to(
        os.path.relpath(source, start=target.parent),
        target_is_directory=source.is_dir(),
    )


def _try_link_derived_ghost_eval(
    args: argparse.Namespace, jobs: List[Dict[str, Any]]
) -> bool:
    """Link complete Ghost evals and leave only missing dataset jobs to run."""
    recovery_name = str(args.recovery_plan_name or "")
    if recovery_name not in _DERIVED_GHOST_RECOVERY_NAMES or not jobs:
        return False
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None or len(checkpoint.parents) < 2:
        return False
    # Recovery fitting creates this directory symlink only after confirming
    # that the physical prune plan is one contiguous interval.
    if not checkpoint.parent.parent.is_symlink():
        return False

    links: List[tuple[Path, Path, str]] = []
    remaining_jobs: List[Dict[str, Any]] = []
    for job in jobs:
        output_value = job.get("output_dir")
        output_dataset = str(job.get("output_dataset") or job.get("dataset") or "")
        if not output_value:
            return False
        target = Path(str(output_value))
        if target.name != recovery_name:
            return False
        source = target.parent / "recovery_ghost"
        if not eval_result_complete(
            source,
            selector=target.parent.name,
            recovery_plan_name="recovery_ghost",
        ):
            remaining_jobs.append(job)
            continue
        links.append((target, source, output_dataset))

    for target, source, output_dataset in links:
        _replace_with_relative_symlink(target, source)
        print(
            "\n  eval reuse\n"
            f"    recovery: {recovery_name.removeprefix('recovery_')}\n"
            f"    dataset: {output_dataset}\n"
            "    source: ghost\n"
            f"    link: {target} -> {source}\n",
            file=sys.stderr,
            flush=True,
        )
    jobs[:] = remaining_jobs
    return not jobs


def _result_selector_from_prune_plan(prune_plan: UnitPrunePlan, fallback: str) -> str:
    extra = prune_plan.extra if isinstance(prune_plan.extra, dict) else {}
    for key in ("eval_selector", "interval_selector", "selector", "plan_group"):
        value = str(extra.get(key, "")).strip()
        if value:
            return value
    return str(fallback or "").strip()


def _eval_scene_count(
    ds_plugin,
    data_root: Path,
    *,
    max_scenes: int,
    split: str | None,
) -> int | None:
    """Scene count for tqdm total without loading image batches."""
    return ds_plugin.count_scenes(data_root, max_scenes=int(max_scenes), split=split)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified eval (v3).")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, default="", choices=("", *dataset_names()))
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split, validated by the dataset adaptor.",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument(
        "--jobs-manifest",
        type=Path,
        default=None,
        help="JSON list of eval jobs sharing one model/checkpoint load.",
    )
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Output tag (default: same as --eval-mode: upper | lower | model).",
    )
    p.add_argument(
        "--eval-mode",
        type=str,
        choices=_EVAL_MODES,
        default="upper",
        help="upper=full pretrained; lower=prune/skip only; model=prune+LoRA+checkpoint.",
    )
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Max frames per scene (applied in dataset plugin; same as cache/redundancy stages).",
    )
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="Per-backbone prune plans for lower/model modes (default: analysis/prune_plan).",
    )
    p.add_argument(
        "--prune-plan-name",
        dest="prune_plan_name",
        type=str,
        default="",
        help="Prune selector for lower/model modes (e.g. i9_j19).",
    )
    p.add_argument(
        "--recovery-plan-root",
        type=Path,
        default=None,
        help="Recovery plan root for model mode. Defaults to prune-plan-root.",
    )
    p.add_argument(
        "--recovery-plan-name",
        type=str,
        default="",
        help="Recovery plan name for model mode (e.g. recovery_lora_l1_r32).",
    )
    p.add_argument(
        "--export-vis-2d",
        action="store_true",
        help="Save per-scene plot.png trajectory comparison (all datasets).",
    )
    p.add_argument(
        "--export-vis-3d",
        action="store_true",
        help="Save vis/pred (and 7andn vis/gt) point clouds for analysis.view.",
    )
    p.add_argument(
        "--vis-max-points",
        type=int,
        default=0,
        help="Max points in 3D viser NPZ (0 = no cap).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar; restore legacy per-line logging.",
    )
    p.add_argument(
        "--no-async-score",
        action="store_true",
        help="Disable scene-level CUDA inference / CPU scoring overlap.",
    )
    p.add_argument(
        "--score-workers",
        type=int,
        default=DEFAULT_ASYNC_SCORE_WORKERS,
        help="Number of parallel CPU scoring workers.",
    )
    p.add_argument(
        "--score-max-pending",
        type=int,
        default=DEFAULT_ASYNC_SCORE_MAX_PENDING,
        help="Maximum number of staged predictions awaiting CPU scoring.",
    )
    p.add_argument(
        "--timing",
        action="store_true",
        help="Log aggregate asynchronous evaluation pipeline timings.",
    )
    p.add_argument(
        "--no-scene-preload",
        action="store_true",
        help="Disable one-scene CPU preparation lookahead and asynchronous H2D.",
    )
    p.add_argument(
        "--inference-compression",
        choices=("none", "lora_merge", "fastvggt", "quantvggt"),
        default="none",
        help="Deployment-only compression applied after prune/checkpoint restoration.",
    )
    p.add_argument("--fastvggt-merge-start", type=int, default=0)
    p.add_argument("--fastvggt-merge-ratio", type=float, default=0.9)
    p.add_argument(
        "--quantvggt-dtype",
        choices=("w4a4", "w6a6", "w8a8"),
        default="w4a4",
    )
    p.add_argument("--quantvggt-params-dir", type=Path, default=None)
    return p


def evaluate_model_v3(
    *,
    adapter,
    model,
    ds_plugin,
    experiment: str,
    dataset: str,
    data_root: Path,
    backbone: str,
    model_name: str,
    device: torch.device,
    output_dir: Path | None,
    max_scenes: int,
    max_frames: int,
    export_vis_2d: bool = False,
    export_vis_3d: bool = False,
    vis_max_points: int = 0,
    split: str | None = None,
    output_dataset: str | None = None,
    progress: bool = True,
    async_score: bool = True,
    score_workers: int = DEFAULT_ASYNC_SCORE_WORKERS,
    score_max_pending: int = DEFAULT_ASYNC_SCORE_MAX_PENDING,
    timing: bool = False,
    scene_preload: bool = True,
) -> Path:
    """Shared evaluation entry used by stage-4 eval and training periodic eval."""
    split = ds_plugin.validate_split(split)
    profile = ds_plugin.eval_profile()
    tagged_dataset = output_dataset or ds_plugin.output_tag(split)
    out_dir = output_dir or stage_dir(experiment, tagged_dataset, f"eval/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = EvalSceneContext(
        device=device,
        profile=profile,
        export_vis_2d=bool(export_vis_2d),
        export_vis_3d=bool(export_vis_3d),
        vis_save_root=out_dir,
        vis_max_points=int(vis_max_points),
        backbone_name=str(backbone),
        dataset_name=str(ds_plugin.name),
    )

    rows: List[Dict[str, float]] = []
    max_scenes_i = int(max_scenes)
    total_scenes = _eval_scene_count(
        ds_plugin,
        data_root,
        max_scenes=max_scenes_i,
        split=split,
    )

    compact_log = os.environ.get("COMPACT_LOG", "").lower() in {"1", "true", "yes"}
    desc = "    scenes" if compact_log else f"eval {model_name}/{backbone} ({tagged_dataset})"
    scene_source = ds_plugin.iter_scenes(
        data_root,
        max_scenes=max_scenes_i,
        max_frames=int(max_frames),
        split=split,
    )
    pbar = (
        tqdm(
            total=total_scenes,
            desc=desc,
            unit="scene",
            file=sys.stderr,
            dynamic_ncols=True,
        )
        if progress
        else None
    )
    def report_error(sid: str, exc: Exception) -> None:
        msg = f"[warn] {sid} failed: {exc}"
        if pbar is not None:
            tqdm.write(msg, file=sys.stderr)
            pbar.update(1)
        else:
            print(msg, file=sys.stderr)

    def write_scene_result(key: _EvalScoreKey, scene_metrics: Dict[str, float]) -> None:
        scene_metrics["num_frames"] = float(key.num_frames)
        scene_metrics["scene_id"] = key.scene_id
        rows.append(scene_metrics)
        scene_dir = out_dir / key.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "metrics.json").write_text(
            json.dumps(scene_metrics, indent=2), encoding="utf-8"
        )
        if pbar is not None:
            pbar.update(1)

    async_active = (
        async_score_enabled(async_score, device)
        and not bool(export_vis_2d)
        and not bool(export_vis_3d)
    )
    pipeline = None
    pipeline_t0 = time.perf_counter()
    if async_active:
        def score_scene(
            key: _EvalScoreKey, prediction
        ) -> Dict[str, float]:
            metrics = ds_plugin.score_eval_prediction(
                adapter, prediction, key.prepared, ctx
            )
            if key.inference_start is None or key.inference_end is None:
                raise RuntimeError("Async eval task is missing CUDA inference events")
            metrics["inference_time_ms"] = float(
                key.inference_start.elapsed_time(key.inference_end)
            )
            metrics["infer_peak_vram_gb"] = float(key.infer_peak_vram_gb)
            return metrics

        pipeline = AsyncPredictionScorePipeline(
            device=device,
            score_fn=score_scene,
            result_fn=write_scene_result,
            error_fn=lambda key, exc: report_error(key.scene_id, exc),
            timing=bool(timing),
            score_workers=score_workers,
            max_pending=score_max_pending,
            thread_name_prefix="eval-score",
        )

    preload_active = async_active and scene_preload_enabled(
        scene_preload,
        device,
        export_vis_2d=export_vis_2d,
        export_vis_3d=export_vis_3d,
    )
    preloader = None
    h2d_stream = None
    h2d_ms = 0.0
    peak_vram_gib = 0.0
    gpu_inference_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    if preload_active:
        preloader = AsyncScenePreloader(
            scene_source=iter(scene_source),
            prepare_fn=lambda batch: ds_plugin.prepare_eval_scene_cpu(
                adapter, batch, ctx
            ),
        )
        h2d_stream = torch.cuda.Stream(device=device)

    scene_iterator = iter(scene_source) if preloader is None else None
    try:
        while True:
            preload_item = None
            if preloader is not None:
                preload_item = preloader.next()
                if preload_item is None:
                    break
                batch = preload_item.batch
                if preload_item.error is not None:
                    report_error(str(batch.scene_id), preload_item.error)
                    continue
            else:
                assert scene_iterator is not None
                try:
                    batch = next(scene_iterator)
                except StopIteration:
                    break
            sid = str(batch.scene_id)
            if pbar is not None:
                pbar.set_postfix_str(sid, refresh=False)
            if pipeline is None:
                try:
                    with torch.no_grad():
                        scene_metrics = ds_plugin.evaluate_scene(
                            adapter,
                            model,
                            batch=batch,
                            ctx=ctx,
                        )
                except Exception as e:
                    report_error(sid, e)
                    continue
                key = _EvalScoreKey(
                    scene_id=sid,
                    num_frames=count_eval_frames(batch),
                    prepared=None,
                    inference_start=None,
                    inference_end=None,
                    infer_peak_vram_gb=float(scene_metrics.get("infer_peak_vram_gb", 0.0)),
                )
                write_scene_result(key, scene_metrics)
                continue
            h2d_ready = None
            h2d_start = None
            host_images = None
            try:
                if preload_item is not None:
                    prepared = preload_item.prepared
                    assert h2d_stream is not None
                    h2d_ready, h2d_start, host_images = stage_preloaded_scene_to_cuda(
                        prepared,
                        device=device,
                        copy_stream=h2d_stream,
                        timing=bool(timing),
                    )
                else:
                    prepared = ds_plugin.prepare_skip_eval_scene(adapter, batch, ctx)
                torch.cuda.reset_peak_memory_stats(device.index)
                inference_start = torch.cuda.Event(enable_timing=True)
                inference_end = torch.cuda.Event(enable_timing=True)
                inference_start.record(torch.cuda.current_stream(device))
                ac_kw = autocast_device_dtype(device)
                with torch.no_grad(), torch.autocast(**ac_kw):
                    prediction = adapter.forward_predictions(
                        model,
                        prepared.images,
                        defer_cpu_geometry=True,
                    )
                inference_end.record(torch.cuda.current_stream(device))
                gpu_inference_events.append((inference_start, inference_end))
                scene_peak_vram_gb = float(torch.cuda.max_memory_allocated(device.index)) / (
                    1024.0**3
                )
                peak_vram_gib = max(peak_vram_gib, scene_peak_vram_gb)
                key = _EvalScoreKey(
                    scene_id=sid,
                    num_frames=count_eval_frames(batch),
                    prepared=prepared,
                    inference_start=inference_start,
                    inference_end=inference_end,
                    infer_peak_vram_gb=scene_peak_vram_gb,
                )
                pipeline.submit(key, prediction)
                prepared.images = None
                del prediction
            except Exception as e:
                report_error(sid, e)
            finally:
                if h2d_ready is not None:
                    h2d_ready.synchronize()
                    if bool(timing) and h2d_start is not None:
                        h2d_ms += float(h2d_start.elapsed_time(h2d_ready))
                del host_images
    finally:
        try:
            if preloader is not None:
                preloader.close()
        finally:
            if pipeline is not None:
                pipeline.close()

    if bool(timing) and pipeline is not None:
        gpu_inference_s = (
            sum(float(start.elapsed_time(end)) for start, end in gpu_inference_events)
            / 1000.0
        )
        print(
            f"[timing] eval dataset={tagged_dataset} pipeline={time.perf_counter() - pipeline_t0:.3f}s "
            f"iterator_load={preloader.iterator_s if preloader is not None else 0.0:.3f}s "
            f"cpu_prepare={preloader.cpu_prepare_s if preloader is not None else 0.0:.3f}s "
            f"preload_wait={preloader.preload_wait_s if preloader is not None else 0.0:.3f}s "
            f"h2d={h2d_ms / 1000.0:.3f}s "
            f"gpu_inference={gpu_inference_s:.3f}s "
            f"d2h={pipeline.d2h_ms / 1000.0:.3f}s "
            f"cpu_score={pipeline.score_s:.3f}s "
            f"producer_wait={pipeline.producer_wait_s:.3f}s "
            f"score_workers={pipeline.score_workers} "
            f"max_pending={pipeline.max_pending_seen} "
            f"max_lookahead={preloader.max_lookahead_seen if preloader is not None else 0} "
            f"peak_vram={peak_vram_gib:.3f}GiB "
            f"peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2):.3f}GiB",
            file=sys.stderr,
            flush=True,
        )
    if pbar is not None:
        pbar.close()

    avg = average_metrics(rows)
    frame_counts = [float(r["num_frames"]) for r in rows if "num_frames" in r]
    if frame_counts:
        avg["num_frames"] = float(np.mean(frame_counts))
    record = EvalRecordV3(
        backbone=backbone,
        dataset=tagged_dataset,
        model_name=model_name,
        dataset_eval_profile={
            "geometry": asdict(profile.geometry),
            "kf_every": int(profile.kf_every),
            "max_frames": int(max_frames),
        },
        scenes={str(i): r for i, r in enumerate(rows)},
        average=avg,
        extra={
            "eval_mode": str(getattr(model, "_lare_eval_mode", "")),
            "auto_merged_lora_linears": int(
                getattr(model, "_lare_auto_merged_lora", 0)
            ),
            "inference_compression": dict(
                getattr(model, "_inference_compression", {"name": "none"})
            )
        },
    )
    record.save(out_dir / "eval_record_v3.json")
    if not compact_log:
        done_msg = f"Wrote {out_dir / 'eval_record_v3.json'}"
        if progress:
            tqdm.write(done_msg, file=sys.stderr)
        else:
            print(done_msg, file=sys.stderr)
    return out_dir


def main() -> None:
    args = _build_parser().parse_args()
    jobs: List[Dict[str, Any]] = []
    if args.jobs_manifest is not None:
        payload = json.loads(Path(args.jobs_manifest).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
            raise ValueError("--jobs-manifest must contain a non-empty JSON object list")
        jobs = [dict(x) for x in payload]
        first = jobs[0]
        common_fields = (
            "experiment", "backbone", "hf_model", "device", "eval_mode",
            "model_name", "checkpoint", "prune_plan_root", "prune_plan_name",
            "recovery_plan_root", "recovery_plan_name", "inference_compression",
            "fastvggt_merge_start", "fastvggt_merge_ratio", "quantvggt_dtype",
            "quantvggt_params_dir",
        )
        for field in common_fields:
            if field in first:
                value = first[field]
                if field in {"checkpoint", "prune_plan_root", "recovery_plan_root", "quantvggt_params_dir"}:
                    value = Path(value) if value else None
                setattr(args, field, value)
        for index, job in enumerate(jobs[1:], start=1):
            mismatched = [
                field for field in common_fields
                if field in job and job.get(field) != first.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"Eval job {index} changes shared model fields: {mismatched}"
                )
    elif not args.dataset or args.data_root is None:
        raise ValueError("--dataset and --data-root are required without --jobs-manifest")
    if _try_link_derived_ghost_eval(args, jobs):
        return
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    device = torch.device(args.device)
    hf_model = str(args.hf_model or adapter.default_hf_model())
    eval_mode: EvalMode = args.eval_mode  # type: ignore[assignment]
    base_model_name = str(args.model_name).strip() or str(eval_mode)
    compression_config = InferenceCompressionConfig(
        name=str(args.inference_compression),
        fastvggt_merge_start=int(args.fastvggt_merge_start),
        fastvggt_merge_ratio=float(args.fastvggt_merge_ratio),
        quantvggt_dtype=str(args.quantvggt_dtype),
        quantvggt_params_dir=args.quantvggt_params_dir,
    )
    compression_adapter = None
    model_name = base_model_name
    if compression_config.name != "none":
        compression_adapter = get_inference_compression(compression_config.name)
        compression_adapter.validate_backbone(adapter.name)
        compression_adapter.validate_config(compression_config)
        model_name = f"{base_model_name}_{compression_adapter.output_tag(compression_config)}"

    model = adapter.load_pretrained(hf_model=hf_model, device=str(device))

    prune_plan: Optional[UnitPrunePlan] = None
    if eval_mode in ("lower", "model"):
        plan_root = args.prune_plan_root or default_prune_plan_root()
        prune_plan_name = str(args.prune_plan_name or "").strip()
        if prune_plan_name:
            recovery_plan_name = str(args.recovery_plan_name or "").strip()
            if eval_mode == "model" and recovery_plan_name:
                prune_plan = compose_recovery_unit_prune_plan(
                    prune_plan_root=plan_root,
                    recovery_plan_root=args.recovery_plan_root or plan_root,
                    prune_selector=prune_plan_name,
                    recovery_plan=recovery_plan_name,
                )
            else:
                prune_plan = load_named_prune_plan(plan_root, prune_plan_name)
        else:
            prune_plan = UnitPrunePlan.load(resolve_prune_plan_path(plan_root, adapter.name))

    apply_eval_model_structure(
        model,
        adapter,
        eval_mode=eval_mode,
        prune_plan=prune_plan,
        checkpoint=args.checkpoint,
        map_location=device,
    )

    # This is an inference-only model. Fold LoRA branches by default so the
    # measured latency and VRAM match the deployable transformer topology.
    auto_merged_lora = 0
    if eval_mode == "model":
        from analysis.inference_compression.lora import merge_lora_linears

        auto_merged_lora = merge_lora_linears(model)
    model._lare_auto_merged_lora = int(auto_merged_lora)

    if compression_adapter is not None:
        model = compression_adapter.apply(
            model,
            backbone_name=adapter.name,
            config=compression_config,
        )
    else:
        model._inference_compression = {"name": "none"}
    model._lare_eval_mode = str(eval_mode)

    model.eval()
    if args.no_progress:
        print(
            f"[eval_v3] mode={eval_mode} model_name={model_name} backbone={adapter.name} "
            f"checkpoint={args.checkpoint} inference_compression={compression_config.name}",
            flush=True,
        )
    eval_jobs = jobs or [
        {
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir) if args.output_dir is not None else None,
            "split": args.split,
        }
    ]
    total_eval_jobs = len(eval_jobs)
    for job_index, job in enumerate(eval_jobs, start=1):
        ds_plugin = get_dataset(str(job["dataset"]))
        split = ds_plugin.validate_split(job.get("split", args.split))
        output_dataset = str(
            job.get("output_dataset") or ds_plugin.output_tag(split)
        )
        recovery_name = str(
            job.get("recovery_plan_name") or args.recovery_plan_name or ""
        ).removeprefix("recovery_")
        prune_name = str(
            job.get("prune_plan_name") or args.prune_plan_name or ""
        )
        print(
            "\n  eval\n"
            f"    job: {job_index}/{total_eval_jobs}\n"
            f"    model: {adapter.name}\n"
            f"    span: {prune_name.removeprefix('prune_span') or '-'}\n"
            f"    recovery: {recovery_name or '-'}\n"
            f"    dataset: {output_dataset}\n",
            file=sys.stderr,
            flush=True,
        )
        out_dir = evaluate_model_v3(
            adapter=adapter,
            model=model,
            ds_plugin=ds_plugin,
            experiment=str(args.experiment),
            dataset=str(job["dataset"]),
            data_root=Path(str(job["data_root"])),
            backbone=str(args.backbone),
            model_name=model_name,
            device=device,
            output_dir=Path(str(job["output_dir"])) if job.get("output_dir") else None,
            max_scenes=int(job.get("max_scenes", args.max_scenes)),
            max_frames=int(job.get("max_frames", args.max_frames)),
            export_vis_2d=bool(job.get("export_vis_2d", args.export_vis_2d)),
            export_vis_3d=bool(job.get("export_vis_3d", args.export_vis_3d)),
            vis_max_points=int(job.get("vis_max_points", args.vis_max_points)),
            split=split,
            output_dataset=job.get("output_dataset"),
            progress=not bool(job.get("no_progress", args.no_progress)),
            async_score=not bool(job.get("no_async_score", args.no_async_score)),
            score_workers=int(job.get("score_workers", args.score_workers)),
            score_max_pending=int(
                job.get("score_max_pending", args.score_max_pending)
            ),
            timing=bool(job.get("timing", args.timing)),
            scene_preload=not bool(
                job.get("no_scene_preload", args.no_scene_preload)
            ),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if eval_mode in ("lower", "model") and prune_plan is not None:
            selector = _result_selector_from_prune_plan(prune_plan, args.prune_plan_name)
            summary_path = materialize_eval_interval_result(
                out_dir,
                selector=selector,
                drop=list(prune_plan.drop),
                source_eval_mode=str(eval_mode),
                recovery_plan_name=str(args.recovery_plan_name or ""),
            )
            if os.environ.get("COMPACT_LOG", "").lower() not in {"1", "true", "yes"}:
                print(f"Wrote {summary_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

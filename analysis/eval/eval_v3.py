"""Stage 4 (v3): unified evaluation across datasets and backbones."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from tqdm import tqdm

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import EvalSceneContext
from analysis.backbones.registry import get_backbone
from analysis.datasets.batch_frames import count_eval_frames
from analysis.datasets.registry import dataset_names, get_dataset
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

_EVAL_MODES = ("upper", "lower", "model")


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
    p.add_argument("--dataset", type=str, required=True, choices=dataset_names())
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split, validated by the dataset adaptor.",
    )
    p.add_argument("--data-root", type=Path, required=True)
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
    p.add_argument("--skip-depth", action="store_true")
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
    skip_depth: bool,
    max_scenes: int,
    max_frames: int,
    export_vis_2d: bool = False,
    export_vis_3d: bool = False,
    vis_max_points: int = 0,
    split: str | None = None,
    output_dataset: str | None = None,
    progress: bool = True,
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
        skip_depth=bool(skip_depth),
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
    for batch in scene_source:
        sid = batch.scene_id
        if pbar is not None:
            pbar.set_postfix_str(sid, refresh=False)
        try:
            with torch.no_grad():
                scene_metrics = ds_plugin.evaluate_scene(
                    adapter,
                    model,
                    batch=batch,
                    ctx=ctx,
                )
        except Exception as e:
            msg = f"[warn] {sid} failed: {e}"
            if pbar is not None:
                tqdm.write(msg, file=sys.stderr)
            else:
                print(msg, file=sys.stderr)
            if pbar is not None:
                pbar.update(1)
            continue
        scene_metrics["num_frames"] = float(count_eval_frames(batch))
        scene_metrics["scene_id"] = sid
        rows.append(scene_metrics)
        (out_dir / sid).mkdir(parents=True, exist_ok=True)
        (out_dir / sid / "metrics.json").write_text(json.dumps(scene_metrics, indent=2), encoding="utf-8")
        if pbar is not None:
            pbar.update(1)
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
            "depth": asdict(profile.depth),
            "kf_every": int(profile.kf_every),
            "max_frames": int(max_frames),
        },
        scenes={str(i): r for i, r in enumerate(rows)},
        average=avg,
        extra={
            "eval_mode": str(getattr(model, "_lare_eval_mode", "")),
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
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    ds_plugin = get_dataset(args.dataset)
    split = ds_plugin.validate_split(args.split)
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
    out_dir = evaluate_model_v3(
        adapter=adapter,
        model=model,
        ds_plugin=ds_plugin,
        experiment=str(args.experiment),
        dataset=str(args.dataset),
        data_root=Path(args.data_root),
        backbone=str(args.backbone),
        model_name=model_name,
        device=device,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        skip_depth=bool(args.skip_depth),
        max_scenes=int(args.max_scenes),
        max_frames=int(args.max_frames),
        export_vis_2d=bool(args.export_vis_2d),
        export_vis_3d=bool(args.export_vis_3d),
        vis_max_points=int(args.vis_max_points),
        split=split,
        progress=not bool(args.no_progress),
    )
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

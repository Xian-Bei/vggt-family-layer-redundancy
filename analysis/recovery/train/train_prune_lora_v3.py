"""Stage 3 (v3): unified prune + LoRA training via backbone adapters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import TrainFromCacheArgs
from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    default_prune_plan_root,
    prune_plan_bundle_path,
    resolve_named_prune_plan_path,
)
from analysis.prune_plan.recovery_plan import compose_recovery_unit_prune_plan
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recovery prune+LoRA (v3) via adapter.train_from_cache.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Training cache dataset tag, or a comma/space separated list of tags.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma/space separated training cache dataset tags. Overrides --dataset.",
    )
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="Directory with prune plans (default: analysis/prune_plan).",
    )
    p.add_argument(
        "--prune-plan-name",
        dest="prune_plan_name",
        type=str,
        default="",
        help="Required prune selector (e.g. prune_span10).",
    )
    p.add_argument(
        "--recovery-plan-root",
        type=Path,
        default=None,
        help="Directory with recovery plan bundles (default: prune-plan-root).",
    )
    p.add_argument(
        "--recovery-plan-name",
        type=str,
        default="",
        help="Required recovery plan name (e.g. recovery_lora_l1_r32).",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-align-tokens", type=int, default=4096)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eval-epoch", type=int, default=0, help="0 disables periodic eval during training.")
    p.add_argument("--eval-max-scenes", type=int, default=0, help="0 means full periodic eval split.")
    p.add_argument(
        "--max-eval-views",
        type=int,
        default=50,
        help="Max frames per scene for periodic eval.",
    )
    p.add_argument(
        "--save-every-epoch",
        type=int,
        default=10,
        help="Save epoch_N.pt every N epochs (0 disables numbered checkpoints; latest.pt still saved).",
    )
    p.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run periodic eval once before training (epoch=0), independent of --eval-epoch.",
    )
    p.add_argument(
        "--eval-bounds-root",
        type=Path,
        default=Path("analysis_outputs_v3"),
        help="Root containing precomputed upper/lower eval_record_v3.json bounds.",
    )
    p.add_argument(
        "--eval-scannet-dev50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on scannet_dev50.",
    )
    p.add_argument(
        "--eval-scannet-test50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on scannet_test50.",
    )
    p.add_argument(
        "--eval-7scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on 7scenes.",
    )
    p.add_argument(
        "--eval-7scenes-split",
        type=str,
        default="test",
        choices=("train", "test"),
        help="7scenes split for periodic eval (output tag: 7scenes_<split>).",
    )
    p.add_argument(
        "--eval-nrgbd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on nrgbd.",
    )
    p.add_argument(
        "--eval-nuscenes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic eval on nuscenes.",
    )
    p.add_argument(
        "--eval-nuscenes-split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="nuScenes split for periodic eval (output tag: nuscenes_<split>).",
    )
    p.add_argument(
        "--eval-waymo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic eval on waymo.",
    )
    p.add_argument(
        "--eval-waymo-split",
        type=str,
        default="validation",
        choices=("training", "validation", "train", "val"),
        help="Waymo split for periodic eval (output tag: waymo_<split>).",
    )
    p.add_argument("--root-7scenes", type=Path, default=Path("data/7scenes"))
    p.add_argument("--root-nrgbd", type=Path, default=Path("data/neural_rgbd_data"))
    p.add_argument("--root-nuscenes", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--root-waymo", type=Path, default=Path("data/waymo"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16", "float16"))
    return p


def _parse_name_list(raw: str) -> list[str]:
    names: list[str] = []
    for name in str(raw or "").replace(",", " ").split():
        name = name.strip()
        if name:
            names.append(name)
    return names


def main() -> None:
    args = _build_parser().parse_args()
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)

    dataset_names = _parse_name_list(args.datasets or args.dataset)
    if not dataset_names:
        raise SystemExit("--dataset/--datasets is required")
    if any(name == "scannet" for name in dataset_names):
        raise SystemExit("dataset tag 'scannet' is legacy; use scannet_dev50 or another cache output tag")
    cache_dirs = [stage_dir(args.experiment, name, "cache") for name in dataset_names]
    if not str(args.prune_plan_name).strip():
        raise SystemExit("--prune-plan-name is required")
    plan_root = args.prune_plan_root or (Path("analysis_outputs_v3") / adapter.name)
    prune_plan_name = str(args.prune_plan_name).strip()
    recovery_root = args.recovery_plan_root or plan_root
    recovery_plan_name = str(args.recovery_plan_name or "").strip()
    if not recovery_plan_name:
        raise SystemExit("--recovery-plan-name is required")
    compose_recovery_unit_prune_plan(
        prune_plan_root=plan_root,
        recovery_plan_root=recovery_root,
        prune_selector=prune_plan_name,
        recovery_plan=recovery_plan_name,
    )
    prune_path = (
        prune_plan_bundle_path(plan_root)
        if prune_plan_name.startswith("prune_span")
        else resolve_named_prune_plan_path(plan_root, prune_plan_name)
    )
    default_output_root = Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT_DEFAULT)))
    output_kind = str(os.environ.get("OUTPUT_KIND", "train") or "train").strip() or "train"
    out_dir = args.output_dir or (
        default_output_root / str(args.experiment) / output_kind / prune_plan_name / recovery_plan_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model = str(args.hf_model or adapter.default_hf_model())

    train_args = TrainFromCacheArgs(
        experiment=str(args.experiment),
        dataset=",".join(dataset_names),
        cache_dir=cache_dirs[0],
        prune_plan_path=prune_path,
        output_dir=out_dir,
        hf_model=hf_model,
        device=str(args.device),
        dtype=str(args.dtype),
        epochs=int(args.epochs),
        max_scenes=int(args.max_scenes),
        max_steps=int(args.max_steps),
        data_root=args.data_root,
        max_align_tokens=int(args.max_align_tokens),
        eval_epoch=int(args.eval_epoch),
        save_every_epoch=int(args.save_every_epoch),
        eval_max_scenes=int(args.eval_max_scenes),
        max_eval_views=int(args.max_eval_views),
        eval_at_start=bool(args.eval_at_start),
        eval_bounds_root=Path(args.eval_bounds_root),
        eval_scannet_dev50=bool(args.eval_scannet_dev50),
        eval_scannet_test50=bool(args.eval_scannet_test50),
        eval_7scenes=bool(args.eval_7scenes),
        eval_7scenes_split=str(args.eval_7scenes_split),
        eval_nrgbd=bool(args.eval_nrgbd),
        eval_nuscenes=bool(args.eval_nuscenes),
        eval_nuscenes_split=str(args.eval_nuscenes_split),
        eval_waymo=bool(args.eval_waymo),
        eval_waymo_split=str(args.eval_waymo_split),
        root_7scenes=Path(args.root_7scenes),
        root_nrgbd=Path(args.root_nrgbd),
        root_nuscenes=Path(args.root_nuscenes),
        root_waymo=Path(args.root_waymo),
        prune_plan_name=prune_plan_name,
        recovery_plan_root=Path(recovery_root),
        recovery_plan_name=recovery_plan_name,
        cache_dirs=tuple(cache_dirs),
    )
    print(
        f"[train_prune_lora_v3] backbone={adapter.name} datasets={','.join(dataset_names)} "
        f"prune_plan={prune_path} -> adapter.train_from_cache",
        flush=True,
    )
    adapter.train_from_cache(train_args)


if __name__ == "__main__":
    main()

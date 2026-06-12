"""Stage 3 (v3): unified prune + LoRA training via backbone adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import TrainFromCacheArgs
from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    default_prune_plan_root,
    resolve_named_prune_plan_path,
    resolve_prune_plan_path,
)
from analysis.schema.v3 import stage_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train prune+LoRA (v3) via adapter.train_from_cache.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="Directory with prune plans (legacy default: analysis/prune_plan).",
    )
    p.add_argument(
        "--plan-name",
        type=str,
        default="",
        help="Required plan name for multi-plan layout (e.g. plan_t2_l1).",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-align-tokens", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-5)
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
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16", "float16"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)

    cache_dir = stage_dir(args.experiment, args.dataset, "cache")
    if not str(args.plan_name).strip():
        raise SystemExit("--plan-name is required")
    plan_root = args.prune_plan_root or (Path("analysis_outputs_v3") / adapter.name)
    prune_path = resolve_named_prune_plan_path(plan_root, str(args.plan_name))
    if not prune_path.is_file():
        # Backward-compatible fallback if caller still passes legacy root/name-by-backbone.
        legacy = resolve_prune_plan_path(plan_root, adapter.name)
        if legacy.is_file() and str(args.plan_name).strip().lower() in ("legacy", adapter.name):
            prune_path = legacy
        else:
            raise SystemExit(f"Missing prune plan: {prune_path}")
    out_dir = args.output_dir or (stage_dir(args.experiment, args.dataset, str(args.plan_name)) / "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model = str(args.hf_model or adapter.default_hf_model())

    train_args = TrainFromCacheArgs(
        experiment=str(args.experiment),
        dataset=str(args.dataset),
        cache_dir=cache_dir,
        prune_plan_path=prune_path,
        output_dir=out_dir,
        hf_model=hf_model,
        device=str(args.device),
        dtype=str(args.dtype),
        epochs=int(args.epochs),
        max_scenes=int(args.max_scenes),
        max_steps=int(args.max_steps),
        lr=float(args.lr),
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
        plan_name=str(args.plan_name),
    )
    print(
        f"[train_prune_lora_v3] backbone={adapter.name} prune_plan={prune_path} -> adapter.train_from_cache",
        flush=True,
    )
    adapter.train_from_cache(train_args)


if __name__ == "__main__":
    main()

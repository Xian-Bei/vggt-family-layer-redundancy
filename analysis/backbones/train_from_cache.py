"""Launch shared cache-replay trainer for any registered backbone."""

from __future__ import annotations

import os
import subprocess
import sys

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.contracts import TrainFromCacheArgs


def launch_cache_replay_training(backbone: str, args: TrainFromCacheArgs) -> None:
    name = backbone.strip().lower()
    apply_pythonpath(name)
    os.environ["BACKEND"] = name
    train_dtype = str(args.dtype)
    cache_dirs = [str(p) for p in (getattr(args, "cache_dirs", None) or ()) if str(p)]
    if not cache_dirs:
        cache_dirs = [str(args.cache_dir)]
    train_argv = [
        sys.executable,
        "-m",
        "analysis.recover.train.train_lora_skip_baseline",
        "--backbone",
        name,
        "--hf-model",
        str(args.hf_model),
        "--output-dir",
        str(args.output_dir),
        "--prune-plan-file",
        str(args.prune_plan_path),
        "--prune-plan-name",
        str(args.prune_plan_name),
        "--device",
        str(args.device),
        "--dtype",
        train_dtype,
        "--epochs",
        str(int(args.epochs)),
        "--max-scenes",
        str(int(args.max_scenes)),
        "--max-steps",
        str(int(args.max_steps)),
        "--save-every-epoch",
        str(int(args.save_every_epoch)),
        "--eval-epoch",
        str(int(args.eval_epoch)),
        "--eval-max-scenes",
        str(int(args.eval_max_scenes)),
        "--max-eval-views",
        str(int(args.max_eval_views)),
    ]
    for cache_dir in cache_dirs:
        train_argv.extend(["--cache-dir", cache_dir])
    training_plan_root = str(getattr(args, "training_plan_root", "") or "")
    training_plan_name = str(getattr(args, "training_plan_name", "") or "")
    if training_plan_root:
        train_argv.extend(["--training-plan-root", training_plan_root])
    if training_plan_name:
        train_argv.extend(["--training-plan-name", training_plan_name])
    if bool(args.eval_at_start):
        train_argv.append("--eval-at-start")
    else:
        train_argv.append("--no-eval-at-start")
    train_argv.extend(
        [
        "--eval-bounds-root",
        str(args.eval_bounds_root),
        "--target-source",
        "teacher",
        ]
    )
    if bool(args.eval_scannet_dev50):
        train_argv.append("--eval-scannet-dev50")
    else:
        train_argv.append("--no-eval-scannet-dev50")
    if bool(args.eval_scannet_test50):
        train_argv.append("--eval-scannet-test50")
    else:
        train_argv.append("--no-eval-scannet-test50")
    if bool(args.eval_7scenes):
        train_argv.append("--eval-7scenes")
    else:
        train_argv.append("--no-eval-7scenes")
    train_argv.extend(["--eval-7scenes-split", str(args.eval_7scenes_split)])
    if bool(args.eval_nrgbd):
        train_argv.append("--eval-nrgbd")
    else:
        train_argv.append("--no-eval-nrgbd")
    if bool(args.eval_nuscenes):
        train_argv.append("--eval-nuscenes")
    else:
        train_argv.append("--no-eval-nuscenes")
    train_argv.extend(["--eval-nuscenes-split", str(args.eval_nuscenes_split)])
    if bool(args.eval_waymo):
        train_argv.append("--eval-waymo")
    else:
        train_argv.append("--no-eval-waymo")
    train_argv.extend(["--eval-waymo-split", str(args.eval_waymo_split)])
    train_argv.extend(["--root-7scenes", str(args.root_7scenes)])
    train_argv.extend(["--root-nrgbd", str(args.root_nrgbd)])
    train_argv.extend(["--root-nuscenes", str(args.root_nuscenes)])
    train_argv.extend(["--root-waymo", str(args.root_waymo)])
    print("[train] launching cache replay:", " ".join(train_argv[2:]), flush=True)
    subprocess.run(train_argv, check=True)

#!/usr/bin/env python3
"""One-off refresh of block-only parameter logs and existing summary markdown files."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_BACKBONES = ("vggt", "pi3", "vggt_omega")


def _format_param_count(n: int) -> str:
    value = float(n)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return str(int(value))


def _parameter_log_line(*, backbone: str, plan_path: Path) -> str:
    import torch

    from analysis.backbones.bootstrap import apply_pythonpath

    apply_pythonpath(backbone)
    from analysis.backbones.cache_replay import get_cache_replay_engine
    from analysis.backbones.registry import get_backbone
    from analysis.prune_plan import UnitPrunePlan, drop_intervals_from_plan, prune_actions_from_plan
    from analysis.recovery.train.model import build_skipped_layer_indices

    adapter = get_backbone(backbone)
    engine = get_cache_replay_engine(backbone)
    plan = UnitPrunePlan.load(plan_path)
    drop_actions, lora_actions = prune_actions_from_plan(plan)
    frame_intervals, global_intervals, _, _ = drop_intervals_from_plan(plan)
    skipped_frame = build_skipped_layer_indices(frame_intervals)
    skipped_global = build_skipped_layer_indices(global_intervals)
    skipped_layers = sorted(set(skipped_frame).union(skipped_global))

    model, _trainable, pre_prune, _removed = engine.build_train_model(
        hf_model=adapter.default_hf_model(),
        device=torch.device("cpu"),
        drop_actions=drop_actions,
        lora_actions=lora_actions,
        frame_skip_intervals=frame_intervals,
        global_skip_intervals=global_intervals,
        skipped_frame_idxs=skipped_frame,
        skipped_global_idxs=skipped_global,
        prune=True,
        inject_lora=True,
    )
    parameters = list(engine.parameter_accounting_parameters(model))
    post_prune = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    frozen = post_prune - trainable
    post_ratio = 100.0 * post_prune / pre_prune if pre_prune > 0 else 0.0
    frozen_ratio = 100.0 * frozen / post_prune if post_prune > 0 else 0.0
    trainable_ratio = 100.0 * trainable / post_prune if post_prune > 0 else 0.0
    return (
        f"[params] {backbone} transformer blocks (excludes embeddings and heads): "
        f"pre_prune={_format_param_count(pre_prune)} "
        f"post_prune={_format_param_count(post_prune)} ({post_ratio:.2f}% of pre-prune) "
        f"frozen={_format_param_count(frozen)} ({frozen_ratio:.2f}% of post-prune) "
        f"trainable={_format_param_count(trainable)} ({trainable_ratio:.2f}% of post-prune) "
        f"skipped {len(skipped_layers)} layers"
    )


def _refresh_train_logs(root: Path, backbone: str) -> None:
    counts_by_plan: Dict[str, str] = {}
    for log_path in sorted((root / backbone).rglob("train.log")):
        plan_name = log_path.parent.name
        plan_path = root / backbone / f"{plan_name}.json"
        if not plan_path.is_file():
            print(f"[warn] skip {log_path}: missing {plan_path}", flush=True)
            continue
        if plan_name not in counts_by_plan:
            counts_by_plan[plan_name] = _parameter_log_line(
                backbone=backbone,
                plan_path=plan_path,
            )
        replacement = counts_by_plan[plan_name]
        lines = log_path.read_text(encoding="utf-8").splitlines()
        replaced = 0
        updated = []
        for line in lines:
            if line.startswith("[params] "):
                updated.append(replacement)
                replaced += 1
            else:
                updated.append(line)
        if replaced == 0:
            print(f"[warn] skip {log_path}: no [params] line", flush=True)
            continue
        log_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print(f"Updated {log_path} ({replaced} parameter line(s))", flush=True)


def _worker(root: Path, backbone: str) -> None:
    _refresh_train_logs(root, backbone)
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(root)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.eval.summary_v3",
            "--experiment",
            backbone,
            "--prune-plan-root",
            str(root / backbone),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def _backend_pythonpath(*, conda_env: str, backbone: str) -> str:
    result = subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-m",
            "analysis.backbones.bootstrap",
            "--backbone",
            backbone,
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _orchestrate(root: Path, backbones: Sequence[str], conda_env: str) -> None:
    for backbone in backbones:
        env = os.environ.copy()
        backend_path = _backend_pythonpath(conda_env=conda_env, backbone=backbone)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (backend_path, str(REPO_ROOT), existing_pythonpath) if part
        )
        subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                conda_env,
                "python",
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--worker-backbone",
                backbone,
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("analysis_outputs_v3"))
    parser.add_argument("--backbones", type=str, default=",".join(SUPPORTED_BACKBONES))
    parser.add_argument("--conda-env", type=str, default="vggt")
    parser.add_argument("--worker-backbone", choices=SUPPORTED_BACKBONES, default="")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker_backbone:
        _worker(root, args.worker_backbone)
        return
    backbones = [value.strip() for value in args.backbones.split(",") if value.strip()]
    unknown = sorted(set(backbones).difference(SUPPORTED_BACKBONES))
    if unknown:
        raise SystemExit(f"Unsupported backbones: {unknown}")
    _orchestrate(root, backbones, args.conda_env)


if __name__ == "__main__":
    main()

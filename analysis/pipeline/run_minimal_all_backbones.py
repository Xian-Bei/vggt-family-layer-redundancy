#!/usr/bin/env python3
"""Minimal end-to-end v3 pipeline for all backbones with per-stage timing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import list_backbones
from analysis.pipeline.timing import RunTimingReport, StageTiming, append_global_summary, timed_step

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_DATASETS = (
    ("scannet", "data/scans_dev50"),
    ("7scenes", "data/7scenes"),
    ("nrgbd", "data/neural_rgbd_data"),
)


def _run(cmd: List[str], *, env: Dict[str, str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _experiment(backbone: str, base: str) -> str:
    return f"{base}/{backbone}"


def _env_for_backbone(backbone: str) -> Dict[str, str]:
    env = os.environ.copy()
    apply_pythonpath(backbone, repo_root_path=REPO_ROOT)
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    env["BACKEND"] = backbone
    return env


def run_backbone(
    backbone: str,
    *,
    experiment_base: str,
    max_frames: int,
    max_steps: int,
    device: str,
    conda_env: str,
    skip_cache: bool,
    skip_redundancy: bool,
    skip_train: bool,
    skip_eval: bool,
) -> RunTimingReport:
    exp = _experiment(backbone, experiment_base)
    report = RunTimingReport(experiment=exp, backbone=backbone, dataset="scannet")
    py = "python"
    env = _env_for_backbone(backbone)
    cache_dir = REPO_ROOT / "analysis_outputs_v3" / exp / "scannet" / "cache"

    # --- Stage 1+2: cache + redundancy (single forward per scene) ---
    st1 = StageTiming(stage="cache_redundancy")
    red_npz = REPO_ROOT / "analysis_outputs_v3" / exp / "scannet" / "redundancy" / "redundancy_graph_all.npz"
    legacy_npz = red_npz.parent / "redundancy_graph_v3.npz"
    has_red = red_npz.is_file() or legacy_npz.is_file()
    if skip_cache and skip_redundancy and (cache_dir / "cache_manifest_v3.json").is_file() and has_red:
        st1.add("cache_redundancy_v3", 0.0, status="skipped", detail="reuse existing cache+redundancy")
    else:
        with timed_step(st1, "cache_redundancy_v3"):
            _run(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    conda_env,
                    py,
                    "-m",
                    "analysis.redundancy.cache_redundancy_v3",
                    "--experiment",
                    exp,
                    "--dataset",
                    "scannet",
                    "--data-root",
                    str(REPO_ROOT / "data/scans_dev50"),
                    "--backbone",
                    backbone,
                    "--device",
                    device,
                    "--max-scenes",
                    "1",
                    "--max-frames",
                    str(max_frames),
                    "--save-cache",
                ],
                env=env,
                cwd=REPO_ROOT,
            )
    report.add_stage(st1)

    # --- Stage 3: train (prune plan from analysis/prune_plan/{backbone}.json) ---
    st3 = StageTiming(stage="train")
    ckpt = REPO_ROOT / "analysis_outputs_v3" / exp / "scannet" / "train/checkpoints/latest.pt"
    prune_plan_root = REPO_ROOT / "analysis" / "prune_plan"
    if skip_train and ckpt.is_file():
        st3.add("train", 0.0, status="skipped", detail="reuse existing checkpoint")
    else:
        with timed_step(st3, "train"):
            _run(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    conda_env,
                    py,
                    "-m",
                    "analysis.recover.train.train_prune_lora_v3",
                    "--experiment",
                    exp,
                    "--dataset",
                    "scannet",
                    "--backbone",
                    backbone,
                    "--device",
                    device,
                    "--prune-plan-root",
                    str(prune_plan_root),
                    "--max-scenes",
                    "1",
                    "--max-steps",
                    str(max_steps),
                    "--epochs",
                    "1",
                    "--data-root",
                    str(REPO_ROOT / "data/scans_dev50"),
                    "--dtype",
                    "bfloat16",
                ],
                env=env,
                cwd=REPO_ROOT,
            )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint after train: {ckpt}")
    report.add_stage(st3)

    # --- Stage 4: eval one scene per dataset ---
    st4 = StageTiming(stage="eval")
    if skip_eval:
        st4.add("eval_all", 0.0, status="skipped", detail="--skip-eval")
    else:
        for ds_name, data_root in EVAL_DATASETS:
            root = REPO_ROOT / data_root
            if not root.is_dir():
                st4.add(f"eval_{ds_name}", 0.0, status="skipped", detail=f"missing {root}")
                continue
            try:
                with timed_step(st4, f"eval_{ds_name}"):
                    _run(
                        [
                            "conda",
                            "run",
                            "--no-capture-output",
                            "-n",
                            conda_env,
                            py,
                            "-m",
                            "analysis.eval.eval_v3",
                            "--experiment",
                            exp,
                            "--dataset",
                            ds_name,
                            "--data-root",
                            str(root),
                            "--backbone",
                            backbone,
                            "--eval-mode",
                            "model",
                            "--model-name",
                            "model",
                            "--prune-plan-root",
                            str(REPO_ROOT / "analysis" / "prune_plan"),
                            "--device",
                            device,
                            "--max-scenes",
                            "1",
                            "--max-frames",
                            str(max_frames),
                            "--checkpoint",
                            str(ckpt),
                        ],
                        env=env,
                        cwd=REPO_ROOT,
                    )
            except subprocess.CalledProcessError as e:
                print(f"[warn] eval {ds_name} failed: {e}", flush=True)
    report.add_stage(st4)
    if any(s.status == "failed" for s in st4.steps):
        st4.status = "partial"
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal v3 pipeline for all backbones.")
    p.add_argument("--experiment-base", type=str, default="minimal_bb")
    p.add_argument("--backbones", type=str, default="")
    p.add_argument("--max-frames", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conda-env", type=str, default="vggt")
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--skip-redundancy", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    names = [b.strip() for b in args.backbones.split(",") if b.strip()] if args.backbones else list_backbones()
    summary_rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for bb in names:
        t0 = time.perf_counter()
        try:
            rep = run_backbone(
                bb,
                experiment_base=args.experiment_base,
                max_frames=int(args.max_frames),
                max_steps=int(args.max_steps),
                device=str(args.device),
                conda_env=str(args.conda_env),
                skip_cache=bool(args.skip_cache),
                skip_redundancy=bool(args.skip_redundancy),
                skip_train=bool(args.skip_train),
                skip_eval=bool(args.skip_eval),
            )
            out = REPO_ROOT / "analysis_outputs_v3" / rep.experiment / "pipeline_timing.json"
            rep.save(out)
            summary_rows.append(
                {
                    "label": rep.experiment,
                    "status": "ok",
                    "grand_total_seconds": rep.grand_total_seconds,
                    "stages": [
                        {
                            "stage": s.stage,
                            "total_seconds": s.total_seconds,
                            "status": s.status,
                            "steps": [asdict(x) for x in s.steps],
                        }
                        for s in rep.stages
                    ],
                }
            )
            print(f"[ok] {bb} total={rep.grand_total_seconds:.1f}s", flush=True)
        except Exception as e:
            failed.append(bb)
            summary_rows.append(
                {
                    "label": _experiment(bb, args.experiment_base),
                    "status": "failed",
                    "grand_total_seconds": time.perf_counter() - t0,
                    "error": str(e),
                    "stages": [],
                }
            )
            print(f"[fail] {bb}: {e}", flush=True)

    global_path = REPO_ROOT / "analysis_outputs_v3" / args.experiment_base / "timing_summary_all_backbones.md"
    append_global_summary(global_path, summary_rows)
    (REPO_ROOT / "analysis_outputs_v3" / args.experiment_base / "timing_summary_all_backbones.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    if failed:
        raise SystemExit(f"Failed backbones: {failed}")
    print(f"All backbones passed. Summary: {global_path}", flush=True)


if __name__ == "__main__":
    main()

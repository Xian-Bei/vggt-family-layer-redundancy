"""
Stage 3: Load pretrained VGGT, apply a redundancy scheme (global block skip / head prune / FFN prune),
run the same ScanNet-style inference as Stage 1, and compare GT metrics against the Stage 1 baseline.

Output layout per scheme (``--output-dir/<scheme_name>/``) writes ``manifest.json`` and ``scheme_meta.json``.
Metrics are written to ``eval_scannet.json`` (FastVGGT-aligned when deps/data allow; unless ``--skip-gt-eval``).
With ``--stage1-cache-dir``, also writes ``compare_vs_stage1.json``; after all schemes,
``all_schemes_compare_stage1.json`` aggregates per-scheme comparisons.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT

from .collect import infer_bs_and_seq_len, predictions_to_cpu_float
from .fastvggt_scannet_eval import dependencies_available
from .prediction_eval import (
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
    delta_gt_metrics,
    metrics_float_dict,
    read_eval_scannet_rows_by_scene,
    resolve_gt_ply_dir,
    write_eval_scannet_json,
)
from .prediction_save import build_scene_prediction_payload
from .pruning import PruneConfig, apply_structured_pruning, prune_config_from_scheme_dict
from .scannet_eval import (
    SceneData,
    aggregate_metric_dict,
    count_parameters,
    discover_scannet_scenes,
    estimate_global_flops,
    load_scene_images,
    load_schemes_json,
    select_scene_image_paths_fastvggt,
)


def _pick_schemes(path: Path, scheme_name_filter: Optional[str]) -> List[Dict[str, Any]]:
    schemes = load_schemes_json(path)
    if not scheme_name_filter or not scheme_name_filter.strip():
        return schemes
    wanted = {n.strip() for n in scheme_name_filter.split(",") if n.strip()}
    out = [s for s in schemes if str(s.get("scheme_name", "")) in wanted]
    if not out:
        avail = [str(s.get("scheme_name", "")) for s in schemes]
        raise SystemExit(f"No scheme matched {wanted}. Available scheme_name values: {avail}")
    return out


def _write_compare_vs_stage1_json(out_path: Path, stage1_dir_str: str, rows: List[Dict[str, Any]]) -> None:
    agg_gt3 = aggregate_metric_dict(
        [r["gt_metrics_stage3"] for r in rows if r.get("gt_metrics_stage3")]
    )
    agg_gt1 = aggregate_metric_dict(
        [r["gt_metrics_stage1"] for r in rows if r.get("gt_metrics_stage1")]
    )
    agg_delta = aggregate_metric_dict(
        [r["delta_gt_vs_stage1"] for r in rows if r.get("delta_gt_vs_stage1")]
    )
    payload = {
        "stage1_cache_dir": stage1_dir_str,
        "scenes": rows,
        "aggregate_gt_stage3": agg_gt3,
        "aggregate_gt_stage1": agg_gt1,
        "aggregate_delta_gt_vs_stage1": agg_delta,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _short_distill(
    student: VGGT,
    teacher: VGGT,
    images: torch.Tensor,
    steps: int,
    lr: float,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> None:
    """Light MSE alignment on pose/depth/points (optional). ``images`` is ``[S,3,H,W]`` on device."""
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for _ in range(steps):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            t_out = teacher(images)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            s_out = student(images)
        loss = torch.zeros((), device=images.device, dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            for key in ("pose_enc", "depth", "world_points"):
                if key in t_out and key in s_out:
                    loss = loss + F.mse_loss(s_out[key].float(), t_out[key].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.eval()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pruned VGGT baseline on ScanNet layout; compare GT metrics to Stage1.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--schemes-json",
        type=Path,
        default=Path("analysis/redundancy/schemes.example.json"),
        help="JSON with {\"schemes\": [...]} listing scheme_name, skip/prune fields.",
    )
    p.add_argument(
        "--scheme-name",
        type=str,
        default="",
        help="Comma-separated scheme_name values; empty = run every scheme in the file.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/scannet_baseline"))
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-scenes", type=int, default=0, help="0 = all discovered scenes.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="FastVGGT-style frame cap: pose∩images, keep first + ~uniform step. 0 = all intersecting frames.",
    )
    p.add_argument(
        "--recovery-steps",
        type=int,
        default=0,
        help="Optional MSE distillation steps on the first scene (0 = off).",
    )
    p.add_argument("--recovery-lr", type=float, default=1e-5)
    p.add_argument(
        "--skip-gt-eval",
        action="store_true",
        help="Skip pose/geometry/depth metrics vs GT. Stage1 comparison also requires GT eval rows.",
    )
    p.add_argument(
        "--stage1-cache-dir",
        type=Path,
        default=None,
        help="Stage 1 output dir (must contain eval_scannet.json). After each scheme, write compare_vs_stage1.json with GT deltas vs Stage1.",
    )
    p.add_argument(
        "--gt-ply-dir",
        type=Path,
        default=None,
        help="ScanNet ply root (parent of scene folders). Default: <data-root>/scans if present.",
    )
    p.add_argument("--chamfer-max-dist", type=float, default=0.5)
    p.add_argument("--depth-conf-thresh", type=float, default=1.0)
    return p


def _run_scheme(
    cfg: PruneConfig,
    scenes: List[SceneData],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> None:
    safe_scheme = cfg.scheme_name.replace("/", "_").replace(" ", "_")
    scheme_dir = args.output_dir / safe_scheme
    scheme_dir.mkdir(parents=True, exist_ok=True)

    teacher = VGGT.from_pretrained(args.hf_model).to(device).eval()
    student = copy.deepcopy(teacher).eval()
    pruned = apply_structured_pruning(student, cfg)
    student.to(device)

    use_amp = device.type == "cuda"
    if args.recovery_steps > 0 and scenes:
        ref_eff = select_scene_image_paths_fastvggt(scenes[0], args.max_frames)
        if ref_eff.image_paths:
            images = load_scene_images(ref_eff, mode="crop").to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            _short_distill(
                student,
                teacher,
                images,
                steps=args.recovery_steps,
                lr=args.recovery_lr,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )

    del teacher

    sample_eff = select_scene_image_paths_fastvggt(scenes[0], args.max_frames)
    if not sample_eff.image_paths:
        raise SystemExit("No frames after FastVGGT-style selection for the first scene.")
    sample_im = load_scene_images(sample_eff, mode="crop")
    seq_len_est = int(sample_im.shape[0])
    image_size_est = int(sample_im.shape[-1])

    manifest: List[Dict[str, Any]] = []
    gt_eval_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    stage1_eval_rows: Optional[Dict[str, Dict[str, Any]]] = None
    if args.stage1_cache_dir is not None:
        eval_path = args.stage1_cache_dir / "eval_scannet.json"
        if not eval_path.exists():
            raise SystemExit(f"Missing Stage1 eval file: {eval_path}")
        try:
            stage1_eval_rows = read_eval_scannet_rows_by_scene(eval_path)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e

    for si, scene in enumerate(scenes):
        print(f"  [{cfg.scheme_name}] scene {si + 1}/{len(scenes)}: {scene.scene_id}")
        scene_eff = select_scene_image_paths_fastvggt(scene, args.max_frames)
        if not scene_eff.image_paths:
            if args.skip_gt_eval:
                print(f"    [warn] skip scene {scene.scene_id}: no frames after pose/image intersection")
                continue
            raise SystemExit(
                f"Scene {scene.scene_id} has no frames after pose/image intersection. "
                "GT eval expects complete ScanNet data; fix the dataset instead of skipping."
            )
        images = load_scene_images(scene_eff, mode="crop").to(device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            preds = student(images)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        preds_cpu = predictions_to_cpu_float(preds)
        pred_payload = build_scene_prediction_payload(preds_cpu, (int(images.shape[-2]), int(images.shape[-1])))

        _b, num_frames = infer_bs_and_seq_len(images)
        manifest.append(
            {
                "scene_id": scene.scene_id,
                "num_frames": num_frames,
                "infer_time_ms": infer_ms,
            }
        )

        m3_gt: Dict[str, float] = {}
        if not args.skip_gt_eval:
            erow = build_scannet_eval_row_from_payload_with_context(
                pred_payload,
                scene_eff,
                build_fastvggt_scene_eval_context(
                    scene_eff,
                    args.data_root,
                    args.gt_ply_dir,
                    image_hw=(int(images.shape[-2]), int(images.shape[-1])),
                ),
                depth_conf_thresh=args.depth_conf_thresh,
                chamfer_max_dist=args.chamfer_max_dist,
                inference_time_ms=infer_ms,
            )
            gt_eval_rows.append(erow)
            m3_gt = metrics_float_dict(erow)

        if stage1_eval_rows is not None:
            if scene.scene_id not in stage1_eval_rows:
                raise SystemExit(
                    f"Missing Stage1 eval row for compare: {scene.scene_id}. "
                    "Stage1 cache/eval data must be complete."
                )
            elif not args.skip_gt_eval and m3_gt:
                m1_gt = metrics_float_dict(stage1_eval_rows[scene.scene_id])
                compare_rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "infer_time_ms": infer_ms,
                        "gt_metrics_stage3": m3_gt,
                        "gt_metrics_stage1": m1_gt,
                        "delta_gt_vs_stage1": delta_gt_metrics(m3_gt, m1_gt),
                    }
                )

    meta = {
        "scheme_name": cfg.scheme_name,
        "prune_config": {
            "scheme_name": cfg.scheme_name,
            "prune_head_ratio": cfg.prune_head_ratio,
            "prune_ffn_ratio": cfg.prune_ffn_ratio,
            "global_skip_layers": list(cfg.global_skip_layers),
            "global_head_layers": list(cfg.global_head_layers),
            "global_ffn_layers": list(cfg.global_ffn_layers),
        },
        "pruned_layers": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in pruned.items()},
        "hf_model": args.hf_model,
        "recovery_steps": args.recovery_steps,
        "global_module_param_count": count_parameters(student, global_only=True),
        "model_param_count": count_parameters(student, global_only=False),
        "global_module_flops_estimate": estimate_global_flops(student, seq_len=seq_len_est, image_size=image_size_est),
    }
    if gt_eval_rows:
        meta["scene_average_metrics"] = aggregate_metric_dict([metrics_float_dict(r) for r in gt_eval_rows])

    (scheme_dir / "scheme_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest_payload: Dict[str, Any] = {
        "scheme_name": cfg.scheme_name,
        "hf_model": args.hf_model,
        "scenes": manifest,
    }
    if gt_eval_rows:
        manifest_payload["eval_scannet_file"] = "eval_scannet.json"
    if compare_rows:
        manifest_payload["compare_vs_stage1_file"] = "compare_vs_stage1.json"
    (scheme_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    if gt_eval_rows:
        rp = resolve_gt_ply_dir(args.data_root, args.gt_ply_dir)
        write_eval_scannet_json(
            scheme_dir,
            gt_eval_rows,
            extra_top_level={
                "hf_model": args.hf_model,
                "scheme_name": cfg.scheme_name,
                "stage": "run_scannet_baseline",
                "evaluation": "fastvggt_scannet",
                "gt_ply_dir": str(rp.resolve()) if rp else None,
            },
        )
    if compare_rows and args.stage1_cache_dir is not None:
        _write_compare_vs_stage1_json(scheme_dir / "compare_vs_stage1.json", str(args.stage1_cache_dir.resolve()), compare_rows)

    print(f"  Wrote {len(manifest)} scenes under {scheme_dir}")


def main() -> None:
    args = _build_parser().parse_args()
    if not args.hf_model.strip():
        raise SystemExit("--hf-model is required for baseline (pretrained weights).")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map[args.dtype]

    schemes_raw = _pick_schemes(args.schemes_json, args.scheme_name or None)
    cfgs = [prune_config_from_scheme_dict(s) for s in schemes_raw]

    scenes = discover_scannet_scenes(args.data_root)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise SystemExit(f"No scenes under {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_gt_eval and not dependencies_available():
        raise SystemExit("GT eval requires: pip install evo open3d scipy")
    print(f"Stage 3 baseline: {len(cfgs)} scheme(s), {len(scenes)} scene(s) -> {args.output_dir}")

    for cfg in cfgs:
        print(f"== Scheme: {cfg.scheme_name} ==")
        _run_scheme(cfg, scenes, args, device, amp_dtype)

    if args.stage1_cache_dir is not None:
        summary_schemes: List[Dict[str, Any]] = []
        for cfg in cfgs:
            safe_scheme = cfg.scheme_name.replace("/", "_").replace(" ", "_")
            cp = args.output_dir / safe_scheme / "compare_vs_stage1.json"
            if not cp.exists():
                continue
            data = json.loads(cp.read_text(encoding="utf-8"))
            summary_schemes.append(
                {
                    "scheme_name": cfg.scheme_name,
                    "aggregate_gt_stage3": data.get("aggregate_gt_stage3", {}),
                    "aggregate_delta_gt_vs_stage1": data.get("aggregate_delta_gt_vs_stage1", {}),
                }
            )
        if summary_schemes:
            out_summary = args.output_dir / "all_schemes_compare_stage1.json"
            out_summary.write_text(
                json.dumps(
                    {"stage1_cache_dir": str(args.stage1_cache_dir.resolve()), "schemes": summary_schemes},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote {out_summary} ({len(summary_schemes)} scheme(s)).")

    print("Done.")


if __name__ == "__main__":
    main()

"""Local redundancy / Pareto probe using a randomly initialized VGGT (no HF weights).

For ScanNet with pretrained checkpoints and on-disk predictions comparable to Stage 1,
use ``python -m analysis.redundancy.run_scannet_baseline`` instead.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from vggt.models.vggt import VGGT
from .collect import collect_batch, summarize_redundancy
from .scannet_eval import (
    aggregate_metric_dict,
    count_parameters,
    discover_scenes,
    estimate_global_flops,
    load_scene_images,
)


def _avg_offdiag(matrix: list[list[float]]) -> float:
    n = len(matrix)
    if n <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += matrix[i][j]
                count += 1
    return total / max(count, 1)


def _corr(x: Dict[str, float], y: Dict[str, float]) -> float:
    keys = [k for k in x.keys() if k in y]
    if len(keys) < 2:
        return 0.0
    xv = torch.tensor([x[k] for k in keys], dtype=torch.float32)
    yv = torch.tensor([y[k] for k in keys], dtype=torch.float32)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = (xv.norm() * yv.norm()).item()
    if denom == 0:
        return 0.0
    return float(torch.dot(xv, yv).item() / denom)


def _avg_nested_mats(reports, key_a: str, key_b: str):
    mats = [torch.tensor(r[key_a][key_b], dtype=torch.float32) for r in reports]
    return (sum(mats) / len(mats)).tolist()


def _average_reports(reports):
    out = {
        "global_patch_similarity": {},
        "global_special_similarity": {},
        "attention_stats": {"global": []},
        "spectral_stats": [],
        "scene_scores": {},
    }
    for metric in ("cka", "cosine", "pwcca", "neg_mse"):
        out["global_patch_similarity"][metric] = _avg_nested_mats(reports, "global_patch_similarity", metric)
        out["global_special_similarity"][metric] = _avg_nested_mats(reports, "global_special_similarity", metric)
    n_layers = len(reports[0]["attention_stats"]["global"])
    for i in range(n_layers):
        vals = [r["attention_stats"]["global"][i]["mean_head_cosine"] for r in reports]
        out["attention_stats"]["global"].append({"layer": i, "mean_head_cosine": float(sum(vals) / len(vals))})
    for i in range(len(reports[0]["spectral_stats"])):
        keys = [k for k in reports[0]["spectral_stats"][i].keys() if k not in {"layer", "module"}]
        row = {
            "layer": reports[0]["spectral_stats"][i]["layer"],
            "module": reports[0]["spectral_stats"][i]["module"],
        }
        for k in keys:
            row[k] = float(sum(r["spectral_stats"][i][k] for r in reports) / len(reports))
        out["spectral_stats"].append(row)
    score_keys = sorted({k for r in reports for k in r["scene_scores"].keys()})
    for k in score_keys:
        out["scene_scores"][k] = float(sum(r["scene_scores"].get(k, 0.0) for r in reports) / len(reports))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VGGT redundancy baseline study.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/baseline"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    model = VGGT(
        img_size=args.image_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
    ).eval()

    scenes = discover_scenes(args.data_root)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise ValueError(f"No scenes found under {args.data_root}")

    reports = []
    per_scene_metrics = []
    for scene in scenes:
        images = load_scene_images(scene, mode="crop")
        batch_result = collect_batch(model, images.unsqueeze(0))
        report = summarize_redundancy(model, batch_result)
        reports.append(report)

        st = time.perf_counter()
        with torch.no_grad():
            _ = model(images.unsqueeze(0))
        infer_ms = (time.perf_counter() - st) * 1000.0
        per_scene_metrics.append({"scene_id": scene.scene_id, "infer_time_ms": infer_ms})

    report = _average_reports(reports)
    patch_cka = _avg_offdiag(report["global_patch_similarity"]["cka"])
    patch_cos = _avg_offdiag(report["global_patch_similarity"]["cosine"])
    patch_pwcca = _avg_offdiag(report["global_patch_similarity"]["pwcca"])
    patch_neg_mse = _avg_offdiag(report["global_patch_similarity"]["neg_mse"])
    special_cka = _avg_offdiag(report["global_special_similarity"]["cka"])
    head_redundancy = sum(x["mean_head_cosine"] for x in report["attention_stats"]["global"]) / max(
        len(report["attention_stats"]["global"]), 1
    )

    branch_proxy = report["scene_scores"]
    redundancy_scores = {
        "patch_cka": patch_cka,
        "patch_cosine": patch_cos,
        "patch_pwcca": patch_pwcca,
        "patch_neg_mse": patch_neg_mse,
        "special_cka": special_cka,
        "head_redundancy": head_redundancy,
    }
    # Simple internal consistency proxy.
    branch_corr = _corr(
        {f"r_{k}": v for k, v in redundancy_scores.items()},
        {f"r_{k}": float(sum(branch_proxy.values()) / max(len(branch_proxy), 1)) for k in redundancy_scores.keys()},
    )

    summary = {
        "seed": args.seed,
        "input": {
            "num_scenes": len(scenes),
            "image_size": args.image_size,
        },
        "redundancy_scores": redundancy_scores,
        "scene_proxy_scores": branch_proxy,
        "redundancy_branch_proxy_correlation": branch_corr,
        "global_module_param_count": count_parameters(model, global_only=True),
        "model_param_count": count_parameters(model, global_only=False),
        "global_module_flops_estimate": estimate_global_flops(model, seq_len=len(scenes[0].image_paths), image_size=args.image_size),
        "scene_average_metrics": aggregate_metric_dict([{k: v for k, v in r.items() if k != "scene_id"} for r in per_scene_metrics]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "redundancy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "scene_metrics.json").write_text(json.dumps(per_scene_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# VGGT Redundancy Baseline",
        "",
        "## Redundancy scores",
        f"- patch_cka: {patch_cka:.6f}",
        f"- patch_cosine: {patch_cos:.6f}",
        f"- patch_pwcca: {patch_pwcca:.6f}",
        f"- special_cka: {special_cka:.6f}",
        f"- head_redundancy: {head_redundancy:.6f}",
        "",
        "## Scene proxy scores",
    ]
    for k, v in branch_proxy.items():
        md.append(f"- {k}: {v:.6f}")
    md.extend(
        [
            "",
            "## Correlation",
            f"- redundancy_scene_proxy_correlation: {branch_corr:.6f}",
            f"- global_module_param_count: {summary['global_module_param_count']}",
            f"- model_param_count: {summary['model_param_count']}",
            f"- global_module_flops_estimate: {summary['global_module_flops_estimate']:.2f}",
            "",
            "Artifacts:",
            "- `redundancy_report.json` contains per-layer pairwise heatmap matrices.",
            "- `summary.json` contains top-level aggregate metrics.",
            "- `scene_metrics.json` contains per-scene metrics and inference time.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Baseline report written to {args.output_dir}")


if __name__ == "__main__":
    main()


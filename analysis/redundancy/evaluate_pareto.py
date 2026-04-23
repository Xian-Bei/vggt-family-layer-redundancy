import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from vggt.models.vggt import VGGT
from .pruning import apply_structured_pruning, prune_config_from_scheme_dict
from .scannet_eval import (
    count_parameters,
    discover_scenes,
    estimate_global_flops,
    load_scene_images,
    load_schemes_json,
)


def _latency_ms(model: VGGT, images: torch.Tensor, warmup: int = 1, iters: int = 5) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(images)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(images)
        end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _quality_proxy(student: VGGT, teacher: VGGT, images: torch.Tensor) -> float:
    with torch.no_grad():
        t = teacher(images)
        s = student(images)
    score = 0.0
    n = 0
    for key in ("pose_enc", "depth", "world_points"):
        if key in t and key in s:
            mse = torch.nn.functional.mse_loss(s[key], t[key]).item()
            score += 1.0 / (1.0 + mse)
            n += 1
    return score / max(n, 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VGGT pruning Pareto frontier.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--schemes-json", type=Path, required=True)
    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="If set, load pretrained VGGT from HuggingFace (recommended). Empty = random-init probe (legacy).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=518, help="Only used when --hf-model is empty.")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pareto"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.hf_model.strip():
        teacher = VGGT.from_pretrained(args.hf_model).to(device).eval()
    else:
        teacher = VGGT(
            img_size=args.image_size,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_depth=True,
            enable_point=True,
            enable_track=False,
        ).to(device).eval()
    scenes = discover_scenes(args.data_root)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise ValueError(f"No scenes found under {args.data_root}")
    images = load_scene_images(scenes[0], mode="crop").unsqueeze(0).to(device)
    base_latency = _latency_ms(teacher, images)
    seq_len = int(images.shape[1])
    image_size_est = int(images.shape[-1])

    schemes = [prune_config_from_scheme_dict(x) for x in load_schemes_json(args.schemes_json)]

    rows: List[Dict[str, float]] = []
    detail_rows = []
    for cfg in schemes:
        student = copy.deepcopy(teacher).eval()
        apply_structured_pruning(student, cfg)
        student.to(device)
        latency = _latency_ms(student, images)
        quality = _quality_proxy(student, teacher, images)

        rows.append(
            {
                "scheme_name": cfg.scheme_name,
                "prune_head_ratio": cfg.prune_head_ratio,
                "prune_ffn_ratio": cfg.prune_ffn_ratio,
                "latency_ms": latency,
                "speedup_vs_base": base_latency / max(latency, 1e-8),
                "quality_proxy": quality,
                "global_module_param_count": float(count_parameters(student, global_only=True)),
                "global_module_flops_estimate": estimate_global_flops(student, seq_len=seq_len, image_size=image_size_est),
            }
        )
        detail_rows.append({"scheme_name": cfg.scheme_name})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pareto.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "pareto_details.json").write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda x: (-x["quality_proxy"], -x["speedup_vs_base"]))
    md_lines = [
        "# VGGT Pareto Summary",
        "",
        f"- Baseline latency (ms): {base_latency:.6f}",
        "",
        "## Candidates",
    ]
    for row in rows:
        md_lines.append(
            "- {}: head={:.3f}, ffn={:.3f}, latency_ms={:.4f}, speedup={:.4f}, quality={:.6f}".format(
                row["scheme_name"],
                row["prune_head_ratio"],
                row["prune_ffn_ratio"],
                row["latency_ms"],
                row["speedup_vs_base"],
                row["quality_proxy"],
            )
        )
    md_lines.extend(["", "## Recommended", "- " + (str(rows_sorted[0]) if rows_sorted else "(none)")])
    (args.output_dir / "pareto_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Pareto artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()


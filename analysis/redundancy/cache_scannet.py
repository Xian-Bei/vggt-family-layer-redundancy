"""
Step 1: run VGGT on all ScanNet (or compatible) scenes and save global-block
activations for offline redundancy analysis.

Stage1 now saves one canonical activation cache:
full per-layer ``concat(frame_k, global_k)`` tensors.
Stage2 reconstructs both the default intra-layer similarity inputs and the
legacy inter-layer global outputs from this cache directly.
GT evaluation and optional visualization export operate directly on in-memory
prediction payloads; the mainline Stage 1 flow no longer writes per-scene
prediction files for later reload.

Stage1 GT metrics now include both:
- pose / geometry: ``ate``, ``are``, ``rpe_rot``, ``rpe_trans``, ``chamfer_distance``
- dense depth: ``depth_abs_rel`` and ``depth_delta1``

For ScanNet depth GT, raw depth PNGs are aligned to the same RGB ``crop`` preprocessing
used for model input, invalid values are masked from the depth map itself, and values
outside the FastVGGT-style indoor range ``(1e-3m, 10m)`` are ignored.
Frame blocks and head definitions are unchanged at inference time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from vggt.models.vggt import VGGT

from .collect import collect_batch, infer_bs_and_seq_len
from analysis.eval.eval_utils import dependencies_available
from analysis.eval.prediction_eval import (
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
    write_eval_scannet_json,
)
from .prediction_save import build_scene_prediction_payload
from .scannet_eval import discover_scannet_scenes, load_scene_images, select_scene_image_paths_fastvggt
from .visualization_export import export_scene_visualization_assets


def _normalize_scene_pred_for_cache(predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep a compact per-scene prediction subset in cache payload."""
    out: Dict[str, torch.Tensor] = {}
    for key in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
        value = predictions.get(key)
        if value is None or not isinstance(value, torch.Tensor):
            continue
        t = value.detach().cpu()
        if t.dim() > 0 and int(t.shape[0]) == 1:
            t = t.squeeze(0)
        out[key] = t.contiguous()
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache global-block activations for all scenes under data/.")
    p.add_argument("--data-root", type=Path, default=Path("data/scannetv2"), help="Root containing scans/ or scene folders.")
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/scannet_cache"))
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B", help="HuggingFace model id; use empty to skip load (random init, debug only).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-scenes", type=int, default=10, help="0 = all scenes.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Max RGB frames per scene: FastVGGT-style selection (pose ∩ images, first frame + ~uniform step). "
        "Use 0 for all valid intersecting frames.",
    )
    p.add_argument(
        "--max-tokens-per-layer",
        type=int,
        default=0,
        help="Deprecated compatibility knob. Stage1 now always saves full aggregated_concat_per_layer.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-gt-eval",
        action="store_true",
        help="Do not run pose/geometry/depth metrics vs ScanNet GT after saving predictions "
        "(default: eval when GT files exist).",
    )
    p.add_argument(
        "--gt-ply-dir",
        type=Path,
        default=None,
        help="Parent of scene folders containing *_vh_clean_2.ply (e.g. data/scans). Default: <data-root>/scans if it exists.",
    )
    p.add_argument("--chamfer-max-dist", type=float, default=0.5)
    p.add_argument("--depth-conf-thresh", type=float, default=1.0)
    p.add_argument(
        "--export-visualization",
        action="store_true",
        help="Export per-scene visualization assets under <output-dir>/visualization/<scene_id>/.",
    )
    p.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Visualization root. Default: <output-dir>/visualization",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map[args.dtype]

    scenes = discover_scannet_scenes(args.data_root)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise SystemExit(f"No scenes found under {args.data_root} (expected scans/*/color or */images).")

    if not args.skip_gt_eval and not dependencies_available():
        raise SystemExit("GT eval requires: pip install evo open3d scipy")

    if args.hf_model.strip():
        model = VGGT.from_pretrained(args.hf_model).to(device).eval()
    else:
        model = VGGT().to(device).eval()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = args.visualization_dir if args.visualization_dir is not None else (out_dir / "visualization")

    manifest: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for si, scene in enumerate(scenes):
        print(f"Processing scene {si} of {len(scenes)}: {scene.scene_id}")
        scene_eff = select_scene_image_paths_fastvggt(scene, args.max_frames)
        safe_id = scene.scene_id.replace("/", "_")
        scene_dir = out_dir / safe_id
        layers_dir = scene_dir / "layers"
        metadata_path = scene_dir / "metadata.json"
        pred_path = scene_dir / "pred.pt"
        if not scene_eff.image_paths:
            if args.skip_gt_eval:
                print(f"Skipping scene {scene.scene_id}: no frames after pose/image intersection (or empty paths)")
                continue
            raise SystemExit(
                f"Scene {scene.scene_id} has no frames after pose/image intersection (or empty paths). "
                "GT eval expects complete ScanNet data; fix the dataset instead of skipping."
            )
        images = load_scene_images(scene_eff, mode="crop").to(device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        use_amp = device.type == "cuda"
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=use_amp
        ):
            batch = collect_batch(model, images, keep_activations_on_gpu=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        _b, num_frames = infer_bs_and_seq_len(images)
        payload = {
            "scene_id": scene.scene_id,
            "num_frames": num_frames,
            "image_hw": [int(images.shape[-2]), int(images.shape[-1])],
            "patch_start_idx": int(batch.patch_start_idx),
            "replay_cache_format": "aggregated_concat_per_layer_v1",
            "scene_scores": batch.scene_scores,
            "aggregated_concat_per_layer": [t.detach().cpu().contiguous() for t in batch.aggregated_concat_tokens],
            "pred": _normalize_scene_pred_for_cache(batch.predictions),
        }
        scene_dir.mkdir(parents=True, exist_ok=True)
        layers_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, layer_tensor in enumerate(payload["aggregated_concat_per_layer"]):
            torch.save(layer_tensor, layers_dir / f"layer_{layer_idx:02d}.pt")
        metadata = {
            "scene_id": payload["scene_id"],
            "num_frames": payload["num_frames"],
            "image_hw": payload["image_hw"],
            "patch_start_idx": payload["patch_start_idx"],
            "replay_cache_format": payload["replay_cache_format"],
            "scene_scores": payload["scene_scores"],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(payload["pred"], pred_path)
        pred_payload = build_scene_prediction_payload(batch.predictions, (int(images.shape[-2]), int(images.shape[-1])))
        if args.export_visualization:
            vis_dir = export_scene_visualization_assets(
                output_root=vis_root,
                scene_id=scene.scene_id,
                image_paths=scene_eff.image_paths,
                images_tensor=images,
                prediction_payload=pred_payload,
                depth_conf_thresh=args.depth_conf_thresh,
            )
            if vis_dir is not None:
                print(f"    [vis] exported {vis_dir}")
        manifest.append(
            {
                "scene_id": scene.scene_id,
                "dir": safe_id,
                "num_frames": payload["num_frames"],
                "frame_ids": [int(p.stem) for p in scene_eff.image_paths],
            }
        )
        if not args.skip_gt_eval:
            eval_rows.append(
                build_scannet_eval_row_from_payload_with_context(
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
            )

    manifest_payload: Dict[str, Any] = {
        "scenes": manifest,
        "hf_model": args.hf_model,
        "replay_cache_format": "aggregated_concat_per_layer_v1",
    }
    if eval_rows:
        manifest_payload["eval_scannet_file"] = "eval_scannet.json"
    (out_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    if eval_rows:
        from analysis.eval.prediction_eval import resolve_gt_ply_dir

        resolved_ply = resolve_gt_ply_dir(args.data_root, args.gt_ply_dir)
        write_eval_scannet_json(
            out_dir,
            eval_rows,
            extra_top_level={
                "hf_model": args.hf_model,
                "stage": "cache_scannet",
                "evaluation": "fastvggt_scannet",
                "gt_ply_dir": str(resolved_ply.resolve()) if resolved_ply else None,
            },
        )
        print(f"Wrote eval_scannet.json ({len(eval_rows)} scenes) under {out_dir}")
    print(f"Cached {len(manifest)} scenes under {out_dir}")


if __name__ == "__main__":
    main()

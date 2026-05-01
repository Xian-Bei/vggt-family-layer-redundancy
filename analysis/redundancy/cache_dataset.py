"""
Stage1: run VGGT and cache activations for supported datasets.

Datasets:
- scannet
- 7scenes
- nrgbd
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from vggt.models.vggt import VGGT

from .collect import collect_batch, infer_bs_and_seq_len
from .prediction_save import build_scene_prediction_payload
from analysis.eval.eval_utils import dependencies_available
from analysis.eval.prediction_eval import (
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
)
from analysis.eval.scannet_eval import (
    SceneData,
    discover_scannet_scenes,
    load_scene_images,
    select_scene_image_paths_fastvggt,
)


DATASET_CHOICES = ("scannet", "7scenes", "nrgbd")


def _normalize_scene_pred_for_cache(predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("pose_enc", "depth", "depth_conf"):
        value = predictions.get(key)
        if value is None or not isinstance(value, torch.Tensor):
            continue
        t = value.detach().cpu()
        if t.dim() > 0 and int(t.shape[0]) == 1:
            t = t.squeeze(0)
        out[key] = t.contiguous()
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache global-block activations for datasets.")
    p.add_argument("--dataset", type=str, default="scannet", choices=DATASET_CHOICES)
    p.add_argument("--data-root", type=Path, default=Path("data/scans_dev50"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/cache/scannet"))
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes.")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--max-tokens-per-layer", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-gt-eval", action="store_true")
    p.add_argument("--gt-ply-dir", type=Path, default=None)
    p.add_argument("--chamfer-max-dist", type=float, default=0.5)
    p.add_argument("--depth-conf-thresh", type=float, default=1.0)
    # run_eval.sh-aligned knobs for 7scenes/nrgbd.
    p.add_argument("--kf", type=int, default=10)
    p.add_argument("--size", type=int, default=518)
    p.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-scene inference time and breakdown (default: on). Use --no-timing to disable.",
    )
    return p


def _resolve_7n_resolution(size: int) -> Any:
    if int(size) == 512:
        return (512, 384)
    if int(size) == 224:
        return 224
    if int(size) == 518:
        return (518, 392)
    raise SystemExit(f"Unsupported --size for 7scenes/nrgbd: {size}")


def _import_7n_datasets():
    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    eval_dir_s = str(eval_dir)
    if eval_dir_s not in sys.path:
        sys.path.insert(0, eval_dir_s)
    from data import NRGBD, SevenScenes  # type: ignore

    return SevenScenes, NRGBD


def _extract_scene_inputs_from_eval_batch(batch: List[Dict[str, Any]]) -> Tuple[str, List[str], torch.Tensor]:
    if not batch:
        raise SystemExit("Empty batch from 7scenes/nrgbd dataset.")

    labels: List[str] = []
    image_paths: List[str] = []
    imgs: List[torch.Tensor] = []
    for view in batch:
        label_raw = view.get("label", [""])
        label = str(label_raw[0] if isinstance(label_raw, list) and label_raw else label_raw).strip()
        labels.append(label)

        path_raw = view.get("path", view.get("instance", [""]))
        img_path = str(path_raw[0] if isinstance(path_raw, list) and path_raw else path_raw).strip()
        image_paths.append(img_path)

        img_t = view.get("img")
        if not isinstance(img_t, torch.Tensor):
            raise SystemExit("Batch view missing tensor field 'img'.")
        imgs.append((img_t + 1.0) / 2.0)

    first_label = labels[0] if labels else ""
    scene_id = "/".join(first_label.split("/")[:-1]) if "/" in first_label else first_label
    if not scene_id:
        p0 = Path(image_paths[0]) if image_paths else Path("unknown")
        scene_id = p0.parent.name

    images = torch.cat(imgs, dim=0)
    return scene_id, image_paths, images


def _iter_7n_scene_inputs(args: argparse.Namespace) -> List[Tuple[str, List[str], torch.Tensor]]:
    SevenScenes, NRGBD = _import_7n_datasets()
    res = _resolve_7n_resolution(int(args.size))
    seed = 33
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.dataset == "7scenes":
        ds = SevenScenes(
            split="train",
            ROOT=str(args.data_root),
            resolution=res,
            num_seq=1,
            full_video=True,
            kf_every=int(args.kf),
            seed=seed,
        )
    else:
        ds = NRGBD(
            split="train",
            ROOT=str(args.data_root),
            resolution=res,
            num_seq=1,
            full_video=True,
            kf_every=int(args.kf),
            seed=seed,
        )

    total = len(ds)
    if int(args.max_scenes) > 0:
        total = min(total, int(args.max_scenes))

    out: List[Tuple[str, List[str], torch.Tensor]] = []
    for idx in range(total):
        batch = default_collate([ds[idx]])
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected collate output type for {args.dataset}: {type(batch)}")
        scene_id, image_paths, images = _extract_scene_inputs_from_eval_batch(batch)
        out.append((scene_id, image_paths, images))
    return out


def _uniform_sample_after_kf(
    image_paths: List[str],
    images: torch.Tensor,
    max_frames: int,
) -> Tuple[List[str], torch.Tensor]:
    """
    After KF sampling, apply FastVGGT-style frame limiting:
    keep first frame, then approximately uniform step on the rest.
    """
    if max_frames <= 0:
        return image_paths, images
    n = len(image_paths)
    if n <= max_frames:
        return image_paths, images
    if max_frames == 1:
        keep = [0]
    else:
        first = 0
        remaining = list(range(1, n))
        step = max(1, len(remaining) // (max_frames - 1))
        keep = [first] + remaining[::step][: max_frames - 1]
    sampled_paths = [image_paths[i] for i in keep]
    sampled_images = images[keep, ...]
    return sampled_paths, sampled_images


def _write_eval_json(out_dir: Path, rows: List[Dict[str, Any]], dataset: str, hf_model: str, gt_ply_dir: str | None) -> None:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "hf_model": hf_model,
        "evaluation": f"fastvggt_{dataset}",
        "stage": "cache_dataset",
        "gt_ply_dir": gt_ply_dir,
        "scenes": rows,
    }
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    dataset = str(args.dataset).strip().lower()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = device.type == "cuda"

    if not args.skip_gt_eval and not dependencies_available():
        raise SystemExit("GT eval requires: pip install evo open3d scipy")
    if dataset != "scannet" and not args.skip_gt_eval:
        raise SystemExit(f"Non-ScanNet GT eval is not supported in cache_dataset yet: dataset={dataset}")

    if args.hf_model.strip():
        model = VGGT.from_pretrained(args.hf_model).to(device).eval()
    else:
        model = VGGT().to(device).eval()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    if dataset == "scannet":
        scenes = discover_scannet_scenes(args.data_root)
        if args.max_scenes > 0:
            scenes = scenes[: args.max_scenes]
        if not scenes:
            raise SystemExit(f"No scenes found under {args.data_root}")
        iterable: List[Tuple[str, List[str], torch.Tensor, SceneData]] = []
        for scene in scenes:
            scene_eff = select_scene_image_paths_fastvggt(scene, args.max_frames)
            if not scene_eff.image_paths:
                continue
            images = load_scene_images(scene_eff, mode="crop")
            image_paths = [str(p) for p in scene_eff.image_paths]
            iterable.append((scene.scene_id, image_paths, images, scene_eff))
    else:
        iterable = []
        for scene_id, image_paths, images in _iter_7n_scene_inputs(args):
            image_paths, images = _uniform_sample_after_kf(image_paths, images, int(args.max_frames))
            scene_eff = SceneData(
                scene_id=scene_id,
                scene_dir=Path(os.path.commonpath(image_paths)) if image_paths else args.data_root,
                image_paths=[Path(p) for p in image_paths],
            )
            iterable.append((scene_id, image_paths, images, scene_eff))

    for si, (scene_id, image_paths, images_raw, scene_eff) in enumerate(iterable):
        timing_parts: Dict[str, float] | None = {} if args.timing else None

        t_h2d = time.perf_counter()
        images = images_raw.to(device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        _, num_frames = infer_bs_and_seq_len(images)
        to_device_ms = (time.perf_counter() - t_h2d) * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            batch = collect_batch(
                model,
                images,
                keep_activations_on_gpu=False,
                timing_ms=timing_parts,
            )
        wall_collect_ms = (time.perf_counter() - t0) * 1000.0

        if timing_parts is not None and timing_parts:
            infer_ms = (
                float(timing_parts["patch_embed_cache_ms"])
                + float(timing_parts["aggregator_ms"])
                + float(timing_parts["heads_ms"])
            )
        else:
            infer_ms = wall_collect_ms

        if args.timing and timing_parts:
            print(
                f"[cache_dataset] {dataset} scene {si + 1}/{len(iterable)}: {scene_id} | "
                f"frames={num_frames} | "
                f"infer={infer_ms:.1f}ms (to_device={to_device_ms:.1f}ms "
                f"patch_embed_cache={timing_parts['patch_embed_cache_ms']:.1f}ms "
                f"aggregator={timing_parts['aggregator_ms']:.1f}ms "
                f"heads={timing_parts['heads_ms']:.1f}ms; wall_collect={wall_collect_ms:.1f}ms)"
            )
        else:
            print(
                f"[cache_dataset] {dataset} scene {si + 1}/{len(iterable)}: {scene_id} | frames={num_frames}"
            )

        safe_id = scene_id.replace("/", "_")
        scene_dir = out_dir / safe_id
        layers_dir = scene_dir / "layers"
        metadata_path = scene_dir / "metadata.json"
        pred_path = scene_dir / "pred.pt"
        # Match --dtype (e.g. run_cache.sh DTYPE): on-disk replay layers use this dtype; pred stays float32.
        layer_storage_dtype = amp_dtype

        payload = {
            "scene_id": scene_id,
            "dataset": dataset,
            "num_frames": num_frames,
            "image_hw": [int(images.shape[-2]), int(images.shape[-1])],
            "patch_start_idx": int(batch.patch_start_idx),
            "replay_cache_format": "aggregated_concat_per_layer_v2",
            "scene_scores": batch.scene_scores,
            "initial_tokens": None if batch.initial_tokens is None else batch.initial_tokens.detach().cpu().contiguous(),
            "aggregated_concat_per_layer": [t.detach().cpu().contiguous() for t in batch.aggregated_concat_tokens],
            "pred": _normalize_scene_pred_for_cache(batch.predictions),
        }
        scene_dir.mkdir(parents=True, exist_ok=True)
        layers_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(payload["initial_tokens"], torch.Tensor):
            torch.save(
                payload["initial_tokens"].to(layer_storage_dtype),
                layers_dir / "layer_-1.pt",
            )
        for layer_idx, layer_tensor in enumerate(payload["aggregated_concat_per_layer"]):
            torch.save(layer_tensor.to(layer_storage_dtype), layers_dir / f"layer_{layer_idx:02d}.pt")

        metadata = {
            "scene_id": payload["scene_id"],
            "dataset": dataset,
            "num_frames": payload["num_frames"],
            "image_hw": payload["image_hw"],
            "patch_start_idx": payload["patch_start_idx"],
            "replay_cache_format": payload["replay_cache_format"],
            "replay_layer_storage_dtype": str(args.dtype),
            "has_initial_tokens": isinstance(payload["initial_tokens"], torch.Tensor),
            "scene_scores": payload["scene_scores"],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(payload["pred"], pred_path)
        pred_payload = build_scene_prediction_payload(batch.predictions, (int(images.shape[-2]), int(images.shape[-1])))

        frame_ids = [Path(p).stem for p in image_paths]
        manifest.append(
            {
                "dataset": dataset,
                "scene_id": scene_id,
                "dir": safe_id,
                "num_frames": payload["num_frames"],
                "frame_ids": frame_ids,
                "image_paths": image_paths,
            }
        )

        if dataset == "scannet" and not args.skip_gt_eval:
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
        "dataset": dataset,
        "scenes": manifest,
        "hf_model": args.hf_model,
        "replay_cache_format": "aggregated_concat_per_layer_v2",
        "replay_layer_storage_dtype": str(args.dtype),
    }
    if eval_rows:
        manifest_payload["eval_file"] = "eval.json"
    (out_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    if eval_rows:
        _write_eval_json(
            out_dir,
            eval_rows,
            dataset=dataset,
            hf_model=args.hf_model,
            gt_ply_dir=str(args.gt_ply_dir.resolve()) if isinstance(args.gt_ply_dir, Path) else None,
        )
        print(f"Wrote eval.json ({len(eval_rows)} scenes) under {out_dir}")
    print(f"Cached {len(manifest)} scenes for dataset={dataset} under {out_dir}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
import json

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from analysis.train.model import build_model, load_interval_rules
from analysis.eval.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
)
from analysis.eval import compute_average_metrics_and_save, evaluate_scene_and_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default="/data/sy/scannetv2/process_scannet"
    )
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument("--model-name", type=str, default="Baseline", choices=("Baseline", "Ours"))
    parser.add_argument("--backend", type=str, default="vggt", choices=("vggt", "fastvggt"))
    parser.add_argument(
        "--interval-config-file",
        type=Path,
        default=None,
        help="Required for Ours(lowerbound) when no checkpoint is provided.",
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Training output dir for Ours checkpoint resolve.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch index for Ours checkpoint resolve.")
    parser.add_argument("--merging", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=1.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation, distances exceeding this value will be clipped",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=1000,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=0,
        help="If >0, evaluate at most N scenes. If 0, evaluate all scenes under --data_dir.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="",
        help="Optional HuggingFace model id. If set, load via from_pretrained.",
    )
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.9,
        help="Token merge ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--skip_depth_eval",
        action="store_true",
        help="Disable dense depth vs ScanNet GT (depth_abs_rel, depth_delta1); same protocol as "
        "redundancy cache_scannet Stage1 when left unset.",
    )
    args = parser.parse_args()
    torch.manual_seed(33)

    # Scene sampling
    if int(args.num_scenes) > 0:
        scannet_scenes = get_all_scenes(args.data_dir, int(args.num_scenes))
        print(f"Evaluate {len(scannet_scenes)} scenes (sampled, --num_scenes={int(args.num_scenes)})")
    else:
        scannet_scenes = sorted([d.name for d in Path(args.data_dir).iterdir() if d.is_dir()])
        print(f"Evaluate {len(scannet_scenes)} scenes (all under {args.data_dir})")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    # Use fp16 by default for compatibility with baseline VGGT implementations.
    dtype = torch.float16
    interval_rules = None
    if str(args.model_name) == "Ours" and (not args.ckpt_path) and (args.train_dir is None):
        if args.interval_config_file is None:
            raise SystemExit("--interval-config-file is required for Ours without checkpoint (lowerbound).")
        interval_rules = load_interval_rules(args.interval_config_file)

    ckpt_path = args.ckpt_path if str(args.ckpt_path).strip() else None
    model = build_model(
        model_name=str(args.model_name),  # type: ignore[arg-type]
        backend=str(args.backend),  # type: ignore[arg-type]
        hf_model=str(args.hf_model),
        device="cpu",
        interval_rules=interval_rules,
        ckpt_path=ckpt_path,
        train_dir=args.train_dir,
        epoch=args.epoch,
        eval_mode=True,
    )
    model = model.cuda().eval()

    # Process each scene
    for scene in scannet_scenes:
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / scene

        # Load scene data
        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)

        # Frame filtering
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )

        # Get corresponding poses
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths

        if len(image_paths) == 0:
            print(f"No images found in {images_dir}")
            continue

        print("🚩Processing", scene, f"Found {len(image_paths)} images")
        all_cam_to_world_mat = []
        all_world_points = []

        try:
            # Load images
            images = load_images_rgb(image_paths)

            if not images or len(images) < 3:
                print(f"Skipping {scene}: insufficient valid images")
                continue

            frame_ids = selected_frame_ids
            images_array = np.stack(images)
            vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
            print(f"Patch dimensions: {patch_width}x{patch_height}")

            # Update model attention layers with dynamic patch dimensions
            if hasattr(model, "update_patch_dimensions"):
                model.update_patch_dimensions(patch_width, patch_height)

            # Inference + Reconstruction
            (
                extrinsic_np,
                intrinsic_np,
                all_world_points,
                all_point_colors,
                all_cam_to_world_mat,
                inference_time_ms,
                depth_raw_np,
            ) = infer_vggt_and_reconstruct(
                model, vgg_input, dtype, args.depth_conf_thresh, image_paths
            )
            print(f"Inference time: {inference_time_ms:.2f}ms")

            # Process results
            if not all_cam_to_world_mat or not all_world_points:
                print(
                    f"Skipping {scene}: failed to obtain valid camera poses or point clouds"
                )
                continue

            # Evaluate and save
            gt_depth = None
            pred_depth_for_metrics = None
            if not args.skip_depth_eval:
                from analysis.eval.prediction_eval import (
                    build_fastvggt_scene_eval_context,
                    _depth_or_conf_to_shw,
                )
                from analysis.eval.scannet_eval import SceneData

                try:
                    pred_depth_for_metrics = _depth_or_conf_to_shw(
                        depth_raw_np, "depth"
                    )
                except Exception as e:
                    print(f"[warn] pred depth layout normalize failed for {scene}: {e}")
                    pred_depth_for_metrics = np.asarray(
                        depth_raw_np, dtype=np.float32
                    )
                try:
                    scene_data = SceneData(
                        scene_id=scene,
                        scene_dir=scene_dir,
                        image_paths=list(image_paths),
                    )
                    # Pixel (H, W) of vgg_input — same contract as cache_scannet / train_lora_skip_baseline.
                    # Do not pass patch grid counts (patch_height, patch_width); that breaks GT depth
                    # preprocessing and pred/gt shape alignment, so depth metrics are silently skipped.
                    ctx = build_fastvggt_scene_eval_context(
                        scene_data,
                        args.data_dir,
                        args.data_dir,
                        image_hw=(int(vgg_input.shape[2]), int(vgg_input.shape[3])),
                    )
                    gt_depth = ctx.gt_depth
                except Exception as e:
                    print(
                        f"[warn] depth GT load skipped for {scene} (align with Stage1 when data is complete): {e}"
                    )
            metrics = evaluate_scene_and_save(
                scene,
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                args.data_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
                pred_depth=pred_depth_for_metrics,
                gt_depth=gt_depth,
            )
            if metrics is not None:
                summary_keys = [
                    "chamfer_distance",
                    "ate",
                    "are",
                    "rpe_rot",
                    "rpe_trans",
                    "inference_time_ms",
                    "depth_abs_rel",
                    "depth_delta1",
                ]
                all_scenes_metrics["scenes"][scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key in summary_keys
                }
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback

            traceback.print_exc()

    # Summarize average metrics and save
    average_metrics = compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        input_frame=None,  # external dir already encodes input_frame; save directly under output_path
    )
    average_metrics["num_scenes"] = int(len(all_scenes_metrics.get("scenes") or {}))
    with open(args.output_path / "average_metrics.json", "w") as f:
        json.dump(average_metrics, f, indent=4)
    print("\nAverage metrics:")
    for metric_name, value in average_metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{metric_name}: {int(value)}")
        else:
            print(f"{metric_name}: {float(value):.6f}")

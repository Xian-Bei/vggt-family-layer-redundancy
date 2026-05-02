"""7Scenes / NRGBD evaluation entrypoint.

Per-scene forward + ICP + accuracy/completion + average_metrics.json aggregation
all live in :mod:`analysis.eval.eval_7andn_utils` so the same code path is shared
with the training-time periodic eval (`train_lora_skip_baseline.py`). This file
just parses CLI args, builds the model and the SevenScenes/NRGBD datasets, and
delegates each dataset to ``evaluate_7andn_dataset``.
"""

import os
import os.path as osp
import sys

# Mirror prior behavior: ensure project root is importable for `vggt.*`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import random

import numpy as np
import torch

from analysis.eval.eval_7andn_utils import evaluate_7andn_dataset
from analysis.train.model import build_model, load_interval_rules


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument("--ckpt_path", type=str, default="", help="ckpt name")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model-name", type=str, default="Baseline", choices=("Baseline", "Ours"))
    parser.add_argument("--backend", type=str, default="vggt", choices=("vggt", "fastvggt"))
    parser.add_argument(
        "--interval-config-file",
        type=str,
        default="",
        help="Required for Ours(lowerbound) when no checkpoint is provided.",
    )
    parser.add_argument("--train-dir", type=str, default="", help="Training output dir for Ours checkpoint resolve.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch index for Ours checkpoint resolve.")
    parser.add_argument(
        "--conf_thresh", type=float, default=1.0, help="confidence threshold (evaluation uses >=1.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sy/code/FastVGGT/eval_results",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--merging", type=int, default=0, help="VGGT aggregator merging steps")
    parser.add_argument("--merge_ratio", type=float, default=0.9, help="Token merge ratio (0.0-1.0)")
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    parser.add_argument("--root_7scenes", type=str, default="/data/sy/7scenes", help="7Scenes dataset root")
    parser.add_argument("--root_nrgbd", type=str, default="/data/sy/neural_rgbd_data", help="NRGBD dataset root")
    parser.add_argument(
        "--enable-7scenes",
        type=int,
        default=1,
        choices=(0, 1),
        help="Enable 7Scenes evaluation (1=on, 0=off).",
    )
    parser.add_argument(
        "--enable-nrgbd",
        type=int,
        default=1,
        choices=(0, 1),
        help="Enable NRGBD evaluation (1=on, 0=off).",
    )
    parser.add_argument(
        "--max_scenes_per_dataset",
        type=int,
        default=0,
        help="If >0, evaluate only first N scenes per dataset (for quick validation).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, evaluate only first N sampled items per dataset.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="",
        help="Optional HuggingFace model id. If set, load via from_pretrained.",
    )
    parser.add_argument(
        "--export-visualization",
        action="store_true",
        help="Export per-scene visualization assets under <output_dir>/<kf>/<dataset>/<scene_id>/vis/.",
    )
    parser.add_argument(
        "--vis-max-points",
        type=int,
        default=200000,
        help="Max points saved into viser_points(.npz) per scene (0 = keep all).",
    )
    return parser


def _resolve_resolution(size: int):
    if size == 512:
        return (512, 384)
    if size == 224:
        return 224
    if size == 518:
        return (518, 392)
    raise NotImplementedError(f"Unsupported --size={size}")


def main(args: argparse.Namespace) -> None:
    SEED = 33
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rng_np = np.random.default_rng(SEED)

    resolution = _resolve_resolution(int(args.size))

    # data.py imports `from base import ...` and `import dataset_utils.cropping`,
    # both of which require `analysis/eval/` on sys.path. eval_7andn_utils.py
    # already injects that path on import; importing it ensures we're aligned.
    from analysis.eval.eval_7andn_utils import evaluate_7andn_dataset  # noqa: F401
    from data import NRGBD, SevenScenes  # type: ignore

    datasets_all = {}
    if int(args.enable_7scenes) == 1:
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT=args.root_7scenes,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
            seed=SEED,
        )
    if int(args.enable_nrgbd) == 1:
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT=args.root_nrgbd,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
            seed=SEED,
        )
    if not datasets_all:
        raise SystemExit("Both --enable-7scenes and --enable-nrgbd are 0; nothing to evaluate.")

    eval_conf_thresh = max(1.0, float(args.conf_thresh))

    interval_rules = None
    if str(args.model_name) == "Ours" and (not str(args.ckpt_path).strip()) and (not str(args.train_dir).strip()):
        if not str(args.interval_config_file).strip():
            raise SystemExit("--interval-config-file is required for Ours without checkpoint (lowerbound).")
        interval_rules = load_interval_rules(str(args.interval_config_file))

    ckpt_path = str(args.ckpt_path).strip() or None
    train_dir = str(args.train_dir).strip() or None
    model = build_model(
        model_name=str(args.model_name),  # type: ignore[arg-type]
        backend=str(args.backend),  # type: ignore[arg-type]
        hf_model=str(args.hf_model),
        device="cpu",
        interval_rules=interval_rules,
        ckpt_path=ckpt_path,
        train_dir=train_dir,
        epoch=args.epoch,
        eval_mode=True,
    )
    model = model.cuda().eval()

    os.makedirs(osp.join(args.output_dir, f"{args.kf}"), exist_ok=True)

    for name_data, dataset in datasets_all.items():
        if args.max_scenes_per_dataset > 0 and hasattr(dataset, "scene_list"):
            dataset.scene_list = dataset.scene_list[: args.max_scenes_per_dataset]
        save_path = osp.join(osp.join(args.output_dir, f"{args.kf}"), name_data)
        evaluate_7andn_dataset(
            model=model,
            dataset=dataset,
            device=args.device,
            eval_amp_dtype=torch.bfloat16,
            save_path=save_path,
            eval_conf_thresh=eval_conf_thresh,
            max_samples=int(args.max_samples),
            revisit=int(args.revisit),
            use_proj=bool(args.use_proj),
            export_visualization=bool(args.export_visualization),
            vis_max_points=int(args.vis_max_points),
            rng_np=rng_np,
            write_average_metrics_json=True,
        )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

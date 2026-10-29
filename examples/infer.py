"""Run any calibrated backbone on a directory of RGB images."""

import argparse
from pathlib import Path

import torch

from models import load_calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("vggt", "pi3", "vggt_omega"), default="vggt")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-model", help="Original HF model ID, or local Omega base checkpoint")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.pt"))
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    model = load_calibrated(args.backbone, checkpoint=args.checkpoint, base_model=args.base_model)
    predictions = model(model.load_images(paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in predictions.items()}, args.output)
    print({key: tuple(value.shape) for key, value in predictions.items()})


if __name__ == "__main__":
    main()

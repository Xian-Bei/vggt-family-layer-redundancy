import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from vggt.models.vggt import VGGT


@dataclass
class TensorShapeRecord:
    name: str
    shape: List[int]
    dtype: str


def _shape_record(name: str, tensor: torch.Tensor) -> TensorShapeRecord:
    return TensorShapeRecord(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))


def build_architecture_map(model: VGGT, batch_size: int, seq_len: int, image_size: int) -> Dict[str, Any]:
    device = torch.device("cpu")
    model = model.to(device).eval()

    images = torch.rand(batch_size, seq_len, 3, image_size, image_size, device=device)
    with torch.no_grad():
        agg_tokens, patch_start_idx = model.aggregator(images)

    last_tokens = agg_tokens[-1]
    records: List[TensorShapeRecord] = []
    records.append(_shape_record("input_images", images))
    records.append(_shape_record("aggregator_tokens_last", last_tokens))

    outputs: Dict[str, Any] = {
        "input": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "image_size": image_size,
        },
        "aggregator": {
            "depth": model.aggregator.depth,
            "aa_order": model.aggregator.aa_order,
            "patch_start_idx": patch_start_idx,
            "num_intermediate_outputs": len(agg_tokens),
            "last_token_shape": list(last_tokens.shape),
        },
        "branches": {},
        "tensor_shapes": [asdict(r) for r in records],
    }

    if model.camera_head is not None:
        with torch.no_grad():
            pose_enc_list = model.camera_head(agg_tokens)
        outputs["branches"]["camera"] = {
            "enabled": True,
            "num_refinement_steps": len(pose_enc_list),
            "last_pose_shape": list(pose_enc_list[-1].shape),
        }
    else:
        outputs["branches"]["camera"] = {"enabled": False}

    if model.depth_head is not None:
        with torch.no_grad():
            depth, depth_conf = model.depth_head(agg_tokens, images=images, patch_start_idx=patch_start_idx)
        outputs["branches"]["depth"] = {
            "enabled": True,
            "depth_shape": list(depth.shape),
            "depth_conf_shape": list(depth_conf.shape),
        }
    else:
        outputs["branches"]["depth"] = {"enabled": False}

    if model.point_head is not None:
        with torch.no_grad():
            points, points_conf = model.point_head(agg_tokens, images=images, patch_start_idx=patch_start_idx)
        outputs["branches"]["point"] = {
            "enabled": True,
            "point_shape": list(points.shape),
            "point_conf_shape": list(points_conf.shape),
        }
    else:
        outputs["branches"]["point"] = {"enabled": False}

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export VGGT architecture and tensor path map.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/architecture_map.json"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model = VGGT(
        img_size=args.image_size,
        patch_size=14,
        embed_dim=args.embed_dim,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
    )
    result = build_architecture_map(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved architecture map to {args.output}")


if __name__ == "__main__":
    main()


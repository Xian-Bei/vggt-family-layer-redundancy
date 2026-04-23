"""
Visualize bidirectional pair attention from Stage1 caches.

This script reuses ``aggregated_concat_per_layer`` saved by Stage1 to reconstruct
the specified global block's attention queries/keys without rerunning the full
VGGT forward. Given a scene, layer, and two selected-frame indices ``(i, j)``,
it produces a panel for both directions:

- ``i -> j``: frame ``i`` patch queries against frame ``j`` patch keys
- ``j -> i``: frame ``j`` patch queries against frame ``i`` patch keys

The attention score is the mean-over-heads raw ``q @ k^T`` (before softmax).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT

from .scannet_eval import SceneData, discover_scannet_scenes, load_scene_images


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize bidirectional pair attention from Stage1 cache.")
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/scannet_cache"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--scene-id", type=str, default="scene0000_00")
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--frame-a-index", type=int, default=0)
    p.add_argument("--frame-b-index", type=int, default=1)
    p.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="Optional HF model override. Default: read hf_model from manifest.json.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pair_attention"))
    return p


def _read_manifest_entry(cache_dir: Path, scene_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run Stage1 first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes", [])
    for item in scenes:
        if str(item.get("scene_id")) == scene_id:
            return manifest, item
    raise SystemExit(f"Scene {scene_id!r} not found in {manifest_path}.")


def _resolve_scene_from_manifest(data_root: Path, scene_id: str, frame_ids: List[int]) -> SceneData:
    discovered = discover_scannet_scenes(data_root)
    scene_map = {s.scene_id: s for s in discovered}
    if scene_id not in scene_map:
        raise SystemExit(f"Scene {scene_id!r} not found under {data_root}.")
    scene = scene_map[scene_id]
    frame_to_path = {int(p.stem): p for p in scene.image_paths}
    selected_paths: List[Path] = []
    missing: List[int] = []
    for fid in frame_ids:
        path = frame_to_path.get(int(fid))
        if path is None:
            missing.append(int(fid))
        else:
            selected_paths.append(path)
    if missing:
        raise SystemExit(f"Scene {scene_id!r} is missing frame files for ids: {missing}")
    return SceneData(scene_id=scene.scene_id, scene_dir=scene.scene_dir, image_paths=selected_paths)


def _load_cache_row(cache_dir: Path, cache_file: str) -> Dict[str, Any]:
    path = cache_dir / cache_file
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_global_positions(model: VGGT, *, num_frames: int, image_hw: Tuple[int, int], device: torch.device) -> torch.Tensor | None:
    agg = model.aggregator
    if agg.rope is None:
        return None
    height, width = image_hw
    pos_frame = agg.position_getter(
        num_frames,
        height // agg.patch_size,
        width // agg.patch_size,
        device=device,
    )
    if agg.patch_start_idx > 0:
        pos_frame = pos_frame + 1
        pos_special = torch.zeros(
            num_frames,
            agg.patch_start_idx,
            2,
            device=device,
            dtype=pos_frame.dtype,
        )
        pos_frame = torch.cat([pos_special, pos_frame], dim=1)
    _, p, _ = pos_frame.shape
    return pos_frame.view(1, num_frames, p, 2).view(1, num_frames * p, 2)


def _reconstruct_qk(
    model: VGGT,
    frame_layer: torch.Tensor,
    *,
    layer_idx: int,
    pos_global: torch.Tensor | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block = model.aggregator.global_blocks[layer_idx]
    frame_layer = frame_layer.to(device=device, dtype=torch.float32)
    bsz, num_frames, tokens_per_frame, dim = frame_layer.shape
    x = frame_layer.view(bsz, num_frames * tokens_per_frame, dim)
    x = block.norm1(x)
    attn = block.attn
    qkv = attn.qkv(x).reshape(bsz, x.shape[1], 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos_global is not None:
        q = attn.rope(q, pos_global)
        k = attn.rope(k, pos_global)
    return q, k


def _frame_patch_slice(frame_index: int, *, tokens_per_frame: int, patch_start_idx: int) -> slice:
    start = frame_index * tokens_per_frame + patch_start_idx
    stop = (frame_index + 1) * tokens_per_frame
    return slice(start, stop)


def _compute_directional_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frame_q_idx: int,
    frame_k_idx: int,
    tokens_per_frame: int,
    patch_start_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_slice = _frame_patch_slice(frame_q_idx, tokens_per_frame=tokens_per_frame, patch_start_idx=patch_start_idx)
    k_slice = _frame_patch_slice(frame_k_idx, tokens_per_frame=tokens_per_frame, patch_start_idx=patch_start_idx)

    q_sel = q[0, :, q_slice, :]
    k_sel = k[0, :, k_slice, :]
    score = torch.matmul(q_sel, k_sel.transpose(-2, -1)).mean(dim=0)
    src_map = score.max(dim=1).values
    tgt_map = score.max(dim=0).values
    return (
        score.detach().cpu().float().numpy(),
        src_map.detach().cpu().float().numpy(),
        tgt_map.detach().cpu().float().numpy(),
    )


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def _overlay_patch_map(image_tensor: torch.Tensor, patch_scores: np.ndarray, *, patch_hw: Tuple[int, int]) -> np.ndarray:
    img = image_tensor.detach().cpu().permute(1, 2, 0).float().numpy()
    img = np.clip(img, 0.0, 1.0)
    patch_map = patch_scores.reshape(patch_hw[0], patch_hw[1])
    patch_map = _normalize_map(patch_map)
    patch_map_t = torch.from_numpy(patch_map).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(
        patch_map_t,
        size=(img.shape[0], img.shape[1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    color = plt.get_cmap("jet")(up)[..., :3]
    return np.clip(0.5 * img + 0.5 * color, 0.0, 1.0)


def _render_panel(
    *,
    scene_id: str,
    layer_idx: int,
    frame_a_index: int,
    frame_b_index: int,
    frame_a_id: int,
    frame_b_id: int,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    score_ab: np.ndarray,
    src_ab: np.ndarray,
    tgt_ab: np.ndarray,
    score_ba: np.ndarray,
    src_ba: np.ndarray,
    tgt_ba: np.ndarray,
    patch_hw: Tuple[int, int],
    out_path: Path,
) -> None:
    overlay_a_src = _overlay_patch_map(image_a, src_ab, patch_hw=patch_hw)
    overlay_b_tgt = _overlay_patch_map(image_b, tgt_ab, patch_hw=patch_hw)
    overlay_b_src = _overlay_patch_map(image_b, src_ba, patch_hw=patch_hw)
    overlay_a_tgt = _overlay_patch_map(image_a, tgt_ba, patch_hw=patch_hw)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"{scene_id} | layer {layer_idx} | raw mean-head q@k before softmax | "
        f"A=index {frame_a_index} (id={frame_a_id}) | B=index {frame_b_index} (id={frame_b_id})"
    )

    axes[0, 0].imshow(overlay_a_src)
    axes[0, 0].set_title("A as q: max over B patches")
    axes[0, 1].imshow(overlay_b_tgt)
    axes[0, 1].set_title("B as k: max over A queries")
    im0 = axes[0, 2].imshow(score_ab, cmap="viridis")
    axes[0, 2].set_title("A -> B patch scores")
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(overlay_b_src)
    axes[1, 0].set_title("B as q: max over A patches")
    axes[1, 1].imshow(overlay_a_tgt)
    axes[1, 1].set_title("A as k: max over B queries")
    im1 = axes[1, 2].imshow(score_ba, cmap="viridis")
    axes[1, 2].set_title("B -> A patch scores")
    fig.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    manifest, item = _read_manifest_entry(args.cache_dir, args.scene_id)
    hf_model = args.hf_model.strip() or str(manifest.get("hf_model", "")).strip()
    if not hf_model:
        raise SystemExit("No HF model provided and manifest.json does not contain 'hf_model'.")

    frame_ids = [int(x) for x in item.get("frame_ids", [])]
    if not frame_ids:
        raise SystemExit(f"Manifest entry for {args.scene_id!r} lacks frame_ids.")
    if not (0 <= args.frame_a_index < len(frame_ids)) or not (0 <= args.frame_b_index < len(frame_ids)):
        raise SystemExit(
            f"Frame indices out of range for scene {args.scene_id!r}: "
            f"num_frames={len(frame_ids)}, got a={args.frame_a_index}, b={args.frame_b_index}"
        )

    row = _load_cache_row(args.cache_dir, str(item["file"]))
    replay_layers = row.get("aggregated_concat_per_layer")
    if not isinstance(replay_layers, list) or not replay_layers:
        raise SystemExit("Cache row lacks aggregated_concat_per_layer. Please rerun Stage1.")

    layer_idx = int(args.layer_idx)
    if not (0 <= layer_idx < len(replay_layers)):
        raise SystemExit(f"Layer index {layer_idx} out of range; cache has {len(replay_layers)} layers.")

    patch_start_idx = int(row["patch_start_idx"])
    image_hw = tuple(int(x) for x in row["image_hw"])
    scene_eff = _resolve_scene_from_manifest(args.data_root, args.scene_id, frame_ids)
    images = load_scene_images(scene_eff, mode="crop")

    frame_a_id = frame_ids[args.frame_a_index]
    frame_b_id = frame_ids[args.frame_b_index]

    model = VGGT.from_pretrained(hf_model).to(device).eval()
    pos_global = _build_global_positions(model, num_frames=len(frame_ids), image_hw=image_hw, device=device)

    concat = replay_layers[layer_idx]
    if not isinstance(concat, torch.Tensor) or concat.dim() != 4:
        raise SystemExit(f"Layer {layer_idx} replay cache is invalid.")
    two_c = int(concat.shape[-1])
    if two_c % 2 != 0:
        raise SystemExit(f"Layer {layer_idx} replay cache has odd channel dimension: {two_c}")
    frame_layer = concat[..., : two_c // 2]

    tokens_per_frame = int(frame_layer.shape[2])
    patch_tokens = tokens_per_frame - patch_start_idx
    patch_h = image_hw[0] // model.aggregator.patch_size
    patch_w = image_hw[1] // model.aggregator.patch_size
    if patch_h * patch_w != patch_tokens:
        raise SystemExit(
            f"Patch grid mismatch: image_hw={image_hw}, patch_size={model.aggregator.patch_size}, "
            f"derived grid={patch_h}x{patch_w}, cache patch_tokens={patch_tokens}"
        )

    q, k = _reconstruct_qk(
        model,
        frame_layer,
        layer_idx=layer_idx,
        pos_global=pos_global,
        device=device,
    )
    score_ab, src_ab, tgt_ab = _compute_directional_maps(
        q,
        k,
        frame_q_idx=args.frame_a_index,
        frame_k_idx=args.frame_b_index,
        tokens_per_frame=tokens_per_frame,
        patch_start_idx=patch_start_idx,
    )
    score_ba, src_ba, tgt_ba = _compute_directional_maps(
        q,
        k,
        frame_q_idx=args.frame_b_index,
        frame_k_idx=args.frame_a_index,
        tokens_per_frame=tokens_per_frame,
        patch_start_idx=patch_start_idx,
    )

    panel_root = args.output_dir / args.scene_id
    stem = f"layer_{layer_idx:02d}_frame_{args.frame_a_index:02d}_to_{args.frame_b_index:02d}"
    panel_path = panel_root / f"{stem}_panel.png"
    npz_path = panel_root / f"{stem}_scores.npz"

    _render_panel(
        scene_id=args.scene_id,
        layer_idx=layer_idx,
        frame_a_index=args.frame_a_index,
        frame_b_index=args.frame_b_index,
        frame_a_id=frame_a_id,
        frame_b_id=frame_b_id,
        image_a=images[args.frame_a_index],
        image_b=images[args.frame_b_index],
        score_ab=score_ab,
        src_ab=src_ab,
        tgt_ab=tgt_ab,
        score_ba=score_ba,
        src_ba=src_ba,
        tgt_ba=tgt_ba,
        patch_hw=(patch_h, patch_w),
        out_path=panel_path,
    )

    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        score_ab=score_ab,
        score_ba=score_ba,
        src_ab=src_ab,
        tgt_ab=tgt_ab,
        src_ba=src_ba,
        tgt_ba=tgt_ba,
        frame_a_index=np.asarray(args.frame_a_index),
        frame_b_index=np.asarray(args.frame_b_index),
        frame_a_id=np.asarray(frame_a_id),
        frame_b_id=np.asarray(frame_b_id),
        layer_idx=np.asarray(layer_idx),
        image_hw=np.asarray(image_hw),
        patch_hw=np.asarray([patch_h, patch_w]),
        patch_start_idx=np.asarray(patch_start_idx),
    )

    print(f"Wrote panel to {panel_path}")
    print(f"Wrote raw scores to {npz_path}")


if __name__ == "__main__":
    main()

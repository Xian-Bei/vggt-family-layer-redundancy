"""Convert dataset ``EvalSampleBatch.images`` to model input tensors for cache/train."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch.utils.data._utils.collate import default_collate

from analysis.datasets.frame_selection import pick_frame_indices

if TYPE_CHECKING:
    from analysis.backbones.contracts import BackboneAdapter


@dataclass
class PreparedCacheBatch:
    """Model input tensor plus per-frame ids/paths (already selected by dataset adaptors)."""

    images: torch.Tensor
    frame_ids: List[str]
    image_paths: List[str]


def _frame_id_from_view(view: Dict[str, Any]) -> str:
    label = str(view.get("label", "")).strip()
    if "/" in label:
        return label.rsplit("/", 1)[-1]
    if label:
        return label
    inst = str(view.get("instance", "")).strip()
    if inst:
        return Path(inst).stem
    return "0"


def _image_path_from_view(view: Dict[str, Any]) -> str:
    inst = view.get("instance")
    if inst:
        return str(Path(str(inst)).resolve())
    return ""


def prepare_cache_batch(
    images: Any,
    *,
    max_frames: int = 0,
    meta: Optional[Dict[str, Any]] = None,
    backbone_adapter: Optional["BackboneAdapter"] = None,
) -> PreparedCacheBatch:
    """Convert dataset batch to model tensor; frame count is set in dataset adaptors (``max_frames=0`` = no re-cap)."""
    meta = dict(meta or {})

    if images is None:
        paths_raw = meta.get("image_paths") or []
        image_paths = [str(Path(str(p)).resolve()) for p in paths_raw]
        if not image_paths:
            raise ValueError("batch.images is None and meta.image_paths is empty")
        if backbone_adapter is None:
            raise ValueError("batch.images is None; pass backbone_adapter to prepare_cache_batch")
        tensor = backbone_adapter.load_eval_images(image_paths)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 5:
            raise ValueError(f"Backbone load_eval_images expected [S,3,H,W] or [1,S,3,H,W], got {tuple(tensor.shape)}")
        fids_raw = meta.get("frame_ids") or []
        frame_ids = [str(x) for x in fids_raw] if fids_raw else [Path(p).stem for p in image_paths]
        if len(frame_ids) != len(image_paths):
            frame_ids = [Path(p).stem for p in image_paths]
        return PreparedCacheBatch(tensor, frame_ids, image_paths)

    if isinstance(images, list) and images and isinstance(images[0], dict):
        views: List[Any] = list(images)
        if int(max_frames) > 0 and len(views) > int(max_frames):
            pick = pick_frame_indices(len(views), max_frames)
            views = [views[i] for i in pick]
        frame_ids = [_frame_id_from_view(v) for v in views]
        image_paths = [_image_path_from_view(v) for v in views]
        if backbone_adapter is None:
            raise ValueError("7andn-style batch requires backbone_adapter in prepare_cache_batch")
        tensor = backbone_adapter.load_eval_images(image_paths)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 5:
            raise ValueError(f"Backbone load_eval_images expected [S,3,H,W] or [1,S,3,H,W], got {tuple(tensor.shape)}")
        return PreparedCacheBatch(tensor, frame_ids, image_paths)

    paths_raw = meta.get("image_paths") or []
    fids_raw = meta.get("frame_ids") or []
    image_paths = [str(Path(str(p)).resolve()) for p in paths_raw]
    if fids_raw:
        frame_ids = [str(x) for x in fids_raw]
    else:
        frame_ids = [Path(p).stem for p in image_paths]

    if image_paths and frame_ids and len(image_paths) != len(frame_ids):
        frame_ids = [Path(p).stem for p in image_paths]

    n_frames = len(image_paths) if image_paths else 0
    if n_frames == 0:
        tensor = prepare_cache_images(images, max_frames=max_frames)
        n_frames = int(tensor.shape[1])
        pick = pick_frame_indices(n_frames, max_frames)
        frame_ids = [frame_ids[i] if i < len(frame_ids) else str(i) for i in pick]
        image_paths = [image_paths[i] if i < len(image_paths) else "" for i in pick]
        return PreparedCacheBatch(tensor, frame_ids, image_paths)

    pick = pick_frame_indices(n_frames, max_frames)
    frame_ids = [frame_ids[i] for i in pick]
    image_paths = [image_paths[i] for i in pick]
    tensor = prepare_cache_images(images, max_frames=max_frames)
    if int(tensor.shape[1]) != len(frame_ids):
        raise ValueError(
            f"Cache frame count mismatch: tensor S={int(tensor.shape[1])} vs "
            f"frame_ids={len(frame_ids)} (check dataset meta image_paths)."
        )
    return PreparedCacheBatch(tensor, frame_ids, image_paths)


def prepare_cache_images(images: Any, *, max_frames: int = 0) -> torch.Tensor:
    """Return float tensor ``[1, S, 3, H, W]`` in [0, 1] for VGGT / Pi3 cache forward.

    ScanNet plugins usually provide a tensor or list of tensors. 7scenes / NRGBD provide
    a list of per-view dicts (``img`` in [-1, 1]); we mirror ``eval_impl._collate_7andn_views``.
    """
    if isinstance(images, torch.Tensor):
        out = images.float()
        if out.dim() == 4:
            out = out.unsqueeze(0)
        if out.dim() != 5:
            raise ValueError(f"Expected image tensor [B,S,3,H,W] or [S,3,H,W], got {tuple(out.shape)}")
        return _subsample_frames(out, max_frames=max_frames)

    if isinstance(images, list):
        if not images:
            raise ValueError("Empty image list in EvalSampleBatch")
        if isinstance(images[0], dict):
            views: List[Any] = list(images)
            if int(max_frames) > 0 and len(views) > int(max_frames):
                n = len(views)
                pick = torch.linspace(0, n - 1, int(max_frames)).round().long().tolist()
                views = [views[i] for i in pick]
            collated = default_collate(views)
            if not isinstance(collated, dict) or "img" not in collated:
                raise ValueError("7andn-style batch must collate to dict with key 'img'")
            out = (collated["img"].float() + 1.0) / 2.0
            if out.dim() == 4:
                out = out.unsqueeze(0)
            if out.dim() != 5:
                raise ValueError(f"Collated 7andn img expected 4D/5D, got {tuple(out.shape)}")
            return out
        from torch.utils.data._utils.collate import default_collate as dc

        out = dc(images).float()
        if out.dim() == 4:
            out = out.unsqueeze(0)
        if out.dim() != 5:
            raise ValueError(f"Collated image list expected 4D/5D, got {tuple(out.shape)}")
        return _subsample_frames(out, max_frames=max_frames)

    if isinstance(images, dict) and "img" in images:
        out = (images["img"].float() + 1.0) / 2.0
        if out.dim() == 4:
            out = out.unsqueeze(0)
        if out.dim() != 5:
            raise ValueError(f"Image dict['img'] expected 4D/5D, got {tuple(out.shape)}")
        return _subsample_frames(out, max_frames=max_frames)

    raise TypeError(f"Unsupported images type for cache: {type(images)!r}")


def _subsample_frames(images: torch.Tensor, *, max_frames: int) -> torch.Tensor:
    if int(max_frames) <= 0 or int(images.shape[1]) <= int(max_frames):
        return images
    n = int(images.shape[1])
    pick = pick_frame_indices(n, max_frames)
    return images[:, pick].contiguous()

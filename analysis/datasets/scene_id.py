"""Scene id helpers for 7scenes / NRGBD eval batches (per-view ``label`` paths)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _first_label(label: Any) -> Any:
    """Unwrap batched / collated labels (list, tuple, or 0-d / 1-d tensor)."""
    if isinstance(label, (list, tuple)):
        if not label:
            raise ValueError("empty label sequence")
        label = label[0]
    try:
        import torch

        if isinstance(label, torch.Tensor):
            if label.numel() == 0:
                raise ValueError("empty label tensor")
            label = label.reshape(-1)[0]
            if label.ndim != 0:
                label = label.item()
    except ImportError:
        pass
    return label


def scene_id_from_label(label: Any) -> str:
    """Parse scene id from view label ``scene/seq/frame`` or ``scene/frame``."""
    text = str(_first_label(label))
    return text.rsplit("/", 1)[0]


def scene_id_from_eval_views(views: Sequence[Mapping[str, Any]]) -> str:
    if not views:
        raise ValueError("empty view list")
    return scene_id_from_label(views[0]["label"])

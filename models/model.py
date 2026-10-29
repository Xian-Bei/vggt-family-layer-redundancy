"""Inference model definition shared by the three released backbones."""

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.recovery.train.checkpoint_v3 import (
    apply_checkpoint_structure,
    load_train_checkpoint_payload,
)

BACKBONES = ("vggt", "pi3", "vggt_omega")


def _new_backbone(backbone: str) -> nn.Module:
    """Construct the pinned model definition without downloading base weights."""
    if backbone == "vggt":
        from vggt.models.vggt import VGGT

        model = VGGT()
        model.point_head = None
        return model
    if backbone == "pi3":
        from pi3.models.pi3 import Pi3

        return Pi3()
    from vggt_omega.models import VGGTOmega

    return VGGTOmega()


class LaRe(nn.Module):
    """A pretrained backbone with its calibrated pruning and recovery operators."""

    def __init__(self, model: nn.Module, adapter, metadata: dict):
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.metadata = metadata

    @classmethod
    def from_pretrained(
        cls,
        backbone: str,
        *,
        checkpoint: str | Path | None = None,
        base_model: str | None = None,
        device: str = "cuda",
    ) -> "LaRe":
        """Load base weights, then restore the topology and compact recovery delta.

        ``base_model`` is a Hugging Face model ID for VGGT/Pi3 and a local
        original checkpoint path for VGGT-Omega. No calibration data is needed.
        """
        key = backbone.lower().replace("-", "_")
        if key not in BACKBONES:
            raise ValueError(f"Expected one of {BACKBONES}, got {backbone!r}")
        path = Path(checkpoint) if checkpoint is not None else Path("checkpoints") / key / "recovery.pt"
        if path.is_dir():
            path = path / "recovery.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibrated checkpoint: {path}. See README.md → Calibrated models.")
        payload = load_train_checkpoint_payload(path, map_location="cpu")
        recorded = str(payload.get("backbone") or payload.get("fit_configuration", {}).get("backbone") or payload.get("args", {}).get("backbone", "")).lower().replace("-", "_")
        if recorded != key:
            raise ValueError(f"Checkpoint backbone {recorded!r} does not match {key!r}")
        if not payload.get("prune_plan") or not payload.get("recovery_specs"):
            raise ValueError("Expected a calibrated checkpoint containing prune_plan and recovery_specs")
        apply_pythonpath(key)
        adapter = get_backbone(key)
        if payload.get("variant") == "lare_standalone_v1":
            model = _new_backbone(key)
            apply_checkpoint_structure(model, adapter, payload)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(device)
            return cls(model, adapter, {"backbone": key, "checkpoint": str(path)}).eval()
        if key == "vggt_omega":
            source = base_model or str(Path(__file__).resolve().parents[1] / "backbones/vggt-omega/vggt_omega_1b_512.pt")
        else:
            source = base_model or adapter.default_hf_model()
        # Preserve official fp32 parameters. Mixed precision belongs in forward.
        model = adapter.load_pretrained(hf_model=str(source), device=device, dtype=None)
        apply_checkpoint_structure(model, adapter, payload)
        return cls(model, adapter, {"backbone": key, "checkpoint": str(path)}).eval()

    def load_images(self, image_paths: Sequence[str | Path]) -> torch.Tensor:
        """Return RGB images [S,3,H,W] using the backbone's evaluation transform."""
        if not image_paths:
            raise ValueError("Provide at least one image")
        apply_pythonpath(self.metadata["backbone"])
        return self.adapter.load_eval_images(image_paths)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return available geometry fields, with camera_poses in c2w convention."""
        device = next(self.model.parameters()).device
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected RGB images [S,3,H,W] or [B,S,3,H,W]")
        with torch.autocast(**autocast_device_dtype(device)):
            pred = self.adapter.forward_predictions(self.model, images.to(device))
        fields = ("pose_enc", "depth", "depth_conf", "points", "local_points", "conf", "camera_poses")
        return {name: value for name in fields if (value := getattr(pred, name)) is not None}


def load_calibrated(backbone: str, **kwargs) -> LaRe:
    """Convenience alias for :meth:`LaRe.from_pretrained`."""
    return LaRe.from_pretrained(backbone, **kwargs)

"""GPU integration checks for locally prepared model release artifacts."""

import gc
from pathlib import Path

import pytest
import torch

from lare import LaRe


@pytest.mark.parametrize("backbone", ["vggt", "pi3", "vggt_omega"])
def test_standalone_model_forward_without_base_weight_download(backbone, monkeypatch):
    checkpoint = Path("checkpoints") / backbone / "model.pt"
    if not checkpoint.is_file():
        pytest.skip("Standalone release artifact is not present")
    assert torch.cuda.is_available(), "Run release model tests on a GPU"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = LaRe.from_pretrained(backbone, checkpoint=checkpoint, device="cuda")
    side = 512 if backbone == "vggt_omega" else 518
    images = torch.rand(1, 2, 3, side, side, device="cuda")
    predictions = model(images)
    assert "points" in predictions
    assert "camera_poses" in predictions
    assert predictions["camera_poses"].shape[-2:] == (4, 4)
    assert torch.isfinite(predictions["points"]).all()
    assert torch.isfinite(predictions["camera_poses"]).all()
    assert all(not value.requires_grad for value in predictions.values())
    del predictions, model, images
    gc.collect()
    torch.cuda.empty_cache()


def test_checkpoint_backbone_mismatch_is_rejected():
    checkpoint = Path("checkpoints/pi3/recovery.pt")
    if not checkpoint.is_file():
        pytest.skip("Recovery release artifact is not present")
    with pytest.raises(ValueError, match="does not match"):
        LaRe.from_pretrained("vggt", checkpoint=checkpoint)

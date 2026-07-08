from __future__ import annotations

import torch
import torch.nn as nn

from analysis.recover.compact_recovery_checkpoints import _compact_payload
from analysis.recover.operators import apply_recovery_weights, install_recovery_modules


class _Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.block = _Block()


class _Adapter:
    name = "dummy"

    def block_for_unit(self, model: _Model, unit_id: str) -> nn.Module:
        assert unit_id == "frame_1"
        return model.block


def test_compact_payload_extracts_recovery_weight() -> None:
    weight = torch.eye(4)
    payload = {
        "variant": "training_free_recovery",
        "model_state_dict": {"block.recovery_patch_frame_1.weight": weight},
        "recovery_specs": [
            {
                "source_unit_id": "frame_1",
                "module_name": "recovery_patch_frame_1",
                "weight_shape": [4, 4],
            }
        ],
        "optimizer_state_dict": {"unused": True},
    }

    compact = _compact_payload(payload)

    assert compact["variant"] == "training_free_recovery_delta"
    assert "model_state_dict" not in compact
    assert compact["optimizer_state_dict"] == {}
    assert torch.equal(compact["recovery_weights"][0]["weight"], weight)


def test_apply_recovery_weights_loads_installed_patch() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _Model().to(device)
    adapter = _Adapter()
    spec = {
        "source_unit_id": "frame_1",
        "module_name": "recovery_patch_frame_1",
        "weight_shape": [4, 4],
    }
    weight = torch.eye(4) * 2

    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "frame_1", "module_name": "recovery_patch_frame_1", "weight": weight}],
        device=device,
    )

    patch_weight = model.block.recovery_patch_frame_1.weight.detach().cpu()
    assert torch.equal(patch_weight, weight)

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from analysis.recovery.operators import (
    apply_recovery_weights,
    compact_recovery_operators,
    install_recovery_modules,
)


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


class _ChainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleDict({"global_0": _Block(), "frame_1": _Block()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks["frame_1"](self.blocks["global_0"](x))


class _ChainAdapter:
    name = "dummy"

    def block_for_unit(self, model: _ChainModel, unit_id: str) -> nn.Module:
        return model.blocks[unit_id]


class _OmegaBoundaryModel(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.aggregator = nn.Module()
        self.aggregator.inter_frame_attention_types = [attention_type, "global"]
        self.aggregator.inter_frame_blocks = nn.ModuleList([_Block(), _Block()])
        self.aggregator.frame_blocks = nn.ModuleList([_Block(), _Block()])


class _OmegaBoundaryAdapter:
    name = "vggt_omega"

    def block_for_unit(self, model: _OmegaBoundaryModel, unit_id: str) -> nn.Module:
        kind, index_s = unit_id.split("_", 1)
        blocks = (
            model.aggregator.frame_blocks
            if kind == "frame"
            else model.aggregator.inter_frame_blocks
        )
        return blocks[int(index_s)]


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


def test_adjacent_dropped_unit_recovery_patches_compose_in_model_order() -> None:
    device = torch.device("cuda")
    model = _ChainModel().to(device)
    adapter = _ChainAdapter()
    specs = [
        {
            "type": "recovery_ghost",
            "source_unit_id": "global_0",
            "module_name": "recovery_patch_global_0",
            "weight_shape": [2, 2],
        },
        {
            "type": "recovery_ghost",
            "source_unit_id": "frame_1",
            "module_name": "recovery_patch_frame_1",
            "weight_shape": [2, 2],
        },
    ]
    weights = [
        {"source_unit_id": "global_0", "weight": torch.eye(2) * 2.0},
        {"source_unit_id": "frame_1", "weight": torch.eye(2) * 3.0},
    ]
    install_recovery_modules(model, adapter, specs, device=device)
    apply_recovery_weights(model, adapter, specs, weights, device=device)

    actual = model(torch.ones(1, 2, device=device))
    assert torch.equal(actual, torch.full((1, 2), 6.0, device=device))


def test_adjacent_ghost_weights_compact_to_one_operator() -> None:
    specs = [
        {
            "type": "recovery_ghost",
            "source_unit_id": "global_0",
            "target_unit_id": "frame_1",
            "source_unit_index": 1,
            "target_unit_index": 2,
            "module_name": "recovery_patch_global_0",
            "weight_shape": [2, 2],
        },
        {
            "type": "recovery_ghost",
            "source_unit_id": "frame_1",
            "target_unit_id": "global_1",
            "source_unit_index": 2,
            "target_unit_index": 3,
            "module_name": "recovery_patch_frame_1",
            "weight_shape": [2, 2],
        },
    ]
    first = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    second = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    compact_specs, compact_weights = compact_recovery_operators(
        specs,
        [{"weight": first}, {"weight": second}],
    )
    assert len(compact_specs) == len(compact_weights) == 1
    assert compact_specs[0]["source_unit_id"] == "global_0"
    assert compact_specs[0]["target_unit_id"] == "global_1"
    torch.testing.assert_close(compact_weights[0]["weight"], first @ second)

    linearpatch_specs = [
        {**spec, "type": "recovery_linearpatch"} for spec in specs
    ]
    compact_specs, compact_weights = compact_recovery_operators(
        linearpatch_specs,
        [{"weight": first}, {"weight": second}],
    )
    assert len(compact_specs) == len(compact_weights) == 1
    torch.testing.assert_close(compact_weights[0]["weight"], first @ second)


def test_omega_register_global_recovery_patches_full_boundary_consumer() -> None:
    device = torch.device("cuda")
    model = _OmegaBoundaryModel("register").to(device)
    adapter = _OmegaBoundaryAdapter()
    spec = {
        "type": "recovery_ghost",
        "source_unit_id": "global_0",
        "module_name": "recovery_patch_global_0",
        "weight_shape": [2, 2],
    }
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "global_0", "weight": torch.eye(2) * 2.0}],
        device=device,
    )

    special = model.aggregator.inter_frame_blocks[0](
        torch.ones(1, 2, device=device)
    )
    full_boundary = model.aggregator.frame_blocks[1](
        torch.ones(3, 2, device=device)
    )
    assert torch.equal(special, torch.ones_like(special))
    assert torch.equal(full_boundary, torch.full_like(full_boundary, 2.0))


def test_omega_full_global_recovery_stays_on_source_output() -> None:
    device = torch.device("cuda")
    model = _OmegaBoundaryModel("global").to(device)
    adapter = _OmegaBoundaryAdapter()
    spec = {
        "type": "recovery_ghost",
        "source_unit_id": "global_0",
        "module_name": "recovery_patch_global_0",
        "weight_shape": [2, 2],
    }
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "global_0", "weight": torch.eye(2) * 2.0}],
        device=device,
    )

    source = model.aggregator.inter_frame_blocks[0](
        torch.ones(3, 2, device=device)
    )
    consumer = model.aggregator.frame_blocks[1](
        torch.ones(3, 2, device=device)
    )
    assert torch.equal(source, torch.full_like(source, 2.0))
    assert torch.equal(consumer, torch.ones_like(consumer))


def test_omega_register_boundary_and_adjacent_identity_patch_compose() -> None:
    device = torch.device("cuda")
    model = _OmegaBoundaryModel("register").to(device)
    adapter = _OmegaBoundaryAdapter()
    specs = [
        {
            "type": "recovery_ghost_interval_sequential",
            "source_unit_id": "global_0",
            "module_name": "recovery_patch_global_0",
            "weight_shape": [2, 2],
        },
        {
            "type": "recovery_ghost_interval_sequential",
            "source_unit_id": "frame_1",
            "module_name": "recovery_patch_frame_1",
            "weight_shape": [2, 2],
        },
    ]
    install_recovery_modules(model, adapter, specs, device=device)
    apply_recovery_weights(
        model,
        adapter,
        specs,
        [
            {"source_unit_id": "global_0", "weight": torch.eye(2) * 2.0},
            {"source_unit_id": "frame_1", "weight": torch.eye(2) * 3.0},
        ],
        device=device,
    )

    actual = model.aggregator.frame_blocks[1](
        torch.ones(3, 2, device=device)
    )
    assert torch.equal(actual, torch.full_like(actual, 6.0))


def load_tests(_loader, tests, _pattern):
    tests.addTest(unittest.FunctionTestCase(test_apply_recovery_weights_loads_installed_patch))
    tests.addTest(
        unittest.FunctionTestCase(
            test_adjacent_dropped_unit_recovery_patches_compose_in_model_order
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(test_adjacent_ghost_weights_compact_to_one_operator)
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_omega_register_global_recovery_patches_full_boundary_consumer
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_omega_full_global_recovery_stays_on_source_output
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_omega_register_boundary_and_adjacent_identity_patch_compose
        )
    )
    return tests

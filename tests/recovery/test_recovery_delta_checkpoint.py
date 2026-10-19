from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from analysis.recovery.operators import (
    BoundaryRecovery,
    StreamlineFFN,
    apply_recovery_weights,
    compact_recovery_operators,
    install_recovery_modules,
    pack_streamline_ffn_parameters,
)


class _Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.block = _Block()


class _GroupedModel(_Model):
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        del images
        return self.block(tokens)


class _Adapter:
    name = "dummy"

    def block_for_unit(self, model: _Model, unit_id: str) -> nn.Module:
        assert unit_id in {"frame_1", "global_0"}
        return model.block


class _MapGroupedAdapter(_Adapter):
    name = "mapanything"

    def ghost_runtime_token_group_order(self, token_groups):
        groups = tuple(token_groups)
        if groups == ("register", "patch"):
            return ("patch", "register")
        if groups == ("special", "patch"):
            return ("patch", "special")
        raise AssertionError(groups)

    def ghost_runtime_trailing_token_count(self) -> int:
        return 1


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


def test_streamline_ffn_checkpoint_rebuilds_official_module() -> None:
    device = torch.device("cuda")
    model = _Model().to(device)
    adapter = _Adapter()
    teacher = StreamlineFFN(4, expansion_factor=4).to(device)
    packed = pack_streamline_ffn_parameters(teacher)
    spec = {
        "type": "recovery_streamline_ffn",
        "source_unit_id": "frame_1",
        "module_name": "recovery_patch_frame_1",
        "weight_shape": [int(packed.numel())],
        "hidden_size": 4,
        "expansion_factor": 4,
        "activation": "relu",
        "objective": "boundary_mse",
    }
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "frame_1", "weight": packed}],
        device=device,
    )
    x = torch.randn(2, 3, 4, device=device)

    actual = model.block(x)
    expected = teacher(x)

    assert isinstance(model.block.recovery_patch_frame_1, StreamlineFFN)
    assert model.block.recovery_patch_frame_1.fc1.out_features == 16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_boundary_recovery_disables_autocast_and_preserves_input_dtype() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7)
    weight = torch.randn(64, 64, device=device, generator=generator)
    patch = BoundaryRecovery(weight).to(device)

    x_fp32 = torch.randn(2, 5, 64, device=device, generator=generator)
    expected_fp32 = torch.matmul(x_fp32, weight)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual_fp32 = patch(x_fp32)
    assert actual_fp32.dtype == torch.float32
    torch.testing.assert_close(actual_fp32, expected_fp32, rtol=0.0, atol=0.0)

    x_bf16 = x_fp32.to(torch.bfloat16)
    expected_bf16 = torch.matmul(x_bf16.float(), weight).to(torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual_bf16 = patch(x_bf16)
    assert actual_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(actual_bf16, expected_bf16, rtol=0.0, atol=0.0)


def test_grouped_boundary_recovery_routes_public_frame_tokens() -> None:
    device = torch.device("cuda")
    weights = torch.stack([torch.eye(2) * scale for scale in (2.0, 3.0, 5.0)]).to(device)
    patch = BoundaryRecovery(
        weights,
        token_groups=("camera", "register", "patch"),
        token_group_starts=(0, 1, 3),
    ).to(device)
    tokens = torch.ones(2, 4, 6, 2, device=device)

    actual = patch(tokens)

    assert torch.equal(actual[:, :, :1], torch.full_like(actual[:, :, :1], 2.0))
    assert torch.equal(actual[:, :, 1:3], torch.full_like(actual[:, :, 1:3], 3.0))
    assert torch.equal(actual[:, :, 3:], torch.full_like(actual[:, :, 3:], 5.0))


def test_special_patch_boundary_shares_one_matrix_for_camera_and_register() -> None:
    device = torch.device("cuda")
    weights = torch.stack([torch.eye(2) * 2.0, torch.eye(2) * 5.0]).to(device)
    patch = BoundaryRecovery(
        weights,
        token_groups=("special", "patch"),
        token_group_starts=(0, 3),
    ).to(device)
    tokens = torch.ones(2, 4, 6, 2, device=device)

    actual = patch(tokens)

    assert torch.equal(actual[:, :, :3], torch.full_like(actual[:, :, :3], 2.0))
    assert torch.equal(actual[:, :, 3:], torch.full_like(actual[:, :, 3:], 5.0))


def test_grouped_ghost_checkpoint_routes_global_flattened_tokens() -> None:
    device = torch.device("cuda")
    model = _GroupedModel().to(device)
    adapter = _Adapter()
    spec = {
        "type": "recovery_ghost_interval_sequential_grouped",
        "source_unit_id": "global_0",
        "module_name": "recovery_patch_global_0",
        "weight_shape": [2, 2, 2],
        "token_groups": ["special", "patch"],
        "token_group_starts": [0, 3],
    }
    weights = torch.stack([torch.eye(2) * scale for scale in (2.0, 5.0)])
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "global_0", "weight": weights}],
        device=device,
    )
    batch_size, num_frames, tokens_per_frame = 2, 4, 6
    images = torch.zeros(batch_size, num_frames, 3, 8, 8, device=device)
    tokens = torch.ones(batch_size, num_frames * tokens_per_frame, 2, device=device)

    actual = model(images, tokens).reshape(batch_size, num_frames, tokens_per_frame, 2)

    assert torch.equal(actual[:, :, :3], torch.full_like(actual[:, :, :3], 2.0))
    assert torch.equal(actual[:, :, 3:], torch.full_like(actual[:, :, 3:], 5.0))


def test_mapanything_special_patch_group_reorders_internal_tokens() -> None:
    device = torch.device("cuda")
    model = _GroupedModel().to(device)
    adapter = _MapGroupedAdapter()
    spec = {
        "type": "recovery_ghost_interval_sequential_grouped",
        "source_unit_id": "global_0",
        "module_name": "recovery_patch_global_0",
        "weight_shape": [2, 2, 2],
        "token_groups": ["special", "patch"],
        "token_group_starts": [0, 2],
    }
    weights = torch.stack([torch.eye(2) * 2.0, torch.eye(2) * 5.0])
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "global_0", "weight": weights}],
        device=device,
    )
    images = torch.zeros(1, 2, 3, 8, 8, device=device)
    tokens = torch.ones(1, 11, 2, device=device)

    actual = model(images, tokens)
    per_frame = actual[:, :10].reshape(1, 2, 5, 2)

    assert torch.equal(per_frame[:, :, :3], torch.full_like(per_frame[:, :, :3], 5.0))
    assert torch.equal(per_frame[:, :, 3:], torch.full_like(per_frame[:, :, 3:], 2.0))
    assert torch.equal(actual[:, -1:], torch.ones_like(actual[:, -1:]))


def test_coupled_checkpoint_conditions_only_special_per_scene_and_frame() -> None:
    device = torch.device("cuda")
    model = _GroupedModel().to(device)
    adapter = _MapGroupedAdapter()
    spec = {
        "type": "recovery_ghost_interval_sequential_grouped",
        "source_unit_id": "global_0",
        "module_name": "recovery_patch_global_0",
        "weight_shape": [7, 2, 2],
        "token_groups": ["special", "patch"],
        "token_group_starts": [0, 2],
        "ghost_coupling_profile": "shared_residual_patch_mean",
    }
    packed = torch.zeros(7, 2, 2)
    packed[0] = torch.eye(2) * 2.0
    packed[1] = torch.eye(2) * 3.0
    packed[2] = torch.eye(2)
    install_recovery_modules(model, adapter, [spec], device=device)
    apply_recovery_weights(
        model, adapter, [spec], [{"source_unit_id": "global_0", "weight": packed}], device=device
    )
    # Runtime order is patch(3), special(2), then one excluded trailing token.
    batch_size, frames = 2, 2
    frame_tokens = torch.zeros(batch_size, frames, 5, 2, device=device)
    for batch in range(batch_size):
        for frame in range(frames):
            value = float(1 + 10 * batch + 3 * frame)
            frame_tokens[batch, frame, :3] = value
            frame_tokens[batch, frame, 3:] = 100.0 + value
    trailing = torch.full((batch_size, 1, 2), -7.0, device=device)
    tokens = torch.cat([frame_tokens.reshape(batch_size, -1, 2), trailing], dim=1)
    images = torch.zeros(batch_size, frames, 3, 8, 8, device=device)

    actual = model(images, tokens)
    actual_frames = actual[:, :-1].reshape_as(frame_tokens)

    torch.testing.assert_close(actual_frames[:, :, :3], frame_tokens[:, :, :3] * 3.0)
    patch_mean = frame_tokens[:, :, :3].mean(dim=2, keepdim=True)
    torch.testing.assert_close(
        actual_frames[:, :, 3:], frame_tokens[:, :, 3:] * 2.0 + patch_mean.expand(-1, -1, 2, -1)
    )
    torch.testing.assert_close(actual[:, -1:], trailing)


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


def test_adjacent_ghost_weights_remain_per_unit() -> None:
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
    assert len(compact_specs) == len(compact_weights) == 2
    assert compact_specs[0]["source_unit_id"] == "global_0"
    assert compact_specs[0]["target_unit_id"] == "frame_1"
    assert compact_specs[1]["source_unit_id"] == "frame_1"
    assert compact_specs[1]["target_unit_id"] == "global_1"
    torch.testing.assert_close(compact_weights[0]["weight"], first)
    torch.testing.assert_close(compact_weights[1]["weight"], second)

    linearpatch_specs = [
        {**spec, "type": "recovery_linearpatch"} for spec in specs
    ]
    compact_specs, compact_weights = compact_recovery_operators(
        linearpatch_specs,
        [{"weight": first}, {"weight": second}],
    )
    assert len(compact_specs) == len(compact_weights) == 2
    torch.testing.assert_close(compact_weights[0]["weight"], first)
    torch.testing.assert_close(compact_weights[1]["weight"], second)


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
            test_boundary_recovery_disables_autocast_and_preserves_input_dtype
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_grouped_boundary_recovery_routes_public_frame_tokens
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_special_patch_boundary_shares_one_matrix_for_camera_and_register
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_grouped_ghost_checkpoint_routes_global_flattened_tokens
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_mapanything_special_patch_group_reorders_internal_tokens
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_coupled_checkpoint_conditions_only_special_per_scene_and_frame
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(
            test_adjacent_dropped_unit_recovery_patches_compose_in_model_order
        )
    )
    tests.addTest(
        unittest.FunctionTestCase(test_adjacent_ghost_weights_remain_per_unit)
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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from analysis.backbones.registry import get_backbone
from analysis.recovery.operators import fuse_prunecomp_scale, fuse_replaceme_transform
from analysis.recovery.recovery_prune_v3 import (
    _fit_boundary_weight,
    _fit_ghost_cka_only,
    _fit_ghost_cka_ridge,
    _fit_ghost_cos,
    _fit_ghost_interval_sequential,
    _fit_ghost_mse_cka,
    _fit_prunecomp,
    _fit_replaceme_cos,
    _fit_replaceme_cos_sequential,
    _fit_replaceme_ls,
    _hadamard,
    _load_training_scenes,
    _optimize_cosine_transform,
)
from analysis.redundancy.cache_redundancy_v3 import (
    _save_scene_cache,
    forward_cache_with_auxiliaries,
)
from analysis.schema.v3 import LayerUnitSpec


class _LayerScale(nn.Module):
    def __init__(self, gamma: torch.Tensor) -> None:
        super().__init__()
        self.gamma = nn.Parameter(gamma.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.proj = nn.Linear(channels, channels, bias=True)
        self.mlp = nn.Module()
        self.mlp.fc2 = nn.Linear(channels, channels, bias=True)
        self.ls2 = _LayerScale(torch.linspace(0.5, 1.5, channels))


class _PatchSource(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)


class _Model(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.aggregator = nn.Module()
        self.aggregator.camera_token = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.aggregator.register_token = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.aggregator.patch_embed = _PatchSource(channels)
        self.blocks = nn.ModuleList([_Block(channels), _Block(channels)])


class _Adapter:
    name = "dummy"

    def block_for_unit(self, model: _Model, unit_id: str) -> nn.Module:
        return model.blocks[0 if unit_id == "frame_0" else 1]

    def layer_unit_catalog(self, model: _Model):
        return [
            LayerUnitSpec("frame_0", "frame", 0, "blocks.0"),
            LayerUnitSpec("global_0", "global", 0, "blocks.1"),
        ]


def test_replaceme_fusion_matches_residual_transform() -> None:
    device = torch.device("cuda")
    model = _Model().to(device)
    adapter = _Adapter()
    block = model.blocks[0]
    z = torch.randn(8, 4, device=device)
    transform = torch.eye(4, device=device) + 0.05 * torch.randn(4, 4, device=device)
    before = block.ls2(block.mlp.fc2(z)).detach()

    fuse_replaceme_transform(model, adapter, "frame_0", transform)

    after = block.ls2(block.mlp.fc2(z)).detach()
    torch.testing.assert_close(after, before @ transform, rtol=1e-4, atol=1e-4)


def test_prunecomp_fuses_only_through_source_unit() -> None:
    device = torch.device("cuda")
    model = _Model().to(device)
    adapter = _Adapter()
    upstream = model.blocks[0].attn.proj.weight.detach().clone()
    downstream = model.blocks[1].attn.proj.weight.detach().clone()

    fuse_prunecomp_scale(model, adapter, "frame_0", 1.25)

    torch.testing.assert_close(model.blocks[0].attn.proj.weight, upstream * 1.25)
    torch.testing.assert_close(model.blocks[1].attn.proj.weight, downstream)
    torch.testing.assert_close(model.aggregator.camera_token, torch.ones_like(model.aggregator.camera_token) * 1.25)


def test_replaceme_ls_recovers_known_transform() -> None:
    device = torch.device("cuda")
    r = torch.randn(1, 2, 8, 4)
    expected = torch.eye(4) + 0.1 * torch.randn(4, 4)
    h = torch.randn_like(r)
    target = h - r + (r.reshape(-1, 4) @ expected).reshape_as(r)

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return h if unit_id == "frame_0" else target

    with (
        mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit),
        mock.patch("analysis.recovery.recovery_prune_v3._load_auxiliary_tensor", return_value=r),
    ):
        actual, diagnostics = _fit_replaceme_ls(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
            num_layers=1,
            device=device,
            alpha=0.0,
        )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert diagnostics["num_rows"] == 16


def test_replaceme_ls_aligns_global_register_token_residual() -> None:
    device = torch.device("cuda")
    residual = torch.randn(1, 3, 2, 4)
    expected = torch.eye(4) + 0.1 * torch.randn(4, 4)
    source = torch.randn(1, 3, 7, 4)
    target = torch.randn_like(source)
    target[:, :, :2] = (
        source[:, :, :2]
        - residual
        + (residual.reshape(-1, 4) @ expected).reshape_as(residual)
    )

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "global_2" else target

    with (
        mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit),
        mock.patch("analysis.recovery.recovery_prune_v3._load_auxiliary_tensor", return_value=residual),
    ):
        actual, diagnostics = _fit_replaceme_ls(
            source_unit_id="global_2",
            target_unit_id="global_8",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "register_tokens"}],
            num_layers=9,
            device=device,
            alpha=0.0,
        )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert diagnostics["num_rows"] == 6


def test_prunecomp_averages_channel_ratios_to_scalar() -> None:
    source = torch.ones(1, 1, 2, 4)
    target = source * torch.tensor([1.0, 2.0, 3.0, 4.0])

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        weight, diagnostics = _fit_prunecomp(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
            num_layers=1,
            eps=1e-8,
        )

    torch.testing.assert_close(weight, torch.tensor([2.5]))
    assert abs(float(diagnostics["alpha"]) - 2.5) < 1e-7
    assert diagnostics["num_scenes"] == 1
    assert diagnostics["estimator"] == "mean_scene_mean_channel_magnitude_ratio"


def test_prunecomp_averages_scene_ratios_instead_of_global_magnitudes() -> None:
    sources = {
        "small": torch.ones(1, 1, 2, 4),
        "large": torch.full((1, 1, 2, 4), 100.0),
    }
    targets = {
        "small": sources["small"] * 2.0,
        "large": sources["large"],
    }

    def load_unit(_cache, item, unit_id, **_kwargs):
        scene_id = item["scene_id"]
        return sources[scene_id] if unit_id == "frame_0" else targets[scene_id]

    scenes = [
        {"_cache_dir": "/tmp", "scene_id": "small"},
        {"_cache_dir": "/tmp", "scene_id": "large"},
    ]
    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        weight, diagnostics = _fit_prunecomp(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=scenes,
            num_layers=1,
            eps=1e-8,
        )

    # Official MagComp averages the two batch/scene ratios (2 and 1), rather
    # than taking their source-magnitude-weighted global ratio (~1.01).
    torch.testing.assert_close(weight, torch.tensor([1.5]))
    assert diagnostics["num_scenes"] == 2


def test_linearpatch_constructs_hadamard_conjugated_diagonal() -> None:
    device = torch.device("cuda")
    source = torch.randn(1, 2, 16, 4, device=device)
    hmat = _hadamard(4, device=device)
    scale = torch.tensor([0.5, 1.0, 1.5, 2.0], device=device)
    expected = hmat @ torch.diag(scale) @ hmat.T
    target = source @ expected

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        actual = _fit_boundary_weight(
            method="recovery_linearpatch",
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
            num_layers=1,
            device=device,
            ridge=0.0,
        )

    torch.testing.assert_close(actual, expected.cpu(), rtol=2e-5, atol=2e-5)


def test_ghost_cos_improves_boundary_cosine_fit() -> None:
    device = torch.device("cuda")
    expected = torch.tensor(
        [[1.0, 0.4, 0.0, 0.0], [-0.2, 1.0, 0.3, 0.0], [0.0, 0.0, 1.0, 0.5], [0.1, 0.0, 0.0, 1.0]]
    )
    sources = {
        "scene_a": torch.randn(1, 1, 64, 4),
        "scene_b": torch.randn(1, 1, 64, 4),
    }
    targets = {scene_id: source @ expected for scene_id, source in sources.items()}

    def load_unit(_cache, item, unit_id, **_kwargs):
        scene_id = item["scene_id"]
        return sources[scene_id] if unit_id == "frame_0" else targets[scene_id]

    scenes = [
        {"_cache_dir": "/tmp", "scene_id": "scene_a"},
        {"_cache_dir": "/tmp", "scene_id": "scene_b"},
    ]
    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        actual, diagnostics = _fit_ghost_cos(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=scenes,
            num_layers=1,
            device=device,
            epochs=80,
            batch_size=128,
            lr=2e-2,
            seed=0,
        )

    source = torch.cat([value.reshape(-1, 4) for value in sources.values()])
    target = torch.cat([value.reshape(-1, 4) for value in targets.values()])
    initial_cos = torch.cosine_similarity(source, target, dim=-1).mean()
    fitted_cos = torch.cosine_similarity(source @ actual, target, dim=-1).mean()
    assert float(fitted_cos) > float(initial_cos) + 0.03
    assert float(fitted_cos) > 0.99
    assert diagnostics["rows_seen"] == 80 * 128
    assert diagnostics["optimizer_steps"] == 80


def test_ghost_more_loss_variants_change_only_objective_components() -> None:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(1, 1, 96, 4, generator=generator)
    expected = torch.tensor(
        [
            [1.1, 0.2, 0.0, 0.0],
            [-0.1, 0.9, 0.3, 0.0],
            [0.0, 0.0, 1.2, 0.2],
            [0.1, 0.0, 0.0, 0.8],
        ]
    )
    target = source @ expected

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    variants = [
        (("mse",), None),
        (("mse", "cosine"), None),
        (("mse", "cka"), None),
        (("mse", "cosine", "cka"), None),
        (("mse", "cosine"), {"mse": 1.0, "cosine": 0.04}),
        (("mse", "cka"), {"mse": 1.0, "cka": 0.10}),
        (
            ("mse", "cosine", "cka"),
            {"mse": 1.0, "cosine": 0.04, "cka": 0.10},
        ),
    ]
    with mock.patch(
        "analysis.recovery.recovery_prune_v3._load_unit_tensor",
        side_effect=load_unit,
    ):
        for components, component_weights in variants:
            weight, diagnostics = _fit_ghost_cos(
                source_unit_id="frame_0",
                target_unit_id="global_0",
                scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
                num_layers=1,
                device=device,
                epochs=3,
                batch_size=32,
                lr=1e-2,
                seed=0,
                loss_components=components,
                loss_weights=component_weights,
                method_name="synthetic ghost-more variant",
            )
            assert tuple(weight.shape) == (4, 4)
            assert diagnostics["loss_components"] == list(components)
            expected_weights = component_weights or {
                name: 1.0 for name in components
            }
            assert diagnostics["loss_weights"] == expected_weights
            assert diagnostics["loss_balance"] == (
                "raw_equal_coefficients"
                if component_weights is None
                else "explicit_coefficients"
            )
            assert set(diagnostics["initial_component_values"]) == set(components)
            assert set(diagnostics["initial_gradient_norms"]) == set(components)
            assert set(diagnostics["initial_weighted_gradient_norms"]) == set(
                components
            )
            for name in components:
                assert diagnostics["initial_weighted_gradient_norms"][name] == (
                    diagnostics["initial_gradient_norms"][name]
                    * expected_weights[name]
                )
            assert set(diagnostics["final_component_losses"]) == set(components)
            assert len(diagnostics["history"]) == 3
            assert diagnostics["optimizer_steps"] == 9


def test_ghost_more_ema_balances_auxiliary_values_to_mse() -> None:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(43)
    source = torch.randn(512, 4, generator=generator)
    target = source @ torch.tensor(
        [
            [1.1, 0.2, 0.0, 0.0],
            [-0.1, 0.9, 0.3, 0.0],
            [0.0, 0.0, 1.2, 0.2],
            [0.1, 0.0, 0.0, 0.8],
        ]
    )
    _, diagnostics = _optimize_cosine_transform(
        input_rows=[source],
        desired_rows=[target],
        device=device,
        epochs=2,
        batch_size=128,
        lr=1e-3,
        seed=0,
        method_name="synthetic EMA-balanced ghost-more variant",
        loss_components=("mse", "cosine", "cka"),
        loss_balance_targets={"cosine": 0.5, "cka": 0.5},
        loss_balance_ema_beta=0.95,
        loss_balance_weight_bounds={
            "cosine": (0.005, 0.2),
            "cka": (0.01, 1.0),
        },
    )
    assert diagnostics["loss_balance"] == "ema_value_to_mse"
    assert diagnostics["loss_balance_targets"] == {"cosine": 0.5, "cka": 0.5}
    assert diagnostics["loss_balance_ema_beta"] == 0.95
    assert diagnostics["loss_weights"] == {
        "mse": 1.0,
        "cosine": 1.0,
        "cka": 1.0,
    }
    for epoch in diagnostics["history"]:
        assert set(epoch["mean_effective_weights"]) == {"mse", "cosine", "cka"}
        assert set(epoch["mean_weighted_components"]) == {"mse", "cosine", "cka"}
        assert epoch["mean_effective_weights"]["mse"] == 1.0
    assert 0.005 - 1e-7 <= diagnostics["final_effective_weights"]["cosine"] <= 0.2 + 1e-7
    assert 0.01 - 1e-7 <= diagnostics["final_effective_weights"]["cka"] <= 1.0 + 1e-7
    assert set(diagnostics["loss_balance_clamp_counts"]) == {"cosine", "cka"}


def test_ghost_mse_cka_preserves_ls_mse_budget() -> None:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(7)
    sources = {
        "scene_a": torch.randn(1, 1, 64, 4, generator=generator),
        "scene_b": torch.randn(1, 1, 64, 4, generator=generator),
    }
    transform = torch.tensor(
        [[1.0, 0.3, 0.0, 0.0], [-0.1, 1.0, 0.2, 0.0], [0.0, 0.0, 1.0, 0.4], [0.1, 0.0, 0.0, 1.0]]
    )
    targets = {
        scene_id: source @ transform + 0.03 * torch.randn(source.shape, generator=generator)
        for scene_id, source in sources.items()
    }

    def load_unit(_cache, item, unit_id, **_kwargs):
        scene_id = item["scene_id"]
        return sources[scene_id] if unit_id == "frame_0" else targets[scene_id]

    scenes = [
        {"_cache_dir": "/tmp", "scene_id": "scene_a"},
        {"_cache_dir": "/tmp", "scene_id": "scene_b"},
    ]
    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        weight, diagnostics = _fit_ghost_mse_cka(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=scenes,
            num_layers=1,
            device=device,
            ridge=1e-6,
            epochs=3,
            lr=1e-3,
            cka_weight=1.0,
            mse_budget=0.01,
            max_delta_ratio=0.001,
            gradient_mode="scene",
            seed=0,
        )

    assert tuple(weight.shape) == (4, 4)
    assert diagnostics["selected_mse_ratio"] <= 1.01 + 1e-6
    assert diagnostics["weight_delta_ratio"] <= 0.001 + 1e-6
    assert len(diagnostics["history"]) == 3
    assert all("cka" in row and "mse_ratio" in row for row in diagnostics["history"])

    domain_scenes = [
        {"_cache_dir": "/tmp/domain_a/cache", "scene_id": "scene_a"},
        {"_cache_dir": "/tmp/domain_b/cache", "scene_id": "scene_b"},
    ]
    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        domain_weight, domain_diagnostics = _fit_ghost_mse_cka(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=domain_scenes,
            num_layers=1,
            device=device,
            ridge=1e-6,
            epochs=2,
            lr=1e-3,
            cka_weight=1.0,
            mse_budget=0.01,
            max_delta_ratio=0.001,
            gradient_mode="domain_agreement",
            seed=0,
        )
    assert tuple(domain_weight.shape) == (4, 4)
    assert domain_diagnostics["gradient_mode"] == "domain_agreement"
    assert domain_diagnostics["domains"] == {"domain_a": 1, "domain_b": 1}
    assert all(
        0.0 <= row["gradient_agreement_fraction"] <= 1.0
        for row in domain_diagnostics["history"]
    )


def test_ghost_cka_only_saves_deterministic_unconstrained_snapshots() -> None:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(19)
    sources = {
        "scene_a": torch.randn(1, 1, 64, 4, generator=generator),
        "scene_b": torch.randn(1, 1, 64, 4, generator=generator),
    }
    transforms = {
        "scene_a": torch.tensor(
            [[1.0, 0.4, 0.0, 0.0], [0.0, 1.0, 0.2, 0.0], [0.1, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
        ),
        "scene_b": torch.tensor(
            [[1.0, -0.2, 0.0, 0.1], [0.3, 1.0, 0.0, 0.0], [0.0, 0.2, 1.0, -0.2], [0.0, 0.0, 0.1, 1.0]]
        ),
    }
    targets = {
        scene_id: source @ transforms[scene_id]
        + 0.05 * torch.randn(source.shape, generator=generator)
        for scene_id, source in sources.items()
    }

    def load_unit(_cache, item, unit_id, **_kwargs):
        scene_id = item["scene_id"]
        return sources[scene_id] if unit_id == "frame_0" else targets[scene_id]

    scenes = [
        {"_cache_dir": "/tmp", "scene_id": "scene_a"},
        {"_cache_dir": "/tmp", "scene_id": "scene_b"},
    ]
    outputs = []
    with mock.patch(
        "analysis.recovery.recovery_prune_v3._load_unit_tensor",
        side_effect=load_unit,
    ):
        for _ in range(2):
            outputs.append(
                _fit_ghost_cka_only(
                    source_unit_id="frame_0",
                    target_unit_id="global_0",
                    scenes=scenes,
                    num_layers=1,
                    device=device,
                    ridge=1e-6,
                    epochs=5,
                    lr=1e-3,
                    snapshot_epochs="1,3,5",
                    seed=0,
                )
            )

    weight, diagnostics, snapshots = outputs[0]
    other_weight, other_diagnostics, other_snapshots = outputs[1]
    assert tuple(weight.shape) == (4, 4)
    assert sorted(snapshots) == [1, 3, 5]
    assert diagnostics["mse_used_for_optimization"] is False
    assert diagnostics["mse_used_for_selection"] is False
    assert diagnostics["selected_cka"] >= diagnostics["baseline_cka"]
    assert len(diagnostics["history"]) == 5
    assert all("mse_ratio" in row and "weight_delta_ratio" in row for row in diagnostics["history"])
    torch.testing.assert_close(weight, other_weight)
    assert diagnostics["history"] == other_diagnostics["history"]
    for epoch in snapshots:
        torch.testing.assert_close(snapshots[epoch], other_snapshots[epoch])


def test_ghost_cka_ridge_scans_feasible_closed_form_candidates() -> None:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(1, 1, 96, 4, generator=generator)
    source[..., 3] = source[..., 0] + 1e-3 * source[..., 3]
    expected = torch.eye(4) + 0.1 * torch.randn(4, 4, generator=generator)
    target = source @ expected + 0.02 * torch.randn(source.shape, generator=generator)

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    with mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit):
        weight, diagnostics = _fit_ghost_cka_ridge(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "ill_conditioned"}],
            num_layers=1,
            device=device,
            ridge=1e-6,
            ridge_grid="0,1e-8,1e-6,1e-4",
            mse_budget=0.01,
        )

    assert tuple(weight.shape) == (4, 4)
    assert len(diagnostics["candidates"]) == 4
    assert diagnostics["selected_mse_ratio"] <= 1.01 + 1e-6
    assert diagnostics["selected_ridge"] >= 1e-6


def test_replaceme_cos_improves_cosine_fit() -> None:
    device = torch.device("cuda")
    residual = torch.randn(1, 1, 128, 4)
    expected = torch.tensor(
        [[1.0, 0.4, 0.0, 0.0], [-0.2, 1.0, 0.3, 0.0], [0.0, 0.0, 1.0, 0.5], [0.1, 0.0, 0.0, 1.0]]
    )
    source = torch.randn_like(residual)
    desired = (residual.reshape(-1, 4) @ expected).reshape_as(residual)
    target = source - residual + desired

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    with (
        mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit),
        mock.patch("analysis.recovery.recovery_prune_v3._load_auxiliary_tensor", return_value=residual),
    ):
        actual, diagnostics = _fit_replaceme_cos(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
            num_layers=1,
            device=device,
            epochs=80,
            batch_size=128,
            lr=2e-2,
            seed=0,
        )

    fitted = residual.reshape(-1, 4) @ actual
    initial_cos = torch.cosine_similarity(residual.reshape(-1, 4), desired.reshape(-1, 4), dim=-1).mean()
    fitted_cos = torch.cosine_similarity(fitted, desired.reshape(-1, 4), dim=-1).mean()
    assert float(fitted_cos) > float(initial_cos) + 0.03
    assert float(fitted_cos) > 0.99
    assert diagnostics["rows_seen"] == 80 * 128


def test_replaceme_cos_batches_rows_globally_across_scenes() -> None:
    device = torch.device("cuda")
    residuals = {
        "scene_a": torch.randn(1, 1, 2, 4),
        "scene_b": torch.randn(1, 1, 2, 4),
    }
    sources = {scene_id: torch.randn_like(residual) for scene_id, residual in residuals.items()}
    targets = {
        scene_id: sources[scene_id] - residual + residual @ (torch.eye(4) + 0.1)
        for scene_id, residual in residuals.items()
    }

    def load_unit(_cache, item, unit_id, **_kwargs):
        scene_id = item["scene_id"]
        return sources[scene_id] if unit_id == "frame_0" else targets[scene_id]

    def load_auxiliary(_cache, item, _auxiliary_id, **_kwargs):
        return residuals[item["scene_id"]]

    scenes = [
        {"_cache_dir": "/tmp", "scene_id": "scene_a"},
        {"_cache_dir": "/tmp", "scene_id": "scene_b"},
    ]
    with (
        mock.patch("analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit),
        mock.patch("analysis.recovery.recovery_prune_v3._load_auxiliary_tensor", side_effect=load_auxiliary),
    ):
        _actual, diagnostics = _fit_replaceme_cos(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=scenes,
            num_layers=1,
            device=device,
            epochs=1,
            batch_size=4,
            lr=1e-4,
            seed=0,
        )

    assert diagnostics["rows_seen"] == 4
    assert diagnostics["optimizer_steps"] == 1


def test_cosine_scale_variants_restore_closed_form_global_magnitude() -> None:
    device = torch.device("cuda")
    source = torch.randn(1, 1, 128, 4)
    ghost_scale = 3.0
    ghost_target = source * ghost_scale

    def load_ghost_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else ghost_target

    with mock.patch(
        "analysis.recovery.recovery_prune_v3._load_unit_tensor",
        side_effect=load_ghost_unit,
    ):
        ghost_weight, ghost_diagnostics = _fit_ghost_cos(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "ghost_scale"}],
            num_layers=1,
            device=device,
            epochs=1,
            batch_size=128,
            lr=1e-4,
            seed=0,
            restore_global_scale=True,
        )

    torch.testing.assert_close(
        ghost_weight,
        torch.eye(4) * ghost_scale,
        rtol=2e-4,
        atol=2e-4,
    )
    assert ghost_diagnostics["restore_global_scale"] is True
    assert abs(float(ghost_diagnostics["global_scale"]) - ghost_scale) < 2e-4

    residual = torch.randn_like(source)
    replaceme_scale = 2.5
    replaceme_source = torch.randn_like(source)
    replaceme_target = replaceme_source - residual + residual * replaceme_scale

    def load_replaceme_unit(_cache, _item, unit_id, **_kwargs):
        return replaceme_source if unit_id == "frame_0" else replaceme_target

    with (
        mock.patch(
            "analysis.recovery.recovery_prune_v3._load_unit_tensor",
            side_effect=load_replaceme_unit,
        ),
        mock.patch(
            "analysis.recovery.recovery_prune_v3._load_auxiliary_tensor",
            return_value=residual,
        ),
    ):
        replaceme_weight, replaceme_diagnostics = _fit_replaceme_cos(
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "replaceme_scale"}],
            num_layers=1,
            device=device,
            epochs=1,
            batch_size=128,
            lr=1e-4,
            seed=0,
            restore_global_scale=True,
        )

    torch.testing.assert_close(
        replaceme_weight,
        torch.eye(4) * replaceme_scale,
        rtol=2e-4,
        atol=2e-4,
    )
    assert replaceme_diagnostics["restore_global_scale"] is True
    assert abs(float(replaceme_diagnostics["global_scale"]) - replaceme_scale) < 2e-4


def test_all_backbones_normalize_replaceme_residual_layouts() -> None:
    device = torch.device("cuda")
    batch_size, frames, patches, channels = 2, 3, 5, 4
    for backbone_name in ("vggt", "pi3", "vggt_omega"):
        adapter = get_backbone(backbone_name)
        frame_raw = torch.randn(batch_size * frames, patches, channels, device=device)
        global_raw = torch.randn(batch_size, frames * patches, channels, device=device)
        frame = adapter.normalize_unit_activation(
            "frame_0", frame_raw, batch_size=batch_size, num_frames=frames
        )
        global_ = adapter.normalize_unit_activation(
            "global_0", global_raw, batch_size=batch_size, num_frames=frames
        )
        assert tuple(frame.shape) == (batch_size, frames, patches, channels)
        assert tuple(global_.shape) == (batch_size, frames, patches, channels)


def test_replaceme_auxiliary_cache_path_and_manifest_record() -> None:
    device = torch.device("cuda")
    tensor = torch.randn(1, 2, 3, 4, device=device)
    images = torch.randn(1, 2, 3, 8, 8, device=device)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        record = _save_scene_cache(
            cache_root=root,
            scene_id="scene",
            unit_map={"frame_0": tensor},
            pred=SimpleNamespace(),
            meta={"patch_start_idx": 0},
            images=images,
            frame_ids=["0", "1"],
            image_paths=["0.jpg", "1.jpg"],
            scene_meta={},
            infer_s=0.1,
            adapter=SimpleNamespace(name="dummy"),
            model=nn.Module(),
            unit_ids_filter={"frame_0"},
            auxiliary_map={"replaceme_residual:frame_0": tensor},
        )
        expected = root / "scene" / "aux" / "replaceme_residual" / "frame_0.pt"
        assert expected.is_file()
        assert record["auxiliary_files"]["replaceme_residual:frame_0"] == str(expected.relative_to(root))
        metadata = json.loads((root / "scene" / "metadata_v3.json").read_text(encoding="utf-8"))
        assert metadata["auxiliary_files"] == record["auxiliary_files"]
        loaded = torch.load(expected, map_location="cuda", weights_only=False)
        assert tuple(loaded.shape) == (1, 2, 3, 4)


def test_shared_cache_forward_captures_auxiliary_and_removes_hook() -> None:
    device = torch.device("cuda")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ls2 = nn.Identity()

    class Adapter:
        name = "dummy"

        def block_for_unit(self, model, unit_id):
            assert unit_id == "frame_0"
            return model

        def normalize_unit_activation(
            self, _unit_id, tensor, *, batch_size, num_frames
        ):
            assert batch_size == 1
            assert num_frames == 2
            return tensor

        def forward_cache_batch(self, model, _images, _device):
            activation = model.ls2(torch.ones(1, 2, 3, 4, device=device))
            return SimpleNamespace(unit_map={"frame_0": activation})

    model = Model().to(device)
    forward, auxiliary = forward_cache_with_auxiliaries(
        adapter=Adapter(),
        model=model,
        images=torch.ones(1, 2, 3, 8, 8, device=device),
        device=device,
        auxiliary_ids={"replaceme_residual:frame_0"},
    )

    assert "frame_0" in forward.unit_map
    assert tuple(auxiliary["replaceme_residual:frame_0"].shape) == (1, 2, 3, 4)
    assert not model.ls2._forward_hooks


def test_training_scene_loader_merges_progressive_recovery_metadata() -> None:
    cache_dir = Path("/tmp/dummy-cache")
    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}.load_v3_manifest", return_value=object()),
        mock.patch(
            f"{module}.manifest_scenes_for_training",
            return_value=[{"scene_id": "scene", "dir": "scene", "layer_files": {}}],
        ),
        mock.patch(
            f"{module}.load_scene_metadata_v3",
            return_value={
                "image_paths": ["0.jpg", "1.jpg"],
                "frame_ids": ["0", "1"],
                "num_frames": 2,
            },
        ),
    ):
        scenes = _load_training_scenes([cache_dir], max_scenes=0)

    assert scenes[0]["image_paths"] == ["0.jpg", "1.jpg"]
    assert scenes[0]["frame_ids"] == ["0", "1"]
    assert scenes[0]["_cache_dir"] == str(cache_dir)


def test_replaceme_cos_multi_interval_fits_on_progressively_modified_model() -> None:
    class Adapter:
        def load_pretrained(self, **_kwargs):
            return SimpleNamespace(events=[])

        def default_hf_model(self) -> str:
            return "dummy"

        def apply_prune_plan(self, model, actions):
            return SimpleNamespace(
                events=model.events
                + [("drop", tuple(action.unit_id for action in actions))]
            )

    adapter = Adapter()
    records = [
        {
            "source_unit_id": "global_0",
            "target_unit_id": "global_1",
            "drop_start_index": 2,
            "drop_end_index": 3,
        },
        {
            "source_unit_id": "global_2",
            "target_unit_id": "frame_3",
            "drop_start_index": 6,
            "drop_end_index": 6,
        },
    ]
    states_seen = []

    def collect(**kwargs):
        states_seen.append(list(kwargs["model"].events))
        return [torch.eye(2)], [torch.eye(2)]

    def optimize(**_kwargs):
        return torch.eye(2), {}

    def fuse(model, _adapter, source_unit_id, _weight):
        model.events.append(("fuse", source_unit_id))

    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}._collect_current_replaceme_rows", side_effect=collect),
        mock.patch(f"{module}._optimize_cosine_transform", side_effect=optimize),
        mock.patch(f"{module}.fuse_replaceme_transform", side_effect=fuse),
    ):
        results = _fit_replaceme_cos_sequential(
            adapter=adapter,
            hf_model="",
            records=records,
            scenes=[{"scene_id": "dummy"}],
            device=torch.device("cuda"),
            epochs=1,
            batch_size=1,
            lr=1e-4,
            seed=0,
            restore_global_scale=False,
        )

    assert len(results) == 2
    assert states_seen[0] == []
    assert states_seen[1] == [
        ("fuse", "global_0"),
        ("drop", ("frame_1", "global_1")),
    ]
    assert results[0][1]["boundary_index"] == 0
    assert results[1][1]["boundary_index"] == 1


def test_ghost_interval_sequential_fits_on_progressively_modified_model() -> None:
    class Adapter:
        def load_pretrained(self, **_kwargs):
            return SimpleNamespace(events=[])

        def default_hf_model(self) -> str:
            return "dummy"

        def apply_prune_plan(self, model, actions):
            return SimpleNamespace(
                events=model.events
                + [("drop", tuple(action.unit_id for action in actions))]
            )

    records = [
        {
            "source_unit_id": "global_0",
            "target_unit_id": "global_1",
            "drop_start_index": 2,
            "drop_end_index": 3,
        },
        {
            "source_unit_id": "global_2",
            "target_unit_id": "frame_3",
            "drop_start_index": 6,
            "drop_end_index": 6,
        },
    ]
    states_seen = []

    def collect(**kwargs):
        states_seen.append(list(kwargs["model"].events))
        return [torch.eye(2)], [torch.eye(2)]

    def fit(**_kwargs):
        return torch.eye(2), {"fit_mode": "sequential_current_model"}

    def install(model, _adapter, specs, **_kwargs):
        model.events.append(("install", specs[0]["source_unit_id"]))
        return []

    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}._collect_current_boundary_rows", side_effect=collect),
        mock.patch(f"{module}._fit_ghost_rows", side_effect=fit),
        mock.patch(f"{module}.install_recovery_modules", side_effect=install),
        mock.patch(f"{module}.apply_recovery_weights"),
    ):
        results = _fit_ghost_interval_sequential(
            adapter=Adapter(),
            hf_model="",
            records=records,
            scenes=[{"scene_id": "dummy"}],
            device=torch.device("cuda"),
            ridge=1e-6,
        )

    assert len(results) == 2
    assert states_seen[0] == []
    assert states_seen[1] == [
        ("install", "global_0"),
        ("drop", ("frame_1", "global_1")),
    ]
    assert results[0][1]["boundary_index"] == 0
    assert results[1][1]["boundary_index"] == 1


def load_tests(_loader, tests, _pattern):
    for function in (
        test_replaceme_fusion_matches_residual_transform,
        test_prunecomp_fuses_only_through_source_unit,
        test_replaceme_ls_recovers_known_transform,
        test_replaceme_ls_aligns_global_register_token_residual,
        test_prunecomp_averages_channel_ratios_to_scalar,
        test_prunecomp_averages_scene_ratios_instead_of_global_magnitudes,
        test_linearpatch_constructs_hadamard_conjugated_diagonal,
        test_ghost_cos_improves_boundary_cosine_fit,
        test_ghost_more_loss_variants_change_only_objective_components,
        test_ghost_more_ema_balances_auxiliary_values_to_mse,
        test_ghost_mse_cka_preserves_ls_mse_budget,
        test_ghost_cka_only_saves_deterministic_unconstrained_snapshots,
        test_ghost_cka_ridge_scans_feasible_closed_form_candidates,
        test_replaceme_cos_improves_cosine_fit,
        test_replaceme_cos_batches_rows_globally_across_scenes,
        test_cosine_scale_variants_restore_closed_form_global_magnitude,
        test_all_backbones_normalize_replaceme_residual_layouts,
        test_replaceme_auxiliary_cache_path_and_manifest_record,
        test_shared_cache_forward_captures_auxiliary_and_removes_hook,
        test_training_scene_loader_merges_progressive_recovery_metadata,
        test_replaceme_cos_multi_interval_fits_on_progressively_modified_model,
        test_ghost_interval_sequential_fits_on_progressively_modified_model,
    ):
        tests.addTest(unittest.FunctionTestCase(function))
    return tests


if __name__ == "__main__":
    unittest.main()

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
from analysis.recovery.activation_store import RowShard, ShardedRowStore
from analysis.recovery.operators import (
    StreamlineFFN,
    apply_recovery_weights,
    fuse_replaceme_transform,
    load_streamline_ffn_parameters,
)
from analysis.recovery.recovery_prune_v3 import (
    GhostStabilityConfig,
    _close_dynamic_row_store,
    _collect_current_replaceme_rows,
    _accumulate_coupled_statistics,
    _fit_boundary_weight,
    _fit_current_ghost_boundary,
    _fit_ghost_interval_sequential,
    _fit_replaceme_cos,
    _fit_replaceme_cos_sequential,
    _fit_replaceme_ls,
    _fit_replaceme_ls_from_row_store,
    _forward_current_ghost_boundary,
    _ghost_token_layout_for_method,
    _hadamard,
    _load_training_scenes,
    _materialize_cosine_rows,
    _merge_special_ghost_token_groups,
    _optimize_cosine_transform,
    _optimize_streamline_ffn,
    _parse_scene_limits,
    _solve_ghost_from_statistics,
    _solve_ghost_with_prior,
    _solve_coupled_ghost_statistics,
    _solve_grouped_ghost_statistics,
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


def test_trace_scaled_ghost_ridge_is_statistic_scale_invariant() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(40, 4, generator=generator, device=device, dtype=torch.float64)
    desired = torch.randn(40, 4, generator=generator, device=device, dtype=torch.float64)
    xtx, xtd = x.T @ x, x.T @ desired
    prior = torch.eye(4, device=device)
    first, _ = _solve_ghost_with_prior(
        xtx, xtd, strength=1e-2, prior_weight=prior
    )
    second, _ = _solve_ghost_with_prior(
        xtx * 37.0, xtd * 37.0, strength=1e-2, prior_weight=prior
    )
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_hybrid_grouped_ghost_uses_all_solution_only_for_special() -> None:
    device = torch.device("cuda")
    eye = torch.eye(4, device=device, dtype=torch.float64)
    special_xtx, patch_xtx = 2.0 * eye, 7.0 * eye
    special_xtd, patch_xtd = 0.4 * eye, -0.7 * eye
    actual, diagnostics = _solve_grouped_ghost_statistics(
        xtx_by_group=(special_xtx, patch_xtx),
        xtd_by_group=(special_xtd, patch_xtd),
        rows_by_group=(8, 28),
        token_group_names=("special", "patch"),
        legacy_ridge=1e-6,
        stability=GhostStabilityConfig(profile="hybrid"),
    )
    expected_all = _solve_ghost_from_statistics(
        special_xtx + patch_xtx,
        special_xtd + patch_xtd,
        ridge=1e-6,
    )
    expected_patch = _solve_ghost_from_statistics(
        patch_xtx, patch_xtd, ridge=1e-6
    )
    torch.testing.assert_close(actual[0].to(device), expected_all)
    torch.testing.assert_close(actual[1].to(device), expected_patch)
    assert diagnostics["groups"]["special"]["prior"] == "all_exact"
    assert diagnostics["groups"]["patch"]["prior"] == "identity"


def test_autogroup_uses_relative_special_to_all_threshold() -> None:
    device = torch.device("cuda")
    eye = torch.eye(4, device=device, dtype=torch.float64)

    grouped, grouped_diagnostics = _solve_grouped_ghost_statistics(
        xtx_by_group=(2.0 * eye, 8.0 * eye),
        xtd_by_group=(0.4 * eye, 1.6 * eye),
        rows_by_group=(8, 32),
        token_group_names=("special", "patch"),
        legacy_ridge=0.0,
        stability=GhostStabilityConfig(),
        autogroup_threshold=20.0,
    )
    assert tuple(grouped.shape) == (2, 4, 4)
    assert grouped_diagnostics["autogroup"]["decision"] == "grouped"
    assert grouped_diagnostics["autogroup"]["value"] == 0.0

    ungrouped, ungrouped_diagnostics = _solve_grouped_ghost_statistics(
        xtx_by_group=(eye, 100.0 * eye),
        xtd_by_group=(100.0 * eye, torch.zeros_like(eye)),
        rows_by_group=(4, 400),
        token_group_names=("special", "patch"),
        legacy_ridge=0.0,
        stability=GhostStabilityConfig(),
        autogroup_threshold=20.0,
    )
    expected_all = _solve_ghost_from_statistics(
        101.0 * eye, 100.0 * eye, ridge=0.0
    )
    assert tuple(ungrouped.shape) == (4, 4)
    torch.testing.assert_close(ungrouped.to(device), expected_all)
    assert ungrouped_diagnostics["autogroup"]["decision"] == "un-grouped"
    assert ungrouped_diagnostics["autogroup"]["value"] > 20.0


def test_autogroup_patch_only_skips_grouping_and_matches_shared_solve() -> None:
    device = torch.device("cuda")
    eye = torch.eye(4, device=device, dtype=torch.float64)
    xtx = 7.0 * eye
    xtd = 1.4 * eye
    actual, diagnostics = _solve_grouped_ghost_statistics(
        xtx_by_group=(xtx,),
        xtd_by_group=(xtd,),
        rows_by_group=(28,),
        token_group_names=("patch",),
        legacy_ridge=1e-6,
        stability=GhostStabilityConfig(),
        autogroup_threshold=20.0,
    )
    expected = _solve_ghost_from_statistics(xtx, xtd, ridge=1e-6)
    torch.testing.assert_close(actual.to(device), expected)
    assert tuple(actual.shape) == (4, 4)
    assert diagnostics["autogroup"] == {
        "metric": "special_to_all_over_all_delta",
        "value": None,
        "threshold": 20.0,
        "comparison": "not_applicable",
        "decision": "un-grouped",
        "reason": "no_special_tokens",
        "candidate_grouped_weight_shape": None,
        "selected_weight_shape": [4, 4],
    }


def _synthetic_coupled_stats(
    *, device: torch.device, scale: float = 1.0
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(812)
    source = torch.randn(3, 2, 7, 4, generator=generator, device=device) * scale
    w0 = torch.eye(4, device=device) + 0.08 * torch.randn(4, 4, generator=generator, device=device)
    rs = torch.eye(4, device=device) + 0.06 * torch.randn(4, 4, generator=generator, device=device)
    rp = torch.eye(4, device=device) + 0.04 * torch.randn(4, 4, generator=generator, device=device)
    bps = 0.03 * torch.randn(4, 4, generator=generator, device=device)
    z = source @ w0
    target = torch.empty_like(source)
    target[:, :, 3:] = z[:, :, 3:] @ rp
    mean_patch = z[:, :, 3:].mean(dim=2, keepdim=True)
    target[:, :, :3] = z[:, :, :3] @ rs + mean_patch @ bps
    stats = _accumulate_coupled_statistics(
        None, source, target, patch_start_idx=3, device=device
    )
    return stats, source, target, w0


def test_shared_residual_packing_equals_two_linear_applications() -> None:
    device = torch.device("cuda")
    stats, source, _target, _ = _synthetic_coupled_stats(device=device)
    packed, diagnostics = _solve_coupled_ghost_statistics(
        stats,
        legacy_ridge=0.0,
        stability=GhostStabilityConfig(
            coupling_profile="shared_residual", residual_ridge=0.0
        ),
    )
    effective_special, effective_patch, _context, w0, rs, rp, _bps = packed.to(device)
    torch.testing.assert_close(source[:, :, :3] @ effective_special, (source[:, :, :3] @ w0) @ rs)
    torch.testing.assert_close(source[:, :, 3:] @ effective_patch, (source[:, :, 3:] @ w0) @ rp)
    assert diagnostics["groups"]["special"]["prior"] == "all_then_identity"


def test_joint_coupled_block_solve_recovers_special_and_patch_targets() -> None:
    device = torch.device("cuda")
    stats, source, target, _ = _synthetic_coupled_stats(device=device)
    packed, _diagnostics = _solve_coupled_ghost_statistics(
        stats,
        legacy_ridge=0.0,
        stability=GhostStabilityConfig(
            coupling_profile="shared_residual_patch_mean",
            residual_ridge=0.0,
            context_ridge=0.0,
        ),
    )
    effective_special, effective_patch, effective_context = packed[:3].to(device)
    predicted_special = source[:, :, :3] @ effective_special
    predicted_special += source[:, :, 3:].mean(dim=2, keepdim=True) @ effective_context
    predicted_patch = source[:, :, 3:] @ effective_patch
    torch.testing.assert_close(predicted_special, target[:, :, :3], rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(predicted_patch, target[:, :, 3:], rtol=3e-4, atol=3e-4)


def test_coupled_trace_regularization_is_activation_scale_invariant() -> None:
    device = torch.device("cuda")
    stats, _source, _target, _ = _synthetic_coupled_stats(device=device, scale=1.0)
    scaled = {
        key: (value * 49.0 if key not in {"rows_s", "rows_p"} else value.clone())
        for key, value in stats.items()
    }
    config = GhostStabilityConfig(
        coupling_profile="shared_residual_patch_mean",
        residual_ridge=0.1,
        context_ridge=0.01,
    )
    first, _ = _solve_coupled_ghost_statistics(stats, legacy_ridge=0.0, stability=config)
    second, _ = _solve_coupled_ghost_statistics(scaled, legacy_ridge=0.0, stability=config)
    torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)


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


def test_streamline_ffn_boundary_mse_decreases() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(17)
    source = torch.randn(256, 4, generator=generator)
    target = torch.zeros_like(source)
    store = ShardedRowStore(
        [RowShard(scene_id="synthetic", inputs=source, desired=target)]
    )
    try:
        packed, diagnostics = _optimize_streamline_ffn(
            row_store=store,
            device=device,
            epochs=30,
            batch_size=64,
            lr=2e-2,
            min_lr=1e-4,
            weight_decay=1e-4,
            expansion_factor=2,
            seed=0,
        )
    finally:
        store.close()
    fitted = StreamlineFFN(4, expansion_factor=2).to(device)
    load_streamline_ffn_parameters(fitted, packed)
    with torch.no_grad():
        fitted_loss = torch.nn.functional.mse_loss(fitted(source.to(device)), target.to(device))

    assert diagnostics["loss_components"] == ["mse"]
    assert diagnostics["rows_seen"] == 30 * 256
    assert diagnostics["best_loss"] < diagnostics["history"][0]["mean_loss"]
    assert float(fitted_loss) < 1e-3


def test_replaceme_ls_checkpoint_uses_fused_transform() -> None:
    device = torch.device("cuda")
    model = _Model().to(device)
    adapter = _Adapter()
    block = model.blocks[0]
    z = torch.randn(8, 4, device=device)
    transform = torch.eye(4, device=device) + 0.05 * torch.randn(4, 4, device=device)
    before = block.ls2(block.mlp.fc2(z)).detach()
    spec = {
        "type": "recovery_replaceme_ls",
        "source_unit_id": "frame_0",
        "module_name": "recovery_patch_frame_0",
        "weight_shape": [4, 4],
    }

    apply_recovery_weights(
        model,
        adapter,
        [spec],
        [{"source_unit_id": "frame_0", "weight": transform}],
        device=device,
    )

    after = block.ls2(block.mlp.fc2(z)).detach()
    torch.testing.assert_close(after, before @ transform, rtol=1e-4, atol=1e-4)


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


def test_replaceme_cos_clears_cuda_cache_before_capacity_check() -> None:
    events = []
    store = ShardedRowStore(
        [
            RowShard(
                scene_id="synthetic",
                inputs=torch.ones(2, 2),
                desired=torch.ones(2, 2),
            )
        ]
    )

    def empty_cache() -> None:
        events.append("empty_cache")

    def mem_get_info(_device):
        events.append("mem_get_info")
        return 32 * 1024**3, 48 * 1024**3

    with (
        mock.patch("torch.cuda.empty_cache", side_effect=empty_cache),
        mock.patch("torch.cuda.mem_get_info", side_effect=mem_get_info),
    ):
        inputs, desired, storage = _materialize_cosine_rows(
            store, device=torch.device("cuda")
        )

    assert events[:2] == ["empty_cache", "mem_get_info"]
    assert storage == "contiguous_gpu_float32"
    assert inputs is not None and inputs.device.type == "cuda"
    assert desired is not None and desired.device.type == "cuda"


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


def test_replaceme_ls_row_store_recovers_known_transform() -> None:
    device = torch.device("cuda")
    residual = torch.randn(32, 4)
    expected = torch.eye(4) + 0.1 * torch.randn(4, 4)
    store = ShardedRowStore(
        [
            RowShard(
                scene_id="synthetic",
                inputs=residual,
                desired=residual @ expected,
            )
        ]
    )
    try:
        actual, diagnostics = _fit_replaceme_ls_from_row_store(
            row_store=store,
            device=device,
            alpha=0.0,
            chunk_rows=7,
        )
    finally:
        store.close()

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert diagnostics["num_scenes"] == 1
    assert diagnostics["num_rows"] == 32
    assert diagnostics["chunk_rows"] == 7


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


def test_ghost_fits_one_matrix_per_backbone_token_group() -> None:
    device = torch.device("cuda")
    source = torch.randn(2, 3, 8, 4)
    expected = torch.stack(
        [
            torch.eye(4) + 0.05 * torch.randn(4, 4),
            torch.eye(4) + 0.05 * torch.randn(4, 4),
            torch.eye(4) + 0.05 * torch.randn(4, 4),
        ]
    )
    target = torch.empty_like(source)
    for group_index, (start, end) in enumerate(((0, 1), (1, 4), (4, 8))):
        target[:, :, start:end] = source[:, :, start:end] @ expected[group_index]

    def load_unit(_cache, _item, unit_id, **_kwargs):
        return source if unit_id == "frame_0" else target

    with mock.patch(
        "analysis.recovery.recovery_prune_v3._load_unit_tensor", side_effect=load_unit
    ):
        actual = _fit_boundary_weight(
            method="recovery_ghost",
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"_cache_dir": "/tmp", "scene_id": "synthetic"}],
            num_layers=1,
            device=device,
            ridge=0.0,
            ghost_token_group_names=("camera", "register", "patch"),
            ghost_token_group_starts=(0, 1, 4),
        )

    assert tuple(actual.shape) == (3, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_backbones_declare_their_ghost_token_groups() -> None:
    assert get_backbone("vggt").ghost_token_group_layout(patch_start_idx=5) == (
        ("camera", "register", "patch"),
        (0, 1, 5),
    )
    assert get_backbone("vggt_omega").ghost_token_group_layout(patch_start_idx=5) == (
        ("camera", "register", "patch"),
        (0, 1, 5),
    )
    assert get_backbone("mapanything").ghost_token_group_layout(patch_start_idx=4) == (
        ("register", "patch"),
        (0, 4),
    )
    assert get_backbone("pi3").ghost_token_group_layout(patch_start_idx=9) == (
        ("patch",),
        (0,),
    )


def test_special_ghost_layout_merges_camera_and_register() -> None:
    assert _merge_special_ghost_token_groups(
        ("camera", "register", "patch"), (0, 1, 5)
    ) == (("special", "patch"), (0, 5))
    assert _merge_special_ghost_token_groups(
        ("register", "patch"), (0, 4)
    ) == (("special", "patch"), (0, 4))
    assert _merge_special_ghost_token_groups(("patch",), (0,)) == (
        ("patch",),
        (0,),
    )
    assert get_backbone("mapanything").ghost_runtime_token_group_order(
        ("special", "patch")
    ) == ("patch", "special")


def test_special_grouped_method_uses_two_vggt_matrices() -> None:
    for method in (
        "recovery_ghost_interval_sequential_grouped",
        "recovery_ghost_interval_sequential_autogroup",
    ):
        for backbone in ("vggt", "vggt_omega"):
            assert _ghost_token_layout_for_method(
                adapter=get_backbone(backbone),
                method=method,
                patch_start_idx=5,
            ) == (("special", "patch"), (0, 5))
        assert _ghost_token_layout_for_method(
            adapter=get_backbone("mapanything"),
            method=method,
            patch_start_idx=4,
        ) == (("special", "patch"), (0, 4))
        assert _ghost_token_layout_for_method(
            adapter=get_backbone("pi3"),
            method=method,
            patch_start_idx=9,
        ) == (("patch",), (0,))


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
        assert "inference_time_ms" not in metadata
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


def test_training_scene_loader_limits_each_cache_dataset_independently() -> None:
    cache_dirs = [Path("/tmp/dummy-cache-a"), Path("/tmp/dummy-cache-b")]
    module = "analysis.recovery.recovery_prune_v3"

    def manifest_scenes(manifest):
        return [
            {"scene_id": f"{manifest}-{index}", "dir": "scene", "layer_files": {}}
            for index in range(3)
        ]

    with (
        mock.patch(
            f"{module}.load_v3_manifest",
            side_effect=lambda cache_dir: Path(cache_dir).name,
        ),
        mock.patch(
            f"{module}.manifest_scenes_for_training",
            side_effect=manifest_scenes,
        ),
        mock.patch(
            f"{module}.load_scene_metadata_v3",
            return_value={"image_paths": [], "frame_ids": [], "num_frames": 0},
        ),
    ):
        scenes = _load_training_scenes(cache_dirs, max_scenes=2)

    assert [scene["scene_id"] for scene in scenes] == [
        "dummy-cache-a-0",
        "dummy-cache-a-1",
        "dummy-cache-b-0",
        "dummy-cache-b-1",
    ]
    assert [scene["_cache_dir"] for scene in scenes] == [
        str(cache_dirs[0]),
        str(cache_dirs[0]),
        str(cache_dirs[1]),
        str(cache_dirs[1]),
    ]


def test_training_scene_loader_applies_exact_per_dataset_quotas() -> None:
    cache_dirs = [Path("/tmp/dummy-cache-c"), Path("/tmp/dummy-cache-d")]
    module = "analysis.recovery.recovery_prune_v3"

    def manifest_scenes(manifest):
        return [
            {"scene_id": f"{manifest}-{index}", "dir": "scene", "layer_files": {}}
            for index in range(4)
        ]

    with (
        mock.patch(
            f"{module}.load_v3_manifest",
            side_effect=lambda cache_dir: Path(cache_dir).name,
        ),
        mock.patch(
            f"{module}.manifest_scenes_for_training",
            side_effect=manifest_scenes,
        ),
        mock.patch(
            f"{module}.load_scene_metadata_v3",
            return_value={"image_paths": [], "frame_ids": [], "num_frames": 0},
        ),
    ):
        scenes = _load_training_scenes(
            cache_dirs, max_scenes=0, exact_limits=(1, 3)
        )

    assert [scene["scene_id"] for scene in scenes] == [
        "dummy-cache-c-0",
        "dummy-cache-d-0",
        "dummy-cache-d-1",
        "dummy-cache-d-2",
    ]


def test_scene_limit_parser_requires_an_exact_dataset_mapping() -> None:
    assert _parse_scene_limits("base=50,extra=100", ["base", "extra"]) == {
        "base": 50,
        "extra": 100,
    }
    for invalid in (
        "base=50",
        "base=50,extra=0",
        "base=50,extra=100,unknown=1",
        "base=50,base=100,extra=100",
    ):
        with unittest.TestCase().assertRaises(ValueError):
            _parse_scene_limits(invalid, ["base", "extra"])


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
        assert "num_layers" not in kwargs
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
        )

    assert len(results) == 2
    assert states_seen[0] == []
    assert states_seen[1] == [
        ("fuse", "global_0"),
        ("drop", ("frame_1", "global_1")),
    ]
    assert results[0][1]["boundary_index"] == 0
    assert results[1][1]["boundary_index"] == 1
    assert results[1][1]["source_mode"] == "sequential_current_model"
    assert results[1][1]["target_mode"] == "sequential_current_model"


def test_replaceme_current_target_uses_current_source_residual_and_target() -> None:
    device = torch.device("cuda")
    source_unit_id = "frame_0"
    target_unit_id = "global_0"
    current_source = torch.full((1, 1, 2, 2), 2.0, device=device)
    current_residual = torch.full((1, 1, 2, 2), 3.0, device=device)
    current_target = torch.full((1, 1, 2, 2), 7.0, device=device)
    prepared = SimpleNamespace(images=torch.ones(1, 1, 3, 8, 8))

    def forward(**kwargs):
        assert kwargs["unit_ids"] == {source_unit_id, target_unit_id}
        return (
            {
                source_unit_id: current_source,
                target_unit_id: current_target,
            },
            {},
            {f"replaceme_residual:{source_unit_id}": current_residual},
        )

    module = "analysis.recovery.recovery_prune_v3"
    store = None
    with (
        mock.patch(f"{module}.prepare_cache_batch", return_value=prepared),
        mock.patch(
            f"{module}.forward_selected_units_with_auxiliaries",
            side_effect=forward,
        ),
        mock.patch(
            f"{module}._load_unit_tensor",
            side_effect=AssertionError("Full target cache must not be read"),
        ),
    ):
        store = _collect_current_replaceme_rows(
            model=object(),
            adapter=object(),
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            scenes=[
                {
                    "scene_id": "synthetic",
                    "_cache_dir": "/tmp",
                    "image_paths": ["frame.jpg"],
                    "frame_ids": ["frame"],
                }
            ],
            device=device,
        )
    try:
        inputs, desired = store.first(store.num_rows)
        torch.testing.assert_close(inputs, torch.full_like(inputs, 3.0))
        # D = Y_current + R_current - H_current = 7 + 3 - 2 = 8.
        torch.testing.assert_close(desired, torch.full_like(desired, 8.0))
    finally:
        _close_dynamic_row_store(store)


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
    installed_types = []

    def fit(**kwargs):
        states_seen.append(list(kwargs["model"].events))
        return torch.eye(2), {
            "fit_mode": "sequential_current_model",
            "source_mode": "sequential_current_model",
            "target_mode": "sequential_current_model",
        }

    def install(model, _adapter, specs, **_kwargs):
        installed_types.append(specs[0]["type"])
        model.events.append(("install", specs[0]["source_unit_id"]))
        return []

    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}._fit_current_ghost_boundary", side_effect=fit),
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
            method="recovery_ghost_interval_sequential",
        )

    assert len(results) == 2
    assert states_seen[0] == []
    assert states_seen[1] == [
        ("drop", ("frame_1", "global_1")),
        ("install", "global_0"),
    ]
    assert results[0][1]["boundary_index"] == 0
    assert results[1][1]["boundary_index"] == 1
    assert results[1][1]["source_mode"] == "sequential_current_model"
    assert results[1][1]["target_mode"] == "sequential_current_model"
    assert installed_types == [
        "recovery_ghost_interval_sequential",
        "recovery_ghost_interval_sequential",
    ]


def test_current_target_ghost_uses_evolving_target_without_full_cache() -> None:
    device = torch.device("cuda")
    sources = [torch.randn(1, 2, 5, 4, device=device) for _ in range(3)]
    expected = torch.eye(4, device=device) + 0.05 * torch.randn(4, 4, device=device)
    current_targets = [source @ expected for source in sources]

    def forward(**kwargs):
        index = int(kwargs["item"]["scene_id"])
        assert kwargs["target_unit_id"] == "global_0"
        return {
            "frame_0": sources[index],
            "global_0": current_targets[index],
        }, torch.empty((), device=device)

    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}._forward_current_ghost_boundary", side_effect=forward),
        mock.patch(
            f"{module}._load_unit_tensor",
            side_effect=AssertionError("Full target cache must not be read"),
        ),
    ):
        weight, diagnostics = _fit_current_ghost_boundary(
            model=object(),
            adapter=object(),
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[
                {"scene_id": str(index), "_cache_dir": "/tmp"}
                for index in range(3)
            ],
            device=device,
            ridge=0.0,
        )

    torch.testing.assert_close(weight.to(device), expected, rtol=1e-4, atol=1e-4)
    assert diagnostics["source_mode"] == "sequential_current_model"
    assert diagnostics["target_mode"] == "sequential_current_model"


def test_current_target_ghost_fits_one_matrix_per_token_group() -> None:
    device = torch.device("cuda")
    source = torch.randn(1, 1, 12, 4, device=device)
    expected = torch.stack(
        [
            torch.eye(4, device=device) + 0.04 * torch.randn(4, 4, device=device),
            torch.eye(4, device=device) + 0.04 * torch.randn(4, 4, device=device),
        ]
    )
    target = torch.empty_like(source)
    target[:, :, :6] = source[:, :, :6] @ expected[0]
    target[:, :, 6:] = source[:, :, 6:] @ expected[1]

    def forward(**kwargs):
        assert kwargs["target_unit_id"] == "global_0"
        return {
            "frame_0": source,
            "global_0": target,
        }, torch.empty((), device=device)

    module = "analysis.recovery.recovery_prune_v3"
    with (
        mock.patch(f"{module}._forward_current_ghost_boundary", side_effect=forward),
        mock.patch(
            f"{module}._load_unit_tensor",
            side_effect=AssertionError("Full target cache must not be read"),
        ),
    ):
        weight, diagnostics = _fit_current_ghost_boundary(
            model=object(),
            adapter=object(),
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"scene_id": "synthetic", "_cache_dir": "/tmp"}],
            device=device,
            ridge=0.0,
            ghost_token_group_names=("special", "patch"),
            ghost_token_group_starts=(0, 6),
        )

    torch.testing.assert_close(weight.to(device), expected, rtol=3e-4, atol=3e-4)
    assert diagnostics["token_groups"] == ["special", "patch"]
    assert diagnostics["target_mode"] == "sequential_current_model"


def test_current_ghost_forward_requests_only_boundary_units() -> None:
    device = torch.device("cuda")

    class Adapter:
        def extract_layer_units(self, _model, images, *, unit_ids):
            assert unit_ids == {"frame_0", "global_0"}
            activation = torch.ones(1, 2, 3, 4, device=images.device)
            return {"frame_0": activation, "global_0": activation}, {}

    prepared = SimpleNamespace(images=torch.ones(1, 2, 3, 8, 8))
    module = "analysis.recovery.recovery_prune_v3"
    with mock.patch(f"{module}.prepare_cache_batch", return_value=prepared):
        unit_map, images = _forward_current_ghost_boundary(
            model=object(),
            adapter=Adapter(),
            source_unit_id="frame_0",
            target_unit_id="global_0",
            item={"scene_id": "scene", "image_paths": ["0.jpg", "1.jpg"]},
            device=device,
        )

    assert set(unit_map) == {"frame_0", "global_0"}
    assert images.device.type == "cuda"


def load_tests(_loader, tests, _pattern):
    for function in (
        test_trace_scaled_ghost_ridge_is_statistic_scale_invariant,
        test_hybrid_grouped_ghost_uses_all_solution_only_for_special,
        test_autogroup_uses_relative_special_to_all_threshold,
        test_autogroup_patch_only_skips_grouping_and_matches_shared_solve,
        test_shared_residual_packing_equals_two_linear_applications,
        test_joint_coupled_block_solve_recovers_special_and_patch_targets,
        test_coupled_trace_regularization_is_activation_scale_invariant,
        test_replaceme_fusion_matches_residual_transform,
        test_replaceme_ls_checkpoint_uses_fused_transform,
        test_replaceme_ls_recovers_known_transform,
        test_replaceme_cos_clears_cuda_cache_before_capacity_check,
        test_replaceme_ls_aligns_global_register_token_residual,
        test_replaceme_ls_row_store_recovers_known_transform,
        test_linearpatch_constructs_hadamard_conjugated_diagonal,
        test_ghost_fits_one_matrix_per_backbone_token_group,
        test_backbones_declare_their_ghost_token_groups,
        test_special_ghost_layout_merges_camera_and_register,
        test_special_grouped_method_uses_two_vggt_matrices,
        test_replaceme_cos_improves_cosine_fit,
        test_replaceme_cos_batches_rows_globally_across_scenes,
        test_all_backbones_normalize_replaceme_residual_layouts,
        test_replaceme_auxiliary_cache_path_and_manifest_record,
        test_shared_cache_forward_captures_auxiliary_and_removes_hook,
        test_training_scene_loader_merges_progressive_recovery_metadata,
        test_training_scene_loader_limits_each_cache_dataset_independently,
        test_training_scene_loader_applies_exact_per_dataset_quotas,
        test_scene_limit_parser_requires_an_exact_dataset_mapping,
        test_replaceme_cos_multi_interval_fits_on_progressively_modified_model,
        test_replaceme_current_target_uses_current_source_residual_and_target,
        test_ghost_interval_sequential_fits_on_progressively_modified_model,
        test_current_target_ghost_uses_evolving_target_without_full_cache,
        test_current_target_ghost_fits_one_matrix_per_token_group,
        test_current_ghost_forward_requests_only_boundary_units,
    ):
        tests.addTest(unittest.FunctionTestCase(function))
    return tests


if __name__ == "__main__":
    unittest.main()

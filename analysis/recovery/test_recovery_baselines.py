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
from analysis.recovery.operators import fuse_replaceme_transform
from analysis.recovery.recovery_prune_v3 import (
    _fit_boundary_weight,
    _fit_current_ghost_boundary,
    _fit_ghost_interval_sequential,
    _fit_replaceme_cos,
    _fit_replaceme_cos_sequential,
    _fit_replaceme_ls,
    _forward_current_ghost_boundary,
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
    installed_types = []

    def fit(**kwargs):
        states_seen.append(list(kwargs["model"].events))
        return torch.eye(2), {"fit_mode": "sequential_current_model"}

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
    assert installed_types == [
        "recovery_ghost_interval_sequential",
        "recovery_ghost_interval_sequential",
    ]


def test_current_ghost_boundary_streams_sufficient_statistics() -> None:
    device = torch.device("cuda")
    sources = [torch.randn(1, 2, 5, 4, device=device) for _ in range(3)]
    expected = torch.eye(4, device=device) + 0.05 * torch.randn(4, 4, device=device)
    targets = [source @ expected for source in sources]
    calls = []

    def forward(**kwargs):
        index = int(kwargs["item"]["scene_id"])
        calls.append(index)
        return {
            "frame_0": sources[index],
            "global_0": targets[index],
        }, torch.empty((), device=device)

    module = "analysis.recovery.recovery_prune_v3"
    with mock.patch(f"{module}._forward_current_ghost_boundary", side_effect=forward):
        weight, diagnostics = _fit_current_ghost_boundary(
            model=object(),
            adapter=object(),
            source_unit_id="frame_0",
            target_unit_id="global_0",
            scenes=[{"scene_id": str(index)} for index in range(3)],
            device=device,
            ridge=0.0,
        )

    torch.testing.assert_close(weight.to(device), expected, rtol=1e-4, atol=1e-4)
    assert calls == [0, 1, 2]
    assert diagnostics["num_scenes"] == 3
    assert diagnostics["num_rows"] == 30


def test_current_ghost_forward_requests_only_boundary_units() -> None:
    device = torch.device("cuda")

    class Adapter:
        def extract_layer_units(self, _model, images, *, unit_ids):
            assert unit_ids == {"frame_0", "global_0"}
            activation = torch.ones(1, 2, 3, 4, device=images.device)
            return {
                "frame_0": activation,
                "global_0": activation + 1,
            }, {}

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
        test_replaceme_fusion_matches_residual_transform,
        test_replaceme_ls_recovers_known_transform,
        test_replaceme_ls_aligns_global_register_token_residual,
        test_linearpatch_constructs_hadamard_conjugated_diagonal,
        test_replaceme_cos_improves_cosine_fit,
        test_replaceme_cos_batches_rows_globally_across_scenes,
        test_all_backbones_normalize_replaceme_residual_layouts,
        test_replaceme_auxiliary_cache_path_and_manifest_record,
        test_shared_cache_forward_captures_auxiliary_and_removes_hook,
        test_training_scene_loader_merges_progressive_recovery_metadata,
        test_replaceme_cos_multi_interval_fits_on_progressively_modified_model,
        test_ghost_interval_sequential_fits_on_progressively_modified_model,
        test_current_ghost_boundary_streams_sufficient_statistics,
        test_current_ghost_forward_requests_only_boundary_units,
    ):
        tests.addTest(unittest.FunctionTestCase(function))
    return tests


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from analysis.redundancy.graph_io import load_similarity
from analysis.redundancy.remaining_redundancy_v3 import (
    _load_base_prune_plan,
    _prepare_output_dir,
    _selection_provenance,
    _validate_base_selector,
    run_remaining_redundancy,
)
from analysis.schema.v3 import LayerUnitSpec


class _CountingBlock(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = float(delta)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value + self.delta


class _SkipBlock(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_CountingBlock(i + 1) for i in range(4)])


class _FakeAdapter:
    name = "vggt"

    def __init__(self) -> None:
        self.model: _FakeModel | None = None
        self.dropped_original: _CountingBlock | None = None
        self.pruned_before_forward = False
        self.requested_unit_ids: set[str] = set()

    def default_hf_model(self) -> str:
        return "unused"

    def load_pretrained(self, *, hf_model: str, device: str):
        del hf_model
        self.model = _FakeModel().to(device)
        return self.model

    def layer_unit_catalog(self, model: _FakeModel):
        unit_ids = ("frame_0", "global_0", "frame_1", "global_1")
        return [
            LayerUnitSpec(
                unit_id=unit_ids[index],
                kind="global",
                index=index,
                module_path=f"blocks.{index}",
            )
            for index in range(len(model.blocks))
        ]

    def block_for_unit(self, model: _FakeModel, unit_id: str):
        unit_ids = ("frame_0", "global_0", "frame_1", "global_1")
        return model.blocks[unit_ids.index(unit_id)]

    def apply_prune_plan(self, model: _FakeModel, actions):
        unit_ids = ("frame_0", "global_0", "frame_1", "global_1")
        for action in actions:
            index = unit_ids.index(action.unit_id)
            self.dropped_original = model.blocks[index]
            model.blocks[index] = _SkipBlock()
        self.pruned_before_forward = True
        return model

    def extract_layer_units(self, model: _FakeModel, images, unit_ids=None, layer_callback=None):
        del layer_callback
        if not self.pruned_before_forward:
            raise AssertionError("Inference started before pruning")
        self.requested_unit_ids = set(unit_ids or ())
        value = images.permute(0, 1, 3, 4, 2).reshape(
            int(images.shape[0]), int(images.shape[1]), -1, int(images.shape[2])
        )
        captures = {}
        unit_ids = ("frame_0", "global_0", "frame_1", "global_1")
        for index, block in enumerate(model.blocks):
            value = block(value)
            unit_id = unit_ids[index]
            if unit_id in self.requested_unit_ids:
                captures[unit_id] = value.detach()
        return captures, {"patch_start_idx": 1, "camera_token_count": 1}


class _FakeDataset:
    def validate_split(self, split):
        return split

    def output_tag(self, split):
        del split
        return "fake"

    def iter_scenes(self, data_root, *, max_scenes, max_frames, split):
        del data_root, max_scenes, max_frames, split
        yield SimpleNamespace(
            scene_id="scene0",
            images=torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
            meta={},
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by repository test policy")
class RemainingRedundancyGpuTest(unittest.TestCase):
    def test_prunes_before_fresh_inference_without_persisting_activations(self) -> None:
        adapter = _FakeAdapter()
        dataset = _FakeDataset()

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_root = Path(temporary_dir)
            output_dir = output_root / "vggt" / "fake" / "remaining_redundancy"
            selection_path = output_root / "remaining_skip_gt_knee.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "schema_version": "gt_single_interval_knee_v1",
                        "method": "threshold",
                        "selector": "i1_j1",
                        "drop_count": 1,
                        "datasets": ["fake"],
                        "threshold": 0.05,
                        "mean_normalized_degradation": 0.04,
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "analysis.redundancy.remaining_redundancy_v3.get_backbone",
                    return_value=adapter,
                ),
                mock.patch(
                    "analysis.redundancy.remaining_redundancy_v3.get_dataset",
                    return_value=dataset,
                ),
                mock.patch(
                    "analysis.redundancy.remaining_redundancy_v3.apply_pythonpath"
                ),
            ):
                result = run_remaining_redundancy(
                    experiment="vggt",
                    dataset="scannet",
                    data_root=Path(temporary_dir),
                    backbone="vggt",
                    prune_plan_root=Path(temporary_dir),
                    prune_plan_name="i1_j1",
                    expected_drop_count=1,
                    base_selection_path=selection_path,
                    device="cuda",
                    max_scenes=1,
                    max_frames=2,
                    metrics="cka",
                    token_regions="special",
                    output_dir=output_dir,
                    progress=False,
                )
                run_remaining_redundancy(
                    experiment="vggt",
                    dataset="scannet",
                    data_root=Path(temporary_dir),
                    backbone="vggt",
                    prune_plan_root=Path(temporary_dir),
                    prune_plan_name="i1_j1",
                    expected_drop_count=1,
                    base_selection_path=selection_path,
                    device="cuda",
                    max_scenes=1,
                    max_frames=2,
                    metrics="cosine",
                    token_regions="special",
                    output_dir=output_dir,
                    progress=False,
                )

            self.assertEqual(result, output_dir)
            self.assertTrue(adapter.pruned_before_forward)
            self.assertIsNotNone(adapter.dropped_original)
            self.assertEqual(adapter.dropped_original.calls, 0)
            self.assertEqual(
                adapter.requested_unit_ids,
                {"frame_0", "frame_1", "global_1"},
            )

            self.assertFalse((output_dir / "activations").exists())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["activations_persisted"])
            self.assertEqual(manifest["metrics"], ["cka", "cosine"])
            self.assertEqual(manifest["prune_plan_name"], "i1_j1")
            self.assertEqual(manifest["selection_method"], "threshold")
            self.assertEqual(len(manifest["selection_sha256"]), 64)
            self.assertEqual(manifest["dropped_unit_ids"], ["global_0"])
            self.assertEqual(
                manifest["remaining_unit_ids"],
                ["frame_0", "frame_1", "global_1"],
            )
            self.assertNotIn("activation_files", manifest["scenes"][0])
            graph_path = output_dir / "redundancy_graph_special.npz"
            self.assertTrue(graph_path.is_file())
            with np.load(graph_path, allow_pickle=True) as graph:
                self.assertEqual(
                    graph["unit_ids"].tolist(),
                    ["frame_0", "frame_1", "global_1"],
                )
                self.assertEqual(graph["cka_mean"].shape, (3, 3))
                self.assertEqual(graph["cosine_mean"].shape, (3, 3))
            loaded_ids, loaded_cka, provenance = load_similarity(
                experiment="vggt",
                dataset="fake",
                token_region="special",
                metric="cka",
                output_root=output_root,
                artifact_stage="remaining_redundancy",
            )
            self.assertEqual(loaded_ids, ["frame_0", "frame_1", "global_1"])
            self.assertEqual(loaded_cka.shape, (3, 3))
            self.assertEqual(provenance["artifact_stage"], "remaining_redundancy")
            aligned_ids, aligned_cka, aligned_provenance = load_similarity(
                experiment="vggt",
                dataset="fake",
                token_region="special",
                metric="cka",
                output_root=output_root,
                artifact_stage="remaining_redundancy",
                align_dropped_units=True,
            )
            self.assertEqual(
                aligned_ids,
                ["frame_0", "global_0", "frame_1", "global_1"],
            )
            self.assertEqual(aligned_cka.shape, (4, 4))
            self.assertTrue(np.isnan(aligned_cka[1, :]).all())
            self.assertTrue(np.isnan(aligned_cka[:, 1]).all())
            self.assertEqual(aligned_provenance["aligned_dropped_units"], True)

    def test_validates_base_selector(self) -> None:
        _validate_base_selector("i8_j20", 13)
        _validate_base_selector("prune_span12", 12)
        with self.assertRaisesRegex(ValueError, "drops 13 units"):
            _validate_base_selector("i8_j20", 12)
        with self.assertRaisesRegex(ValueError, "encodes 16 drops"):
            _validate_base_selector("prune_span16", 12)
        with self.assertRaisesRegex(ValueError, "exact interval selector"):
            _validate_base_selector("bad_selector", 12)

    def test_builds_exact_interval_plan_and_validates_selection(self) -> None:
        plan = _load_base_prune_plan(
            root=Path("/unused"),
            selector="i1_j2",
            backbone="vggt",
            num_units=4,
        )
        self.assertEqual(plan.drop, [1, 2])
        with tempfile.TemporaryDirectory() as temporary_dir:
            selection_path = Path(temporary_dir) / "knee.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "schema_version": "gt_single_interval_knee_v1",
                        "method": "threshold",
                        "selector": "i1_j2",
                        "drop_count": 2,
                        "datasets": ["scannet_dev50", "nuscenes_train"],
                        "threshold": 0.05,
                        "mean_normalized_degradation": 0.04,
                    }
                ),
                encoding="utf-8",
            )
            provenance = _selection_provenance(
                selection_path,
                selector="i1_j2",
                expected_drop_count=2,
            )
            self.assertEqual(provenance["selection_method"], "threshold")
            self.assertEqual(len(provenance["selection_sha256"]), 64)

    def test_rejects_incompatible_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            existing = {
                "schema": "remaining_redundancy_v1",
                "schema_version": "v3",
                "backbone": "vggt",
                "dataset": "fake",
                "prune_plan_name": "prune_span12",
                "expected_drop_count": 12,
                "dropped_unit_ids": ["global_0"],
                "remaining_unit_ids": ["frame_0"],
                "metrics": ["cka"],
                "token_regions": ["special"],
                "max_tokens": 0,
                "seed": 0,
                "max_frames": 2,
                "scene_ids": ["scene0"],
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(existing),
                encoding="utf-8",
            )
            alias = dict(existing, prune_plan_name="i1_j1")
            _prepare_output_dir(output_dir, alias)
            metric_extension = dict(existing, metrics=["cosine", "angular"])
            _prepare_output_dir(output_dir, metric_extension)
            incompatible = dict(
                existing,
                prune_plan_name="i2_j2",
                dropped_unit_ids=["frame_0"],
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _prepare_output_dir(output_dir, incompatible)


if __name__ == "__main__":
    unittest.main()

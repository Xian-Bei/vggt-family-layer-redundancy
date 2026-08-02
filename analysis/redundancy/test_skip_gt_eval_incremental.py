from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from analysis.backbones.adapters._vggt_base import VGGTFamilyAdapterMixin
from analysis.backbones.contracts import PredictionBundle
from analysis.redundancy.skip_gt_eval_v3 import (
    _append_scene_checkpoint,
    _finalize_skip_outputs_from_disk,
    _load_scene_checkpoint,
    _prepare_checkpoint_manifest,
    iter_unit_spans,
)


class _AffineBlock(nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.bias = float(bias)

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        del pos
        return x * self.scale + self.bias


class _Aggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_blocks = nn.ModuleList(
            [_AffineBlock(1.1, 1.0), _AffineBlock(1.3, 3.0)]
        )
        self.global_blocks = nn.ModuleList(
            [_AffineBlock(1.2, 2.0), _AffineBlock(1.4, 4.0)]
        )
        self.patch_start_idx = 0
        self.rope = None

    def forward(self, images: torch.Tensor):
        batch_size, frames = int(images.shape[0]), int(images.shape[1])
        patches, channels = 5, 4
        tokens = torch.zeros(
            (batch_size, frames, patches, channels),
            device=images.device,
        )
        outputs = []
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            frame = frame_block(tokens.reshape(batch_size * frames, patches, channels))
            global_ = global_block(frame.reshape(batch_size, frames * patches, channels))
            tokens = global_.reshape(batch_size, frames, patches, channels)
            outputs.append(tokens)
        return outputs, 2


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _Aggregator()
        self.camera_head = object()
        self.depth_head = None


class _TestAdapter(VGGTFamilyAdapterMixin):
    name = "test_vggt"

    def load_pretrained(self, **kwargs):
        del kwargs
        return _Model()

    def build_initial_tokens(self, model, images):
        del model
        return torch.zeros((1, int(images.shape[1]), 1, 1), device=images.device)

    def required_head_layers_for_cache(self, model, num_layers):
        del model
        return [num_layers - 1]

    def run_heads_from_aggregated_tokens(self, model, aggregated_tokens_list, patch_start_idx, images, query_points=None):
        del model, patch_start_idx, images, query_points
        return {"head": aggregated_tokens_list[-1]}

    def prediction_bundle_from_raw(self, raw, images):
        del images
        return PredictionBundle(raw=raw)


class IncrementalSkipReplayTest(unittest.TestCase):
    def test_hook_normalizes_isolated_surviving_global_unit(self) -> None:
        adapter = _TestAdapter()
        model = _Model().eval()
        images = torch.zeros((1, 30, 3, 4, 4))
        unit_map, meta = adapter.extract_layer_units(
            model,
            images,
            unit_ids={"global_1"},
        )
        self.assertEqual(tuple(unit_map["global_1"].shape), (1, 30, 5, 4))
        self.assertEqual(meta["patch_start_idx"], 2)

    def test_incremental_matches_direct_skips(self) -> None:
        adapter = _TestAdapter()
        model = _Model().eval()
        images = torch.zeros((1, 2, 3, 4, 4))
        spans = [(0, 0), (1, 1), (1, 2), (0, 3), (3, 3)]
        actual = {
            (i, j): pred.raw["head"]
            for i, j, pred in adapter.iter_skip_span_predictions(model, images, spans)
        }

        for i, j in spans:
            tokens = adapter.build_initial_tokens(model, images)
            frame = None
            final_concat = None
            for unit_idx in range(4):
                if not (i <= unit_idx <= j):
                    tokens = adapter._run_incremental_skip_unit(
                        model, tokens, unit_idx, pos_frame=None, pos_global=None
                    )
                if unit_idx % 2 == 0:
                    frame = tokens
                else:
                    final_concat = torch.cat([frame, tokens], dim=-1)
            torch.testing.assert_close(actual[(i, j)], final_concat)


class SkipCheckpointTest(unittest.TestCase):
    def test_span_enumeration(self) -> None:
        self.assertEqual(len(list(iter_unit_spans(48, max_span=8))), 356)

    def test_point_metric_family_is_retained(self) -> None:
        from analysis.redundancy.skip_gt_eval_v3 import normalize_metrics

        row = normalize_metrics({"acc": 1.0, "comp": 2.0, "cd": 3.0, "nc": 4.0})
        self.assertEqual(
            row,
            {"acc": 1.0, "comp": 2.0, "chamfer_distance": 3.0, "nc": 4.0},
        )

    def test_jsonl_resume_uses_latest_and_ignores_bad_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _append_scene_checkpoint(root, scene_id="a/b", span=(1, 2), metrics={"ate": 1.0})
            _append_scene_checkpoint(root, scene_id="a/b", span=(1, 2), metrics={"ate": 2.0})
            path = root / "checkpoints" / "a__b.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write('{"broken":')
            rows = _load_scene_checkpoint(root, "a/b")
            self.assertEqual(rows[(1, 2)]["ape"], 2.0)

    def test_manifest_rejects_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _prepare_checkpoint_manifest(root, config={"x": 1}, resume=True)
            with self.assertRaises(RuntimeError):
                _prepare_checkpoint_manifest(root, config={"x": 2}, resume=True)
            payload = json.loads((root / "skip_eval_manifest_v3.json").read_text())
            self.assertEqual(payload["config"], {"x": 1})

    def test_checkpoint_finalization_keeps_public_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spans = list(iter_unit_spans(2, max_span=0))
            for index, span in enumerate(spans):
                _append_scene_checkpoint(
                    root,
                    scene_id="scene0",
                    span=span,
                    metrics={"ape": 1.0 + index},
                )
            _finalize_skip_outputs_from_disk(
                out_root=root,
                spans=spans,
                n_units=2,
                unit_ids=["frame_0", "global_0"],
                metrics=["ape"],
                upper_metrics={"ape": 1.0},
                upper_scene_metrics={"scene0": {"ape": 1.0}},
                backbone="vggt",
                dataset="test",
                max_span=0,
                fusion_mode="mixed_units",
                elapsed_seconds=1.0,
                ensure_scene_reports=True,
            )
            for name in (
                "skip_eval_report_v3.json",
                "skip_eval_scenes_v3.json",
                "skip_eval_v3.npz",
                "skip_eval_summary_v3.json",
                "skip_eval_interval_index_v3.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            self.assertTrue((root / "i0_j0" / "summary.json").is_file())
            summary = json.loads((root / "skip_eval_summary_v3.json").read_text())
            self.assertEqual(summary["num_scenes_completed"], 1)


if __name__ == "__main__":
    unittest.main()

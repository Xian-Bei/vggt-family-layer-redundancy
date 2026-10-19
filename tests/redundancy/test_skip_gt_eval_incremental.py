from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from analysis.backbones.adapters._vggt_base import VGGTFamilyAdapterMixin
from analysis.backbones.contracts import EvalSceneContext, PredictionBundle
from analysis.backbones.prediction import bundle_from_pose_depth
from analysis.backbones.pose_geometry import unproject_depth_payload
from analysis.datasets.contracts import PreparedSkipEvalScene
from analysis.eval.async_pipeline import (
    ASYNC_SCORE_DEFERRED_FIELDS,
    ASYNC_SCORE_FIELDS,
    DEFAULT_ASYNC_SCORE_MAX_PENDING,
    DEFAULT_ASYNC_SCORE_WORKERS,
    AsyncPredictionScorePipeline,
    async_score_enabled,
)
from analysis.redundancy.metrics import upper_packed_to_symmetric_matrix
from analysis.redundancy.skip_gt_eval_v3 import (
    _append_scene_checkpoint,
    _build_parser,
    _finalize_skip_outputs_from_disk,
    _load_scene_checkpoint,
    _prepare_checkpoint_manifest,
    build_packed_matrices,
    iter_unit_spans,
    normalize_metrics,
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

    def prediction_bundle_from_raw(
        self,
        raw,
        images,
        *,
        defer_depth_unprojection=False,
    ):
        del images, defer_depth_unprojection
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

    def test_negative_max_span_is_relative_to_unit_count(self) -> None:
        spans = list(iter_unit_spans(48, max_span=-1))

        self.assertEqual(len(spans), 1175)
        self.assertNotIn((0, 47), spans)
        self.assertEqual(max(end - start + 1 for start, end in spans), 47)

    def test_negative_max_span_rejects_non_positive_resolved_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "resolves to 0"):
            list(iter_unit_spans(48, max_span=-48))

    def test_packed_matrices_leave_unevaluated_spans_as_nan(self) -> None:
        spans = list(iter_unit_spans(4, max_span=2))
        span_metrics = {span: {"ape": 2.0} for span in spans}

        delta, pct = build_packed_matrices(
            n_units=4,
            spans=spans,
            upper_metrics={"ape": 1.0},
            span_metrics=span_metrics,
            metrics=["ape"],
        )

        for packed in (delta["ape"], pct["ape"]):
            matrix = np.asarray(upper_packed_to_symmetric_matrix(packed, 4))
            self.assertTrue(np.isfinite(matrix[0, 1]))
            self.assertTrue(np.isnan(matrix[0, 2]))
            self.assertTrue(np.isnan(matrix[0, 3]))
            self.assertTrue(np.isnan(matrix[1, 3]))

    def test_removed_point_metric_is_ignored(self) -> None:
        from analysis.redundancy.skip_gt_eval_v3 import normalize_metrics

        row = normalize_metrics({"acc": 1.0, "comp": 2.0, "cd": 3.0, "nc": 4.0})
        self.assertEqual(
            row,
            {"acc": 1.0, "comp": 2.0, "chamfer_distance": 3.0},
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for async score tests")
class AsyncScorePipelineTest(unittest.TestCase):
    def _context(self) -> EvalSceneContext:
        return EvalSceneContext(
            device=torch.device("cuda"),
            profile=object(),
            backbone_name="test",
            dataset_name="test",
        )

    def test_stages_only_score_fields_and_bounds_pending_work(self) -> None:
        self.assertEqual(ASYNC_SCORE_FIELDS, ("points", "camera_poses"))
        worker_threads = set()

        def scorer(_adapter, prediction, _prepared, _ctx):
            worker_threads.add(threading.get_ident())
            self.assertIsNone(prediction.depth)
            self.assertEqual(prediction.raw, {})
            self.assertFalse(prediction.points.is_cuda)
            self.assertTrue(prediction.points.is_pinned())
            self.assertTrue(prediction.camera_poses.is_pinned())
            return {"ate": float(prediction.points.reshape(-1)[0])}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared = PreparedSkipEvalScene(
                scene_id="scene",
                images=torch.empty(0, device="cuda"),
            )
            pipeline = AsyncPredictionScorePipeline(
                device=torch.device("cuda"),
                score_fn=lambda _span, prediction: normalize_metrics(
                    scorer(object(), prediction, prepared, self._context())
                ),
                result_fn=lambda span, metrics: _append_scene_checkpoint(
                    root, scene_id="scene", span=span, metrics=metrics
                ),
                error_fn=lambda _span, exc: (_ for _ in ()).throw(exc),
                timing=True,
            )
            for index in range(DEFAULT_ASYNC_SCORE_MAX_PENDING + 2):
                prediction = PredictionBundle(
                    points=torch.full((1, 2, 3), float(index), device="cuda"),
                    camera_poses=torch.eye(4, device="cuda").reshape(1, 1, 4, 4),
                    depth=torch.ones((64, 64), device="cuda"),
                    raw={"not_staged": torch.ones(1, device="cuda")},
                )
                pipeline.submit((index, index), prediction)
                self.assertLessEqual(
                    len(pipeline.pending), DEFAULT_ASYNC_SCORE_MAX_PENDING
                )
            pipeline.close()
            self.assertEqual(pipeline.score_workers, DEFAULT_ASYNC_SCORE_WORKERS)
            self.assertEqual(
                pipeline.max_pending_seen, DEFAULT_ASYNC_SCORE_MAX_PENDING
            )
            self.assertGreaterEqual(pipeline.d2h_ms, 0.0)
            rows = [json.loads(line) for line in (root / "checkpoints" / "scene.jsonl").read_text().splitlines()]
            expected = list(range(DEFAULT_ASYNC_SCORE_MAX_PENDING + 2))
            self.assertEqual([row["span_i"] for row in rows], expected)
            self.assertEqual(
                [row["metrics"]["ape"] for row in rows],
                [float(x) for x in expected],
            )
            self.assertGreaterEqual(len(worker_threads), 1)
            self.assertLessEqual(len(worker_threads), DEFAULT_ASYNC_SCORE_WORKERS)
            self.assertNotIn(threading.get_ident(), worker_threads)

    def test_default_score_workers_run_configured_tasks_concurrently(self) -> None:
        worker_threads: set[int] = set()
        worker_lock = threading.Lock()
        both_started = threading.Event()
        release_workers = threading.Event()

        def scorer(_key, _prediction):
            with worker_lock:
                worker_threads.add(threading.get_ident())
                if len(worker_threads) == DEFAULT_ASYNC_SCORE_WORKERS:
                    both_started.set()
            if not release_workers.wait(timeout=5.0):
                raise TimeoutError("score workers did not run concurrently")
            return {"ape": 1.0}

        results = []
        pipeline = AsyncPredictionScorePipeline(
            device=torch.device("cuda"),
            score_fn=scorer,
            result_fn=lambda key, metrics: results.append((key, metrics)),
            error_fn=lambda _key, exc: (_ for _ in ()).throw(exc),
            timing=False,
        )
        try:
            for index in range(DEFAULT_ASYNC_SCORE_WORKERS):
                pipeline.submit(
                    index,
                    PredictionBundle(
                        points=torch.ones((1, 1, 3), device="cuda"),
                        camera_poses=torch.eye(4, device="cuda").reshape(1, 1, 4, 4),
                    ),
                )
            self.assertTrue(both_started.wait(timeout=5.0))
        finally:
            release_workers.set()
            pipeline.close()
        self.assertEqual(len(worker_threads), DEFAULT_ASYNC_SCORE_WORKERS)
        self.assertEqual(
            [key for key, _metrics in results],
            list(range(DEFAULT_ASYNC_SCORE_WORKERS)),
        )

    def test_worker_failure_leaves_span_for_resume(self) -> None:
        def scorer(_adapter, prediction, _prepared, _ctx):
            value = int(prediction.points.reshape(-1)[0])
            if value == 2:
                raise RuntimeError("expected scoring failure")
            return {"ate": float(value)}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prepared = PreparedSkipEvalScene("scene", torch.empty(0, device="cuda"))
            pipeline = AsyncPredictionScorePipeline(
                device=torch.device("cuda"),
                score_fn=lambda _span, prediction: normalize_metrics(
                    scorer(object(), prediction, prepared, self._context())
                ),
                result_fn=lambda span, metrics: _append_scene_checkpoint(
                    root, scene_id="scene", span=span, metrics=metrics
                ),
                error_fn=lambda _span, _exc: None,
                timing=False,
            )
            for index in range(4):
                pipeline.submit(
                    (index, index),
                    PredictionBundle(
                        points=torch.full((1, 1, 3), float(index), device="cuda"),
                        camera_poses=torch.eye(4, device="cuda").reshape(1, 1, 4, 4),
                    ),
                )
            pipeline.close()
            completed = _load_scene_checkpoint(root, "scene")
            self.assertEqual(set(completed), {(0, 0), (1, 1), (3, 3)})

    def test_deferred_vggt_unprojection_runs_in_cpu_worker(self) -> None:
        self.assertEqual(
            ASYNC_SCORE_DEFERRED_FIELDS,
            (
                "depth",
                "deferred_unprojection_extrinsic",
                "deferred_unprojection_intrinsic",
                "camera_poses",
            ),
        )
        worker_threads = set()
        depth = torch.tensor(
            [[[[[1.0], [2.0]], [[3.0], [4.0]]]]], device="cuda"
        )
        extrinsic = torch.eye(4, device="cuda")[:3].reshape(1, 1, 3, 4)
        intrinsic = torch.tensor(
            [[[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]]],
            device="cuda",
        )
        prediction = PredictionBundle(
            depth=depth,
            camera_poses=torch.eye(4, device="cuda").reshape(1, 1, 4, 4),
            raw={
                "deferred_unprojection_backbone": "vggt",
                "deferred_unprojection_extrinsic": extrinsic,
                "deferred_unprojection_intrinsic": intrinsic,
            },
        )
        expected = torch.from_numpy(
            unproject_depth_payload(
                depth.cpu(),
                extrinsic.cpu(),
                intrinsic.cpu(),
                backbone="vggt",
            )
        ).float().unsqueeze(0)

        def scorer(_key, staged):
            worker_threads.add(threading.get_ident())
            self.assertIsNotNone(staged.points)
            self.assertEqual(staged.points.device.type, "cpu")
            self.assertEqual(tuple(staged.points.shape), (1, 1, 2, 2, 3))
            self.assertTrue(torch.equal(staged.points, expected))
            return {"ape": float(staged.points.sum())}

        results = []
        pipeline = AsyncPredictionScorePipeline(
            device=torch.device("cuda"),
            score_fn=scorer,
            result_fn=lambda key, metrics: results.append((key, metrics)),
            error_fn=lambda _key, exc: (_ for _ in ()).throw(exc),
            timing=True,
        )
        pipeline.submit((0, 0), prediction)
        self.assertIsNone(prediction.points)
        pipeline.close()
        staged_slot_fields = set().union(
            *(set(slot.tensors) for slot in pipeline.free_slots)
        )
        self.assertEqual(staged_slot_fields, set(ASYNC_SCORE_DEFERRED_FIELDS))
        self.assertEqual([key for key, _metrics in results], [(0, 0)])
        self.assertGreaterEqual(len(worker_threads), 1)
        self.assertLessEqual(len(worker_threads), DEFAULT_ASYNC_SCORE_WORKERS)
        self.assertNotIn(threading.get_ident(), worker_threads)

    def test_vggt_bundle_defers_without_gpu_cpu_roundtrip(self) -> None:
        depth = torch.ones((1, 1, 2, 2, 1), device="cuda")
        pose_enc = torch.zeros((1, 1, 9), device="cuda")
        images = torch.zeros((1, 1, 3, 2, 2), device="cuda")
        extrinsic = torch.eye(4, device="cuda")[:3].reshape(1, 1, 3, 4)
        intrinsic = torch.tensor(
            [[[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]]],
            device="cuda",
        )
        with mock.patch(
            "analysis.backbones.prediction.decode_pose_enc_to_extrinsics_intrinsics",
            return_value=(extrinsic, intrinsic),
        ), mock.patch(
            "analysis.backbones.prediction.unproject_depth_payload"
        ) as unproject:
            prediction = bundle_from_pose_depth(
                raw={"pose_enc": pose_enc, "depth": depth},
                images=images,
                backbone_name="vggt",
                defer_depth_unprojection=True,
            )
        unproject.assert_not_called()
        self.assertIsNone(prediction.points)
        self.assertIs(prediction.depth, depth)
        self.assertIs(
            prediction.raw["deferred_unprojection_extrinsic"], extrinsic
        )
        self.assertIs(
            prediction.raw["deferred_unprojection_intrinsic"], intrinsic
        )
        self.assertTrue(prediction.camera_poses.is_cuda)

    def test_cli_and_cpu_fallback(self) -> None:
        args = _build_parser().parse_args(
            [
                "--dataset",
                "scannet",
                "--data-root",
                "data/scans_dev50",
                "--no-async-score",
                "--score-workers",
                "5",
                "--score-max-pending",
                "15",
            ]
        )
        self.assertTrue(args.no_async_score)
        self.assertEqual(args.score_workers, 5)
        self.assertEqual(args.score_max_pending, 15)
        self.assertFalse(async_score_enabled(True, torch.device("cpu")))
        self.assertFalse(async_score_enabled(False, torch.device("cuda")))
        self.assertTrue(async_score_enabled(True, torch.device("cuda")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from analysis.backbones.contracts import PredictionBundle
from analysis.datasets.contracts import EvalSampleBatch, PreparedSkipEvalScene
from analysis.eval.async_pipeline import (
    DEFAULT_ASYNC_SCORE_MAX_PENDING,
    DEFAULT_ASYNC_SCORE_WORKERS,
    AsyncScenePreloader,
)
from analysis.eval.eval_v3 import _build_parser, evaluate_model_v3
from analysis.schema.v3 import DatasetEvalProfile


class _Adapter:
    def __init__(self) -> None:
        self.defer_cpu_geometry_flags: list[bool] = []

    def forward_predictions(
        self,
        _model,
        images: torch.Tensor,
        *,
        defer_cpu_geometry: bool = False,
    ) -> PredictionBundle:
        self.defer_cpu_geometry_flags.append(bool(defer_cpu_geometry))
        value = images[:, :, :1, :1, :1].reshape(1, 1, 1)
        points = value.expand(1, 1, 3).contiguous()
        poses = torch.eye(4, device=images.device).reshape(1, 1, 4, 4)
        return PredictionBundle(points=points, camera_poses=poses)


class _Dataset:
    name = "fake"

    def __init__(self) -> None:
        self.score_threads: set[int] = set()
        self.prepare_threads: set[int] = set()

    def validate_split(self, split=None):
        return split

    def eval_profile(self):
        return DatasetEvalProfile(dataset=self.name)

    def output_tag(self, _split=None):
        return self.name

    def count_scenes(self, _root, *, max_scenes=0, split=None):
        del split
        return min(5, max_scenes) if max_scenes > 0 else 5

    def iter_scenes(self, _root, *, max_scenes=0, max_frames=0, split=None):
        del max_frames, split
        count = min(5, max_scenes) if max_scenes > 0 else 5
        for index in range(count):
            yield EvalSampleBatch(scene_id=f"scene{index}", images=None, frame_ids=[index])

    def prepare_eval_scene_cpu(self, _adapter, batch, _ctx):
        self.prepare_threads.add(threading.get_ident())
        index = int(str(batch.scene_id).removeprefix("scene"))
        images = torch.full((1, 1, 3, 2, 2), float(index), device="cpu")
        return PreparedSkipEvalScene(str(batch.scene_id), images, {"index": index})

    def score_eval_prediction(self, _adapter, prediction, prepared, _ctx):
        self.score_threads.add(threading.get_ident())
        self.assert_cpu_prediction(prediction)
        expected = float(prepared.state["index"])
        if float(prediction.points.reshape(-1)[0]) != expected:
            raise AssertionError("H2D/inference ordering produced stale image data")
        return {"ape": float(prepared.state["index"])}

    def prepare_skip_eval_scene(self, adapter, batch, ctx):
        prepared = self.prepare_eval_scene_cpu(adapter, batch, ctx)
        prepared.images = prepared.images.to(ctx.device)
        return prepared

    @staticmethod
    def assert_cpu_prediction(prediction):
        if prediction.points.is_cuda or prediction.camera_poses.is_cuda:
            raise AssertionError("score prediction was not staged to CPU")

    def evaluate_scene(self, adapter, model, batch, ctx):
        prepared = self.prepare_eval_scene_cpu(adapter, batch, ctx)
        prepared.images = prepared.images.to(ctx.device)
        prediction = adapter.forward_predictions(model, prepared.images)
        cpu_prediction = PredictionBundle(
            points=prediction.points.detach().float().cpu(),
            camera_poses=prediction.camera_poses.detach().float().cpu(),
        )
        metrics = self.score_eval_prediction(adapter, cpu_prediction, prepared, ctx)
        metrics.update({"inference_time_ms": 0.0, "infer_peak_vram_gb": 0.0})
        return metrics


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for async eval tests")
class EvalAsyncScoreTest(unittest.TestCase):
    def test_score_pipeline_cli_defaults_and_overrides(self) -> None:
        required = ["--dataset", "scannet", "--data-root", "data/scans_dev50"]
        defaults = _build_parser().parse_args(required)
        self.assertEqual(defaults.score_workers, DEFAULT_ASYNC_SCORE_WORKERS)
        self.assertEqual(
            defaults.score_max_pending, DEFAULT_ASYNC_SCORE_MAX_PENDING
        )
        configured = _build_parser().parse_args(
            [
                *required,
                "--score-workers",
                "7",
                "--score-max-pending",
                "21",
            ]
        )
        self.assertEqual(configured.score_workers, 7)
        self.assertEqual(configured.score_max_pending, 21)

    def test_scene_pipeline_writes_ordered_equivalent_results(self) -> None:
        adapter = _Adapter()
        model = nn.Identity().cuda().eval()
        main_thread = threading.get_ident()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            async_ds = _Dataset()
            async_dir = evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=async_ds,
                experiment="test",
                dataset="fake",
                data_root=root,
                backbone="fake",
                model_name="async",
                device=torch.device("cuda"),
                output_dir=root / "async",
                max_scenes=5,
                max_frames=1,
                progress=False,
                async_score=True,
                timing=True,
            )
            payload = json.loads((async_dir / "eval_record_v3.json").read_text())
            rows = list(payload["scenes"].values())
            self.assertEqual([row["scene_id"] for row in rows], [f"scene{i}" for i in range(5)])
            self.assertEqual([row["ape"] for row in rows], [float(i) for i in range(5)])
            self.assertGreaterEqual(len(async_ds.score_threads), 1)
            self.assertLessEqual(
                len(async_ds.score_threads), DEFAULT_ASYNC_SCORE_WORKERS
            )
            self.assertNotIn(main_thread, async_ds.score_threads)
            self.assertEqual(len(async_ds.prepare_threads), 1)
            self.assertNotIn(main_thread, async_ds.prepare_threads)
            self.assertTrue(all((async_dir / f"scene{i}" / "metrics.json").is_file() for i in range(5)))
            self.assertEqual(adapter.defer_cpu_geometry_flags, [True] * 5)
            sync_ds = _Dataset()
            sync_dir = evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=sync_ds,
                experiment="test",
                dataset="fake",
                data_root=root,
                backbone="fake",
                model_name="sync",
                device=torch.device("cuda"),
                output_dir=root / "sync",
                max_scenes=5,
                max_frames=1,
                progress=False,
                async_score=False,
            )
            sync_payload = json.loads((sync_dir / "eval_record_v3.json").read_text())
            sync_rows = list(sync_payload["scenes"].values())
            self.assertEqual(
                [(row["scene_id"], row["ape"]) for row in rows],
                [(row["scene_id"], row["ape"]) for row in sync_rows],
            )
            self.assertEqual(
                adapter.defer_cpu_geometry_flags,
                [True] * 5 + [False] * 5,
            )

    def test_preloader_is_single_worker_ordered_pinned_and_bounded(self) -> None:
        worker_threads: set[int] = set()

        def prepare(batch):
            worker_threads.add(threading.get_ident())
            return PreparedSkipEvalScene(
                scene_id=str(batch.scene_id),
                images=torch.ones((1, 2, 3, 4, 4)),
            )

        source = iter(
            [EvalSampleBatch(scene_id=f"s{i}", images=None) for i in range(6)]
        )
        preloader = AsyncScenePreloader(scene_source=source, prepare_fn=prepare)
        items = []
        while True:
            item = preloader.next()
            if item is None:
                break
            items.append(item)
            self.assertLessEqual(preloader.max_lookahead_seen, 1)
            self.assertTrue(item.prepared.images.is_pinned())
            self.assertTrue(item.prepared.images.is_contiguous())
            self.assertEqual(item.prepared.images.dtype, torch.float32)
        preloader.close()
        self.assertEqual([item.batch.scene_id for item in items], [f"s{i}" for i in range(6)])
        self.assertEqual(len(worker_threads), 1)
        self.assertNotIn(threading.get_ident(), worker_threads)

    def test_preloader_reports_prepare_failure_and_continues(self) -> None:
        def prepare(batch):
            if batch.scene_id == "bad":
                raise ValueError("expected prepare failure")
            return PreparedSkipEvalScene(
                scene_id=str(batch.scene_id), images=torch.zeros((1, 1, 3, 2, 2))
            )

        source = iter(
            [
                EvalSampleBatch(scene_id="first", images=None),
                EvalSampleBatch(scene_id="bad", images=None),
                EvalSampleBatch(scene_id="last", images=None),
            ]
        )
        preloader = AsyncScenePreloader(scene_source=source, prepare_fn=prepare)
        first = preloader.next()
        failed = preloader.next()
        last = preloader.next()
        exhausted = preloader.next()
        preloader.close()
        self.assertEqual(first.batch.scene_id, "first")
        self.assertEqual(failed.batch.scene_id, "bad")
        self.assertIsInstance(failed.error, ValueError)
        self.assertEqual(last.batch.scene_id, "last")
        self.assertIsNone(exhausted)

    def test_disabled_preload_and_visualization_use_fallbacks(self) -> None:
        adapter = _Adapter()
        model = nn.Identity().cuda().eval()
        main_thread = threading.get_ident()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            no_preload_ds = _Dataset()
            evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=no_preload_ds,
                experiment="test",
                dataset="fake",
                data_root=root,
                backbone="fake",
                model_name="no_preload",
                device=torch.device("cuda"),
                output_dir=root / "no_preload",
                max_scenes=2,
                max_frames=1,
                progress=False,
                async_score=True,
                scene_preload=False,
            )
            self.assertEqual(no_preload_ds.prepare_threads, {main_thread})
            self.assertNotIn(main_thread, no_preload_ds.score_threads)

            visual_ds = _Dataset()
            evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=visual_ds,
                experiment="test",
                dataset="fake",
                data_root=root,
                backbone="fake",
                model_name="visual",
                device=torch.device("cuda"),
                output_dir=root / "visual",
                max_scenes=2,
                max_frames=1,
                progress=False,
                async_score=True,
                scene_preload=True,
                export_vis_2d=True,
            )
            self.assertEqual(visual_ds.prepare_threads, {main_thread})
            self.assertEqual(visual_ds.score_threads, {main_thread})


if __name__ == "__main__":
    unittest.main()

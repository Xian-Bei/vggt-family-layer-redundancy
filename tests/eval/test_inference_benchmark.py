from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from analysis.backbones.contracts import PredictionBundle
from analysis.datasets.contracts import EvalSampleBatch, PreparedEvalScene
from analysis.eval.inference_benchmark import (
    BenchmarkJob,
    ComputeApp,
    ExclusiveGpuGuard,
    _load_model,
    _validate_variant_compression,
    atomic_merge_result,
    benchmark_config_id,
    benchmark_dataset,
    benchmark_protocol,
    benchmark_result_complete,
    evenly_spaced_indices,
    load_selection_aliases,
    query_compute_apps,
    summarize_samples,
)
from analysis.inference_compression.contracts import InferenceCompressionConfig


class InferenceBenchmarkHelpersTest(unittest.TestCase):
    def test_evenly_spaced_indices_and_short_dataset(self) -> None:
        self.assertEqual(
            evenly_spaced_indices(100, 10),
            [0, 11, 22, 33, 44, 55, 66, 77, 88, 99],
        )
        self.assertEqual(evenly_spaced_indices(10, 10), list(range(10)))
        with self.assertRaisesRegex(ValueError, "requires 10"):
            evenly_spaced_indices(9, 10)

    def test_summary_uses_population_std_and_linear_p95(self) -> None:
        summary = summarize_samples(list(range(10)))
        self.assertEqual(summary["mean"], 4.5)
        self.assertEqual(summary["median"], 4.5)
        self.assertAlmostEqual(summary["std"], 2.8722813232690143)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["max"], 9.0)
        self.assertAlmostEqual(summary["p95"], 8.55)

    def test_config_id_is_stable_and_includes_compression(self) -> None:
        config = InferenceCompressionConfig(
            name="fastvggt",
            fastvggt_merge_start=2,
            fastvggt_merge_ratio=0.75,
        )
        expected = (
            "recovery_fastvggt__i3_j20__"
            "recovery_ghost_interval_sequential"
            "__fastvggt_m2_r0p75"
        )
        self.assertEqual(
            benchmark_config_id(
                variant="recovery_fastvggt",
                selector="i3_j20",
                recovery_plan_name="recovery_ghost_interval_sequential",
                compression_config=config,
            ),
            expected,
        )

    def test_full_config_id_is_stable(self) -> None:
        self.assertEqual(
            benchmark_config_id(
                variant="full",
                selector="full",
                recovery_plan_name="",
                compression_config=InferenceCompressionConfig(name="none"),
            ),
            "full__full",
        )
        self.assertEqual(
            benchmark_config_id(
                variant="full_fastvggt",
                selector="full",
                recovery_plan_name="",
                compression_config=InferenceCompressionConfig(
                    name="fastvggt",
                    fastvggt_merge_start=2,
                    fastvggt_merge_ratio=0.75,
                ),
            ),
            "full_fastvggt__full__fastvggt_m2_r0p75",
        )
        self.assertEqual(
            benchmark_config_id(
                variant="full_quantvggt",
                selector="full",
                recovery_plan_name="",
                compression_config=InferenceCompressionConfig(
                    name="quantvggt",
                    quantvggt_dtype="w4a4",
                ),
            ),
            "full_quantvggt__full__quantvggt_w4a4",
        )

    def test_variant_requires_its_matching_compression(self) -> None:
        _validate_variant_compression(
            "full_fastvggt", InferenceCompressionConfig(name="fastvggt")
        )
        _validate_variant_compression(
            "full_quantvggt", InferenceCompressionConfig(name="quantvggt")
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            _validate_variant_compression(
                "full_fastvggt", InferenceCompressionConfig(name="none")
            )

    def test_full_model_keeps_pretrained_structure(self) -> None:
        class FullAdapter:
            name = "fake"

            @staticmethod
            def default_hf_model() -> str:
                return "fake/full"

            @staticmethod
            def load_pretrained(*, hf_model: str, device: str) -> nn.Module:
                self.assertEqual(hf_model, "fake/full")
                self.assertEqual(device, "cpu")
                return nn.Identity()

        adapter = FullAdapter()
        args = SimpleNamespace(backbone="fake", variant="full")
        compression = InferenceCompressionConfig(name="none")
        with (
            mock.patch(
                "analysis.eval.inference_benchmark.get_backbone",
                return_value=adapter,
            ),
            mock.patch("analysis.eval.inference_benchmark.apply_pythonpath"),
            mock.patch(
                "analysis.eval.inference_benchmark.apply_eval_model_structure"
            ) as apply_structure,
        ):
            loaded_adapter, model = _load_model(
                args,
                torch.device("cpu"),
                compression,
            )

        self.assertIs(loaded_adapter, adapter)
        self.assertFalse(model.training)
        apply_structure.assert_called_once_with(
            model,
            adapter,
            eval_mode="upper",
            prune_plan=None,
            checkpoint=None,
            map_location=torch.device("cpu"),
        )

    def test_compressed_full_model_keeps_pretrained_structure(self) -> None:
        class FullAdapter:
            name = "fake"

            @staticmethod
            def default_hf_model() -> str:
                return "fake/full"

            @staticmethod
            def load_pretrained(*, hf_model: str, device: str) -> nn.Module:
                return nn.Identity()

        class CompressionAdapter:
            def validate_backbone(self, backbone_name: str) -> None:
                self.backbone_name = backbone_name

            def validate_config(self, config: InferenceCompressionConfig) -> None:
                self.config = config

            def apply(self, model, *, backbone_name, config):
                model.compression_applied = (backbone_name, config.name)
                return model

        adapter = FullAdapter()
        compression_adapter = CompressionAdapter()
        args = SimpleNamespace(backbone="fake", variant="full_fastvggt")
        compression = InferenceCompressionConfig(name="fastvggt")
        with (
            mock.patch(
                "analysis.eval.inference_benchmark.get_backbone",
                return_value=adapter,
            ),
            mock.patch("analysis.eval.inference_benchmark.apply_pythonpath"),
            mock.patch(
                "analysis.eval.inference_benchmark.apply_eval_model_structure"
            ) as apply_structure,
            mock.patch(
                "analysis.eval.inference_benchmark.get_inference_compression",
                return_value=compression_adapter,
            ),
        ):
            _loaded_adapter, model = _load_model(
                args,
                torch.device("cpu"),
                compression,
            )

        apply_structure.assert_called_once_with(
            model,
            adapter,
            eval_mode="upper",
            prune_plan=None,
            checkpoint=None,
            map_location=torch.device("cpu"),
        )
        self.assertEqual(model.compression_applied, ("fake", "fastvggt"))

    def test_compute_app_parser(self) -> None:
        output = "GPU-a, 101, python, 2048\nGPU-b, 202, worker, N/A\n"
        apps = query_compute_apps(lambda _command: output)
        self.assertEqual(
            apps,
            [
                ComputeApp("GPU-a", 101, "python", 2048),
                ComputeApp("GPU-b", 202, "worker", None),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            query_compute_apps(lambda _command: "GPU-a, 101\n")

    def test_gpu_guard_accepts_self_and_other_gpu_but_rejects_target_gpu(self) -> None:
        guard = ExclusiveGpuGuard(torch.device("cuda"))
        guard.gpu_uuid = "GPU-a"
        guard.pid = 101
        allowed = [
            ComputeApp("GPU-a", 101, "python", 100),
            ComputeApp("GPU-b", 202, "worker", 100),
        ]
        with mock.patch(
            "analysis.eval.inference_benchmark.query_compute_apps",
            return_value=allowed,
        ):
            guard.check()
        rejected = [*allowed, ComputeApp("GPU-a", 303, "foreign", 200)]
        with mock.patch(
            "analysis.eval.inference_benchmark.query_compute_apps",
            return_value=rejected,
        ):
            with self.assertRaisesRegex(RuntimeError, "pid=303"):
                guard.check()
        with mock.patch(
            "analysis.eval.inference_benchmark.query_compute_apps",
            side_effect=RuntimeError("query failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                guard.check()

    def test_atomic_merge_preserves_other_configs_and_replaces_matching_one(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "inference_benchmark_v3.json"
            atomic_merge_result(
                path,
                backbone="vggt",
                dataset="fake",
                config_id="one",
                result={"value": 1},
            )
            atomic_merge_result(
                path,
                backbone="vggt",
                dataset="fake",
                config_id="two",
                result={"value": 2},
            )
            atomic_merge_result(
                path,
                backbone="vggt",
                dataset="fake",
                config_id="one",
                result={"value": 3},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                payload["results"], {"one": {"value": 3}, "two": {"value": 2}}
            )
            self.assertFalse(any(path.parent.glob(f".{path.name}.*.tmp")))

    def test_selection_aliases_preserve_all_logical_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "selection.tsv"
            path.write_text(
                "i3_j20\t16\tCKA\tCKA_single\t0.1\n"
                "i3_j20\t18\tretention\tretention_double\t0.2\n"
                "i5_j22\t20\tCKA\tCKA_single\t0.3\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_selection_aliases(path, "i3_j20"),
                [
                    {
                        "drop_count": 16,
                        "family": "CKA",
                        "selected_strategy": "CKA_single",
                        "selection_score": 0.1,
                    },
                    {
                        "drop_count": 18,
                        "family": "retention",
                        "selected_strategy": "retention_double",
                        "selection_score": 0.2,
                    },
                ],
            )

    def test_complete_result_is_reused_only_when_inputs_match(self) -> None:
        protocol = benchmark_protocol(
            sample_count=10,
            max_scenes=0,
            max_frames=30,
            gpu_poll_interval_ms=200,
        )
        selection = {
            "datasets": "train_a,train_b",
            "aliases": [
                {
                    "drop_count": 20,
                    "family": "CKA",
                    "selected_strategy": "CKA_single",
                    "selection_score": 0.1,
                }
            ],
        }
        scenes = [
            {
                "latency_ms": float(index + 1),
                "memory_gib": {
                    "baseline_allocated": 1.0,
                    "baseline_reserved": 2.0,
                    "peak_allocated": 3.0,
                    "peak_reserved": 4.0,
                },
            }
            for index in range(10)
        ]
        result = {
            "config_id": "pruning__i3_j22",
            "variant": "pruning",
            "selector": "i3_j22",
            "selection": selection,
            "recovery_plan_name": None,
            "compression": {"name": "none"},
            "hf_model": "fake/model",
            "checkpoint": None,
            "protocol": protocol,
            "scenes": scenes,
            "summary": {
                key: {}
                for key in (
                    "latency_ms",
                    "baseline_allocated_gib",
                    "baseline_reserved_gib",
                    "peak_allocated_gib",
                    "peak_reserved_gib",
                )
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "inference_benchmark_v3.json"
            atomic_merge_result(
                path,
                backbone="vggt",
                dataset="fake",
                config_id="pruning__i3_j22",
                result=result,
            )
            common = {
                "backbone": "vggt",
                "dataset": "fake",
                "config_id": "pruning__i3_j22",
                "variant": "pruning",
                "selector": "i3_j22",
                "selection": selection,
                "recovery_plan_name": None,
                "compression": {"name": "none"},
                "hf_model": "fake/model",
                "checkpoint": None,
            }
            self.assertTrue(
                benchmark_result_complete(path, protocol=protocol, **common)
            )
            changed_protocol = {**protocol, "max_frames": 20}
            self.assertFalse(
                benchmark_result_complete(path, protocol=changed_protocol, **common)
            )
            self.assertFalse(
                benchmark_result_complete(
                    path,
                    protocol=protocol,
                    **{**common, "compression": {"name": "fastvggt"}},
                )
            )

            full_result = {
                **result,
                "config_id": "full__full",
                "variant": "full",
                "selector": "full",
                "selection": None,
            }
            atomic_merge_result(
                path,
                backbone="vggt",
                dataset="fake",
                config_id="full__full",
                result=full_result,
            )
            self.assertTrue(
                benchmark_result_complete(
                    path,
                    backbone="vggt",
                    dataset="fake",
                    config_id="full__full",
                    variant="full",
                    selector="full",
                    selection=None,
                    recovery_plan_name=None,
                    compression={"name": "none"},
                    hf_model="fake/model",
                    checkpoint=None,
                    protocol=protocol,
                )
            )


class _FakeAdapter:
    def __init__(self) -> None:
        self.forward_count = 0

    def forward_predictions(self, _model, images: torch.Tensor) -> PredictionBundle:
        self.forward_count += 1
        return PredictionBundle(points=images.mean(dim=(-1, -2)).contiguous())


class _FakeDataset:
    name = "fake"

    def __init__(
        self, *, mismatch_last: bool = False, variable_frames_last: bool = False
    ) -> None:
        self.mismatch_last = mismatch_last
        self.variable_frames_last = variable_frames_last
        self.score_calls = 0

    def count_scenes(self, _root, *, max_scenes=0, split=None):
        del split
        return min(100, max_scenes) if max_scenes > 0 else 100

    def iter_scenes(self, _root, *, max_scenes=0, max_frames=0, split=None):
        del max_frames, split
        count = min(100, max_scenes) if max_scenes > 0 else 100
        for index in range(count):
            frame_ids = [0] if self.variable_frames_last and index == 99 else [0, 1]
            yield EvalSampleBatch(
                scene_id=f"scene{index}",
                images=None,
                frame_ids=frame_ids,
            )

    def prepare_inference_scene_cpu(self, _adapter, batch):
        index = int(str(batch.scene_id).removeprefix("scene"))
        height = 5 if self.mismatch_last and index == 99 else 4
        images = torch.full(
            (1, len(batch.frame_ids), 3, height, 4), float(index), device="cpu"
        )
        return PreparedEvalScene(scene_id=str(batch.scene_id), images=images)

    def score_eval_prediction(self, *_args, **_kwargs):
        self.score_calls += 1
        raise AssertionError("forward-only benchmark must not score predictions")


class _FakeGuard:
    def __init__(self) -> None:
        self.check_count = 0

    def check(self) -> None:
        self.check_count += 1


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA is required for inference benchmark tests"
)
class InferenceBenchmarkGpuTest(unittest.TestCase):
    def test_ten_scenes_have_one_warmup_and_no_scoring(self) -> None:
        adapter = _FakeAdapter()
        dataset = _FakeDataset()
        guard = _FakeGuard()
        rows = benchmark_dataset(
            adapter=adapter,
            model=nn.Identity().cuda().eval(),
            dataset_plugin=dataset,
            job=BenchmarkJob("fake", Path("."), "fake"),
            device=torch.device("cuda"),
            guard=guard,
            sample_count=10,
            max_frames=2,
        )
        self.assertEqual(adapter.forward_count, 11)
        self.assertEqual(dataset.score_calls, 0)
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["dataset_index"] for row in rows],
            [0, 11, 22, 33, 44, 55, 66, 77, 88, 99],
        )
        self.assertTrue(all(row["latency_ms"] > 0.0 for row in rows))
        self.assertTrue(
            all(
                row["memory_gib"]["peak_allocated"]
                >= row["memory_gib"]["baseline_allocated"]
                for row in rows
            )
        )
        self.assertGreaterEqual(guard.check_count, 22)

    def test_input_signature_mismatch_aborts(self) -> None:
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            benchmark_dataset(
                adapter=_FakeAdapter(),
                model=nn.Identity().cuda().eval(),
                dataset_plugin=_FakeDataset(mismatch_last=True),
                job=BenchmarkJob("fake", Path("."), "fake"),
                device=torch.device("cuda"),
                guard=_FakeGuard(),
                sample_count=10,
                max_frames=2,
            )

    def test_variable_frame_mode_warms_each_sequence_length(self) -> None:
        adapter = _FakeAdapter()
        rows = benchmark_dataset(
            adapter=adapter,
            model=nn.Identity().cuda().eval(),
            dataset_plugin=_FakeDataset(variable_frames_last=True),
            job=BenchmarkJob("fake", Path("."), "fake"),
            device=torch.device("cuda"),
            guard=_FakeGuard(),
            sample_count=10,
            max_frames=2,
            allow_variable_frames=True,
        )
        self.assertEqual(adapter.forward_count, 12)
        self.assertEqual([row["num_frames"] for row in rows[:-1]], [2] * 9)
        self.assertEqual(rows[-1]["num_frames"], 1)


if __name__ == "__main__":
    unittest.main()

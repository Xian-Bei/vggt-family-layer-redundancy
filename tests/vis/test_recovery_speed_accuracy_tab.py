from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from analysis.prune_plan.eval_plots import GT_MEAN_METRICS
from analysis.vis.recovery_speed_accuracy_tab import (
    _BENCHMARK_FILENAME,
    _DATASET_AVERAGE_GROUPS,
    _POINT_STYLES,
    _TABLE_HEADERS,
    _load_benchmark_measurement,
    _load_backbone_tradeoff_points,
    _load_full_accuracy_metrics,
    _load_full_benchmark_measurement,
    _minimum_fastvggt_aliases,
    _render_tradeoff_figure,
    _render_tradeoff_radar_figure,
    _scatter_area,
    _tradeoff_table_rows,
    build_recovery_speed_accuracy_tab,
)


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in GT_MEAN_METRICS}


class RecoverySpeedAccuracyTabTest(unittest.TestCase):
    def test_all_ours_states_are_above_org(self) -> None:
        org_zorder = _POINT_STYLES["upper"]["zorder"]
        self.assertGreater(_POINT_STYLES["before"]["zorder"], org_zorder)
        self.assertGreater(_POINT_STYLES["after"]["zorder"], org_zorder)

    def test_parameter_count_scales_marker_radius(self) -> None:
        small = _scatter_area(1, 2)
        large = _scatter_area(2, 2)
        self.assertAlmostEqual(large / small, 4.0)

    def test_builder_defaults_to_all_backbones_autogroup_and_same_dataset(self) -> None:
        with (
            patch(
                "analysis.vis.recovery_speed_accuracy_tab.discover_recovery_overview_options",
                return_value=(
                    ["vggt", "pi3", "vggt_omega"],
                    ["recovery_ghost_interval_sequential_autogroup"],
                ),
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_recovery_speed_accuracy_tab(
                    blocks=blocks,
                    output_root=Path("/unused"),
                )

        self.assertEqual(ui["backbone_cb"].value, ["vggt", "pi3", "vggt_omega"])
        self.assertEqual(
            ui["method_dd"].value,
            "recovery_ghost_interval_sequential_autogroup",
        )
        self.assertEqual(ui["dataset_dd"].value, "same-dataset testing average")
        self.assertEqual(
            [choice[1] for choice in ui["dataset_dd"].choices],
            [
                "training average",
                "same-dataset testing average",
                "cross-dataset testing average",
            ],
        )
        self.assertNotIn("image", ui)
        self.assertEqual(
            ui["radar_image"].label,
            "Recovery speed–accuracy radar plots",
        )
        self.assertEqual(
            ui["results_table"].label,
            "Full / Recovery / compression measurements",
        )

    def test_table_has_identity_seven_metrics_time_and_vram_columns(self) -> None:
        points = [
            {
                "backbone": "vggt",
                "state": "after",
                "drop_count": 16,
                "selector": "i1_j16",
                "inference_time_ms": 1250.0,
                "peak_allocated_gib": 7.5,
                "raw_metric": _metrics(3.0),
            },
            {
                "backbone": "vggt",
                "state": "upper",
                "drop_count": 0,
                "selector": "upper",
                "inference_time_ms": 2000.0,
                "peak_allocated_gib": 9.0,
                "raw_metric": _metrics(1.0),
            },
            {
                "backbone": "vggt",
                "state": "full_fastvggt",
                "drop_count": 0,
                "selector": "full",
                "inference_time_ms": 1750.0,
                "peak_allocated_gib": 8.5,
                "raw_metric": _metrics(1.5),
            },
            {
                "backbone": "vggt",
                "state": "full_quantvggt",
                "drop_count": 0,
                "selector": "full",
                "inference_time_ms": 1600.0,
                "peak_allocated_gib": 8.25,
                "raw_metric": _metrics(1.75),
            },
            {
                "backbone": "vggt",
                "state": "before",
                "drop_count": 16,
                "selector": "i1_j16",
                "inference_time_ms": 1500.0,
                "peak_allocated_gib": 8.0,
                "raw_metric": _metrics(2.0),
            },
        ]

        rows = _tradeoff_table_rows(points)

        self.assertEqual(len(rows), 5)
        self.assertTrue(all(len(row) == len(_TABLE_HEADERS) == 13 for row in rows))
        self.assertEqual(
            [(row[1], row[2]) for row in rows],
            [
                ("—", "Original upper"),
                ("—", "Full + FastVGGT"),
                ("—", "Full + QuantVGGT"),
                ("16", "Recovery"),
                ("16", "Recovery + FastVGGT"),
            ],
        )
        self.assertEqual(rows[0][3:10], ["1.00"] * 7)
        self.assertEqual(rows[1][3:10], ["1.50"] * 7)
        self.assertEqual(rows[2][3:10], ["1.75"] * 7)
        self.assertEqual(rows[3][-2:], ["1500.00", "8.00"])
        self.assertEqual(rows[4][-2:], ["1250.00", "7.50"])

    def test_table_falls_back_to_synchronized_eval_time_and_vram(self) -> None:
        raw_metric = {
            **_metrics(1.2345),
            "inference_time_ms": 1234.567,
            "infer_peak_vram_gb": 6.789,
        }

        rows = _tradeoff_table_rows(
            [
                {
                    "backbone": "vggt_omega",
                    "state": "after",
                    "drop_count": 20,
                    "selector": "i2_j16_i41_j45",
                    "raw_metric": raw_metric,
                }
            ]
        )

        self.assertEqual(rows[0][3:10], ["1.23"] * 7)
        self.assertEqual(rows[0][-2:], ["1234.57", "6.79"])

    def test_benchmark_measurement_aggregates_raw_scene_latency(self) -> None:
        def result(variant: str, compression: dict) -> dict:
            return {
                "variant": variant,
                "selector": "i1_j20",
                "recovery_plan_name": "recovery_method",
                "selection": {
                    "aliases": [
                        {
                            "drop_count": 20,
                            "selected_strategy": "retention_single",
                        }
                    ]
                },
                "compression": compression,
                "scenes": [],
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for dataset, latency in (("dataset_a", 100.0), ("dataset_b", 200.0)):
                before = result("recovery", {"name": "none"})
                before["scenes"] = [
                    {
                        "latency_ms": latency,
                        "num_frames": 30,
                        "memory_gib": {"peak_allocated": 4.0},
                    },
                    {
                        "latency_ms": latency,
                        "num_frames": 30,
                        "memory_gib": {"peak_allocated": 4.0},
                    },
                ]
                path = (
                    root / dataset / "eval" / "testtime" / "inference_benchmark_v3.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "dataset": dataset,
                            "results": {"before": before},
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.dict(
                _DATASET_AVERAGE_GROUPS,
                {"two datasets": ("dataset_a", "dataset_b")},
            ):
                measurement, warnings = _load_benchmark_measurement(
                    root,
                    recovery_method="recovery_method",
                    dataset="two datasets",
                    drop_count=20,
                    selector="i1_j20",
                    prune_strategy="retention_single",
                    state="before",
                    compression_name="none",
                )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement["inference_time_ms"], 150.0)
        self.assertAlmostEqual(measurement["peak_allocated_gib"], 4.0)
        self.assertEqual(measurement["benchmark_scene_count"], 4)
        self.assertEqual(measurement["benchmark_datasets"], ["dataset_a", "dataset_b"])

    def test_minimum_fastvggt_aliases_requires_both_families(self) -> None:
        def result(selector: str, aliases: list[dict]) -> dict:
            return {
                "variant": "recovery_fastvggt",
                "recovery_plan_name": "recovery_method",
                "selector": selector,
                "selection": {"aliases": aliases},
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = (
                root / "dataset" / "eval" / "testtime" / "inference_benchmark_v3.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "results": {
                            "drop14_cka_only": result(
                                "i1_j14",
                                [
                                    {
                                        "drop_count": 14,
                                        "family": "CKA",
                                        "selected_strategy": "CKA_single",
                                    }
                                ],
                            ),
                            "drop16_cka": result(
                                "i1_j16",
                                [
                                    {
                                        "drop_count": 16,
                                        "family": "CKA",
                                        "selected_strategy": "CKA_single",
                                    }
                                ],
                            ),
                            "drop16_retention": result(
                                "i2_j16",
                                [
                                    {
                                        "drop_count": 16,
                                        "family": "retention",
                                        "selected_strategy": "retention_double",
                                    }
                                ],
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            drop_count, aliases = _minimum_fastvggt_aliases(
                root,
                recovery_method="recovery_method",
            )

        self.assertEqual(drop_count, 16)
        self.assertEqual(
            aliases,
            {
                "CKA": {
                    "selected_strategy": "CKA_single",
                    "selector": "i1_j16",
                },
                "retention": {
                    "selected_strategy": "retention_double",
                    "selector": "i2_j16",
                },
            },
        )

    def test_full_benchmark_measurement_uses_exclusive_gpu_scene_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            records = {
                "dataset_a": [(100.0, 4.0), (300.0, 6.0)],
                "dataset_b": [(500.0, 7.0), (700.0, 9.0)],
            }
            for dataset, values in records.items():
                path = (
                    root / dataset / "eval" / "testtime" / "inference_benchmark_v3.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "dataset": dataset,
                            "results": {
                                variant: {
                                    "variant": "full",
                                    "selector": "full",
                                    "selection": None,
                                    "recovery_plan_name": None,
                                    "compression": {"name": "none"},
                                    "checkpoint": None,
                                    "scenes": [
                                        {
                                            "latency_ms": inference_time_ms,
                                            "num_frames": 30,
                                            "memory_gib": {
                                                "peak_allocated": peak_vram_gib
                                            },
                                        }
                                        for inference_time_ms, peak_vram_gib in values
                                    ],
                                }
                                for variant in ("full__full",)
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.dict(
                _DATASET_AVERAGE_GROUPS,
                {"two datasets": ("dataset_a", "dataset_b")},
            ):
                measurement, warnings = _load_full_benchmark_measurement(
                    root,
                    dataset="two datasets",
                )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement["inference_time_ms"], 400.0)
        self.assertAlmostEqual(measurement["peak_allocated_gib"], 6.5)
        self.assertEqual(measurement["benchmark_scene_count"], 4)
        self.assertEqual(measurement["benchmark_datasets"], ["dataset_a", "dataset_b"])
        self.assertEqual(measurement["timing_source"], "exclusive_gpu_benchmark")

    def test_full_compressed_benchmark_measurement_matches_exact_variant(self) -> None:
        def result(variant: str, compression: str, latency_ms: float) -> dict:
            return {
                "variant": variant,
                "selector": "full",
                "selection": None,
                "recovery_plan_name": None,
                "compression": {"name": compression},
                "checkpoint": None,
                "scenes": [
                    {
                        "latency_ms": latency_ms,
                        "num_frames": 30,
                        "memory_gib": {"peak_allocated": latency_ms / 100.0},
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = root / "dataset" / "eval" / "testtime" / _BENCHMARK_FILENAME
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "dataset",
                        "results": {
                            "full": result("full", "none", 300.0),
                            "fast": result(
                                "full_fastvggt", "fastvggt", 200.0
                            ),
                            "quant": result(
                                "full_quantvggt", "quantvggt", 100.0
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            fast, fast_warnings = _load_full_benchmark_measurement(
                root,
                dataset="dataset",
                variant="full_fastvggt",
            )
            quant, quant_warnings = _load_full_benchmark_measurement(
                root,
                dataset="dataset",
                variant="full_quantvggt",
            )

        self.assertEqual(fast_warnings, [])
        self.assertEqual(quant_warnings, [])
        self.assertIsNotNone(fast)
        self.assertIsNotNone(quant)
        assert fast is not None and quant is not None
        self.assertEqual(fast["inference_time_ms"], 200.0)
        self.assertEqual(fast["peak_allocated_gib"], 2.0)
        self.assertEqual(quant["inference_time_ms"], 100.0)
        self.assertEqual(quant["peak_allocated_gib"], 1.0)

    def test_full_compressed_accuracy_uses_matching_upper_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for variant, compression, tag, value in (
                (
                    "full_fastvggt",
                    {"name": "fastvggt", "merge_start": 0, "merge_ratio": 0.9},
                    "fastvggt_m0_r0p9",
                    1.5,
                ),
                (
                    "full_quantvggt",
                    {"name": "quantvggt", "dtype": "w4a4"},
                    "quantvggt_w4a4",
                    1.75,
                ),
            ):
                path = (
                    root
                    / "dataset"
                    / "eval"
                    / f"upper_{tag}"
                    / "eval_record_v3.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "v3",
                            "dataset": "dataset",
                            "average": _metrics(value),
                            "extra": {
                                "eval_mode": "upper",
                                "inference_compression": compression,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                metrics, warnings = _load_full_accuracy_metrics(
                    root,
                    dataset="dataset",
                    variant=variant,
                    compression=compression,
                )
                self.assertEqual(warnings, [])
                self.assertIsNotNone(metrics)
                assert metrics is not None
                self.assertEqual(metrics, _metrics(value))

    def test_loader_keeps_org_and_all_fastvggt_pairs(self) -> None:
        method = "recovery_ghost_interval_sequential"
        recovery_points = {
            str(drop): {
                "available": True,
                "metrics_mean": _metrics(float(drop)),
            }
            for drop in (16, 18, 20)
        }
        recovery_payload = {
            "dataset_average": {"strategies": {method: {"points": recovery_points}}}
        }
        testtime_points = [
            {
                "compression": "fastvggt_m0_r0p9",
                "drop_count": drop,
                "selector": selector,
                "raw_metric": _metrics(float(drop) + 0.5),
            }
            for drop, selector in ((20, "i1_j20"), (16, "i1_j16"), (18, "i1_j18"))
        ]
        testtime_points.append(
            {
                "compression": "quantvggt_w4a4",
                "drop_count": 16,
                "selector": "i1_j16",
                "raw_metric": _metrics(99.0),
            }
        )

        def load_testtime(_root, *, prune_strategy, **_kwargs):
            return (
                (testtime_points, [])
                if prune_strategy == "retention_single"
                else ([], [])
            )

        def load_benchmark(_root, *, state, **_kwargs):
            return (
                {
                    "inference_time_ms": 1500.0 if state == "before" else 1250.0,
                    "peak_allocated_gib": 8.0,
                    "benchmark_scene_count": 40,
                    "benchmark_datasets": ["a", "b", "c", "d"],
                },
                [],
            )

        def load_full(_root, *, variant="full", **_kwargs):
            timings = {
                "full": (2000.0, 9.0),
                "full_fastvggt": (1750.0, 8.5),
                "full_quantvggt": (1600.0, 8.25),
            }
            compressions = {
                "full": {"name": "none"},
                "full_fastvggt": {
                    "name": "fastvggt",
                    "merge_start": 0,
                    "merge_ratio": 0.9,
                },
                "full_quantvggt": {"name": "quantvggt", "dtype": "w4a4"},
            }
            inference_time_ms, peak_allocated_gib = timings[variant]
            return (
                {
                    "inference_time_ms": inference_time_ms,
                    "peak_allocated_gib": peak_allocated_gib,
                    "benchmark_scene_count": 40,
                    "benchmark_datasets": ["a", "b", "c", "d"],
                    "timing_source": "exclusive_gpu_benchmark",
                    "compression": compressions[variant],
                },
                [],
            )

        def load_full_accuracy(_root, *, variant, **_kwargs):
            values = {"full_fastvggt": 1.5, "full_quantvggt": 1.75}
            return _metrics(values[variant]), []

        with (
            patch(
                "analysis.vis.recovery_speed_accuracy_tab.load_prune_plan_bundle",
                return_value={"num_units": 24},
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab.collect_pruning_eval_payload",
                return_value={"dataset_average": {"upper_average": _metrics(1.0)}},
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab.collect_recovery_eval_payload",
                return_value=recovery_payload,
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab._load_testtime_points",
                side_effect=load_testtime,
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab._filter_testtime_to_family_winners",
                side_effect=lambda payload, **_kwargs: (payload, []),
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab.compute_parameter_series_for_backbone",
                return_value=(
                    {
                        0: (1_000_000_000, 600_000_000),
                        16: (800_000_000, 400_000_000),
                        18: (750_000_000, 350_000_000),
                        20: (700_000_000, 300_000_000),
                    },
                    [],
                ),
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab._load_benchmark_measurement",
                side_effect=load_benchmark,
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab._load_full_benchmark_measurement",
                side_effect=load_full,
            ),
            patch(
                "analysis.vis.recovery_speed_accuracy_tab._load_full_accuracy_metrics",
                side_effect=load_full_accuracy,
            ),
        ):
            points, messages = _load_backbone_tradeoff_points(
                Path("/unused"),
                backbone="vggt",
                recovery_method=method,
                dataset="mean",
            )

        self.assertEqual(len(points), 9)
        self.assertEqual([point["state"] for point in points].count("upper"), 1)
        self.assertEqual(
            [point["state"] for point in points].count("full_fastvggt"), 1
        )
        self.assertEqual(
            [point["state"] for point in points].count("full_quantvggt"), 1
        )
        self.assertEqual([point["state"] for point in points].count("before"), 3)
        self.assertEqual([point["state"] for point in points].count("after"), 3)
        self.assertEqual(points[0]["inference_time_ms"], 2000.0)
        self.assertEqual(points[1]["inference_time_ms"], 1750.0)
        self.assertEqual(points[2]["inference_time_ms"], 1600.0)
        self.assertEqual(points[1]["raw_metric"], _metrics(1.5))
        self.assertEqual(points[2]["raw_metric"], _metrics(1.75))
        self.assertEqual(
            {
                point["inference_time_ms"]
                for point in points
                if point["state"] == "before"
            },
            {1500.0},
        )
        self.assertEqual(
            {
                point["inference_time_ms"]
                for point in points
                if point["state"] == "after"
            },
            {1250.0},
        )
        self.assertNotIn(
            99.0, [point["raw_metric"].get("ape") for point in points]
        )
        for drop in (16, 18):
            pair = [point for point in points if point["drop_count"] == drop]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["param_count"], pair[1]["param_count"])
        self.assertTrue(any(point["drop_count"] == 20 for point in points))
        self.assertIn("3 Retention interval pair(s)", messages[-1])

    def test_figure_has_one_row_of_seven_and_reuses_x_and_size(self) -> None:
        points = [
            {
                "backbone": "vggt",
                "state": "upper",
                "drop_count": 0,
                "inference_time_ms": 2000.0,
                "peak_allocated_gib": 9.0,
                "param_count": 1_000_000_000,
                "raw_metric": _metrics(1.0),
            },
            {
                "backbone": "vggt",
                "state": "before",
                "drop_count": 16,
                "inference_time_ms": 1500.0,
                "peak_allocated_gib": 8.0,
                "param_count": 800_000_000,
                "raw_metric": _metrics(2.0),
            },
            {
                "backbone": "vggt",
                "state": "after",
                "drop_count": 16,
                "inference_time_ms": 1250.0,
                "param_count": 800_000_000,
                "raw_metric": _metrics(3.0),
            },
        ]
        fig = _render_tradeoff_figure(points)
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(len(fig.axes), 7)
        for axis in fig.axes:
            actual = [
                collection
                for collection in axis.collections
                if len(collection.get_offsets())
            ]
            self.assertEqual(len(actual), 3)
            self.assertEqual(
                {float(collection.get_offsets()[0, 0]) for collection in actual},
                {1250.0, 1500.0, 2000.0},
            )
            self.assertEqual(
                {collection.get_hatch() for collection in actual},
                {None, "//////", "//"},
            )
            self.assertEqual(
                {
                    tuple(map(tuple, collection.get_paths()[0].vertices))
                    for collection in actual
                },
                {tuple(map(tuple, actual[0].get_paths()[0].vertices))},
            )
            self.assertEqual(len(axis.lines), 0)
            self.assertEqual(len(axis.texts), 0)
            self.assertEqual(axis.get_ylim()[0], 0.0)
            self.assertEqual(axis.get_ylabel(), "")
            self.assertEqual(axis.get_xlabel(), "Inference time (ms)")
            self.assertEqual(axis.get_box_aspect(), 1.0)
        first_sizes = [
            float(collection.get_sizes()[0])
            for collection in fig.axes[0].collections
            if len(collection.get_offsets())
        ]
        for axis in fig.axes[1:]:
            self.assertEqual(
                [
                    float(collection.get_sizes()[0])
                    for collection in axis.collections
                    if len(collection.get_offsets())
                ],
                first_sizes,
            )
        fig.canvas.draw()
        self.assertEqual(len(fig.legends), 1)
        legend_bounds = fig.legends[0].get_window_extent()
        canvas_width, canvas_height = fig.canvas.get_width_height()
        self.assertGreaterEqual(legend_bounds.x0, 0.0)
        self.assertGreaterEqual(legend_bounds.y0, 0.0)
        self.assertLessEqual(legend_bounds.x1, float(canvas_width))
        self.assertLessEqual(legend_bounds.y1, float(canvas_height))
        self.assertIn("Ours", [text.get_text() for text in fig.legends[0].texts])
        self.assertIn("Org", [text.get_text() for text in fig.legends[0].texts])
        self.assertIn(
            "Ours(FastVGGT)",
            [text.get_text() for text in fig.legends[0].texts],
        )
        plt.close(fig)

    def test_radar_only_draws_full_models_with_shared_scale(self) -> None:
        points = [
            {
                "backbone": "vggt",
                "state": "upper",
                "drop_count": 0,
                "selector": "upper",
                "inference_time_ms": 2000.0,
                "peak_allocated_gib": 9.0,
                "param_count": 1_000_000_000,
                "raw_metric": _metrics(1.0),
            },
            {
                "backbone": "vggt",
                "state": "before",
                "radar_family": "CKA",
                "drop_count": 16,
                "selector": "i1_j16",
                "inference_time_ms": 1500.0,
                "peak_allocated_gib": 8.0,
                "param_count": 800_000_000,
                "raw_metric": _metrics(2.0),
            },
            {
                "backbone": "vggt",
                "state": "before",
                "radar_family": "retention",
                "drop_count": 16,
                "selector": "i1_j16",
                "inference_time_ms": 1200.0,
                "peak_allocated_gib": 7.0,
                "param_count": 800_000_000,
                "raw_metric": _metrics(3.0),
            },
        ]
        fig = _render_tradeoff_radar_figure(points)
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(len(fig.axes), 1)
        axis = fig.axes[0]
        axis_text = [text.get_text() for text in axis.get_xticklabels()]
        self.assertEqual(
            axis_text,
            [
                "APE↓ (m)",
                "ARE↓ (°)",
                "RPEₜ↓ (m)",
                "RPEᵣ↓ (°)",
                "Acc.↓ (m)",
                "Comp.↓ (m)",
                "CD↓ (m)",
                "Inference\nTime (ms) ↓",
                "Peak VRAM\n(GiB) ↓",
                "Parameters (M) ↓",
            ],
        )
        self.assertTrue(
            all(label.get_fontweight() == "bold" for label in axis.get_xticklabels())
        )
        data_lines = [line for line in axis.lines if line.get_label() == "VGGT"]
        self.assertEqual(len(data_lines), 1)
        org_scores = list(data_lines[0].get_ydata())[:-1]
        self.assertEqual(org_scores, [0.5] * 10)
        self.assertEqual(axis.get_ylim(), (0.0, 1.0))
        self.assertEqual(axis.get_yticks().tolist(), [0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertTrue(all(label.get_text() == "" for label in axis.get_yticklabels()))
        fig.canvas.draw()
        self.assertEqual(len(fig.legends), 2)
        self.assertEqual(
            [text.get_text() for text in fig.legends[0].texts],
            ["VGGT"],
        )
        self.assertTrue(
            all(text.get_fontweight() == "bold" for text in fig.legends[0].texts)
        )
        self.assertEqual(
            [text.get_text() for text in fig.legends[1].texts],
            ["Estimation metrics", "Point-cloud metrics", "Efficiency metrics"],
        )
        legend_bounds = fig.legends[0].get_window_extent()
        canvas_width, canvas_height = fig.canvas.get_width_height()
        self.assertGreaterEqual(legend_bounds.x0, 0.0)
        self.assertGreaterEqual(legend_bounds.y0, 0.0)
        self.assertLessEqual(legend_bounds.x1, float(canvas_width))
        self.assertLessEqual(legend_bounds.y1, float(canvas_height))
        plt.close(fig)

    def test_radar_overlays_all_backbones_in_one_panel(self) -> None:
        points = []
        for backbone, value in (("vggt", 1.0), ("pi3", 2.0), ("vggt_omega", 4.0)):
            points.append(
                {
                    "backbone": backbone,
                    "state": "upper",
                    "inference_time_ms": value,
                    "peak_allocated_gib": value,
                    "param_count": value,
                    "raw_metric": _metrics(value),
                }
            )

        fig = _render_tradeoff_radar_figure(points)
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_title(), "")
        data_lines = [
            line
            for line in fig.axes[0].lines
            if line.get_label() in {"VGGT", "π³", "VGGT-Ω"}
        ]
        self.assertEqual(len(data_lines), 3)
        first_scores = list(data_lines[0].get_ydata())[:-1]
        self.assertEqual(first_scores, [0.5] * 10)
        middle_scores = list(data_lines[1].get_ydata())[:-1]
        self.assertTrue(all(abs(score - (19.0 / 30.0)) < 1e-12 for score in middle_scores))
        worst_scores = list(data_lines[2].get_ydata())[:-1]
        np.testing.assert_allclose(worst_scores, [0.9] * 10)
        self.assertEqual(
            [text.get_text() for text in fig.legends[0].texts],
            ["VGGT", "π³", "VGGT-Ω"],
        )
        self.assertTrue(
            all(text.get_fontweight() == "bold" for text in fig.legends[0].texts)
        )
        self.assertEqual(
            [text.get_text() for text in fig.legends[1].texts],
            ["Estimation metrics", "Point-cloud metrics", "Efficiency metrics"],
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

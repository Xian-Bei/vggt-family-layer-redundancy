from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from analysis.eval.summary_v3 import METRIC_HEADERS
from analysis.schema.v3 import EvalRecordV3, UNIFIED_METRIC_KEYS
from analysis.vis.summary_tab import (
    SummaryRow,
    _add_dataset_average_rows,
    build_summary_tab,
    _compression_is_visible,
    _default_recovery_methods,
    _default_span_values,
    _format_mode_cell,
    _load_live_eval_rows,
    _latex_escape_cell,
    _mode_group_for_row,
    _mode_sort_key,
    _pruning_sources_for_mode,
    _pruning_structures_for_sources,
    _recovery_method_for_row,
    _render_summary_html,
    _select_best_recovery_rows,
    _span_value_for_row,
)


class SummaryTabIntervalDisplayTest(unittest.TestCase):
    def test_interval_mode_displays_drop_count_with_strategy_tooltip(self) -> None:
        cell = _format_mode_cell(
            "i5_j16_recovery_ghost",
            ("CKA_single", "retention_double"),
        )

        self.assertIn(">12</abbr>", cell)
        self.assertIn('title="CKA_single, retention_double"', cell)
        self.assertIn("recovery_ghost", cell)
        self.assertNotIn("i5_j16", cell)

    def test_multiple_intervals_count_unique_dropped_layers(self) -> None:
        self.assertEqual(_span_value_for_row("i2_j4_i8_j10"), "6")
        self.assertEqual(_span_value_for_row("i2_j5_i4_j7"), "6")

    def test_non_interval_mode_is_unchanged(self) -> None:
        self.assertEqual(_format_mode_cell("upper"), "upper")
        self.assertEqual(_span_value_for_row("prune_span16/recovery_ghost"), "16")

    def test_summary_table_uses_drop_count_and_hides_interval(self) -> None:
        rendered = _render_summary_html(
            ["Dataset", "Backbone", "Mode", "APE ↓"],
            [
                SummaryRow(
                    dataset="example",
                    backbone="vggt",
                    mode="i2_j4_i8_j10",
                    metrics={"APE ↓": "1.0"},
                    pruning_sources=("CKA_double",),
                )
            ],
            metric_headers=["APE ↓"],
        )

        self.assertIn("Dropped layers / Mode", rendered)
        self.assertIn(">6</abbr>", rendered)
        self.assertIn('title="CKA_double"', rendered)
        self.assertNotIn("i2_j4_i8_j10", rendered)

    def test_summary_table_renders_rpe_headers_with_html_subscripts(self) -> None:
        rendered = _render_summary_html(
            ["Dataset", "Backbone", "Mode", "RPEₜ↓ (m)", "RPEᵣ↓ (°)"],
            [
                SummaryRow(
                    dataset="example",
                    backbone="vggt",
                    mode="upper",
                    metrics={"RPEₜ↓ (m)": "1.0", "RPEᵣ↓ (°)": "2.0"},
                )
            ],
            metric_headers=["RPEₜ↓ (m)", "RPEᵣ↓ (°)"],
        )

        self.assertIn("<th>RPE<sub>t</sub>↓ (m)</th>", rendered)
        self.assertIn("<th>RPE<sub>r</sub>↓ (°)</th>", rendered)
        self.assertNotIn("RPEₜ", rendered)
        self.assertNotIn("RPEᵣ", rendered)

    def test_summary_table_rows_are_selectable_and_embed_latex(self) -> None:
        rendered = _render_summary_html(
            ["Dataset", "Backbone", "Mode", "APE ↓"],
            [
                SummaryRow(
                    dataset="same-dataset testing average",
                    backbone="vggt_omega",
                    mode="i3_j20_recovery_ghost_interval_sequential_autogroup",
                    metrics={"APE ↓": "1.234"},
                )
            ],
            metric_headers=["APE ↓"],
        )

        self.assertIn('class="summary-row-select"', rendered)
        self.assertIn("Same-Dataset<br>Generalization", rendered)
        self.assertIn("1.23", rendered)
        self.assertIn(r"\\", rendered)
        data_latex = rendered.split('data-latex="', 1)[1].split('"', 1)[0]
        self.assertNotIn("Same-Dataset", data_latex)
        self.assertNotIn("vggt", data_latex)
        self.assertNotIn("recovery", data_latex)
        self.assertEqual(data_latex, r"1.23 \\")

    def test_latex_cell_escapes_reserved_characters(self) -> None:
        self.assertEqual(
            _latex_escape_cell("a_b & 10%"),
            r"a\_b \& 10\%",
        )

    def test_resource_columns_put_units_in_headers_and_use_one_decimal(self) -> None:
        rendered = _render_summary_html(
            [
                "Dataset",
                "Backbone",
                "Mode",
                "Block params (M) ↓",
                "Latency (ms) ↓",
                "Peak VRAM (GiB) ↓",
            ],
            [
                SummaryRow(
                    "example",
                    "vggt",
                    "upper",
                    {
                        "Block params (M) ↓": "604.729344",
                        "Latency (ms) ↓": "110.04",
                        "Peak VRAM (GiB) ↓": "9.04",
                    },
                )
            ],
            metric_headers=[
                "Block params (M) ↓",
                "Latency (ms) ↓",
                "Peak VRAM (GiB) ↓",
            ],
        )

        self.assertIn("Block params (M) ↓", rendered)
        self.assertIn("Latency (ms) ↓", rendered)
        self.assertIn("Peak VRAM (GiB) ↓", rendered)
        self.assertIn(">604.7</td>", rendered)
        self.assertIn(">110.0</td>", rendered)
        self.assertIn(">9.0</td>", rendered)
        self.assertNotIn("Inference time", rendered)

    def test_interval_modes_sort_by_dropped_layer_count(self) -> None:
        modes = [
            "i1_j20_recovery_ghost",
            "i5_j16_recovery_ghost",
            "upper",
            "i4_j17_recovery_ghost",
            "prune_span8/recovery_ghost",
        ]

        self.assertEqual(
            sorted(modes, key=_mode_sort_key),
            [
                "upper",
                "prune_span8/recovery_ghost",
                "i5_j16_recovery_ghost",
                "i4_j17_recovery_ghost",
                "i1_j20_recovery_ghost",
            ],
        )

    def test_recovery_method_is_separate_from_mode_group(self) -> None:
        modes = (
            "i3_j20_recovery_ghost_interval_sequential",
            "i3_j20_recovery_ghost_interval_sequential+fastvggt_m0_r0p9",
            "prune_span18/recovery_ghost",
        )

        self.assertEqual(
            [_mode_group_for_row(mode) for mode in modes],
            ["recovery", "recovery", "recovery"],
        )
        self.assertEqual(
            [_recovery_method_for_row(mode) for mode in modes],
            ["ghost_interval_sequential", "ghost_interval_sequential", "ghost"],
        )
        self.assertEqual(_mode_group_for_row("upper+fastvggt_m0_r0p9"), "upper")
        self.assertEqual(_mode_group_for_row("i3_j20"), "lower")

    def test_pruning_sources_follow_selector_for_recovery_and_compression(self) -> None:
        source_map = {
            "i3_j20": ("CKA_single", "retention_single"),
        }

        self.assertEqual(
            _pruning_sources_for_mode(
                "i3_j20_recovery_ghost_interval_sequential"
                "+fastvggt_m0_r0p9",
                source_map,
            ),
            ("CKA_single", "retention_single"),
        )

    def test_pruning_structures_are_derived_from_source_suffixes(self) -> None:
        self.assertEqual(
            _pruning_structures_for_sources(
                ("CKA_single", "retention_double", "retention_single")
            ),
            ("single", "double"),
        )
        self.assertEqual(_pruning_structures_for_sources(()), ())

    def test_only_current_target_recovery_method_is_selected_by_default(self) -> None:
        self.assertEqual(
            _default_recovery_methods(
                ["ghost", "ghost_interval_sequential", "linearpatch"]
            ),
            ["ghost_interval_sequential"],
        )
        self.assertEqual(_default_recovery_methods(["ghost"]), [])

    def test_drop_counts_default_to_twelve_and_above(self) -> None:
        self.assertEqual(
            _default_span_values(["4", "8", "12", "16", "24"]),
            ["12", "16", "24"],
        )

    def test_compressed_rows_require_their_checkbox(self) -> None:
        self.assertTrue(_compression_is_visible("upper", []))
        self.assertFalse(
            _compression_is_visible("upper+fastvggt_m0_r0p9", [])
        )
        self.assertTrue(
            _compression_is_visible(
                "upper+fastvggt_m0_r0p9",
                ["fastvggt"],
            )
        )
        self.assertFalse(
            _compression_is_visible(
                "i3_j20_recovery_ghost+quantvggt_w4a4",
                ["fastvggt"],
            )
        )
        self.assertTrue(
            _compression_is_visible(
                "i3_j20_recovery_ghost+quantvggt_w4a4",
                ["quantvggt"],
            )
        )

    def test_dataset_groups_replace_legacy_mean(self) -> None:
        rows = [
            SummaryRow("mean", "vggt", "upper", {"APE ↓": "99"}),
            SummaryRow("scannet_dev50", "vggt", "upper", {"APE ↓": "1"}),
            SummaryRow("nuscenes_train", "vggt", "upper", {"APE ↓": "3"}),
            SummaryRow("scannet_test50", "vggt", "upper", {"APE ↓": "5"}),
            SummaryRow("nuscenes_val", "vggt", "upper", {"APE ↓": "7"}),
            SummaryRow("7scenes_test", "vggt", "upper", {"APE ↓": "9"}),
            SummaryRow("waymo_validation", "vggt", "upper", {"APE ↓": "11"}),
        ]

        result = _add_dataset_average_rows(rows, ["APE ↓"])
        averages = {
            row.dataset: row.metrics["APE ↓"]
            for row in result
            if row.dataset.endswith("average")
        }

        self.assertNotIn("mean", {row.dataset for row in result})
        self.assertEqual(
            averages,
            {
                "training average": "2.00",
                "same-dataset testing average": "6.00",
                "cross-dataset testing average": "10.00",
            },
        )

    def test_dataset_group_uses_publication_label_in_table(self) -> None:
        rendered = _render_summary_html(
            ["Dataset", "Backbone", "Mode", "APE ↓"],
            [
                SummaryRow(
                    dataset="same-dataset testing average",
                    backbone="vggt",
                    mode="upper",
                    metrics={"APE ↓": "1.0"},
                )
            ],
            metric_headers=["APE ↓"],
        )

        self.assertIn("Same-Dataset<br>Generalization", rendered)
        self.assertNotIn("same-dataset testing average", rendered)

    def test_dataset_averages_accept_final_detail_groups(self) -> None:
        rows = [
            SummaryRow(dataset, "vggt", "upper", {"APE ↓": str(value)})
            for dataset, value in (
                ("scannet_dev50", 1),
                ("7scenes_train", 3),
                ("nuscenes_train", 5),
                ("waymo_training", 7),
                ("scannet_test50", 2),
                ("7scenes_test", 4),
                ("nuscenes_val", 6),
                ("waymo_validation", 8),
            )
        ]
        groups = {
            "training average": (
                "scannet_dev50",
                "7scenes_train",
                "nuscenes_train",
                "waymo_training",
            ),
            "four-dataset testing average": (
                "7scenes_test",
                "scannet_test50",
                "nuscenes_val",
                "waymo_validation",
            ),
            "two-dataset testing average": (
                "scannet_test50",
                "nuscenes_val",
            ),
        }

        result = _add_dataset_average_rows(rows, ["APE ↓"], groups)
        averages = {
            row.dataset: row.metrics["APE ↓"]
            for row in result
            if row.dataset in groups
        }

        self.assertEqual(
            averages,
            {
                "training average": "4.00",
                "four-dataset testing average": "5.00",
                "two-dataset testing average": "4.00",
            },
        )

    def test_recovery_rows_choose_one_whole_four_pose_sum_point(self) -> None:
        headers = (
            "APE↓ (m)",
            "ARE↓ (°)",
            "RPEₜ↓ (m)",
            "RPEᵣ↓ (°)",
            "Acc.↓ (m)",
            "Comp.↓ (m)",
            "CD↓ (m)",
        )

        def row(
            mode: str,
            values: tuple[float, ...],
            sources=(),
            dataset: str = "training average",
        ) -> SummaryRow:
            return SummaryRow(
                dataset,
                "vggt",
                mode,
                {
                    header: str(value)
                    for header, value in zip(headers, values)
                },
                tuple(sources),
            )

        rows = [
            row("upper", (0, 0, 0, 0, 0, 0, 0)),
            row("i1_j2", (100, 1, 1, 1, 1, 1, 1), ("CKA_single",)),
            row("i3_j4", (100, 1, 1, 1, 1, 1, 1), ("CKA_double",)),
            row("i5_j6", (100, 1, 1, 1, 1, 1, 1), ("retention_single",)),
            row("i7_j8", (100, 1, 1, 1, 1, 1, 1), ("retention_double",)),
            # The raw seven-metric mean determines the shared Single/Double choice.
            row(
                "i1_j2_recovery_ghost",
                (50, .2, .2, .2, .2, .2, .2),
                ("CKA_single",),
            ),
            row(
                "i3_j4_recovery_ghost",
                (10, .3, .3, .3, .3, .3, .3),
                ("CKA_double",),
            ),
            row(
                "i5_j6_recovery_ghost",
                (60, .4, .4, .4, .4, .4, .4),
                ("retention_single",),
            ),
            row(
                "i7_j8_recovery_ghost",
                (70, .5, .5, .5, .5, .5, .5),
                ("retention_double",),
            ),
            # Held-out scores favor the opposite structures. They must not
            # alter the winners selected by the calibration average above.
            row("upper", (0, 0, 0, 0, 0, 0, 0), dataset="scannet_test50"),
            row("i1_j2", (90, .9, .9, .9, .9, .9, .9), ("CKA_single",), "scannet_test50"),
            row("i3_j4", (1, .1, .1, .1, .1, .1, .1), ("CKA_double",), "scannet_test50"),
            row("i5_j6", (90, .9, .9, .9, .9, .9, .9), ("retention_single",), "scannet_test50"),
            row("i7_j8", (1, .1, .1, .1, .1, .1, .1), ("retention_double",), "scannet_test50"),
            row("i1_j2_recovery_ghost", (90, .9, .9, .9, .9, .9, .9), ("CKA_single",), "scannet_test50"),
            row("i3_j4_recovery_ghost", (1, .1, .1, .1, .1, .1, .1), ("CKA_double",), "scannet_test50"),
            row("i5_j6_recovery_ghost", (90, .9, .9, .9, .9, .9, .9), ("retention_single",), "scannet_test50"),
            row("i7_j8_recovery_ghost", (1, .1, .1, .1, .1, .1, .1), ("retention_double",), "scannet_test50"),
        ]

        selected = _select_best_recovery_rows(rows)
        recovery = [
            item for item in selected
            if item.dataset == "scannet_test50"
            and _recovery_method_for_row(item.mode)
        ]
        lower = [
            item for item in selected
            if item.dataset == "scannet_test50"
            and _mode_group_for_row(item.mode) == "lower"
        ]

        self.assertEqual(
            [item.mode for item in recovery],
            ["i3_j4_recovery_ghost", "i5_j6_recovery_ghost"],
        )
        self.assertEqual([item.mode for item in lower], ["i1_j2", "i5_j6"])
        self.assertEqual(lower[0].pruning_sources, ("CKA_single",))

    def test_autogroup_method_display_uses_last_two_name_parts(self) -> None:
        cell = _format_mode_cell(
            "i3_j18_recovery_ghost_interval_sequential_autogroup",
            ("retention_single",),
        )

        self.assertIn("sequential_autogroup", cell)
        self.assertNotIn("ghost_interval", cell)
        slash_cell = _format_mode_cell(
            "prune_span16/recovery_ghost_interval_sequential_autogroup"
        )
        self.assertEqual(slash_cell, "16 · sequential_autogroup")

    def test_final_mode_cell_shows_unrecovered_replaceme_explicitly(self) -> None:
        cell = _format_mode_cell(
            "i8_j17",
            ("retention_single", "replaceme_single"),
            show_replaceme_unrecovered=True,
        )

        self.assertIn("ReplaceMe Single · no recovery", cell)

    def test_final_filter_keeps_only_calibration_recovery_winner(self) -> None:
        headers = tuple(METRIC_HEADERS[key] for key in UNIFIED_METRIC_KEYS)

        def row(dataset, mode, score, source):
            return SummaryRow(
                dataset,
                "pi3",
                mode,
                {header: str(score) for header in headers},
                (source,),
                {header: score for header in headers},
            )

        single = "i3_j18_recovery_ghost_interval_sequential_autogroup"
        double = "i3_j6_i9_j20_recovery_ghost_interval_sequential_autogroup"
        rows = [
            row("training average", single, 2.00, "retention_single"),
            row("training average", double, 1.95, "retention_double"),
            row("testing average", single, 9.00, "retention_single"),
            row("testing average", double, 1.00, "retention_double"),
            row("testing average", f"{single}+fastvggt_m0_r0p9", 9.00, "retention_single"),
            row("testing average", f"{double}+fastvggt_m0_r0p9", 1.00, "retention_double"),
            row("testing average", "i3_j18", 9.00, "retention_single"),
            row("testing average", "i3_j6_i9_j20", 1.00, "retention_double"),
        ]

        selected = _select_best_recovery_rows(
            rows,
            include_lower=False,
            share_recovery_winner_across_variants=True,
        )
        selected_modes = {item.mode for item in selected if item.dataset == "testing average"}

        # Double improves by only 0.05, so the shared 0.10 margin keeps Single.
        self.assertIn(single, selected_modes)
        self.assertIn(f"{single}+fastvggt_m0_r0p9", selected_modes)
        self.assertNotIn(double, selected_modes)
        self.assertNotIn(f"{double}+fastvggt_m0_r0p9", selected_modes)
        # Final filtering is recovery-only; lower rows remain independently visible.
        self.assertIn("i3_j18", selected_modes)
        self.assertIn("i3_j6_i9_j20", selected_modes)

    def test_live_loader_uses_eval_records_instead_of_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone_root = root / "vggt"
            backbone_root.mkdir()
            (backbone_root / "summary_comparison.md").write_text(
                "this stale file must not be parsed",
                encoding="utf-8",
            )
            grouped = {
                "scannet_dev50": {
                    "upper": EvalRecordV3(average={"ape": 1.25})
                }
            }
            with (
                patch(
                    "analysis.vis.summary_tab.collect_summary_records",
                    return_value=grouped,
                ) as collect,
                patch(
                    "analysis.vis.summary_tab.collect_summary_parameter_counts",
                    return_value=(100, {}),
                ),
            ):
                header, rows, live_roots = _load_live_eval_rows(root)

        self.assertEqual(live_roots, [backbone_root])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].metrics["APE↓ (m)"], "1.2500")
        self.assertEqual(rows[0].metrics["Block params (M) ↓"], "0.0")
        self.assertIn("Backbone", header)
        collect.assert_called_once_with(
            backbone_root,
            prune_plan_root=backbone_root,
            prune_strategies=("CKA", "retention"),
        )

    def test_live_loader_uses_configured_pruning_strategies_everywhere(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone_root = root / "vggt"
            backbone_root.mkdir()
            grouped = {
                "scannet_test50": {
                    "upper": EvalRecordV3(average={"ape": 1.0})
                }
            }
            with (
                patch(
                    "analysis.vis.summary_tab.collect_summary_records",
                    return_value=grouped,
                ) as collect,
                patch(
                    "analysis.vis.summary_tab.collect_summary_parameter_counts",
                    return_value=(100, {}),
                ),
                patch(
                    "analysis.vis.summary_tab._load_pruning_source_map",
                    return_value={},
                ) as source_map,
            ):
                _load_live_eval_rows(
                    root,
                    prune_strategies=("retention",),
                )

        collect.assert_called_once_with(
            backbone_root,
            prune_plan_root=backbone_root,
            prune_strategies=("retention",),
        )
        source_map.assert_called_once_with(
            backbone_root,
            prune_strategies=("retention",),
        )

    def test_live_loader_adds_exclusive_gpu_benchmark_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone_root = root / "vggt"
            benchmark_root = backbone_root / "scannet_test50" / "eval" / "testtime"
            benchmark_root.mkdir(parents=True)
            benchmark_payload = {
                "schema_version": 1,
                "backbone": "vggt",
                "dataset": "scannet_test50",
                "results": {
                    "full": {
                        "variant": "full",
                        "selector": "full",
                        "compression": {"name": "none"},
                        "scenes": [
                            {
                                "latency_ms": 100.0,
                                "num_frames": 30,
                                "memory_gib": {"peak_allocated": 8.0},
                            },
                            {
                                "latency_ms": 120.0,
                                "num_frames": 30,
                                "memory_gib": {"peak_allocated": 10.0},
                            },
                        ],
                    },
                    "full_fast": {
                        "variant": "full_fastvggt",
                        "selector": "full",
                        "compression": {
                            "name": "fastvggt",
                            "merge_start": 0,
                            "merge_ratio": 0.9,
                        },
                        "scenes": [
                            {
                                "latency_ms": 80.0,
                                "num_frames": 30,
                                "memory_gib": {"peak_allocated": 7.5},
                            }
                        ],
                    },
                    "recovery_quant": {
                        "variant": "recovery_quantvggt",
                        "selector": "i3_j20",
                        "recovery_plan_name": "recovery_ghost_interval_sequential",
                        "compression": {"name": "quantvggt", "dtype": "w4a4"},
                        "scenes": [
                            {
                                "latency_ms": 90.0,
                                "num_frames": 30,
                                "memory_gib": {"peak_allocated": 12.5},
                            }
                        ],
                    },
                },
            }
            (benchmark_root / "inference_benchmark_v3.json").write_text(
                json.dumps(benchmark_payload),
                encoding="utf-8",
            )
            grouped = {
                "scannet_test50": {
                    "upper": EvalRecordV3(average={"ape": 1.0}),
                    "upper+fastvggt_m0_r0p9": EvalRecordV3(average={"ape": 1.1}),
                    (
                        "i3_j20_recovery_ghost_interval_sequential"
                        "+quantvggt_w4a4"
                    ): EvalRecordV3(average={"ape": 1.2}),
                }
            }
            with (
                patch(
                    "analysis.vis.summary_tab.collect_summary_records",
                    return_value=grouped,
                ),
                patch(
                    "analysis.vis.summary_tab.collect_summary_parameter_counts",
                    return_value=(100, {}),
                ),
            ):
                header, rows, _ = _load_live_eval_rows(root)

        by_mode = {row.mode: row for row in rows}
        self.assertIn("Latency (ms) ↓", header)
        self.assertIn("Peak VRAM (GiB) ↓", header)
        self.assertEqual(by_mode["upper"].metrics["Latency (ms) ↓"], "110.0")
        self.assertEqual(by_mode["upper"].metrics["Peak VRAM (GiB) ↓"], "9.0")
        self.assertEqual(
            by_mode["upper+fastvggt_m0_r0p9"].metrics["Latency (ms) ↓"],
            "80.0",
        )
        quant_mode = (
            "i3_j20_recovery_ghost_interval_sequential+quantvggt_w4a4"
        )
        self.assertEqual(
            by_mode[quant_mode].metrics["Peak VRAM (GiB) ↓"], "12.5"
        )

    def test_live_loader_can_disable_compression_and_efficiency_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone_root = root / "vggt"
            backbone_root.mkdir()
            grouped = {
                "scannet_test50": {
                    "upper": EvalRecordV3(average={"ape": 1.0}),
                    "upper+fastvggt_m0_r0p9": EvalRecordV3(
                        average={"ape": 1.1}
                    ),
                    "i3_j20_recovery_ghost": EvalRecordV3(
                        average={"ape": 1.2}
                    ),
                    (
                        "i3_j20_recovery_ghost"
                        "+fastvggt_m0_r0p9"
                    ): EvalRecordV3(average={"ape": 1.3}),
                }
            }
            with (
                patch(
                    "analysis.vis.summary_tab.collect_summary_records",
                    return_value=grouped,
                ),
                patch(
                    "analysis.vis.summary_tab.collect_summary_parameter_counts",
                    return_value=(100, {}),
                ) as parameter_counts,
                patch(
                    "analysis.vis.summary_tab._load_benchmark_metrics",
                    return_value={},
                ) as benchmark_metrics,
            ):
                header, rows, _roots = _load_live_eval_rows(
                    root,
                    include_compression=False,
                    include_efficiency_metrics=False,
                )

        self.assertEqual(
            {row.mode for row in rows},
            {"upper", "i3_j20_recovery_ghost"},
        )
        self.assertNotIn("Block params (M) ↓", header)
        self.assertNotIn("Latency (ms) ↓", header)
        self.assertNotIn("Peak VRAM (GiB) ↓", header)
        parameter_counts.assert_not_called()
        benchmark_metrics.assert_not_called()

    def test_builder_exposes_recovery_method_multiselect_with_target_default(self) -> None:
        rows = [
            SummaryRow("scannet_test50", "vggt", "upper", {"APE ↓": "1"}),
            SummaryRow(
                "scannet_test50",
                "vggt",
                "upper+fastvggt_m0_r0p9",
                {"APE ↓": "1"},
            ),
            SummaryRow(
                "scannet_test50",
                "vggt",
                "upper+quantvggt_w4a4",
                {"APE ↓": "1"},
            ),
            SummaryRow(
                "scannet_test50",
                "vggt",
                "i3_j20_recovery_ghost",
                {"APE ↓": "2"},
            ),
            SummaryRow(
                "scannet_test50",
                "vggt",
                "i3_j20_recovery_ghost_interval_sequential",
                {"APE ↓": "1.5"},
            ),
        ]
        with patch(
            "analysis.vis.summary_tab._load_live_eval_rows",
            return_value=(
                ["Dataset", "Backbone", "Mode", "APE ↓"],
                rows,
                [Path("vggt")],
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_summary_tab(blocks=blocks, output_root=Path("unused"))

        self.assertIsInstance(ui["recovery_method_cb"], gr.CheckboxGroup)
        self.assertEqual(
            ui["recovery_method_cb"].choices,
            [("ghost", "ghost"), ("ghost_interval_sequential", "ghost_interval_sequential")],
        )
        self.assertEqual(
            ui["recovery_method_cb"].value,
            ["ghost_interval_sequential"],
        )
        self.assertEqual(
            ui["mode_cb"].choices,
            [("upper", "upper"), ("recovery", "recovery")],
        )
        self.assertEqual(ui["span_cb"].value, ["18"])
        self.assertEqual(
            ui["compression_cb"].choices,
            [("FastVGGT", "fastvggt"), ("QuantVGGT", "quantvggt")],
        )
        self.assertEqual(ui["compression_cb"].value, [])
        self.assertEqual(
            ui["pruning_strategy_cb"].choices,
            [("CKA", "CKA"), ("Degradation", "retention")],
        )
        self.assertEqual(
            ui["pruning_strategy_cb"].value,
            ["CKA", "retention"],
        )

    def test_builder_accepts_table_name_for_recovery_and_final_copies(self) -> None:
        with (
            patch(
                "analysis.vis.summary_tab._load_live_eval_rows",
                return_value=([], [], []),
            ),
            patch(
                "analysis.vis.summary_tab._select_best_recovery_rows",
                wraps=_select_best_recovery_rows,
            ) as select_best,
        ):
            with gr.Blocks() as blocks:
                recovery_ui = build_summary_tab(
                    blocks=blocks,
                    output_root=Path("analysis_outputs_v3"),
                    tab_id="recovery_table",
                    tab_name="Recovery Table",
                    include_compression=False,
                    include_efficiency_metrics=False,
                )
                final_ui = build_summary_tab(
                    blocks=blocks,
                    output_root=Path("main_results"),
                    tab_id="final_table",
                    tab_name="Final Table",
                    separate_pruning_structures=True,
                    winner_only_recovery=True,
                )

        # Recovery Table filters lower and recovery rows. Final Table preserves
        # separate lower structures but filters recovery rows to their winner.
        self.assertEqual(select_best.call_count, 2)
        self.assertEqual(
            select_best.call_args_list[1].kwargs,
            {
                "include_lower": False,
                "share_recovery_winner_across_variants": True,
            },
        )

        self.assertEqual(recovery_ui["tab"].label, "Recovery Table")
        self.assertEqual(recovery_ui["tab"].id, "recovery_table")
        self.assertEqual(final_ui["tab"].label, "Final Table")
        self.assertEqual(final_ui["tab"].id, "final_table")
        self.assertIsInstance(recovery_ui["copy_latex_btn"], gr.Button)
        self.assertIsInstance(final_ui["copy_latex_btn"], gr.Button)
        self.assertIsNone(recovery_ui["compression_cb"])
        self.assertIsInstance(final_ui["compression_cb"], gr.CheckboxGroup)
        self.assertIsNone(recovery_ui["pruning_structure_cb"])
        self.assertIsInstance(final_ui["pruning_structure_cb"], gr.CheckboxGroup)
        self.assertEqual(
            final_ui["pruning_structure_cb"].choices,
            [("Single", "single"), ("Double", "double")],
        )
        self.assertEqual(
            final_ui["pruning_structure_cb"].value,
            ["single", "double"],
        )

    def test_builder_accepts_table_specific_default_selections(self) -> None:
        rows = [
            SummaryRow(
                dataset,
                "vggt",
                "i3_j20_recovery_ghost_interval_sequential_autogroup",
                {"APE ↓": "1"},
            )
            for dataset in ("scannet_test50", "nuscenes_val")
        ]
        with patch(
            "analysis.vis.summary_tab._load_live_eval_rows",
            return_value=(
                ["Dataset", "Backbone", "Mode", "APE ↓"],
                rows,
                [Path("vggt")],
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_summary_tab(
                    blocks=blocks,
                    output_root=Path("unused"),
                    dataset_average_groups={
                        "same-dataset testing average": (
                            "scannet_test50",
                            "nuscenes_val",
                        ),
                        "four-dataset testing average": (
                            "scannet_test50",
                            "nuscenes_val",
                        ),
                    },
                    default_datasets=("four-dataset testing average",),
                    default_recovery_methods=(
                        "ghost_interval_sequential_autogroup",
                    ),
                )

        self.assertEqual(
            ui["dataset_cb"].value,
            ["four-dataset testing average"],
        )
        self.assertEqual(
            ui["recovery_method_cb"].value,
            ["ghost_interval_sequential_autogroup"],
        )

    def test_builder_can_default_to_drop_counts_with_recovery_results(self) -> None:
        rows = [
            SummaryRow(
                "scannet_test50",
                "vggt",
                mode,
                {"APE ↓": "1"},
            )
            for mode in (
                "prune_span8",
                "prune_span12/recovery_ghost_interval_sequential_autogroup",
                "prune_span16",
                "prune_span20/recovery_ghost_interval_sequential_autogroup",
            )
        ]
        with patch(
            "analysis.vis.summary_tab._load_live_eval_rows",
            return_value=(
                ["Dataset", "Backbone", "Mode", "APE ↓"],
                rows,
                [Path("vggt")],
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_summary_tab(
                    blocks=blocks,
                    output_root=Path("unused"),
                    default_spans_with_recovery=True,
                )

        self.assertEqual(
            ui["span_cb"].choices,
            [("8", "8"), ("12", "12"), ("16", "16"), ("20", "20")],
        )
        self.assertEqual(ui["span_cb"].value, ["12", "20"])


if __name__ == "__main__":
    unittest.main()

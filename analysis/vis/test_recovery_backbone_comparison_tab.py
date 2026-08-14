from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from analysis.vis.recovery_backbone_comparison_tab import (
    _BACKBONE_COLORS,
    _global_metric_ranges,
    _collapse_min_prune_strategies,
    _filter_testtime_to_family_winners,
    _load_testtime_points,
    _normalized_mean,
    _recovery_state_style,
    _render_combined_figure,
    build_recovery_backbone_comparison_tab,
    build_recovery_backbone_remaining_comparison_tab,
    render_recovery_backbone_comparison,
    render_recovery_backbone_remaining_comparison,
)


class RecoveryBackboneComparisonTabTest(unittest.TestCase):
    def test_testtime_filter_reuses_family_winner_selector(self) -> None:
        payload = {
            "testtime_points": [
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 16,
                    "selector": "i7_j22",
                    "prune_strategy": "gt_single",
                },
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 16,
                    "selector": "i8_j20_i41_j43",
                    "prune_strategy": "gt_double",
                },
            ]
        }
        with patch(
            "analysis.vis.recovery_backbone_comparison_tab.best_recovery_result",
            return_value={"strategy": "gt_double", "selector": "i8_j20_i41_j43"},
        ) as select:
            filtered, failures = _filter_testtime_to_family_winners(
                payload,
                root=Path("/unused"),
                recovery_method="recovery_ghost_interval_sequential",
                families=["gt"],
                selection_datasets=["scannet_dev50", "nuscenes_train"],
            )

        self.assertEqual(failures, [])
        self.assertEqual(
            [point["selector"] for point in filtered["testtime_points"]],
            ["i8_j20_i41_j43"],
        )
        self.assertEqual(
            filtered["testtime_points"][0]["testtime_winner_families"], ["gt"]
        )
        self.assertEqual(select.call_args.kwargs["strategies"], ("gt_single", "gt_double"))

    def test_min_selects_whole_points_and_testtime_per_structure_and_drop(self) -> None:
        metrics = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")

        def raw(value: float, marker: float) -> dict:
            result = {metric: value for metric in metrics}
            result["ape"] = marker
            return result

        payload = {"upper_average": {}, "strategies": {}, "testtime_points": []}
        for variant, value, marker in (
            ("gt_single", 0.8, 0.0), ("ours_single", 0.2, 1.0),
            ("gt_double", 0.1, 2.0), ("ours_double", 0.9, 3.0),
        ):
            payload["strategies"][f"recovery_linearpatch_{variant}"] = {
                "points": {"2": {"available": True, "raw_metric": raw(value, marker)}}
            }
            payload["strategies"][f"no_recovery_{variant}"] = {
                "points": {
                    "1": {
                        "available": True,
                        "raw_metric": raw(
                            0.1 if variant.startswith("gt_") else 0.9,
                            marker + 20.0,
                        ),
                    },
                    "2": {
                        "available": True,
                        "raw_metric": raw(1.0 - value, marker + 10.0),
                    },
                }
            }
            payload["testtime_points"].append({
                "compression": "fast", "drop_count": 2,
                "prune_strategy": variant, "raw_metric": raw(value, marker),
            })

        merged, method_keys = _collapse_min_prune_strategies(
            payload, method="recovery_linearpatch", structures=["single", "double"]
        )
        self.assertEqual(method_keys, [
            "recovery_linearpatch_min_single", "recovery_linearpatch_min_double"
        ])
        single = merged["strategies"][method_keys[0]]["points"]["2"]
        double = merged["strategies"][method_keys[1]]["points"]["2"]
        self.assertEqual(single["min_prune_strategy"], "ours_single")
        self.assertEqual(double["min_prune_strategy"], "gt_double")
        self.assertEqual(single["raw_metric"]["ape"], 1.0)
        self.assertEqual(double["raw_metric"]["ape"], 2.0)
        single_baseline = merged["strategies"]["no_recovery_single"]["points"]["2"]
        double_baseline = merged["strategies"]["no_recovery_double"]["points"]["2"]
        self.assertEqual(single_baseline["min_prune_strategy"], "ours_single")
        self.assertEqual(double_baseline["min_prune_strategy"], "gt_double")
        self.assertEqual(single_baseline["raw_metric"]["ape"], 11.0)
        self.assertEqual(double_baseline["raw_metric"]["ape"], 12.0)
        single_fallback = merged["strategies"]["no_recovery_single"]["points"]["1"]
        double_fallback = merged["strategies"]["no_recovery_double"]["points"]["1"]
        self.assertTrue(single_fallback["no_recovery_fallback"])
        self.assertTrue(double_fallback["no_recovery_fallback"])
        self.assertEqual(single_fallback["min_prune_strategy"], "gt_single")
        self.assertEqual(double_fallback["min_prune_strategy"], "gt_double")
        self.assertEqual(
            {point["prune_strategy"] for point in merged["testtime_points"]},
            {"gt_double"},
        )

    def test_testtime_loader_matches_strategy_selector_and_averages_datasets(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, ape in (("scannet_dev50", 2.0), ("nuscenes_train", 4.0)):
                out = (
                    root / dataset / "eval" / "testtime"
                    / "recovery_ghost_interval_sequential"
                    / "fastvggt_m0_r0p9" / "i3_j22"
                )
                out.mkdir(parents=True)
                (out / "summary.json").write_text(
                    json.dumps({
                        "selector": "i3_j22",
                        "drop_count": 20,
                        "recovery_plan_name": "recovery_ghost_interval_sequential",
                        "metrics_mean": {"ape": ape},
                    }),
                    encoding="utf-8",
                )

            with patch(
                "analysis.vis.recovery_backbone_comparison_tab._strategy_selectors_by_drop",
                return_value={20: "i3_j22", 22: "i1_j22"},
            ):
                points, warnings = _load_testtime_points(
                    root,
                    recovery_method="recovery_ghost_interval_sequential",
                    prune_strategy="gt_single",
                    dataset="training average",
                )

        self.assertEqual(warnings, [])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["drop_count"], 20)
        self.assertEqual(points[0]["selector"], "i3_j22")
        self.assertEqual(points[0]["raw_metric"]["ape"], 3.0)

    def test_testtime_points_are_isolated_and_compressions_use_distinct_markers(self) -> None:
        payload = {
            "upper_average": {},
            "strategies": {},
            "testtime_points": [
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 20,
                    "raw_metric": {"ape": 1.0},
                },
                {
                    "compression": "quantvggt_w4a4",
                    "drop_count": 22,
                    "raw_metric": {"ape": 2.0},
                },
            ],
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=[],
            title="test",
            show_no_recovery=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(len(fig.axes[0].lines), 0)
        self.assertEqual(len(fig.axes[0].collections), 2)
        self.assertEqual(
            [float(collection.get_offsets()[0, 0]) for collection in fig.axes[0].collections],
            [20.0, 22.0],
        )
        first_path = fig.axes[0].collections[0].get_paths()[0].vertices
        second_path = fig.axes[0].collections[1].get_paths()[0].vertices
        self.assertFalse(np.array_equal(first_path, second_path))
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertIn("Test-time · fastvggt_m0_r0p9", legend_labels)
        self.assertIn("Test-time · quantvggt_w4a4", legend_labels)
        plt.close(fig)

    def test_backbone_palette_matches_pruning_plan_overview(self) -> None:
        from analysis.prune_plan.eval_plots import STRATEGY_PLOT_COLORS

        self.assertEqual(
            _BACKBONE_COLORS[:6],
            tuple(
                STRATEGY_PLOT_COLORS[key]
                for key in ("ours", "gt", "replaceme", "inv_mse", "cosine", "cca")
            ),
        )

    def test_remaining_axis_uses_each_backbones_total_unit_count(self) -> None:
        payload = {
            "upper_average": {},
            "strategies": {
                "recovery_linearpatch": {
                    "display_name": "linearpatch",
                    "points": {
                        "8": {"available": True, "raw_metric": {"ape": 1.0}},
                        "10": {"available": True, "raw_metric": {"ape": 2.0}},
                    },
                }
            },
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch"],
            title="test",
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [38, 40])
        self.assertEqual(fig.axes[0].get_xlabel(), "")
        self.assertIn("Remaining layers", [text.get_text() for text in fig.texts])
        plt.close(fig)

    def test_x_axis_can_be_divided_by_two(self) -> None:
        payload = {
            "upper_average": {},
            "strategies": {
                "recovery_linearpatch": {
                    "points": {
                        "8": {"available": True, "raw_metric": {"ape": 1.0}},
                        "10": {"available": True, "raw_metric": {"ape": 2.0}},
                    }
                }
            },
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch"],
            title="test",
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48},
            halve_x_axis=True,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [19.0, 20.0])
        self.assertEqual(fig.axes[0].get_xlabel(), "")
        self.assertIn(
            "Remaining layers ÷ 2",
            [text.get_text() for text in fig.texts],
        )
        plt.close(fig)

    def test_remaining_renderer_reads_num_units_from_each_prune_bundle(self) -> None:
        captured = {}

        def capture(payloads, **kwargs):
            captured.update(kwargs)
            return object()

        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.load_prune_plan_bundle",
                side_effect=[{"num_units": 48}, {"num_units": 36}],
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value={"datasets": {}},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                side_effect=lambda payload, _pruning, _variant: payload,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._pruning_only_recovery_like_payload",
                return_value={"strategies": {}},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                side_effect=capture,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            render_recovery_backbone_remaining_comparison(
                Path("/unused"), ["vggt", "pi3"], "recovery_linearpatch"
            )

        self.assertEqual(
            captured["total_units_by_backbone"], {"vggt": 48, "pi3": 36}
        )

    def test_recovery_state_has_distinct_weight_and_marker_fill(self) -> None:
        no_recovery = _recovery_state_style("no_recovery_single")
        recovery = _recovery_state_style("recovery_linearpatch_ours_single")

        self.assertEqual(no_recovery["markerfacecolor"], "none")
        self.assertIsNone(recovery["markerfacecolor"])
        self.assertLess(no_recovery["linewidth"], recovery["linewidth"])
        self.assertLess(no_recovery["alpha"], recovery["alpha"])

    def test_average_status_warns_about_missing_datasets(self) -> None:
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "strategies": {"recovery_linearpatch": {"points": {}}}
                }
            }
        }
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value=recovery,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                return_value=recovery,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                return_value=object(),
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            _, status = render_recovery_backbone_comparison(
                Path("/unused"),
                ["vggt"],
                "recovery_linearpatch",
                "training average",
            )

        self.assertIn("⚠️ WARNING", status)
        self.assertIn("1/2 datasets", status)
        self.assertIn("Missing: nuscenes_train", status)

    def test_mean_uses_one_range_shared_by_backbones(self) -> None:
        def payload(low: float, high: float) -> dict:
            return {
                "upper_average": {metric: low for metric in (
                    "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance"
                )},
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {
                            "2": {
                                "available": True,
                                "raw_metric": {metric: high for metric in (
                                    "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance"
                                )},
                            }
                        }
                    }
                },
            }

        first, second = payload(0.0, 2.0), payload(1.0, 4.0)
        ranges = _global_metric_ranges(
            [first, second], ["recovery_linearpatch"]
        )
        self.assertEqual(ranges["ape"], (0.0, 4.0))
        self.assertAlmostEqual(
            _normalized_mean(first["strategies"]["recovery_linearpatch"]["points"]["2"]["raw_metric"], ranges),
            0.5,
        )

    def test_builder_has_single_method_and_no_normalization_control(self) -> None:
        with gr.Blocks() as blocks:
            ui = build_recovery_backbone_comparison_tab(
                blocks=blocks, output_root="/path/that/does/not/exist"
            )
        self.assertIsInstance(ui["method_dd"], gr.Dropdown)
        self.assertIsInstance(ui["image"], gr.Image)
        self.assertFalse(ui["method_dd"].multiselect)
        self.assertIsInstance(ui["show_no_recovery_cb"], gr.Checkbox)
        self.assertTrue(ui["show_no_recovery_cb"].value)
        self.assertIsInstance(ui["halve_x_axis_cb"], gr.Checkbox)
        self.assertFalse(ui["halve_x_axis_cb"].value)
        self.assertEqual(
            ui["prune_strategy_dd"].choices,
            [("gt", "gt"), ("ours", "ours"), ("min", "min")],
        )
        self.assertEqual(ui["prune_strategy_dd"].value, "ours")
        self.assertIsInstance(ui["prune_structure_cb"], gr.CheckboxGroup)
        self.assertEqual(ui["prune_structure_cb"].value, ["single", "double"])
        self.assertNotIn("gallery", ui)
        self.assertNotIn("normalize_metrics", ui)

    def test_builder_defaults_to_ghost_interval_sequential_when_available(self) -> None:
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.discover_recovery_overview_options",
                return_value=(
                    ["vggt"],
                    [
                        "recovery_linearpatch",
                        "recovery_ghost",
                        "recovery_ghost_interval_sequential",
                    ],
                ),
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.discover_recovery_overview_datasets",
                return_value=["training average"],
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_recovery_backbone_comparison_tab(
                    blocks=blocks,
                    output_root="/unused",
                )

        self.assertEqual(
            ui["method_dd"].value,
            "recovery_ghost_interval_sequential",
        )
        self.assertEqual(ui["tab"].label, "Recovery Backbone")

    def test_remaining_builder_has_separate_tab(self) -> None:
        with gr.Blocks() as blocks:
            ui = build_recovery_backbone_remaining_comparison_tab(
                blocks=blocks, output_root="/path/that/does/not/exist"
            )
        self.assertIsInstance(ui["image"], gr.Image)
        self.assertEqual(ui["tab"].id, "recovery_backbone_remaining_comparison")
        self.assertEqual(ui["tab"].label, "Recovery Remaining")

    def test_selected_structures_load_and_merge_both_strategy_variants(self) -> None:
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {"ape": 1.0},
                    "strategies": {
                        "recovery_linearpatch": {
                            "display_name": "linearpatch",
                            "points": {"2": {"available": True, "raw_metric": {"ape": 2.0}}},
                        }
                    },
                }
            }
        }
        captured = {}

        def capture(payloads, **kwargs):
            captured["payloads"] = payloads
            captured.update(kwargs)
            return object()

        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value=recovery,
            ) as collect_recovery,
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                side_effect=lambda payload, _pruning, _variant: payload,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                side_effect=capture,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            render_recovery_backbone_comparison(
                Path("/unused"), ["vggt"], "recovery_linearpatch",
                "scannet_dev50", "gt", True, ["single", "double"],
            )

        self.assertEqual(
            [call.kwargs["prune_strategy"] for call in collect_recovery.call_args_list],
            ["gt_single", "gt_double"],
        )
        strategies = captured["payloads"][0][1]["strategies"]
        self.assertIn("recovery_linearpatch_gt_single", strategies)
        self.assertIn("recovery_linearpatch_gt_double", strategies)
        self.assertEqual(
            captured["method_keys"],
            ["recovery_linearpatch_gt_single", "recovery_linearpatch_gt_double"],
        )


if __name__ == "__main__":
    unittest.main()

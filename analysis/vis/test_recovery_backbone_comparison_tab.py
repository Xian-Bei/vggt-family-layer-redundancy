from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np

from analysis.vis.recovery_backbone_comparison_tab import (
    _global_metric_ranges,
    _normalized_mean,
    _recovery_state_style,
    build_recovery_backbone_comparison_tab,
    render_recovery_backbone_comparison,
)


class RecoveryBackboneComparisonTabTest(unittest.TestCase):
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
        self.assertEqual(ui["prune_strategy_dd"].choices, [("gt", "gt"), ("ours", "ours")])
        self.assertEqual(ui["prune_strategy_dd"].value, "ours")
        self.assertIsInstance(ui["prune_structure_cb"], gr.CheckboxGroup)
        self.assertEqual(ui["prune_structure_cb"].value, ["single", "double"])
        self.assertNotIn("gallery", ui)
        self.assertNotIn("normalize_metrics", ui)

    def test_builder_defaults_to_ghost_when_available(self) -> None:
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.discover_recovery_overview_options",
                return_value=(
                    ["vggt"],
                    ["recovery_linearpatch", "recovery_ghost"],
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

        self.assertEqual(ui["method_dd"].value, "recovery_ghost")

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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
from matplotlib.lines import Line2D

from analysis.vis.final_results_tab import (
    FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS,
    FINAL_ANALYSIS_SAME_DATASET_TESTING_DATASETS,
    FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
    FINAL_DATASET_AVERAGE_GROUPS,
    FINAL_DATASETS,
    FINAL_FOUR_DATASET_TESTING_AVERAGE,
    FINAL_OVERVIEW_AVERAGE,
    FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS,
    FINAL_OVERVIEW_DATASETS,
    FINAL_SAME_DATASET_TESTING_DATASETS,
    FINAL_TABLE_DATASET_AVERAGE_GROUPS,
    FINAL_TRAINING_DATASETS,
    FINAL_TWO_DATASET_TESTING_AVERAGE,
    build_final_results_tabs,
)
from analysis.vis.recovery_tab import (
    _AUTOGROUP_METHOD,
    _append_comparison_source_legend_entries,
    _apply_backbone_normalization_profiles,
    _dataset_group_average,
    _overlay_recovery_overview_methods,
    _render_recovery_legend,
    render_recovery_detail,
    render_recovery_detail_matrix,
)


class FinalResultsTabTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path) -> None:
        backbone_root = root / "vggt"
        backbone_root.mkdir(parents=True)
        (backbone_root / "prune_plan_all.json").write_text(
            json.dumps({"strategies": {"retention_single": {"points": {}}}}),
            encoding="utf-8",
        )
        (backbone_root / "recovery_plan_all.json").write_text(
            json.dumps(
                {
                    "methods": {
                        "recovery_ghost_interval": {
                            "method": "recovery_ghost_interval"
                        },
                        "recovery_ghost_interval_sequential": {
                            "method": "recovery_ghost_interval_sequential"
                        },
                        "recovery_ghost_interval_sequential_grouped": {
                            "method": "recovery_ghost_interval_sequential_grouped"
                        },
                        "recovery_ghost_interval_sequential_autogroup": {
                            "method": "recovery_ghost_interval_sequential_autogroup"
                        },
                        "recovery_replaceme_ls": {
                            "method": "recovery_replaceme_ls"
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        for dataset in FINAL_DATASETS:
            (backbone_root / dataset / "eval" / "recovery").mkdir(
                parents=True
            )
        (backbone_root / "nrgbd" / "eval" / "recovery").mkdir(parents=True)

    def test_final_groups_match_between_both_curve_sources(self) -> None:
        self.assertEqual(
            FINAL_DATASET_AVERAGE_GROUPS,
            {
                FINAL_FOUR_DATASET_TESTING_AVERAGE: (
                    *FINAL_SAME_DATASET_TESTING_DATASETS,
                ),
                FINAL_TWO_DATASET_TESTING_AVERAGE: (
                    *FINAL_ANALYSIS_SAME_DATASET_TESTING_DATASETS,
                ),
            },
        )
        self.assertNotIn(
            "cross-dataset testing average", FINAL_DATASET_AVERAGE_GROUPS
        )
        self.assertEqual(
            FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS,
            FINAL_DATASET_AVERAGE_GROUPS,
        )

        payload = {
            "datasets": {
                dataset: {"upper_average": {"ape": float(index)}}
                for index, dataset in enumerate(FINAL_DATASETS, start=1)
            }
        }
        four_dataset_testing = _dataset_group_average(
            payload,
            FINAL_FOUR_DATASET_TESTING_AVERAGE,
            FINAL_DATASET_AVERAGE_GROUPS,
        )
        two_dataset_testing = _dataset_group_average(
            payload,
            FINAL_TWO_DATASET_TESTING_AVERAGE,
            FINAL_DATASET_AVERAGE_GROUPS,
        )
        self.assertEqual(four_dataset_testing["num_datasets"], 4)
        self.assertEqual(four_dataset_testing["upper_average"]["ape"], 6.5)
        self.assertEqual(two_dataset_testing["num_datasets"], 2)
        self.assertEqual(two_dataset_testing["upper_average"]["ape"], 6.5)

    def test_final_table_groups_match_final_detail_dataset_definitions(self) -> None:
        self.assertEqual(
            FINAL_TABLE_DATASET_AVERAGE_GROUPS,
            {
                "training average": FINAL_TRAINING_DATASETS,
                FINAL_FOUR_DATASET_TESTING_AVERAGE: (
                    *FINAL_SAME_DATASET_TESTING_DATASETS,
                ),
                FINAL_TWO_DATASET_TESTING_AVERAGE: (
                    *FINAL_ANALYSIS_SAME_DATASET_TESTING_DATASETS,
                ),
            },
        )

    def test_final_builder_forwards_hidden_calibration_group(self) -> None:
        with mock.patch(
            "analysis.vis.final_results_tab.build_recovery_tabs",
            return_value={"ok": True},
        ) as build_tabs:
            result = build_final_results_tabs(
                blocks=mock.sentinel.blocks,
                output_root=Path("/main-results"),
                recovery_output_root=Path("/analysis-results"),
            )

        self.assertEqual(result, {"ok": True})
        kwargs = build_tabs.call_args.kwargs
        self.assertEqual(
            kwargs["calibration_dataset_average_groups"],
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
        )
        self.assertEqual(
            kwargs["comparison_calibration_dataset_average_groups"],
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
        )
        self.assertNotIn(
            "training average", kwargs["dataset_average_groups"]
        )
        self.assertNotIn(
            "training average", kwargs["overview_dataset_average_groups"]
        )

    def test_final_refresh_callbacks_use_correct_groups_and_allow_empty_results(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with (
                mock.patch(
                    "analysis.vis.recovery_tab.render_recovery_overview_matrix",
                    return_value=([], "overview empty"),
                ) as render_overview,
                mock.patch(
                    "analysis.vis.recovery_tab.render_recovery_detail_matrix",
                    return_value=([], "detail empty"),
                ) as render_detail,
            ):
                with gr.Blocks() as blocks:
                    ui = build_final_results_tabs(
                        blocks=blocks,
                        output_root=root,
                        recovery_output_root=root,
                    )

                callbacks = {
                    getattr(block_fn.fn, "__name__", ""): block_fn.fn
                    for block_fn in blocks.fns.values()
                }
                overview_result = callbacks["_refresh_options"](
                    ui["backbone_cb"].value,
                    ui["method_cb"].value,
                    ui["dataset_cb"].value,
                    ui["prune_strategy_cb"].value,
                    ui["simplify_dataset_names"].value,
                )
                detail_result = callbacks["_refresh_detail_options"](
                    ui["detail_backbone_cb"].value,
                    ui["detail_method_cb"].value,
                    ui["detail_dataset_dd"].value,
                    ui["detail_prune_strategy_dd"].value,
                )

        overview_kwargs = render_overview.call_args.kwargs
        self.assertEqual(
            overview_kwargs["dataset_average_groups"],
            FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS,
        )
        self.assertEqual(
            overview_kwargs["calibration_dataset_average_groups"],
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
        )
        self.assertIsNone(overview_result[-2])
        self.assertEqual(overview_result[-1], "overview empty")

        detail_kwargs = render_detail.call_args.kwargs
        self.assertEqual(
            detail_kwargs["dataset_average_groups"],
            FINAL_DATASET_AVERAGE_GROUPS,
        )
        self.assertEqual(
            detail_kwargs["calibration_dataset_average_groups"],
            FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
        )
        self.assertIsNone(detail_result[-2])
        self.assertEqual(detail_result[-1], "detail empty")

    def test_builder_exposes_final_tabs_without_cross_dataset_or_cka(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with gr.Blocks() as blocks:
                ui = build_final_results_tabs(
                    blocks=blocks,
                    output_root=root,
                    recovery_output_root=root,
                )

        dataset_values = [value for _label, value in ui["dataset_cb"].choices]
        detail_dataset_values = [
            value for _label, value in ui["detail_dataset_dd"].choices
        ]
        self.assertEqual(
            ui["dataset_cb"].value,
            [FINAL_OVERVIEW_AVERAGE],
        )
        self.assertEqual(
            dataset_values,
            [FINAL_OVERVIEW_AVERAGE, *FINAL_OVERVIEW_DATASETS],
        )
        self.assertNotIn("training average", dataset_values)
        self.assertNotIn(FINAL_FOUR_DATASET_TESTING_AVERAGE, dataset_values)
        self.assertNotIn(FINAL_TWO_DATASET_TESTING_AVERAGE, dataset_values)
        self.assertNotIn("cross-dataset testing average", dataset_values)
        self.assertNotIn("nrgbd", dataset_values)
        self.assertNotIn(
            "cross-dataset testing average", detail_dataset_values
        )
        self.assertNotIn("training average", detail_dataset_values)
        self.assertIn(
            ("Four-Dataset\nTest Average", FINAL_FOUR_DATASET_TESTING_AVERAGE),
            ui["detail_dataset_dd"].choices,
        )
        self.assertIn(
            ("Two-Dataset\nTest Average", FINAL_TWO_DATASET_TESTING_AVERAGE),
            ui["detail_dataset_dd"].choices,
        )
        self.assertEqual(
            ui["detail_dataset_dd"].value,
            FINAL_FOUR_DATASET_TESTING_AVERAGE,
        )
        self.assertTrue(set(FINAL_DATASETS).issubset(detail_dataset_values))
        self.assertEqual(
            ui["prune_strategy_cb"].choices,
            [("Degradation", "Retention_min")],
        )
        self.assertEqual(
            ui["detail_prune_strategy_dd"].choices,
            [("Degradation", "Retention_min")],
        )
        self.assertEqual(ui["tab"].label, "Final")
        self.assertEqual(ui["detail_tab"].label, "Final")
        self.assertNotIn("detail_normalize_metrics", ui)
        self.assertNotIn("normalize_metrics", ui)
        for checkbox in (ui["method_cb"], ui["detail_method_cb"]):
            self.assertNotIn("recovery_ghost_interval", checkbox.value)
            self.assertNotIn(
                "recovery_ghost_interval_sequential", checkbox.value
            )
            self.assertNotIn(
                "recovery_ghost_interval_sequential_grouped", checkbox.value
            )
            self.assertIn(
                "recovery_ghost_interval_sequential_autogroup", checkbox.value
            )

        overview_average = _dataset_group_average(
            {
                "datasets": {
                    dataset: {"upper_average": {"ape": float(index)}}
                    for index, dataset in enumerate(FINAL_OVERVIEW_DATASETS, start=1)
                }
            },
            FINAL_OVERVIEW_AVERAGE,
            FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS,
        )
        self.assertEqual(overview_average["num_datasets"], 4)
        self.assertEqual(overview_average["upper_average"]["ape"], 2.5)

    def test_final_detail_forwards_analysis_comparison_root(self) -> None:
        main_root = Path("/main-results")
        analysis_root = Path("/analysis-results")
        method = "recovery_ghost_interval_sequential_autogroup"

        with mock.patch(
            "analysis.vis.recovery_tab.render_recovery_overview",
            return_value=([], ""),
        ) as render_overview:
            render_recovery_detail(
                main_root,
                ["vggt"],
                [method],
                FINAL_FOUR_DATASET_TESTING_AVERAGE,
                "Retention_min",
                comparison_dataset_average_groups=(
                    FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS
                ),
                comparison_output_root=analysis_root,
            )
        self.assertEqual(
            render_overview.call_args.kwargs["comparison_output_root"],
            analysis_root,
        )
        self.assertEqual(
            render_overview.call_args.kwargs[
                "comparison_dataset_average_groups"
            ],
            FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS,
        )

        with mock.patch(
            "analysis.vis.recovery_tab.render_recovery_detail",
            return_value=([], ""),
        ) as render_detail:
            render_recovery_detail_matrix(
                main_root,
                ["vggt"],
                [method],
                FINAL_FOUR_DATASET_TESTING_AVERAGE,
                ["Retention_min"],
                allowed_prune_strategies=("Retention_min",),
                comparison_dataset_average_groups=(
                    FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS
                ),
                comparison_output_root=analysis_root,
            )
        self.assertEqual(
            render_detail.call_args.kwargs["comparison_output_root"],
            analysis_root,
        )
        self.assertEqual(
            render_detail.call_args.kwargs[
                "comparison_dataset_average_groups"
            ],
            FINAL_ANALYSIS_DATASET_AVERAGE_GROUPS,
        )

    def test_final_points_use_recovery_overview_normalization_bounds(self) -> None:
        payload = {
            "datasets": {
                "7scenes_test": {
                    "upper_average": {"ape": 12.0},
                    "strategies": {
                        "recovery_replaceme_ls": {
                            "points": {
                                "2": {
                                    "available": True,
                                    "metrics_mean": {"ape": 16.0},
                                }
                            }
                        }
                    },
                }
            }
        }
        profiles = {
            "vggt": {
                "7scenes_test": {
                    "upper_average": {"ape": 10.0},
                    "normalization_denominator": {"ape": 10.0},
                }
            }
        }
        normalized = _apply_backbone_normalization_profiles(
            payload,
            profiles,
            backbone="vggt",
            use_profile_upper=True,
        )
        dataset = normalized["datasets"]["7scenes_test"]
        self.assertEqual(
            dataset["normalization_bounds"]["ape"],
            {"min": 10.0, "max": 20.0},
        )
        self.assertEqual(dataset["normalized_upper_average"]["ape"], 0.2)
        self.assertEqual(
            dataset["strategies"]["recovery_replaceme_ls"]["points"]["2"]
            ["normalized_metric"]["ape"],
            0.6,
        )

    def test_recovery_overlay_pairs_color_and_line_style(self) -> None:
        method = "recovery_replaceme_ls"
        point = {"available": True, "normalized_metric": {"ape": 0.5}}
        final_payload = {
            "strategies": {
                "no_recovery": {"display_name": "No Recovery", "points": {"2": point}},
                method: {"display_name": "ReplaceMe LS", "points": {"2": point}},
            }
        }
        recovery_payload = {
            "strategies": {
                "no_recovery": {"display_name": "No Recovery", "points": {"2": point}},
                method: {"display_name": "ReplaceMe LS", "points": {"2": point}},
            }
        }
        merged, selected, styles = _overlay_recovery_overview_methods(
            final_payload, recovery_payload, [method]
        )
        baseline_reference_key = "no_recovery__recovery_overview"
        reference_key = f"{method}__recovery_overview"
        self.assertEqual(
            selected,
            ["no_recovery", baseline_reference_key, method, reference_key],
        )
        self.assertEqual(styles["no_recovery"]["linestyle"], "-")
        self.assertEqual(styles[baseline_reference_key]["linestyle"], "--")
        self.assertIsNone(styles["no_recovery"]["dashes"])
        self.assertEqual(styles[baseline_reference_key]["dashes"], (5.0, 2.0))
        self.assertEqual(
            styles["no_recovery"]["color"],
            styles[baseline_reference_key]["color"],
        )
        self.assertEqual(styles[method]["linestyle"], "-")
        self.assertEqual(styles[reference_key]["linestyle"], "--")
        self.assertEqual(styles[method]["color"], styles[reference_key]["color"])
        self.assertIn(reference_key, merged["strategies"])

        named, _selected, _styles = _overlay_recovery_overview_methods(
            final_payload,
            recovery_payload,
            [method],
            primary_source_label="Four-Dataset Test Average",
            comparison_source_label="Two-Dataset Test Average",
        )
        self.assertEqual(
            named["strategies"][method]["display_name"],
            "ReplaceMe LS",
        )
        self.assertEqual(
            named["strategies"][reference_key]["display_name"],
            "ReplaceMe LS",
        )

    def test_final_legend_separates_method_colors_from_source_line_styles(self) -> None:
        entries = [(Line2D([], [], color="red"), "ReplaceMe LS")]
        _append_comparison_source_legend_entries(
            entries,
            "Final",
            "Recovery Overview",
        )
        with mock.patch(
            "analysis.vis.recovery_tab.render_shared_legend_banner",
            return_value=None,
        ) as render:
            _render_recovery_legend(entries, 26)

        rendered_entries = render.call_args.args[0]
        self.assertEqual(
            [label for _handle, label in rendered_entries],
            ["ReplaceMe LS", "Final", "Recovery Overview"],
        )
        method_handle, final_handle, comparison_handle = [
            handle for handle, _label in rendered_entries
        ]
        self.assertEqual(method_handle.get_color(), "red")
        self.assertEqual(method_handle.get_linestyle(), "None")
        self.assertEqual(method_handle.get_marker(), "o")
        self.assertEqual(final_handle.get_color(), "#222222")
        self.assertEqual(final_handle.get_linestyle(), "-")
        self.assertEqual(final_handle.get_marker(), "None")
        self.assertEqual(comparison_handle.get_color(), "#222222")
        self.assertEqual(comparison_handle.get_linestyle(), "--")
        self.assertEqual(comparison_handle.get_marker(), "None")

    def test_ours_overlay_is_explicitly_above_other_curves(self) -> None:
        point = {"available": True, "normalized_metric": {"ape": 0.5}}
        payload = {
            "strategies": {
                "no_recovery": {
                    "display_name": "No Recovery",
                    "points": {"2": point},
                },
                _AUTOGROUP_METHOD: {
                    "display_name": "Ours",
                    "points": {"2": point},
                },
            }
        }
        _merged, _selected, styles = _overlay_recovery_overview_methods(
            payload, payload, [_AUTOGROUP_METHOD]
        )
        ours_reference = f"{_AUTOGROUP_METHOD}__recovery_overview"
        self.assertGreater(
            styles[_AUTOGROUP_METHOD]["zorder"],
            styles["no_recovery"]["zorder"],
        )
        self.assertGreater(
            styles[ours_reference]["zorder"],
            styles["no_recovery__recovery_overview"]["zorder"],
        )
        self.assertGreater(
            styles[_AUTOGROUP_METHOD]["marker_zorder"],
            styles[_AUTOGROUP_METHOD]["zorder"],
        )


if __name__ == "__main__":
    unittest.main()

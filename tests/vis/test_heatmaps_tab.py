from __future__ import annotations

from pathlib import Path
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import gradio as gr
import numpy as np

from analysis import view
from analysis.vis import build_similarity_tab, similarity_tab


class HeatmapsTabTest(unittest.TestCase):
    def test_spearman_comparison_uses_average_ranks_for_ties(self) -> None:
        reference = [0.0, 0.0, 1.0, 2.0]
        candidate = [0.0, 1.0, 1.0, 2.0]

        stats = view._packed_heatmap_comparison_stats(reference, candidate)

        self.assertIsNotNone(stats)
        assert stats is not None
        _mae, _pearson, spearman, count = stats
        expected = np.corrcoef(
            [1.5, 1.5, 3.0, 4.0],
            [1.0, 2.5, 2.5, 4.0],
        )[0, 1]
        self.assertEqual(count, 4)
        self.assertAlmostEqual(spearman, float(expected))

    def test_similarity_metrics_composite_preserves_source_raster_scale(self) -> None:
        widths = [120, 90, 70, 70, 90]
        first_row = [
            (np.zeros((100 + column, width, 3), dtype=np.uint8), "VGGT" if column == 0 else "")
            for column, width in enumerate(widths)
        ]
        second_row = [
            (np.zeros((95, width - 5, 3), dtype=np.uint8), "Pi3" if column == 0 else "")
            for column, width in enumerate(widths)
        ]

        with mock.patch.object(
            similarity_tab,
            "compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            similarity_tab._compose_similarity_metrics(
                [*first_row, *second_row]
            )

        section = compose.call_args.args[0][0]
        expected_widths = [widths[index] for index in (0, 2, 1, 3, 4)]
        self.assertEqual(
            section.column_labels,
            [
                "Normalized Degradation",
                "1 − CKA",
                "1 − Cosine",
                "1 − CCA",
                "1 − Inv_MSE",
            ],
        )
        self.assertEqual(
            section.column_width_weights,
            [float(x) for x in expected_widths],
        )
        self.assertEqual(section.column_label_centers, [0.5] * len(widths))
        self.assertTrue(section.column_labels_below)
        for row in section.fixed_rows:
            self.assertEqual(
                [cell.image.width for cell in row],
                expected_widths,
            )
            self.assertEqual(len({cell.image.height for cell in row}), 1)

    def test_similarity_metric_label_uses_heatmap_axis_center(self) -> None:
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        image[10:90, 20:120] = (220, 40, 20)
        image[10:90, 140:150] = (220, 40, 20)

        center, width = similarity_tab._heatmap_axis_geometry(
            similarity_tab.Image.fromarray(image)
        )

        self.assertEqual(width, 100)
        self.assertEqual(center, 70.0)

    def test_similarity_crops_only_internal_metric_left_margins(self) -> None:
        images = []
        for column in range(5):
            image = np.full((100, 100, 3), 255, dtype=np.uint8)
            image[10:90, 10:90] = (220, 40, 20)
            images.append((image, "VGGT" if column == 0 else ""))

        with mock.patch.object(
            similarity_tab,
            "compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            similarity_tab._compose_similarity_metrics(images)

        section = compose.call_args.args[0][0]
        self.assertEqual(section.column_width_weights, [100.0, 100.0, 92.0, 92.0, 92.0])
        self.assertEqual(section.column_gaps, [2, 6, 6, 6])
        self.assertEqual(
            [cell.image.width for cell in section.fixed_rows[0]],
            [100, 100, 92, 92, 92],
        )

    def test_single_similarity_row_puts_backbone_above_first_plot(self) -> None:
        images = [
            (np.zeros((100, 100, 3), dtype=np.uint8), "VGGT-Ω" if column == 0 else "")
            for column in range(5)
        ]

        with mock.patch.object(
            similarity_tab,
            "_draw_single_row_backbone_label",
            side_effect=lambda image, _label: image,
        ) as draw_label, mock.patch.object(
            similarity_tab,
            "compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            similarity_tab._compose_similarity_metrics(images)

        section = compose.call_args.args[0][0]
        self.assertIsNone(section.row_labels)
        self.assertIsNone(section.group_headings)
        self.assertEqual(draw_label.call_args.args[1], "VGGT-Ω")

    def test_multi_row_similarity_puts_backbones_above_first_plots(self) -> None:
        images = []
        for label in ("VGGT-Ω", "π³", "VGGT"):
            images.extend(
                (np.zeros((100, 100, 3), dtype=np.uint8), label if column == 0 else "")
                for column in range(5)
            )

        with mock.patch.object(
            similarity_tab,
            "_draw_multi_row_backbone_label",
            side_effect=lambda image, _reference, _label, **_kwargs: image,
        ) as draw_label, mock.patch.object(
            similarity_tab,
            "compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            similarity_tab._compose_similarity_metrics(images)

        section = compose.call_args.args[0][0]
        self.assertIsNone(section.row_labels)
        self.assertEqual(
            [call.args[2] for call in draw_label.call_args_list],
            ["VGGT-Ω", "π³", "VGGT"],
        )
        self.assertTrue(
            all(call.kwargs["font_size"] > 0 for call in draw_label.call_args_list)
        )

    def test_similarity_columns_compare_to_degradation_and_apply_axis_policy(self) -> None:
        similarity = {
            metric: [1.0, 0.8, 0.6, 1.0, 0.7, 1.0]
            for metric in ("cka", "cosine", "cca", "inv_mse")
        }
        rendered = np.zeros((8, 8, 3), dtype=np.uint8)
        with mock.patch.object(
            view,
            "_heatmap_rgb_from_packed",
            return_value=rendered,
        ) as render:
            images = view._render_similarity_gallery(
                similarity,
                3,
                role_short="A",
                sim_gamma=1.0,
                sim_cutoff=0.0,
                comparison_reference_packed=[
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0.1,
                    0.5,
                    0.3,
                ],
                show_colorbar_metric_keys={"inv_mse"},
                show_yindex_metric_keys={"cka"},
                show_titles=False,
            )

        self.assertEqual(len(images), 4)
        calls = render.call_args_list
        for call in calls:
            np.testing.assert_allclose(
                call.args[0],
                [np.nan, np.nan, np.nan, 0.2, 0.4, 0.3],
                equal_nan=True,
            )
            self.assertEqual(call.kwargs["cmap"], "coolwarm_r")
        self.assertEqual(
            [call.kwargs["show_colorbar"] for call in calls],
            [False, False, False, True],
        )
        self.assertEqual(
            [call.kwargs["show_yindex"] for call in calls],
            [False, True, False, False],
        )
        self.assertTrue(
            all(
                len(call.args[2].splitlines()) == 3
                and call.args[2].splitlines()[0].startswith("MAE =")
                and "%" not in call.args[2].splitlines()[0]
                and call.args[2].splitlines()[1].startswith("Pearson CC =")
                and call.args[2].splitlines()[2].startswith("Spearman CC =")
                for call in calls
            )
        )
        self.assertTrue(
            all(
                call.kwargs["title_fontsize"]
                == similarity_tab.SIMILARITY_STATS_FONT_SIZE
                for call in calls
            )
        )

    def test_name_and_default_backbone_filter(self) -> None:
        class Controller:
            dataset = "mean"
            analysis_similarity_dir = Path("similarity")
            analysis_skip_dir = Path("skip")

        with gr.Blocks():
            ui = build_similarity_tab(
                controller=Controller(),
                dataset_choices=["min", "mean", "scannet_dev50"],
                backbone_choices=["pi3", "vggt", "vggt_omega"],
                initial_backbone="pi3",
                stage2_modes=["mixed_units"],
                stage2_scene_ids=["mean"],
                initial_stage2_scene="mean",
                stage2_metric_row_pose="pose",
                stage2_metric_row_point="point",
                stage2_token_row_special="special",
                stage2_token_row_camera="camera",
                stage2_token_row_patch="patch",
                stage2_token_row_special_patch_mean="special_patch_mean",
                stage2_token_row_all="all",
                heatmaps_resolve_filters_cb=lambda *_args: (
                    "absolute",
                    [],
                    ["all", "special_patch_mean", "mean", "min"],
                    [],
                    ["all", "special_patch_mean", "mean", "min"],
                    True,
                    True,
                    True,
                ),
                refresh_stage2_full_cb=lambda *_args: (),
            )

        self.assertEqual(ui["tab"].label, "Similarity")
        self.assertEqual(
            ui["stage2_backbone_filter"].value,
            ["vggt_omega", "pi3", "vggt"],
        )
        self.assertEqual(ui["stage2_dataset_dd"].value, "mean")
        self.assertNotIn("min", ui["stage2_dataset_dd"].choices)
        self.assertEqual(ui["stage2_scene_dd"].value, "mean")
        self.assertFalse(ui["stage2_scene_dd"].interactive)
        self.assertNotIn("min", ui["stage2_token_rows_filter"].choices)
        self.assertNotIn("min", ui["stage2_token_rows_filter"].value)
        self.assertNotIn("mean", ui["stage2_token_rows_filter"].choices)
        self.assertNotIn("mean", ui["stage2_token_rows_filter"].value)
        self.assertIn(
            ("special-patch-mean", "special_patch_mean"),
            ui["stage2_token_rows_filter"].choices,
        )
        self.assertEqual(ui["stage2_token_rows_filter"].value, ["all"])
        self.assertIsInstance(ui["stage2_image"], gr.Image)
        self.assertEqual(ui["stage2_image"].type, "numpy")
        self.assertNotIn("stage2_normalize_skip_metrics", ui)
        self.assertNotIn("stage2_colorbar_histogram", ui)

    def test_similarity_does_not_offer_mean_plot_option(self) -> None:
        tab_source = inspect.getsource(build_similarity_tab)
        self.assertNotIn("Show row/column mean plots", tab_source)
        self.assertNotIn("colorbar_histogram", tab_source)

        render_source = inspect.getsource(view._prepare_stage2_render)
        self.assertNotIn("show_colorbar_histogram", render_source)

        build_ui_source = inspect.getsource(view._build_ui)
        scope_source = build_ui_source.split(
            "def _heatmaps_scope_changed(", 1
        )[1].split("with gr.Blocks(", 1)[0]
        self.assertNotIn("show_colorbar_histogram", scope_source)

    def test_similarity_blocks_include_special_patch_mean_region(self) -> None:
        block_keys = view._stage2_block_keys_for_mode("mixed_units")
        self.assertEqual(len(block_keys), 8)
        self.assertIn("sim_mixed_units_special_patch_mean", block_keys)

    def test_unevaluated_skip_spans_are_masked_in_existing_reports(self) -> None:
        maps = view._extract_skip_metric_maps_by_mode(
            {
                "num_layers": 4,
                "max_span": 2,
                "fusion_by_mode": {
                    "mixed_units": {
                        "gt_delta_upper_by_metric": {"ape": [0.0] * 10},
                    }
                },
            }
        )

        matrix = view.upper_packed_to_symmetric_matrix(
            maps["mixed_units"]["ape"],
            4,
        )
        self.assertEqual(matrix[0][1], 0.0)
        self.assertTrue(np.isnan(matrix[0][2]))
        self.assertTrue(np.isnan(matrix[0][3]))
        self.assertTrue(np.isnan(matrix[1][3]))

    def test_similarity_filter_defaults_to_all_and_rejects_stale_mean(self) -> None:
        report = SimpleNamespace(
            report={
                "fusion_by_mode": {
                    "mixed_units": {
                        "global_patch_similarity_upper_by_metric": {
                            "cka": [1.0]
                        },
                        "global_all_similarity_upper_by_metric": {
                            "cka": [1.0]
                        },
                    }
                }
            }
        )

        class Controller:
            dataset = "scannet_dev50"
            stage1_eval_by_scene = {}

            @staticmethod
            def get_skip_report(_scene_id: str) -> None:
                return None

            @staticmethod
            def get_similarity_report(_scene_id: str) -> SimpleNamespace:
                return report

        for requested in (None, ["mean"]):
            with self.subTest(requested=requested):
                resolved = view._heatmaps_resolve_filters(
                    Controller(), "mean", "absolute", None, requested
                )
                self.assertEqual(resolved[2], ["all"])
                self.assertEqual(resolved[4], ["patch", "all"])

    def test_similarity_filter_offers_special_patch_mean(self) -> None:
        report = SimpleNamespace(
            report={
                "fusion_by_mode": {
                    "mixed_units": {
                        "global_special_similarity_upper_by_metric": {
                            "cka": [0.2, 0.4]
                        },
                        "global_patch_similarity_upper_by_metric": {
                            "cka": [0.6, 0.8]
                        },
                    }
                }
            }
        )

        class Controller:
            dataset = "scannet_dev50"
            stage1_eval_by_scene = {}

            @staticmethod
            def get_skip_report(_scene_id: str) -> None:
                return None

            @staticmethod
            def get_similarity_report(_scene_id: str) -> SimpleNamespace:
                return report

        resolved = view._heatmaps_resolve_filters(
            Controller(), "mean", "absolute", None, ["special_patch_mean"]
        )
        self.assertEqual(resolved[2], ["special_patch_mean"])
        self.assertIn("special_patch_mean", resolved[4])
        maps = view._extract_similarity_metric_maps_by_mode(report.report)
        np.testing.assert_allclose(
            maps["mixed_units"]["special_patch_mean"]["cka"],
            [0.4, 0.6],
        )

    def test_remaining_labels_every_x_axis_and_only_leftmost_y_axis(self) -> None:
        render_source = inspect.getsource(view._prepare_stage2_render)
        self.assertEqual(render_source.count("show_ylabel=False"), 5)

        ui_source = inspect.getsource(view._build_ui)
        remaining_source = ui_source.split(
            "def _refresh_remaining_tab", 1
        )[1].split("def _refresh_brute_force", 1)[0]
        self.assertIn("show_xlabel=True", remaining_source)
        self.assertIn("show_xlabels=True", remaining_source)
        self.assertNotIn(
            "show_xlabel=backbone_index == len(available_backbones) - 1",
            remaining_source,
        )
        self.assertNotIn(
            "show_xlabels=backbone_index == len(available_backbones) - 1",
            remaining_source,
        )
        self.assertIn("show_ylabel=backbone_index == 0", remaining_source)
        self.assertIn("show_ylabel=False", remaining_source)
        self.assertIn("show_colorbar=False", remaining_source)
        self.assertIn("show_yindex=False", remaining_source)

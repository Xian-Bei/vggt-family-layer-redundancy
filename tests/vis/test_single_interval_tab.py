from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np

from analysis import view
from analysis.vis.single_interval_tab import (
    BRUTE_FORCE_DETAIL_COLUMN_LABELS,
    BRUTE_FORCE_OVERVIEW_COLUMN_LABELS,
    BRUTE_FORCE_OVERVIEW_LEGACY_COLUMN_LABELS,
    BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE,
    BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE,
    _compose_brute_force_overview,
    _compose_brute_force_detail,
    build_single_interval_tabs,
    filter_brute_force_detail_datasets,
)


class BruteForceTabTest(unittest.TestCase):
    def test_overview_uses_one_shared_coolwarm_colormap(self) -> None:
        source = inspect.getsource(view._build_ui)
        overview_source = source.split(
            "def _refresh_brute_force_overview(", 1
        )[1].split("def _refresh_brute_force(", 1)[0]

        self.assertNotIn('"viridis"', overview_source)
        self.assertEqual(overview_source.count('cmap="coolwarm"'), 1)

    def test_overview_only_last_column_has_colorbar(self) -> None:
        source = inspect.getsource(view._build_ui)
        overview_source = source.split(
            "def _refresh_brute_force_overview(", 1
        )[1].split("def _refresh_brute_force(", 1)[0]

        self.assertIn(
            "show_colorbar=column_index == overview_columns - 1",
            overview_source,
        )

    def test_overview_splits_pose_into_rotation_and_translation(self) -> None:
        self.assertEqual(
            BRUTE_FORCE_OVERVIEW_COLUMN_LABELS,
            (
                "Rotation Metrics",
                "Translation Metrics",
                "Geometry Metrics",
                "Rotation Metrics",
                "Translation Metrics",
                "Geometry Metrics",
            ),
        )

    def test_overview_composer_can_render_legacy_four_column_labels(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.single_interval_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_brute_force_overview(
                [gr.update(value=[tile] * 4, columns=4)],
                ["VGGT-Omega"],
                BRUTE_FORCE_OVERVIEW_LEGACY_COLUMN_LABELS,
            )

        section = compose.call_args.args[0][0]
        self.assertEqual(section.columns, 4)
        self.assertEqual(
            section.column_labels,
            list(BRUTE_FORCE_OVERVIEW_LEGACY_COLUMN_LABELS),
        )
        self.assertEqual(
            section.group_headings,
            [
                ("Indoor Normalized Degradation", 2),
                ("Outdoor Normalized Degradation", 2),
            ],
        )
        self.assertEqual(section.group_heading_anchor_columns, [0.5, 2.5])

    def test_overview_composer_preserves_source_scale_across_columns(self) -> None:
        widths = [90, 100, 110, 120, 130, 140]
        rows = [
            gr.update(
                value=[
                    np.zeros((height, width, 3), dtype=np.uint8)
                    for width, height in zip(widths, [70, 75, 80, 85, 90, 95])
                ],
                columns=6,
            ),
            gr.update(
                value=[
                    np.zeros((height, width - 5, 3), dtype=np.uint8)
                    for width, height in zip(widths, [68, 73, 78, 83, 88, 93])
                ],
                columns=6,
            ),
        ]
        with patch(
            "analysis.vis.single_interval_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_brute_force_overview(rows, ["VGGT-Omega", "Pi3"])

        section = compose.call_args.args[0][0]
        self.assertEqual(section.column_width_weights, [float(x) for x in widths])
        self.assertEqual(compose.call_args.kwargs["heatmap_column_gap"], 2)
        self.assertEqual(compose.call_args.kwargs["heatmap_row_gap"], 2)
        self.assertEqual(
            compose.call_args.kwargs["outer_font_size"],
            BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE,
        )
        self.assertTrue(section.column_labels_below)
        self.assertEqual(
            section.group_headings,
            [
                ("Indoor Normalized Degradation", 3),
                ("Outdoor Normalized Degradation", 3),
            ],
        )
        self.assertEqual(section.group_heading_anchor_columns, [1.0, 4.0])
        self.assertEqual(len(section.row_label_centers), 2)
        self.assertEqual(section.column_label_centers, [0.5] * 6)
        self.assertEqual(
            [[cell.image.size for cell in row] for row in section.fixed_rows],
            [
                [(widths[index], 95) for index in range(6)],
                [(widths[index], 93) for index in range(6)],
            ],
        )

    def test_overview_typography_uses_the_two_public_constants(self) -> None:
        source = inspect.getsource(view._build_ui)
        overview_source = source.split(
            "def _refresh_brute_force_overview(", 1
        )[1].split("def _refresh_brute_force(", 1)[0]

        self.assertIn(
            "label_fontsize=BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE",
            overview_source,
        )
        self.assertGreater(BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE, 0)
        self.assertGreater(BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE, 0)

    def test_detail_metric_labels_are_pillow_safe_and_use_canonical_units(self) -> None:
        self.assertEqual(
            BRUTE_FORCE_DETAIL_COLUMN_LABELS,
            (
                "APE↓ (m)",
                "ARE↓ (°)",
                "RPEₜ↓ (m)",
                "RPEᵣ↓ (°)",
                "Acc.↓ (m)",
                "Comp.↓ (m)",
                "CD↓ (m)",
            ),
        )

    def test_detail_composer_uses_one_seven_metric_row_per_scope(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.single_interval_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_brute_force_detail(
                [
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                ],
                ["vggt_omega"],
                ["mixed_units"],
                ["scannet_dev50", "nuscenes_train"],
            )

        sections = compose.call_args.args[0]
        self.assertEqual([section.columns for section in sections], [7, 7])
        self.assertEqual(
            [section.heading for section in sections],
            ["VGGT-Omega ScanNet", "VGGT-Omega nuScenes"],
        )
        self.assertEqual(
            sections[1].column_labels,
            list(BRUTE_FORCE_DETAIL_COLUMN_LABELS),
        )
        self.assertIsNone(sections[0].column_labels)
        self.assertTrue(sections[1].column_labels_below)
        self.assertTrue(all(section.heading_y_offset == 2 for section in sections))
        self.assertTrue(all(section.row_labels is None for section in sections))
        self.assertEqual([len(section.cells) for section in sections], [7, 7])
        self.assertEqual(compose.call_args.kwargs["section_gap"], -20)

    def test_detail_dataset_filter_is_shared_with_refresh_scopes(self) -> None:
        self.assertEqual(
            filter_brute_force_detail_datasets(
                [
                    "7scenes_test",
                    "7scenes_train",
                    "nuscenes_test",
                    "nuscenes_train",
                    "nuscenes_val",
                    "scannet_dev50",
                    "scannet_test50",
                    "waymo_validation",
                    "waymo_training",
                    "mean",
                ]
            ),
            [
                "scannet_dev50",
                "nuscenes_train",
                "7scenes_train",
                "waymo_training",
            ],
        )

    def test_single_interval_does_not_offer_mean_plot_option(self) -> None:
        source = inspect.getsource(build_single_interval_tabs)
        self.assertNotIn("Show row/column mean plots", source)
        self.assertNotIn("colorbar_histogram", source)

        build_ui_source = inspect.getsource(view._build_ui)
        overview_refresh_source = build_ui_source.split(
            "def _refresh_brute_force_overview(", 1
        )[1].split("def _heatmaps_remaining_images(", 1)[0]
        detail_refresh_source = build_ui_source.split(
            "def _refresh_brute_force(", 1
        )[1].split("def _refresh_outputs(", 1)[0]
        self.assertNotIn("show_colorbar_histogram", overview_refresh_source)
        self.assertNotIn("show_colorbar_histogram", detail_refresh_source)

    def test_detail_does_not_offer_individual_metric_normalization(self) -> None:
        source = inspect.getsource(build_single_interval_tabs)
        self.assertNotIn("Normalize individual skip metrics", source)
        self.assertNotIn("normalize_skip_metrics", source)

    def test_overview_and_detail_browser_options(self) -> None:
        with gr.Blocks():
            ui = build_single_interval_tabs(
                backbone_choices=["vggt_omega", "pi3", "vggt"],
                dataset_choices=[
                    "7scenes_test",
                    "7scenes_train",
                    "scannet_dev50",
                    "scannet_test50",
                    "nuscenes_test",
                    "nuscenes_train",
                    "nuscenes_val",
                    "waymo_validation",
                    "waymo_training",
                    "mean",
                ],
                initial_skip_datasets=[
                    "7scenes_train",
                    "scannet_dev50",
                    "nuscenes_train",
                    "waymo_training",
                ],
                stage2_modes=["mixed_units"],
                refresh_overview_cb=lambda *_args: ((), (), (), ""),
                refresh_detail_cb=lambda *_args: (),
            )

        self.assertEqual(ui["overview_tab"].label, "Single-interval")
        self.assertEqual(ui["detail_tab"].label, "Single-interval")
        self.assertIsInstance(ui["overview_image"], gr.Image)
        self.assertEqual(ui["overview_image"].type, "numpy")
        self.assertIn("Mean metric sources", ui["overview_info"].value)
        self.assertNotIn("scannet_dev50", ui["overview_info"].value)
        self.assertEqual(
            ui["overview_datasets"],
            [
                "scannet_dev50",
                "nuscenes_train",
                "7scenes_train",
                "waymo_training",
            ],
        )
        self.assertEqual(
            [checkbox.value for checkbox in ui["overview_dataset_checks"]],
            [True, True, True, True],
        )
        self.assertEqual(
            [checkbox.interactive for checkbox in ui["overview_dataset_checks"]],
            [True, True, True, True],
        )
        self.assertEqual(
            ui["detail_datasets"],
            [
                "scannet_dev50",
                "nuscenes_train",
                "7scenes_train",
                "waymo_training",
            ],
        )
        self.assertIsInstance(ui["detail_image"], gr.Image)
        self.assertEqual(ui["detail_image"].type, "numpy")


if __name__ == "__main__":
    unittest.main()

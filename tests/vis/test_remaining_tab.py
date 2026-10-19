from __future__ import annotations

import unittest
from unittest import mock

import gradio as gr
import numpy as np
from PIL import Image

from analysis.view import (
    _gallery_item_with_context,
    _preferred_remaining_mode,
    _prepare_skip_maps_for_dataset_aggregation,
    _render_skip_seven_metric_mean,
    _degradation_comparison_caption,
    _degradation_comparison_stats,
)
from analysis.vis.composite import Cell, Section, compose_sections
from analysis.vis.remaining_tab import (
    REMAINING_BETWEEN_BACKBONE_GAP,
    REMAINING_BETWEEN_POINT_GAP,
    REMAINING_COLUMN_LABELS,
    REMAINING_WITHIN_BACKBONE_GAP,
    _bottom_axis_label_center,
    _compose_remaining,
    _crop_white_side,
    _draw_pruning_arrows,
    build_remaining_tab,
)


class RemainingTabTest(unittest.TestCase):
    def test_seven_metric_renderer_forwards_gap_annotation_style(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with mock.patch(
            "analysis.view._heatmap_rgb_from_packed",
            return_value=image,
        ) as render:
            result = _render_skip_seven_metric_mean(
                {"ape": [0.0]},
                1,
                value_mode="absolute",
                normalize_seven_metrics=True,
                normalized_input=True,
                gap_annotation="MAE 4.2%",
                gap_annotation_fontsize=44,
                gap_annotation_color="#666666",
                show_xindex=False,
            )

        self.assertIs(result, image)
        self.assertEqual(render.call_args.kwargs["gap_annotation"], "MAE 4.2%")
        self.assertEqual(render.call_args.kwargs["gap_annotation_fontsize"], 44)
        self.assertEqual(render.call_args.kwargs["gap_annotation_color"], "#666666")
        self.assertFalse(render.call_args.kwargs["show_xindex"])

    def test_degradation_comparison_uses_jointly_finite_final_values(self) -> None:
        original = {"ape": [0.0, 0.5, 1.0, float("nan")]}
        marginal = {"ape": [0.0, 0.25, 1.0, 0.5]}

        stats = _degradation_comparison_stats(original, marginal)

        self.assertIsNotNone(stats)
        assert stats is not None
        mae, pearson, spearman, count = stats
        original_degradation = np.asarray([0.0, 0.5, 1.0])
        marginal_degradation = np.asarray([0.0, 0.25, 1.0])
        self.assertEqual(count, 3)
        self.assertAlmostEqual(
            mae,
            float(np.mean(np.abs(marginal_degradation - original_degradation))),
        )
        self.assertAlmostEqual(
            pearson,
            float(np.corrcoef(original_degradation, marginal_degradation)[0, 1]),
        )
        self.assertAlmostEqual(spearman, 1.0)
        self.assertEqual(
            _degradation_comparison_caption(original, marginal),
            (
                f"MAE = {mae:.2f}\n"
                f"Pearson = {pearson:.2f}\n"
                f"Spearman = {spearman:.2f}"
            ),
        )

    def test_weighted_columns_do_not_distribute_residual_scale_pixels(self) -> None:
        red = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
        blue = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)

        composed = compose_sections(
            [
                Section(
                    columns=2,
                    cells=[Cell(Image.fromarray(red)), Cell(Image.fromarray(blue))],
                    column_width_weights=[100.0, 100.0],
                )
            ],
            width=205,
            outer_font_size=1,
            left_label_to_image_gap=0,
            top_label_to_image_gap=0,
            bottom_label_to_image_gap=0,
            heatmap_column_gap=2,
        )

        red_columns = np.flatnonzero(
            np.any(np.all(composed == (255, 0, 0), axis=2), axis=0)
        )
        blue_columns = np.flatnonzero(
            np.any(np.all(composed == (0, 0, 255), axis=2), axis=0)
        )
        self.assertEqual(red_columns.size, blue_columns.size)

    def test_bottom_column_labels_preserve_descenders(self) -> None:
        composed = compose_sections(
            [
                Section(
                    columns=1,
                    cells=[Cell(Image.new("RGB", (100, 20), "white"))],
                    column_labels=["Geometry gyp"],
                    column_labels_below=True,
                )
            ],
            width=320,
            outer_font_size=40,
            left_label_to_image_gap=0,
            top_label_to_image_gap=0,
            bottom_label_to_image_gap=0,
            section_gap=0,
        )

        ink_rows = np.flatnonzero(np.any(composed < 250, axis=(1, 2)))
        self.assertGreater(ink_rows.size, 0)
        self.assertLessEqual(int(ink_rows[-1]), composed.shape[0] - 3)

    def test_multiline_row_label_centers_each_line(self) -> None:
        composed = compose_sections(
            [
                Section(
                    columns=1,
                    cells=[Cell(Image.new("RGB", (100, 100), "white"))],
                    row_labels=["Calibration\nDataset"],
                    row_labels_bold=True,
                )
            ],
            width=320,
            outer_font_size=30,
            left_label_to_image_gap=0,
            top_label_to_image_gap=0,
            bottom_label_to_image_gap=0,
        )

        label_ink = np.any(composed[:, :198] < 250, axis=2)
        ink_rows = np.flatnonzero(np.any(label_ink, axis=1))
        row_groups = np.split(
            ink_rows,
            np.flatnonzero(np.diff(ink_rows) > 1) + 1,
        )
        self.assertEqual(len(row_groups), 2)
        line_centers = []
        for rows in row_groups:
            columns = np.flatnonzero(np.any(label_ink[rows], axis=0))
            line_centers.append((columns[0] + columns[-1]) / 2.0)
        self.assertAlmostEqual(line_centers[0], line_centers[1], delta=2.0)

    def test_raw_width_weighting_keeps_pair_at_equal_pixel_scale(self) -> None:
        left = np.zeros((105, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 150, 3), dtype=np.uint8)

        with mock.patch(
            "analysis.vis.remaining_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_remaining([[left, right]], ["VGGT"])

        section = compose.call_args.args[0][0]
        self.assertEqual(section.column_width_weights, [100.0, 150.0])
        self.assertEqual(section.column_gaps, [REMAINING_WITHIN_BACKBONE_GAP])
        self.assertEqual(section.column_label_centers, [0.5, 0.5])
        self.assertEqual(
            section.column_labels,
            ["Original\nDegradation", "Marginal\nDegradation"],
        )
        padded_left, padded_right = section.fixed_rows[0]
        self.assertEqual(padded_left.image.height, 105)
        self.assertEqual(padded_right.image.height, 105)
        self.assertTrue(np.all(np.asarray(padded_right.image)[:5] == 255))
        self.assertTrue(np.all(np.asarray(padded_right.image)[5:] == 0))

    def test_remaining_uses_tighter_gaps_within_backbones(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch(
            "analysis.vis.remaining_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_remaining(
                [[image, image], [image, image], [image, image]],
                ["A", "B", "C"],
            )

        section = compose.call_args.args[0][0]
        self.assertEqual(
            section.column_gaps,
            [
                REMAINING_WITHIN_BACKBONE_GAP,
                REMAINING_BETWEEN_BACKBONE_GAP,
                REMAINING_WITHIN_BACKBONE_GAP,
                REMAINING_BETWEEN_BACKBONE_GAP,
                REMAINING_WITHIN_BACKBONE_GAP,
            ],
        )
        self.assertEqual(section.group_heading_anchor_boundaries, [1, 3, 5])

    def test_multiple_points_share_one_set_of_outer_labels(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch(
            "analysis.vis.remaining_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_remaining(
                [
                    [
                        (image, "16\nMAE = 1.0%\nPearson = 0.90\nSpearman = 0.91"),
                        (image, "16\nMAE = 1.0%\nPearson = 0.90\nSpearman = 0.91"),
                        (image, "20\nMAE = 2.0%\nPearson = 0.80\nSpearman = 0.81"),
                        (image, "20\nMAE = 2.0%\nPearson = 0.80\nSpearman = 0.81"),
                    ],
                    [
                        (image, "12\nMAE = 1.0%\nPearson = 0.90\nSpearman = 0.91"),
                        (image, "12\nMAE = 1.0%\nPearson = 0.90\nSpearman = 0.91"),
                        (image, "16\nMAE = 2.0%\nPearson = 0.80\nSpearman = 0.81"),
                        (image, "16\nMAE = 2.0%\nPearson = 0.80\nSpearman = 0.81"),
                    ],
                ],
                ["Pi3", "VGGT"],
            )

        sections = compose.call_args.args[0]
        self.assertEqual(len(sections), 1)
        self.assertEqual(
            sections[0].group_headings,
            [("Pi3", 2), ("VGGT", 2)],
        )
        self.assertEqual(
            sections[0].column_labels,
            ["Original\nDegradation", "Marginal\nDegradation"] * 2,
        )
        self.assertEqual(len(sections[0].fixed_rows), 1)
        cells = sections[0].fixed_rows[0]
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].image.size, cells[1].image.size)
        self.assertEqual(
            cells[1].image.size,
            (100, 200 + REMAINING_BETWEEN_POINT_GAP),
        )
        self.assertEqual(cells[2].image.size, cells[3].image.size)
        self.assertEqual(
            cells[3].image.size,
            (100, 200 + REMAINING_BETWEEN_POINT_GAP),
        )

    def test_first_row_variant_aligns_pair_without_arrows(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch(
            "analysis.vis.remaining_tab._draw_pruning_arrows",
        ) as draw_arrows, mock.patch(
            "analysis.vis.remaining_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_remaining(
                [
                    [
                        (image, "16\nMAE = 1.0%"),
                        (image, "16\nMAE = 1.0%"),
                        (image, "20\nMAE = 2.0%"),
                        (image, "20\nMAE = 2.0%"),
                    ]
                ],
                ["VGGT"],
                first_row_only=True,
                draw_comparison_arrows=False,
            )

        draw_arrows.assert_not_called()
        cells = compose.call_args.args[0][0].fixed_rows[0]
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[0].image.size, cells[1].image.size)
        self.assertEqual(cells[0].image.size, (100, 100))

    def test_pruning_arrows_label_the_two_comparisons(self) -> None:
        image = Image.new("RGB", (300, 600), "white")

        annotated = _draw_pruning_arrows(
            image,
            degradation_box=(0, 150, 300, 450),
            marginal_centers=[150.0, 450.0],
            comparison_labels=[
                "MAE = 2.9%\nPearson = 0.98\nSpearman = 0.98",
                "MAE = 4.2%\nPearson = 0.97\nSpearman = 0.97",
            ],
        )

        self.assertFalse(np.array_equal(np.asarray(annotated), np.asarray(image)))
        self.assertTrue(np.any(np.asarray(annotated)[40:150] < 250))
        self.assertTrue(np.any(np.asarray(annotated)[450:560] < 250))

    def test_remaining_label_tracks_bottom_axis_label_center(self) -> None:
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        image[10:65, 5:155] = (180, 20, 20)
        image[88:96, 20:120] = 0

        self.assertEqual(_bottom_axis_label_center(Image.fromarray(image)), 0.4375)
        self.assertEqual(
            REMAINING_COLUMN_LABELS,
            ("Original\nDegradation", "Marginal\nDegradation"),
        )

    def test_inner_white_margin_crop_preserves_ink_and_guard(self) -> None:
        image = np.full((20, 40, 3), 255, dtype=np.uint8)
        image[4:16, 10:31] = 0

        left = _crop_white_side(Image.fromarray(image), side="left", padding=2)
        right = _crop_white_side(Image.fromarray(image), side="right", padding=2)

        self.assertEqual(left.size, (32, 20))
        self.assertEqual(right.size, (33, 20))
        self.assertTrue(np.all(np.asarray(left)[4:16, 2:23] == 0))
        self.assertTrue(np.all(np.asarray(right)[4:16, 10:31] == 0))

    def test_marginal_normalization_reuses_original_metric_bounds(self) -> None:
        original = {
            "mixed_units": {
                "ape": [float("nan"), 0.0, 10.0, 20.0],
            }
        }
        marginal = {
            "mixed_units": {
                "ape": [float("nan"), 5.0, 10.0, 30.0],
            }
        }

        normalized = _prepare_skip_maps_for_dataset_aggregation(
            marginal,
            value_mode="absolute",
            normalize_metrics=True,
            normalization_reference_maps=original,
        )

        values = normalized["mixed_units"]["ape"]
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_allclose(values[1:], [0.25, 0.5, 1.0])

    def test_one_preferred_skip_mode_is_selected(self) -> None:
        self.assertEqual(
            _preferred_remaining_mode(["default", "mixed_units"]),
            "mixed_units",
        )
        self.assertEqual(_preferred_remaining_mode(["custom"]), "custom")

    def test_existing_score_caption_is_flattened_for_gradio_gallery(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        item = _gallery_item_with_context((image, "detail"), "vggt")

        self.assertIs(item[0], image)
        self.assertEqual(item[1], "vggt · detail")
        gr.Gallery().postprocess([item])

    def test_tab_has_one_composite_image(self) -> None:
        with gr.Blocks():
            ui = build_remaining_tab(
                dataset_choices=["min", "mean", "scannet_dev50"],
                backbone_choices=["vggt_omega", "pi3"],
                initial_dataset="mean",
                refresh_remaining_cb=lambda *_args: ([], [], True, True, ""),
            )

        self.assertEqual(ui["tab"].label, "Remaining")
        self.assertEqual(ui["dataset"].value, "mean")
        self.assertNotIn("min", ui["dataset"].choices)
        self.assertIsInstance(ui["image"], gr.Image)
        self.assertEqual(ui["image"].type, "numpy")
        self.assertEqual(ui["image"].label, "Original / marginal 7-metric mean")


if __name__ == "__main__":
    unittest.main()

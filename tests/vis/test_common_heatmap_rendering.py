from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import analysis.view as view


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by repository test policy")
class CommonHeatmapRenderingGpuTest(unittest.TestCase):
    def test_axes_use_one_based_ticks_and_include_maximum_index(self) -> None:
        axes = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            axes.append(ax)
            return original_imshow(ax, arr, n, **kwargs)

        num_layers = view.HEATMAP_DISPLAY_NUM_LAYERS
        packed = [1.0] * (num_layers * (num_layers + 1) // 2)
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            view._heatmap_rgb_from_packed(packed, num_layers, "")

        self.assertEqual(len(axes), 1)
        expected_labels = [1, *range(4, num_layers + 1, 4)]
        expected_positions = [value - 1 for value in expected_labels]
        np.testing.assert_allclose(axes[0].get_xticks(), expected_positions)
        np.testing.assert_allclose(axes[0].get_yticks(), expected_positions)
        self.assertEqual(
            [label.get_text() for label in axes[0].get_xticklabels()],
            [str(value) for value in expected_labels],
        )
        self.assertEqual(
            [label.get_text() for label in axes[0].get_yticklabels()],
            [str(value) for value in expected_labels],
        )
        self.assertEqual(axes[0].xaxis.label.get_fontweight(), "normal")
        self.assertEqual(axes[0].yaxis.label.get_fontweight(), "normal")
        self.assertTrue(
            all(
                label.get_fontweight() == "normal"
                for label in axes[0].get_xticklabels()
                + axes[0].get_yticklabels()
            )
        )
        self.assertTrue(
            all(
                label.get_fontweight() == "normal"
                for label in axes[0].figure.axes[1].get_yticklabels()
            )
        )

    def test_gap_annotation_is_centered_in_internal_missing_region(self) -> None:
        axes = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            axes.append(ax)
            return original_imshow(ax, arr, n, **kwargs)

        num_layers = view.HEATMAP_DISPLAY_NUM_LAYERS
        matrix = np.ones((num_layers, num_layers), dtype=np.float64)
        matrix[7:23, :] = np.nan
        matrix[:, 7:23] = np.nan
        packed = view.upper_triangle_pack(torch.from_numpy(matrix))
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            view._heatmap_rgb_from_packed(
                packed,
                num_layers,
                "",
                gap_annotation="MAE 20.2%\nPearson CC = 0.95",
                gap_annotation_fontsize=44,
            )

        self.assertEqual(len(axes), 1)
        annotations = [
            text
            for text in axes[0].texts
            if text.get_text().startswith("MAE")
        ]
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].get_position(), (23.5, 14.5))
        self.assertEqual(annotations[0].get_fontweight(), "bold")
        self.assertEqual(annotations[0].get_fontsize(), 44.0)
        self.assertIsNone(annotations[0].get_bbox_patch())

    def test_isolated_missing_corner_cells_use_hatching(self) -> None:
        axes = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            axes.append(ax)
            return original_imshow(ax, arr, n, **kwargs)

        num_layers = view.HEATMAP_DISPLAY_NUM_LAYERS
        packed = [1.0] * (num_layers * (num_layers + 1) // 2)
        packed[num_layers - 1] = float("nan")
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            view._heatmap_rgb_from_packed(packed, num_layers, "")

        self.assertEqual(len(axes), 1)
        rectangles = axes[0].patches
        self.assertEqual(len(rectangles), 2)
        positions = {
            (rectangle.get_x(), rectangle.get_y())
            for rectangle in rectangles
        }
        self.assertEqual(positions, {(-0.5, 46.5), (46.5, -0.5)})
        self.assertTrue(
            all(
                rectangle.get_width() == rectangle.get_height() == 1.0
                for rectangle in rectangles
            )
        )

    def test_shared_canvas_padding_uses_missing_layer_hatching(self) -> None:
        axes = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            axes.append(ax)
            return original_imshow(ax, arr, n, **kwargs)

        num_layers = 36
        packed = [1.0] * (num_layers * (num_layers + 1) // 2)
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            view._heatmap_rgb_from_packed(packed, num_layers, "")

        self.assertEqual(len(axes), 1)
        rectangles = axes[0].patches
        self.assertEqual(len(rectangles), 2)
        self.assertEqual(
            (
                rectangles[0].get_x(),
                rectangles[0].get_y(),
                rectangles[0].get_width(),
                rectangles[0].get_height(),
            ),
            (35.5, -0.5, 12.0, 48.0),
        )
        self.assertEqual(
            (
                rectangles[1].get_x(),
                rectangles[1].get_y(),
                rectangles[1].get_width(),
                rectangles[1].get_height(),
            ),
            (-0.5, 35.5, 48.0, 12.0),
        )
        self.assertTrue(
            all(rectangle.get_linewidth() == 0.0 for rectangle in rectangles)
        )
        self.assertTrue(all(rectangle.get_hatch() == "/" for rectangle in rectangles))

    def test_merged_axis_labels_do_not_resize_heatmap_area(self) -> None:
        axes = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            axes.append(ax)
            return original_imshow(ax, arr, n, **kwargs)

        packed = [1.0, 0.5, 1.0]
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            view._heatmap_rgb_from_packed(
                packed,
                2,
                "",
                show_xlabel=True,
                show_ylabel=True,
            )
            view._heatmap_rgb_from_packed(
                packed,
                2,
                "",
                show_xlabel=False,
                show_ylabel=False,
            )

        self.assertEqual(len(axes), 2)
        np.testing.assert_allclose(
            axes[0].get_position().bounds,
            axes[1].get_position().bounds,
        )

    def test_packed_heatmap_is_globally_expanded_to_full_symmetric_matrix(self) -> None:
        captured: list[tuple[np.ndarray, int]] = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            captured.append((np.asarray(arr, dtype=np.float64).copy(), n))
            return original_imshow(ax, arr, n, **kwargs)

        packed = [
            1.0,
            0.2,
            float("nan"),
            1.0,
            0.4,
            1.0,
        ]
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            image = view._heatmap_rgb_from_packed(
                packed,
                3,
                "full matrix",
                vmin=0.0,
                vmax=1.0,
            )

        self.assertEqual(image.ndim, 3)
        self.assertEqual(len(captured), 1)
        values, display_num_layers = captured[0]
        self.assertEqual(display_num_layers, view.HEATMAP_DISPLAY_NUM_LAYERS)
        self.assertEqual(
            values.shape,
            (view.HEATMAP_DISPLAY_NUM_LAYERS, view.HEATMAP_DISPLAY_NUM_LAYERS),
        )
        expected = np.array(
            [
                [1.0, 0.2, np.nan],
                [0.2, 1.0, 0.4],
                [np.nan, 0.4, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_equal(values[:3, :3], expected)
        self.assertTrue(np.isnan(values[3:, :]).all())
        self.assertTrue(np.isnan(values[:, 3:]).all())


if __name__ == "__main__":
    unittest.main()

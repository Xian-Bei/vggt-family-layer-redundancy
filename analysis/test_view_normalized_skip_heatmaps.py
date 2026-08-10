from __future__ import annotations

from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from analysis import view


class NormalizedSkipHeatmapTest(unittest.TestCase):
    def test_absolute_clamps_each_dataset_before_cross_dataset_mean(self) -> None:
        better = {"mixed_units": {"ape": [-2.0, 2.0]}}
        worse = {"mixed_units": {"ape": [4.0, 4.0]}}
        left = view._prepare_skip_maps_for_dataset_aggregation(
            better, value_mode="absolute", signed_absolute_input=True
        )
        right = view._prepare_skip_maps_for_dataset_aggregation(
            worse, value_mode="absolute", signed_absolute_input=True
        )
        averaged = view._packed_elementwise_mean(
            [left["mixed_units"]["ape"], right["mixed_units"]["ape"]]
        )
        np.testing.assert_allclose(averaged, [2.0, 3.0])

    def test_cross_dataset_normalized_mean_matches_expected_retention(self) -> None:
        absolute_rows = []
        for absolute_maximum in (10.0, 20.0):
            absolute = {
                "mixed_units": {
                    metric: [0.0, absolute_maximum / 2.0, absolute_maximum]
                    for metric in view.SKIP_GT_METRIC_KEYS
                }
            }
            absolute_rows.append(
                view._prepare_skip_maps_for_dataset_aggregation(
                    absolute, value_mode="absolute", normalize_metrics=True
                )["mixed_units"]
            )
        absolute_mean = {
            metric: view._packed_elementwise_mean([row[metric] for row in absolute_rows])
            for metric in view.SKIP_GT_METRIC_KEYS
        }
        absolute_display = view._skip_seven_metric_display_packed(
            absolute_mean,
            value_mode="absolute",
            normalize_seven_metrics=True,
            normalized_input=True,
        )
        np.testing.assert_allclose(absolute_display, [1.0, 0.5, 0.0])

    def test_cross_dataset_normalized_mean_is_not_minmaxed_again(self) -> None:
        metric_map = {
            metric: [0.25, 0.5, 0.75]
            for metric in view.SKIP_GT_METRIC_KEYS
        }
        display = view._skip_seven_metric_display_packed(
            metric_map,
            value_mode="absolute",
            normalize_seven_metrics=True,
            normalized_input=True,
        )
        np.testing.assert_allclose(display, [0.75, 0.5, 0.25])

    def test_single_metric_heatmap_clamps_better_than_upper_before_display(self) -> None:
        calls: list[list[float]] = []

        def capture(
            packed: list[float],
            _num_layers: int,
            _title: str,
            **_kwargs: Any,
        ) -> np.ndarray:
            calls.append(packed)
            return np.zeros((1, 1, 3), dtype=np.uint8)

        metric_map = {"ape": [-5.0, 0.0, 5.0]}
        with patch.object(view, "_heatmap_rgb_from_packed", side_effect=capture):
            view._render_skip_gallery(
                metric_map,
                2,
                [("ape", "APE")],
                value_mode="absolute",
                normalize_seven_metrics=False,
            )
            view._render_skip_gallery(
                metric_map,
                2,
                [("ape", "APE")],
                value_mode="absolute",
                normalize_seven_metrics=True,
            )

        np.testing.assert_allclose(calls[0], [0.0, 0.0, -5.0])
        np.testing.assert_allclose(calls[1], [1.0, 1.0, 0.0])

    def test_all_renderers_use_zero_to_one_retention(self) -> None:
        calls: list[tuple[list[float], dict[str, Any]]] = []

        def capture(
            packed: list[float],
            _num_layers: int,
            _title: str,
            **kwargs: Any,
        ) -> np.ndarray:
            calls.append((packed, kwargs))
            return np.zeros((1, 1, 3), dtype=np.uint8)

        metric_map = {"ape": [0.0, 1.0, 2.0]}
        with patch.object(view, "_heatmap_rgb_from_packed", side_effect=capture):
            view._render_skip_gallery(
                metric_map,
                2,
                [("ape", "APE")],
                value_mode="absolute",
                normalize_seven_metrics=True,
            )
            view._render_skip_seven_metric_mean(
                metric_map,
                2,
                value_mode="absolute",
                normalize_seven_metrics=True,
            )
            view._render_skip_metric_mean(
                metric_map,
                2,
                [("ape", "APE")],
                title="Pose",
                cmap="viridis",
                value_mode="absolute",
                normalize_metrics=True,
            )

        self.assertEqual(len(calls), 3)
        for packed, kwargs in calls:
            np.testing.assert_allclose(packed, [1.0, 0.5, 0.0])
            self.assertEqual(kwargs["vmin"], 0.0)
            self.assertEqual(kwargs["vmax"], 1.0)

    def test_heatmap_scores_use_common_finite_strict_upper_entries(self) -> None:
        left = [999.0, 0.25, 0.5, -999.0, 0.75, 999.0]
        right = [0.0, 0.25, np.nan, 0.0, 0.75, 0.0]

        scores = view._packed_strict_upper_heatmap_scores(left, right, 3)

        self.assertIsNotNone(scores)
        assert scores is not None
        self.assertAlmostEqual(float(scores["soft_iou"]), 1.0)
        self.assertAlmostEqual(float(scores["soft_dice"]), 1.0)
        self.assertAlmostEqual(float(scores["sinkhorn_2d"]), 0.0)
        self.assertAlmostEqual(float(scores["js_divergence"]), 0.0)

    def test_heatmap_scores_preserve_strength_and_spatial_distance(self) -> None:
        # Moving mass from (0, 2) to (1, 2) costs one 2D grid cell even
        # though those entries are two positions apart in the packed array.
        left = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        right = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        scores = view._packed_strict_upper_heatmap_scores(left, right, 3)

        self.assertIsNotNone(scores)
        assert scores is not None
        self.assertAlmostEqual(float(scores["soft_iou"]), 0.0)
        self.assertAlmostEqual(float(scores["soft_dice"]), 0.0)
        self.assertAlmostEqual(float(scores["sinkhorn_2d"]), 1.0)
        self.assertAlmostEqual(float(scores["js_divergence"]), 1.0)

    def test_similarity_suffix_reports_horizontal_and_vertical_scores(
        self,
    ) -> None:
        current = {"cka": [1.0, 0.2, 0.4, 1.0, 0.6, 1.0]}
        retention = [1.0, 0.2, 0.4, 1.0, 0.6, 1.0]
        original = {"cka": [1.0, 0.1, 0.2, 1.0, 0.3, 1.0]}

        suffixes = view._similarity_score_suffixes(
            current,
            3,
            horizontal_reference=retention,
            vertical_reference=original,
            sim_gamma=1.0,
            sim_cutoff=0.0,
        )

        self.assertEqual(
            suffixes["cka"],
            " | H IoU=1.000 Dice=1.000 Sinkhorn=0.000 JS=0.000 · "
            "V IoU=0.500 Dice=0.800 Sinkhorn=0.000 JS=0.000",
        )

    def test_marginal_score_compares_displayed_map_with_row_and_column_bases(
        self,
    ) -> None:
        marginal = {"cka": [1.0, 0.2, 0.4, 1.0, 0.8, 1.0]}
        row_base = [1.0, 0.2, 0.4, 1.0, 0.8, 1.0]
        column_base = {
            "cka": [1.0, 0.2, 0.4, 1.0, 0.8, 1.0]
        }

        suffixes = view._similarity_score_suffixes(
            marginal,
            3,
            horizontal_reference=row_base,
            vertical_reference=column_base,
            sim_gamma=2.0,
            sim_cutoff=0.0,
            current_is_displayed=True,
            vertical_reference_is_displayed=True,
        )

        self.assertEqual(
            suffixes["cka"],
            " | H IoU=1.000 Dice=1.000 Sinkhorn=0.000 JS=0.000 · "
            "V IoU=1.000 Dice=1.000 Sinkhorn=0.000 JS=0.000",
        )

    def test_similarity_score_is_gallery_caption_not_raster_title(self) -> None:
        rendered_titles: list[str] = []

        def capture(
            _packed: list[float],
            _num_layers: int,
            title: str,
            **_kwargs: Any,
        ) -> np.ndarray:
            rendered_titles.append(title)
            return np.zeros((1, 1, 3), dtype=np.uint8)

        with patch.object(view, "_heatmap_rgb_from_packed", side_effect=capture):
            items = view._render_similarity_gallery(
                {"cka": [1.0, 0.5, 1.0]},
                2,
                role_short="A",
                sim_gamma=1.0,
                sim_cutoff=0.0,
                title_suffix_by_metric={
                    "cka": " | H IoU=0.800 · V IoU=0.900"
                },
            )

        self.assertEqual(rendered_titles, ["A·CKA"])
        self.assertEqual(items[0][1], "H IoU=0.800 · V IoU=0.900")

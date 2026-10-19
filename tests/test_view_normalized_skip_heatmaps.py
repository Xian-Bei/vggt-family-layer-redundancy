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

    def test_cross_dataset_normalized_mean_matches_expected_degradation(self) -> None:
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
        np.testing.assert_allclose(absolute_display, [0.0, 0.5, 1.0])

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
        np.testing.assert_allclose(display, [0.25, 0.5, 0.75])

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
        np.testing.assert_allclose(calls[1], [0.0, 0.0, 1.0])

    def test_single_metric_gallery_accepts_external_titles_and_coolwarm(self) -> None:
        rendered: list[tuple[str, str, bool, bool]] = []

        def capture(
            _packed: list[float],
            _num_layers: int,
            title: str,
            **kwargs: Any,
        ) -> np.ndarray:
            rendered.append(
                (
                    title,
                    kwargs["cmap"],
                    kwargs["show_xlabel"],
                    kwargs["show_ylabel"],
                )
            )
            return np.zeros((1, 1, 3), dtype=np.uint8)

        with patch.object(view, "_heatmap_rgb_from_packed", side_effect=capture):
            view._render_skip_gallery(
                {"ape": [0.0, 1.0, 2.0]},
                2,
                [("ape", "APE")],
                value_mode="absolute",
                cmap="coolwarm",
                show_title=False,
                show_xlabels=False,
                show_ylabel=False,
            )

        self.assertEqual(rendered, [("", "coolwarm", False, False)])

    def test_all_renderers_use_zero_to_one_degradation(self) -> None:
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
        for (packed, kwargs), expected_cmap in zip(
            calls,
            ("viridis_r", "coolwarm_r", "viridis_r"),
        ):
            np.testing.assert_allclose(packed, [0.0, 0.5, 1.0])
            self.assertEqual(kwargs["vmin"], 0.0)
            self.assertEqual(kwargs["vmax"], 1.0)
            self.assertEqual(kwargs["cmap"], expected_cmap)

    def test_similarity_heading_is_rendered_inside_figure(self) -> None:
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
            )

        self.assertEqual(rendered_titles, ["A·1 − CKA"])
        self.assertIsInstance(items[0], np.ndarray)

    def test_similarity_keeps_title_and_outer_axis_labels_in_figure(self) -> None:
        rendered: list[tuple[str, bool]] = []

        def capture(
            _packed: list[float],
            _num_layers: int,
            title: str,
            **kwargs: Any,
        ) -> np.ndarray:
            rendered.append(
                (
                    title,
                    bool(kwargs["show_xlabel"] or kwargs["show_ylabel"]),
                )
            )
            return np.zeros((1, 1, 3), dtype=np.uint8)

        with patch.object(view, "_heatmap_rgb_from_packed", side_effect=capture):
            items = view._render_similarity_gallery(
                {"cka": [1.0, 0.5, 1.0]},
                2,
                role_short="A",
                sim_gamma=1.0,
                sim_cutoff=0.0,
            )

        self.assertEqual(rendered, [("A·1 − CKA", True)])
        self.assertIsInstance(items[0], np.ndarray)

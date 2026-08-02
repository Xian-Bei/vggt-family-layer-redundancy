from __future__ import annotations

from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from analysis import view


class NormalizedSkipHeatmapTest(unittest.TestCase):
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

    def test_pearson_uses_common_finite_strict_upper_entries(self) -> None:
        left = [999.0, 1.0, 2.0, -999.0, 3.0, 999.0]
        right = [0.0, 2.0, np.nan, 0.0, 6.0, 0.0]

        score = view._packed_strict_upper_pearson(left, right, 3)

        self.assertIsNotNone(score)
        self.assertAlmostEqual(float(score), 1.0)

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

        self.assertEqual(suffixes["cka"], " | H r=1.000 · V r=1.000")

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

        self.assertEqual(suffixes["cka"], " | H r=1.000 · V r=1.000")

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
                    "cka": " | H r=0.800 · V r=0.900"
                },
            )

        self.assertEqual(rendered_titles, ["A·CKA"])
        self.assertEqual(items[0][1], "H r=0.800 · V r=0.900")

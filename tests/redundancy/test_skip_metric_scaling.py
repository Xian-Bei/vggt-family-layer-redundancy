from __future__ import annotations

import unittest

import numpy as np

from analysis.redundancy.skip_metric_scaling import (
    normalize_canonical_degradation,
    normalize_raw_metric_against_upper_worst,
    raw_metric_to_degradation,
)


class NormalizedDegradationTest(unittest.TestCase):
    def test_absolute_degradation_uses_upper_as_zero_anchor(self) -> None:
        raw = np.asarray([8.0, 10.0, 15.0, 20.0])
        absolute = raw_metric_to_degradation(
            raw, metric="ape", upper_baseline=10.0
        )
        normalized = normalize_canonical_degradation(absolute, maximum=10.0)

        np.testing.assert_allclose(absolute, [0.0, 0.0, 5.0, 10.0])
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.5, 1.0])

    def test_plot_mode_preserves_better_than_upper_values(self) -> None:
        normalized = normalize_raw_metric_against_upper_worst(
            np.asarray([8.0, 10.0, 15.0, 20.0]),
            upper_baseline=10.0,
            worst_value=20.0,
            clamp_better_than_upper=False,
        )
        np.testing.assert_allclose(normalized, [-0.2, 0.0, 0.5, 1.0])

    def test_heatmap_mode_clamps_better_than_upper_values(self) -> None:
        normalized = normalize_raw_metric_against_upper_worst(
            np.asarray([8.0, 10.0, 15.0, 20.0]),
            upper_baseline=10.0,
            worst_value=20.0,
            clamp_better_than_upper=True,
        )
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.5, 1.0])

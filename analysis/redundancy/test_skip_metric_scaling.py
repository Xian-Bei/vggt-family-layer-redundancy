from __future__ import annotations

import unittest

import numpy as np

from analysis.redundancy.skip_metric_scaling import (
    normalize_canonical_degradation,
    normalized_degradation_to_retention,
    raw_metric_to_degradation,
)


class NormalizedDegradationToRetentionTest(unittest.TestCase):
    def test_absolute_degradation_uses_upper_as_zero_anchor(self) -> None:
        raw = np.asarray([8.0, 10.0, 15.0, 20.0])
        absolute = raw_metric_to_degradation(
            raw, metric="ape", upper_baseline=10.0
        )
        normalized = normalize_canonical_degradation(absolute, maximum=10.0)

        np.testing.assert_allclose(absolute, [0.0, 0.0, 5.0, 10.0])
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.5, 1.0])

    def test_shifts_display_range(self) -> None:
        degradation = np.asarray([0.0, 0.25, 1.0, np.nan])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention[:3], [1.0, 0.75, 0.0])
        self.assertTrue(np.isnan(retention[3]))

    def test_clips_roundoff(self) -> None:
        degradation = np.asarray([-1e-12, 1.0 + 1e-12])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention, [1.0, 0.0])

from __future__ import annotations

import unittest

import numpy as np

from analysis.redundancy.skip_metric_scaling import (
    normalized_degradation_to_retention,
)


class NormalizedDegradationToRetentionTest(unittest.TestCase):
    def test_shifts_display_range(self) -> None:
        degradation = np.asarray([0.0, 0.25, 1.0, np.nan])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention[:3], [1.0, 0.75, 0.0])
        self.assertTrue(np.isnan(retention[3]))

    def test_clips_roundoff(self) -> None:
        degradation = np.asarray([-1e-12, 1.0 + 1e-12])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention, [1.0, 0.0])

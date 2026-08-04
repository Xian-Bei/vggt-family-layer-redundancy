from __future__ import annotations

import unittest

import numpy as np

from analysis.redundancy.skip_metric_scaling import (
    normalize_canonical_degradation,
    normalized_degradation_to_retention,
    raw_metric_to_degradation,
    raw_metric_to_upper_multiple_score,
)


class NormalizedDegradationToRetentionTest(unittest.TestCase):
    def test_absolute_and_multiple_share_canonical_normalization(self) -> None:
        raw = np.asarray([8.0, 10.0, 15.0, 20.0])
        absolute = raw_metric_to_degradation(
            raw, metric="ape", upper_baseline=10.0, value_mode="absolute"
        )
        multiple = raw_metric_to_degradation(
            raw, metric="ape", upper_baseline=10.0, value_mode="multiple"
        )
        normalized = normalize_canonical_degradation(absolute, maximum=10.0)

        np.testing.assert_allclose(absolute, [0.0, 0.0, 5.0, 10.0])
        np.testing.assert_allclose(multiple, [1.0, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.5, 1.0])

    def test_upper_multiple_metadata_omits_unused_floor_count(self) -> None:
        scores, metadata = raw_metric_to_upper_multiple_score(
            np.asarray([8.0, 10.0, 15.0]),
            metric="ape",
            upper_baseline=10.0,
        )

        np.testing.assert_allclose(scores, [1.0, 1.0, 1.5])
        self.assertTrue(metadata["enabled"])
        self.assertNotIn("num_floored_to_one", metadata)

    def test_shifts_display_range(self) -> None:
        degradation = np.asarray([0.0, 0.25, 1.0, np.nan])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention[:3], [1.0, 0.75, 0.0])
        self.assertTrue(np.isnan(retention[3]))

    def test_clips_roundoff(self) -> None:
        degradation = np.asarray([-1e-12, 1.0 + 1e-12])

        retention = normalized_degradation_to_retention(degradation)

        np.testing.assert_allclose(retention, [1.0, 0.0])

from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis.redundancy.metrics import (
    sample_angular_distance,
    sample_cosine_similarity,
)
from analysis.redundancy.similarity_v3 import (
    compute_scene_similarity,
    unit_similarity_to_interval_score,
)


class UnitOutputSimilarityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU is required for this test")

    def test_compute_scene_similarity_returns_unit_interval_similarity(self) -> None:
        base = torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            device="cuda",
        )
        layers = [
            base + 0.25,
            base * 0.5 + 1.0,
            torch.flip(base, dims=(0,)),
        ]
        result = compute_scene_similarity(
            layers,
            metric_names=("cka", "cosine", "inv_mse"),
            metric_device="cuda",
            progress=False,
            progress_desc="test",
        )
        for matrix in result.values():
            np.testing.assert_array_equal(matrix, matrix.T)
            np.testing.assert_array_equal(np.diag(matrix), np.ones(3))
        self.assertAlmostEqual(
            float(result["cosine"][0, 2]),
            (sample_cosine_similarity(layers[0], layers[2]) + 1.0) / 2.0,
            places=7,
        )
        self.assertGreaterEqual(float(result["cosine"].min()), 0.0)
        self.assertLessEqual(float(result["cosine"].max()), 1.0)

    def test_cosine_maps_opposite_vectors_to_zero(self) -> None:
        x = torch.tensor([[1.0, 0.0]], device="cuda")
        y = torch.tensor([[-1.0, 0.0]], device="cuda")

        result = compute_scene_similarity(
            [x, y],
            metric_names=("cosine",),
            metric_device="cuda",
            progress=False,
            progress_desc="test",
        )

        self.assertAlmostEqual(float(result["cosine"][0, 1]), 0.0, places=7)

    def test_angular_distance_is_computed_before_token_mean(self) -> None:
        x = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device="cuda")
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")
        result = compute_scene_similarity(
            [x, y],
            metric_names=("angular",),
            metric_device="cuda",
            progress=False,
            progress_desc="test",
        )
        self.assertAlmostEqual(sample_angular_distance(x, y), 0.25, places=7)
        self.assertAlmostEqual(float(result["angular"][0, 1]), 0.25, places=7)
        np.testing.assert_array_equal(np.diag(result["angular"]), np.zeros(2))

    def test_interval_conversion_has_structural_nan_row_zero(self) -> None:
        similarity = np.array(
            [
                [1.0, 0.4, 0.9],
                [0.4, 1.0, 0.7],
                [0.9, 0.7, 1.0],
            ],
            dtype=np.float64,
        )
        score = unit_similarity_to_interval_score(similarity)
        self.assertTrue(np.all(np.isnan(score[0, :])))
        self.assertEqual(float(score[1, 2]), float(similarity[0, 2]))
        self.assertEqual(float(score[2, 2]), float(similarity[1, 2]))


if __name__ == "__main__":
    unittest.main()

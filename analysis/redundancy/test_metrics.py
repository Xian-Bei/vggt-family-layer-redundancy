from __future__ import annotations

import unittest

import torch

from analysis.redundancy.metrics import inverse_mean_squared_error
from analysis.redundancy.metrics import pairwise_similarity_matrices


class InverseMSETest(unittest.TestCase):
    def test_inverse_mse_matches_definition_on_gpu(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU is required for this test")
        x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], device="cuda")
        y = torch.tensor([[[[1.0, 1.0], [0.0, 3.0]]]], device="cuda")
        mse = torch.mean((x - y) ** 2).item()

        score = inverse_mean_squared_error(x, y)

        self.assertAlmostEqual(score, 1.0 / (1.0 + mse), places=7)

    def test_pairwise_inverse_mse_is_higher_for_identical_activations(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU is required for this test")
        x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], device="cuda")
        y = x + 2.0

        matrix = pairwise_similarity_matrices(
            (x, y), metric_names=("inv_mse",), device="cuda"
        )["inv_mse"]

        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=7)
        self.assertLess(float(matrix[0, 1]), float(matrix[0, 0]))


if __name__ == "__main__":
    unittest.main()

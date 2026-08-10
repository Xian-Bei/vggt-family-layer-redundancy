from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from analysis.redundancy.metrics import (
    upper_packed_to_symmetric_matrix,
    upper_triangle_pack,
)
from analysis.redundancy.remaining_skip_rebase import (
    build_marginal_skip_maps,
    resolve_plan_interval,
)


def _packed(matrix: list[list[float]]) -> list[float]:
    return upper_triangle_pack(torch.tensor(matrix, dtype=torch.float64))


class RemainingSkipRebaseTest(unittest.TestCase):
    def test_resolves_exact_contiguous_plan_interval(self) -> None:
        plan = SimpleNamespace(drop=[1, 2])
        self.assertEqual(
            resolve_plan_interval(
                plan,
                full_unit_ids=["u0", "u1", "u2", "u3"],
                expected_dropped_unit_ids=["u1", "u2"],
            ),
            (1, 2, "i1_j2"),
        )

    def test_rebases_remaining_delta_to_base_pruned_metric(self) -> None:
        original = {
            "unit_ids": ["u0", "u1", "u2", "u3"],
            "upper_average": {"ape": 10.0},
            "fusion_by_mode": {
                "mixed_units": {
                    "gt_delta_upper_by_metric": {
                        "ape": _packed(
                            [
                                [0.0, 0.5, 1.0, 1.5],
                                [0.5, 0.0, 2.0, 2.5],
                                [1.0, 2.0, 0.0, 3.0],
                                [1.5, 2.5, 3.0, 0.0],
                            ]
                        )
                    }
                }
            },
        }
        remaining = {
            "unit_ids": ["u0", "u3"],
            "upper_average": {"ape": 10.0},
            "fusion_by_mode": {
                "mixed_units": {
                    "gt_delta_upper_by_metric": {
                        "ape": _packed([[3.0, 5.0], [5.0, 4.0]])
                    }
                }
            },
        }

        rebased = build_marginal_skip_maps(
            original_report=original,
            remaining_report=remaining,
            base_interval=(1, 2),
        )
        matrix = np.asarray(
            upper_packed_to_symmetric_matrix(
                rebased["mixed_units"]["ape"], 2
            )
        )
        np.testing.assert_allclose(matrix, [[1.0, 3.0], [3.0, 2.0]])


if __name__ == "__main__":
    unittest.main()

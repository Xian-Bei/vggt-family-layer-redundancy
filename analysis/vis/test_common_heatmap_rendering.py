from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import analysis.view as view


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by repository test policy")
class CommonHeatmapRenderingGpuTest(unittest.TestCase):
    def test_packed_heatmap_is_globally_expanded_to_full_symmetric_matrix(self) -> None:
        captured: list[np.ndarray] = []
        original_imshow = view.imshow_upper_triangle

        def capture_imshow(ax, arr, n, **kwargs):
            captured.append(np.asarray(arr, dtype=np.float64).copy())
            return original_imshow(ax, arr, n, **kwargs)

        packed = [
            1.0,
            0.2,
            float("nan"),
            1.0,
            0.4,
            1.0,
        ]
        with mock.patch.object(
            view,
            "imshow_upper_triangle",
            side_effect=capture_imshow,
        ):
            image = view._heatmap_rgb_from_packed(
                packed,
                3,
                "full matrix",
                vmin=0.0,
                vmax=1.0,
            )

        self.assertEqual(image.ndim, 3)
        self.assertEqual(len(captured), 1)
        expected = np.array(
            [
                [1.0, 0.2, np.nan],
                [0.2, 1.0, 0.4],
                [np.nan, 0.4, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_equal(captured[0], expected)


if __name__ == "__main__":
    unittest.main()

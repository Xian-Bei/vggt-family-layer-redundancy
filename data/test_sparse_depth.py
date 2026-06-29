from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.datasets.driving_depth import (
    rasterize_for_omega_balanced,
    rasterize_for_vggt_crop,
    vggt_crop_output_hw,
    omega_balanced_output_hw,
)
from analysis.datasets.sparse_depth import load_sparse_depth, save_sparse_depth


class DrivingSparseDepthTest(unittest.TestCase):
    def test_round_trip_filters_invalid_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            saved = save_sparse_depth(
                path,
                u=np.array([10.0, -1.0, 20.0, np.nan]),
                v=np.array([12.0, 12.0, 30.0, 4.0]),
                depth=np.array([5.0, 6.0, 100.0, 3.0]),
                image_width=100,
                image_height=50,
                source="unit_test",
                min_depth=0.1,
                max_depth=80.0,
            )
            loaded = load_sparse_depth(path)
            self.assertEqual(saved.count, 1)
            self.assertEqual(loaded.depth_type, "camera_z")
            self.assertAlmostEqual(float(loaded.depth[0]), 5.0)

    def test_vggt_shapes_and_nearest_depth_z_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            save_sparse_depth(
                path,
                u=np.array([800.0, 800.2, 1200.0]),
                v=np.array([450.0, 450.1, 600.0]),
                depth=np.array([8.0, 3.0, 12.0]),
                image_width=1600,
                image_height=900,
                source="unit_test",
            )
            item = load_sparse_depth(path)
            self.assertEqual(vggt_crop_output_hw(1600, 900), (294, 518))
            self.assertEqual(vggt_crop_output_hw(1920, 1280), (350, 518))
            dense = rasterize_for_vggt_crop(item, (294, 518))
            self.assertEqual(dense.shape, (294, 518))
            self.assertAlmostEqual(float(dense[147, 259]), 3.0)

    def test_omega_balanced_shape_and_rasterize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            save_sparse_depth(
                path,
                u=np.array([800.0, 1200.0]),
                v=np.array([450.0, 600.0]),
                depth=np.array([8.0, 12.0]),
                image_width=1600,
                image_height=900,
                source="unit_test",
            )
            item = load_sparse_depth(path)
            self.assertEqual(omega_balanced_output_hw(900, 1600), (384, 688))
            dense = rasterize_for_omega_balanced(item, (384, 688))
            self.assertEqual(dense.shape, (384, 688))
            self.assertGreater(float(np.count_nonzero(dense)), 0.0)


if __name__ == "__main__":
    unittest.main()

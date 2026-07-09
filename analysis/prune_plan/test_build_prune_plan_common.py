from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import analysis.prune_plan.build_prune_plan_common as common
import analysis.schema.v3 as schema_v3


def _write_graph(root: Path, experiment: str, dataset: str, region: str, sim: np.ndarray) -> None:
    red_dir = root / experiment / dataset / "redundancy"
    red_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        red_dir / f"redundancy_graph_{region}.npz",
        unit_ids=np.asarray(["frame_0", "global_0"], dtype=object),
        cka_mean=np.asarray(sim, dtype=np.float64),
    )


class LoadSimilarityDatasetListTest(unittest.TestCase):
    def test_dataset_list_uses_mean_over_explicit_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_graph(root, "vggt", "scannet_dev50", "all", np.array([[1.0, 0.2], [0.2, 1.0]]))
            _write_graph(root, "vggt", "nuscenes_train", "all", np.array([[1.0, 0.8], [0.8, 1.0]]))

            with (
                mock.patch.object(common, "OUTPUT_ROOT_DEFAULT", root),
                mock.patch.object(schema_v3, "OUTPUT_ROOT_DEFAULT", root),
            ):
                unit_ids, sim, provenance = common.load_similarity(
                    experiment="vggt",
                    dataset="scannet_dev50,nuscenes_train",
                    token_region="all",
                    metric="cka",
                )

        self.assertEqual(unit_ids, ["frame_0", "global_0"])
        np.testing.assert_allclose(sim, np.array([[1.0, 0.5], [0.5, 1.0]]))
        self.assertEqual(provenance["datasets"], ["scannet_dev50", "nuscenes_train"])
        self.assertEqual(provenance["dataset_selector"], "scannet_dev50,nuscenes_train")
        self.assertEqual(provenance["dataset_aggregation"], "mean")

    def test_single_dataset_is_not_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = np.array([[1.0, 0.3], [0.3, 1.0]])
            _write_graph(root, "vggt", "scannet_dev50", "all", expected)

            with (
                mock.patch.object(common, "OUTPUT_ROOT_DEFAULT", root),
                mock.patch.object(schema_v3, "OUTPUT_ROOT_DEFAULT", root),
            ):
                _unit_ids, sim, provenance = common.load_similarity(
                    experiment="vggt",
                    dataset="scannet_dev50",
                    token_region="all",
                    metric="cka",
                )

        np.testing.assert_allclose(sim, expected)
        self.assertIsNone(provenance["dataset_aggregation"])

    def test_dataset_mean_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly"):
            common.parse_dataset_list("mean")

        with self.assertRaisesRegex(ValueError, "explicitly"):
            common.parse_dataset_list("scannet_dev50,mean")


if __name__ == "__main__":
    unittest.main()

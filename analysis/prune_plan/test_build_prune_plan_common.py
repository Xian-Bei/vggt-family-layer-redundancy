from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import analysis.prune_plan.build_prune_plan_common as common
import analysis.schema.v3 as schema_v3
from analysis.redundancy.similarity_v3 import UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS


def _write_graph(
    root: Path,
    experiment: str,
    dataset: str,
    region: str,
    sim: np.ndarray,
    *,
    semantics: str = UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS,
) -> None:
    red_dir = root / experiment / dataset / "redundancy"
    red_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit_ids": np.asarray(["frame_0", "global_0"], dtype=object),
        "cka_mean": np.asarray(sim, dtype=np.float64),
    }
    if semantics:
        payload["graph_semantics"] = np.asarray([semantics], dtype=object)
    np.savez(red_dir / f"redundancy_graph_{region}.npz", **payload)


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
        self.assertEqual(provenance["graph_semantics"], UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS)

    def test_region_mean_and_min_compose_with_dataset_mean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for dataset, special, patch in (
                ("d1", 0.2, 0.6),
                ("d2", 0.4, 0.8),
            ):
                _write_graph(
                    root,
                    "vggt",
                    dataset,
                    "special",
                    np.array([[1.0, special], [special, 1.0]]),
                )
                _write_graph(
                    root,
                    "vggt",
                    dataset,
                    "patch",
                    np.array([[1.0, patch], [patch, 1.0]]),
                )

            with (
                mock.patch.object(common, "OUTPUT_ROOT_DEFAULT", root),
                mock.patch.object(schema_v3, "OUTPUT_ROOT_DEFAULT", root),
            ):
                _unit_ids, mean_sim, mean_provenance = common.load_similarity(
                    experiment="vggt",
                    dataset="d1,d2",
                    token_region="mean",
                    metric="cka",
                )
                _unit_ids, min_sim, min_provenance = common.load_similarity(
                    experiment="vggt",
                    dataset="d1,d2",
                    token_region="min",
                    metric="cka",
                )

        np.testing.assert_allclose(mean_sim, np.array([[1.0, 0.5], [0.5, 1.0]]))
        np.testing.assert_allclose(min_sim, np.array([[1.0, 0.3], [0.3, 1.0]]))
        self.assertEqual(mean_provenance["dataset_aggregation"], "mean")
        self.assertEqual(min_provenance["dataset_aggregation"], "mean")

    def test_legacy_graph_without_interval_semantics_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_graph(
                root,
                "vggt",
                "scannet_dev50",
                "all",
                np.array([[1.0, 0.3], [0.3, 1.0]]),
                semantics="",
            )

            with (
                mock.patch.object(common, "OUTPUT_ROOT_DEFAULT", root),
                mock.patch.object(schema_v3, "OUTPUT_ROOT_DEFAULT", root),
            ):
                with self.assertRaisesRegex(ValueError, "tmp_migrate_interval"):
                    common.load_similarity(
                        experiment="vggt",
                        dataset="scannet_dev50",
                        token_region="all",
                        metric="cka",
                    )

    def test_dataset_mean_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicitly"):
            common.parse_dataset_list("mean")

        with self.assertRaisesRegex(ValueError, "explicitly"):
            common.parse_dataset_list("scannet_dev50,mean")


class RecoveryIntervalPlanTest(unittest.TestCase):
    def test_closed_interval_drop_uses_left_boundary_as_lora_anchor(self) -> None:
        pick = SimpleNamespace(i=2, j=4, unit_i="frame_1", unit_j="frame_2")

        plan, lora_indices = common.build_recovery_plan_from_interval(
            picks_prefix=[pick],
            lora_prefix=1,
            lora_rank=8,
            lora_alpha=16.0,
            backbone="vggt",
            dataset="scannet_dev50",
            num_units=8,
            source_cache="",
        )

        self.assertEqual(plan.drop, [2, 3, 4])
        self.assertEqual(lora_indices, [1])
        self.assertEqual(sorted(plan.lora), ["1"])
        self.assertEqual(plan.lora["1"].alignment_teacher_unit_index, 4)

    def test_prefix_interval_without_left_boundary_is_rejected(self) -> None:
        pick = SimpleNamespace(i=0, j=1, unit_i="frame_0", unit_j="global_0")
        with self.assertRaisesRegex(ValueError, "no physical left-boundary"):
            common.build_recovery_plan_from_interval(
                picks_prefix=[pick],
                lora_prefix=1,
                lora_rank=8,
                lora_alpha=16.0,
                backbone="vggt",
                dataset="scannet_dev50",
                num_units=8,
                source_cache="",
            )


if __name__ == "__main__":
    unittest.main()

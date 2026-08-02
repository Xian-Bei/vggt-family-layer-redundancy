from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan.build_prune_plan_v3 import (
    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP,
)
from analysis.prune_plan.gt_single_interval_knee import (
    round_degradation_for_selection,
    select_gt_single_interval_knee,
    write_knee_plot,
)


METRICS = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "acc",
    "comp",
    "chamfer_distance",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class GtSingleIntervalKneeTest(unittest.TestCase):
    def test_rounds_to_two_decimals_before_threshold_comparison(self) -> None:
        self.assertEqual(round_degradation_for_selection(0.054), 0.05)
        self.assertEqual(round_degradation_for_selection(0.055), 0.06)
        self.assertLessEqual(round_degradation_for_selection(0.054), 0.05)

    def _write_dataset(self, root: Path, dataset: str) -> None:
        eval_root = root / dataset / "eval"
        pruning_root = eval_root / "pruning"
        _write_json(
            pruning_root / "skip_eval_manifest_v3.json",
            {
                "fingerprint": f"{dataset}-fingerprint",
                "config": {
                    "unit_ids": [f"unit_{index}" for index in range(4)],
                },
            },
        )
        _write_json(
            eval_root / "upper" / "eval_record_v3.json",
            {
                "average": {metric: 1.0 for metric in METRICS},
                "scenes": {"scene0": {"scene_id": "scene0"}},
            },
        )
        scores = {
            "i1_j1": 1.1,
            "i2_j2": 1.2,
            # The ascending path anchors on this best one-unit interval, which
            # forces the two-unit interval to overlap it.
            "i3_j3": 1.0,
            "i1_j2": 1.05,
            "i2_j3": 1.1,
            "i1_j3": 2.0,
        }
        for selector, score in scores.items():
            _write_json(
                pruning_root / selector / "summary.json",
                {
                    "selector": selector,
                    "num_scenes": 1,
                    "metrics_mean": {metric: score for metric in METRICS},
                },
            )
        _write_json(
            pruning_root / "i1_j2_i3_j3" / "summary.json",
            {
                "selector": "i1_j2_i3_j3",
                "num_scenes": 1,
                "metrics_mean": {metric: 0.0 for metric in METRICS},
            },
        )

    def test_selects_largest_eligible_best_single_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root, "dataset_a")
            self._write_dataset(root, "dataset_b")

            selected = select_gt_single_interval_knee(
                plan_root=root,
                datasets=["dataset_a", "dataset_b"],
                threshold=0.1,
            )

            self.assertEqual(selected["selector"], "i1_j2")
            self.assertEqual(selected["drop_count"], 2)
            self.assertTrue(selected["single_interval_only"])
            self.assertEqual(
                selected["cross_budget_interval_overlap_required"],
                GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP,
            )
            self.assertEqual(
                selected["path_selection"],
                "canonical_pruning_plan_gt_single_interval_path",
            )
            self.assertEqual(len(selected["curve"]), 3)
            self.assertEqual(
                {
                    point["drop_count"]: point["selector"]
                    for point in selected["curve"]
                },
                {
                    1: "i1_j1",
                    2: "i1_j2",
                    3: "i1_j3",
                },
            )
            self.assertAlmostEqual(
                selected["mean_normalized_degradation"],
                0.05,
            )
            self.assertEqual(
                selected["cross_budget_scan_direction"],
                "large_to_small",
            )
            plot_path = write_knee_plot(
                root / "knee.png",
                {**selected, "backbone": "vggt"},
            )
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_rejects_incomplete_single_interval_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root, "dataset")
            (root / "dataset" / "eval" / "pruning" / "i2_j3" / "summary.json").unlink()

            with self.assertRaisesRegex(ValueError, "missing 1 required"):
                select_gt_single_interval_knee(
                    plan_root=root,
                    datasets=["dataset"],
                )


if __name__ == "__main__":
    unittest.main()

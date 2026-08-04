from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan.gt_single_interval_at_point import (
    select_gt_single_interval_at_point,
)


METRICS = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class GtSingleIntervalAtPointTest(unittest.TestCase):
    def _write_dataset(self, root: Path, dataset: str) -> None:
        eval_root = root / dataset / "eval"
        pruning_root = eval_root / "pruning"
        _write_json(
            pruning_root / "skip_eval_manifest_v3.json",
            {"config": {"unit_ids": [f"unit_{index}" for index in range(4)]}},
        )
        _write_json(
            eval_root / "upper" / "eval_record_v3.json",
            {"average": {metric: 1.0 for metric in METRICS}, "scenes": {"s": {}}},
        )
        scores = {
            "i1_j1": 1.1,
            "i2_j2": 1.2,
            "i3_j3": 1.0,
            "i1_j2": 1.05,
            "i2_j3": 1.1,
            "i1_j3": 2.0,
        }
        for selector, score in scores.items():
            _write_json(
                pruning_root / selector / "summary.json",
                {"selector": selector, "num_scenes": 1, "metrics_mean": {m: score for m in METRICS}},
            )

    def test_maps_manual_point_to_canonical_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root, "a")
            self._write_dataset(root, "b")
            self.assertEqual(
                select_gt_single_interval_at_point(
                    plan_root=root, datasets=["a", "b"], point=2
                ),
                "i1_j2",
            )

    def test_rejects_unavailable_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root, "a")
            with self.assertRaisesRegex(ValueError, "point=4 is unavailable"):
                select_gt_single_interval_at_point(
                    plan_root=root, datasets=["a"], point=4
                )


if __name__ == "__main__":
    unittest.main()

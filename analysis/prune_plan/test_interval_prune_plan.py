from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    load_named_prune_plan,
    strategy_interval_selectors,
)
from analysis.prune_plan.build_prune_plan_v3 import build_interval_score_matrix
from analysis.prune_plan.migrate_lower_to_intervals import run as run_migration


def _bundle() -> dict:
    return {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "scannet_dev50,nuscenes_train",
        "num_units": 8,
        "param_accounting": {
            "available": True,
            "total_params": 800,
        },
        "source_cache": "",
        "intervals": {
            "i1_j3": {
                "selector": "i1_j3",
                "i": 1,
                "j": 3,
                "start": 1,
                "end": 3,
                "span": 3,
                "drop_count": 3,
                "unit_i": "global_0",
                "unit_j": "global_1",
                "drop": [1, 2, 3],
                "dropped_unit_ids": ["global_0", "frame_1", "global_1"],
                "aggregator_params": 600,
            },
            "i2_j5": {
                "selector": "i2_j5",
                "i": 2,
                "j": 5,
                "start": 2,
                "end": 5,
                "span": 4,
                "drop_count": 4,
                "unit_i": "frame_1",
                "unit_j": "global_2",
                "drop": [2, 3, 4, 5],
                "dropped_unit_ids": ["frame_1", "global_1", "frame_2", "global_2"],
                "aggregator_params": 500,
            },
        },
        "strategies": {
            "ours": {
                "display_name": "Ours",
                "points": {"3": "i1_j3", "4": "i2_j5"},
            },
            "gt": {
                "display_name": "GT",
                "points": {"3": "i1_j3"},
            },
        },
        "spans": {
            "3": {
                "selector": "i1_j3",
                "i": 1,
                "j": 3,
                "start": 1,
                "end": 3,
                "span": 3,
                "drop_count": 3,
                "unit_i": "global_0",
                "unit_j": "global_1",
                "drop": [1, 2, 3],
                "dropped_unit_ids": ["global_0", "frame_1", "global_1"],
            }
        },
        "extra": {},
    }


class IntervalPrunePlanTest(unittest.TestCase):
    def test_interval_selector_loads_unit_prune_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            plan = load_named_prune_plan(root, "i2_j5")

        self.assertEqual(plan.drop, [2, 3, 4, 5])
        self.assertEqual(plan.drop_length, 4)
        self.assertEqual(plan.extra["interval_selector"], "i2_j5")

    def test_strategy_interval_selectors_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            selectors = strategy_interval_selectors(
                root,
                strategies=["ours", "gt"],
                min_drop=3,
                max_drop=4,
                drop_step=1,
            )

        self.assertEqual(selectors, ["i1_j3", "i2_j5"])

    def test_migration_maps_open_left_interval_dirs_to_closed_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "analysis_outputs_v3"
            backbone_root = output_root / "vggt"
            backbone_root.mkdir(parents=True)
            (backbone_root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
                json.dumps(_bundle()),
                encoding="utf-8",
            )
            source = backbone_root / "scannet_dev50" / "eval" / "lower" / "i1_j3"
            source.mkdir(parents=True)
            (source / "eval_record_v3.json").write_text("{}", encoding="utf-8")

            moves = run_migration(output_root=output_root, backbone="vggt", commit=True)

            target = backbone_root / "scannet_dev50" / "eval" / "lower" / "i2_j3"
            self.assertEqual(moves, [(source, target)])
            self.assertFalse(source.exists())
            self.assertTrue((target / "eval_record_v3.json").is_file())

    def test_interval_score_matrix_uses_left_boundary(self) -> None:
        sim = np.arange(25, dtype=np.float64).reshape(5, 5)
        score = build_interval_score_matrix(sim)

        self.assertEqual(float(score[3, 4]), float(sim[2, 4]))
        self.assertEqual(float(score[0, 4]), 0.0)
        self.assertTrue(np.isnan(score[4, 3]))


if __name__ == "__main__":
    unittest.main()

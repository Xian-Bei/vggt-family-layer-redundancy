from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.eval.interval_results import (
    RECOVERY_INDEX_FILENAME,
    materialize_eval_interval_result,
)
from analysis.schema.v3 import EvalRecordV3


class IntervalResultsTest(unittest.TestCase):
    def test_eval_record_materializes_interval_summary_with_metric_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "i1_j2"
            record = EvalRecordV3(
                backbone="vggt",
                dataset="scannet_dev50",
                model_name="lower",
                scenes={
                    "0": {"scene_id": "scene0", "ape": 1.0, "cd": 0.25},
                    "1": {"scene_id": "scene1", "ape": 3.0, "cd": 0.75},
                },
                average={"ape": 2.0, "cd": 0.5},
            )
            record.save(out_dir / "eval_record_v3.json")

            summary_path = materialize_eval_interval_result(
                out_dir,
                selector="i1_j2",
                drop=[1, 2],
                source_eval_mode="lower",
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selector"], "i1_j2")
            self.assertEqual(summary["start"], 1)
            self.assertEqual(summary["end"], 2)
            self.assertEqual(summary["drop_count"], 2)
            self.assertEqual(summary["metrics_mean"]["cd"], 0.5)
            self.assertEqual(summary["metrics_mean"]["chamfer_distance"], 0.5)
            self.assertTrue((out_dir / "scenes.jsonl").is_file())
            self.assertTrue((out_dir.parent / "skip_eval_interval_index_v3.json").is_file())

    def test_recovery_result_writes_method_index_under_selector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            selector_root = Path(td) / "i1_j2"
            out_dir = selector_root / "recovery_ghost"
            EvalRecordV3(
                backbone="vggt",
                dataset="scannet_dev50",
                model_name="model",
                scenes={"0": {"scene_id": "scene0", "ape": 1.0}},
                average={"ape": 1.0},
            ).save(out_dir / "eval_record_v3.json")

            materialize_eval_interval_result(
                out_dir,
                selector="i1_j2",
                drop=[1, 2],
                source_eval_mode="model",
                recovery_plan_name="recovery_ghost",
            )

            index = json.loads(
                (selector_root / RECOVERY_INDEX_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(index["selector"], "i1_j2")
            self.assertIn("recovery_ghost", index["methods"])
            self.assertFalse((selector_root / "skip_eval_interval_index_v3.json").exists())


if __name__ == "__main__":
    unittest.main()

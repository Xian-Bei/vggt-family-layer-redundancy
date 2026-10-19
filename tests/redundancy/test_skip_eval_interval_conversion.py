from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.redundancy.convert_skip_eval_to_intervals import (
    INDEX_FILENAME,
    materialize_skip_eval_intervals,
)


class SkipEvalIntervalConversionTest(unittest.TestCase):
    def test_checkpoint_span_row_materializes_closed_interval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "skip_eval"
            ckpt = root / "checkpoints"
            ckpt.mkdir(parents=True)
            (ckpt / "scene0.jsonl").write_text(
                json.dumps(
                    {
                        "scene_id": "scene0",
                        "span_i": 1,
                        "span_j": 3,
                        "metrics": {"ape": 2.0},
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )

            index = materialize_skip_eval_intervals(root, commit=True)

            self.assertIn("i1_j3", index["intervals"])
            self.assertTrue((root / "i1_j3" / "summary.json").is_file())
            self.assertTrue((root / "i1_j3" / "scenes.jsonl").is_file())
            self.assertTrue((root / INDEX_FILENAME).is_file())
            summary = json.loads((root / "i1_j3" / "summary.json").read_text())
            self.assertEqual(summary["drop"], [1, 2, 3])
            self.assertEqual(summary["drop_count"], 3)


if __name__ == "__main__":
    unittest.main()

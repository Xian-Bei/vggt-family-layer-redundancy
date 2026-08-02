from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan.knee_point import load_knee_point


class KneePointTest(unittest.TestCase):
    def test_loads_selected_plan_and_drop_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "prune_plan_all.json").write_text(
                json.dumps(
                    {
                        "knee_points": {
                            "ours": {
                                "plan_name": "prune_span18",
                                "drop_count": 18,
                            },
                            "gt": {
                                "plan_name": "prune_span16",
                                "drop_count": 16,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            knee = load_knee_point(root)
            self.assertEqual(knee["plan_name"], "prune_span18")
            self.assertEqual(knee["drop_count"], 18)
            gt_knee = load_knee_point(root, strategy="gt")
            self.assertEqual(gt_knee["plan_name"], "prune_span16")

    def test_rejects_bundle_without_knee_point(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "prune_plan_all.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "knee_points.ours.plan_name is missing"):
                load_knee_point(root)


if __name__ == "__main__":
    unittest.main()

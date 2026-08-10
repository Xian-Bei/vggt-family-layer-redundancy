from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.recovery.shared_results import (
    eval_result_complete,
    recovery_result_complete,
    selectors_for_strategies,
)


class SharedResultsTest(unittest.TestCase):
    def _write_bundle(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "selector": "i1_j2", "i": 1, "j": 2, "start": 1, "end": 2,
            "drop": [1, 2], "drop_count": 2,
        }
        (root / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt", "dataset": "test", "num_units": 4,
                    "intervals": {"i1_j2": record},
                    "sets": {},
                    "strategies": {
                        "ours": {"points": {"2": "i1_j2"}},
                        "gt": {"points": {"2": "i1_j2"}},
                    },
                    "spans": {"2": record},
                }
            ),
            encoding="utf-8",
        )

    def test_recovery_requires_complete_matching_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            out = root / "recovery" / "i1_j2" / "recovery_ghost"
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            (out / "config.json").write_text(
                json.dumps(
                    {
                        "backbone": "vggt",
                        "prune_plan_name": "prune_span2",
                        "recovery_plan_name": "recovery_ghost",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                recovery_result_complete(
                    out, plan_root=root, backbone="vggt", selector="i1_j2",
                    recovery_plan_name="recovery_ghost",
                )
            )
            (out / "config.json").unlink()
            self.assertFalse(
                recovery_result_complete(
                    out, plan_root=root, backbone="vggt", selector="i1_j2",
                    recovery_plan_name="recovery_ghost",
                )
            )

    def test_eval_requires_record_and_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "i1_j2" / "recovery_ghost"
            out.mkdir(parents=True)
            (out / "eval_record_v3.json").write_text("{}", encoding="utf-8")
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "recovery_plan_name": "recovery_ghost",
                    }
                ),
                encoding="utf-8",
            )
            # An empty eval record is not a complete result.
            self.assertFalse(
                eval_result_complete(
                    out, selector="i1_j2", recovery_plan_name="recovery_ghost"
                )
            )
            (out / "eval_record_v3.json").write_text(
                json.dumps({"schema_version": "v3"}), encoding="utf-8"
            )
            self.assertTrue(
                eval_result_complete(
                    out, selector="i1_j2", recovery_plan_name="recovery_ghost"
                )
            )

    def test_strategy_union_deduplicates_same_physical_selector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            self.assertEqual(
                selectors_for_strategies(
                    root, ["ours", "gt"], ["prune_span2"]
                ),
                ["i1_j2"],
            )


if __name__ == "__main__":
    unittest.main()

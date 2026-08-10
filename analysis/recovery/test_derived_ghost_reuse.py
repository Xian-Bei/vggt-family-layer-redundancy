from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from analysis.eval.eval_v3 import _try_link_derived_ghost_eval
from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION, UnitPrunePlan
from analysis.recovery.recovery_prune_v3 import _try_link_derived_ghost_recovery


class DerivedGhostReuseTest(unittest.TestCase):
    def _write_bundle(self, root: Path) -> None:
        record = {
            "selector": "i1_j2", "i": 1, "j": 2, "start": 1, "end": 2,
            "drop": [1, 2], "drop_count": 2,
        }
        (root / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt", "dataset": "test", "num_units": 4,
                    "intervals": {"i1_j2": record}, "sets": {},
                    "strategies": {"ours": {"points": {"2": "i1_j2"}}},
                    "spans": {"2": record},
                }
            ),
            encoding="utf-8",
        )

    def test_recovery_link_requires_complete_ghost_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            ghost = root / "recovery" / "i1_j2" / "recovery_ghost"
            (ghost / "checkpoints").mkdir(parents=True)
            (ghost / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            derived = ghost.parent / "recovery_ghost_interval"
            plan = UnitPrunePlan(
                backbone="vggt", dataset="test", num_units=4,
                drop=[1, 2], drop_length=2,
            )
            kwargs = dict(
                method="recovery_ghost_interval", prune_plan=plan,
                recovery_plan_name="recovery_ghost_interval", out_dir=derived,
                plan_root=root, backbone="vggt", selector="i1_j2",
            )
            self.assertFalse(_try_link_derived_ghost_recovery(**kwargs))
            self.assertFalse(derived.exists())
            (ghost / "config.json").write_text(
                json.dumps(
                    {
                        "backbone": "vggt", "prune_plan_name": "i1_j2",
                        "recovery_plan_name": "recovery_ghost",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_try_link_derived_ghost_recovery(**kwargs))
            self.assertTrue(derived.is_symlink())

    def test_eval_link_requires_record_and_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ghost_recovery = root / "recovery" / "i1_j2" / "recovery_ghost"
            (ghost_recovery / "checkpoints").mkdir(parents=True)
            (ghost_recovery / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            derived_recovery = ghost_recovery.parent / "recovery_ghost_interval"
            derived_recovery.symlink_to("recovery_ghost", target_is_directory=True)

            ghost_eval = root / "test" / "eval" / "recovery" / "i1_j2" / "recovery_ghost"
            ghost_eval.mkdir(parents=True)
            (ghost_eval / "eval_record_v3.json").write_text(
                json.dumps({"schema_version": "v3"}), encoding="utf-8"
            )
            derived_eval = ghost_eval.parent / "recovery_ghost_interval"
            args = argparse.Namespace(
                recovery_plan_name="recovery_ghost_interval",
                checkpoint=derived_recovery / "checkpoints" / "latest.pt",
            )
            jobs = [{"output_dir": str(derived_eval), "output_dataset": "test"}]
            self.assertFalse(_try_link_derived_ghost_eval(args, jobs))
            self.assertFalse(derived_eval.exists())
            (ghost_eval / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "recovery_plan_name": "recovery_ghost",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_try_link_derived_ghost_eval(args, jobs))
            self.assertTrue(derived_eval.is_symlink())

    def test_eval_links_complete_datasets_and_leaves_only_missing_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ghost_recovery = root / "recovery" / "i1_j2" / "recovery_ghost"
            (ghost_recovery / "checkpoints").mkdir(parents=True)
            (ghost_recovery / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            derived_recovery = ghost_recovery.parent / "recovery_ghost_interval"
            derived_recovery.symlink_to("recovery_ghost", target_is_directory=True)
            args = argparse.Namespace(
                recovery_plan_name="recovery_ghost_interval",
                checkpoint=derived_recovery / "checkpoints" / "latest.pt",
            )

            outputs = []
            for dataset in ("complete", "missing"):
                ghost = root / dataset / "eval" / "recovery" / "i1_j2" / "recovery_ghost"
                ghost.mkdir(parents=True)
                (ghost / "eval_record_v3.json").write_text(
                    json.dumps({"schema_version": "v3"}), encoding="utf-8"
                )
                if dataset == "complete":
                    (ghost / "summary.json").write_text(
                        json.dumps(
                            {
                                "selector": "i1_j2",
                                "recovery_plan_name": "recovery_ghost",
                            }
                        ),
                        encoding="utf-8",
                    )
                outputs.append(ghost.parent / "recovery_ghost_interval")
            jobs = [
                {"output_dir": str(output), "output_dataset": dataset}
                for output, dataset in zip(outputs, ("complete", "missing"))
            ]
            self.assertFalse(_try_link_derived_ghost_eval(args, jobs))
            self.assertTrue(outputs[0].is_symlink())
            self.assertFalse(outputs[1].exists())
            self.assertEqual([job["output_dataset"] for job in jobs], ["missing"])


if __name__ == "__main__":
    unittest.main()

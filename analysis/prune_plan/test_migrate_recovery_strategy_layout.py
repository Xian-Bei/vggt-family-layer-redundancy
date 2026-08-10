from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.migrate_recovery_strategy_layout import migrate


class RecoveryMigrationTest(unittest.TestCase):
    def _backbone(self, root: Path) -> Path:
        backbone = root / "vggt"
        backbone.mkdir(parents=True)
        record = {
            "selector": "i1_j2", "i": 1, "j": 2, "start": 1, "end": 2,
            "drop": [1, 2], "drop_count": 2,
        }
        (backbone / "prune_plan_all.json").write_text(
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
        return backbone

    def _legacy_result(self, backbone: Path, strategy: str) -> None:
        eval_dir = (
            backbone / "test" / "eval" / "recovery" / strategy
            / "recovery_ghost" / "prune_span2"
        )
        eval_dir.mkdir(parents=True)
        (eval_dir / "summary.json").write_text("{}", encoding="utf-8")
        (eval_dir / "eval_record_v3.json").write_text("{}", encoding="utf-8")
        checkpoint_dir = (
            backbone / "recovery" / strategy / "prune_span2"
            / "recovery_ghost" / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "latest.pt").write_bytes(b"checkpoint")

    def test_apply_is_canonical_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            backbone = self._backbone(root)
            self._legacy_result(backbone, "ours")
            self.assertEqual(migrate(root, apply=False), 2)
            self.assertEqual(migrate(root, apply=True), 2)
            self.assertTrue(
                (backbone / "test" / "eval" / "recovery" / "i1_j2"
                 / "recovery_ghost" / "eval_record_v3.json").is_file()
            )
            self.assertTrue(
                (backbone / "recovery" / "i1_j2" / "recovery_ghost"
                 / "checkpoints" / "latest.pt").is_file()
            )
            self.assertEqual(migrate(root, apply=False), 0)

    def test_refuses_duplicate_shared_destination(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            backbone = self._backbone(root)
            self._legacy_result(backbone, "ours")
            self._legacy_result(backbone, "gt")
            with self.assertRaises(RuntimeError):
                migrate(root, apply=False)

    def test_rewrites_derived_method_symlink_for_selector_first_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            backbone = self._backbone(root)
            self._legacy_result(backbone, "ours")
            derived_root = (
                backbone / "test" / "eval" / "recovery" / "ours"
                / "recovery_ghost_interval"
            )
            derived_root.mkdir(parents=True)
            (derived_root / "prune_span2").symlink_to(
                Path("../recovery_ghost/prune_span2"), target_is_directory=True
            )
            self.assertEqual(migrate(root, apply=True), 3)
            derived = (
                backbone / "test" / "eval" / "recovery" / "i1_j2"
                / "recovery_ghost_interval"
            )
            self.assertTrue(derived.is_symlink())
            self.assertEqual(derived.readlink(), Path("recovery_ghost"))
            self.assertTrue((derived / "eval_record_v3.json").is_file())
            self.assertEqual(migrate(root, apply=False), 0)


if __name__ == "__main__":
    unittest.main()

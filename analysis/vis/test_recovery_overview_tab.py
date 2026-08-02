from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.recovery_overview_tab import (
    discover_recovery_overview_datasets,
    discover_recovery_overview_options,
    render_recovery_overview,
)


class RecoveryOverviewTabTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        backbone = root / "vggt"
        backbone.mkdir(parents=True)
        (backbone / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt",
                    "dataset": "scannet_dev50",
                    "num_units": 4,
                    "intervals": {},
                    "strategies": {
                        "ours": {"display_name": "Ours", "points": {"2": "i1_j2"}}
                    },
                    "spans": {},
                }
            ),
            encoding="utf-8",
        )
        upper = backbone / "scannet_dev50" / "eval" / "upper"
        upper.mkdir(parents=True)
        EvalRecordV3(
            backbone="vggt",
            dataset="scannet_dev50",
            model_name="upper",
            average={"ape": 1.0, "are": 1.0, "rpe_trans": 1.0, "rpe_rot": 1.0,
                     "acc": 1.0, "comp": 1.0, "chamfer_distance": 1.0},
        ).save(upper / "eval_record_v3.json")
        for result_root in (
            backbone / "scannet_dev50" / "eval" / "pruning",
            backbone / "scannet_dev50" / "eval" / "recovery" / "recovery_linearpatch",
        ):
            target = result_root / "i1_j2"
            target.mkdir(parents=True)
            (target / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "drop_count": 2,
                        "num_scenes": 1,
                        "metrics_mean": {"ape": 2.0, "are": 2.0, "rpe_trans": 2.0,
                                         "rpe_rot": 2.0, "acc": 2.0, "comp": 2.0,
                                         "chamfer_distance": 2.0},
                    }
                ),
                encoding="utf-8",
            )

    def test_discovers_controls_and_redraws_from_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["vggt"], ["recovery_linearpatch"]),
            )
            self.assertEqual(
                discover_recovery_overview_datasets(root),
                ["mean", "scannet_dev50"],
            )
            images, status = render_recovery_overview(
                root, ["vggt"], ["recovery_linearpatch"], "scannet_dev50"
            )
            self.assertEqual(len(images), 1, status)
            self.assertEqual(images[0][0].ndim, 3)
            self.assertGreater(images[0][0].shape[0], 1000)
            self.assertIn("redrawn from current summaries", status)
            self.assertIn("/ scannet_dev50", status)

    def test_discovers_plan_before_evaluation_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_ghost.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["pi3"], ["recovery_ghost"]),
            )

    def test_discovers_methods_from_recovery_plan_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_plan_all.json").write_text(
                json.dumps(
                    {
                        "methods": {
                            "recovery_ghost_interval": {},
                            "recovery_ghost_interval_sequential": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                discover_recovery_overview_options(root),
                (
                    ["pi3"],
                    [
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                    ],
                ),
            )


if __name__ == "__main__":
    unittest.main()

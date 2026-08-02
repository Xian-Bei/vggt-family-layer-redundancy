from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
)
from analysis.prune_plan.stage7 import (
    existing_summary_path,
    missing_plans_for_dataset,
    selected_plans,
)


def _write_bundle(root: Path) -> None:
    payload = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "scannet_dev50,nuscenes_train",
        "num_units": 8,
        "intervals": {
            "i1_j2": {
                "selector": "i1_j2",
                "drop": [1, 2],
                "drop_count": 2,
            }
        },
        "sets": {
            "ours_k2": {
                "selector": "ours_k2",
                "selector_kind": "set",
                "drop": [1, 4],
                "drop_count": 2,
                "num_intervals": 2,
            }
        },
        "strategies": {
            "ours": {"points": {"2": "ours_k2"}},
            "gt": {"points": {"2": "i1_j2"}},
        },
        "spans": {},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


class Stage7Test(unittest.TestCase):
    def test_selected_plans_deduplicate_physical_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root)
            plans = selected_plans(
                root,
                strategies=["ours", "ours"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
        self.assertEqual([plan.selector for plan in plans], ["i1_j1_i4_j4"])
        self.assertEqual(plans[0].references, ("ours@2",))

    def test_summary_only_reversed_result_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result_root = Path(td)
            reversed_dir = result_root / "i4_j4_i1_j1"
            reversed_dir.mkdir()
            (reversed_dir / "summary.json").write_text("{}", encoding="utf-8")
            path = existing_summary_path(result_root, "i1_j1_i4_j4")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.parent.name, "i4_j4_i1_j1")

    def test_training_missing_single_is_fatal_but_double_is_scheduled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root)
            single = selected_plans(
                root,
                strategies=["gt"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "training single-interval"
            ):
                missing_plans_for_dataset(
                    result_root=root / "results",
                    plans=single,
                    training_dataset=True,
                    dataset="scannet_dev50",
                )
            double = selected_plans(
                root,
                strategies=["ours"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            missing = missing_plans_for_dataset(
                result_root=root / "results",
                plans=double,
                training_dataset=True,
                dataset="scannet_dev50",
            )
        self.assertEqual([plan.selector for plan in missing], ["i1_j1_i4_j4"])


if __name__ == "__main__":
    unittest.main()

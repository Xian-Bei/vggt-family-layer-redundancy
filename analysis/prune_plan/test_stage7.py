from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
)
from analysis.prune_plan.stage7 import (
    DatasetSpec,
    existing_summary_path,
    missing_plan_schedule,
    missing_plans_for_dataset,
    selected_plans,
    main as stage7_main,
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

    def test_batch_schedule_groups_missing_datasets_by_selector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root)
            plans = selected_plans(
                root,
                strategies=["ours"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            complete = (
                root / "dataset_a" / "eval" / "pruning" / "i4_j4_i1_j1"
            )
            complete.mkdir(parents=True)
            (complete / "summary.json").write_text("{}", encoding="utf-8")
            schedule, dataset_results = missing_plan_schedule(
                prune_plan_root=root,
                plans=plans,
                datasets=[
                    DatasetSpec("DATASET_A", "dataset_a", False),
                    DatasetSpec("DATASET_B", "dataset_b", True),
                    DatasetSpec("DATASET_C", "dataset_c", False),
                ],
            )

        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0].selector, "i1_j1_i4_j4")
        self.assertEqual(schedule[0].dataset_keys, ("DATASET_B", "DATASET_C"))
        self.assertEqual(
            [len(missing) for _dataset, missing in dataset_results],
            [0, 1, 1],
        )

    def test_batch_cli_emits_tsv_schedule_and_logs_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bundle(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "stage7",
                "--prune-plan-root",
                str(root),
                "--strategies",
                "ours",
                "--min-drop",
                "2",
                "--max-drop",
                "2",
                "--drop-step",
                "2",
                "--dataset-spec",
                "DATASET_B:dataset_b:1",
                "--dataset-spec",
                "DATASET_C:dataset_c:0",
            ]
            with (
                mock.patch("sys.argv", argv),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                stage7_main()

        self.assertEqual(
            stdout.getvalue(),
            "i1_j1_i4_j4\tDATASET_B,DATASET_C\n",
        )
        self.assertIn("[missing]", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

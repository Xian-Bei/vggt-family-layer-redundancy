from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.eval_plots import GT_MEAN_METRICS
from analysis.prune_plan.recovery_plan import RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.recovery.shared_results import (
    PRUNE_FAMILY_STRATEGIES,
    best_recovery_result,
    eval_result_complete,
    main,
    recovery_result_complete,
    selectors_for_strategies,
)


class SharedResultsTest(unittest.TestCase):
    def test_prune_families_only_adapt_single_vs_double(self) -> None:
        self.assertEqual(PRUNE_FAMILY_STRATEGIES["ours"], ("ours_single", "ours_double"))
        self.assertEqual(PRUNE_FAMILY_STRATEGIES["gt"], ("gt_single", "gt_double"))

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

    def test_best_recovery_result_uses_final_recovered_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            bundle_path = root / "prune_plan_all.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            gt_record = {
                "selector": "i3_j4", "i": 3, "j": 4, "start": 3, "end": 4,
                "drop": [3, 4], "drop_count": 2,
            }
            bundle["intervals"]["i3_j4"] = gt_record
            bundle["strategies"]["gt"]["points"]["2"] = "i3_j4"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            payload = {
                "datasets": {
                    dataset: {
                        "normalization_profile": {
                            "upper_average": {metric: 0.0 for metric in GT_MEAN_METRICS},
                            "maximum_single_interval_metric": {
                                metric: 10.0 for metric in GT_MEAN_METRICS
                            },
                        },
                    }
                    for dataset in ("train_a", "train_b")
                }
            }
            for dataset in ("train_a", "train_b"):
                for selector, value in (("i1_j2", 1.0), ("i3_j4", 4.0)):
                    out = root / dataset / "eval" / "recovery" / selector / "recovery_ghost"
                    out.mkdir(parents=True)
                    (out / "summary.json").write_text(
                        json.dumps({
                            "selector": selector,
                            "drop_count": 2,
                            "recovery_plan_name": "recovery_ghost",
                            "metrics_mean": {
                                metric: value for metric in GT_MEAN_METRICS
                            },
                        }),
                        encoding="utf-8",
                    )
            with patch(
                "analysis.recovery.shared_results.collect_pruning_eval_payload",
                return_value=payload,
            ):
                result = best_recovery_result(
                    root,
                    drop_count=2,
                    datasets=["train_a", "train_b"],
                    recovery_plan_name="recovery_ghost",
                    strategies=["gt", "ours"],
                )
        self.assertEqual(result["strategy"], "ours")
        self.assertEqual(result["selector"], "i1_j2")
        self.assertAlmostEqual(result["score"], 0.1)

    def test_selector_cli_prints_stable_union(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "selectors-for-strategies",
                        "--plan-root",
                        str(root),
                        "--strategies",
                        "ours,gt",
                        "--selectors",
                        "prune_span2",
                    ]
                )
            self.assertEqual(output.getvalue(), "i1_j2\n")

    def test_recovery_plan_names_cli_prints_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "recovery_plan_all.json").write_text(
                json.dumps(
                    {
                        "schema_version": RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
                        "backbone": "vggt",
                        "dataset": "test",
                        "num_units": 4,
                        "methods": {
                            "recovery_replaceme_ls": {
                                "method": "recovery_replaceme_ls"
                            },
                            "recovery_ghost": {"method": "recovery_ghost"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                main(["recovery-plan-names", "--plan-root", str(root)])
            self.assertEqual(
                output.getvalue(), "recovery_ghost,recovery_replaceme_ls\n"
            )


if __name__ == "__main__":
    unittest.main()

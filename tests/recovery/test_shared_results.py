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
from analysis.prune_plan.recovery_plan import (
    RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
    TOKEN_GROUPED_RECOVERY_BASE_METHODS,
    recovery_method_semantics_version,
)
from analysis.recovery.shared_results import (
    PRUNE_FAMILY_STRATEGIES,
    SINGLE_DOUBLE_MEAN_MARGIN,
    SINGLE_DOUBLE_SELECTION_DATASETS,
    best_recovery_result,
    eval_result_complete,
    main,
    recovery_result_complete,
    seven_metric_mean_score,
    single_double_mean_decision,
    selectors_for_strategies,
)


class SharedResultsTest(unittest.TestCase):
    def test_prune_families_only_adapt_single_vs_double(self) -> None:
        self.assertEqual(PRUNE_FAMILY_STRATEGIES["CKA"], ("CKA_single", "CKA_double"))
        self.assertEqual(PRUNE_FAMILY_STRATEGIES["retention"], ("retention_single", "retention_double"))

    def _write_bundle(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "selector": "i1_j2", "i": 1, "j": 2, "start": 1, "end": 2,
            "drop": [1, 2], "drop_count": 2,
        }
        multi_record = {
            "selector": "i1_j1_i3_j3",
            "selector_kind": "set",
            "drop": [1, 3],
            "drop_count": 2,
        }
        (root / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt", "dataset": "test", "num_units": 4,
                    "intervals": {"i1_j2": record},
                    "sets": {"i1_j1_i3_j3": multi_record},
                    "strategies": {
                        "CKA": {"points": {"2": "i1_j2"}},
                        "retention": {"points": {"2": "i1_j2"}},
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
            config = {
                "backbone": "vggt",
                "prune_plan_name": "prune_span2",
                "recovery_plan_name": "recovery_ghost",
            }
            config_path = out / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertTrue(
                recovery_result_complete(
                    out, plan_root=root, backbone="vggt", selector="i1_j2",
                    recovery_plan_name="recovery_ghost",
                )
            )
            config["fit_signature"] = "expected-signature"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertTrue(
                recovery_result_complete(
                    out, plan_root=root, backbone="vggt", selector="i1_j2",
                    recovery_plan_name="recovery_ghost",
                    expected_fit_signature="expected-signature",
                )
            )
            self.assertFalse(
                recovery_result_complete(
                    out, plan_root=root, backbone="vggt", selector="i1_j2",
                    recovery_plan_name="recovery_ghost",
                    expected_fit_signature="different-signature",
                )
            )
            config_path.unlink()
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
                        "recovery_semantics_version": recovery_method_semantics_version(
                            "recovery_ghost"
                        ),
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

    def test_legacy_single_component_is_reused_but_multi_requires_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            for method in ("recovery_replaceme_cos",):
                for selector in (
                    "i1_j2",
                    "i1_j1_i3_j3",
                ):
                    legacy_complete = selector == "i1_j2"
                    version = recovery_method_semantics_version(method)
                    out = root / "recovery" / selector / method
                    (out / "checkpoints").mkdir(parents=True)
                    (out / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
                    config = {
                        "backbone": "vggt",
                        "prune_plan_name": selector,
                        "recovery_plan_name": method,
                    }
                    config_path = out / "config.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    self.assertEqual(
                        recovery_result_complete(
                            out,
                            plan_root=root,
                            backbone="vggt",
                            selector=selector,
                            recovery_plan_name=method,
                        ),
                        legacy_complete,
                    )
                    config["recovery_semantics_version"] = version
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    self.assertTrue(
                        recovery_result_complete(
                            out,
                            plan_root=root,
                            backbone="vggt",
                            selector=selector,
                            recovery_plan_name=method,
                        )
                    )

                    eval_out = root / "eval" / selector / method
                    eval_out.mkdir(parents=True)
                    (eval_out / "eval_record_v3.json").write_text(
                        json.dumps({"schema_version": "v3"}), encoding="utf-8"
                    )
                    summary = {"selector": selector, "recovery_plan_name": method}
                    summary_path = eval_out / "summary.json"
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    self.assertEqual(
                        eval_result_complete(
                            eval_out, selector=selector, recovery_plan_name=method
                        ),
                        legacy_complete,
                    )
                    summary["recovery_semantics_version"] = version
                    summary_path.write_text(json.dumps(summary), encoding="utf-8")
                    self.assertTrue(
                        eval_result_complete(
                            eval_out, selector=selector, recovery_plan_name=method
                        )
                    )

    def test_token_grouped_ghost_has_independent_versioned_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            for method, base in TOKEN_GROUPED_RECOVERY_BASE_METHODS.items():
                with self.subTest(method=method):
                    out = root / "recovery" / "i1_j2" / method
                    (out / "checkpoints").mkdir(parents=True)
                    (out / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
                    config = {
                        "backbone": "vggt",
                        "prune_plan_name": "i1_j2",
                        "recovery_plan_name": method,
                    }
                    config_path = out / "config.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    self.assertFalse(
                        recovery_result_complete(
                            out,
                            plan_root=root,
                            backbone="vggt",
                            selector="i1_j2",
                            recovery_plan_name=method,
                        )
                    )
                    config["recovery_semantics_version"] = (
                        recovery_method_semantics_version(method)
                    )
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    self.assertTrue(
                        recovery_result_complete(
                            out,
                            plan_root=root,
                            backbone="vggt",
                            selector="i1_j2",
                            recovery_plan_name=method,
                        )
                    )
                    self.assertNotEqual(
                        out,
                        root / "recovery" / "i1_j2" / base,
                    )

    def test_old_full_target_cosine_is_reused_only_for_single_interval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            for selector, expected in (("i1_j2", True), ("i1_j1_i3_j3", False)):
                out = root / "recovery" / selector / "recovery_replaceme_cos"
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
                (out / "config.json").write_text(
                    json.dumps(
                        {
                            "backbone": "vggt",
                            "prune_plan_name": selector,
                            "recovery_plan_name": "recovery_replaceme_cos",
                            "recovery_semantics_version": (
                                "current_source_full_target_v2"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    recovery_result_complete(
                        out,
                        plan_root=root,
                        backbone="vggt",
                        selector=selector,
                        recovery_plan_name="recovery_replaceme_cos",
                    ),
                    expected,
                )

    def test_single_interval_current_target_ghost_symlink_remains_reusable(self) -> None:
        method = "recovery_ghost_interval_sequential"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            ghost = root / "recovery" / "i1_j2" / "recovery_ghost"
            (ghost / "checkpoints").mkdir(parents=True)
            (ghost / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            (ghost / "config.json").write_text(
                json.dumps(
                    {
                        "backbone": "vggt",
                        "prune_plan_name": "i1_j2",
                        "recovery_plan_name": "recovery_ghost",
                        "recovery_semantics_version": recovery_method_semantics_version(
                            "recovery_ghost"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            derived = ghost.parent / method
            derived.symlink_to("recovery_ghost", target_is_directory=True)
            self.assertTrue(
                recovery_result_complete(
                    derived,
                    plan_root=root,
                    backbone="vggt",
                    selector="i1_j2",
                    recovery_plan_name=method,
                )
            )

            ghost_eval = root / "eval" / "i1_j2" / "recovery_ghost"
            ghost_eval.mkdir(parents=True)
            (ghost_eval / "eval_record_v3.json").write_text(
                json.dumps({"schema_version": "v3"}), encoding="utf-8"
            )
            (ghost_eval / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "recovery_plan_name": "recovery_ghost",
                        "recovery_semantics_version": recovery_method_semantics_version(
                            "recovery_ghost"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            derived_eval = ghost_eval.parent / method
            derived_eval.symlink_to("recovery_ghost", target_is_directory=True)
            self.assertTrue(
                eval_result_complete(
                    derived_eval, selector="i1_j2", recovery_plan_name=method
                )
            )

    def test_strategy_union_deduplicates_same_physical_selector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            self.assertEqual(
                selectors_for_strategies(
                    root, ["CKA", "retention"], ["prune_span2"]
                ),
                ["i1_j2"],
            )

    def test_best_recovery_result_uses_all_seven_final_recovered_metrics(self) -> None:
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
            bundle["strategies"]["CKA_single"] = {"points": {"2": "i1_j2"}}
            bundle["strategies"]["CKA_double"] = {"points": {"2": "i3_j4"}}
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
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
                                metric: (
                                    value if metric in GT_MEAN_METRICS[:4]
                                    else 1000.0 if selector == "i1_j2" else 0.0
                                )
                                for metric in GT_MEAN_METRICS
                            },
                        }),
                        encoding="utf-8",
                    )
            result = best_recovery_result(
                root,
                drop_count=2,
                datasets=["train_a", "train_b"],
                recovery_plan_name="recovery_ghost",
                strategies=PRUNE_FAMILY_STRATEGIES["CKA"],
            )
            for dataset in ("train_a", "train_b"):
                for selector, value in (("i1_j2", 1.0), ("i3_j4", 0.91)):
                    path = (
                        root / dataset / "eval" / "recovery" / selector
                        / "recovery_ghost" / "summary.json"
                    )
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    summary["metrics_mean"] = {
                        metric: value for metric in GT_MEAN_METRICS
                    }
                    path.write_text(json.dumps(summary), encoding="utf-8")
            margin_result = best_recovery_result(
                root,
                drop_count=2,
                datasets=["train_a", "train_b"],
                recovery_plan_name="recovery_ghost",
                strategies=PRUNE_FAMILY_STRATEGIES["CKA"],
            )
        self.assertEqual(result["strategy"], "CKA_double")
        self.assertEqual(result["selector"], "i3_j4")
        self.assertAlmostEqual(result["score"], 16.0 / 7.0)
        self.assertEqual(margin_result["strategy"], "CKA_single")
        self.assertTrue(margin_result["margin_override"])

    def test_shared_single_double_mean_margin_prefers_single_within_threshold(self) -> None:
        self.assertEqual(SINGLE_DOUBLE_MEAN_MARGIN, 0.1)
        self.assertEqual(
            SINGLE_DOUBLE_SELECTION_DATASETS,
            (
                "scannet_dev50",
                "7scenes_train",
                "nuscenes_train",
                "waymo_training",
            ),
        )
        self.assertEqual(
            seven_metric_mean_score({metric: 0.91 for metric in GT_MEAN_METRICS}),
            0.91,
        )
        within = single_double_mean_decision(1.0, 0.91)
        self.assertEqual(within["winner"], "single")
        self.assertTrue(within["margin_override"])
        beyond = single_double_mean_decision(1.0, 0.89)
        self.assertEqual(beyond["winner"], "double")
        self.assertFalse(beyond["margin_override"])

    def test_best_recovery_result_rejects_cross_family_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one pruning family"):
            best_recovery_result(
                Path("/unused"),
                drop_count=2,
                datasets=["train"],
                recovery_plan_name="recovery_ghost",
                strategies=["CKA_single", "retention_single"],
            )

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
                        "CKA,retention",
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

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from analysis.eval.eval_v3 import _try_link_derived_ghost_eval
from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION, UnitPrunePlan
from analysis.prune_plan.recovery_plan import recovery_method_semantics_version
from analysis.prune_plan.recovery_plan import TOKEN_GROUPED_RECOVERY_BASE_METHODS
from analysis.recovery.recovery_prune_v3 import _try_link_derived_ghost_recovery
from analysis.recovery.shared_results import eval_result_complete, recovery_result_complete


class DerivedGhostReuseTest(unittest.TestCase):
    def _write_bundle(self, root: Path) -> None:
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
                    "strategies": {"CKA": {"points": {"2": "i1_j2"}}},
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
            config = {
                "backbone": "vggt",
                "prune_plan_name": "i1_j2",
                "recovery_plan_name": "recovery_ghost",
            }
            config_path = ghost / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
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
                        "recovery_semantics_version": recovery_method_semantics_version(
                            "recovery_ghost"
                        ),
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
                                "recovery_semantics_version": recovery_method_semantics_version(
                                    "recovery_ghost"
                                ),
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

    def test_pi3_token_grouped_recovery_links_to_corresponding_base(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            plan = UnitPrunePlan(
                backbone="pi3", dataset="test", num_units=4,
                drop=[1, 3], drop_length=2,
            )
            selector = "i1_j1_i3_j3"
            for grouped, base in TOKEN_GROUPED_RECOVERY_BASE_METHODS.items():
                with self.subTest(grouped=grouped):
                    source = root / "recovery" / selector / base
                    (source / "checkpoints").mkdir(parents=True, exist_ok=True)
                    (source / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
                    (source / "config.json").write_text(
                        json.dumps(
                            {
                                "backbone": "pi3",
                                "prune_plan_name": selector,
                                "recovery_plan_name": base,
                                "recovery_semantics_version": (
                                    recovery_method_semantics_version(base)
                                ),
                            }
                        ),
                        encoding="utf-8",
                    )
                    target = source.parent / grouped
                    self.assertTrue(
                        _try_link_derived_ghost_recovery(
                            method=grouped,
                            prune_plan=plan,
                            recovery_plan_name=grouped,
                            out_dir=target,
                            plan_root=root,
                            backbone="pi3",
                            selector=selector,
                        )
                    )
                    self.assertTrue(target.is_symlink())
                    self.assertEqual(target.resolve(), source.resolve())
                    self.assertTrue(
                        recovery_result_complete(
                            target,
                            plan_root=root,
                            backbone="pi3",
                            selector=selector,
                            recovery_plan_name=grouped,
                        )
                    )

    def test_pi3_autogroup_does_not_link_to_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            plan = UnitPrunePlan(
                backbone="pi3", dataset="test", num_units=4,
                drop=[1, 3], drop_length=2,
            )
            selector = "i1_j1_i3_j3"
            source = (
                root / "recovery" / selector
                / "recovery_ghost_interval_sequential"
            )
            (source / "checkpoints").mkdir(parents=True)
            (source / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "backbone": "pi3",
                        "prune_plan_name": selector,
                        "recovery_plan_name": (
                            "recovery_ghost_interval_sequential"
                        ),
                        "recovery_semantics_version": (
                            recovery_method_semantics_version(
                                "recovery_ghost_interval_sequential"
                            )
                        ),
                    }
                ),
                encoding="utf-8",
            )
            target = (
                source.parent
                / "recovery_ghost_interval_sequential_autogroup"
            )
            self.assertFalse(
                _try_link_derived_ghost_recovery(
                    method="recovery_ghost_interval_sequential_autogroup",
                    prune_plan=plan,
                    recovery_plan_name=(
                        "recovery_ghost_interval_sequential_autogroup"
                    ),
                    out_dir=target,
                    plan_root=root,
                    backbone="pi3",
                    selector=selector,
                )
            )
            self.assertFalse(target.exists())

    def test_pi3_token_grouped_direct_outputs_are_migrated_to_links(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_bundle(root)
            grouped = "recovery_ghost_interval_sequential_grouped"
            direct = root / "recovery" / "i1_j2" / grouped
            (direct / "checkpoints").mkdir(parents=True)
            (direct / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
            (direct / "config.json").write_text(
                json.dumps(
                    {
                        "backbone": "pi3",
                        "prune_plan_name": "i1_j2",
                        "recovery_plan_name": grouped,
                        "recovery_semantics_version": (
                            recovery_method_semantics_version(grouped)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                recovery_result_complete(
                    direct,
                    plan_root=root,
                    backbone="pi3",
                    selector="i1_j2",
                    recovery_plan_name=grouped,
                )
            )

            direct_eval = root / "test" / "eval" / "recovery" / "i1_j2" / grouped
            direct_eval.mkdir(parents=True)
            (direct_eval / "eval_record_v3.json").write_text(
                json.dumps({"schema_version": "v3", "backbone": "pi3"}),
                encoding="utf-8",
            )
            (direct_eval / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "recovery_plan_name": grouped,
                        "recovery_semantics_version": (
                            recovery_method_semantics_version(grouped)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                eval_result_complete(
                    direct_eval,
                    selector="i1_j2",
                    recovery_plan_name=grouped,
                )
            )

    def test_pi3_token_grouped_eval_links_to_corresponding_base(self) -> None:
        for grouped, base in TOKEN_GROUPED_RECOVERY_BASE_METHODS.items():
            with self.subTest(grouped=grouped), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                base_recovery = root / "recovery" / "i1_j2" / base
                (base_recovery / "checkpoints").mkdir(parents=True)
                (base_recovery / "checkpoints" / "latest.pt").write_bytes(
                    b"checkpoint"
                )
                grouped_recovery = base_recovery.parent / grouped
                grouped_recovery.symlink_to(base, target_is_directory=True)

                base_eval = root / "test" / "eval" / "recovery" / "i1_j2" / base
                base_eval.mkdir(parents=True)
                (base_eval / "eval_record_v3.json").write_text(
                    json.dumps({"schema_version": "v3", "backbone": "pi3"}),
                    encoding="utf-8",
                )
                (base_eval / "summary.json").write_text(
                    json.dumps(
                        {
                            "selector": "i1_j2",
                            "recovery_plan_name": base,
                            "recovery_semantics_version": (
                                recovery_method_semantics_version(base)
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                grouped_eval = base_eval.parent / grouped
                args = argparse.Namespace(
                    backbone="pi3",
                    recovery_plan_name=grouped,
                    checkpoint=grouped_recovery / "checkpoints" / "latest.pt",
                )
                jobs = [{"output_dir": str(grouped_eval), "output_dataset": "test"}]
                self.assertTrue(_try_link_derived_ghost_eval(args, jobs))
                self.assertTrue(grouped_eval.is_symlink())
                self.assertEqual(grouped_eval.resolve(), base_eval.resolve())
                self.assertTrue(
                    eval_result_complete(
                        grouped_eval,
                        selector="i1_j2",
                        recovery_plan_name=grouped,
                    )
                )


if __name__ == "__main__":
    unittest.main()

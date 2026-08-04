from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.build_recovery_plan import run_build_recovery_plan
from analysis.prune_plan.recovery_plan import (
    RECOVERY_PLAN_BUNDLE_FILENAME,
    RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
    compose_recovery_unit_prune_plan,
    list_recovery_plan_names,
    load_recovery_plan_bundle,
)


def _write_recovery_all(root: Path, methods: dict | None = None, schema: str | None = None) -> None:
    payload = {
        "schema_version": schema or RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "synthetic",
        "num_units": 8,
        "methods": methods or {
            "recovery_ghost": {"method": "recovery_ghost"},
            "recovery_replaceme_ls": {"method": "recovery_replaceme_ls"},
        },
    }
    (root / RECOVERY_PLAN_BUNDLE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _write_prune_all(root: Path, *, start: int = 2, end: int = 4) -> None:
    drop = list(range(start, end + 1))
    record = {
        "selector": f"i{start}_j{end}",
        "i": start,
        "j": end,
        "start": start,
        "end": end,
        "span": len(drop),
        "drop_count": len(drop),
        "drop": drop,
    }
    payload = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "synthetic",
        "num_units": 8,
        "source_cache": "",
        "intervals": {record["selector"]: record},
        "sets": {},
        "strategies": {"ours": {"points": {str(len(drop)): record["selector"]}}},
        "spans": {str(len(drop)): record},
    }
    (root / "prune_plan_all.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_prune_set(root: Path, drop: list[int]) -> None:
    record = {
        "selector": f"ours_double_k{len(drop)}",
        "selector_kind": "set",
        "span": len(drop),
        "drop_count": len(drop),
        "drop": list(drop),
    }
    payload = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "synthetic",
        "num_units": 8,
        "source_cache": "",
        "intervals": {},
        "sets": {record["selector"]: record},
        "strategies": {"ours_double": {"points": {str(len(drop)): record["selector"]}}},
        "spans": {str(len(drop)): record},
    }
    (root / "prune_plan_all.json").write_text(json.dumps(payload), encoding="utf-8")


class RecoveryPlanSchemaTest(unittest.TestCase):
    def test_loads_named_methods_from_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recovery_all(root)
            loaded = load_recovery_plan_bundle(root, "recovery_ghost")
            self.assertEqual(loaded["method"], "recovery_ghost")
            self.assertEqual(list_recovery_plan_names(root), ["recovery_ghost", "recovery_replaceme_ls"])
            self.assertEqual(list(root.glob("recovery*.json")), [root / RECOVERY_PLAN_BUNDLE_FILENAME])

    def test_builder_accumulates_training_free_methods_without_span_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_all(root)
            methods = (
                "recovery_linearpatch",
                "recovery_ghost",
                "recovery_ghost_sequential",
                "recovery_ghost_interval",
                "recovery_ghost_interval_sequential",
                "recovery_replaceme_ls",
                "recovery_replaceme_cos",
            )
            for method in methods:
                run_build_recovery_plan(
                    experiment="vggt",
                    backbone="vggt",
                    prune_plan_root=root,
                    recovery_plan_root=root,
                    method=method,
                )
            raw = json.loads((root / RECOVERY_PLAN_BUNDLE_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(set(raw["methods"]), set(methods))
            self.assertTrue(all(set(entry) == {"method"} for entry in raw["methods"].values()))

    def test_derives_boundary_and_replaceme_auxiliary_from_prune_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_all(root)
            _write_recovery_all(root)
            plan = compose_recovery_unit_prune_plan(
                prune_plan_root=root,
                recovery_plan_root=root,
                prune_selector="prune_span3",
                recovery_plan="recovery_replaceme_ls",
            )
            self.assertEqual(plan.extra["recovery_source_unit_index"], 1)
            self.assertEqual(plan.extra["recovery_target_unit_index"], 4)
            self.assertEqual(plan.extra["cache_unit_ids"], ["global_0", "frame_2"])
            self.assertEqual(plan.extra["recovery_auxiliary_ids"], ["replaceme_residual:global_0"])

    def test_ghost_non_contiguous_plan_uses_one_boundary_per_dropped_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_set(root, [2, 3, 6])
            _write_recovery_all(root)
            plan = compose_recovery_unit_prune_plan(
                prune_plan_root=root,
                recovery_plan_root=root,
                prune_selector="prune_span3",
                recovery_plan="recovery_ghost",
            )
            boundaries = plan.extra["recovery_boundaries"]
            self.assertEqual(
                [(x["source_unit_index"], x["target_unit_index"]) for x in boundaries],
                [(1, 2), (2, 3), (5, 6)],
            )
            self.assertEqual(
                plan.extra["cache_unit_ids"],
                ["global_0", "frame_1", "global_1", "global_2", "frame_3"],
            )

    def test_ghost_variants_use_one_boundary_per_dropped_unit_even_for_one_interval(self) -> None:
        for method in ("recovery_ghost", "recovery_ghost_sequential"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_prune_all(root, start=2, end=4)
                _write_recovery_all(root, methods={method: {"method": method}})
                plan = compose_recovery_unit_prune_plan(
                    prune_plan_root=root,
                    recovery_plan_root=root,
                    prune_selector="prune_span3",
                    recovery_plan=method,
                )
                self.assertEqual(
                    [
                        (x["source_unit_index"], x["target_unit_index"])
                        for x in plan.extra["recovery_boundaries"]
                    ],
                    [(1, 2), (2, 3), (3, 4)],
                )

    def test_linearpatch_non_contiguous_plan_uses_one_boundary_per_dropped_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_set(root, [2, 3, 6])
            _write_recovery_all(
                root,
                methods={"recovery_linearpatch": {"method": "recovery_linearpatch"}},
            )
            plan = compose_recovery_unit_prune_plan(
                prune_plan_root=root,
                recovery_plan_root=root,
                prune_selector="prune_span3",
                recovery_plan="recovery_linearpatch",
            )
            self.assertEqual(
                [
                    (x["source_unit_index"], x["target_unit_index"])
                    for x in plan.extra["recovery_boundaries"]
                ],
                [(1, 2), (2, 3), (5, 6)],
            )

    def test_replaceme_non_contiguous_plan_uses_one_boundary_per_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_set(root, [2, 3, 6])
            _write_recovery_all(root)
            plan = compose_recovery_unit_prune_plan(
                prune_plan_root=root,
                recovery_plan_root=root,
                prune_selector="prune_span3",
                recovery_plan="recovery_replaceme_ls",
            )
            boundaries = plan.extra["recovery_boundaries"]
            self.assertEqual(
                [(x["source_unit_index"], x["target_unit_index"]) for x in boundaries],
                [(1, 3), (5, 6)],
            )
            self.assertEqual(
                plan.extra["recovery_auxiliary_ids"],
                ["replaceme_residual:global_0", "replaceme_residual:global_2"],
            )

    def test_interval_ghost_variants_use_one_boundary_per_interval(self) -> None:
        for method in (
            "recovery_ghost_interval",
            "recovery_ghost_interval_sequential",
        ):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_prune_set(root, [2, 3, 6])
                _write_recovery_all(root, methods={method: {"method": method}})
                plan = compose_recovery_unit_prune_plan(
                    prune_plan_root=root,
                    recovery_plan_root=root,
                    prune_selector="prune_span3",
                    recovery_plan=method,
                )
                self.assertEqual(
                    [
                        (x["source_unit_index"], x["target_unit_index"])
                        for x in plan.extra["recovery_boundaries"]
                    ],
                    [(1, 3), (5, 6)],
                )
                self.assertEqual(
                    plan.extra["cache_unit_ids"],
                    ["global_0", "global_1", "global_2", "frame_3"],
                )

    def test_rejects_pruning_protected_unit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_prune_all(root, start=0, end=2)
            _write_recovery_all(root)
            with self.assertRaisesRegex(ValueError, "protected unit 0"):
                compose_recovery_unit_prune_plan(
                    prune_plan_root=root,
                    recovery_plan_root=root,
                    prune_selector="prune_span3",
                    recovery_plan="recovery_ghost",
                )

    def test_rejects_legacy_per_method_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recovery_ghost.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "unified recovery plan"):
                load_recovery_plan_bundle(root, "recovery_ghost")

    def test_rejects_legacy_schema_and_removed_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recovery_all(root, schema="closed_interval_recovery_plan_bundle_v2")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_recovery_plan_bundle(root, "recovery_ghost")
        for method in ("recovery_diag", "recovery_rotate"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_recovery_all(root, methods={method: {"method": method}})
                with self.assertRaisesRegex(ValueError, "method"):
                    load_recovery_plan_bundle(root, method)

    def test_rejects_redundant_training_free_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recovery_all(
                root,
                methods={"recovery_ghost": {"method": "recovery_ghost", "spans": {}}},
            )
            with self.assertRaisesRegex(ValueError, "redundant"):
                load_recovery_plan_bundle(root, "recovery_ghost")


if __name__ == "__main__":
    unittest.main()

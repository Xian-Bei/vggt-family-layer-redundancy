from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.redundancy import skip_gt_eval_v3 as base
from analysis.redundancy.remaining_skip_gt_eval_v3 import (
    _load_base_prune_plan,
    _prepare_manifest_allowing_selector_alias,
    _validate_base_selector,
    build_remaining_span_mapping,
)
from analysis.redundancy.remaining_skip_layout import (
    build_remaining_span_records,
    evaluated_logical_spans,
)


class RemainingSkipSpanMappingTest(unittest.TestCase):
    def test_exact_base_selector_builds_one_contiguous_plan(self) -> None:
        _validate_base_selector("i2_j4", 3)
        with self.assertRaisesRegex(ValueError, "drops 3 units, expected 2"):
            _validate_base_selector("i2_j4", 2)
        plan = _load_base_prune_plan(
            root=Path("/unused"),
            selector="i2_j4",
            backbone="vggt",
            num_units=8,
        )
        self.assertEqual(plan.drop, [2, 3, 4])
        self.assertEqual(plan.extra["interval_selector"], "i2_j4")

    def test_equivalent_legacy_selector_manifest_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_config = {
                "base_selector": "i2_j4",
                "base_prune_plan": {
                    "prune_plan_name": "prune_span3",
                    "dropped_unit_ids": ["unit_2", "unit_3", "unit_4"],
                },
                "dataset": "dataset",
            }
            base._prepare_checkpoint_manifest(root, config=old_config, resume=True)
            new_config = {
                **old_config,
                "base_prune_plan": {
                    "prune_plan_name": "i2_j4",
                    "dropped_unit_ids": ["unit_2", "unit_3", "unit_4"],
                    "selection_path": "/selection.json",
                },
            }
            _prepare_manifest_allowing_selector_alias(
                root,
                config=new_config,
                resume=True,
            )
            manifest = json.loads(
                (root / "skip_eval_manifest_v3.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["config"], new_config)

    def test_intervals_are_indexed_over_remaining_units(self) -> None:
        remaining, logical_spans, physical_to_logical = build_remaining_span_mapping(
            full_unit_ids=[f"unit_{index}" for index in range(8)],
            dropped_indices=[1, 4, 6],
            max_span=0,
        )

        self.assertEqual(
            remaining,
            ["unit_0", "unit_2", "unit_3", "unit_5", "unit_7"],
        )
        self.assertEqual(len(logical_spans), 15)
        self.assertEqual(physical_to_logical[(0, 0)], (0, 0))
        self.assertEqual(physical_to_logical[(2, 5)], (1, 3))
        self.assertEqual(physical_to_logical[(0, 7)], (0, 4))

    def test_max_span_caps_logical_drop_count(self) -> None:
        _remaining, logical_spans, physical_to_logical = build_remaining_span_mapping(
            full_unit_ids=[f"unit_{index}" for index in range(7)],
            dropped_indices=[1, 3],
            max_span=2,
        )

        self.assertTrue(all(end - start + 1 <= 2 for start, end in logical_spans))
        self.assertEqual(len(logical_spans), 9)
        self.assertEqual(physical_to_logical[(2, 4)], (1, 2))

    def test_contiguous_unions_are_reused_and_disjoint_unions_are_evaluated(self) -> None:
        full = [f"unit_{index}" for index in range(8)]
        remaining, logical_spans, physical_to_logical = build_remaining_span_mapping(
            full_unit_ids=full,
            dropped_indices=[3, 4],
            max_span=0,
        )
        records = build_remaining_span_records(
            full_unit_ids=full,
            remaining_unit_ids=remaining,
            dropped_indices=[3, 4],
            logical_spans=logical_spans,
            physical_to_logical=physical_to_logical,
        )

        adjacent = records[(2, 2)]
        self.assertEqual(adjacent["result_kind"], "reused_single_interval")
        self.assertEqual(adjacent["result_selector"], "i2_j4")

        crossing = records[(2, 3)]
        self.assertEqual(crossing["result_kind"], "reused_single_interval")
        self.assertEqual(crossing["result_selector"], "i2_j5")

        disjoint = records[(1, 1)]
        self.assertEqual(disjoint["result_kind"], "evaluated_double_interval")
        self.assertEqual(disjoint["result_selector"], "i3_j4_i1_j1")
        self.assertIn((1, 1), evaluated_logical_spans(records))

    def test_finalization_merges_reused_metrics_without_copying_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base._append_scene_checkpoint(
                root,
                scene_id="scene0",
                span=(1, 1),
                metrics={"ape": 12.0},
            )
            report_path = base._finalize_skip_outputs_from_disk(
                out_root=root,
                spans=[(0, 0), (1, 1)],
                n_units=2,
                unit_ids=["unit_0", "unit_1"],
                metrics=["ape"],
                upper_metrics={"ape": 10.0},
                upper_scene_metrics={"scene0": {"ape": 10.0}},
                backbone="vggt",
                dataset="dataset",
                max_span=0,
                fusion_mode="mixed_units",
                elapsed_seconds=0.0,
                ensure_scene_reports=True,
                supplemental_span_metrics_by_scene={
                    "scene0": {(0, 0): {"ape": 11.0}}
                },
                materialize_intervals=False,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["scene_ids"], ["scene0"])
            checkpoint_text = (
                root / "checkpoints" / "scene0.jsonl"
            ).read_text(encoding="utf-8")
            self.assertNotIn('"span_i":0', checkpoint_text)
            self.assertFalse((root / "i0_j0").exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.eval.summary_v3 import (
    _collect_by_dataset,
    _filter_to_current_prune_spans,
    _interval_span_from_label,
    _label_base,
    _record_label,
)
from analysis.schema.v3 import EvalRecordV3


class SummaryV3SelectedPlanTest(unittest.TestCase):
    def test_collection_reads_recovery_method_symlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost_root = (
                root
                / "dataset"
                / "eval"
                / "recovery"
                / "i1_j2"
                / "recovery_ghost"
            )
            EvalRecordV3(average={"ape": 1.25}).save(
                ghost_root / "eval_record_v3.json"
            )
            ghost_root.with_name(
                "recovery_ghost_interval_sequential"
            ).symlink_to("recovery_ghost", target_is_directory=True)

            grouped = _collect_by_dataset(root)

        self.assertEqual(
            set(grouped["dataset"]),
            {
                "i1_j2_recovery_ghost",
                "i1_j2_recovery_ghost_interval_sequential",
            },
        )
        self.assertEqual(
            grouped["dataset"][
                "i1_j2_recovery_ghost_interval_sequential"
            ].average["ape"],
            1.25,
        )

    def test_multi_interval_recovery_label_resolves_physical_selector(self) -> None:
        label = "i2_j4_i8_j10_recovery_ghost_interval_sequential"

        self.assertEqual(_label_base(label), "i2_j4_i8_j10")
        self.assertEqual(_interval_span_from_label(label), (2, 10, 6))

    def test_testtime_label_keeps_selector_and_compression(self) -> None:
        record = EvalRecordV3(
            model_name="model",
            extra={
                "eval_mode": "model",
                "inference_compression": {
                    "name": "fastvggt",
                    "merge_start": 0,
                    "merge_ratio": 0.9,
                },
            },
        )

        mode, label, selector = _record_label(
            record,
            Path(
                "testtime/recovery_ghost_interval_sequential/"
                "fastvggt_m0_r0p9/i2_j4_i8_j10"
            ),
        )

        self.assertEqual(mode, "model")
        self.assertEqual(selector, "i2_j4_i8_j10")
        self.assertEqual(
            label,
            "i2_j4_i8_j10_recovery_ghost_interval_sequential+fastvggt_m0_r0p9",
        )

    def test_filter_keeps_only_selected_physical_intervals(self) -> None:
        record = EvalRecordV3()
        grouped = {
            "dataset": {
                "upper": record,
                "upper+fastvggt_m0_r0p9": record,
                "i2_j4": record,
                "i2_j4_recovery_ghost": record,
                "i8_j10_i14_j16_recovery_ghost": record,
                "i20_j22": record,
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prune_plan_all.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "analysis.eval.summary_v3.load_prune_plan_bundle",
                    return_value={},
                ),
                patch(
                    "analysis.eval.summary_v3.strategy_interval_selectors",
                    return_value=["i2_j4", "i8_j10_i14_j16"],
                ) as selected,
            ):
                result = _filter_to_current_prune_spans(
                    grouped,
                    root,
                    ["CKA", "retention"],
                )

        self.assertEqual(
            list(result["dataset"]),
            [
                "upper",
                "upper+fastvggt_m0_r0p9",
                "i2_j4",
                "i2_j4_recovery_ghost",
                "i8_j10_i14_j16_recovery_ghost",
            ],
        )
        selected.assert_called_once_with(
            root,
            strategies=["CKA", "retention"],
        )

    def test_selected_strategy_resolution_failure_does_not_load_all(self) -> None:
        record = EvalRecordV3()
        grouped = {"dataset": {"upper": record, "i2_j4": record}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prune_plan_all.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "analysis.eval.summary_v3.load_prune_plan_bundle",
                    return_value={},
                ),
                patch(
                    "analysis.eval.summary_v3.strategy_interval_selectors",
                    side_effect=ValueError("bad strategy"),
                ),
            ):
                result = _filter_to_current_prune_spans(
                    grouped,
                    root,
                    ["CKA", "retention"],
                )

        self.assertEqual(list(result["dataset"]), ["upper"])

    def test_missing_bundle_does_not_load_unfiltered_intervals(self) -> None:
        record = EvalRecordV3()
        grouped = {"dataset": {"upper": record, "i2_j4": record}}

        with tempfile.TemporaryDirectory() as tmp:
            result = _filter_to_current_prune_spans(
                grouped,
                Path(tmp),
                ["CKA", "retention"],
            )

        self.assertEqual(list(result["dataset"]), ["upper"])


if __name__ == "__main__":
    unittest.main()

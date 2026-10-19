from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.results.init_main_results import (
    INIT_MANIFEST,
    initialize_main_results,
)


class InitializeMainResultsTest(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_source_tree(self, root: Path) -> None:
        backbone = "vggt"
        for dataset in ("train_a", "eval_a"):
            self._write_json(
                root / backbone / dataset / "eval" / "upper" / "eval_record_v3.json",
                {
                    "schema_version": "v3",
                    "backbone": backbone,
                    "dataset": dataset,
                    "model_name": "upper",
                },
            )
        pruning_root = root / backbone / "train_a" / "eval" / "pruning"
        self._write_json(
            pruning_root / "skip_eval_report_v3.json",
            {
                "schema_version": "v3",
                "backbone": backbone,
                "dataset": "train_a",
                "num_layers": 2,
                "unit_ids": ["u0", "u1"],
                "fusion_modes": ["mixed_units"],
                "max_span": 0,
                "fusion_by_mode": {
                    "mixed_units": {
                        "gt_degrade_pct_upper_by_metric": {
                            "ape": [0.0, 1.0, 2.0],
                        }
                    }
                },
            },
        )
        for start, end in ((0, 0), (0, 1), (1, 1)):
            selector = f"i{start}_j{end}"
            self._write_json(
                pruning_root / selector / "summary.json",
                {
                    "schema_version": "skip_eval_interval_v3",
                    "selector": selector,
                    "drop": list(range(start, end + 1)),
                    "drop_count": end - start + 1,
                    "metrics_mean": {"ape": 1.0},
                },
            )
        eval_selector = root / backbone / "eval_a" / "eval" / "pruning" / "i1_j1"
        self._write_json(
            eval_selector / "summary.json",
            {
                "schema_version": "eval_interval_v3",
                "selector": "i1_j1",
                "eval_record_path": str(eval_selector / "eval_record_v3.json"),
            },
        )
        self._write_json(
            eval_selector / "eval_record_v3.json",
            {
                "schema_version": "v3",
                "backbone": backbone,
                "dataset": "eval_a",
                "model_name": "lower",
            },
        )

    def test_reinitialization_overwrites_imports_and_preserves_other_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "analysis_outputs_v3"
            output = root / "main_results"
            self._write_source_tree(source)

            status = initialize_main_results(
                source_root=source,
                output_root=output,
                backbones=["vggt"],
                calib_datasets=["train_a"],
                eval_datasets=["eval_a"],
            )

            self.assertEqual(status, "copied")
            self.assertTrue((output / INIT_MANIFEST).is_file())
            self.assertTrue(
                (
                    output
                    / "vggt"
                    / "train_a"
                    / "eval"
                    / "pruning"
                    / "skip_eval_report_v3.json"
                ).is_file()
            )
            copied_summary = (
                output
                / "vggt"
                / "eval_a"
                / "eval"
                / "pruning"
                / "i1_j1"
                / "summary.json"
            )
            payload = json.loads(copied_summary.read_text(encoding="utf-8"))
            self.assertTrue(str(payload["eval_record_path"]).startswith(str(output)))

            source_summary = source / copied_summary.relative_to(output)
            updated = json.loads(source_summary.read_text(encoding="utf-8"))
            updated["metrics_mean"] = {"ape": 42.0}
            self._write_json(source_summary, updated)
            recovery = output / "vggt" / "recovery" / "result.json"
            self._write_json(recovery, {"keep": True})
            second_status = initialize_main_results(
                source_root=source,
                output_root=output,
                backbones=["vggt"],
                calib_datasets=["train_a"],
                eval_datasets=["eval_a"],
            )
            self.assertEqual(second_status, "copied")
            refreshed = json.loads(copied_summary.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["metrics_mean"], {"ape": 42.0})
            self.assertTrue(str(refreshed["eval_record_path"]).startswith(str(output)))
            self.assertEqual(json.loads(recovery.read_text()), {"keep": True})
            self.assertEqual(list(root.glob(".main_results.init.*")), [])

    def test_incomplete_skip_eval_does_not_create_destination(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "analysis_outputs_v3"
            output = root / "main_results"
            self._write_source_tree(source)
            report_path = (
                source
                / "vggt"
                / "train_a"
                / "eval"
                / "pruning"
                / "skip_eval_report_v3.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["fusion_by_mode"]["mixed_units"][
                "gt_degrade_pct_upper_by_metric"
            ]["ape"][2] = None
            self._write_json(report_path, report)

            with self.assertRaisesRegex(ValueError, "incomplete skip-eval report"):
                initialize_main_results(
                    source_root=source,
                    output_root=output,
                    backbones=["vggt"],
                    calib_datasets=["train_a"],
                    eval_datasets=["eval_a"],
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

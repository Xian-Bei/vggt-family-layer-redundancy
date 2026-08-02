from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.redundancy.migrate_remaining_skip_results import (
    commit_migration,
    inspect_migration,
)
from analysis.redundancy.remaining_skip_gt_eval_v3 import (
    build_remaining_span_mapping,
)
from analysis.redundancy.remaining_skip_layout import (
    REMAINING_INDEX_FILENAME,
    build_remaining_span_records,
    logical_selector,
    manifest_payload,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class RemainingSkipMigrationTest(unittest.TestCase):
    def test_dry_run_and_commit_keep_only_true_double_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            eval_root = Path(temporary) / "vggt" / "dataset" / "eval"
            old_root = eval_root / "remaining_pruning"
            pruning_root = eval_root / "pruning"
            full = [f"unit_{index}" for index in range(8)]
            remaining, spans, physical_to_logical = build_remaining_span_mapping(
                full_unit_ids=full,
                dropped_indices=[3, 4],
                max_span=0,
            )
            records = build_remaining_span_records(
                full_unit_ids=full,
                remaining_unit_ids=remaining,
                dropped_indices=[3, 4],
                logical_spans=spans,
                physical_to_logical=physical_to_logical,
            )
            config = {
                "backbone": "vggt",
                "dataset": "dataset",
                "full_unit_ids": full,
                "unit_ids": remaining,
                "spans": [list(span) for span in spans],
                "physical_spans": [
                    list(physical) for physical in physical_to_logical
                ],
                "base_prune_plan": {
                    "prune_plan_root": str(Path(temporary) / "vggt"),
                    "prune_plan_name": "prune_span2",
                    "expected_drop_count": 2,
                    "dropped_unit_ids": ["unit_3", "unit_4"],
                    "remaining_unit_ids": remaining,
                },
            }
            _write_json(
                old_root / "skip_eval_manifest_v3.json",
                manifest_payload(config),
            )
            _write_json(old_root / "skip_eval_interval_index_v3.json", {})
            checkpoint_lines = []
            for span, record in records.items():
                metric = float(sum(record["combined_drop"]))
                checkpoint_lines.append(
                    json.dumps(
                        {
                            "schema_version": "v3",
                            "kind": "skip_gt_eval_v3_span",
                            "scene_id": "scene0",
                            "span_i": span[0],
                            "span_j": span[1],
                            "metrics": {"ape": metric},
                        },
                        separators=(",", ":"),
                    )
                )
                if record["result_kind"] == "reused_single_interval":
                    direct = pruning_root / str(record["result_selector"])
                    _write_json(
                        direct / "summary.json",
                        {
                            "selector": record["result_selector"],
                            "drop": record["combined_drop"],
                            "num_scenes": 1,
                            "metrics_mean": {"ape": metric},
                        },
                    )
                    (direct / "scenes.jsonl").write_text(
                        json.dumps(
                            {
                                "scene_id": "scene0",
                                "metrics": {"ape": metric},
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            checkpoint = old_root / "checkpoints" / "scene0.jsonl"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("\n".join(checkpoint_lines) + "\n", encoding="utf-8")

            dry_run = inspect_migration(old_root)
            self.assertEqual(dry_run["num_candidates"], len(records))
            self.assertFalse(dry_run["destination_collisions"])
            self.assertTrue(old_root.is_dir())

            report = commit_migration(old_root)
            self.assertTrue(report["committed"])
            self.assertFalse(old_root.exists())
            self.assertTrue((eval_root / "remaining_pruning.legacy").is_dir())
            staging = pruning_root / "_remaining_skip" / "i3_j4"
            self.assertTrue((staging / "skip_eval_manifest_v3.json").is_file())
            migrated_rows = (
                staging / "checkpoints" / "scene0.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            evaluated_count = sum(
                record["result_kind"] == "evaluated_double_interval"
                for record in records.values()
            )
            self.assertEqual(len(migrated_rows), evaluated_count)
            for span, record in records.items():
                destination = pruning_root / str(record["result_selector"])
                if record["result_kind"] == "evaluated_double_interval":
                    self.assertTrue((destination / "summary.json").is_file())
                else:
                    self.assertNotIn(
                        str(record["base_selector"]) + "_",
                        destination.name,
                    )
            index = json.loads(
                (pruning_root / REMAINING_INDEX_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(index["active_base_selector"], "i3_j4")
            self.assertEqual(
                index["plans"]["i3_j4"]["num_candidates"],
                len(records),
            )


if __name__ == "__main__":
    unittest.main()

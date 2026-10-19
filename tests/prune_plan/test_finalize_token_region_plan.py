from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    load_prune_plan_bundle,
    strategy_interval_selectors,
)
from analysis.prune_plan.eval_plots import GT_MEAN_METRICS
from analysis.prune_plan.finalize_token_region_plan import (
    finalize_token_region_plan,
)
from analysis.prune_plan.stage7 import (
    DatasetSpec,
    missing_plan_schedule,
    selected_plans,
)


def _record(plan_key: str, selector: str, drop: list[int]) -> dict:
    return {
        "selector": plan_key,
        "eval_selector": selector,
        "selector_kind": "set",
        "drop_count": len(drop),
        "drop": drop,
    }


def _strategy(plan_key: str, source_region: str) -> dict:
    return {
        "display_name": plan_key.rsplit("_k", 1)[0],
        "points": {"2": plan_key},
        "scores": {"2": 0.5},
        "source": {"token_region": source_region},
    }


def _bundle(
    *,
    token_region: str,
    cka_selector: str,
    include_fixed: bool,
) -> dict:
    cka_key = "CKA_single_k2"
    cka_double_key = "CKA_double_k2"
    cka_drop = [1, 2] if cka_selector == "i1_j2" else [3, 4]
    strategies = {
        "CKA_single": _strategy(cka_key, token_region),
        "CKA_double": _strategy(cka_double_key, token_region),
    }
    sets = {
        cka_key: _record(cka_key, cka_selector, cka_drop),
        cka_double_key: _record(cka_double_key, cka_selector, cka_drop),
    }
    if include_fixed:
        for strategy, selector, drop in (
            ("retention_single", "i5_j6", [5, 6]),
            ("replaceme_single", "i7_j8", [7, 8]),
            ("shortgpt", "i1_j1_i9_j9", [1, 9]),
            ("gardener", "i2_j2_i9_j9", [2, 9]),
            ("simpler", "i8_j9", [8, 9]),
        ):
            plan_key = f"{strategy}_k2"
            strategies[strategy] = _strategy(plan_key, token_region)
            sets[plan_key] = _record(plan_key, selector, drop)
    return {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "train",
        "num_units": 10,
        "unit_ids": [
            value
            for layer in range(5)
            for value in (f"frame_{layer}", f"global_{layer}")
        ],
        "intervals": {},
        "sets": sets,
        "strategies": strategies,
        "spans": {},
        "extra": {"token_region": token_region, "max_drop_budget": 2},
    }


def _write_summary(root: Path, selector: str, value: float) -> None:
    path = root / selector
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "selector": selector,
                "drop_count": 2,
                "metrics_mean": {metric: value for metric in GT_MEAN_METRICS},
            }
        ),
        encoding="utf-8",
    )


class FinalizeTokenRegionPlanTest(unittest.TestCase):
    def _fixture(self, root: Path, *, special_value: float = 1.5) -> tuple[Path, Path]:
        all_path = root / "all.json"
        special_path = root / "special.json"
        all_path.write_text(
            json.dumps(
                _bundle(
                    token_region="all",
                    cka_selector="i1_j2",
                    include_fixed=True,
                )
            ),
            encoding="utf-8",
        )
        special_path.write_text(
            json.dumps(
                _bundle(
                    token_region="special_patch_mean",
                    cka_selector="i3_j4",
                    include_fixed=False,
                )
            ),
            encoding="utf-8",
        )
        upper = root / "train" / "eval" / "upper"
        upper.mkdir(parents=True)
        (upper / "eval_record_v3.json").write_text(
            json.dumps({"average": {metric: 1.0 for metric in GT_MEAN_METRICS}}),
            encoding="utf-8",
        )
        pruning = root / "train" / "eval" / "pruning"
        _write_summary(pruning, "i1_j2", 2.0)
        _write_summary(pruning, "i3_j4", special_value)
        _write_summary(pruning, "i5_j6", 1.2)
        _write_summary(pruning, "i7_j8", 1.4)
        return all_path, special_path

    def test_selects_special_candidate_and_keeps_all_only_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root)
            output = root / "prune_plan_all.json"
            finalize_token_region_plan(
                experiment_root=root,
                all_bundle_path=all_path,
                special_patch_mean_bundle_path=special_path,
                selection_datasets=["train"],
                output_path=output,
                viz_out_dir=root,
            )
            final = json.loads(output.read_text(encoding="utf-8"))
            loaded = load_prune_plan_bundle(root)
            cka_selectors = strategy_interval_selectors(
                root, strategies=["CKA_single"]
            )
            grid_exists = (root / "prune_plan_vggt_drop_grid.png").is_file()

        self.assertEqual(
            final["sets"]["CKA_single_k2"]["eval_selector"],
            "i3_j4",
        )
        selection = final["sets"]["CKA_single_k2"]["token_region_selection"]
        self.assertEqual(selection["winner_token_region"], "special_patch_mean")
        self.assertTrue(grid_exists)
        self.assertEqual(cka_selectors, ["i3_j4"])
        self.assertEqual(loaded["spans"]["2"]["eval_selector"], "i3_j4")
        self.assertEqual(
            final["sets"]["retention_single_k2"]["eval_selector"],
            "i5_j6",
        )
        self.assertEqual(
            final["sets"]["replaceme_single_k2"]["eval_selector"],
            "i7_j8",
        )
        fixed = final["extra"]["token_region_policy"]["fixed_all_strategies"]
        self.assertIn("retention_single", fixed)
        self.assertIn("replaceme_single", fixed)
        self.assertIn("shortgpt", fixed)
        self.assertIn("gardener", fixed)
        self.assertIn("simpler", fixed)

    def test_heldout_schedule_contains_winner_but_not_loser(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root)
            finalize_token_region_plan(
                experiment_root=root,
                all_bundle_path=all_path,
                special_patch_mean_bundle_path=special_path,
                selection_datasets=["train"],
                output_path=root / "prune_plan_all.json",
            )
            plans = selected_plans(
                root,
                strategies=["CKA_single", "retention_single"],
                min_drop=0,
                max_drop=2,
                drop_step=2,
            )
            schedule, _ = missing_plan_schedule(
                prune_plan_root=root,
                plans=plans,
                datasets=[
                    DatasetSpec(
                        key="heldout",
                        directory="heldout",
                        training=False,
                    )
                ],
            )

        scheduled = {item.selector for item in schedule}
        self.assertEqual(scheduled, {"i3_j4", "i5_j6"})
        self.assertNotIn("i1_j2", scheduled)

    def test_selects_each_drop_budget_independently(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root)
            all_bundle = json.loads(all_path.read_text(encoding="utf-8"))
            special_bundle = json.loads(special_path.read_text(encoding="utf-8"))
            for bundle, selector, drop in (
                (all_bundle, "i1_j3", [1, 2, 3]),
                (special_bundle, "i4_j6", [4, 5, 6]),
            ):
                bundle["sets"]["CKA_single_k3"] = _record(
                    "CKA_single_k3", selector, drop
                )
                bundle["strategies"]["CKA_single"]["points"]["3"] = (
                    "CKA_single_k3"
                )
                bundle["strategies"]["CKA_single"]["scores"]["3"] = 0.4
                bundle["extra"]["max_drop_budget"] = 3
            all_path.write_text(json.dumps(all_bundle), encoding="utf-8")
            special_path.write_text(json.dumps(special_bundle), encoding="utf-8")
            pruning = root / "train" / "eval" / "pruning"
            _write_summary(pruning, "i1_j3", 1.2)
            _write_summary(pruning, "i4_j6", 2.0)
            output = root / "prune_plan_all.json"
            finalize_token_region_plan(
                experiment_root=root,
                all_bundle_path=all_path,
                special_patch_mean_bundle_path=special_path,
                selection_datasets=["train"],
                output_path=output,
            )
            final = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            final["sets"]["CKA_single_k2"]["token_region_selection"][
                "winner_token_region"
            ],
            "special_patch_mean",
        )
        self.assertEqual(
            final["sets"]["CKA_single_k3"]["token_region_selection"][
                "winner_token_region"
            ],
            "all",
        )

    def test_exact_score_tie_prefers_special_patch_mean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root, special_value=2.0)
            output = root / "prune_plan_all.json"
            finalize_token_region_plan(
                experiment_root=root,
                all_bundle_path=all_path,
                special_patch_mean_bundle_path=special_path,
                selection_datasets=["train"],
                output_path=output,
            )
            final = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            final["sets"]["CKA_single_k2"]["token_region_selection"][
                "winner_token_region"
            ],
            "special_patch_mean",
        )

    def test_budget_filter_removes_unevaluated_dual_region_points(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root)
            all_bundle = json.loads(all_path.read_text(encoding="utf-8"))
            special_bundle = json.loads(special_path.read_text(encoding="utf-8"))
            for bundle, selector, drop in (
                (all_bundle, "i1_j3", [1, 2, 3]),
                (special_bundle, "i4_j6", [4, 5, 6]),
            ):
                bundle["sets"]["CKA_single_k3"] = _record(
                    "CKA_single_k3", selector, drop
                )
                bundle["strategies"]["CKA_single"]["points"]["3"] = (
                    "CKA_single_k3"
                )
                bundle["strategies"]["CKA_single"]["scores"]["3"] = 0.4
                bundle["extra"]["max_drop_budget"] = 3
            all_path.write_text(json.dumps(all_bundle), encoding="utf-8")
            special_path.write_text(json.dumps(special_bundle), encoding="utf-8")
            output = root / "prune_plan_all.json"
            finalize_token_region_plan(
                experiment_root=root,
                all_bundle_path=all_path,
                special_patch_mean_bundle_path=special_path,
                selection_datasets=["train"],
                output_path=output,
                min_drop=0,
                max_drop=3,
                drop_step=2,
            )
            final = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            final["strategies"]["CKA_single"]["points"],
            {"2": "CKA_single_k2"},
        )
        self.assertNotIn("CKA_single_k3", final["sets"])

    def test_rejects_all_only_strategy_in_special_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            all_path, special_path = self._fixture(root)
            special = json.loads(special_path.read_text(encoding="utf-8"))
            special["strategies"]["shortgpt"] = _strategy(
                "shortgpt_k2", "special_patch_mean"
            )
            special_path.write_text(json.dumps(special), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must remain all-only"):
                finalize_token_region_plan(
                    experiment_root=root,
                    all_bundle_path=all_path,
                    special_patch_mean_bundle_path=special_path,
                    selection_datasets=["train"],
                    output_path=root / "prune_plan_all.json",
                )


if __name__ == "__main__":
    unittest.main()

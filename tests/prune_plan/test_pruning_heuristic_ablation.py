from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.build_prune_plan_v3 import (
    DOUBLE_GAP_COMPARISON_TOLERANCE,
    SINGLE_DOUBLE_COMPARISON_TOLERANCE,
    _build_parser,
    _build_similarity_metric_strategy_pair,
    _one_or_two_interval_drop_sets_from_score,
)
from analysis.results.pruning_heuristic_ablation import (
    MANIFEST_FILENAME,
    STRATEGIES,
    VARIANT_ORDER,
    VARIANT_TOLERANCES,
    _selector_origin,
    aggregate_rows,
    collect_rows,
    init_manifest,
    summarize,
)
from analysis.redundancy.skip_metric_scaling import GT_METRIC_KEYS


def _components(extra: dict[int, dict], count: int) -> list[tuple[int, int]]:
    return [
        (int(component["start"]), int(component["end"]))
        for component in extra[count]["component_intervals"]
    ]


def _write_summary(root: Path, selector: str, value: float) -> None:
    path = root / selector
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({
            "selector": selector,
            "num_scenes": 2,
            "metrics_mean": {metric: value for metric in GT_METRIC_KEYS},
        }),
        encoding="utf-8",
    )


def _write_eval_inputs(root: Path, dataset: str, selectors: dict[str, float]) -> None:
    upper = root / dataset / "eval" / "upper"
    upper.mkdir(parents=True, exist_ok=True)
    (upper / "eval_record_v3.json").write_text(
        json.dumps({"average": {metric: 1.0 for metric in GT_METRIC_KEYS}}),
        encoding="utf-8",
    )
    pruning = root / dataset / "eval" / "pruning"
    for selector, value in selectors.items():
        _write_summary(pruning, selector, value)
    if dataset == "scannet_dev50":
        (pruning / "normalization_profile.json").write_text(
            json.dumps({
                "schema_version": "stage2_upper_zero_worst_all_intervals_v3",
                "source_dataset": dataset,
                "upper_average": {metric: 1.0 for metric in GT_METRIC_KEYS},
                "maximum_brute_force_metric": {
                    metric: 5.0 for metric in GT_METRIC_KEYS
                },
            }),
            encoding="utf-8",
        )


def _write_bundle(root: Path, variant: str, selector: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    strategies = {}
    for strategy in STRATEGIES:
        key = f"{strategy}_k2"
        record = {
            "selector": key,
            "eval_selector": selector,
            "selector_kind": "set",
            "drop": [int(selector[1]), int(selector[4])],
            "drop_count": 2,
        }
        if strategy.startswith("CKA_"):
            record["token_region_selection"] = {
                "winner_token_region": "special_patch_mean"
            }
        records[key] = record
        strategies[strategy] = {"display_name": strategy, "points": {"2": key}}
    single_double = VARIANT_TOLERANCES[variant]
    payload = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "scannet_dev50",
        "num_units": 10,
        "unit_ids": [str(index) for index in range(10)],
        "intervals": {},
        "sets": records,
        "strategies": strategies,
        "spans": {},
        "extra": {
            "similarity_selection_controls": {
                "single_double_comparison_tolerance": single_double,
            }
        },
    }
    (root / "prune_plan_all.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class PruningHeuristicSelectorTest(unittest.TestCase):
    def test_gap_tolerance_only_changes_narrower_gap_comparison(self) -> None:
        for maximize in (False, True):
            score = np.full((7, 7), np.nan)
            score[1, 1] = score[5, 5] = 0.5
            score[3, 3] = 0.504 if maximize else 0.496
            kwargs = dict(
                maximize=maximize, selection_mode="two",
                require_cross_budget_interval_overlap=False, max_drop_budget=2,
            )
            _, _, tolerant = _one_or_two_interval_drop_sets_from_score(
                score, double_gap_comparison_tolerance=0.005, **kwargs,
            )
            _, _, strict = _one_or_two_interval_drop_sets_from_score(score, **kwargs)
            self.assertEqual(_components(tolerant, 2), [(1, 1), (5, 5)])
            self.assertEqual(_components(strict, 2), [(1, 1), (3, 3)])
            self.assertNotIn("double_gap_comparison_tolerance", strict[2])
            score[3, 3] = 0.506 if maximize else 0.494
            _, _, better = _one_or_two_interval_drop_sets_from_score(
                score, double_gap_comparison_tolerance=0.005, **kwargs,
            )
            self.assertEqual(_components(better, 2), [(1, 1), (3, 3)])

    def test_all_similarity_families_pass_and_record_gap_tolerance(self) -> None:
        proxy = np.full((7, 7), np.nan)
        proxy[1, 2] = 2.0
        proxy[1, 1] = proxy[5, 5] = 0.5
        proxy[3, 3] = 0.496
        for family in ("CKA", "cosine", "inv_mse", "cca"):
            for tolerance, expected in ((0.005, [1, 5]), (0.0, [1, 3])):
                records = {}
                with mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3.REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP",
                    False,
                ):
                    strategies = _build_similarity_metric_strategy_pair(
                        family_key=family, display_name=family, metric=family,
                        interval_score=1.0 - proxy, similarity_source={},
                        token_region="all", dataset="fixture",
                        unit_ids=[str(i) for i in range(7)], num_units=7,
                        param_accounting={"available": False}, sets=records,
                        double_gap_comparison_tolerance=tolerance, max_drop_budget=2,
                    )
                double = strategies[f"{family}_double"]
                point = records[double["points"]["2"]]
                self.assertEqual(point["drop"], expected)
                self.assertEqual(point["double_gap_comparison_tolerance"], tolerance)
                self.assertEqual(double["source"]["double_gap_comparison_tolerance"], tolerance)
                self.assertNotIn("double_gap_comparison_tolerance", strategies[f"{family}_single"]["source"])

    def test_gap_tolerance_rejects_invalid_values(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "double_gap_comparison_tolerance"):
                _one_or_two_interval_drop_sets_from_score(
                    np.full((3, 3), np.nan), maximize=False,
                    double_gap_comparison_tolerance=value,
                )

    def test_double_candidates_use_strict_score_and_exact_tie_prefers_wider_gap(self) -> None:
        score = np.full((7, 7), np.nan)
        score[1, 1] = 0.500
        score[3, 3] = 0.496
        score[5, 5] = 0.500
        _, _, extra = _one_or_two_interval_drop_sets_from_score(
            score, maximize=False, selection_mode="two",
            require_cross_budget_interval_overlap=False, max_drop_budget=2,
        )
        self.assertEqual(_components(extra, 2), [(1, 1), (3, 3)])
        self.assertNotIn("single_double_comparison_tolerance", extra[2])

        score[3, 3] = 0.500
        _, _, tied = _one_or_two_interval_drop_sets_from_score(
            score, maximize=False, selection_mode="two",
            require_cross_budget_interval_overlap=False, max_drop_budget=2,
        )
        self.assertEqual(_components(tied, 2), [(1, 1), (5, 5)])

    def test_single_double_tolerance_is_optional(self) -> None:
        score = np.full((6, 6), np.nan)
        score[1, 2] = 1.000
        score[1, 1] = 0.498
        score[3, 3] = 0.498
        _, _, similarity = _one_or_two_interval_drop_sets_from_score(
            score, maximize=False, selection_mode="best",
            single_double_comparison_tolerance=0.005,
            require_cross_budget_interval_overlap=False, max_drop_budget=2,
        )
        _, _, retention = _one_or_two_interval_drop_sets_from_score(
            score, maximize=False, selection_mode="best",
            require_cross_budget_interval_overlap=False, max_drop_budget=2,
        )
        self.assertEqual(similarity[2]["num_intervals"], 1)
        self.assertEqual(retention[2]["num_intervals"], 2)
        self.assertNotIn("single_double_comparison_tolerance", retention[2])

    def test_cli_exposes_similarity_tolerances(self) -> None:
        common = ["--dataset", "scannet_dev50"]
        defaults = _build_parser().parse_args(common)
        self.assertEqual(defaults.double_gap_comparison_tolerance, DOUBLE_GAP_COMPARISON_TOLERANCE)
        self.assertEqual(
            defaults.single_double_comparison_tolerance,
            SINGLE_DOUBLE_COMPARISON_TOLERANCE,
        )
        custom = _build_parser().parse_args([
            *common, "--single-double-comparison-tolerance", "0.001",
            "--double-gap-comparison-tolerance", "0.002",
        ])
        self.assertEqual(custom.double_gap_comparison_tolerance, 0.002)
        self.assertEqual(custom.single_double_comparison_tolerance, 0.001)


class PruningHeuristicReportTest(unittest.TestCase):
    def test_manifest_preserves_initial_snapshot_and_origins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            result = source / "vggt" / "scannet_dev50" / "eval" / "pruning" / "i1_j2"
            result.mkdir(parents=True)
            (result / "summary.json").write_text("{}", encoding="utf-8")
            output = root / "ablation"
            manifest_path = init_manifest(
                output_root=output,
                source_output_root=source,
                backbones=["vggt"],
                training_datasets=["scannet_dev50"],
                heldout_datasets=["scannet_test50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            first = json.loads(manifest_path.read_text(encoding="utf-8"))
            new_result = result.parent / "i3_j4"
            new_result.mkdir()
            (new_result / "summary.json").write_text("{}", encoding="utf-8")
            init_manifest(
                output_root=output,
                source_output_root=source,
                backbones=["vggt"],
                training_datasets=["scannet_dev50"],
                heldout_datasets=["scannet_test50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            second = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(first["initial_existing_selectors"], second["initial_existing_selectors"])
        self.assertEqual(
            _selector_origin(
                first,
                variant="baseline",
                backbone="vggt",
                dataset="scannet_dev50",
                selector="i1_j2",
            ),
            "preexisting_main",
        )

    def test_collects_metrics_baseline_deltas_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source" / "vggt"
            selectors = {
                "i1_j2": 2.0,
                "i2_j3": 2.4,
                "i3_j4": 2.8,
                "i4_j5": 3.2,
            }
            _write_eval_inputs(source, "scannet_dev50", selectors)
            _write_eval_inputs(source, "scannet_test50", selectors)
            output = root / "ablation"
            manifest_path = init_manifest(
                output_root=output,
                source_output_root=root / "source",
                backbones=["vggt"],
                training_datasets=["scannet_dev50"],
                heldout_datasets=["scannet_test50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            for variant, selector in zip(VARIANT_ORDER, selectors):
                _write_bundle(output / variant / "vggt", variant, selector)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows, missing = collect_rows(output_root=output, manifest=manifest)
            aggregates = aggregate_rows(rows)
            payload = summarize(output)
            self.assertTrue((output / "pruning_heuristic_ablation_results.json").is_file())
            self.assertTrue((output / "pruning_heuristic_ablation_results.csv").is_file())
            self.assertEqual(len(payload["figures"]), 1)
            self.assertTrue((output / "figures" / "heuristic_ablation_heldout_by_backbone.png").is_file())
            self.assertTrue(all(item.get("raw_metric_sum") is not None for item in aggregates))
        self.assertEqual(missing, [])
        self.assertEqual(len(rows), 16)
        baseline = next(
            row for row in rows
            if row["variant"] == "baseline"
            and row["dataset"] == "scannet_dev50"
            and row["strategy"] == "retention_double"
        )
        ablated = next(
            row for row in rows
            if row["variant"] == "no_single_double"
            and row["dataset"] == "scannet_dev50"
            and row["strategy"] == "retention_double"
        )
        self.assertAlmostEqual(baseline["gt_mean"], 0.25)
        self.assertAlmostEqual(ablated["delta_gt_mean_vs_baseline"], 0.1)
        self.assertEqual(len(aggregates), 16)

    def test_driver_has_complete_matrix_and_valid_shell_syntax(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        subprocess.run(
            ["bash", "-n", "experiments/ablation_prune_heuristics.sh"],
            cwd=repo,
            check=True,
        )
        script = (repo / "experiments/ablation_prune_heuristics.sh").read_text(encoding="utf-8")
        for variant in VARIANT_ORDER:
            self.assertIn(variant, script)
        self.assertIn("vggt,pi3,vggt_omega", script)
        self.assertIn("special_patch_mean", script)


if __name__ == "__main__":
    unittest.main()

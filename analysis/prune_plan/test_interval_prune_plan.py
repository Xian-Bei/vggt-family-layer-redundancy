from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    canonical_physical_pruning_selector,
    drop_from_physical_pruning_selector,
    expand_pruning_strategy_keys,
    load_named_prune_plan,
    physical_pruning_selector_aliases,
    physical_pruning_selector_from_drop,
    strategy_interval_selectors,
)
from analysis.prune_plan.build_prune_plan_v3 import _load_gt_aggregate_from_pruning_dir
from analysis.prune_plan.build_prune_plan_v3 import _shortgpt_drop_sets_from_unit_similarity
from analysis.prune_plan.build_prune_plan_v3 import _simpler_cutoff_scores
from analysis.prune_plan.build_prune_plan_v3 import (
    GT_TWO_INTERVAL_SCORE_AGGREGATION,
    GT_OURS_COMPARISON_TOLERANCE,
    GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE,
    GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP,
    GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE,
    OURS_TWO_INTERVAL_SCORE_AGGREGATION,
    _build_ours_metric_strategy_pair,
    _each_interval_overlaps_previous_plan,
    _official_replaceme_drop_sets_from_distance,
    _one_or_two_interval_drop_sets_from_score,
    _parse_pruning_strategies,
    _similarity_proxy_evidence,
    _similarity_to_proxy_drop_score,
    _strategy_from_drop_sets,
    _strategy_from_picks,
)
from analysis.prune_plan.migrate_lower_to_intervals import run as run_migration
from analysis.prune_plan.triangle_prune import (
    interval_pick_from_bounds,
    scan_endpoint_linked_closed_intervals_with_candidates,
)
from analysis.prune_plan.visualize_triangle_prune import (
    _draw_drop_grid_evidence_panel,
    _group_drop_grid_items_by_method,
    largest_adjacent_similarity_gaps,
    max_similarity_by_drop_count,
    render_max_similarity_by_drop_count,
    render_prune_strategy_drop_grids,
    render_similarity_gap_heatmaps,
)
from analysis.redundancy.similarity_v3 import (
    CLOSED_INTERVAL_GRAPH_SEMANTICS,
    UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS,
    unit_similarity_to_interval_score,
)
from analysis.triangular_heatmap import draw_symmetric_heatmap_panel
from scripts.tmp_migrate_interval_graphs_to_pairwise_semantics import convert_payload


def _bundle() -> dict:
    return {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "scannet_dev50,nuscenes_train",
        "num_units": 8,
        "param_accounting": {
            "available": True,
            "total_params": 800,
        },
        "source_cache": "",
        "intervals": {
            "i1_j3": {
                "selector": "i1_j3",
                "i": 1,
                "j": 3,
                "start": 1,
                "end": 3,
                "span": 3,
                "drop_count": 3,
                "unit_i": "global_0",
                "unit_j": "global_1",
                "drop": [1, 2, 3],
                "dropped_unit_ids": ["global_0", "frame_1", "global_1"],
                "aggregator_params": 600,
            },
            "i2_j5": {
                "selector": "i2_j5",
                "i": 2,
                "j": 5,
                "start": 2,
                "end": 5,
                "span": 4,
                "drop_count": 4,
                "unit_i": "frame_1",
                "unit_j": "global_2",
                "drop": [2, 3, 4, 5],
                "dropped_unit_ids": ["frame_1", "global_1", "frame_2", "global_2"],
                "aggregator_params": 500,
            },
        },
        "sets": {
            "shortgpt_k2": {
                "selector": "shortgpt_k2",
                "selector_kind": "set",
                "drop_count": 2,
                "span": 2,
                "drop": [1, 4],
                "dropped_unit_ids": ["global_0", "frame_2"],
                "source_strategy": "shortgpt",
                "score_name": "unit_redundancy_score",
                "score": 0.95,
                "ranking": [4, 1, 2, 3, 5, 6, 7, 0],
                "aggregator_params": 600,
            },
        },
        "strategies": {
            "ours": {
                "display_name": "Ours",
                "points": {"3": "i1_j3", "4": "i2_j5"},
            },
            "shortgpt": {
                "display_name": "ShortGPT",
                "points": {"2": "shortgpt_k2"},
            },
            "inv_mse": {
                "display_name": "INV_MSE",
                "points": {"3": "i1_j3", "4": "i2_j5"},
            },
            "gt": {
                "display_name": "GT",
                "points": {"3": "i1_j3"},
            },
        },
        "spans": {
            "3": {
                "selector": "i1_j3",
                "i": 1,
                "j": 3,
                "start": 1,
                "end": 3,
                "span": 3,
                "drop_count": 3,
                "unit_i": "global_0",
                "unit_j": "global_1",
                "drop": [1, 2, 3],
                "dropped_unit_ids": ["global_0", "frame_1", "global_1"],
            }
        },
        "extra": {},
    }


class IntervalPrunePlanTest(unittest.TestCase):
    def test_gt_and_ours_use_separate_two_interval_aggregations(self) -> None:
        self.assertEqual(GT_TWO_INTERVAL_SCORE_AGGREGATION, "sum")
        self.assertEqual(OURS_TWO_INTERVAL_SCORE_AGGREGATION, "max")

    def test_ours_metric_families_share_identical_single_double_selection(self) -> None:
        score = np.full((6, 6), np.nan, dtype=np.float64)
        for start in range(1, 6):
            for end in range(start, 6):
                score[start, end] = 0.55 + 0.03 * start + 0.01 * end

        def build(family: str, metric: str):
            sets: dict[str, dict] = {}
            strategies = _build_ours_metric_strategy_pair(
                family_key=family,
                display_name=family,
                metric=metric,
                interval_score=score,
                similarity_source={"metric": metric},
                token_region="mean",
                dataset="scannet_dev50,nuscenes_train",
                unit_ids=[f"unit_{index}" for index in range(6)],
                num_units=6,
                param_accounting={},
                sets=sets,
            )
            drops = {
                variant: {
                    drop_count: sets[selector]["drop"]
                    for drop_count, selector in strategies[
                        f"{family}_{variant}"
                    ]["points"].items()
                }
                for variant in ("single", "double")
            }
            return strategies, drops

        ours, ours_drops = build("ours", "cka")
        cosine, cosine_drops = build("cosine", "cosine")
        inv_mse, inv_mse_drops = build("inv_mse", "inv_mse")
        self.assertEqual(ours_drops, cosine_drops)
        self.assertEqual(ours_drops, inv_mse_drops)

        for strategies, family, metric in (
            (ours, "ours", "cka"),
            (cosine, "cosine", "cosine"),
            (inv_mse, "inv_mse", "inv_mse"),
        ):
            self.assertEqual(
                set(strategies),
                {f"{family}_single", f"{family}_double"},
            )
            single_source = strategies[f"{family}_single"]["source"]
            double_source = strategies[f"{family}_double"]["source"]
            self.assertEqual(single_source["metric"], metric)
            self.assertEqual(single_source["proxy_drop"], "D[start,end]=1-C[start,end]")
            self.assertEqual(single_source["optimization_direction"], "minimize")
            self.assertEqual(single_source["max_num_intervals"], 1)
            self.assertNotIn(
                "two_interval_score_aggregation",
                single_source,
            )
            self.assertEqual(double_source["max_num_intervals"], 2)
            self.assertEqual(
                double_source["two_interval_score_aggregation"],
                OURS_TWO_INTERVAL_SCORE_AGGREGATION,
            )
            self.assertEqual(
                double_source["comparison_tolerance"],
                GT_OURS_COMPARISON_TOLERANCE,
            )
            self.assertEqual(
                double_source["double_gap_comparison_tolerance"],
                GT_OURS_DOUBLE_GAP_COMPARISON_TOLERANCE,
            )
            self.assertEqual(
                double_source["single_double_comparison_tolerance"],
                GT_OURS_SINGLE_DOUBLE_COMPARISON_TOLERANCE,
            )
            self.assertEqual(double_source["tie_policy"], "prefer_single_interval")
            self.assertEqual(
                double_source["cross_budget_path_constraint"],
                (
                    "each_component_interval_at_drop_count_n_overlaps_at_least_one_"
                    "component_interval_at_drop_count_n_plus_1;"
                    "double_to_double_requires_one_to_one_overlap"
                    if GT_OURS_REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                    else False
                ),
            )
            self.assertEqual(
                double_source["cross_budget_scan_direction"],
                "large_to_small",
            )
            self.assertTrue(double_source["double_one_split_only"])

    def test_legacy_strategy_family_names_expand_for_explicit_bundles(self) -> None:
        available = {
            "gt_single",
            "gt_double",
            "ours_single",
            "ours_double",
            "cosine_single",
            "cosine_double",
            "inv_mse_single",
            "inv_mse_double",
        }
        self.assertEqual(
            expand_pruning_strategy_keys(
                ["gt", "ours_double", "ours", "cosine", "inv_mse"],
                available=available,
            ),
            [
                "gt_single",
                "gt_double",
                "ours_double",
                "ours_single",
                "cosine_single",
                "cosine_double",
                "inv_mse_single",
                "inv_mse_double",
            ],
        )
        self.assertEqual(
            expand_pruning_strategy_keys(["gt"], available={"gt"}),
            ["gt"],
        )

    def test_pruning_strategy_selector_preserves_requested_subset_order(self) -> None:
        self.assertEqual(
            _parse_pruning_strategies(
                "gt,ours,cosine,inv_mse,replaceme,shortgpt,simpler,ours"
            ),
            [
                "gt_single",
                "gt_double",
                "ours_single",
                "ours_double",
                "cosine_single",
                "cosine_double",
                "inv_mse_single",
                "inv_mse_double",
                "replaceme_single",
                "replaceme_double",
                "shortgpt",
                "simpler",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _parse_pruning_strategies("ours,unknown")

    def test_drop_grid_groups_single_and_double_variants_by_method(self) -> None:
        rows = _group_drop_grid_items_by_method(
            [
                ("GT Double", {1: [1]}),
                ("Ours Single", {1: [2]}),
                ("ShortGPT", {1: [3]}),
                ("GT Single", {1: [4]}),
                ("Ours Double", {1: [5]}),
            ]
        )
        self.assertEqual(
            [
                (
                    None if single is None else single[0],
                    None if double is None else double[0],
                )
                for single, double in rows
            ],
            [
                ("GT Single", "GT Double"),
                ("Ours Single", "Ours Double"),
                ("ShortGPT", None),
            ],
        )

    def test_drop_grid_packs_shortgpt_and_simpler_into_one_row(self) -> None:
        rows = _group_drop_grid_items_by_method(
            [
                ("SIMPLER", {1: [4]}),
                ("Ours_CKA Single", {1: [2]}),
                ("ShortGPT", {1: [3]}),
                ("Ours_CKA Double", {1: [2, 5]}),
            ]
        )
        self.assertEqual(
            [
                (
                    None if left is None else left[0],
                    None if right is None else right[0],
                )
                for left, right in rows
            ],
            [
                ("ShortGPT", "SIMPLER"),
                ("Ours_CKA Single", "Ours_CKA Double"),
            ],
        )

    def test_drop_grid_renders_two_method_variant_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = render_prune_strategy_drop_grids(
                num_units=6,
                strategy_drop_indices=[
                    ("GT Single", {1: [1]}),
                    ("GT Double", {1: [1, 4]}),
                    ("Ours Single", {1: [2]}),
                    ("Ours Double", {1: [2, 5]}),
                ],
                out_dir=Path(td),
                backbone="vggt",
            )
            from matplotlib import image as mpl_image

            image = mpl_image.imread(path)
        height, width = image.shape[:2]
        self.assertGreater(width / height, 0.7)
        self.assertLess(width / height, 1.1)

    def test_drop_grid_evidence_can_show_only_similarity_diagonal(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            with mock.patch(
                "analysis.prune_plan.visualize_triangle_prune."
                "draw_symmetric_heatmap_panel",
                wraps=draw_symmetric_heatmap_panel,
            ) as public_draw:
                _draw_drop_grid_evidence_panel(
                    ax,
                    evidence={
                        "values": np.arange(25, dtype=np.float64).reshape(5, 5),
                        "mode": "diagonal",
                        "title": "ShortGPT evidence",
                    },
                    num_units=5,
                )
            public_draw.assert_called_once()
            displayed = np.ma.asarray(ax.images[0].get_array())
            self.assertEqual(int(np.count_nonzero(~np.ma.getmaskarray(displayed))), 5)
            self.assertEqual(len(ax.lines), 0)
        finally:
            plt.close(fig)

    def test_public_evidence_heatmap_is_symmetric_square_and_reversed_coolwarm(self) -> None:
        import matplotlib.pyplot as plt

        upper = np.full((4, 4), np.nan, dtype=np.float64)
        upper[np.triu_indices(4)] = np.arange(10, dtype=np.float64)
        fig, ax = plt.subplots()
        try:
            image, _colorbar = draw_symmetric_heatmap_panel(ax, upper)
            displayed = np.ma.asarray(image.get_array())
            self.assertEqual(
                int(np.count_nonzero(~np.ma.getmaskarray(displayed))),
                16,
            )
            np.testing.assert_equal(
                np.asarray(displayed),
                np.asarray(displayed).T,
            )
            self.assertEqual(image.get_cmap().name, "coolwarm_r")
        finally:
            plt.close(fig)

    def test_drop_grid_renders_raw_evidence_as_third_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = render_prune_strategy_drop_grids(
                num_units=6,
                strategy_drop_indices=[
                    ("Ours_CKA Single", {1: [1]}),
                    ("Ours_CKA Double", {1: [1, 4]}),
                ],
                strategy_evidence={
                    "Ours_CKA": {
                        "values": np.triu(np.full((6, 6), 0.25)),
                        "title": "1 − CKA",
                    }
                },
                out_dir=Path(td),
                backbone="vggt",
            )
            from matplotlib import image as mpl_image

            image = mpl_image.imread(path)
        height, width = image.shape[:2]
        self.assertGreater(width / height, 2.0)

    def test_similarity_proxy_evidence_uses_unit_color_scale(self) -> None:
        score = np.asarray([[1.0, 0.25], [np.nan, 1.0]], dtype=np.float64)
        evidence = _similarity_proxy_evidence(
            score,
            display_name="Ours_Cosine",
            metric_label="cosine",
        )
        np.testing.assert_allclose(
            evidence["values"],
            np.asarray([[0.0, 0.75], [np.nan, 0.0]], dtype=np.float64),
            equal_nan=True,
        )
        self.assertEqual(evidence["vmin"], 0.0)
        self.assertEqual(evidence["vmax"], 1.0)
        self.assertEqual(evidence["title"], "Ours_Cosine evidence: 1 − cosine")

    def test_largest_adjacent_similarity_gaps(self) -> None:
        score = np.full((4, 4), np.nan, dtype=np.float64)
        score[0, :] = 1.0  # Placeholder row must not participate.
        score[1, 1], score[1, 2], score[1, 3] = 0.1, 0.2, 0.9
        score[2, 2], score[2, 3] = 0.8, 0.85
        score[3, 3] = 0.1
        gaps = largest_adjacent_similarity_gaps(score)
        self.assertEqual(gaps["row"]["first"], (2, 3))
        self.assertEqual(gaps["row"]["second"], (3, 3))
        self.assertAlmostEqual(gaps["row"]["delta"], 0.75)
        self.assertEqual(gaps["column"]["first"], (1, 2))
        self.assertEqual(gaps["column"]["second"], (1, 3))
        self.assertAlmostEqual(gaps["column"]["delta"], 0.7)

    def test_render_similarity_gap_heatmaps(self) -> None:
        score = np.full((4, 4), np.nan, dtype=np.float64)
        score[1, 1], score[1, 2], score[1, 3] = 0.9, 0.8, 0.2
        score[2, 2], score[2, 3], score[3, 3] = 0.7, 0.6, 0.1
        with tempfile.TemporaryDirectory() as td:
            path = render_similarity_gap_heatmaps(
                {"cka": score, "cca": score},
                out_dir=Path(td),
                backbone="vggt",
            )
            self.assertEqual(
                path.name,
                "prune_plan_vggt_similarity_gap_heatmaps.png",
            )
            self.assertTrue(path.is_file())

    def test_max_similarity_curve_uses_closed_interval_drop_count(self) -> None:
        score = np.full((5, 5), np.nan, dtype=np.float64)
        score[0, :] = 1.0  # Placeholder row must not participate.
        score[1, 1], score[2, 2] = 0.4, 0.7
        score[1, 2], score[2, 3] = 0.8, 0.9
        score[1, 3] = 0.6
        series = max_similarity_by_drop_count(score)
        self.assertEqual(
            series,
            [(1, 0.7, 2, 2), (2, 0.9, 2, 3), (3, 0.6, 1, 3)],
        )

    def test_render_max_similarity_curve(self) -> None:
        score = np.full((4, 4), np.nan, dtype=np.float64)
        score[1, 1], score[1, 2], score[1, 3] = 0.9, 0.8, 0.7
        with tempfile.TemporaryDirectory() as td:
            path = render_max_similarity_by_drop_count(
                {"cka": score, "cosine": score},
                out_dir=Path(td),
                backbone="vggt",
                gt_degradation_series=[(0, 0.0), (1, 0.2), (2, 0.5)],
            )
            self.assertEqual(
                path.name,
                "prune_plan_vggt_max_similarity_by_drop_count.png",
            )
            self.assertTrue(path.is_file())

    def test_replaceme_style_scan_can_select_each_drop_count_independently(self) -> None:
        score = np.full((4, 4), np.nan, dtype=np.float64)
        score[1, 1], score[2, 2], score[3, 3] = 0.9, 0.2, 0.3
        score[1, 2], score[2, 3] = 0.2, 0.1
        score[1, 3] = 0.7
        picks, _ = scan_endpoint_linked_closed_intervals_with_candidates(
            score,
            [f"unit_{i}" for i in range(4)],
            require_endpoint_linked_spans=False,
        )
        self.assertEqual(
            [(pick.span, pick.i, pick.j) for pick in picks],
            [(1, 1, 1), (2, 1, 2), (3, 1, 3)],
        )

    def test_official_replaceme_double_uses_equal_blocks_and_even_budgets(self) -> None:
        distance = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                distance[start, end] = 1.0
        distance[1, 2] = 0.1
        distance[3, 4] = 0.2

        drops, _scores, extra = _official_replaceme_drop_sets_from_distance(
            distance,
            num_blocks=2,
        )

        self.assertTrue(all(total % 2 == 0 for total in drops))
        self.assertEqual(drops[4], [1, 2, 3, 4])
        self.assertEqual(extra[4]["layers_to_skip"], 2)
        self.assertEqual(
            [item["drop_count"] for item in extra[4]["official_selection_order"]],
            [2, 2],
        )

    def test_official_replaceme_is_greedy_and_allows_adjacent_blocks(self) -> None:
        distance = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                distance[start, end] = 2.0
        # For layers_to_skip=3, greedy takes [2,4] first. This blocks the
        # collectively better adjacent pair [1,3] + [4,6].
        distance[2, 4] = 0.1
        distance[1, 3] = 0.2
        distance[4, 6] = 0.2
        distance[5, 7] = 0.9

        drops, _scores, extra = _official_replaceme_drop_sets_from_distance(
            distance,
            num_blocks=2,
        )

        self.assertEqual(drops[6], [2, 3, 4, 5, 6, 7])
        self.assertEqual(
            [
                (item["start"], item["end"])
                for item in extra[6]["official_selection_order"]
            ],
            [(2, 4), (5, 7)],
        )
        self.assertEqual(extra[6]["selection_objective"], "greedy_individual_angular_distance")

    def test_interval_selector_loads_unit_prune_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            plan = load_named_prune_plan(root, "i2_j5")

        self.assertEqual(plan.drop, [2, 3, 4, 5])
        self.assertEqual(plan.drop_length, 4)
        self.assertEqual(plan.extra["interval_selector"], "i2_j5")

    def test_set_selector_loads_unit_prune_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            plan = load_named_prune_plan(root, "shortgpt_k2")

        self.assertEqual(plan.drop, [1, 4])
        self.assertEqual(plan.drop_length, 2)
        self.assertEqual(plan.extra["selector_kind"], "set")
        self.assertEqual(plan.extra["score_name"], "unit_redundancy_score")
        self.assertEqual(plan.extra["eval_selector"], "i1_j1_i4_j4")

    def test_physical_two_interval_selector_loads_set_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
                json.dumps(_bundle()), encoding="utf-8"
            )
            plan = load_named_prune_plan(root, "i4_j4_i1_j1")

        self.assertEqual(plan.drop, [1, 4])
        self.assertEqual(plan.extra["eval_selector"], "i1_j1_i4_j4")

    def test_physical_selector_canonicalization_and_validation(self) -> None:
        self.assertEqual(
            physical_pruning_selector_from_drop([6, 2, 1, 5]),
            "i1_j2_i5_j6",
        )
        self.assertEqual(
            canonical_physical_pruning_selector("i5_j6_i1_j2"),
            "i1_j2_i5_j6",
        )
        self.assertEqual(
            canonical_physical_pruning_selector("i1_j2_i3_j4"),
            "i1_j4",
        )
        self.assertEqual(
            physical_pruning_selector_aliases("i1_j2_i5_j6"),
            ["i1_j2_i5_j6", "i5_j6_i1_j2"],
        )
        self.assertEqual(
            drop_from_physical_pruning_selector("i1_j2_i5_j6"),
            [1, 2, 5, 6],
        )
        self.assertEqual(
            physical_pruning_selector_from_drop([5, 1, 3]),
            "i1_j1_i3_j3_i5_j5",
        )
        self.assertEqual(
            canonical_physical_pruning_selector("i5_j5_i1_j1_i3_j3"),
            "i1_j1_i3_j3_i5_j5",
        )
        self.assertEqual(
            physical_pruning_selector_aliases("i1_j1_i3_j3_i5_j5"),
            [
                "i1_j1_i3_j3_i5_j5",
                "i5_j5_i3_j3_i1_j1",
            ],
        )
        with self.assertRaisesRegex(ValueError, "protected unit 0"):
            physical_pruning_selector_from_drop([0, 1])

    def test_prune_span_alias_loads_ours_set_selector(self) -> None:
        bundle = _bundle()
        bundle["strategies"]["ours"]["points"] = {"2": "shortgpt_k2"}
        bundle["sets"]["shortgpt_k2"]["source_strategy"] = "ours"
        bundle["sets"]["shortgpt_k2"]["component_intervals"] = [
            {"start": 1, "end": 1},
            {"start": 4, "end": 4},
        ]
        bundle["spans"] = {"2": bundle["sets"]["shortgpt_k2"]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
                json.dumps(bundle), encoding="utf-8"
            )

            plan = load_named_prune_plan(root, "prune_span2")

        self.assertEqual(plan.drop, [1, 4])
        self.assertEqual(plan.extra["selector_kind"], "set")
        self.assertEqual(len(plan.extra["component_intervals"]), 2)

    def test_strategy_interval_selectors_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            selectors = strategy_interval_selectors(
                root,
                strategies=["ours", "gt"],
                min_drop=3,
                max_drop=4,
                drop_step=1,
            )

        self.assertEqual(selectors, ["i1_j3", "i2_j5"])

    def test_strategy_name_lookup_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
                json.dumps(_bundle()), encoding="utf-8"
            )

            selectors = strategy_interval_selectors(root, strategies=["INV_MSE"])

        self.assertEqual(selectors, ["i1_j3", "i2_j5"])

    def test_strategy_selectors_include_set_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(json.dumps(_bundle()), encoding="utf-8")

            selectors = strategy_interval_selectors(
                root,
                strategies=["shortgpt"],
                min_drop=2,
                max_drop=2,
                drop_step=1,
            )

        self.assertEqual(selectors, ["i1_j1_i4_j4"])

    def test_migration_maps_open_left_interval_dirs_to_closed_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "analysis_outputs_v3"
            backbone_root = output_root / "vggt"
            backbone_root.mkdir(parents=True)
            (backbone_root / PRUNE_PLAN_BUNDLE_FILENAME).write_text(
                json.dumps(_bundle()),
                encoding="utf-8",
            )
            source = backbone_root / "scannet_dev50" / "eval" / "lower" / "i1_j3"
            source.mkdir(parents=True)
            (source / "eval_record_v3.json").write_text("{}", encoding="utf-8")

            moves = run_migration(output_root=output_root, backbone="vggt", commit=True)

            target = backbone_root / "scannet_dev50" / "eval" / "lower" / "i2_j3"
            self.assertEqual(moves, [(source, target)])
            self.assertFalse(source.exists())
            self.assertTrue((target / "eval_record_v3.json").is_file())

    def test_unit_similarity_to_interval_score_uses_left_boundary(self) -> None:
        sim = np.arange(25, dtype=np.float64).reshape(5, 5)
        score = unit_similarity_to_interval_score(sim)

        self.assertEqual(float(score[3, 4]), float(sim[2, 4]))
        self.assertTrue(np.isnan(score[0, 4]))
        self.assertTrue(np.isnan(score[4, 3]))

    def test_scanner_naturally_excludes_unit_zero_from_derived_score(self) -> None:
        sim = np.eye(4, dtype=np.float64)
        sim[0, 1:] = sim[1:, 0] = 0.9
        score = unit_similarity_to_interval_score(sim)
        picks, candidates = scan_endpoint_linked_closed_intervals_with_candidates(
            score,
            [f"unit_{i}" for i in range(4)],
            require_endpoint_linked_spans=False,
        )

        self.assertTrue(picks)
        self.assertTrue(candidates)
        self.assertTrue(
            all(
                pick.i >= 1
                for drop_candidates in candidates.values()
                for pick in drop_candidates
            )
        )

    def test_shortgpt_uses_bi_scores_with_at_most_two_intervals(self) -> None:
        sim = np.array(
            [
                [1.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 0.9, 0.0, 0.0],
                [0.0, 0.9, 1.0, 0.1, 0.0],
                [0.0, 0.0, 0.1, 1.0, 0.8],
                [0.0, 0.0, 0.0, 0.8, 1.0],
            ],
            dtype=np.float64,
        )
        drops, scores, ranking, extra = _shortgpt_drop_sets_from_unit_similarity(
            sim
        )

        self.assertEqual(ranking, [2, 4, 1, 3])
        self.assertEqual(drops[2], [2, 4])
        self.assertAlmostEqual(scores[2], 1.7)
        self.assertEqual(extra[2]["num_intervals"], 2)
        self.assertAlmostEqual(extra[2]["block_influence"]["2"], 0.1)

    def test_shortgpt_keeps_unconstrained_nested_global_ranking(self) -> None:
        sim = np.eye(7, dtype=np.float64)
        for idx, value in enumerate([0.9, -1.0, 0.8, -1.0, 0.7, -1.0], start=1):
            sim[idx - 1, idx] = value
            sim[idx, idx - 1] = value

        drops, _, ranking, extra = _shortgpt_drop_sets_from_unit_similarity(sim)

        self.assertEqual(ranking[:3], [1, 3, 5])
        self.assertEqual(drops[1], [1])
        self.assertEqual(drops[2], [1, 3])
        self.assertEqual(drops[3], [1, 3, 5])
        self.assertEqual(extra[3]["num_intervals"], 3)
        self.assertTrue(extra[3]["budget_path_nested"])
        self.assertIsNone(extra[3]["interval_count_constraint"])
        self.assertIsNone(extra[3]["interval_gap_constraint"])

    def test_two_interval_wins_when_sum_is_better_for_exact_total(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 0.01
        score[1, 2] = 0.9
        score[5, 7] = 0.8

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[5], [1, 2, 5, 6, 7])
        self.assertAlmostEqual(scores[5], 1.7)
        self.assertEqual(
            [(x["start"], x["end"]) for x in extra[5]["component_intervals"]],
            [(1, 2), (5, 7)],
        )
        self.assertEqual(extra[5]["num_intervals"], 2)
        self.assertFalse(extra[5]["scores_recomputed_after_first_drop"])
        self.assertEqual(
            extra[5]["best_single_interval"],
            {"start": 1, "end": 5, "drop_count": 5, "score": 0.01},
        )

    def test_max_drop_budget_caps_gt_ours_path_before_scanning(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = float(start + end)

        drops, _, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="single",
            max_drop_budget=4,
            require_cross_budget_interval_overlap=True,
        )

        self.assertEqual(sorted(drops, reverse=True), [4, 3, 2, 1])
        self.assertEqual(extra[4]["linked_to_drop_count"], None)
        self.assertEqual(extra[4]["max_drop_budget"], 4)
        self.assertTrue(all(len(drop) <= 4 for drop in drops.values()))

    def test_max_drop_budget_requires_positive_integer(self) -> None:
        score = np.eye(4, dtype=np.float64)
        for invalid in (0, -1, True, 2.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _one_or_two_interval_drop_sets_from_score(
                        score,
                        maximize=False,
                        max_drop_budget=invalid,
                    )

    def test_max_drop_budget_caps_set_and_interval_baseline_plans(self) -> None:
        unit_ids = [f"unit_{index}" for index in range(30)]
        sets: dict[str, dict] = {}
        set_strategy = _strategy_from_drop_sets(
            key="baseline",
            display_name="Baseline",
            drops_by_count={
                24: list(range(1, 25)),
                25: list(range(1, 26)),
            },
            scores_by_count={24: 1.0, 25: 2.0},
            unit_ids=unit_ids,
            num_units=len(unit_ids),
            param_accounting={},
            score_name="score",
            source={"kind": "baseline"},
            sets=sets,
            max_drop_budget=24,
        )
        self.assertEqual(set(set_strategy["points"]), {"24"})
        self.assertEqual(set(sets), {"baseline_k24"})
        self.assertEqual(set_strategy["source"]["max_drop_budget"], 24)

        intervals: dict[str, dict] = {}
        interval_strategy = _strategy_from_picks(
            key="suffix_baseline",
            display_name="Suffix Baseline",
            picks=[
                interval_pick_from_bounds(unit_ids, 1, 24),
                interval_pick_from_bounds(unit_ids, 1, 25),
            ],
            unit_ids=unit_ids,
            num_units=len(unit_ids),
            param_accounting={},
            score=None,
            score_name="score",
            source={"kind": "suffix_baseline"},
            intervals=intervals,
            max_drop_budget=24,
        )
        self.assertEqual(set(interval_strategy["points"]), {"24"})
        self.assertEqual(set(intervals), {"i1_j24"})
        self.assertEqual(interval_strategy["source"]["max_drop_budget"], 24)

    def test_ours_proxy_drop_prefers_single_when_split_has_no_net_gain(self) -> None:
        similarity = np.full((6, 6), np.nan, dtype=np.float64)
        for start in range(1, 6):
            for end in range(start, 6):
                similarity[start, end] = 0.1
        similarity[1, 2] = 0.9
        similarity[1, 1] = 0.9
        similarity[3, 3] = 1.0

        proxy_drop = _similarity_to_proxy_drop_score(similarity)
        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            proxy_drop,
            maximize=False,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[2], [1, 2])
        self.assertAlmostEqual(scores[2], 0.1)
        self.assertEqual(extra[2]["num_intervals"], 1)

    def test_single_interval_wins_when_better_or_tied(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 0.01
        score[1, 5] = 2.0
        score[1, 2] = 0.9
        score[5, 7] = 0.8

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[5], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(scores[5], 2.0)
        self.assertEqual(extra[5]["num_intervals"], 1)
        self.assertEqual(extra[5]["joint_score_aggregation"], "single_interval_score")

        # Use exactly representable binary fractions for an exact floating tie.
        score[1, 5] = 0.75
        score[1, 2] = 0.5
        score[5, 7] = 0.25
        drops, _, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            require_cross_budget_interval_overlap=False,
        )
        self.assertEqual(drops[5], [1, 2, 3, 4, 5])
        self.assertEqual(extra[5]["num_intervals"], 1)

    def test_explicit_single_and_forced_two_interval_modes(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 0.01
        score[1, 5] = 10.0
        score[1, 2] = 0.9
        score[5, 7] = 0.8

        single_drops, single_scores, single_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=True,
                selection_mode="single",
            )
        )
        two_drops, two_scores, two_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=True,
                selection_mode="two",
                min_retained_units_between_intervals=1,
            )
        )

        self.assertEqual(single_drops[5], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(single_scores[5], 10.0)
        self.assertEqual(single_extra[5]["num_intervals"], 1)
        self.assertNotIn(1, two_drops)
        self.assertEqual(two_drops[5], [1, 2, 5, 6, 7])
        self.assertAlmostEqual(two_scores[5], 1.7)
        self.assertEqual(two_extra[5]["num_intervals"], 2)
        self.assertTrue(
            all(item["num_intervals"] == 2 for item in two_extra.values())
        )

    def test_two_intervals_require_one_retained_unit_between_them(self) -> None:
        score = np.full((6, 6), np.nan, dtype=np.float64)
        score[1, 1] = 10.0
        score[3, 3] = 9.0
        score[4, 4] = 8.0

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            selection_mode="two",
            min_retained_units_between_intervals=1,
        )

        self.assertEqual(drops[2], [1, 3])
        self.assertAlmostEqual(scores[2], 19.0)
        self.assertEqual(extra[2]["num_intervals"], 2)
        self.assertEqual(
            extra[2]["min_retained_units_between_intervals"],
            1,
        )

    def test_two_interval_candidates_prefer_larger_gap_within_tolerance(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 1] = 0.1
        score[3, 3] = 0.1
        score[5, 5] = 0.105

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="two",
            double_gap_comparison_tolerance=0.01,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[2], [1, 5])
        self.assertAlmostEqual(scores[2], 0.205)
        self.assertEqual(extra[2]["double_gap_comparison_tolerance"], 0.01)
        self.assertEqual(
            extra[2]["double_candidate_tie_policy"],
            "prefer_larger_interval_gap",
        )

    def test_smaller_gap_double_must_win_beyond_gap_tolerance(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 1] = 0.1
        score[3, 3] = 0.1
        score[5, 5] = 0.111

        drops, scores, _ = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="two",
            double_gap_comparison_tolerance=0.01,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[2], [1, 3])
        self.assertAlmostEqual(scores[2], 0.2)

    def test_double_gap_preference_does_not_chain_across_tolerance(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        score[1, 1] = 0.0
        score[3, 3] = 0.0
        score[5, 5] = 0.009
        score[7, 7] = 0.018

        drops, scores, _ = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="two",
            two_interval_score_aggregation="max",
            double_gap_comparison_tolerance=0.01,
            require_cross_budget_interval_overlap=False,
        )

        # [1, 3] beats the widest [1, 7] by 0.018. The intermediate [1, 5]
        # candidate is within 0.01 of both and must not cause a chained win.
        self.assertEqual(drops[2], [1, 3])
        self.assertAlmostEqual(scores[2], 0.0)

    def test_gt_minimization_compares_single_with_two_interval_sum(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 10.0
        score[1, 5] = 0.5
        score[1, 2] = 0.1
        score[5, 7] = 0.2

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[5], [1, 2, 5, 6, 7])
        self.assertAlmostEqual(scores[5], 0.3)
        self.assertEqual(extra[5]["num_intervals"], 2)
        self.assertEqual(extra[5]["joint_score_aggregation"], "sum")
        self.assertAlmostEqual(extra[5]["best_two_interval_score_sum"], 0.3)

    def test_max_aggregation_can_choose_double_when_sum_chooses_single(self) -> None:
        score = np.full((5, 5), np.nan, dtype=np.float64)
        score[1, 2] = 0.7
        score[1, 1] = 0.2
        score[3, 3] = 0.6

        sum_drops, sum_scores, sum_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=False,
                two_interval_score_aggregation="sum",
                min_retained_units_between_intervals=1,
            )
        )
        max_drops, max_scores, max_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=False,
                two_interval_score_aggregation="max",
                min_retained_units_between_intervals=1,
            )
        )

        self.assertEqual(sum_drops[2], [1, 2])
        self.assertAlmostEqual(sum_scores[2], 0.7)
        self.assertEqual(sum_extra[2]["num_intervals"], 1)
        self.assertAlmostEqual(sum_extra[2]["best_two_interval_score_sum"], 0.8)

        self.assertEqual(max_drops[2], [1, 3])
        self.assertAlmostEqual(max_scores[2], 0.6)
        self.assertEqual(max_extra[2]["num_intervals"], 2)
        self.assertEqual(max_extra[2]["joint_score_aggregation"], "max")
        self.assertAlmostEqual(max_extra[2]["best_two_interval_score_max"], 0.6)
        self.assertNotIn("best_two_interval_score_sum", max_extra[2])

    def test_rejects_unknown_two_interval_score_aggregation(self) -> None:
        score = np.full((4, 4), np.nan, dtype=np.float64)
        score[1, 1] = 0.1
        score[3, 3] = 0.2

        with self.assertRaisesRegex(
            ValueError,
            "two_interval_score_aggregation",
        ):
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=False,
                two_interval_score_aggregation="mean",
            )

    def test_minimization_requires_configured_two_interval_gain(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 10.0
        score[1, 5] = 0.5
        score[1, 2] = 0.2
        score[5, 7] = 0.291

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            single_double_comparison_tolerance=0.01,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
        )
        self.assertEqual(drops[5], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(scores[5], 0.5)
        self.assertEqual(extra[5]["num_intervals"], 1)

        score[5, 7] = 0.289
        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            single_double_comparison_tolerance=0.01,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
        )
        self.assertEqual(drops[5], [1, 2, 5, 6, 7])
        self.assertAlmostEqual(scores[5], 0.489)
        self.assertEqual(extra[5]["num_intervals"], 2)

    def test_maximization_requires_configured_two_interval_gain(self) -> None:
        score = np.full((8, 8), np.nan, dtype=np.float64)
        for start in range(1, 8):
            for end in range(start, 8):
                score[start, end] = 0.01
        score[1, 5] = 1.0
        score[1, 2] = 0.5
        score[5, 7] = 0.509

        drops, _, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            single_double_comparison_tolerance=0.01,
            min_retained_units_between_intervals=1,
        )
        self.assertEqual(drops[5], [1, 2, 3, 4, 5])
        self.assertEqual(extra[5]["num_intervals"], 1)

        score[5, 7] = 0.511
        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            single_double_comparison_tolerance=0.01,
            min_retained_units_between_intervals=1,
        )
        self.assertEqual(drops[5], [1, 2, 5, 6, 7])
        self.assertAlmostEqual(scores[5], 1.011)
        self.assertEqual(extra[5]["num_intervals"], 2)

    def test_cross_budget_overlap_checks_every_component_interval(self) -> None:
        previous_double = [(2, 6), (10, 12)]

        self.assertTrue(
            _each_interval_overlaps_previous_plan(
                [(2, 3), (5, 6)],
                previous_double,
            )
        )
        self.assertTrue(
            _each_interval_overlaps_previous_plan(
                [(3, 4), (11, 13)],
                previous_double,
            )
        )
        self.assertTrue(
            _each_interval_overlaps_previous_plan(
                [(4, 5)],
                previous_double,
            )
        )
        self.assertFalse(
            _each_interval_overlaps_previous_plan(
                [(3, 4), (7, 8)],
                previous_double,
            )
        )
        self.assertTrue(
            _each_interval_overlaps_previous_plan(
                [(2, 2), (5, 6)],
                [(1, 6)],
            )
        )
        self.assertFalse(
            _each_interval_overlaps_previous_plan(
                [(2, 2), (8, 9)],
                [(1, 6)],
            )
        )

    def test_adaptive_double_allows_only_one_single_to_double_split(self) -> None:
        rng = np.random.default_rng(3147)
        score = np.full((9, 9), np.nan, dtype=np.float64)
        for start in range(1, 9):
            for end in range(start, 9):
                score[start, end] = float(rng.uniform(0.0, 1.0))

        _, _, unrestricted = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
            limit_double_to_one_split=False,
        )
        _, _, limited = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            min_retained_units_between_intervals=1,
            require_cross_budget_interval_overlap=False,
            limit_double_to_one_split=True,
        )

        totals = sorted(limited, reverse=True)
        self.assertEqual(
            [unrestricted[total]["num_intervals"] for total in totals],
            [1, 2, 2, 2, 2, 1, 2, 1],
        )
        self.assertEqual(
            [limited[total]["num_intervals"] for total in totals],
            [1, 2, 2, 2, 2, 1, 1, 1],
        )
        self.assertEqual(
            limited[7]["interval_count_transition"],
            "single_to_double",
        )
        self.assertEqual(
            limited[3]["interval_count_transition"],
            "double_to_single",
        )
        self.assertTrue(limited[2]["single_to_double_split_has_occurred"])

    def test_adaptive_double_starting_split_cannot_split_again(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 1] = 10.0
        score[3, 6] = 10.0
        score[1, 5] = 1.0
        score[2, 5] = 30.0
        score[3, 4] = 10.0
        score[2, 4] = 1.0

        _, _, unrestricted = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            max_drop_budget=5,
            require_cross_budget_interval_overlap=False,
            limit_double_to_one_split=False,
        )
        _, _, limited = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            max_drop_budget=5,
            require_cross_budget_interval_overlap=False,
            limit_double_to_one_split=True,
        )

        self.assertEqual(
            [unrestricted[total]["num_intervals"] for total in (5, 4, 3)],
            [2, 1, 2],
        )
        self.assertEqual(
            [limited[total]["num_intervals"] for total in (5, 4, 3)],
            [2, 1, 1],
        )
        self.assertTrue(limited[5]["single_to_double_split_has_occurred"])
        self.assertIsNone(limited[5]["interval_count_transition"])
        self.assertEqual(
            limited[4]["interval_count_transition"],
            "double_to_single",
        )
        self.assertTrue(limited[3]["single_to_double_split_has_occurred"])
        self.assertIsNone(limited[3]["interval_count_transition"])

    def test_double_continuity_requires_one_to_one_interval_matching(self) -> None:
        score = np.full((11, 11), np.nan, dtype=np.float64)
        # At k=7, select two branches a=[2,7] and b=[10,10].
        score[2, 7] = 27.0
        score[10, 10] = 10.0
        score[1, 7] = 1.0
        # At k=6, the unconstrained best Double drops b and maps both current
        # intervals [1,2] and [4,7] to a. The best Single is selected once that
        # many-to-one continuity candidate is rejected.
        score[1, 2] = 15.0
        score[4, 7] = 15.0

        _, _, independent = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            max_drop_budget=7,
            require_cross_budget_interval_overlap=False,
            limit_double_to_one_split=False,
        )
        _, _, continuous = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            max_drop_budget=7,
            require_cross_budget_interval_overlap=True,
            limit_double_to_one_split=False,
        )

        self.assertEqual(independent[7]["num_intervals"], 2)
        self.assertEqual(independent[6]["num_intervals"], 2)
        self.assertEqual(
            [
                (item["start"], item["end"])
                for item in independent[6]["component_intervals"]
            ],
            [(1, 2), (4, 7)],
        )
        self.assertEqual(continuous[7]["num_intervals"], 2)
        self.assertEqual(continuous[6]["num_intervals"], 1)
        self.assertEqual(
            [
                (item["start"], item["end"])
                for item in continuous[6]["component_intervals"]
            ],
            [(2, 7)],
        )
        self.assertTrue(
            continuous[6][
                "cross_budget_double_interval_one_to_one_required"
            ]
        )
        self.assertFalse(continuous[6]["double_one_split_only"])
        self.assertEqual(
            continuous[6]["interval_count_transition"],
            None,
        )

    def test_cross_budget_overlap_rejects_better_discontinuous_plan(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 3] = 0.0
        score[1, 2] = 1.0
        score[4, 5] = -10.0

        independent_drops, _, _ = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            require_cross_budget_interval_overlap=False,
        )
        linked_drops, linked_scores, linked_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=False,
                require_cross_budget_interval_overlap=True,
            )
        )

        self.assertEqual(independent_drops[2], [4, 5])
        self.assertEqual(linked_drops[3], [1, 2, 3])
        self.assertEqual(linked_drops[2], [1, 2])
        self.assertAlmostEqual(linked_scores[2], 1.0)
        self.assertEqual(linked_extra[2]["linked_to_drop_count"], 3)
        self.assertTrue(
            linked_extra[2]["cross_budget_interval_overlap_required"]
        )

    def test_single_mode_rejects_better_discontinuous_interval(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 3] = 0.0
        score[1, 2] = 1.0
        score[4, 5] = -10.0

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="single",
            require_cross_budget_interval_overlap=True,
        )

        self.assertEqual(drops[3], [1, 2, 3])
        self.assertEqual(drops[2], [1, 2])
        self.assertAlmostEqual(scores[2], 1.0)
        self.assertEqual(extra[2]["num_intervals"], 1)
        self.assertEqual(extra[2]["linked_to_drop_count"], 3)

    def test_cross_budget_overlap_holds_across_full_double_plan_curve(self) -> None:
        rng = np.random.default_rng(7)
        score = np.full((10, 10), np.nan, dtype=np.float64)
        for start in range(1, 10):
            for end in range(start, 10):
                score[start, end] = float(rng.uniform(0.1, 1.0))

        _, _, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            require_cross_budget_interval_overlap=True,
        )

        totals = sorted(extra, reverse=True)
        self.assertTrue(any(extra[total]["num_intervals"] == 2 for total in totals))
        for previous_total, total in zip(totals, totals[1:]):
            self.assertEqual(previous_total, total + 1)
            previous_components = [
                (item["start"], item["end"])
                for item in extra[previous_total]["component_intervals"]
            ]
            components = [
                (item["start"], item["end"])
                for item in extra[total]["component_intervals"]
            ]
            self.assertTrue(
                _each_interval_overlaps_previous_plan(
                    components,
                    previous_components,
                )
            )

    def test_simpler_scores_cutoffs_from_tl_minus_br_delta(self) -> None:
        sim = np.array(
            [
                [1.0, 0.9, 0.1, 0.1],
                [0.9, 1.0, 0.2, 0.2],
                [0.1, 0.2, 1.0, 0.8],
                [0.1, 0.2, 0.8, 1.0],
            ],
            dtype=np.float64,
        )

        scores, meta = _simpler_cutoff_scores(sim)

        self.assertAlmostEqual(float(scores[2]), -0.1)
        self.assertEqual(meta["paper_valid_cutoffs"], [2])

    def test_tmp_migration_restores_pairwise_mean_and_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "redundancy_graph_all.npz"
            sim = np.array(
                [
                    [1.0, 0.4, 0.9],
                    [0.4, 1.0, 0.7],
                    [0.9, 0.7, 1.0],
                ],
                dtype=np.float64,
            )
            interval = unit_similarity_to_interval_score(sim)
            interval[0, :] = np.array([0.1, 0.2, 0.3])
            np.savez(
                path,
                unit_ids=np.asarray(["u0", "u1", "u2"], dtype=object),
                scene_ids=np.asarray(["a", "b"], dtype=object),
                graph_semantics=np.asarray([CLOSED_INTERVAL_GRAPH_SEMANTICS]),
                start_zero_value=np.asarray([0.0]),
                legacy_unit_similarity_recoverable=np.asarray([True]),
                cka_mean=interval,
                cka_scenes=np.stack([interval, interval], axis=0),
            )

            payload = convert_payload(path)
            self.assertEqual(
                str(payload["graph_semantics"][0]),
                UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS,
            )
            np.testing.assert_array_equal(payload["cka_mean"], sim)
            np.testing.assert_array_equal(payload["cka_scenes"][1], sim)
            self.assertNotIn("start_zero_value", payload)

    def test_load_gt_aggregate_from_pruning_interval_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            eval_root = Path(td) / "eval"
            pruning = eval_root / "pruning"
            upper = eval_root / "upper"
            upper.mkdir(parents=True)
            upper.joinpath("eval_record_v3.json").write_text(
                json.dumps(
                    {
                        "average": {
                            "ape": 1.0,
                            "are": 1.0,
                            "rpe_trans": 1.0,
                            "rpe_rot": 1.0,
                            "acc": 1.0,
                            "comp": 1.0,
                            "cd": 1.0,
                            "nc": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for selector, value in {"i0_j0": 1.2, "i1_j1": 1.1, "i0_j1": 1.5}.items():
                start, end = [int(part[1:]) for part in selector.split("_")]
                out_dir = pruning / selector
                out_dir.mkdir(parents=True)
                out_dir.joinpath("summary.json").write_text(
                    json.dumps(
                        {
                            "selector": selector,
                            "start": start,
                            "end": end,
                            "metrics_mean": {
                                "ape": value,
                                "are": value,
                                "rpe_trans": value,
                                "rpe_rot": value,
                                "acc": value,
                                "comp": value,
                                "chamfer_distance": value,
                                "nc": 1.0 / value,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            score, components, picks, meta = _load_gt_aggregate_from_pruning_dir(
                pruning,
                unit_ids=["u0", "u1"],
                metric_selector="all",
            )

        self.assertAlmostEqual(float(score[1, 1]), 0.0)
        self.assertTrue(np.isnan(float(score[0, 0])))
        self.assertNotIn("nc", components)
        self.assertEqual([(p.i, p.j, p.span) for p in picks], [(1, 1, 1)])
        self.assertEqual(meta["source_kind"], "eval_pruning_interval_summaries")
        self.assertNotIn("i0_j1", meta["selector_scores"])
        self.assertEqual(
            meta["component_normalization"]["method"],
            "per_metric_finite_minmax_then_nanmean",
        )


if __name__ == "__main__":
    unittest.main()

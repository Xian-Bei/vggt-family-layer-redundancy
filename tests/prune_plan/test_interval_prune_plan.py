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
    load_prune_plan_bundle,
    load_named_prune_plan,
    physical_pruning_selector_aliases,
    physical_pruning_selector_from_drop,
    strategy_interval_selectors,
)
from analysis.prune_plan.build_prune_plan_v3 import _load_gt_aggregate_from_pruning_dir
from analysis.prune_plan.build_prune_plan_v3 import _gardener_drop_sets_from_scores
from analysis.prune_plan.build_prune_plan_v3 import _gardener_exact_value_entropy
from analysis.prune_plan.build_prune_plan_v3 import _shortgpt_drop_sets_from_unit_similarity
from analysis.prune_plan.build_prune_plan_v3 import _simpler_cutoff_scores
from analysis.prune_plan.build_prune_plan_v3 import (
    REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP,
    SINGLE_DOUBLE_COMPARISON_TOLERANCE,
    _build_parser,
    _build_similarity_metric_strategy_pair,
    _each_interval_overlaps_previous_plan,
    _gt_picks_from_score,
    _load_unit_ids_from_skip_eval_reports,
    _official_replaceme_drop_sets_from_distance,
    _one_or_two_interval_drop_sets_from_score,
    _parse_pruning_strategies,
    _similarity_proxy_evidence,
    _similarity_to_proxy_drop_score,
    _strategy_from_drop_sets,
    _strategy_from_picks,
    run_build_prune_plan,
)
from analysis.prune_plan.triangle_prune import (
    interval_pick_from_bounds,
    scan_endpoint_linked_closed_intervals_with_candidates,
)
from analysis.prune_plan.visualize_triangle_prune import (
    _draw_drop_grid_panel,
    _draw_drop_grid_evidence_panel,
    _drop_grid_display_title,
    _group_drop_grid_items_by_method,
    largest_adjacent_similarity_gaps,
    max_similarity_by_drop_count,
    render_max_similarity_by_drop_count,
    render_prune_strategy_drop_grids,
    render_similarity_gap_heatmaps,
)
from analysis.redundancy.similarity_v3 import (
    unit_similarity_to_interval_score,
)
from analysis.triangular_heatmap import draw_symmetric_heatmap_panel


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
            "CKA": {
                "display_name": "CKA",
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
            "retention": {
                "display_name": "Retention",
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
    def test_bundle_loader_rejects_noncanonical_strategy_keys_and_old_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / PRUNE_PLAN_BUNDLE_FILENAME

            noncanonical = _bundle()
            noncanonical["strategies"]["cka_single"] = noncanonical["strategies"].pop("CKA")
            path.write_text(json.dumps(noncanonical), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical keys"):
                load_prune_plan_bundle(root)

            old_schema = _bundle()
            old_schema["schema_version"] = PRUNE_PLAN_BUNDLE_SCHEMA_VERSION.replace("v5", "v4")
            path.write_text(json.dumps(old_schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid prune plan bundle schema"):
                load_prune_plan_bundle(root)

    def test_cka_metric_families_share_identical_single_double_selection(self) -> None:
        score = np.full((6, 6), np.nan, dtype=np.float64)
        for start in range(1, 6):
            for end in range(start, 6):
                score[start, end] = 0.55 + 0.03 * start + 0.01 * end

        def build(family: str, metric: str):
            sets: dict[str, dict] = {}
            strategies = _build_similarity_metric_strategy_pair(
                family_key=family,
                display_name=family,
                metric=metric,
                interval_score=score,
                similarity_source={"metric": metric},
                token_region="special_patch_mean",
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

        cka, cka_drops = build("CKA", "cka")
        cosine, cosine_drops = build("cosine", "cosine")
        inv_mse, inv_mse_drops = build("inv_mse", "inv_mse")
        cca, cca_drops = build("cca", "cca")
        self.assertEqual(cka_drops, cosine_drops)
        self.assertEqual(cka_drops, inv_mse_drops)
        self.assertEqual(cka_drops, cca_drops)

        for strategies, family, metric in (
            (cka, "CKA", "cka"),
            (cosine, "cosine", "cosine"),
            (inv_mse, "inv_mse", "inv_mse"),
            (cca, "cca", "cca"),
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
                "sum",
            )
            self.assertEqual(
                double_source["single_double_comparison_tolerance"],
                SINGLE_DOUBLE_COMPARISON_TOLERANCE,
            )
            self.assertEqual(double_source["tie_policy"], "prefer_single_interval")
            self.assertEqual(
                double_source["cross_budget_path_constraint"],
                (
                    "each_component_interval_at_drop_count_n_overlaps_at_least_one_"
                    "component_interval_at_drop_count_n_plus_1;"
                    "double_to_double_requires_one_to_one_overlap"
                    if REQUIRE_CROSS_BUDGET_INTERVAL_OVERLAP
                    else False
                ),
            )
            self.assertEqual(
                double_source["cross_budget_scan_direction"],
                "large_to_small",
            )

    def test_strategy_family_names_expand_for_explicit_bundles(self) -> None:
        available = {
            "retention_single",
            "retention_double",
            "CKA_single",
            "CKA_double",
            "cosine_single",
            "cosine_double",
            "inv_mse_single",
            "inv_mse_double",
            "cca_single",
            "cca_double",
        }
        self.assertEqual(
            expand_pruning_strategy_keys(
                ["retention", "CKA_double", "CKA", "cosine", "inv_mse", "cca"],
                available=available,
            ),
            [
                "retention_single",
                "retention_double",
                "CKA_double",
                "CKA_single",
                "cosine_single",
                "cosine_double",
                "inv_mse_single",
                "inv_mse_double",
                "cca_single",
                "cca_double",
            ],
        )
        self.assertEqual(
            expand_pruning_strategy_keys(["retention"], available={"retention"}),
            ["retention"],
        )

    def test_pruning_strategy_selector_preserves_requested_subset_order(self) -> None:
        self.assertEqual(
            _parse_pruning_strategies(
                "RETENTION,cka,cosine,inv_mse,cca,replaceme,shortgpt,gardener,simpler,CKA"
            ),
            [
                "retention_single",
                "retention_double",
                "CKA_single",
                "CKA_double",
                "cosine_single",
                "cosine_double",
                "inv_mse_single",
                "inv_mse_double",
                "cca_single",
                "cca_double",
                "replaceme_single",
                "replaceme_double",
                "shortgpt",
                "gardener",
                "simpler",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _parse_pruning_strategies("cka,unknown")

    def test_cli_accepts_special_patch_mean_token_region(self) -> None:
        args = _build_parser().parse_args(
            ["--dataset", "scannet_dev50", "--token-region", "special_patch_mean"]
        )
        self.assertEqual(args.token_region, "special_patch_mean")

    def test_cli_only_accepts_supported_pruning_token_regions(self) -> None:
        parser = _build_parser()
        action = next(
            action for action in parser._actions if action.dest == "token_region"
        )
        self.assertEqual(
            tuple(action.choices),
            ("all", "special", "patch", "special_patch_mean"),
        )
        self.assertEqual(
            parser.parse_args(["--dataset", "scannet_dev50"]).token_region,
            "special_patch_mean",
        )

    def test_drop_grid_groups_single_and_double_variants_by_method(self) -> None:
        rows = _group_drop_grid_items_by_method(
            [
                ("Retention Double", {1: [1]}),
                ("CKA Single", {1: [2]}),
                ("ShortGPT", {1: [3]}),
                ("Retention Single", {1: [4]}),
                ("CKA Double", {1: [5]}),
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
                ("Retention Single", "Retention Double"),
                ("CKA Single", "CKA Double"),
                ("ShortGPT", None),
            ],
        )

    def test_drop_grid_packs_shortgpt_and_simpler_into_one_row(self) -> None:
        rows = _group_drop_grid_items_by_method(
            [
                ("SIMPLER", {1: [4]}),
                ("CKA Single", {1: [2]}),
                ("ShortGPT", {1: [3]}),
                ("CKA Double", {1: [2, 5]}),
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
                ("CKA Single", "CKA Double"),
            ],
        )

    def test_drop_grid_renders_two_method_variant_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = render_prune_strategy_drop_grids(
                num_units=6,
                strategy_drop_indices=[
                    ("Retention Single", {1: [1]}),
                    ("Retention Double", {1: [1, 4]}),
                    ("CKA Single", {1: [2]}),
                    ("CKA Double", {1: [2, 5]}),
                ],
                out_dir=Path(td),
                backbone="vggt",
            )
            from matplotlib import image as mpl_image

            image = mpl_image.imread(path)
        height, width = image.shape[:2]
        self.assertGreater(width / height, 1.5)
        self.assertLess(width / height, 2.5)

    def test_drop_grid_uses_actual_budget_and_publication_labels(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            _draw_drop_grid_panel(
                ax,
                num_units=12,
                span_to_drop_indices={1: [2], 8: [2, 3, 4, 5, 6, 7, 8, 9]},
                title="Degradation (Single)",
            )
            self.assertEqual(ax.get_xlabel(), "Droped Layers")
            self.assertEqual(ax.get_ylabel(), "Budget")
            self.assertEqual(tuple(ax.get_xlim()), (-0.5, 11.5))
            self.assertEqual(tuple(ax.get_ylim()), (0.5, 8.5))
            self.assertEqual(list(ax.get_xticks()), [0, 4, 8])
            self.assertEqual(list(ax.get_yticks()), [4, 8])
            self.assertGreaterEqual(ax.title.get_fontsize(), 24)
        finally:
            plt.close(fig)

    def test_drop_grid_titles_rename_retention_and_parenthesize_structure(self) -> None:
        self.assertEqual(
            _drop_grid_display_title("Retention Single"),
            "Degradation (Single)",
        )
        self.assertEqual(_drop_grid_display_title("CKA Double"), "CKA (Double)")

    def test_drop_grid_defaults_to_retention_and_cka(self) -> None:
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "analysis.prune_plan.visualize_triangle_prune._draw_drop_grid_panel"
        ) as draw_panel:
            render_prune_strategy_drop_grids(
                num_units=12,
                strategy_drop_indices=[
                    ("Retention Single", {4: [1, 2, 3, 4]}),
                    ("CKA Single", {4: [2, 3, 4, 5]}),
                    ("Cosine Single", {4: [3, 4, 5, 6]}),
                ],
                out_dir=Path(td),
                backbone="vggt",
            )
        self.assertEqual(draw_panel.call_count, 2)
        self.assertEqual(
            [call.kwargs["title"] for call in draw_panel.call_args_list],
            ["Degradation (Single)", "CKA (Single)"],
        )

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
                    ("CKA Single", {1: [1]}),
                    ("CKA Double", {1: [1, 4]}),
                ],
                strategy_evidence={
                    "CKA": {
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
            display_name="Cosine",
            metric_label="cosine",
        )
        np.testing.assert_allclose(
            evidence["values"],
            np.asarray([[0.0, 0.75], [np.nan, 0.0]], dtype=np.float64),
            equal_nan=True,
        )
        self.assertEqual(evidence["vmin"], 0.0)
        self.assertEqual(evidence["vmax"], 1.0)
        self.assertEqual(evidence["title"], "Cosine evidence: 1 − cosine")

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

    def test_prune_span_alias_loads_cka_set_selector(self) -> None:
        bundle = _bundle()
        bundle["strategies"]["CKA"]["points"] = {"2": "shortgpt_k2"}
        bundle["sets"]["shortgpt_k2"]["source_strategy"] = "CKA"
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
                strategies=["CKA", "retention"],
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

    def test_gardener_entropy_uses_only_exact_rank_two_weight_values(self) -> None:
        import torch

        class ToyBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kernel = torch.nn.Parameter(
                    torch.tensor([[0.0, 0.0], [1.0, 2.0]])
                )
                self.bias = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

        entropy, details = _gardener_exact_value_entropy(ToyBlock())

        self.assertAlmostEqual(entropy, 1.0397207708399179)
        self.assertEqual(details["selected_parameter_names"], ["kernel"])
        self.assertEqual(details["selected_parameter_count"], 4)
        self.assertEqual(details["unique_value_count"], 3)

    def test_gardener_uses_protected_nested_low_entropy_prefix(self) -> None:
        drops, scores, ranking, extra = _gardener_drop_sets_from_scores(
            np.asarray([0.0, 0.4, 0.2, 0.2], dtype=np.float64)
        )

        self.assertEqual(ranking, [2, 3, 1])
        self.assertEqual(drops, {1: [2], 2: [2, 3], 3: [1, 2, 3]})
        self.assertAlmostEqual(scores[2], 0.4)
        self.assertEqual(extra[2]["protected_unit_indices"], [0])
        self.assertEqual(extra[2]["global_pruning_order"], ranking)
        self.assertTrue(extra[2]["budget_path_nested"])
        self.assertIsNone(extra[2]["interval_count_constraint"])

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

    def test_max_drop_budget_caps_gt_cka_path_before_scanning(self) -> None:
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

    def test_cka_proxy_drop_prefers_single_when_split_has_no_net_gain(self) -> None:
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

    def test_two_interval_candidates_use_strict_score(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 1] = 0.1
        score[3, 3] = 0.1
        score[5, 5] = 0.105

        drops, scores, extra = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="two",
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[2], [1, 3])
        self.assertAlmostEqual(scores[2], 0.2)
        self.assertNotIn("single_double_comparison_tolerance", extra[2])
        self.assertEqual(
            extra[2]["double_candidate_tie_policy"],
            "strict_score_then_prefer_larger_interval_gap_on_exact_tie",
        )

    def test_two_interval_exact_score_tie_prefers_larger_gap(self) -> None:
        score = np.full((7, 7), np.nan, dtype=np.float64)
        score[1, 1] = 0.1
        score[3, 3] = 0.1
        score[5, 5] = 0.1

        drops, scores, _ = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=False,
            selection_mode="two",
            require_cross_budget_interval_overlap=False,
        )

        self.assertEqual(drops[2], [1, 5])
        self.assertAlmostEqual(scores[2], 0.2)

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

    def test_joint_score_adds_components_instead_of_taking_maximum(self) -> None:
        score = np.full((5, 5), np.nan, dtype=np.float64)
        score[1, 2] = 0.7
        score[1, 1] = 0.2
        score[3, 3] = 0.6

        sum_drops, sum_scores, sum_extra = (
            _one_or_two_interval_drop_sets_from_score(
                score,
                maximize=False,
                min_retained_units_between_intervals=1,
            )
        )

        self.assertEqual(sum_drops[2], [1, 2])
        self.assertAlmostEqual(sum_scores[2], 0.7)
        self.assertEqual(sum_extra[2]["num_intervals"], 1)
        self.assertAlmostEqual(sum_extra[2]["best_two_interval_score_sum"], 0.8)

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

    def test_adaptive_double_allows_repeated_single_to_double_transitions(self) -> None:
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
        )

        totals = sorted(unrestricted, reverse=True)
        self.assertEqual(
            [unrestricted[total]["num_intervals"] for total in totals],
            [1, 2, 2, 2, 2, 1, 2, 1],
        )

    def test_adaptive_double_can_split_again_after_returning_to_single(self) -> None:
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
        )

        self.assertEqual(
            [unrestricted[total]["num_intervals"] for total in (5, 4, 3)],
            [2, 1, 2],
        )

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
        )
        _, _, continuous = _one_or_two_interval_drop_sets_from_score(
            score,
            maximize=True,
            max_drop_budget=7,
            require_cross_budget_interval_overlap=True,
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

        self.assertAlmostEqual(float(score[1, 1]), 1.0)
        self.assertTrue(np.isnan(float(score[0, 0])))
        self.assertNotIn("nc", components)
        self.assertEqual([(p.i, p.j, p.span) for p in picks], [(1, 1, 1)])
        self.assertEqual(meta["source_kind"], "eval_pruning_interval_summaries")
        self.assertNotIn("i0_j1", meta["selector_scores"])
        self.assertEqual(
            meta["component_normalization"]["method"],
            "per_metric_upper_zero_all_interval_worst_then_nanmean",
        )


class RetentionOnlyBuildTest(unittest.TestCase):
    @staticmethod
    def _write_skip_report(root: Path, dataset: str, unit_ids: list[str]) -> None:
        report_path = root / dataset / "eval" / "pruning" / "skip_eval_report_v3.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": "v3",
                    "kind": "skip_eval_report_v3",
                    "dataset": dataset,
                    "num_layers": len(unit_ids),
                    "unit_ids": unit_ids,
                }
            ),
            encoding="utf-8",
        )

    def test_unit_catalog_is_loaded_and_validated_across_skip_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = ["u0", "u1", "u2", "u3"]
            self._write_skip_report(root, "d1", expected)
            self._write_skip_report(root, "d2", expected)

            unit_ids, source = _load_unit_ids_from_skip_eval_reports(
                plan_root=root,
                datasets=["d1", "d2"],
            )

        self.assertEqual(unit_ids, expected)
        self.assertEqual(source["kind"], "skip_eval_report_unit_catalog")
        self.assertEqual(source["datasets"], ["d1", "d2"])

    def test_retention_only_build_never_loads_redundancy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unit_ids = ["u0", "u1", "u2", "u3"]
            self._write_skip_report(root, "d1", unit_ids)
            self._write_skip_report(root, "d2", unit_ids)
            score = np.full((len(unit_ids), len(unit_ids)), np.nan, dtype=np.float64)
            for start in range(1, len(unit_ids)):
                for end in range(start, len(unit_ids)):
                    score[start, end] = float(start + end)
            picks = _gt_picks_from_score(score, unit_ids)
            gt_payload = (
                score,
                {"ape": score},
                picks,
                {"metrics": ["ape"], "sources": ["d1", "d2"]},
                {},
                {},
            )

            with (
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3.load_similarity",
                    side_effect=AssertionError("redundancy must not be loaded"),
                ) as load_mock,
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3._load_gt_aggregate_for_datasets",
                    return_value=gt_payload,
                ),
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3.render_prune_strategy_drop_grids",
                    return_value=root / "drop_grid.png",
                ),
            ):
                run_build_prune_plan(
                    experiment="vggt",
                    dataset="d1,d2",
                    backbone="vggt",
                    prune_plan_root=root,
                    viz_out_dir=root,
                    strategy_selector="retention_single,retention_double",
                    max_drop_budget=2,
                    skip_param_accounting=True,
                )

            bundle = json.loads(
                (root / PRUNE_PLAN_BUNDLE_FILENAME).read_text(encoding="utf-8")
            )

        load_mock.assert_not_called()
        self.assertEqual(bundle["unit_ids"], unit_ids)
        self.assertEqual(
            list(bundle["strategies"]),
            ["retention_single", "retention_double"],
        )
        self.assertIsNone(bundle["extra"]["similarity_source"])
        self.assertNotIn("similarity_selection_controls", bundle["extra"])
        retention_payload = {
            "strategies": bundle["strategies"],
            "sets": bundle["sets"],
        }
        self.assertNotIn(
            "single_double_comparison_tolerance",
            json.dumps(retention_payload),
        )
        self.assertNotIn("double_gap_comparison_tolerance", json.dumps(bundle))
        self.assertEqual(
            bundle["extra"]["unit_catalog_source"]["kind"],
            "skip_eval_report_unit_catalog",
        )


class GardenerOnlyBuildTest(unittest.TestCase):
    def test_gardener_only_build_does_not_load_redundancy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unit_ids = ["u0", "u1", "u2", "u3"]
            loaded = (
                unit_ids,
                np.asarray([0.8, 0.4, 0.1, 0.2], dtype=np.float64),
                {
                    "kind": "gardener_official_code_semantics",
                    "data_free": True,
                },
                1234,
            )
            with (
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3.load_similarity",
                    side_effect=AssertionError("redundancy must not be loaded"),
                ) as load_mock,
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3._load_gardener_weight_entropy",
                    return_value=loaded,
                ),
                mock.patch(
                    "analysis.prune_plan.build_prune_plan_v3.render_prune_strategy_drop_grids",
                    return_value=root / "drop_grid.png",
                ),
            ):
                run_build_prune_plan(
                    experiment="vggt",
                    dataset="d1,d2",
                    backbone="vggt",
                    prune_plan_root=root,
                    viz_out_dir=root,
                    strategy_selector="gardener",
                    max_drop_budget=2,
                    skip_param_accounting=True,
                )

            bundle = json.loads(
                (root / PRUNE_PLAN_BUNDLE_FILENAME).read_text(encoding="utf-8")
            )

        load_mock.assert_not_called()
        self.assertEqual(bundle["unit_ids"], unit_ids)
        self.assertEqual(list(bundle["strategies"]), ["gardener"])
        self.assertEqual(
            bundle["strategies"]["gardener"]["points"],
            {"1": "gardener_k1", "2": "gardener_k2"},
        )
        self.assertEqual(bundle["sets"]["gardener_k1"]["drop"], [2])
        self.assertEqual(bundle["sets"]["gardener_k2"]["drop"], [2, 3])
        self.assertEqual(
            bundle["extra"]["unit_catalog_source"]["kind"],
            "pretrained_backbone_droppable_unit_catalog",
        )


if __name__ == "__main__":
    unittest.main()

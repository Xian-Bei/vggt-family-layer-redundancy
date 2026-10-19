from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    canonical_plan_name_for_group,
)
from analysis.prune_plan.eval_plots import (
    _apply_metric_profile_to_point,
    _fit_pruning_normalization_profile,
    _write_normalization_profile,
    _with_dataset_average,
    OURS_CURVE_ZORDER,
    OURS_MARKER_ZORDER,
    PRUNING_METHOD_LABELS,
    RECOVERY_METRIC_GRID,
    STRATEGY_PLOT_COLORS,
    STRATEGY_PLOT_ORDER,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    gt_mean_degradation_series,
    pruning_average_strategy_keys,
    pruning_structure_strategy_keys,
    pose_selection_score,
    render_strategy_metric_figure,
    move_y_axis_label_to_top,
    set_y_axis_max_decimals,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
    strategy_plot_marker_offset,
    strategy_plot_zorder,
    write_gt_degradation_plot,
    write_pruning_eval_plots,
    write_source_pruning_eval_plots,
)
from analysis.prune_plan.build_prune_plan_v3 import (
    _raw_metric_to_upper_degradation_score,
)
from analysis.redundancy.skip_metric_scaling import (
    GT_METRIC_KEYS,
    normalize_metric_components,
)
from analysis.schema.v3 import EvalRecordV3


def _write_plan_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
        "backbone": "vggt",
        "dataset": "scannet_dev50,nuscenes_train",
        "num_units": 4,
        "intervals": {
            "i1_j2": {
                "selector": "i1_j2",
                "i": 1,
                "j": 2,
                "start": 1,
                "end": 2,
                "drop": [1, 2],
                "drop_count": 2,
            }
        },
        "strategies": {
            "CKA": {
                "display_name": "CKA",
                "points": {"2": "i1_j2"},
            },
            "retention": {
                "display_name": "Retention",
                "points": {"2": "i1_j2"},
            },
        },
        "spans": {
            "2": {
                "selector": "i1_j2",
                "i": 1,
                "j": 2,
                "start": 1,
                "end": 2,
                "drop": [1, 2],
                "drop_count": 2,
            }
        },
    }
    (root / "prune_plan_all.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_upper(path: Path, *, ape: float = 1.0, cd: float = 0.1) -> None:
    EvalRecordV3(
        backbone="vggt",
        dataset="scannet_dev50",
        model_name="upper",
        average={"ape": ape, "cd": cd, "chamfer_distance": cd},
    ).save(path / "eval_record_v3.json")


def _write_summary(
    path: Path,
    selector: str = "i1_j2",
    metric_value: float = 2.0,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "eval_interval_v3",
                "kind": "eval_interval_summary",
                "selector": selector,
                "start": 1,
                "end": 2,
                "drop": [1, 2],
                "drop_count": 2,
                "num_scenes": 1,
                "metrics_mean": {
                    "ape": metric_value,
                    "cd": metric_value / 10.0,
                    "chamfer_distance": metric_value / 10.0,
                },
            }
        ),
        encoding="utf-8",
    )


class EvalPlotsTest(unittest.TestCase):
    def test_top_metric_label_preserves_subplot_layout(self) -> None:
        import matplotlib.pyplot as plt

        reference, reference_axis = plt.subplots(figsize=(7.4, 4.8))
        moved, moved_axis = plt.subplots(figsize=(7.4, 4.8))
        try:
            reference_axis.set_ylabel(
                "Comp.↓ (m)", fontsize=18, fontweight="bold"
            )
            move_y_axis_label_to_top(
                moved_axis,
                "Comp.↓ (m)",
                fontsize=18,
            )
            reference.tight_layout()
            moved.tight_layout()
            np.testing.assert_allclose(
                moved_axis.get_position().bounds,
                reference_axis.get_position().bounds,
            )
        finally:
            plt.close(reference)
            plt.close(moved)

    def test_y_axis_formatter_uses_one_precision_with_at_most_two_places(self) -> None:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        try:
            axis.set_yticks([0.375, 0.4, 0.425, 0.5])
            set_y_axis_max_decimals(axis, 2)
            fig.canvas.draw()
            labels = [label.get_text() for label in axis.get_yticklabels()]
            self.assertEqual(labels, ["0.38", "0.40", "0.42", "0.50"])
            self.assertTrue(
                all(
                    len(label.rsplit(".", 1)[1]) == 2
                    for label in labels
                )
            )
        finally:
            plt.close(fig)

    def test_y_axis_formatter_keeps_half_step_ticks_consistent(self) -> None:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        try:
            axis.set_yticks([1.0, 1.5, 2.0, 2.5])
            set_y_axis_max_decimals(axis, 2)
            fig.canvas.draw()
            labels = [label.get_text() for label in axis.get_yticklabels()]
            self.assertEqual(labels, ["1.0", "1.5", "2.0", "2.5"])
        finally:
            plt.close(fig)

    def test_retention_uses_upper_zero_and_all_interval_worst(self) -> None:
        raw = np.asarray([0.5, 1.0, 1.5, 3.0], dtype=np.float64)
        degradation, _meta = _raw_metric_to_upper_degradation_score(
            raw,
            upper_baseline=1.0,
        )
        normalized, aggregate, meta = normalize_metric_components(
            {"ape": degradation}, metric_names=("ape",)
        )
        np.testing.assert_allclose(degradation, [0.0, 0.0, 0.5, 2.0])
        np.testing.assert_allclose(normalized["ape"], [0.0, 0.0, 0.25, 1.0])
        np.testing.assert_allclose(aggregate, [0.0, 0.0, 0.25, 1.0])
        self.assertEqual(
            meta["method"],
            "per_metric_upper_zero_all_interval_worst_then_nanmean",
        )

    def test_normalization_profile_uses_only_stage2_single_intervals(self) -> None:
        metrics_24 = {metric: 2.0 for metric in GT_METRIC_KEYS}
        metrics_26 = {metric: 3.0 for metric in GT_METRIC_KEYS}
        metrics_47 = {metric: 100.0 for metric in GT_METRIC_KEYS}
        metrics_double = {metric: 200.0 for metric in GT_METRIC_KEYS}
        profile = _fit_pruning_normalization_profile(
            summaries={
                "i1_j24": {"drop_count": 24, "metrics_mean": metrics_24},
                "i1_j26": {"drop_count": 26, "metrics_mean": metrics_26},
                "i1_j47": {"drop_count": 47, "metrics_mean": metrics_47},
                "i1_j2_i4_j5": {
                    "drop_count": 4,
                    "metrics_mean": metrics_double,
                },
            },
            upper_average={metric: 1.0 for metric in GT_METRIC_KEYS},
            result_root=Path("/tmp/pruning"),
            max_drop_count=26,
        )
        self.assertEqual(profile["max_drop_budget"], 26)
        self.assertEqual(
            profile["candidate_scope"],
            "stage2_contiguous_intervals_with_drop_count_at_most_max_drop_budget",
        )
        self.assertEqual(profile["candidate_counts"]["ape"], 2)
        self.assertEqual(profile["maximum_brute_force_metric"]["ape"], 3.0)
        self.assertEqual(
            profile["worst_source_by_metric"]["ape"]["selector"],
            "i1_j26",
        )

    def test_pose_selection_score_sums_only_four_pose_metrics(self) -> None:
        metrics = {
            "ape": 1.0, "are": 2.0, "rpe_trans": 3.0, "rpe_rot": 4.0,
            "acc": 1000.0, "comp": 2000.0, "chamfer_distance": 3000.0,
        }
        self.assertEqual(pose_selection_score(metrics), 10.0)

    def test_pruning_collector_uses_all_interval_cap_not_visible_points(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            bundle_path = root / "prune_plan_all.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["num_units"] = 50
            bundle["intervals"] = {
                f"i1_j{drop_count}": {
                    "selector": f"i1_j{drop_count}",
                    "i": 1,
                    "j": drop_count,
                    "start": 1,
                    "end": drop_count,
                    "drop": list(range(1, drop_count + 1)),
                    "drop_count": drop_count,
                }
                for drop_count in (24, 26)
            }
            bundle["strategies"] = {
                "CKA": {
                    "display_name": "CKA",
                    "points": {
                        "24": "i1_j24",
                        "26": "i1_j26",
                    },
                }
            }
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            pruning_root = root / "scannet_dev50" / "eval" / "pruning"
            _write_summary(pruning_root / "i1_j24", "i1_j24", 2.0)
            _write_summary(pruning_root / "i1_j26", "i1_j26", 3.0)
            _write_summary(pruning_root / "i1_j47", "i1_j47", 100.0)

            payload = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                datasets=["scannet_dev50"],
            )

            dataset = payload["datasets"]["scannet_dev50"]
            profile = dataset["normalization_profile"]
            self.assertEqual(profile["max_drop_budget"], 48)
            self.assertEqual(
                profile["maximum_brute_force_metric"]["ape"],
                100.0,
            )
            self.assertEqual(
                profile["worst_source_by_metric"]["ape"]["drop_count"], 47
            )
            self.assertAlmostEqual(
                dataset["strategies"]["CKA"]["points"]["26"]
                ["normalized_metric"]["ape"],
                2.0 / 99.0,
            )

    def test_generalization_reuses_calibration_stage2_span(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            source_root = root / "scannet_dev50" / "eval" / "pruning"
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(source_root / "i1_j24", "i1_j24", 3.0)
            profile = _fit_pruning_normalization_profile(
                summaries={
                    "i1_j24": {
                        "metrics_mean": {"ape": 3.0, "cd": 0.3},
                        "summary_path": str(source_root / "i1_j24" / "summary.json"),
                    }
                },
                upper_average={"ape": 1.0, "cd": 0.1},
                result_root=source_root,
                max_drop_count=24,
            )
            _write_normalization_profile(source_root, profile)

            target_root = root / "scannet_test50" / "eval" / "pruning"
            _write_upper(
                root / "scannet_test50" / "eval" / "upper",
                ape=10.0,
                cd=1.0,
            )
            _write_summary(target_root / "i1_j2", "i1_j2", 12.0)
            payload = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                datasets=["scannet_test50"],
            )
            target = payload["datasets"]["scannet_test50"]
            target_profile = target["normalization_profile"]
            self.assertEqual(
                target_profile["normalization_source_dataset"],
                "scannet_dev50",
            )
            self.assertEqual(target_profile["maximum_brute_force_metric"]["ape"], 12.0)
            self.assertEqual(target["normalized_upper_average"]["ape"], 0.0)
            self.assertEqual(
                target["strategies"]["CKA"]["points"]["2"]
                ["normalized_metric"]["ape"],
                1.0,
            )

    def test_plot_uses_current_and_upper_anchored_normalization(self) -> None:
        point = {
            "metrics_mean": {metric: 0.5 for metric in GT_METRIC_KEYS}
        }
        _apply_metric_profile_to_point(
            point,
            upper_average={metric: 1.0 for metric in GT_METRIC_KEYS},
            profile={
                "maximum_single_interval_metric": {
                    metric: 2.0 for metric in GT_METRIC_KEYS
                }
            },
        )
        self.assertEqual(point["raw_metric"]["ape"], 0.5)
        self.assertEqual(point["normalized_metric"]["ape"], -0.5)

        fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "normalized_upper_average": {"ape": 0.0},
                "normalization_bounds": {"ape": {"min": 0.0, "max": 2.0}},
                "strategies": {
                    "CKA_double": {
                        "display_name": "CKA",
                        "points": {"2": {"available": True, **point}},
                    }
                }
            },
            title="better than upper",
            metric_grid=(("ape",),),
            normalize_metrics=True,
            include_upper_anchor=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        axis = fig.axes[0]
        self.assertEqual(float(axis.lines[0].get_ydata()[0]), 0.0)
        self.assertEqual(float(axis.lines[-1].get_ydata()[0]), -0.5)
        self.assertEqual(axis.get_ylim()[1], 1.0)

        raw_fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "normalization_bounds": {"ape": {"min": 0.0, "max": 2.0}},
                "strategies": {
                    "CKA_double": {
                        "display_name": "CKA",
                        "points": {"2": {"available": True, **point}},
                    }
                },
            },
            title="raw current metric",
            metric_grid=(("ape",),),
            normalize_metrics=False,
            include_upper_anchor=False,
        )
        self.assertIsNotNone(raw_fig)
        assert raw_fig is not None
        raw_axis = raw_fig.axes[0]
        self.assertEqual(float(raw_axis.lines[0].get_ydata()[0]), 1.0)
        self.assertEqual(float(raw_axis.lines[-1].get_ydata()[0]), 0.5)
        self.assertEqual(raw_axis.get_ylim()[1], 2.0)

        uncapped_fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "normalized_upper_average": {"ape": 0.0},
                "normalization_bounds": {"ape": {"min": 0.0, "max": 2.0}},
                "strategies": {
                    "CKA_double": {
                        "display_name": "CKA",
                        "points": {
                            "2": {
                                "available": True,
                                "normalized_metric": {"ape": 1.5},
                            }
                        },
                    }
                },
            },
            title="uncapped normalized metric",
            metric_grid=(("ape",),),
            normalize_metrics=True,
            include_upper_anchor=False,
            cap_y_at_normalization_max=False,
            minimum_y_axis_upper=0.6,
        )
        self.assertIsNotNone(uncapped_fig)
        assert uncapped_fig is not None
        self.assertGreater(uncapped_fig.axes[0].get_ylim()[1], 1.5)

        minimum_upper_fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "normalized_upper_average": {"ape": 0.0},
                "strategies": {
                    "CKA_double": {
                        "display_name": "CKA",
                        "points": {
                            "2": {
                                "available": True,
                                "normalized_metric": {"ape": 0.2},
                            }
                        },
                    }
                },
            },
            title="minimum normalized y-axis upper bound",
            metric_grid=(("ape",),),
            normalize_metrics=True,
            include_upper_anchor=False,
            cap_y_at_normalization_max=False,
            minimum_y_axis_upper=0.6,
        )
        self.assertIsNotNone(minimum_upper_fig)
        assert minimum_upper_fig is not None
        self.assertEqual(minimum_upper_fig.axes[0].get_ylim()[1], 0.6)

        fixed_upper_fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "normalized_upper_average": {"ape": 0.0},
                "strategies": {
                    "CKA_double": {
                        "display_name": "CKA",
                        "points": {
                            "2": {
                                "available": True,
                                "normalized_metric": {"ape": 0.8},
                            }
                        },
                    }
                },
            },
            title="fixed normalized y-axis upper bound",
            metric_grid=(("ape",),),
            normalize_metrics=True,
            include_upper_anchor=False,
            cap_y_at_normalization_max=False,
            fixed_y_axis_upper=0.5,
        )
        self.assertIsNotNone(fixed_upper_fig)
        assert fixed_upper_fig is not None
        self.assertEqual(fixed_upper_fig.axes[0].get_ylim()[1], 0.5)

    def test_dataset_average_preserves_mean_normalization_upper_bounds(self) -> None:
        payload = _with_dataset_average({
            "datasets": {
                "first": {
                    "normalization_bounds": {"ape": {"min": 0.0, "max": 2.0}},
                    "strategies": {},
                },
                "second": {
                    "normalization_bounds": {"ape": {"min": 0.0, "max": 4.0}},
                    "strategies": {},
                },
            }
        })
        self.assertEqual(
            payload["dataset_average"]["normalization_bounds"]["ape"]["max"],
            3.0,
        )

    def test_pruning_method_labels_and_training_plot_groups(self) -> None:
        self.assertEqual(PRUNING_METHOD_LABELS["CKA"], "CKA")
        self.assertEqual(PRUNING_METHOD_LABELS["cosine"], "Cosine")
        self.assertEqual(PRUNING_METHOD_LABELS["inv_mse"], "INV_MSE")
        self.assertEqual(PRUNING_METHOD_LABELS["cca"], "CCA")
        self.assertEqual(PRUNING_METHOD_LABELS["gardener"], "Gardener")

        strategies = (
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
            "shortgpt",
            "gardener",
            "simpler",
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "CKA"),
            (
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
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "CKA_single"),
            (
                "retention_single",
                "CKA_single",
                "cosine_single",
                "inv_mse_single",
                "cca_single",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "CKA_multi"),
            (
                "retention_double",
                "CKA_double",
                "cosine_double",
                "inv_mse_double",
                "cca_double",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "others"),
            (
                "retention_single",
                "retention_double",
                "replaceme_single",
                "shortgpt",
                "gardener",
                "simpler",
            ),
        )
        self.assertEqual(
            pruning_structure_strategy_keys(strategies, "single"),
            (
                "retention_single",
                "CKA_single",
                "cosine_single",
                "inv_mse_single",
                "cca_single",
                "replaceme_single",
                "simpler",
            ),
        )
        self.assertEqual(
            pruning_structure_strategy_keys(strategies, "non_single"),
            (
                "retention_double",
                "CKA_double",
                "shortgpt",
                "gardener",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "single"),
            pruning_structure_strategy_keys(strategies, "single"),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "non-single"),
            pruning_structure_strategy_keys(strategies, "non_single"),
        )

    def test_compact_legends_are_stacked_in_upper_left(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")

        normalized = {key: 0.05 for key in GT_METRIC_KEYS}
        strategies = {
            key: {
                "display_name": key,
                "points": {
                    "2": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                },
            }
            for key in ("CKA_single", "CKA_double", "simpler")
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="legend placement",
            metric_grid=(("gt_mean",),),
            compact_pruning_legend=True,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            ax = fig.axes[0]
            style_legend = ax.get_legend()
            method_legend = next(
                artist
                for artist in ax.artists
                if artist.get_title().get_text() == "Method (color)"
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            method_bbox = method_legend.get_window_extent(renderer=renderer)
            style_bbox = style_legend.get_window_extent(renderer=renderer)
            self.assertAlmostEqual(method_bbox.x0, style_bbox.x0, delta=1.0)
            self.assertLessEqual(style_bbox.y1, method_bbox.y0)
            self.assertEqual(
                [text.get_text() for text in style_legend.get_texts()],
                ["Single", "Multi"],
            )
            self.assertAlmostEqual(fig.get_size_inches()[0], 7.4)
        finally:
            plt.close(fig)

    def test_collects_reversed_two_interval_summary_and_requires_complete_plot(
        self,
    ) -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            bundle_path = root / "prune_plan_all.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["sets"] = {
                "CKA_k2": {
                    "selector": "CKA_k2",
                    "selector_kind": "set",
                    "drop": [1, 3],
                    "drop_count": 2,
                    "num_intervals": 2,
                }
            }
            bundle["strategies"]["CKA"]["points"] = {"2": "CKA_k2"}
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(
                root
                / "scannet_dev50"
                / "eval"
                / "pruning"
                / "i1_j1",
                selector="i1_j1",
            )
            _write_summary(
                root
                / "scannet_dev50"
                / "eval"
                / "pruning"
                / "i3_j3_i1_j1",
                selector="i3_j3_i1_j1",
            )
            payload = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                strategies={"CKA": bundle["strategies"]["CKA"]},
                datasets=["scannet_dev50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            point = payload["datasets"]["scannet_dev50"]["strategies"]["CKA"][
                "points"
            ]["2"]
            self.assertTrue(point["available"])
            self.assertEqual(point["selector"], "i1_j1_i3_j3")
            paths = write_pruning_eval_plots(
                experiment_root=root,
                backbone="vggt",
                payload=payload,
                write_dataset_plots=True,
            )
            self.assertEqual(
                paths[0].name, "scannet_dev50_CKA_multi_pruning_drop.png"
            )

            point["available"] = False
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                write_pruning_eval_plots(
                    experiment_root=root,
                    backbone="vggt",
                    payload=payload,
                    write_dataset_plots=False,
                    average_filename="strict.png",
                )

    def test_held_out_payload_includes_training_selected_gt_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            _write_upper(root / "held_out" / "eval" / "upper")
            _write_summary(root / "held_out" / "eval" / "pruning" / "i1_j2")
            payload = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                datasets=["held_out"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            strategies = payload["datasets"]["held_out"]["strategies"]
            self.assertIn("CKA", strategies)
            self.assertIn("retention", strategies)
            self.assertTrue(strategies["retention"]["points"]["2"]["available"])
            average_strategies = payload["dataset_average"]["strategies"]
            self.assertIn("CKA", average_strategies)
            self.assertIn("retention", average_strategies)

    def test_single_variants_use_dashed_plot_lines(self) -> None:
        self.assertEqual(strategy_plot_linestyle("retention"), "-")
        self.assertEqual(strategy_plot_linestyle("CKA_double"), "-")
        self.assertEqual(strategy_plot_linestyle("replaceme_double"), "-")
        self.assertEqual(strategy_plot_linestyle("recovery_replaceme_ls"), "-")
        self.assertEqual(strategy_plot_linestyle("recovery_ghost"), "-")
        self.assertEqual(strategy_plot_linestyle("no_recovery"), "--")
        self.assertEqual(strategy_plot_linestyle("no_recovery_single"), "--")
        self.assertEqual(strategy_plot_linestyle("no_recovery_double"), "-")
        self.assertEqual(
            STRATEGY_PLOT_COLORS["no_recovery_single"],
            STRATEGY_PLOT_COLORS["no_recovery_double"],
        )
        self.assertEqual(
            len(
                {
                    STRATEGY_PLOT_COLORS[key]
                    for key in (
                        "recovery_ghost",
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                        "recovery_ghost_interval_sequential_autogroup",
                    )
                }
            ),
            4,
        )
        self.assertEqual(
            STRATEGY_PLOT_COLORS[
                "recovery_ghost_interval_sequential_autogroup"
            ],
            "#d62728",
        )
        self.assertNotEqual(
            STRATEGY_PLOT_COLORS["recovery_ghost_interval_sequential"],
            "#d62728",
        )
        self.assertEqual(strategy_plot_linestyle("retention_single"), "--")
        self.assertEqual(strategy_plot_linestyle("CKA_single"), "--")
        self.assertEqual(strategy_plot_linestyle("replaceme_single"), "--")
        self.assertEqual(strategy_plot_linestyle("shortgpt"), "-")
        self.assertEqual(strategy_plot_linestyle("gardener"), "-")
        self.assertEqual(strategy_plot_linestyle("simpler"), "--")

    def test_pruning_structure_controls_dash_marker_offset_and_layer(self) -> None:
        self.assertEqual(strategy_plot_dashes("CKA_single"), (5.0, 2.0))
        self.assertIsNone(strategy_plot_dashes("CKA_double"))
        self.assertEqual(strategy_plot_dashes("simpler"), (5.0, 2.0))

        self.assertEqual(strategy_plot_marker("CKA_single"), "o")
        self.assertEqual(strategy_plot_marker("CKA_double"), "s")
        self.assertEqual(strategy_plot_marker("shortgpt"), "s")
        self.assertEqual(strategy_plot_marker("gardener"), "s")
        self.assertEqual(strategy_plot_marker("simpler"), "o")

        self.assertEqual(strategy_plot_marker_offset("CKA_single"), -0.24)
        self.assertEqual(strategy_plot_marker_offset("CKA_double"), 0.24)
        self.assertEqual(strategy_plot_marker_offset("shortgpt"), 0.24)
        self.assertEqual(strategy_plot_marker_offset("gardener"), 0.24)
        self.assertEqual(strategy_plot_marker_offset("simpler"), -0.24)
        self.assertGreater(
            strategy_plot_zorder("CKA_single"),
            strategy_plot_zorder("CKA_double"),
        )
        self.assertEqual(
            strategy_plot_zorder(
                "recovery_ghost_interval_sequential_autogroup"
            ),
            OURS_CURVE_ZORDER,
        )

    def test_render_dodges_markers_but_keeps_lines_at_true_x_positions(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")

        normalized = {key: 0.25 for key in GT_METRIC_KEYS}
        strategies = {
            key: {
                "display_name": key,
                "points": {
                    "2": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                },
            }
            for key in ("CKA_single", "CKA_double")
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="overlap styles",
            metric_grid=(("gt_mean",),),
            draw_solid_before_dashed=True,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            ax = fig.axes[0]
            lines = {line.get_label(): line for line in ax.lines}
            plotted_labels = [
                line.get_label()
                for line in ax.lines
                if line.get_label() in {"CKA_single", "CKA_double"}
            ]
            self.assertEqual(plotted_labels, ["CKA_double", "CKA_single"])
            self.assertEqual(list(lines["CKA_single"].get_xdata()), [0, 2])
            self.assertEqual(list(lines["CKA_double"].get_xdata()), [0, 2])
            self.assertGreater(
                lines["CKA_single"].get_zorder(),
                lines["CKA_double"].get_zorder(),
            )

            marker_offsets = [
                [tuple(float(value) for value in xy) for xy in collection.get_offsets()]
                for collection in ax.collections
            ]
            self.assertIn([(0.0, 0.0), (1.76, 0.25)], marker_offsets)
            self.assertIn([(0.0, 0.0), (2.24, 0.25)], marker_offsets)
        finally:
            plt.close(fig)

    def test_render_can_keep_markers_at_true_x_positions(self) -> None:
        import matplotlib.pyplot as plt

        normalized = {key: 0.25 for key in GT_METRIC_KEYS}
        strategies = {
            key: {
                "display_name": key,
                "points": {
                    "2": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                },
            }
            for key in ("CKA_single", "CKA_double")
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="aligned markers",
            metric_grid=(("gt_mean",),),
            dodge_structure_markers=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            marker_offsets = [
                [tuple(float(value) for value in xy) for xy in collection.get_offsets()]
                for collection in fig.axes[0].collections
            ]
            self.assertEqual(marker_offsets.count([(0.0, 0.0), (2.0, 0.25)]), 2)
        finally:
            plt.close(fig)

    def test_render_can_anchor_upper_near_axis_bottom_and_clip_low_outlier(self) -> None:
        import matplotlib.pyplot as plt

        fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"acc": 1.0},
                "strategies": {
                    "simpler": {
                        "display_name": "SIMPLER",
                        "points": {
                            "2": {
                                "available": True,
                                "raw_metric": {"acc": 0.1},
                            },
                            "4": {
                                "available": True,
                                "raw_metric": {"acc": 2.0},
                            },
                        },
                    }
                },
            },
            title="upper anchored",
            metric_grid=(("acc",),),
            normalize_metrics=False,
            cap_y_at_normalization_max=False,
            upper_anchor_bottom_margin_fraction=0.10,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            bottom, top = fig.axes[0].get_ylim()
            self.assertGreater(bottom, 0.1)
            self.assertAlmostEqual(
                (1.0 - bottom) / (top - bottom),
                1.0 / 11.0,
            )
        finally:
            plt.close(fig)

    def test_render_forces_ours_curve_and_markers_to_top_layer(self) -> None:
        import matplotlib.pyplot as plt

        normalized = {key: 0.25 for key in GT_METRIC_KEYS}
        strategies = {
            "comparison": {
                "display_name": "Comparison",
                "points": {
                    "2": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                },
            },
            "candidate": {
                "display_name": "Ours · Final",
                "points": {
                    "2": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                },
            },
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="ours layering",
            metric_grid=(("gt_mean",),),
            strategy_style_overrides={
                "candidate": {"zorder": -1.0, "marker_zorder": -1.0}
            },
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            lines = {line.get_label(): line for line in fig.axes[0].lines}
            self.assertEqual(
                lines["Ours · Final"].get_zorder(), OURS_CURVE_ZORDER
            )
            self.assertEqual(
                max(collection.get_zorder() for collection in fig.axes[0].collections),
                OURS_MARKER_ZORDER,
            )
        finally:
            plt.close(fig)

    def test_pruning_plot_contains_single_and_double_versions_and_only_mean_panel(
        self,
    ) -> None:
        try:
            import matplotlib.image as mpimg
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            bundle_path = root / "prune_plan_all.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["sets"] = {
                "retention_single_k2": {
                    "selector": "retention_single_k2",
                    "selector_kind": "set",
                    "drop": [1, 2],
                    "drop_count": 2,
                    "num_intervals": 1,
                },
                "retention_double_k2": {
                    "selector": "retention_double_k2",
                    "selector_kind": "set",
                    "drop": [1, 3],
                    "drop_count": 2,
                    "num_intervals": 2,
                },
            }
            bundle["strategies"] = {
                "retention_single": {
                    "display_name": "Retention Single",
                    "points": {"2": "retention_single_k2"},
                },
                "retention_double": {
                    "display_name": "Retention Double",
                    "points": {"2": "retention_double_k2"},
                },
            }
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            pruning_root = root / "scannet_dev50" / "eval" / "pruning"
            _write_summary(pruning_root / "i1_j1_i3_j3", "i1_j1_i3_j3")
            _write_summary(pruning_root / "i1_j2", "i1_j2")

            payload = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                strategies=bundle["strategies"],
                datasets=["scannet_dev50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            strategies = payload["datasets"]["scannet_dev50"]["strategies"]
            self.assertEqual(
                strategies["retention_single"]["points"]["2"]["selector"], "i1_j2"
            )
            self.assertEqual(
                strategies["retention_double"]["points"]["2"]["selector"],
                "i1_j1_i3_j3",
            )

            paths = write_pruning_eval_plots(
                experiment_root=root,
                backbone="vggt",
                payload=payload,
                write_dataset_plots=True,
            )
            self.assertEqual(
                {path.name for path in paths},
                {
                    "scannet_dev50_CKA_single_pruning_drop.png",
                    "scannet_dev50_CKA_multi_pruning_drop.png",
                    "scannet_dev50_CKA_pruning_drop.png",
                    "scannet_dev50_others_pruning_drop.png",
                },
            )
            image = mpimg.imread(paths[0])
            self.assertLess(image.shape[1], image.shape[0] * 2)

    def test_prune_span_canonicalizes_to_interval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_plan_bundle(root)
            self.assertEqual(canonical_plan_name_for_group(root, "prune_span2"), "i1_j2")

    def test_collectors_read_pruning_and_recovery_interval_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(root / "scannet_dev50" / "eval" / "pruning" / "i1_j2")
            _write_summary(root / "scannet_dev50" / "eval" / "recovery" / "recovery_linearpatch" / "i1_j2")
            _write_summary(
                root
                / "scannet_dev50"
                / "eval"
                / "recovery"
                / "recovery_linearpatch"
                / "i2_j3",
                selector="i2_j3",
            )
            _write_summary(root / "scannet_dev50" / "eval" / "recovery" / "recovery_diag" / "i1_j2")
            _write_summary(root / "scannet_dev50" / "eval" / "recovery" / "recovery_rotate" / "i1_j2")
            _write_upper(root / "7scenes_test" / "eval" / "upper")
            _write_summary(root / "7scenes_test" / "eval" / "pruning" / "i1_j2")
            _write_summary(root / "7scenes_test" / "eval" / "recovery" / "recovery_linearpatch" / "i1_j2")

            pruning = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
            )
            recovery = collect_recovery_eval_payload(experiment_root=root)
            pruning["datasets"]["scannet_dev50"]["strategies"]["CKA"]["points"]["3"] = {
                **pruning["datasets"]["scannet_dev50"]["strategies"]["CKA"]["points"]["2"],
                "drop_count": 3,
            }
            recovery = add_recovery_comparison_strategies(recovery, pruning)
            self.assertIn("scannet_dev50", pruning["datasets"])
            self.assertIn("scannet_dev50", recovery["datasets"])
            self.assertIn("CKA", pruning["datasets"]["scannet_dev50"]["strategies"])
            self.assertIn("recovery_linearpatch", recovery["datasets"]["scannet_dev50"]["strategies"])
            self.assertEqual(
                recovery["datasets"]["scannet_dev50"]["strategies"]
                ["recovery_linearpatch"]["points"]["2"]["selector"],
                "i1_j2",
            )
            self.assertNotIn("recovery_diag", recovery["datasets"]["scannet_dev50"]["strategies"])
            self.assertNotIn("recovery_rotate", recovery["datasets"]["scannet_dev50"]["strategies"])
            self.assertIn("gt_drop", recovery["datasets"]["scannet_dev50"]["strategies"])
            self.assertIn("no_recovery", recovery["datasets"]["scannet_dev50"]["strategies"])
            self.assertEqual(
                recovery["datasets"]["scannet_dev50"]["strategies"]["no_recovery"]["display_name"],
                "No recovery · CKA",
            )
            self.assertNotIn("gt_drop", recovery["datasets"]["7scenes_test"]["strategies"])
            self.assertIn("no_recovery", recovery["datasets"]["7scenes_test"]["strategies"])
            self.assertNotIn("gt_drop", recovery["dataset_average"]["strategies"])

            self.assertEqual(
                set(recovery["datasets"]["scannet_dev50"]["strategies"]["gt_drop"]["points"]),
                {"2"},
            )
            self.assertEqual(
                set(recovery["datasets"]["scannet_dev50"]["strategies"]["no_recovery"]["points"]),
                {"2", "3"},
            )
            self.assertEqual(
                set(recovery["dataset_average"]["strategies"]["no_recovery"]["points"]),
                {
                    drop_key
                    for drop_key in pruning["dataset_average"]["strategies"]["CKA"]["points"]
                    if int(drop_key) % 2 == 0
                },
            )
            pruning_average_point = pruning["dataset_average"]["strategies"]["CKA"]["points"]["2"]
            self.assertEqual(pruning_average_point["relative_to_upper"]["ape"], 2.0)
            self.assertEqual(pruning_average_point["relative_to_upper"]["chamfer_distance"], 2.0)
            self.assertEqual(pruning_average_point["normalized_relative_to_upper"]["ape"], 1.0)
            self.assertEqual(
                gt_mean_degradation_series(pruning),
                [(0, 0.0), (2, 1.0)],
            )
            self.assertEqual(
                STRATEGY_PLOT_ORDER,
                (
                    "no_recovery_single",
                    "no_recovery_double",
                    "no_recovery",
                    "gt_drop",
                    "recovery_linearpatch",
                    "recovery_ghost",
                    "recovery_ghost_interval",
                    "recovery_ghost_interval_sequential",
                    "recovery_ghost_interval_sequential_grouped",
                    "recovery_ghost_interval_sequential_autogroup",
                    "recovery_replaceme_cos",
                    "recovery_replaceme_ls",
                    "recovery_streamline_ffn",
                    "recovery_lora",
                    "recovery_train",
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
                ),
            )
            pruning_versions = (
                "retention_single",
                "retention_double",
                "CKA_single",
                "CKA_double",
                "replaceme_single",
                "replaceme_double",
                "shortgpt",
                "gardener",
                "simpler",
            )
            self.assertEqual(
                len(
                    {
                        (
                            STRATEGY_PLOT_COLORS[key],
                            strategy_plot_linestyle(key),
                        )
                        for key in pruning_versions
                    }
                ),
                len(pruning_versions),
            )
            self.assertEqual(RECOVERY_METRIC_GRID[-1][-1], "gt_mean")
            linearpatch_point = recovery["datasets"]["scannet_dev50"]["strategies"][
                "recovery_linearpatch"
            ]["points"]["2"]
            self.assertEqual(
                set(linearpatch_point["normalized_relative_to_upper"]),
                set(GT_METRIC_KEYS) & set(linearpatch_point["relative_to_upper"]),
            )

    def test_recovery_comparison_adds_explicit_single_and_double_baselines(self) -> None:
        metric_values = {key: 2.0 for key in GT_METRIC_KEYS}
        relative_values = {key: 2.0 for key in GT_METRIC_KEYS}
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {key: 1.0 for key in GT_METRIC_KEYS},
                    "strategies": {
                        "recovery_ghost": {
                            "display_name": "ghost",
                            "points": {
                                "2": {
                                    "available": True,
                                    "drop_count": 2,
                                    "metrics_mean": {key: 1.5 for key in GT_METRIC_KEYS},
                                    "relative_to_upper": {
                                        key: 1.5 for key in GT_METRIC_KEYS
                                    },
                                }
                            },
                        }
                    },
                }
            }
        }
        pruning = {
            "gt_route_datasets": [],
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {key: 1.0 for key in GT_METRIC_KEYS},
                    "normalization_profile": {
                        "maximum_single_interval_metric": {
                            key: 2.0 for key in GT_METRIC_KEYS
                        }
                    },
                    "strategies": {
                        name: {
                            "display_name": name,
                            "points": {
                                "2": {
                                    "available": True,
                                    "drop_count": 2,
                                    "metrics_mean": metric_values,
                                    "relative_to_upper": relative_values,
                                }
                            },
                        }
                        for name in ("CKA_single", "CKA_double")
                    }
                }
            },
            # Deliberately incompatible values.  The recovery mean must be
            # rebuilt from the jointly-normalized per-dataset payload instead.
            "dataset_average": {
                "strategies": {
                    "CKA_double": {
                        "points": {
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "normalized_relative_to_upper": {
                                    key: 0.0 for key in GT_METRIC_KEYS
                                },
                            }
                        }
                    }
                }
            },
        }

        merged = add_recovery_comparison_strategies(recovery, pruning)
        dataset_strategies = merged["datasets"]["scannet_dev50"]["strategies"]
        self.assertEqual(
            dataset_strategies["no_recovery_single"]["display_name"],
            "No recovery · CKA single",
        )
        self.assertEqual(
            dataset_strategies["no_recovery_double"]["display_name"],
            "No recovery · CKA double",
        )
        average_strategies = merged["dataset_average"]["strategies"]
        recovery_point = average_strategies["recovery_ghost"]["points"]["2"]
        no_recovery_point = average_strategies["no_recovery_double"]["points"]["2"]
        self.assertTrue(
            all(
                no_recovery_point["normalized_relative_to_upper"][key]
                > recovery_point["normalized_relative_to_upper"][key]
                for key in GT_METRIC_KEYS
            )
        )

    def test_recovery_collector_shares_selector_across_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            recovery_root = root / "scannet_dev50" / "eval" / "recovery"
            _write_summary(recovery_root / "i1_j2" / "recovery_linearpatch")

            cka = collect_recovery_eval_payload(
                experiment_root=root,
                prune_strategy="CKA",
            )
            strategies = cka["datasets"]["scannet_dev50"]["strategies"]
            self.assertIn("recovery_linearpatch", strategies)
            self.assertEqual(cka["prune_strategy"], "CKA")
            retention = collect_recovery_eval_payload(
                experiment_root=root,
                prune_strategy="retention",
            )
            self.assertIn(
                "recovery_linearpatch",
                retention["datasets"]["scannet_dev50"]["strategies"],
            )

    def test_render_strategy_metric_figure_accepts_larger_legend_font(self) -> None:
        strategies = {
            "recovery_ghost": {
                "display_name": "Ghost",
                "points": {
                    "2": {
                        "available": True,
                        "drop_count": 2,
                        "metrics_mean": {"ape": 2.0},
                    }
                },
            }
        }
        fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "strategies": strategies,
            },
            title="legend size",
            metric_grid=(("ape",),),
            legend_fontsize=11,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        legend = fig.axes[0].get_legend()
        self.assertIsNotNone(legend)
        assert legend is not None
        self.assertEqual(legend.get_texts()[0].get_fontsize(), 11.0)

    def test_fixed_layout_keeps_actual_axes_geometry_uniform(self) -> None:
        import matplotlib.pyplot as plt

        strategy = {
            "display_name": "CKA",
            "points": {
                "2": {
                    "available": True,
                    "drop_count": 2,
                    "metrics_mean": {"ape": 2.0, "are": 20.0},
                }
            },
        }
        figures = [
            render_strategy_metric_figure(
                payload={
                    "upper_average": {"ape": 1.0, "are": 10.0},
                    "strategies": {"CKA_single": strategy},
                },
                title="",
                metric_grid=((metric,),),
                strategy_keys=("CKA_single",),
                show_figure_title=False,
                show_metric_titles=False,
                metric_labels_as_ylabels=show_ylabel,
                metric_ylabel="Normalized Degradation",
                shared_xlabel=None,
                subplot_box_aspect=0.704,
                fixed_layout_margins=(0.10, 0.09, 0.99, 0.99),
            )
            for metric, show_ylabel in (("ape", True), ("are", False))
        ]
        self.assertTrue(all(figure is not None for figure in figures))
        first, second = figures
        assert first is not None and second is not None
        self.assertTrue(
            np.allclose(
                first.axes[0].get_position().bounds,
                second.axes[0].get_position().bounds,
            )
        )
        plt.close(first)
        plt.close(second)

    def test_compact_legend_and_overview_axis_can_be_publication_styled(self) -> None:
        import matplotlib.pyplot as plt

        normalized = {key: 0.25 for key in GT_METRIC_KEYS}
        strategies = {
            key: {
                "display_name": key,
                "points": {
                    "2": {
                        "available": True,
                        "drop_count": 2,
                        "normalized_metric": normalized,
                    }
                },
            }
            for key in (
                "retention_single",
                "CKA_single",
                "inv_mse_single",
                "cca_single",
            )
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="overview styling",
            metric_grid=(("gt_mean",),),
            strategy_keys=tuple(strategies),
            compact_pruning_legend=True,
            compact_method_labels={
                "retention": "Retention",
                "CKA": "CKA",
                "inv_mse": "Inv_MSE",
                "cca": "CCA",
            },
            compact_method_order=("retention", "CKA", "cca", "inv_mse"),
            compact_structure_labels=("Multi",),
            combine_compact_legend=True,
            legend_fontsize=15,
            legend_fontweight="bold",
            normalize_metrics=True,
            include_upper_anchor=False,
            show_metric_titles=False,
            shared_ylabel="Normalized Degradation",
            y_tick_percentage=True,
            figure_height_per_row=5.2,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            ax = fig.axes[0]
            legends = [legend for legend in [*ax.artists, ax.get_legend()] if legend]
            self.assertEqual(len(legends), 1)
            self.assertEqual(legends[0].get_title().get_text(), "")
            self.assertEqual(
                [text.get_text() for text in legends[0].get_texts()],
                ["Retention", "CKA", "CCA", "Inv_MSE"],
            )
            self.assertTrue(
                all(
                    text.get_fontweight() == "bold"
                    for legend in legends
                    for text in legend.get_texts()
                )
            )
            self.assertEqual(ax.get_title(), "")
            self.assertIn(
                "Normalized Degradation",
                [text.get_text() for text in fig.texts],
            )
            fig.canvas.draw()
            self.assertTrue(
                all(label.get_text().endswith("%") for label in ax.get_yticklabels())
            )
            self.assertAlmostEqual(fig.get_size_inches()[1], 5.2)
        finally:
            plt.close(fig)

    def test_render_strategy_metric_figure_can_delegate_x_label_to_gallery(self) -> None:
        fig = render_strategy_metric_figure(
            payload={
                "upper_average": {"ape": 1.0},
                "strategies": {
                    "recovery_ghost": {
                        "display_name": "Ghost",
                        "points": {
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "metrics_mean": {"ape": 2.0},
                            }
                        },
                    }
                },
            },
            title="shared gallery axis",
            metric_grid=(("ape",),),
            shared_xlabel=None,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(fig.axes[0].get_xlabel(), "")
        self.assertNotIn("Dropped Layers", [text.get_text() for text in fig.texts])

    def test_metric_mean_can_stay_normalized_when_individual_metrics_are_raw(self) -> None:
        absolute = {key: 10.0 for key in GT_METRIC_KEYS}
        normalized = {key: 0.25 for key in GT_METRIC_KEYS}
        absolute["ape"] = 4.0
        normalized["ape"] = 0.4
        fig = render_strategy_metric_figure(
            payload={
                "strategies": {
                    "recovery_ghost": {
                        "display_name": "Ghost",
                        "points": {
                            "2": {
                                "available": True,
                                "raw_metric": absolute,
                                "normalized_metric": normalized,
                            }
                        },
                    }
                }
            },
            title="mixed normalization",
            metric_grid=(("ape", "gt_mean"),),
            normalize_metrics=False,
            always_normalize_metric_mean=True,
            include_upper_anchor=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertAlmostEqual(float(fig.axes[0].lines[-1].get_ydata()[0]), 4.0)
        self.assertAlmostEqual(
            float(fig.axes[1].lines[-1].get_ydata()[0]),
            (0.4 + 6 * 0.25) / 7,
        )
        self.assertNotIn("raw metric", fig.axes[0].get_title())
        self.assertNotIn("normalized metric", fig.axes[1].get_title())
        self.assertIn("Individual metrics: raw", fig._suptitle.get_text())
        self.assertIn("7-metric mean: normalized", fig._suptitle.get_text())

    def test_pruning_collector_accepts_in_memory_bundle_before_plan_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(root / "scannet_dev50" / "eval" / "pruning" / "i1_j2")
            strategies = {
                "CKA": {
                    "display_name": "CKA",
                    "points": {"2": "i1_j2"},
                }
            }

            pruning = collect_pruning_eval_payload(
                experiment_root=root,
                prune_plan_root=root,
                strategies=strategies,
                datasets=["scannet_dev50"],
                bundle={"dataset": "scannet_dev50"},
            )

            self.assertFalse((root / "prune_plan_all.json").exists())
            self.assertEqual(pruning["gt_route_datasets"], ["scannet_dev50"])
            self.assertIn("scannet_dev50", pruning["datasets"])
            self.assertIn("CKA", pruning["datasets"]["scannet_dev50"]["strategies"])

    def test_pruning_plot_outputs_use_eval_directory(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_plan_bundle(root)
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(root / "scannet_dev50" / "eval" / "pruning" / "i1_j2")
            _write_summary(root / "scannet_dev50" / "eval" / "recovery" / "recovery_linearpatch" / "i1_j2")
            _write_upper(root / "7scenes_test" / "eval" / "upper")
            _write_summary(root / "7scenes_test" / "eval" / "pruning" / "i1_j2")
            _write_summary(root / "7scenes_test" / "eval" / "recovery" / "recovery_linearpatch" / "i1_j2")
            pruning = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)

            pruning_paths = write_source_pruning_eval_plots(
                experiment_root=root,
                backbone="vggt",
                payload=pruning,
            )
            gt_path = write_gt_degradation_plot(
                experiment_root=root,
                backbone="vggt",
                payload=pruning,
            )
            self.assertIsNotNone(gt_path)
            self.assertEqual(gt_path.name, "prune_plan_vggt_gt_degradation.png")
            self.assertTrue(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "scannet_dev50_CKA_multi_pruning_drop.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "scannet_dev50_others_pruning_drop.png"
                ).is_file()
            )
            self.assertFalse(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "scannet_dev50_pruning_drop.png"
                ).exists()
            )
            self.assertTrue(
                any(p.name == "prune_plan_vggt_pruning_drop.png" for p in pruning_paths)
            )
            self.assertFalse(
                (
                    root
                    / "7scenes_test"
                    / "eval"
                    / "7scenes_test_CKA_multi_pruning_drop.png"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()

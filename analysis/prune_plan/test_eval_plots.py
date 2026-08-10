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
    _with_dataset_average,
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
    render_strategy_metric_figure,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
    strategy_plot_marker_offset,
    strategy_plot_zorder,
    write_gt_degradation_plot,
    write_recovery_eval_plots,
    write_pruning_eval_plots,
    write_source_pruning_eval_plots,
)
from analysis.prune_plan.build_prune_plan_v3 import _raw_metric_to_upper_clamped_score
from analysis.redundancy.skip_metric_scaling import (
    GT_METRIC_KEYS,
    minmax_normalize_finite,
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
            "ours": {
                "display_name": "Ours",
                "points": {"2": "i1_j2"},
            },
            "gt": {
                "display_name": "GT",
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


def _write_upper(path: Path) -> None:
    EvalRecordV3(
        backbone="vggt",
        dataset="scannet_dev50",
        model_name="upper",
        average={"ape": 1.0, "cd": 0.1, "chamfer_distance": 0.1},
    ).save(path / "eval_record_v3.json")


def _write_summary(path: Path, selector: str = "i1_j2") -> None:
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
                "metrics_mean": {"ape": 2.0, "cd": 0.2, "chamfer_distance": 0.2},
            }
        ),
        encoding="utf-8",
    )


class EvalPlotsTest(unittest.TestCase):
    def test_raw_upper_clamp_preserves_previous_minmax_result(self) -> None:
        raw = np.asarray([0.5, 1.0, 1.5, 3.0], dtype=np.float64)
        clamped, _meta = _raw_metric_to_upper_clamped_score(
            raw,
            upper_baseline=1.0,
        )
        previous = np.maximum(raw / 1.0, 1.0)
        np.testing.assert_allclose(
            minmax_normalize_finite(clamped)[0],
            minmax_normalize_finite(previous)[0],
        )

    def test_normalization_profile_uses_single_intervals_up_to_24_drops(self) -> None:
        metrics_24 = {metric: 2.0 for metric in GT_METRIC_KEYS}
        metrics_47 = {metric: 100.0 for metric in GT_METRIC_KEYS}
        metrics_double = {metric: 200.0 for metric in GT_METRIC_KEYS}
        profile = _fit_pruning_normalization_profile(
            summaries={
                "i1_j24": {"drop_count": 24, "metrics_mean": metrics_24},
                "i1_j47": {"drop_count": 47, "metrics_mean": metrics_47},
                "i1_j2_i4_j5": {
                    "drop_count": 4,
                    "metrics_mean": metrics_double,
                },
            },
            upper_average={metric: 1.0 for metric in GT_METRIC_KEYS},
            result_root=Path("/tmp/pruning"),
        )
        self.assertEqual(profile["max_drop_budget"], 24)
        self.assertEqual(
            profile["candidate_scope"],
            "single_intervals_with_drop_count_at_most_24",
        )
        self.assertEqual(profile["candidate_counts"]["ape"], 1)
        self.assertEqual(profile["maximum_single_interval_metric"]["ape"], 2.0)

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
                    "ours_double": {
                        "display_name": "Ours",
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
                    "ours_double": {
                        "display_name": "Ours",
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
        self.assertEqual(PRUNING_METHOD_LABELS["ours"], "Ours_CKA")
        self.assertEqual(PRUNING_METHOD_LABELS["cosine"], "Ours_Cosine")
        self.assertEqual(PRUNING_METHOD_LABELS["inv_mse"], "Ours_INV_MSE")
        self.assertEqual(PRUNING_METHOD_LABELS["cca"], "Ours_CCA")

        strategies = (
            "gt_single",
            "gt_double",
            "ours_single",
            "ours_double",
            "cosine_single",
            "cosine_double",
            "inv_mse_single",
            "inv_mse_double",
            "cca_single",
            "cca_double",
            "replaceme_single",
            "shortgpt",
            "simpler",
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "ours"),
            (
                "gt_single",
                "gt_double",
                "ours_single",
                "ours_double",
                "cosine_single",
                "cosine_double",
                "inv_mse_single",
                "inv_mse_double",
                "cca_single",
                "cca_double",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "ours_single"),
            (
                "gt_single",
                "ours_single",
                "cosine_single",
                "inv_mse_single",
                "cca_single",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "ours_multi"),
            (
                "gt_double",
                "ours_double",
                "cosine_double",
                "inv_mse_double",
                "cca_double",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "others"),
            (
                "gt_single",
                "gt_double",
                "ours_single",
                "ours_double",
                "replaceme_single",
                "shortgpt",
                "simpler",
            ),
        )
        self.assertEqual(
            pruning_structure_strategy_keys(strategies, "single"),
            (
                "gt_single",
                "ours_single",
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
                "gt_double",
                "ours_double",
                "shortgpt",
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
            for key in ("ours_single", "ours_double", "simpler")
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
                "ours_k2": {
                    "selector": "ours_k2",
                    "selector_kind": "set",
                    "drop": [1, 3],
                    "drop_count": 2,
                    "num_intervals": 2,
                }
            }
            bundle["strategies"]["ours"]["points"] = {"2": "ours_k2"}
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
                strategies={"ours": bundle["strategies"]["ours"]},
                datasets=["scannet_dev50"],
                min_drop=2,
                max_drop=2,
                drop_step=2,
            )
            point = payload["datasets"]["scannet_dev50"]["strategies"]["ours"][
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
                paths[0].name, "scannet_dev50_ours_multi_pruning_drop.png"
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
            self.assertIn("ours", strategies)
            self.assertIn("gt", strategies)
            self.assertTrue(strategies["gt"]["points"]["2"]["available"])
            average_strategies = payload["dataset_average"]["strategies"]
            self.assertIn("ours", average_strategies)
            self.assertIn("gt", average_strategies)

    def test_single_variants_use_dashed_plot_lines(self) -> None:
        self.assertEqual(strategy_plot_linestyle("gt"), "-")
        self.assertEqual(strategy_plot_linestyle("ours_double"), "-")
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
                        "recovery_ghost_sequential",
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                    )
                }
            ),
            4,
        )
        self.assertEqual(strategy_plot_linestyle("gt_single"), "--")
        self.assertEqual(strategy_plot_linestyle("ours_single"), "--")
        self.assertEqual(strategy_plot_linestyle("replaceme_single"), "--")
        self.assertEqual(strategy_plot_linestyle("shortgpt"), "-")
        self.assertEqual(strategy_plot_linestyle("simpler"), "--")

    def test_pruning_structure_controls_dash_marker_offset_and_layer(self) -> None:
        self.assertEqual(strategy_plot_dashes("ours_single"), (5.0, 2.0))
        self.assertIsNone(strategy_plot_dashes("ours_double"))
        self.assertEqual(strategy_plot_dashes("simpler"), (5.0, 2.0))

        self.assertEqual(strategy_plot_marker("ours_single"), "o")
        self.assertEqual(strategy_plot_marker("ours_double"), "s")
        self.assertEqual(strategy_plot_marker("shortgpt"), "s")
        self.assertEqual(strategy_plot_marker("simpler"), "o")

        self.assertEqual(strategy_plot_marker_offset("ours_single"), -0.12)
        self.assertEqual(strategy_plot_marker_offset("ours_double"), 0.12)
        self.assertEqual(strategy_plot_marker_offset("shortgpt"), 0.12)
        self.assertEqual(strategy_plot_marker_offset("simpler"), -0.12)
        self.assertGreater(
            strategy_plot_zorder("ours_single"),
            strategy_plot_zorder("ours_double"),
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
            for key in ("ours_single", "ours_double")
        }
        fig = render_strategy_metric_figure(
            payload={"strategies": strategies},
            title="overlap styles",
            metric_grid=(("gt_mean",),),
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        try:
            ax = fig.axes[0]
            lines = {line.get_label(): line for line in ax.lines}
            self.assertEqual(list(lines["ours_single"].get_xdata()), [0, 2])
            self.assertEqual(list(lines["ours_double"].get_xdata()), [0, 2])
            self.assertGreater(
                lines["ours_single"].get_zorder(),
                lines["ours_double"].get_zorder(),
            )

            marker_offsets = [
                [tuple(float(value) for value in xy) for xy in collection.get_offsets()]
                for collection in ax.collections
            ]
            self.assertIn([(0.0, 0.0), (1.88, 0.25)], marker_offsets)
            self.assertIn([(0.0, 0.0), (2.12, 0.25)], marker_offsets)
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
                "gt_single_k2": {
                    "selector": "gt_single_k2",
                    "selector_kind": "set",
                    "drop": [1, 2],
                    "drop_count": 2,
                    "num_intervals": 1,
                },
                "gt_double_k2": {
                    "selector": "gt_double_k2",
                    "selector_kind": "set",
                    "drop": [1, 3],
                    "drop_count": 2,
                    "num_intervals": 2,
                },
            }
            bundle["strategies"] = {
                "gt_single": {
                    "display_name": "GT Single",
                    "points": {"2": "gt_single_k2"},
                },
                "gt_double": {
                    "display_name": "GT Double",
                    "points": {"2": "gt_double_k2"},
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
                strategies["gt_single"]["points"]["2"]["selector"], "i1_j2"
            )
            self.assertEqual(
                strategies["gt_double"]["points"]["2"]["selector"],
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
                    "scannet_dev50_ours_single_pruning_drop.png",
                    "scannet_dev50_ours_multi_pruning_drop.png",
                    "scannet_dev50_ours_pruning_drop.png",
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
            pruning["datasets"]["scannet_dev50"]["strategies"]["ours"]["points"]["3"] = {
                **pruning["datasets"]["scannet_dev50"]["strategies"]["ours"]["points"]["2"],
                "drop_count": 3,
            }
            recovery = add_recovery_comparison_strategies(recovery, pruning)
            self.assertIn("scannet_dev50", pruning["datasets"])
            self.assertIn("scannet_dev50", recovery["datasets"])
            self.assertIn("ours", pruning["datasets"]["scannet_dev50"]["strategies"])
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
                "No recovery",
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
                    for drop_key in pruning["dataset_average"]["strategies"]["ours"]["points"]
                    if int(drop_key) % 2 == 0
                },
            )
            pruning_average_point = pruning["dataset_average"]["strategies"]["ours"]["points"]["2"]
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
                    "recovery_ghost_sequential",
                    "recovery_ghost_interval",
                    "recovery_ghost_interval_sequential",
                    "recovery_replaceme_ls",
                    "recovery_replaceme_cos",
                    "recovery_lora",
                    "recovery_train",
                    "gt_single",
                    "gt_double",
                    "ours_single",
                    "ours_double",
                    "cosine_single",
                    "cosine_double",
                    "inv_mse_single",
                    "inv_mse_double",
                    "cca_single",
                    "cca_double",
                    "replaceme_single",
                    "replaceme_double",
                    "shortgpt",
                    "simpler",
                ),
            )
            pruning_versions = (
                "gt_single",
                "gt_double",
                "ours_single",
                "ours_double",
                "replaceme_single",
                "replaceme_double",
                "shortgpt",
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
                        for name in ("ours_single", "ours_double")
                    }
                }
            },
            # Deliberately incompatible values.  The recovery mean must be
            # rebuilt from the jointly-normalized per-dataset payload instead.
            "dataset_average": {
                "strategies": {
                    "ours_double": {
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

            ours = collect_recovery_eval_payload(
                experiment_root=root,
                prune_strategy="ours",
            )
            strategies = ours["datasets"]["scannet_dev50"]["strategies"]
            self.assertIn("recovery_linearpatch", strategies)
            self.assertEqual(ours["prune_strategy"], "ours")
            gt = collect_recovery_eval_payload(
                experiment_root=root,
                prune_strategy="gt",
            )
            self.assertIn(
                "recovery_linearpatch",
                gt["datasets"]["scannet_dev50"]["strategies"],
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
        self.assertIn("raw metric", fig.axes[0].get_title())
        self.assertIn("normalized metric", fig.axes[1].get_title())

    def test_pruning_collector_accepts_in_memory_bundle_before_plan_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vggt"
            _write_upper(root / "scannet_dev50" / "eval" / "upper")
            _write_summary(root / "scannet_dev50" / "eval" / "pruning" / "i1_j2")
            strategies = {
                "ours": {
                    "display_name": "Ours",
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
            self.assertIn("ours", pruning["datasets"]["scannet_dev50"]["strategies"])

    def test_plot_outputs_use_recovery_names_in_eval_directory(self) -> None:
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
            recovery = collect_recovery_eval_payload(experiment_root=root)
            recovery = add_recovery_comparison_strategies(recovery, pruning)

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
            recovery_paths = write_recovery_eval_plots(
                experiment_root=root,
                backbone="vggt",
                payload=recovery,
            )
            self.assertIsNotNone(gt_path)
            self.assertEqual(gt_path.name, "prune_plan_vggt_gt_degradation.png")
            self.assertTrue(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "scannet_dev50_ours_multi_pruning_drop.png"
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
                    / "7scenes_test_ours_multi_pruning_drop.png"
                ).exists()
            )
            self.assertTrue(any(p.name.endswith("_recovery_drop.png") for p in recovery_paths))
            self.assertTrue(
                any(
                    p.name == "recovery_plan_scannet_dev50_recovery_drop.png"
                    for p in recovery_paths
                )
            )
            self.assertTrue(
                any(p.name == "recovery_plan_vggt_recovery_drop.png" for p in recovery_paths)
            )
            self.assertFalse(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "recovery"
                    / "recovery_plan_scannet_dev50_recovery_drop.png"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "scannet_dev50"
                    / "eval"
                    / "recovery_plan_scannet_dev50_recovery_drop.png"
                ).is_file()
            )
            self.assertFalse((root / "prune_plan_vggt_recovery_drop.png").exists())


if __name__ == "__main__":
    unittest.main()

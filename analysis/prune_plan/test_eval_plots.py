from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    canonical_plan_name_for_group,
)
from analysis.prune_plan.eval_plots import (
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
    select_max_drop_below_mean_degradation,
    strategy_plot_dashes,
    strategy_plot_linestyle,
    strategy_plot_marker,
    strategy_plot_marker_offset,
    strategy_plot_zorder,
    write_pruning_strategy_knee_plot,
    write_gt_degradation_plot,
    write_recovery_eval_plots,
    write_pruning_eval_plots,
    write_source_pruning_eval_plots,
)
from analysis.redundancy.skip_metric_scaling import SEVEN_METRIC_KEYS
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
    def test_pruning_method_labels_and_training_plot_groups(self) -> None:
        self.assertEqual(PRUNING_METHOD_LABELS["ours"], "Ours_CKA")
        self.assertEqual(PRUNING_METHOD_LABELS["cosine"], "Ours_Cosine")
        self.assertEqual(PRUNING_METHOD_LABELS["inv_mse"], "Ours_INV_MSE")

        strategies = (
            "gt_single",
            "gt_double",
            "ours_single",
            "ours_double",
            "cosine_single",
            "cosine_double",
            "inv_mse_single",
            "inv_mse_double",
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
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "ours_single"),
            (
                "gt_single",
                "ours_single",
                "cosine_single",
                "inv_mse_single",
            ),
        )
        self.assertEqual(
            pruning_average_strategy_keys(strategies, "ours_multi"),
            (
                "gt_double",
                "ours_double",
                "cosine_double",
                "inv_mse_double",
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

        normalized = {key: 0.05 for key in SEVEN_METRIC_KEYS}
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

    def test_writes_pruning_strategy_knee_plot(self) -> None:
        strategy_keys = (
            "ours",
            "cosine",
            "gt",
            "replaceme",
            "inv_mse",
            "shortgpt",
            "simpler",
        )
        self.assertEqual(
            len({STRATEGY_PLOT_COLORS[key] for key in strategy_keys}),
            len(strategy_keys),
        )
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"matplotlib unavailable: {exc}")
        normalized = {key: 0.05 for key in SEVEN_METRIC_KEYS}
        strategies = {
            key: {
                "points": {
                    "1": {
                        "available": True,
                        "normalized_relative_to_upper": normalized,
                    }
                }
            }
            for key in strategy_keys
        }
        knees = {
            key: {"drop_count": 1}
            for key in strategy_keys
        }
        with tempfile.TemporaryDirectory() as td:
            path = write_pruning_strategy_knee_plot(
                experiment_root=Path(td),
                backbone="vggt",
                payload={"dataset_average": {"strategies": strategies}},
                knee_points=knees,
            )
            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(
                path.name,
                "prune_plan_vggt_strategy_knees.png",
            )
            self.assertTrue(path.is_file())

    def test_selects_largest_complete_ours_point_at_threshold(self) -> None:
        metrics_at_threshold = {key: 0.1 for key in SEVEN_METRIC_KEYS}
        metrics_below = {key: 0.05 for key in SEVEN_METRIC_KEYS}
        metrics_above = {key: 0.11 for key in SEVEN_METRIC_KEYS}
        payload = {
            "dataset_average": {
                "strategies": {
                    "ours": {
                        "points": {
                            "12": {
                                "available": True,
                                "selector": "i1_j12",
                                "normalized_relative_to_upper": metrics_below,
                            },
                            "17": {
                                "available": True,
                                "selector": "i1_j17",
                                "normalized_relative_to_upper": metrics_below,
                            },
                            "16": {
                                "available": True,
                                "selector": "i1_j16",
                                "normalized_relative_to_upper": metrics_at_threshold,
                            },
                            "18": {
                                "available": True,
                                "selector": "i1_j18",
                                "normalized_relative_to_upper": metrics_above,
                            },
                        }
                    }
                }
            }
        }
        selected = select_max_drop_below_mean_degradation(payload)
        self.assertEqual(selected["drop_count"], 17)
        self.assertEqual(selected["drop_step"], 1)
        self.assertEqual(selected["plan_name"], "prune_span17")
        self.assertEqual(selected["selector"], "i1_j17")
        self.assertAlmostEqual(selected["mean_normalized_degradation"], 0.05)

    def test_knee_selection_falls_back_to_unpruned_for_incomplete_points(self) -> None:
        payload = {
            "dataset_average": {
                "strategies": {
                    "ours": {
                        "points": {
                            "12": {
                                "available": True,
                                "selector": "i1_j12",
                                "normalized_relative_to_upper": {"ape": 0.01},
                            }
                        }
                    }
                }
            }
        }
        selected = select_max_drop_below_mean_degradation(payload)
        self.assertEqual(selected["drop_count"], 0)
        self.assertEqual(selected["plan_name"], "upper")

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
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                    )
                }
            ),
            3,
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

        normalized = {key: 0.25 for key in SEVEN_METRIC_KEYS}
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
                    if 2 <= int(drop_key) <= 24 and int(drop_key) % 2 == 0
                },
            )
            pruning_average_point = pruning["dataset_average"]["strategies"]["ours"]["points"]["2"]
            self.assertEqual(pruning_average_point["relative_to_upper"]["ape"], 2.0)
            self.assertEqual(pruning_average_point["relative_to_upper"]["chamfer_distance"], 2.0)
            self.assertEqual(pruning_average_point["normalized_relative_to_upper"]["ape"], 0.0)
            self.assertEqual(
                gt_mean_degradation_series(pruning),
                [(0, 0.0), (2, 0.0)],
            )
            self.assertEqual(
                STRATEGY_PLOT_ORDER[:29],
                (
                    "no_recovery_single",
                    "no_recovery_double",
                    "no_recovery",
                    "gt_drop",
                    "recovery_linearpatch",
                    "recovery_ghost",
                    "recovery_ghost_interval",
                    "recovery_ghost_interval_sequential",
                    "recovery_ghost_cos",
                    "recovery_ghost_cos_more",
                    "recovery_ghost_cos_scale",
                    "recovery_ghost_mse_more",
                    "recovery_ghost_mse_cos_more",
                    "recovery_ghost_mse_cka_more",
                    "recovery_ghost_mse_cos_cka_more",
                    "recovery_ghost_mse_cos_balanced_more",
                    "recovery_ghost_mse_cka_balanced_more",
                    "recovery_ghost_mse_cos_cka_balanced_more",
                    "recovery_ghost_mse_cka_ema_more",
                    "recovery_ghost_mse_cos_cka_ema_half_more",
                    "recovery_ghost_mse_cos_cka_ema_one_more",
                    "recovery_ghost_mse_cka",
                    "recovery_ghost_cka_only",
                    "recovery_ghost_cka_ridge",
                    "recovery_replaceme_ls",
                    "recovery_replaceme_cos",
                    "recovery_replaceme_cos_more",
                    "recovery_replaceme_cos_scale",
                    "recovery_prunecomp",
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
                set(SEVEN_METRIC_KEYS) & set(linearpatch_point["relative_to_upper"]),
            )

    def test_recovery_comparison_adds_explicit_single_and_double_baselines(self) -> None:
        metric_values = {key: 2.0 for key in SEVEN_METRIC_KEYS}
        relative_values = {key: 2.0 for key in SEVEN_METRIC_KEYS}
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {key: 1.0 for key in SEVEN_METRIC_KEYS},
                    "strategies": {
                        "recovery_ghost": {
                            "display_name": "ghost",
                            "points": {
                                "2": {
                                    "available": True,
                                    "drop_count": 2,
                                    "metrics_mean": {key: 1.5 for key in SEVEN_METRIC_KEYS},
                                    "relative_to_upper": {
                                        key: 1.5 for key in SEVEN_METRIC_KEYS
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
                                    key: 0.0 for key in SEVEN_METRIC_KEYS
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
                for key in SEVEN_METRIC_KEYS
            )
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

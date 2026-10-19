from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from analysis.vis.display_names import backbone_display_name

from analysis.vis.recovery_backbone_comparison_tab import (
    RECOVERY_REMAINING_OUTER_FONT_SIZE,
    RECOVERY_REMAINING_SUBPLOT_FONT_SIZE,
    _BACKBONE_COLORS,
    _calibration_color,
    _average_backbone_payloads,
    discover_calibration_sizes,
    _collapse_structure_min_strategy,
    _filter_testtime_to_family_winners,
    _load_testtime_points,
    _prune_family_plot_style,
    _recovery_method_legend_label,
    _recovery_state_style,
    _render_combined_figure,
    _fill_missing_recovery_points,
    _selected_dataset,
    _shared_profile_maxima,
    build_recovery_ablation_size_tab,
    build_recovery_backbone_remaining_comparison_tab,
    render_recovery_backbone_remaining_comparison,
)
from analysis.vis.recovery_tab import _minimum_strategy_plot_payload


class RecoveryBackboneComparisonTabTest(unittest.TestCase):
    def test_three_backbone_mean_matches_size_budget_and_raw_metrics(self) -> None:
        payloads, totals = [], {}
        for size in (25, 50):
            for backbone, value in zip(('vggt', 'pi3', 'vggt_omega'), (1., 4., 10.)):
                name = f'{backbone}_{size}'
                totals[name] = 48
                payloads.append((name, {
                    'backbone': backbone, 'calibration_size': size,
                    'upper_average': {'ape': value},
                    'strategies': {'recovery_test': {'points': {
                        '12': {'available': True, 'raw_metric': {'ape': value * size},
                               'normalized_metric': {'ape': 999.}},
                        '14': {'available': backbone != 'pi3', 'raw_metric': {'ape': value}},
                    }}},
                }))
        averaged, averaged_totals, messages = _average_backbone_payloads(payloads, totals)
        self.assertEqual(len(averaged), 2)
        for _, payload in averaged:
            points = payload['strategies']['recovery_test']['points']
            self.assertEqual(set(points), {'12'})
            self.assertEqual(points['12']['raw_metric']['ape'], 5. * payload['calibration_size'])
            self.assertEqual(payload['upper_average']['ape'], 5.)
        self.assertEqual(set(averaged_totals.values()), {48})
        self.assertTrue(any('skipped' in message for message in messages))
        missing, _, warnings = _average_backbone_payloads(payloads[:2], totals)
        self.assertEqual(missing, [])
        self.assertTrue(any('missing vggt_omega' in message for message in warnings))
        totals['pi3_25'] = 40
        unequal, _, warnings = _average_backbone_payloads(payloads[:3], totals)
        self.assertEqual(len(unequal), 1)
        self.assertEqual(unequal[0][1]['strategies']['recovery_test']['points']['12']['raw_metric']['ape'], 125.)
        for selected, mean in ((['vggt', 'vggt_omega'], 5.5), (['pi3'], 4.0)):
            averaged, _, _ = _average_backbone_payloads(payloads, totals, selected)
            self.assertEqual(len(averaged), 2)
            for _, payload in averaged:
                point = payload['strategies']['recovery_test']['points']['12']
                self.assertEqual(point['raw_metric']['ape'], mean * payload['calibration_size'])
                self.assertEqual(point['num_backbones'], len(selected))
                self.assertEqual(payload['upper_average']['ape'], mean)
                self.assertTrue(payload['is_backbone_average'])
            if len(selected) == 2:
                self.assertIn('14', averaged[0][1]['strategies']['recovery_test']['points'])

    def test_backbone_mean_can_align_different_models_by_remaining_layers(self) -> None:
        payloads = [
            ("vggt", {
                "backbone": "vggt",
                "upper_average": {"ape": 1.0},
                "strategies": {"recovery_test": {"points": {
                    "18": {"available": True, "raw_metric": {"ape": 2.0}},
                }}},
            }),
            ("pi3", {
                "backbone": "pi3",
                "upper_average": {"ape": 3.0},
                "strategies": {"recovery_test": {"points": {
                    "6": {"available": True, "raw_metric": {"ape": 4.0}},
                }}},
            }),
        ]
        averaged, totals, messages = _average_backbone_payloads(
            payloads,
            {"vggt": 48, "pi3": 36},
            ["vggt", "pi3"],
            coordinate="remaining",
        )
        self.assertEqual(messages, [])
        self.assertEqual(totals, {})
        self.assertEqual(len(averaged), 1)
        payload = averaged[0][1]
        self.assertEqual(payload["point_axis"], "remaining")
        self.assertNotIn("calibration_size", payload)
        point = payload["strategies"]["recovery_test"]["points"]["30"]
        self.assertEqual(point["raw_metric"]["ape"], 3.0)

    def test_average_mode_requires_checked_backbones(self) -> None:
        images, status = render_recovery_backbone_remaining_comparison(
            Path('/unused'), [], 'recovery_ghost', average_backbones=True,
        )
        self.assertEqual(images, [])
        self.assertIn('Select at least one backbone', status)

    def test_shades_rerank_after_deselecting_maximum(self) -> None:
        base = _BACKBONE_COLORS[0]
        darkest = _calibration_color(base, 150, [25, 50, 75, 100, 125, 150])
        np.testing.assert_allclose(_calibration_color(base, 125, [25, 50, 75, 100, 125]), darkest)
        np.testing.assert_allclose(_calibration_color(base, 25, [25]), darkest)
        self.assertLess(sum(_calibration_color(base, 100, [50, 100])),
                        sum(_calibration_color(base, 100, [50, 100, 150])))
        colors = [_calibration_color(base, n, [25, 75, 125]) for n in (0, 25, 75, 125)]
        self.assertTrue(all(sum(a) > sum(b) for a, b in zip(colors, colors[1:])))
        self.assertGreater(np.mean(colors[1]) - np.mean(colors[-1]), 0.4)

    def test_builder_selects_all_discovered_sizes_by_default(self) -> None:
        with patch('analysis.vis.recovery_backbone_comparison_tab._calibration_choices',
                   return_value=[(str(n), str(n)) for n in (25, 50, 75, 100, 125, 150)]):
            with gr.Blocks() as blocks:
                ui = build_recovery_backbone_remaining_comparison_tab(
                    blocks=blocks, output_root=Path('/unused'))
        self.assertNotIn('calibration_sizes_cb', ui)

        with patch('analysis.vis.recovery_backbone_comparison_tab._calibration_choices',
                   return_value=[(str(n), str(n)) for n in (25, 50, 75, 100, 125, 150)]):
            with gr.Blocks() as blocks:
                ui = build_recovery_ablation_size_tab(
                    blocks=blocks, output_root=Path('/unused'))
        self.assertEqual(ui['calibration_sizes_cb'].value, ['25', '50', '75', '100', '125', '150'])

    def test_discover_calibration_sizes_from_saved_results(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'main'
            samples = Path(tmp) / 'main_samples_ablation'
            self.assertEqual(discover_calibration_sizes(root), ['50'])
            for size in ('025', '075', '125', '150'):
                record = samples / f'samples_{size}/pi3/scannet_test50/eval/recovery/i1_j12/recovery_ghost/summary.json'
                record.parent.mkdir(parents=True)
                record.write_text('{}')
            (samples / 'samples_100').mkdir()  # unfinished run with no evaluations
            self.assertEqual(discover_calibration_sizes(root), ['25', '50', '75', '125', '150'])
            self.assertGreater(sum(_calibration_color('blue', 25)), sum(_calibration_color('blue', 75)))

    def test_calibration_shades_and_single_light_baseline(self) -> None:
        key = 'recovery_ghost_retention_min'
        payloads = []
        for size in (50, 100, 150):
            point = {'available': True, 'raw_metric': {'ape': float(size)}}
            payloads.append((f'pi3_{size}', {
                'backbone': 'pi3', 'calibration_size': size,
                'series_label': f'Pi3 {size}',
                'strategies': {key: {'points': {'12': point}},
                               'no_recovery_retention_min': {'points': {'12': point}}},
            }))
        fig = _render_combined_figure(payloads, method_keys=[key])
        self.assertIsNotNone(fig)
        try:
            lines = fig.axes[0].lines
            self.assertEqual(len(lines), 4)  # one shared baseline, three sizes
            expected = [_calibration_color(_BACKBONE_COLORS[0], size)
                        for size in (0, 50, 100, 150)]
            for line, color in zip(lines, expected):
                np.testing.assert_allclose(to_rgb(line.get_color()), color)
            self.assertTrue(all(np.mean(a) > np.mean(b)
                                for a, b in zip(expected, expected[1:])))
            legend = fig.legends[0]
            self.assertGreaterEqual(legend.labelspacing, 0.45)
            self.assertGreaterEqual(legend.borderpad, 0.7)
            self.assertGreaterEqual(legend.handletextpad, 0.8)
            entries = dict(zip((text.get_text() for text in legend.get_texts()), legend.get_lines()))
            self.assertEqual(list(entries).count(backbone_display_name('pi3')), 1)
            self.assertNotIn('Pi3 50', entries)
            for size in (50, 100, 150):
                np.testing.assert_allclose(
                    to_rgb(entries[f'{size} scenes/domain'].get_color()),
                    _calibration_color('#333333', size),
                )
            self.assertFalse(fig.axes[7].axison)
            self.assertEqual(len(fig.axes[7].lines), 0)
        finally:
            plt.close(fig)

    def test_calibration_sizes_empty_selection(self) -> None:
        images, status = render_recovery_backbone_remaining_comparison(
            Path('/unused'), ['pi3'], 'recovery_ghost', calibration_sizes=[]
        )
        self.assertEqual(images, [])
        self.assertIn('Select at least one calibration size', status)

    def test_calibration_sizes_load_separate_roots_with_shared_plans(self) -> None:
        module = 'analysis.vis.recovery_backbone_comparison_tab.'
        metrics = {key: 2.0 for key in (
            'ape', 'are', 'rpe_trans', 'rpe_rot', 'acc', 'comp', 'chamfer_distance'
        )}
        recovery = {'datasets': {'scannet_test50': {
            'upper_average': metrics,
            'strategies': {'recovery_ghost': {'points': {'12': {
                'available': True, 'raw_metric': metrics, 'normalized_metric': metrics,
            }}}},
        }}}
        with (
            patch(module + 'load_prune_plan_bundle', return_value={'num_units': 48}),
            patch(module + 'collect_pruning_eval_payload', return_value={}),
            patch(module + 'collect_recovery_eval_payload', side_effect=lambda **kw: json.loads(json.dumps(recovery))) as collect,
            patch(module + 'add_recovery_comparison_strategies', side_effect=lambda p, *_: p),
            patch(module + '_render_combined_figure', return_value=None) as render,
        ):
            render_recovery_backbone_remaining_comparison(
                Path('/main'), ['pi3'], 'recovery_ghost', 'scannet_test50',
                calibration_sizes=['50', '100', '150'], samples_output_root=Path('/samples'),
            )
        self.assertEqual([c.kwargs['experiment_root'] for c in collect.call_args_list],
                         [Path('/main/pi3')] * 2 + [Path('/samples/samples_100/pi3')] * 2
                         + [Path('/samples/samples_150/pi3')] * 2)
        self.assertEqual(collect.call_args_list[-1].kwargs['prune_plan_root'], Path('/main/pi3'))
        payloads = render.call_args.args[0]
        self.assertEqual([p['calibration_size'] for _, p in payloads], [50, 100, 150])
        self.assertIn('100 scenes/domain', payloads[1][1]['series_label'])
        self.assertIn('150 scenes/domain', payloads[2][1]['series_label'])

    def test_missing_calibration_data_not_replaced_by_pruning(self) -> None:
        module = 'analysis.vis.recovery_backbone_comparison_tab.'
        with (
            patch(module + 'load_prune_plan_bundle', return_value={'num_units': 48}),
            patch(module + 'collect_pruning_eval_payload', return_value={}),
            patch(module + 'collect_recovery_eval_payload', return_value={'datasets': {}}),
            patch(module + '_render_combined_figure', return_value=None) as render,
        ):
            _, status = render_recovery_backbone_remaining_comparison(
                Path('/main'), ['pi3'], 'recovery_ghost', calibration_sizes=['150'],
            )
        self.assertEqual(render.call_args.args[0], [])
        self.assertIn('missing calibration results skipped', status)

    def test_generalization_scopes_load_recovery_group_averages(self) -> None:
        payload = {"datasets": {"scannet_test50": {}, "7scenes_test": {}}}
        for scope in ("same-dataset testing average", "cross-dataset testing average"):
            expected = {"strategies": {"recovery_linearpatch": {"points": {}}}}
            with patch(
                "analysis.vis.recovery_backbone_comparison_tab._dataset_group_average",
                return_value=expected,
            ) as average:
                self.assertIs(_selected_dataset(payload, scope), expected)
                average.assert_called_once_with(payload, scope)

    def test_baseline_only_fills_missing_recovery_coordinates(self) -> None:
        key = "recovery_linearpatch_retention_min"
        payload = {"strategies": {
            key: {"points": {"12": {"available": True, "value": 1},
                              "10": {"available": False}}},
            "no_recovery_retention_min": {"points": {
                "12": {"available": True, "value": 9},
                "10": {"available": True, "value": 2},
                "2": {"available": True, "value": 3},
            }},
        }}
        merged = _fill_missing_recovery_points(payload, [key])
        points = merged["strategies"][key]["points"]
        self.assertEqual({k: p["value"] for k, p in points.items()},
                         {"12": 1, "10": 2, "2": 3})
        self.assertEqual(
            list(merged["strategies"]),
            [key, "no_recovery_retention_min"],
        )
        self.assertEqual(
            set(merged["strategies"]["no_recovery_retention_min"]["points"]),
            {"12", "10", "2"},
        )
        self.assertFalse(payload["strategies"][key]["points"]["10"]["available"])

    def test_remaining_draws_complete_no_recovery_curve(self) -> None:
        point = lambda value: {
            "available": True,
            "raw_metric": {"ape": value},
        }
        payload = {"strategies": {
            "recovery_linearpatch_retention_min": {
                "points": {"12": point(1.0)},
            },
            "no_recovery_retention_min": {
                "points": {
                    "12": point(2.0),
                    "10": point(3.0),
                    "2": point(4.0),
                },
            },
        }}
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch_retention_min"],
            total_units_by_backbone={"vggt": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        baseline = fig.axes[0].lines[0]
        self.assertEqual(list(baseline.get_xdata()), [36.0, 38.0, 46.0])
        self.assertEqual(baseline.get_linestyle(), "--")
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertIn("No Recovery", legend_labels)
        plt.close(fig)

    def test_remaining_method_legend_uses_publication_name(self) -> None:
        self.assertEqual(
            _recovery_method_legend_label(
                "recovery_ghost_interval_sequential"
            ),
            "CKA",
        )
        self.assertEqual(
            _recovery_method_legend_label(
                "recovery_ghost_interval_sequential", "retention"
            ),
            "Degradation",
        )

    def test_testtime_filter_reuses_family_winner_selector(self) -> None:
        payload = {
            "testtime_points": [
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 16,
                    "selector": "i7_j22",
                    "prune_strategy": "retention_single",
                },
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 16,
                    "selector": "i8_j20_i41_j43",
                    "prune_strategy": "retention_double",
                },
            ]
        }
        with patch(
            "analysis.vis.recovery_backbone_comparison_tab.best_recovery_result",
            return_value={"strategy": "retention_double", "selector": "i8_j20_i41_j43"},
        ) as select:
            filtered, failures = _filter_testtime_to_family_winners(
                payload,
                root=Path("/unused"),
                recovery_method="recovery_ghost_interval_sequential",
                families=["retention"],
                selection_datasets=["scannet_dev50", "nuscenes_train"],
            )

        self.assertEqual(failures, [])
        self.assertEqual(
            [point["selector"] for point in filtered["testtime_points"]],
            ["i8_j20_i41_j43"],
        )
        self.assertEqual(
            filtered["testtime_points"][0]["testtime_winner_families"], ["retention"]
        )
        self.assertEqual(select.call_args.kwargs["strategies"], ("retention_single", "retention_double"))

    def test_family_min_selects_single_double_independently_per_curve(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def point(score: float, selector: str) -> dict:
            return {
                "available": True,
                "selector": selector,
                "raw_metric": {metric: score for metric in metrics},
            }

        payload = {
            "strategies": {
                "recovery_linearpatch_CKA_single": {
                    "points": {
                        "2": point(0.1, "recovery_single_2"),
                        "4": point(0.9, "recovery_single_4"),
                    }
                },
                "recovery_linearpatch_CKA_double": {
                    "points": {
                        "2": point(0.9, "recovery_double_2"),
                        "4": point(0.1, "recovery_double_4"),
                    }
                },
                "no_recovery_CKA_single": {
                    "points": {
                        "2": point(0.8, "baseline_single_2"),
                        "4": point(0.2, "baseline_single_4"),
                    }
                },
                "no_recovery_CKA_double": {
                    "points": {
                        "2": point(0.2, "baseline_double_2"),
                        "4": point(0.8, "baseline_double_4"),
                    }
                },
            },
            "testtime_points": [],
        }

        merged, method_keys = _collapse_structure_min_strategy(
            payload, method="recovery_linearpatch", family="CKA"
        )

        self.assertEqual(method_keys, ["recovery_linearpatch_CKA_min"])
        recovery = merged["strategies"][method_keys[0]]["points"]
        baseline = merged["strategies"]["no_recovery"]["points"]
        self.assertEqual(recovery["2"]["min_prune_strategy"], "CKA_single")
        self.assertEqual(recovery["4"]["min_prune_strategy"], "CKA_double")
        self.assertEqual(baseline["2"]["min_prune_strategy"], "CKA_double")
        self.assertEqual(baseline["4"]["min_prune_strategy"], "CKA_single")
        self.assertEqual(recovery["2"]["selector"], "recovery_single_2")
        self.assertEqual(baseline["2"]["selector"], "baseline_double_2")
        self.assertEqual(set(recovery["2"]["raw_metric"].values()), {0.1})
        self.assertEqual(set(baseline["2"]["raw_metric"].values()), {0.2})

    def test_structure_min_reuses_main_calibration_winner_for_every_curve(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def point(score: float, selector: str) -> dict:
            return {
                "available": True,
                "selector": selector,
                "normalized_metric": {metric: score for metric in metrics},
                "raw_metric": {metric: score for metric in metrics},
            }

        payload = {
            "strategies": {
                "recovery_linearpatch_retention_single": {
                    "points": {"12": point(0.9, "test-single")}
                },
                "recovery_linearpatch_retention_double": {
                    "points": {"12": point(0.1, "test-double")}
                },
                "no_recovery_retention_single": {
                    "points": {"12": point(0.8, "baseline-single")}
                },
                "no_recovery_retention_double": {
                    "points": {"12": point(0.2, "baseline-double")}
                },
            },
            "testtime_points": [],
        }
        merged, method_keys = _collapse_structure_min_strategy(
            payload,
            method="recovery_linearpatch",
            family="retention",
            calibration_selection={"12": "retention_single"},
        )

        recovery = merged["strategies"][method_keys[0]]["points"]["12"]
        baseline = merged["strategies"]["no_recovery"]["points"]["12"]
        self.assertEqual(recovery["selector"], "test-single")
        self.assertEqual(baseline["selector"], "baseline-single")
        self.assertEqual(recovery["min_prune_strategy"], "retention_single")
        self.assertEqual(baseline["min_prune_strategy"], "retention_single")
        self.assertEqual(recovery["selection_scope"], "main_calibration")
        self.assertEqual(baseline["selection_scope"], "main_calibration")

    def test_family_min_matches_recovery_overview_four_pose_selection(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def candidate(raw_value: float, normalized_value: float) -> dict:
            return {
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {"2": {
                            "available": True,
                            "raw_metric": {
                                metric: raw_value for metric in metrics
                            },
                            "normalized_metric": {
                                metric: normalized_value for metric in metrics
                            },
                        }}
                    }
                }
            }

        single = candidate(100.0, 0.2)
        double = candidate(1.0, 0.8)
        overview = _minimum_strategy_plot_payload([
            ("CKA_single", single), ("CKA_double", double),
        ])
        overview_point = overview["strategies"]["recovery_linearpatch"]["points"]["2"]

        remaining_payload = {
            "strategies": {
                "recovery_linearpatch_CKA_single": single["strategies"]["recovery_linearpatch"],
                "recovery_linearpatch_CKA_double": double["strategies"]["recovery_linearpatch"],
            },
            "testtime_points": [],
        }
        remaining, method_keys = _collapse_structure_min_strategy(
            remaining_payload,
            method="recovery_linearpatch",
            family="CKA",
        )
        remaining_point = remaining["strategies"][method_keys[0]]["points"]["2"]

        self.assertEqual(overview_point["min_prune_strategy"], "CKA_double")
        self.assertEqual(
            remaining_point["min_prune_strategy"],
            overview_point["min_prune_strategy"],
        )
        self.assertEqual(remaining_point["raw_metric"]["ape"], 1.0)

        fig = _render_combined_figure(
            [("vggt", remaining)],
            method_keys=method_keys,
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [46.0])
        self.assertAlmostEqual(float(fig.axes[0].lines[0].get_ydata()[0]), 1.0)
        self.assertFalse(fig.axes[-1].axison)
        plt.close(fig)

    def test_testtime_loader_matches_strategy_selector_and_averages_datasets(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, ape in (("scannet_dev50", 2.0), ("nuscenes_train", 4.0)):
                out = (
                    root / dataset / "eval" / "testtime"
                    / "recovery_ghost_interval_sequential"
                    / "fastvggt_m0_r0p9" / "i3_j22"
                )
                out.mkdir(parents=True)
                (out / "summary.json").write_text(
                    json.dumps({
                        "selector": "i3_j22",
                        "drop_count": 20,
                        "recovery_plan_name": (
                            "recovery_ghost_interval_sequential"
                        ),
                        "metrics_mean": {"ape": ape},
                    }),
                    encoding="utf-8",
                )

            with patch(
                "analysis.vis.recovery_backbone_comparison_tab._strategy_selectors_by_drop",
                return_value={20: "i3_j22", 22: "i1_j22"},
            ):
                points, warnings = _load_testtime_points(
                    root,
                    recovery_method=(
                        "recovery_ghost_interval_sequential"
                    ),
                    prune_strategy="retention_single",
                    dataset="training average",
                )

        self.assertEqual(warnings, [])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["drop_count"], 20)
        self.assertEqual(points[0]["selector"], "i3_j22")
        self.assertEqual(points[0]["raw_metric"]["ape"], 3.0)

    def test_testtime_compressions_are_not_rendered(self) -> None:
        payload = {
            "upper_average": {},
            "strategies": {
                "recovery_linearpatch_CKA_min": {
                    "display_name": "CKA",
                    "points": {
                        "2": {"available": True, "raw_metric": {"ape": 3.0}}
                    },
                }
            },
            "testtime_points": [
                {
                    "compression": "fastvggt_m0_r0p9",
                    "drop_count": 20,
                    "raw_metric": {"ape": 1.0},
                },
                {
                    "compression": "quantvggt_w4a4",
                    "drop_count": 22,
                    "raw_metric": {"ape": 2.0},
                },
            ],
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch_CKA_min"],
            show_no_recovery=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        figure_width, figure_height = fig.get_size_inches()
        self.assertAlmostEqual(figure_width, 29.6)
        self.assertAlmostEqual(figure_height, 12.0)
        for axis in fig.axes:
            self.assertAlmostEqual(axis.get_box_aspect(), 0.6875)
        self.assertEqual(len(fig.axes[0].lines), 1)
        self.assertEqual(len(fig.axes[0].collections), 1)
        self.assertEqual(fig.legends, [])
        self.assertEqual(
            [text.get_text() for text in fig.axes[7].texts],
            ["VGGT"],
        )
        self.assertIsNone(fig._suptitle)
        self.assertEqual(fig.axes[0].get_title(), "")
        self.assertEqual(fig.axes[0].get_ylabel(), "APE↓ (m)")
        self.assertEqual(fig.axes[0].yaxis.label.get_alpha(), 0.0)
        self.assertEqual(
            [
                text.get_text()
                for text in fig.axes[0].texts
                if text.get_gid() == "metric-label-top"
            ],
            ["APE↓ (m)"],
        )
        top_label = next(
            text
            for text in fig.axes[0].texts
            if text.get_gid() == "metric-label-top"
        )
        self.assertEqual(top_label.get_position(), (0.5, 0.98))
        self.assertEqual(top_label.get_ha(), "center")
        self.assertEqual(
            fig.axes[0].yaxis.label.get_fontsize(),
            RECOVERY_REMAINING_SUBPLOT_FONT_SIZE,
        )
        self.assertTrue(fig.axes[0].yaxis.label.get_fontweight() == "bold")
        self.assertTrue(
            all(
                text.get_fontsize() == RECOVERY_REMAINING_SUBPLOT_FONT_SIZE
                for text in fig.axes[0].get_xticklabels()
            )
        )
        self.assertTrue(
            all(
                text.get_fontweight() == "bold"
                for axis in fig.axes[:7]
                for text in (
                    *axis.get_xticklabels(),
                    *axis.get_yticklabels(),
                )
            )
        )
        fig.canvas.draw()
        for axis in fig.axes[:7]:
            self.assertTrue(
                all(
                    "." not in label.get_text()
                    or len(label.get_text().rsplit(".", 1)[1]) <= 2
                    for label in axis.get_yticklabels()
                )
            )
        self.assertEqual(fig.axes[7].texts[0].get_fontweight(), "bold")
        self.assertEqual(
            fig.axes[7].texts[0].get_fontsize(),
            RECOVERY_REMAINING_OUTER_FONT_SIZE,
        )
        plt.close(fig)

    def test_backbone_palette_matches_pruning_plan_overview(self) -> None:
        from analysis.prune_plan.eval_plots import STRATEGY_PLOT_COLORS

        self.assertEqual(
            _BACKBONE_COLORS[:6],
            tuple(
                STRATEGY_PLOT_COLORS[key]
                for key in ("CKA", "retention", "replaceme", "inv_mse", "cosine", "cca")
            ),
        )

    def test_remaining_axis_uses_each_backbones_total_unit_count(self) -> None:
        payload = {
            "upper_average": {"ape": 3.0},
            "strategies": {
                "recovery_linearpatch": {
                    "display_name": "linearpatch",
                    "points": {
                        "8": {"available": True, "raw_metric": {"ape": 1.0}},
                        "10": {"available": True, "raw_metric": {"ape": 2.0}},
                    },
                }
            },
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch"],
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [38, 40, 48])
        self.assertEqual(fig.axes[0].lines[0].get_ydata()[-1], 3.0)
        self.assertEqual(fig.axes[0].get_xlim(), (1.0, 48.0))
        self.assertIn(1.0, fig.axes[0].get_xticks())
        self.assertIn(48.0, fig.axes[0].get_xticks())
        horizontal_upper = fig.axes[0].lines[1]
        self.assertEqual(list(horizontal_upper.get_ydata()), [3.0, 3.0])
        self.assertEqual(len(fig.axes[0].lines), 2)
        self.assertTrue(
            all(axis.get_xlabel() == "Remaining units" for axis in fig.axes[:7])
        )
        self.assertTrue(
            all(np.allclose(np.diff(axis.get_xticks()[1:]), 4.0) for axis in fig.axes[:7])
        )
        plt.close(fig)

    def test_x_axis_can_be_divided_by_two(self) -> None:
        payload = {
            "upper_average": {},
            "strategies": {
                "recovery_linearpatch": {
                    "points": {
                        "8": {"available": True, "raw_metric": {"ape": 1.0}},
                        "10": {"available": True, "raw_metric": {"ape": 2.0}},
                    }
                }
            },
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=["recovery_linearpatch"],
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48},
            halve_x_axis=True,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [19.0, 20.0])
        self.assertEqual(len(fig.axes[0].lines), 1)
        self.assertTrue(
            all(
                axis.get_xlabel() == "Remaining units ÷ 2"
                for axis in fig.axes[:7]
            )
        )
        plt.close(fig)

    def test_strategy_legend_factors_state_and_structure(self) -> None:
        point = {"available": True, "raw_metric": {"ape": 1.0}}
        payload = {
            "upper_average": {},
            "strategies": {
                "no_recovery_single": {
                    "display_name": "No Recovery · Single",
                    "points": {"8": point},
                },
                "no_recovery_double": {
                    "display_name": "No Recovery · Multi",
                    "points": {"8": point},
                },
                "recovery_linearpatch_CKA_single": {
                    "display_name": "CKA · Single",
                    "points": {"8": point},
                },
                "recovery_linearpatch_CKA_double": {
                    "display_name": "CKA · Multi",
                    "points": {"8": point},
                },
            },
        }
        fig = _render_combined_figure(
            [("vggt", payload)],
            method_keys=[
                "recovery_linearpatch_CKA_single",
                "recovery_linearpatch_CKA_double",
            ],
            total_units_by_backbone={"vggt": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertIn("No Recovery", legend_labels)
        self.assertNotIn("Recovery", legend_labels)
        self.assertIn("Single", legend_labels)
        self.assertIn("Multi", legend_labels)
        self.assertNotIn("No Recovery · Single", legend_labels)
        self.assertNotIn("CKA · Multi", legend_labels)
        plt.close(fig)

    def test_remaining_renderer_reads_num_units_from_each_prune_bundle(self) -> None:
        captured = {}

        def capture(payloads, **kwargs):
            captured.update(kwargs)
            return object()

        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.load_prune_plan_bundle",
                side_effect=[{"num_units": 48}, {"num_units": 36}],
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value={"datasets": {}},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                side_effect=lambda payload, _pruning, _variant: payload,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._pruning_only_recovery_like_payload",
                return_value={"strategies": {}},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                side_effect=capture,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            render_recovery_backbone_remaining_comparison(
                Path("/unused"), ["vggt", "pi3"], "recovery_linearpatch"
            )

        self.assertEqual(
            captured["total_units_by_backbone"], {"vggt": 48, "pi3": 36}
        )
        self.assertTrue(captured["show_no_recovery"])

    def test_recovery_state_has_distinct_weight_and_marker_fill(self) -> None:
        no_recovery = _recovery_state_style("no_recovery_single")
        recovery = _recovery_state_style("recovery_linearpatch_CKA_single")

        self.assertEqual(no_recovery["markerfacecolor"], "none")
        self.assertIsNone(recovery["markerfacecolor"])
        self.assertLess(no_recovery["linewidth"], recovery["linewidth"])
        self.assertLess(no_recovery["alpha"], recovery["alpha"])

    def test_retention_uses_solid_line_with_extra_weight(self) -> None:
        cka = _prune_family_plot_style("recovery_linearpatch_CKA_min")
        retention = _prune_family_plot_style(
            "recovery_linearpatch_retention_min"
        )

        self.assertEqual(cka["linestyle"], "-")
        self.assertEqual(retention["linestyle"], "-")
        self.assertGreater(
            _recovery_state_style("recovery_linearpatch_retention_min")["linewidth"],
            _recovery_state_style("recovery_linearpatch_CKA_min")["linewidth"],
        )

    def test_average_status_warns_about_missing_datasets(self) -> None:
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "strategies": {"recovery_linearpatch": {"points": {}}}
                }
            }
        }
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.load_prune_plan_bundle",
                return_value={"num_units": 48},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value=recovery,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                return_value=recovery,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                return_value=object(),
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            _, status = render_recovery_backbone_remaining_comparison(
                Path("/unused"),
                ["vggt"],
                "recovery_linearpatch",
                "training average",
            )

        self.assertIn("⚠️ WARNING", status)
        self.assertIn("1/2 datasets", status)
        self.assertIn("Missing: nuscenes_train", status)

    def test_remaining_mean_uses_largest_profile_max_across_backbones(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def payload(value: float, profile_maximum: float) -> dict:
            return {
                "upper_average": {metric: 1.0 for metric in metrics},
                "normalized_upper_average": {metric: 0.0 for metric in metrics},
                "normalization_bounds": {
                    metric: {"min": 0.0, "max": profile_maximum}
                    for metric in metrics
                },
                "strategies": {
                    "recovery_linearpatch_CKA_min": {
                        "points": {
                            "2": {
                                "available": True,
                                "raw_metric": {metric: value for metric in metrics},
                            }
                        }
                    }
                },
            }

        first = payload(2.0, 3.0)
        second = payload(3.0, 5.0)
        self.assertEqual(
            set(_shared_profile_maxima([first, second]).values()), {5.0}
        )

        fig = _render_combined_figure(
            [("vggt", first), ("pi3", second)],
            method_keys=["recovery_linearpatch_CKA_min"],
            show_no_recovery=False,
            total_units_by_backbone={"vggt": 48, "pi3": 48},
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        mean_curves = [
            line
            for line in fig.axes[-1].lines
            if np.array_equal(line.get_xdata(), [46.0, 48.0])
        ]
        self.assertEqual(mean_curves, [])
        self.assertFalse(fig.axes[-1].axison)
        vertical_bounds = [
            line
            for line in fig.axes[-1].lines
            if np.array_equal(line.get_xdata(), [48.0, 48.0])
        ]
        self.assertEqual(vertical_bounds, [])
        plt.close(fig)

    def test_three_coincident_upper_lines_use_three_dash_phases(self) -> None:
        def payload(value: float) -> dict:
            return {
                "upper_average": {"ape": 3.0},
                "strategies": {
                    "recovery_linearpatch_CKA_min": {
                        "points": {
                            "2": {
                                "available": True,
                                "raw_metric": {"ape": value},
                            }
                        }
                    }
                },
            }

        fig = _render_combined_figure(
            [
                ("vggt", payload(1.0)),
                ("vggt_omega", payload(1.5)),
                ("pi3", payload(2.0)),
            ],
            method_keys=["recovery_linearpatch_CKA_min"],
            show_no_recovery=False,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        horizontal_bounds = [
            line
            for line in fig.axes[0].lines
            if np.array_equal(line.get_ydata(), [3.0, 3.0])
        ]
        self.assertEqual(len(horizontal_bounds), 3)
        self.assertTrue(
            all(
                line._us_dashSeq == horizontal_bounds[0]._us_dashSeq
                for line in horizontal_bounds[1:]
            )
        )
        self.assertEqual(
            len({line._us_dashOffset for line in horizontal_bounds}), 3
        )
        plt.close(fig)

    def test_builder_has_single_method_and_no_normalization_control(self) -> None:
        with gr.Blocks() as blocks:
            ui = build_recovery_backbone_remaining_comparison_tab(
                blocks=blocks, output_root="/path/that/does/not/exist"
            )
        self.assertIsInstance(ui["method_dd"], gr.Dropdown)
        self.assertNotIn("calibration_sizes_cb", ui)
        self.assertIsInstance(ui["average_backbones_cb"], gr.Checkbox)
        self.assertFalse(ui["average_backbones_cb"].value)
        self.assertIsInstance(ui["prune_families_cb"], gr.CheckboxGroup)
        self.assertEqual(ui["prune_families_cb"].value, ["retention", "CKA"])
        self.assertIsInstance(ui["image"], gr.Image)
        self.assertEqual(ui["dataset_dd"].choices, [
            ("Training", "training average"),
            ("Same-Dataset Generalization", "same-dataset testing average"),
            ("Cross-Dataset Generalization", "cross-dataset testing average"),
        ])
        self.assertEqual(ui["dataset_dd"].value, "same-dataset testing average")
        self.assertFalse(ui["method_dd"].multiselect)
        self.assertIsInstance(ui["show_no_recovery_cb"], gr.Checkbox)
        self.assertTrue(ui["show_no_recovery_cb"].value)
        self.assertIsInstance(ui["halve_x_axis_cb"], gr.Checkbox)
        self.assertFalse(ui["halve_x_axis_cb"].value)
        self.assertNotIn("prune_strategy_dd", ui)
        self.assertNotIn("prune_structure_cb", ui)
        self.assertNotIn("gallery", ui)
        self.assertNotIn("normalize_metrics", ui)

    def test_builder_defaults_to_autogroup_when_available(self) -> None:
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.discover_recovery_overview_options",
                return_value=(
                    ["vggt"],
                    [
                        "recovery_linearpatch",
                        "recovery_ghost",
                        "recovery_ghost_interval_sequential",
                        "recovery_ghost_interval_sequential_autogroup",
                    ],
                ),
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_recovery_backbone_remaining_comparison_tab(
                    blocks=blocks,
                    output_root="/unused",
                )

        self.assertEqual(
            ui["method_dd"].value,
            "recovery_ghost_interval_sequential_autogroup",
        )
        self.assertEqual(ui["tab"].label, "Recovery Remaining")

    def test_remaining_builder_has_separate_tab(self) -> None:
        with gr.Blocks() as blocks:
            ui = build_recovery_backbone_remaining_comparison_tab(
                blocks=blocks, output_root="/path/that/does/not/exist"
            )
        self.assertIsInstance(ui["image"], gr.Image)
        self.assertEqual(ui["tab"].id, "recovery_backbone_remaining_comparison")
        self.assertEqual(ui["tab"].label, "Recovery Remaining")

    def test_ablation_size_builder_has_calibration_and_average_controls(self) -> None:
        with gr.Blocks() as blocks:
            ui = build_recovery_ablation_size_tab(
                blocks=blocks, output_root="/path/that/does/not/exist"
            )
        self.assertEqual(ui["tab"].id, "recovery_ablation_size")
        self.assertEqual(ui["tab"].label, "Ablation Size")
        self.assertIsInstance(ui["calibration_sizes_cb"], gr.CheckboxGroup)
        self.assertEqual(ui["calibration_sizes_cb"].value, ["50"])
        self.assertIsInstance(ui["average_backbones_cb"], gr.Checkbox)
        self.assertTrue(ui["average_backbones_cb"].value)

    def test_ablation_size_uses_dropped_layer_axis_for_each_backbone(self) -> None:
        captured = {}

        def capture(payloads, **kwargs):
            captured.update(kwargs)
            return object()

        metrics = {"ape": 1.0}
        recovery = {"datasets": {"scannet_test50": {
            "upper_average": metrics,
            "strategies": {"recovery_ghost": {"points": {
                "12": {"available": True, "raw_metric": metrics},
            }}},
        }}}
        module = "analysis.vis.recovery_backbone_comparison_tab."
        with (
            patch(module + "load_prune_plan_bundle", return_value={"num_units": 48}),
            patch(module + "collect_pruning_eval_payload", return_value={}),
            patch(module + "collect_recovery_eval_payload", return_value=recovery),
            patch(module + "add_recovery_comparison_strategies", side_effect=lambda p, *_: p),
            patch(module + "_render_combined_figure", side_effect=capture),
            patch(module + "_figure_to_rgb", return_value=np.zeros((1, 1, 3), dtype=np.uint8)),
            patch(module + "plt.close"),
        ):
            render_recovery_backbone_remaining_comparison(
                Path("/unused"), ["vggt"], "recovery_ghost",
                "scannet_test50", calibration_sizes=["50"], x_axis="dropped",
            )

        self.assertIsNone(captured["total_units_by_backbone"])
        self.assertFalse(captured["show_upper"])
        self.assertEqual(captured["calibration_size_multiplier"], 2)
        self.assertEqual(captured["x_axis_left"], 12.0)

        fig = _render_combined_figure(
            [("backbone_mean_50", {
                "backbone": "Mean (1 backbone)",
                "is_backbone_average": True,
                "calibration_size": 50,
                "strategies": {"recovery_ghost_retention_min": {"points": {
                    "12": {"available": True, "raw_metric": metrics},
                }}},
            })],
            method_keys=["recovery_ghost_retention_min"],
            show_no_recovery=False,
            x_axis_left=12.0,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [12.0])
        self.assertTrue(
            all(axis.get_xlabel() == "Dropped Layers" for axis in fig.axes[:7])
        )
        self.assertTrue(
            all(axis.get_xlim()[0] == 12.0 for axis in fig.axes[:7])
        )
        plt.close(fig)

    def test_ablation_size_hides_upper_and_averaged_backbone_legend(self) -> None:
        metrics = {"ape": 1.0}
        sizes = (50, 100, 150)
        fig = _render_combined_figure(
            [(f"backbone_mean_{size}", {
                "backbone": "Mean (2 backbones)",
                "is_backbone_average": True,
                "calibration_size": size,
                "upper_average": {"ape": 3.0},
                "strategies": {"recovery_ghost_retention_min": {"points": {
                    "12": {"available": True, "raw_metric": metrics},
                }}},
            }) for size in sizes],
            method_keys=["recovery_ghost_retention_min"],
            show_no_recovery=False,
            show_upper=False,
            show_backbone_legend=False,
            calibration_size_multiplier=2,
        )
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertFalse(any(
            np.array_equal(line.get_ydata(), [3.0, 3.0])
            for line in fig.axes[0].lines
        ))
        legend_labels = [
            text.get_text()
            for legend in fig.legends
            for text in legend.get_texts()
        ]
        self.assertNotIn("Upper", legend_labels)
        self.assertNotIn("Mean (2 backbones)", legend_labels)
        self.assertNotIn("Calibration (shade)", legend_labels)
        self.assertNotIn("Degradation", legend_labels)
        self.assertIn("100 scenes", legend_labels)
        self.assertTrue(all("/domain" not in label for label in legend_labels))
        legend = fig.legends[0]
        entries = dict(zip(
            (text.get_text() for text in legend.get_texts()),
            legend.get_lines(),
        ))
        for size in sizes:
            handle = entries[f"{size * 2} scenes"]
            self.assertEqual(handle.get_marker(), "o")
            np.testing.assert_allclose(
                to_rgb(handle.get_color()),
                _calibration_color(_BACKBONE_COLORS[0], size, sizes),
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bounds = legend.get_window_extent(renderer)
        axes_bounds = fig.axes[7].get_window_extent(renderer)
        self.assertAlmostEqual(legend_bounds.y0, axes_bounds.y0, delta=1.0)
        self.assertAlmostEqual(legend_bounds.y1, axes_bounds.y1, delta=1.0)
        plt.close(fig)

    def test_remaining_loads_and_collapses_degradation_and_cka(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )
        recovery = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {"ape": 1.0},
                    "strategies": {
                        "recovery_linearpatch": {
                            "display_name": "linearpatch",
                            "points": {"2": {
                                "available": True,
                                "raw_metric": {metric: 2.0 for metric in metrics},
                            }},
                        }
                    },
                }
            }
        }
        captured = {}

        def capture(payloads, **kwargs):
            captured["payloads"] = payloads
            captured.update(kwargs)
            return object()

        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.load_prune_plan_bundle",
                return_value={"num_units": 48},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value=recovery,
            ) as collect_recovery,
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                side_effect=lambda payload, _pruning, _variant: payload,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._main_calibration_structure_selection",
                side_effect=lambda **kwargs: (
                    {"2": f"{kwargs['family']}_double"},
                    [],
                ),
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                side_effect=capture,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_backbone_comparison_tab.plt.close"),
        ):
            render_recovery_backbone_remaining_comparison(
                Path("/unused"), ["vggt"], "recovery_linearpatch",
                "scannet_dev50", True,
            )

        self.assertEqual(
            [call.kwargs["prune_strategy"] for call in collect_recovery.call_args_list],
            [
                "retention_single", "retention_double",
                "CKA_single", "CKA_double",
            ],
        )
        strategies = captured["payloads"][0][1]["strategies"]
        self.assertEqual(list(strategies), [
            "recovery_linearpatch_retention_min",
            "recovery_linearpatch_CKA_min",
        ])
        point = strategies["recovery_linearpatch_retention_min"]["points"]["2"]
        self.assertEqual(point["min_prune_strategy"], "retention_double")
        self.assertEqual(point["selection_scope"], "main_calibration")
        self.assertEqual(
            captured["method_keys"],
            [
                "recovery_linearpatch_retention_min",
                "recovery_linearpatch_CKA_min",
            ],
        )

    def test_remaining_pruning_criteria_can_select_cka_only(self) -> None:
        recovery = {"datasets": {"scannet_dev50": {
            "strategies": {"recovery_linearpatch": {"points": {}}},
        }}}
        with (
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.load_prune_plan_bundle",
                return_value={"num_units": 48},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.collect_recovery_eval_payload",
                return_value=recovery,
            ) as collect_recovery,
            patch(
                "analysis.vis.recovery_backbone_comparison_tab.add_recovery_comparison_strategies",
                side_effect=lambda payload, _pruning, _variant: payload,
            ),
            patch(
                "analysis.vis.recovery_backbone_comparison_tab._render_combined_figure",
                return_value=None,
            ),
        ):
            render_recovery_backbone_remaining_comparison(
                Path("/unused"), ["vggt"], "recovery_linearpatch",
                "scannet_dev50", prune_families=["CKA"],
            )
        self.assertEqual(
            [call.kwargs["prune_strategy"] for call in collect_recovery.call_args_list],
            ["CKA_single", "CKA_double"],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.shared_legend import SHARED_LEGEND_CAPTION
from analysis.vis.recovery_tab import (
    _apply_backbone_recovery_method_aliases,
    RECOVERY_DETAIL_COLUMN_LABELS,
    RECOVERY_DETAIL_OUTER_FONT_SIZE,
    RECOVERY_DETAIL_SUBPLOT_FONT_SIZE,
    RECOVERY_OVERVIEW_OUTER_FONT_SIZE,
    RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
    RECOVERY_SHARED_LEGEND_FONT_SIZE,
    _OVERVIEW_DATASET_AVERAGE_GROUPS,
    _compose_recovery_detail,
    _compose_recovery_overview,
    _dataset_group_average,
    _apply_backbone_normalization_profiles,
    _collect_backbone_normalization_profiles,
    _figure_to_rgb,
    _formalize_recovery_legend_names,
    _keep_only_first_column_ylabel,
    _label_recovery_detail_figure,
    _minimum_strategy_plot_payload,
    _minimum_strategy_selection,
    _prune_strategy_candidates,
    _render_recovery_legend,
    build_recovery_tabs,
    discover_recovery_overview_datasets,
    discover_recovery_overview_options,
    discover_recovery_overview_prune_strategies,
    render_recovery_detail,
    render_recovery_detail_matrix,
    render_recovery_overview,
    render_recovery_overview_matrix,
)


class RecoveryOverviewTabTest(unittest.TestCase):
    def test_analysis_groups_use_two_calibration_same_and_cross_datasets(self) -> None:
        self.assertEqual(
            _OVERVIEW_DATASET_AVERAGE_GROUPS,
            {
                "training average": ("scannet_dev50", "nuscenes_train"),
                "same-dataset testing average": (
                    "scannet_test50",
                    "nuscenes_val",
                ),
                "cross-dataset testing average": (
                    "7scenes_test",
                    "waymo_validation",
                ),
            },
        )

    def test_detail_uses_four_by_two_grid_with_label_in_last_cell(self) -> None:
        figure, axes = plt.subplots(2, 4)
        try:
            _label_recovery_detail_figure(
                figure,
                "VGGT · Same-Dataset\nGeneralization · CKA-based",
            )
            self.assertEqual(
                [axis.get_title() for axis in figure.axes[:7]],
                [""] * 7,
            )
            self.assertEqual(
                [axis.get_ylabel() for axis in figure.axes[:7]],
                list(RECOVERY_DETAIL_COLUMN_LABELS),
            )
            self.assertTrue(
                all(axis.yaxis.label.get_alpha() == 0.0 for axis in figure.axes[:7])
            )
            self.assertEqual(
                [
                    [text.get_text() for text in axis.texts if text.get_gid() == "metric-label-top"]
                    for axis in figure.axes[:7]
                ],
                [[label] for label in RECOVERY_DETAIL_COLUMN_LABELS],
            )
            self.assertTrue(
                all(
                    text.get_fontsize() == RECOVERY_DETAIL_SUBPLOT_FONT_SIZE
                    for axis in figure.axes[:7]
                    for text in axis.texts
                    if text.get_gid() == "metric-label-top"
                )
            )
            self.assertEqual(
                [text.get_text() for text in axes[1, 3].texts],
                ["VGGT", "Same-Dataset Generalization", "CKA-based"],
            )
            self.assertEqual(
                [text.get_position()[1] for text in axes[1, 3].texts],
                [0.74, 0.50, 0.26],
            )
            self.assertTrue(
                all(not text.get_wrap() for text in axes[1, 3].texts)
            )
            self.assertFalse(axes[1, 3].axison)
        finally:
            plt.close(figure)

    def test_detail_uses_based_pruning_names_in_row_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            for strategy, expected_name in (
                ("CKA_min", "CKA-based"),
                ("Retention_min", "Degradation-based"),
            ):
                with self.subTest(strategy=strategy), patch(
                    "analysis.vis.recovery_tab._label_recovery_detail_figure"
                ) as label:
                    images, status = render_recovery_detail(
                        root,
                        ["vggt"],
                        ["recovery_linearpatch"],
                        "scannet_dev50",
                        strategy,
                    )
                    self.assertEqual(len(images), 1, status)
                    self.assertEqual(
                        label.call_args.args[1],
                        "VGGT",
                    )
                    label.reset_mock()
                    images, status = render_recovery_detail(
                        root,
                        ["vggt"],
                        ["recovery_linearpatch"],
                        "scannet_dev50",
                        strategy,
                        show_prune_strategy_label=True,
                    )
                    self.assertEqual(len(images), 1, status)
                    self.assertEqual(
                        label.call_args.args[1],
                        f"VGGT · {expected_name}",
                    )

    def test_figure_export_keeps_canvas_size_independent_of_y_axis_label(
        self,
    ) -> None:
        figures = []
        for ylabel in ("Normalized Degradation", ""):
            fig, axis = plt.subplots(figsize=(7.4, 5.8))
            axis.plot([0, 1], [0, 1])
            axis.set_ylabel(ylabel)
            figures.append(fig)
        try:
            self.assertEqual(
                _figure_to_rgb(figures[0]).shape,
                _figure_to_rgb(figures[1]).shape,
            )
        finally:
            for figure in figures:
                plt.close(figure)

    def test_recovery_profiles_are_per_backbone_and_use_stage2_bounds(
        self,
    ) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def pruning(upper: float, cka: float, retention: float) -> dict:
            maximum = max(cka, retention)
            return {
                "datasets": {
                    "test": {
                        "upper_average": {
                            metric: upper for metric in metrics
                        },
                        "normalization_profile": {
                            "method": "stage2_brute_force_single_interval_raw_metric_max",
                            "upper_average": {
                                metric: upper for metric in metrics
                            },
                            "maximum_brute_force_metric": {
                                metric: maximum for metric in metrics
                            },
                        },
                        "strategies": {
                            "CKA_single": {
                                "points": {
                                    "22": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: 100.0 for metric in metrics
                                        },
                                    },
                                    "24": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: cka for metric in metrics
                                        },
                                    },
                                }
                            },
                            "retention_single": {
                                "points": {
                                    "24": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: retention for metric in metrics
                                        },
                                    },
                                }
                            }
                        },
                    }
                }
            }

        profiles = _collect_backbone_normalization_profiles({
                "first": pruning(1.0, 3.0, 4.0),
                "second": pruning(10.0, 15.0, 13.0),
            })
        first_profile = profiles["first"]["test"]
        second_profile = profiles["second"]["test"]
        self.assertEqual(
            first_profile["normalization_denominator"],
            {metric: 3.0 for metric in metrics},
        )
        self.assertEqual(
            second_profile["normalization_denominator"],
            {metric: 5.0 for metric in metrics},
        )
        self.assertEqual(
            first_profile["method"],
            "stage2_brute_force_single_interval_raw_metric_max",
        )

        def recovery(upper: float, no_recovery: float, recovered: float) -> dict:
            return {
                "datasets": {
                    "test": {
                        "upper_average": {
                            metric: upper for metric in metrics
                        },
                        "strategies": {
                            "no_recovery": {
                                "points": {
                                    "24": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: no_recovery for metric in metrics
                                        },
                                        "normalized_metric": {
                                            metric: 999.0 for metric in metrics
                                        },
                                    }
                                }
                            },
                            "recovery_linearpatch": {
                                "points": {
                                    "24": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: recovered for metric in metrics
                                        },
                                    }
                                }
                            },
                        },
                    }
                }
            }

        first = _apply_backbone_normalization_profiles(
            recovery(1.0, 3.0, 2.0),
            profiles,
            backbone="first",
        )
        second = _apply_backbone_normalization_profiles(
            recovery(10.0, 15.0, 12.0),
            profiles,
            backbone="second",
        )
        self.assertEqual(
            first["datasets"]["test"]["strategies"]["no_recovery"]
            ["points"]["24"]["normalized_metric"],
            {metric: 2.0 / 3.0 for metric in metrics},
        )
        self.assertEqual(
            first["datasets"]["test"]["strategies"]
            ["recovery_linearpatch"]["points"]["24"]
            ["normalized_metric"],
            {metric: 1.0 / 3.0 for metric in metrics},
        )
        self.assertEqual(
            second["datasets"]["test"]["strategies"]["no_recovery"]
            ["points"]["24"]["normalized_metric"],
            {metric: 1.0 for metric in metrics},
        )
        self.assertEqual(
            first["datasets"]["test"]["normalization_bounds"]["ape"],
            {"min": 1.0, "max": 4.0},
        )

    def test_backbone_profile_uses_stage2_brute_force_metric_maxima(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def pruning() -> dict:
            return {
                "datasets": {
                    "test": {
                        "upper_average": {
                            metric: 1.0 for metric in metrics
                        },
                        "normalization_profile": {
                            "method": "stage2_brute_force_single_interval_raw_metric_max",
                            "upper_average": {
                                metric: 1.0 for metric in metrics
                            },
                            "maximum_brute_force_metric": {
                                metric: 7.0 for metric in metrics
                            },
                        },
                        "strategies": {
                            strategy: {
                                "points": {
                                    "24": {
                                        "available": True,
                                        "metrics_mean": {
                                            metric: endpoint
                                            for metric in metrics
                                        },
                                    }
                                }
                            }
                            for strategy, endpoint in (
                                ("CKA_double", 3.0),
                                ("CKA_single", 5.0),
                                ("retention_double", 6.0),
                                ("retention_single", 4.0),
                            )
                        },
                    }
                }
            }

        profile = _collect_backbone_normalization_profiles(
            {"vggt": pruning()}
        )["vggt"]["test"]
        self.assertEqual(
            profile["normalization_denominator"],
            {metric: 6.0 for metric in metrics},
        )
        self.assertEqual(
            profile["method"],
            "stage2_brute_force_single_interval_raw_metric_max",
        )

    def test_only_first_metric_column_keeps_its_y_axis_title(self) -> None:
        class Axis:
            def __init__(self) -> None:
                self.ylabel = "metric"

            def set_ylabel(self, value: str) -> None:
                self.ylabel = value

        figure = type("Figure", (), {"axes": [Axis() for _ in range(7)]})()
        _keep_only_first_column_ylabel(
            figure,
            ((
                "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
                "chamfer_distance",
            ),),
        )
        self.assertEqual(
            [axis.ylabel for axis in figure.axes],
            ["metric", "", "", "", "", "", ""],
        )

    def test_overview_composer_uses_dataset_rows_and_backbone_column_groups(
        self,
    ) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, f"{backbone} · {dataset} · {strategy}")
            for backbone in ("VGGT", "π³")
            for dataset, strategy in (
                ("training average", "CKA_min"),
                ("training average", "Retention_min"),
                ("same-dataset testing average", "CKA_min"),
                ("same-dataset testing average", "Retention_min"),
                ("cross-dataset testing average", "CKA_min"),
                ("cross-dataset testing average", "Retention_min"),
            )
        ]
        with patch(
            "analysis.vis.recovery_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_recovery_overview(items)

        section = compose.call_args.args[0][0]
        self.assertEqual(section.columns, 4)
        self.assertEqual(
            section.column_labels,
            ["VGGT", "VGGT", "π³", "π³"],
        )
        self.assertIsNone(section.group_headings)
        self.assertTrue(section.column_labels_below)
        self.assertFalse(section.row_banners_align_content)
        self.assertEqual(
            section.row_labels,
            [
                "Calibration\nDataset",
                "Same-Dataset\nGeneralization",
                "Cross-Dataset\nGeneralization",
            ],
        )
        self.assertEqual([len(row) for row in section.fixed_rows], [4, 4, 4])
        self.assertTrue(section.row_labels_bold)
        self.assertTrue(section.column_labels_bold)

    def test_overview_composer_can_replace_dataset_names_with_row_letters(
        self,
    ) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, f"VGGT · {dataset} · Retention_min")
            for dataset in (
                "training average",
                "same-dataset testing average",
                "cross-dataset testing average",
            )
        ]
        with patch(
            "analysis.vis.recovery_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_recovery_overview(items, simplify_dataset_names=True)

        self.assertEqual(
            compose.call_args.args[0][0].row_labels,
            ["(a)", "(b)", "(c)"],
        )
        self.assertEqual(
            compose.call_args.kwargs["outer_font_size"],
            RECOVERY_OVERVIEW_OUTER_FONT_SIZE,
        )

    def test_overview_legend_is_vertical_and_appended_on_the_right(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        legend = np.zeros((40, 10, 3), dtype=np.uint8)
        with (
            patch(
                "analysis.vis.recovery_tab.compose_sections",
                return_value=tile,
            ),
            patch(
                "analysis.vis.recovery_tab.append_shared_legends_right",
                return_value=np.zeros((20, 42, 3), dtype=np.uint8),
            ) as append,
        ):
            result = _compose_recovery_overview([
                (tile, "VGGT · training average · Retention_min"),
                (legend, SHARED_LEGEND_CAPTION),
            ])

        append.assert_called_once()
        self.assertFalse(append.call_args.kwargs["frameon"])
        self.assertTrue(
            np.array_equal(np.asarray(append.call_args.args[1][0][0]), legend)
        )
        self.assertEqual(result.shape, (20, 42, 3))

        entries = [(Line2D([], [], color="red"), "Ours")]
        with patch(
            "analysis.vis.recovery_tab.render_shared_legend_banner",
            return_value=legend,
        ) as render:
            _render_recovery_legend(entries, 26, vertical=True)
        self.assertEqual(render.call_args.kwargs["max_columns"], 1)
        self.assertFalse(render.call_args.kwargs["fit_single_row"])
        self.assertFalse(render.call_args.kwargs["frameon"])

        with patch(
            "analysis.vis.recovery_tab.render_shared_legend_banner",
            return_value=legend,
        ) as render:
            _render_recovery_legend(entries, 26)
        self.assertTrue(render.call_args.kwargs["fit_single_row"])
        self.assertFalse(render.call_args.kwargs["frameon"])

    def test_detail_legend_expands_instead_of_clipping_one_row(self) -> None:
        entries = [
            (
                Line2D([], [], color=f"C{index % 10}"),
                f"Recovery method with a long label {index}",
            )
            for index in range(10)
        ]
        legend = _render_recovery_legend(entries, 32)
        self.assertIsNotNone(legend)
        assert legend is not None
        self.assertGreater(legend.shape[1], 2400)
        self.assertTrue(np.all(legend[:, :4] >= 250))
        self.assertTrue(np.all(legend[:, -4:] >= 250))

    def test_detail_composer_labels_every_backbone_strategy_row(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        with (
            patch(
                "analysis.vis.recovery_tab.compose_sections",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ) as compose,
            patch(
                "analysis.vis.recovery_tab.prepend_shared_legend",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ) as prepend,
        ):
            _compose_recovery_detail([
                (tile, "VGGT · Same-Dataset\nGeneralization · CKA_min"),
                (tile, "π³ · Same-Dataset\nGeneralization · CKA_min"),
                (tile, "VGGT · Same-Dataset\nGeneralization · Retention_min"),
                (tile, "π³ · Same-Dataset\nGeneralization · Retention_min"),
                (tile, SHARED_LEGEND_CAPTION),
            ])

        sections = compose.call_args.args[0]
        self.assertEqual(len(sections), 4)
        self.assertTrue(all(section.heading == "" for section in sections))
        self.assertTrue(all(section.columns == 1 for section in sections))
        self.assertTrue(all(len(section.cells) == 1 for section in sections))
        self.assertTrue(all(section.row_labels is None for section in sections))
        self.assertTrue(all(section.column_labels is None for section in sections))
        self.assertEqual(compose.call_args.kwargs["top_label_to_image_gap"], 0)
        self.assertEqual(compose.call_args.kwargs["section_gap"], 0)
        prepend.assert_called_once()
        self.assertIsNotNone(prepend.call_args.args[1])

    def test_overview_composer_shrinks_to_one_column_per_backbone(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.recovery_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_recovery_overview([
                (tile, "VGGT · training average · Retention_min"),
                (tile, "π³ · training average · Retention_min"),
            ])

        section = compose.call_args.args[0][0]
        self.assertEqual(section.columns, 2)
        self.assertEqual(section.column_labels, ["VGGT", "π³"])
        self.assertIsNone(section.group_headings)
        self.assertTrue(section.column_labels_below)
        self.assertEqual([len(row) for row in section.fixed_rows], [2])

    def test_publication_legend_names_are_formalized(self) -> None:
        payload = {
            "strategies": {
                "recovery_replaceme_ls": {"display_name": "replaceme_ls"},
                "recovery_replaceme_cos": {"display_name": "replaceme_cos"},
                "recovery_ghost": {"display_name": "ghost"},
                "recovery_ghost_interval": {"display_name": "ghost_interval"},
                "recovery_linearpatch": {"display_name": "linearpatch"},
                "recovery_ghost_interval_sequential": {
                    "display_name": "ghost_interval_sequential_current_target"
                },
                "recovery_ghost_interval_sequential_grouped": {
                    "display_name": "ghost_interval_sequential_special_patch"
                },
                "recovery_ghost_interval_sequential_autogroup": {
                    "display_name": "ghost_interval_sequential_autogroup"
                },
                "no_recovery": {"display_name": "No recovery Retention Double"},
            }
        }

        renamed = _formalize_recovery_legend_names(payload)

        self.assertEqual(
            {
                key: value["display_name"]
                for key, value in renamed["strategies"].items()
            },
            {
                "recovery_replaceme_ls": "ReplaceMe LS",
                "recovery_replaceme_cos": "ReplaceMe Cos",
                "recovery_ghost": "Ghost",
                "recovery_ghost_interval": "Ghost",
                "recovery_linearpatch": "LinearPatch",
                "recovery_ghost_interval_sequential": "Current Target",
                "recovery_ghost_interval_sequential_grouped": (
                    "Current Target (Special/Patch)"
                ),
                "recovery_ghost_interval_sequential_autogroup": "Ours",
                "no_recovery": "No Recovery",
            },
        )
        self.assertEqual(
            payload["strategies"]["recovery_ghost"]["display_name"], "ghost"
        )

    def test_min_uses_four_pose_sum_then_copies_the_whole_winning_point(self) -> None:
        metrics = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")

        def candidate(
            first: float,
            last: float,
            marker: str,
            ghost_score: float,
            baseline_score: float,
        ) -> dict:
            normalized = {metric: first for metric in metrics}
            normalized["chamfer_distance"] = last
            return {
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {"2": {
                            "available": True,
                            "normalized_metric": normalized,
                            "raw_metric": {metric: first for metric in metrics},
                            "marker": marker,
                        }}
                    },
                    "recovery_ghost": {
                        "points": {"2": {
                            "available": True,
                            "normalized_metric": {
                                metric: ghost_score for metric in metrics
                            },
                            "raw_metric": {
                                metric: ghost_score for metric in metrics
                            },
                            "marker": f"{marker}_ghost",
                        }}
                    },
                    "no_recovery": {
                        "points": {"2": {
                            "available": True,
                            "normalized_metric": {
                                metric: baseline_score for metric in metrics
                            },
                            "raw_metric": {
                                metric: baseline_score for metric in metrics
                            },
                            "marker": f"{marker}_baseline",
                        }}
                    },
                }
            }

        # Geometry normalization is not used; the lower raw seven-metric mean wins.
        merged = _minimum_strategy_plot_payload([
            (
                "retention_single",
                candidate(0.4, 1.0, "retention", 0.1, 0.2),
            ),
            ("CKA_single", candidate(0.5, 0.0, "CKA", 0.9, 0.8)),
        ])
        point = merged["strategies"]["recovery_linearpatch"]["points"]["2"]
        self.assertEqual(point["min_prune_strategy"], "retention_single")
        self.assertEqual(point["marker"], "retention")
        ghost = merged["strategies"]["recovery_ghost"]["points"]["2"]
        self.assertEqual(ghost["min_prune_strategy"], "retention_single")
        self.assertEqual(ghost["marker"], "retention_ghost")
        baseline = merged["strategies"]["no_recovery"]["points"]["2"]
        self.assertEqual(baseline["min_prune_strategy"], "retention_single")
        self.assertEqual(
            baseline["marker"], "retention_baseline"
        )
        self.assertNotIn("no_recovery__recovery_linearpatch", merged["strategies"])

    def test_metric_min_views_compare_single_and_double_per_drop(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def candidate(scores: dict[str, float], marker: str) -> dict:
            return {
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {
                            drop: {
                                "available": True,
                                "normalized_metric": {
                                    metric: score for metric in metrics
                                },
                                "raw_metric": {
                                    metric: score for metric in metrics
                                },
                                "marker": f"{marker}_{drop}",
                            }
                            for drop, score in scores.items()
                        }
                    }
                }
            }

        self.assertEqual(
            _prune_strategy_candidates("Retention_min"),
            ("retention_double", "retention_single"),
        )
        self.assertEqual(
            _prune_strategy_candidates("CKA_min"),
            ("CKA_double", "CKA_single"),
        )

        payloads = {
            "retention_double": candidate(
                {"2": 0.1, "4": 0.8, "6": 0.45}, "double"
            ),
            "retention_single": candidate(
                {"2": 0.8, "4": 0.1, "6": 0.5}, "single"
            ),
        }
        merged = _minimum_strategy_plot_payload([
            (name, payloads[name])
            for name in _prune_strategy_candidates("Retention_min")
        ])
        points = merged["strategies"]["recovery_linearpatch"]["points"]
        self.assertEqual(points["2"]["min_prune_strategy"], "retention_double")
        self.assertEqual(points["4"]["min_prune_strategy"], "retention_single")
        self.assertEqual(points["6"]["min_prune_strategy"], "retention_single")
        self.assertEqual(points["2"]["marker"], "double_2")
        self.assertEqual(points["4"]["marker"], "single_4")
        self.assertEqual(points["6"]["marker"], "single_6")

    def test_generalization_reuses_calibration_single_double_winner(self) -> None:
        metrics = (
            "ape", "are", "rpe_trans", "rpe_rot", "acc", "comp",
            "chamfer_distance",
        )

        def candidate(score: float, marker: str) -> dict:
            return {
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {"8": {
                            "available": True,
                            "normalized_metric": {
                                metric: score for metric in metrics
                            },
                            "raw_metric": {
                                metric: score for metric in metrics
                            },
                            "marker": marker,
                        }}
                    }
                }
            }

        calibration = [
            ("retention_single", candidate(0.1, "cal-single")),
            ("retention_double", candidate(0.8, "cal-double")),
        ]
        generalization = [
            ("retention_single", candidate(0.9, "test-single")),
            ("retention_double", candidate(0.2, "test-double")),
        ]
        selection = _minimum_strategy_selection(calibration)
        merged = _minimum_strategy_plot_payload(generalization, selection)
        point = merged["strategies"]["recovery_linearpatch"]["points"]["8"]
        self.assertEqual(point["min_prune_strategy"], "retention_single")
        self.assertEqual(point["selection_scope"], "calibration")
        self.assertEqual(point["marker"], "test-single")

    def test_overview_matrix_defaults_to_retention_only(self) -> None:
        tile = np.zeros((2, 3, 3), dtype=np.uint8)

        def render_overview(*args, **kwargs):
            entries = kwargs["shared_legend_entries"]
            if not entries:
                entries.extend((
                    (Line2D([], [], color="black"), "Full Model"),
                    (Line2D([], [], color="red"), "LinearPatch"),
                ))
            return [(tile, "VGGT")], "ok"

        with (
            patch(
                "analysis.vis.recovery_tab.render_recovery_overview",
                side_effect=render_overview,
            ) as render,
            patch(
                "analysis.vis.recovery_tab._render_recovery_legend",
                return_value=None,
            ) as render_legend,
        ):
            gallery, status = render_recovery_overview_matrix(
                Path("/unused"), ["vggt"], ["recovery_linearpatch"]
            )

        actual_calls = render.call_args_list
        self.assertEqual(
            [
                (call.args[3], call.args[5])
                for call in actual_calls
            ],
            [
                ("same-dataset testing average", "Retention_min"),
                ("cross-dataset testing average", "Retention_min"),
            ],
        )
        self.assertEqual(
            [call.kwargs["show_bottom_xlabel"] for call in actual_calls],
            [False, True],
        )
        self.assertEqual(
            [call.kwargs["show_metric_ylabels"] for call in actual_calls],
            [True, True],
        )
        self.assertEqual(
            [call.kwargs["show_strategy_title"] for call in actual_calls],
            [False, False],
        )
        self.assertTrue(
            all(
                call.kwargs["label_only_first_backbone_yaxis"]
                for call in actual_calls
            )
        )
        self.assertEqual(
            [caption for _, caption in gallery],
            [
                "VGGT · same-dataset testing average · Retention_min",
                "VGGT · cross-dataset testing average · Retention_min",
            ],
        )
        self.assertEqual(status.splitlines(), ["ok"] * 2)
        self.assertEqual(
            render_legend.call_args.args[1],
            RECOVERY_SHARED_LEGEND_FONT_SIZE,
        )
        self.assertEqual(
            [label for _, label in render_legend.call_args.args[0]],
            ["LinearPatch"],
        )
        self.assertTrue(render_legend.call_args.kwargs["vertical"])

    def test_overview_matrix_uses_selected_datasets_only(self) -> None:
        tile = np.zeros((2, 3, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.recovery_tab.render_recovery_overview",
            return_value=([(tile, "VGGT")], "ok"),
        ) as render:
            gallery, status = render_recovery_overview_matrix(
                Path("/unused"),
                ["vggt"],
                ["recovery_linearpatch"],
                ["scannet_dev50"],
                ["CKA_min", "Retention_min"],
            )

        actual_calls = render.call_args_list
        self.assertEqual(
            [(call.args[3], call.args[5]) for call in actual_calls],
            [
                ("scannet_dev50", "CKA_min"),
                ("scannet_dev50", "Retention_min"),
            ],
        )
        self.assertEqual(
            [call.kwargs["show_strategy_title"] for call in actual_calls],
            [True, True],
        )
        self.assertEqual(
            [caption for _, caption in gallery],
            [
                "VGGT · scannet_dev50 · CKA_min",
                "VGGT · scannet_dev50 · Retention_min",
            ],
        )
        self.assertEqual(status.splitlines(), ["ok", "ok"])

        gallery, status = render_recovery_overview_matrix(
            Path("/unused"), ["vggt"], ["recovery_linearpatch"], []
        )
        self.assertEqual(gallery, [])
        self.assertEqual(status, "Select at least one dataset.")

    def test_detail_matrix_renders_selected_strategies_as_separate_rows(self) -> None:
        tile = np.zeros((2, 3, 3), dtype=np.uint8)

        def render_detail(*args, **kwargs):
            kwargs["shared_legend_entries"].append((Line2D([], [], color="red"), "LinearPatch"))
            return [(tile, "VGGT"), (tile, "π³")], "ok"

        with (
            patch(
                "analysis.vis.recovery_tab.render_recovery_detail",
                side_effect=render_detail,
            ) as render,
            patch(
                "analysis.vis.recovery_tab.render_shared_legend_banner",
                return_value=tile,
            ) as render_legend,
        ):
            gallery, status = render_recovery_detail_matrix(
                Path("/unused"),
                ["vggt", "pi3"],
                ["recovery_linearpatch"],
                "same-dataset testing average",
                ["CKA_min", "Retention_min"],
            )

        self.assertEqual(
            [call.args[4] for call in render.call_args_list],
            ["CKA_min", "Retention_min"],
        )
        self.assertEqual(
            [call.kwargs["show_bottom_xlabel"] for call in render.call_args_list],
            [False, True],
        )
        self.assertTrue(
            all(
                call.kwargs["show_prune_strategy_label"]
                for call in render.call_args_list
            )
        )
        shared_entries = [
            call.kwargs["shared_legend_entries"]
            for call in render.call_args_list
        ]
        self.assertIs(shared_entries[0], shared_entries[1])
        render_legend.assert_called_once()
        self.assertEqual(render_legend.call_args.kwargs["font_size_pixels"], RECOVERY_DETAIL_OUTER_FONT_SIZE)
        self.assertEqual(render_legend.call_args.kwargs["max_columns"], len(shared_entries[0]))
        for handle, _ in render_legend.call_args.args[0]:
            self.assertEqual(handle.get_marker(), "o")
            self.assertEqual(handle.get_linestyle(), "None")
        self.assertEqual(
            [caption for _, caption in gallery],
            [
                "VGGT · Same-Dataset\nGeneralization · CKA_min",
                "π³ · Same-Dataset\nGeneralization · CKA_min",
                "VGGT · Same-Dataset\nGeneralization · Retention_min",
                "π³ · Same-Dataset\nGeneralization · Retention_min",
                SHARED_LEGEND_CAPTION,
            ],
        )
        self.assertEqual(status.splitlines(), ["ok", "ok"])

    def _write_fixture(self, root: Path) -> None:
        backbone = root / "vggt"
        backbone.mkdir(parents=True)
        (backbone / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt",
                    "dataset": "scannet_dev50",
                    "num_units": 4,
                    "intervals": {
                        "i1_j2": {
                            "selector": "i1_j2", "i": 1, "j": 2,
                            "start": 1, "end": 2, "drop": [1, 2],
                            "drop_count": 2,
                        }
                    },
                    "strategies": {
                        "CKA_double": {
                            "display_name": "CKA Double", "points": {"2": "i1_j2"}
                        },
                        "CKA_single": {
                            "display_name": "CKA Single", "points": {"2": "i1_j2"}
                        },
                        "retention_double": {
                            "display_name": "Retention Double", "points": {"2": "i1_j2"}
                        },
                        "retention_single": {
                            "display_name": "Retention Single", "points": {"2": "i1_j2"}
                        },
                        "cosine_double": {
                            "display_name": "Cosine double", "points": {"2": "i1_j2"}
                        },
                    },
                    "spans": {},
                }
            ),
            encoding="utf-8",
        )
        upper = backbone / "scannet_dev50" / "eval" / "upper"
        upper.mkdir(parents=True)
        EvalRecordV3(
            backbone="vggt",
            dataset="scannet_dev50",
            model_name="upper",
            average={"ape": 1.0, "are": 1.0, "rpe_trans": 1.0, "rpe_rot": 1.0,
                     "acc": 1.0, "comp": 1.0, "chamfer_distance": 1.0},
        ).save(upper / "eval_record_v3.json")
        for result_root in (
            backbone / "scannet_dev50" / "eval" / "pruning",
            backbone / "scannet_dev50" / "eval" / "recovery" / "i1_j2",
        ):
            target = (
                result_root / "recovery_linearpatch"
                if result_root.name == "i1_j2" and result_root.parent.name == "recovery"
                else result_root / "i1_j2"
            )
            target.mkdir(parents=True)
            (target / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "drop_count": 2,
                        "num_scenes": 1,
                        "metrics_mean": {"ape": 2.0, "are": 2.0, "rpe_trans": 2.0,
                                         "rpe_rot": 2.0, "acc": 2.0, "comp": 2.0,
                                         "chamfer_distance": 2.0},
                    }
                ),
                encoding="utf-8",
            )

    def test_discovers_controls_and_redraws_from_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["vggt"], ["recovery_linearpatch"]),
            )
            self.assertEqual(
                discover_recovery_overview_datasets(root),
                [
                    "training average",
                    "testing average",
                    "scannet_dev50",
                ],
            )
            images, status = render_recovery_overview(
                root, ["vggt"], ["recovery_linearpatch"], "scannet_dev50"
            )
            self.assertEqual(len(images), 1, status)
            self.assertEqual(images[0][0].ndim, 3)
            self.assertGreater(images[0][0].shape[0], 400)
            self.assertIn("redrawn from current summaries", status)
            self.assertIn("/ scannet_dev50", status)

            detail_images, detail_status = render_recovery_detail(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
            )
            self.assertEqual(len(detail_images), 1, detail_status)
            self.assertGreater(
                detail_images[0][0].shape[1], detail_images[0][0].shape[0] * 2
            )
            self.assertLess(
                detail_images[0][0].shape[1], detail_images[0][0].shape[0] * 4
            )

            _, average_status = render_recovery_overview(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "training average",
            )
            self.assertIn("⚠️ WARNING", average_status)
            self.assertIn("1/2 datasets", average_status)
            self.assertIn("Missing: nuscenes_train", average_status)

            matrix, matrix_status = render_recovery_overview_matrix(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                ["training average"],
            )
            self.assertEqual(len(matrix), 2, matrix_status)
            self.assertTrue(
                matrix[-1][1].startswith(f"{SHARED_LEGEND_CAPTION}|entries=")
            )

            detail_matrix, detail_matrix_status = render_recovery_detail_matrix(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
                ["CKA_min"],
            )
            self.assertEqual(len(detail_matrix), 2, detail_matrix_status)
            self.assertEqual(
                detail_matrix[-1][1], SHARED_LEGEND_CAPTION
            )

    def test_discovers_plan_before_evaluation_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_ghost.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["pi3"], ["recovery_ghost"]),
            )

    def test_discovers_and_renders_shared_selector_for_prune_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(
                discover_recovery_overview_prune_strategies(root),
                ["CKA_double", "CKA_single", "retention_double", "retention_single"],
            )
            images, status = render_recovery_overview(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
                prune_strategy="CKA_double",
            )
            self.assertEqual(len(images), 1, status)
            self.assertIn("CKA_double", status)

            for minimum_strategy in ("Retention_min", "CKA_min"):
                with self.subTest(prune_strategy=minimum_strategy):
                    images, status = render_recovery_overview(
                        root,
                        ["vggt"],
                        ["recovery_linearpatch"],
                        "scannet_dev50",
                        prune_strategy=minimum_strategy,
                    )
                    self.assertEqual(len(images), 1, status)
                    self.assertIn(minimum_strategy, status)

    def test_discovers_methods_from_recovery_plan_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_plan_all.json").write_text(
                json.dumps(
                    {
                        "methods": {
                            "recovery_linearpatch_compacted": {},
                            "recovery_ghost_compacted": {},
                            "recovery_ghost_interval": {},
                            "recovery_ghost_interval_sequential": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                discover_recovery_overview_options(root),
                (
                    ["pi3"],
                    [
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                        "recovery_ghost_interval_sequential_autogroup",
                    ],
                ),
            )

    def test_pi3_uses_sequential_as_adaptive_ours_fallback(self) -> None:
        sequential = "recovery_ghost_interval_sequential"
        autogroup = "recovery_ghost_interval_sequential_autogroup"
        payload = {
            "datasets": {
                "test": {
                    "strategies": {
                        autogroup: {
                            "display_name": "Ours",
                            "points": {"20": {"available": False}},
                        },
                        sequential: {
                            "display_name": "Current Target",
                            "points": {"20": {"available": True}},
                        }
                    }
                }
            }
        }
        pi3 = _apply_backbone_recovery_method_aliases(
            payload, backbone="pi3"
        )
        alias = pi3["datasets"]["test"]["strategies"][autogroup]
        self.assertEqual(alias["display_name"], "Ours")
        self.assertEqual(alias["source_method"], sequential)
        self.assertEqual(alias["points"], {"20": {"available": True}})
        self.assertFalse(
            payload["datasets"]["test"]["strategies"][autogroup]
            ["points"]["20"]["available"]
        )
        vggt = _apply_backbone_recovery_method_aliases(
            payload, backbone="vggt"
        )
        self.assertFalse(
            vggt["datasets"]["test"]["strategies"][autogroup]
            ["points"]["20"]["available"]
        )

    def test_builder_exposes_cka_retention_checkboxes_defaulting_to_retention(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            recovery_root = (
                root / "vggt" / "scannet_dev50" / "eval" / "recovery" / "i1_j2"
            )
            (recovery_root / "recovery_ghost_interval").mkdir()
            (
                recovery_root
                / "recovery_ghost_interval_sequential"
            ).mkdir()
            (
                recovery_root
                / "recovery_ghost_interval_sequential_grouped"
            ).mkdir()
            (
                recovery_root
                / "recovery_ghost_interval_sequential_autogroup"
            ).mkdir()
            with gr.Blocks() as blocks:
                ui = build_recovery_tabs(blocks=blocks, output_root=root)
            self.assertEqual(ui["tab"].label, "Recovery")
            self.assertEqual(ui["detail_tab"].label, "Recovery")
            self.assertNotIn("value_mode", ui)
            self.assertNotIn("prune_strategy_dd", ui)
            self.assertNotIn("dataset_dd", ui)
            self.assertIsInstance(ui["dataset_cb"], gr.CheckboxGroup)
            self.assertEqual(
                ui["dataset_cb"].value,
                [
                    "same-dataset testing average",
                    "cross-dataset testing average",
                ],
            )
            self.assertIn(
                ("Calibration\nDataset", "training average"),
                ui["dataset_cb"].choices,
            )
            self.assertIn(
                ("scannet_dev50", "scannet_dev50"),
                ui["dataset_cb"].choices,
            )
            self.assertIsInstance(ui["prune_strategy_cb"], gr.CheckboxGroup)
            self.assertEqual(
                ui["prune_strategy_cb"].choices,
                [("CKA", "CKA_min"), ("Degradation", "Retention_min")],
            )
            self.assertEqual(
                ui["prune_strategy_cb"].value,
                ["Retention_min"],
            )
            self.assertIsInstance(ui["simplify_dataset_names"], gr.Checkbox)
            self.assertTrue(ui["simplify_dataset_names"].value)
            self.assertIsInstance(
                ui["detail_prune_strategy_dd"], gr.CheckboxGroup
            )
            self.assertEqual(
                ui["detail_prune_strategy_dd"].choices,
                [
                    ("CKA", "CKA_min"),
                    ("Degradation", "Retention_min"),
                ],
            )
            self.assertEqual(
                ui["detail_prune_strategy_dd"].value,
                ["Retention_min"],
            )
            self.assertEqual(
                ui["detail_dataset_dd"].value,
                "same-dataset testing average",
            )
            self.assertIn(
                (
                    "Cross-Dataset\nGeneralization",
                    "cross-dataset testing average",
                ),
                ui["detail_dataset_dd"].choices,
            )
            self.assertNotIn("normalize_metrics", ui)
            self.assertNotIn("detail_normalize_metrics", ui)
            self.assertIsInstance(ui["detail_tab"], gr.Tab)
            self.assertIsInstance(ui["detail_gallery"], gr.Image)
            self.assertIn(("Mean", "mean"), ui["backbone_cb"].choices)
            self.assertIn(("VGGT", "vggt"), ui["backbone_cb"].choices)
            self.assertEqual(ui["backbone_cb"].value, ["vggt"])
            self.assertNotIn("recovery_ghost_interval", ui["method_cb"].value)
            self.assertNotIn(
                "recovery_ghost_interval", ui["detail_method_cb"].value
            )
            self.assertNotIn(
                "recovery_ghost_interval_sequential",
                ui["method_cb"].value,
            )
            self.assertNotIn(
                "recovery_ghost_interval_sequential",
                ui["detail_method_cb"].value,
            )
            self.assertNotIn(
                "recovery_ghost_interval_sequential_grouped",
                ui["method_cb"].value,
            )
            self.assertNotIn(
                "recovery_ghost_interval_sequential_grouped",
                ui["detail_method_cb"].value,
            )
            self.assertIn(
                "recovery_ghost_interval_sequential_autogroup",
                ui["method_cb"].value,
            )
            self.assertIn(
                "recovery_ghost_interval_sequential_autogroup",
                ui["detail_method_cb"].value,
            )

    def test_mean_backbone_averages_shared_recovery_points(self) -> None:
        def plot_payload(value: float) -> dict:
            return {
                "upper_average": {"ape": value},
                "strategies": {
                    "recovery_linearpatch": {
                        "display_name": "Linear patch",
                        "points": {
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "metrics_mean": {"ape": value},
                            }
                        },
                    }
                },
            }

        figure = SimpleNamespace(axes=[])
        with (
            patch(
                "analysis.vis.recovery_tab.discover_recovery_overview_options",
                return_value=(["vggt", "pi3"], ["recovery_linearpatch"]),
            ),
            patch(
                "analysis.vis.recovery_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_tab.collect_recovery_eval_payload",
                side_effect=[
                    {"datasets": {"scannet_dev50": plot_payload(2.0)}},
                    {"datasets": {"scannet_dev50": plot_payload(4.0)}},
                ],
            ),
            patch(
                "analysis.vis.recovery_tab.add_recovery_comparison_strategies",
                side_effect=lambda recovery, *_: recovery,
            ),
            patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["mean"],
                ["recovery_linearpatch"],
                "scannet_dev50",
            )

        self.assertEqual(len(images), 1, status)
        averaged = render.call_args.kwargs["payload"]
        self.assertEqual(averaged["num_backbones"], 2)
        self.assertEqual(averaged["upper_average"]["ape"], 3.0)
        self.assertEqual(
            averaged["strategies"]["recovery_linearpatch"]["points"]["2"]
            ["metrics_mean"]["ape"],
            3.0,
        )
        self.assertIn("backbones=vggt,pi3", status)

    def test_recovery_plot_does_not_prepend_upper_anchor_to_strategies(self) -> None:
        payload = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {"ape": 1.0},
                    "strategies": {
                        "gt_drop": {"points": {}},
                        "recovery_linearpatch": {
                            "points": {
                                "2": {
                                    "available": True,
                                    "drop_count": 2,
                                    "metrics_mean": {"ape": 5.0},
                                    "raw_metric": {"ape": 5.0},
                                }
                            }
                        },
                        "recovery_ghost_interval_sequential_autogroup": {
                            "display_name": "Ours",
                            "points": {
                                "2": {
                                    "available": True,
                                    "drop_count": 2,
                                    "metrics_mean": {"ape": 4.0},
                                    "raw_metric": {"ape": 4.0},
                                }
                            },
                        },
                    },
                }
            }
        }
        figure = SimpleNamespace(axes=[])
        with (
            patch(
                "analysis.vis.recovery_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_tab.collect_recovery_eval_payload",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.add_recovery_comparison_strategies",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["vggt"],
                [
                    "recovery_linearpatch",
                    "recovery_ghost_interval_sequential_autogroup",
                ],
                "scannet_dev50",
            )

        self.assertEqual(len(images), 1, status)
        overview_kwargs = render.call_args.kwargs
        self.assertEqual(overview_kwargs["metric_grid"], (("gt_mean",),))
        self.assertEqual(
            overview_kwargs["strategy_keys"],
            [
                "recovery_linearpatch",
                "recovery_ghost_interval_sequential_autogroup",
            ],
        )
        self.assertEqual(
            overview_kwargs["strategy_style_overrides"],
            {
                "recovery_ghost_interval_sequential_autogroup": {
                    "linewidth": 4.4,
                }
            },
        )
        self.assertFalse(overview_kwargs["include_upper_anchor"])
        self.assertEqual(overview_kwargs["subplot_width"], 7.4)
        self.assertEqual(overview_kwargs["subplot_box_aspect"], 0.55)
        self.assertEqual(overview_kwargs["figure_height_per_row"], 4.8)
        self.assertEqual(
            overview_kwargs["fixed_layout_margins"],
            (0.10, 0.09, 0.99, 0.99),
        )
        self.assertIsNone(overview_kwargs["fixed_layout_wspace"])
        self.assertEqual(
            overview_kwargs["legend_fontsize"],
            RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(overview_kwargs["legend_fontweight"], "bold")
        self.assertEqual(overview_kwargs["legend_loc"], "upper left")
        self.assertEqual(overview_kwargs["upper_label"], "Full Model")
        self.assertEqual(overview_kwargs["default_linewidth"], 3.2)
        self.assertEqual(overview_kwargs["upper_linewidth"], 3.2)
        self.assertEqual(
            overview_kwargs["legend_order"],
            (
                "Full Model",
                "No Recovery",
                "LinearPatch",
                "ReplaceMe Cos",
                "ReplaceMe LS",
                "Streamline FFN",
                "Ghost",
                "Current Target",
                "Current Target (Special/Patch)",
                "Ours",
            ),
        )
        self.assertEqual(
            overview_kwargs["tick_label_fontsize"],
            RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(
            overview_kwargs["axis_label_fontsize"],
            RECOVERY_OVERVIEW_SUBPLOT_FONT_SIZE,
        )
        self.assertFalse(overview_kwargs["show_figure_title"])
        self.assertFalse(overview_kwargs["show_metric_titles"])
        self.assertTrue(overview_kwargs["metric_labels_as_ylabels"])
        self.assertEqual(
            overview_kwargs["metric_ylabel"], "Normalized Degradation"
        )
        self.assertTrue(overview_kwargs["y_tick_percentage"])
        self.assertFalse(
            overview_kwargs["cap_y_at_normalization_max"]
        )
        self.assertIsNone(overview_kwargs["minimum_y_axis_upper"])
        self.assertEqual(overview_kwargs["fixed_y_axis_upper"], 0.40)
        self.assertIsNone(overview_kwargs["shared_xlabel"])
        self.assertEqual(
            overview_kwargs["bottom_row_xlabel"], "Dropped Layers"
        )
        self.assertTrue(overview_kwargs["always_normalize_metric_mean"])
        self.assertNotIn("overview_normalization", overview_kwargs["payload"])
        with (
            patch(
                "analysis.vis.recovery_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_tab.collect_recovery_eval_payload",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.add_recovery_comparison_strategies",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as detail_render,
            patch(
                "analysis.vis.recovery_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_tab.plt.close"),
        ):
            detail_images, detail_status = render_recovery_detail(
                Path("/unused"),
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
            )

        self.assertEqual(len(detail_images), 1, detail_status)
        self.assertEqual(
            detail_render.call_args.kwargs["metric_grid"],
            (
                ("ape", "are", "rpe_trans", "rpe_rot"),
                ("acc", "comp", "chamfer_distance"),
            ),
        )
        self.assertTrue(
            detail_render.call_args.kwargs["metric_labels_as_ylabels"]
        )
        self.assertIsNone(detail_render.call_args.kwargs["metric_ylabel"])
        self.assertFalse(
            detail_render.call_args.kwargs["y_tick_percentage"]
        )
        self.assertEqual(
            detail_render.call_args.kwargs["y_tick_max_decimals"], 2
        )
        self.assertEqual(
            detail_render.call_args.kwargs["legend_fontsize"],
            RECOVERY_DETAIL_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(
            detail_render.call_args.kwargs["tick_label_fontsize"],
            RECOVERY_DETAIL_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(
            detail_render.call_args.kwargs["axis_label_fontsize"],
            RECOVERY_DETAIL_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(
            detail_render.call_args.kwargs["x_tick_step"], 4
        )
        self.assertAlmostEqual(
            detail_render.call_args.kwargs["subplot_box_aspect"],
            0.6875,
        )
        self.assertAlmostEqual(
            detail_render.call_args.kwargs["figure_height_per_row"], 6.0
        )
        self.assertTrue(detail_render.call_args.kwargs["all_axes_xlabel"])
        self.assertFalse(
            detail_render.call_args.kwargs["cap_y_at_normalization_max"]
        )
        self.assertFalse(
            detail_render.call_args.kwargs["normalize_metrics"]
        )
        self.assertIsNone(
            detail_render.call_args.kwargs["fixed_layout_margins"]
        )
        self.assertIsNone(
            detail_render.call_args.kwargs["fixed_layout_wspace"]
        )
        self.assertNotIn(
            "overview_normalization",
            detail_render.call_args.kwargs["payload"],
        )

    def test_overview_labels_every_backbone_and_detail_labels_last_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                wraps=__import__(
                    "analysis.vis.recovery_tab",
                    fromlist=["render_strategy_metric_figure"],
                ).render_strategy_metric_figure,
            ) as render:
                render_recovery_overview(
                    root,
                    ["mean", "vggt"],
                    ["recovery_linearpatch"],
                    "scannet_dev50",
                )

            self.assertEqual(render.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["bottom_row_xlabel"] == "Dropped Layers"
                    for call in render.call_args_list
                )
            )

            with patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                wraps=__import__(
                    "analysis.vis.recovery_tab",
                    fromlist=["render_strategy_metric_figure"],
                ).render_strategy_metric_figure,
            ) as detail_render:
                render_recovery_detail(
                    root,
                    ["mean", "vggt"],
                    ["recovery_linearpatch"],
                    "scannet_dev50",
                )

            self.assertEqual(detail_render.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["bottom_row_xlabel"] == "Dropped Layers"
                    and call.kwargs["all_axes_xlabel"]
                    for call in detail_render.call_args_list
                )
            )
            self.assertTrue(
                all(
                    call.kwargs["shared_xlabel"] is None
                    for call in (*render.call_args_list, *detail_render.call_args_list)
                )
            )

    def test_training_average_uses_only_training_datasets(self) -> None:
        def dataset_payload(value: float) -> dict:
            return {
                "upper_average": {"ape": value},
                "strategies": {
                    "recovery_linearpatch": {
                        "display_name": "Linear patch",
                        "points": {
                            "1": {
                                "available": True,
                                "drop_count": 1,
                                "metrics_mean": {"ape": value},
                            },
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "metrics_mean": {"ape": value},
                            },
                            "3": {
                                "available": True,
                                "drop_count": 3,
                                "metrics_mean": {"ape": value},
                            },
                            "4": {
                                "available": True,
                                "drop_count": 4,
                                "metrics_mean": {"ape": value},
                            },
                        },
                    }
                },
            }

        payload = {
            "datasets": {
                "scannet_dev50": dataset_payload(2.0),
                "nuscenes_train": dataset_payload(4.0),
                "scannet_test50": dataset_payload(100.0),
            }
        }
        figure = SimpleNamespace(axes=[])
        with (
            patch(
                "analysis.vis.recovery_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_tab.collect_recovery_eval_payload",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.add_recovery_comparison_strategies",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["vggt"],
                ["recovery_linearpatch"],
                "training average",
            )

        self.assertEqual(len(images), 1, status)
        averaged = render.call_args.kwargs["payload"]
        self.assertEqual(averaged["num_datasets"], 2)
        self.assertEqual(averaged["upper_average"]["ape"], 3.0)
        self.assertEqual(
            averaged["strategies"]["recovery_linearpatch"]["points"]["2"]
            ["metrics_mean"]["ape"],
            3.0,
        )

    def test_test_averages_separate_same_and_cross_dataset_generalization(
        self,
    ) -> None:
        payload = {
            "datasets": {
                "scannet_test50": {"upper_average": {"ape": 1.0}},
                "nuscenes_val": {"upper_average": {"ape": 3.0}},
                "7scenes_test": {"upper_average": {"ape": 10.0}},
                "waymo_validation": {"upper_average": {"ape": 30.0}},
            }
        }

        same_dataset = _dataset_group_average(
            payload, "same-dataset testing average"
        )
        cross_dataset = _dataset_group_average(
            payload, "cross-dataset testing average"
        )

        self.assertEqual(same_dataset["num_datasets"], 2)
        self.assertEqual(same_dataset["upper_average"]["ape"], 2.0)
        self.assertEqual(cross_dataset["num_datasets"], 2)
        self.assertEqual(cross_dataset["upper_average"]["ape"], 20.0)

    def test_training_average_no_recovery_uses_even_drop_counts(self) -> None:
        no_recovery = {
            "display_name": "No recovery",
            "points": {
                str(drop): {
                    "available": True,
                    "drop_count": drop,
                    "metrics_mean": {"ape": float(drop)},
                }
                for drop in (*range(1, 6), 26)
            },
        }
        payload = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {"ape": 0.0},
                    "strategies": {"no_recovery": no_recovery},
                }
            }
        }

        average = _dataset_group_average(payload, "training average")
        points = average["strategies"]["no_recovery"]["points"]
        self.assertEqual(list(points), ["2", "4", "26"])


if __name__ == "__main__":
    unittest.main()

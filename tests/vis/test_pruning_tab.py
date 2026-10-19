from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
from PIL import Image

from analysis import view as analysis_view
from analysis.metric_labels import SEVEN_METRIC_KEYS
from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.eval_plots import (
    STRATEGY_PLOT_COLORS,
    pruning_strategy_family,
)
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.shared_legend import (
    SHARED_LEGEND_CAPTION,
    _redistribute_vertical_legend,
)
from analysis.vis.pruning_tab import (
    PRUNING_PLAN_DETAIL_COLUMN_LABELS,
    PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
    _compose_pruning_detail,
    _compose_pruning_overview,
    _label_pruning_detail_figure,
    _mean_backbone_plot_payloads,
    _merge_replaceme_strategies,
    _replaceme_selection,
    build_pruning_tabs,
    discover_pruning_plan_backbones,
    discover_pruning_plan_scopes,
    render_pruning_plan_gallery,
    resolve_pruning_scope_datasets,
)


class PruningPlanTabTest(unittest.TestCase):
    def test_frameless_vertical_legend_preserves_both_horizontal_edges(self) -> None:
        source = np.full((30, 80, 3), 255, dtype=np.uint8)
        source[8:13, 3:10] = 0
        source[8:13, 70:77] = 0

        result = np.asarray(
            _redistribute_vertical_legend(
                Image.fromarray(source),
                50,
                expected_count=1,
                frameon=False,
            )
        )

        ink_columns = np.flatnonzero(np.any(result < 245, axis=(0, 2)))
        self.assertEqual(int(ink_columns[0]), 3)
        self.assertEqual(int(ink_columns[-1]), 76)

    def test_detail_uses_native_two_by_four_metric_layout(self) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 4)
        try:
            _label_pruning_detail_figure(
                figure,
                "VGGT · Same-Dataset\nGeneralization",
            )
            self.assertEqual(
                [axis.get_ylabel() for axis in figure.axes[:7]],
                [
                    "APE↓ (m)",
                    "ARE↓ (°)",
                    "RPEₜ↓ (m)",
                    "RPEᵣ↓ (°)",
                    "Acc.↓ (m)",
                    "Comp.↓ (m)",
                    "CD↓ (m)",
                ],
            )
            self.assertTrue(
                all(axis.yaxis.label.get_alpha() == 0.0 for axis in figure.axes[:7])
            )
            self.assertEqual(
                [
                    [text.get_text() for text in axis.texts if text.get_gid() == "metric-label-top"]
                    for axis in figure.axes[:7]
                ],
                [[label] for label in PRUNING_PLAN_DETAIL_COLUMN_LABELS],
            )
            self.assertTrue(
                all(
                    text.get_position() == (0.5, 0.98)
                    and text.get_ha() == "center"
                    for axis in figure.axes[:7]
                    for text in axis.texts
                    if text.get_gid() == "metric-label-top"
                )
            )
            self.assertEqual(
                [text.get_text() for text in axes[1, 3].texts],
                ["VGGT", "Same-Dataset Generalization"],
            )
            self.assertGreater(
                axes[1, 3].texts[0].get_position()[1],
                axes[1, 3].texts[1].get_position()[1],
            )
            self.assertTrue(
                all(not text.get_wrap() for text in axes[1, 3].texts)
            )
            self.assertFalse(axes[1, 3].axison)
        finally:
            plt.close(figure)

    def test_overview_composer_uses_dataset_rows_and_backbone_columns(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, "VGGT · Calibration\nDataset"),
            (tile, "VGGT · Same-Dataset\nGeneralization"),
            (tile, "VGGT · Cross-Dataset\nGeneralization"),
            (tile, "π³ · Calibration\nDataset"),
            (tile, "π³ · Same-Dataset\nGeneralization"),
            (tile, "π³ · Cross-Dataset\nGeneralization"),
            (tile, f"{SHARED_LEGEND_CAPTION}:training"),
            (tile, f"{SHARED_LEGEND_CAPTION}:testing"),
        ]
        with (
            mock.patch(
                "analysis.vis.pruning_tab.compose_sections",
                return_value=np.zeros((60, 100, 3), dtype=np.uint8),
            ) as compose,
            mock.patch(
                "analysis.vis.pruning_tab.append_shared_legends_right",
                return_value=np.zeros((60, 120, 3), dtype=np.uint8),
            ) as append,
        ):
            result = _compose_pruning_overview(items)

        section = compose.call_args.args[0][0]
        self.assertEqual(section.columns, 2)
        self.assertEqual(
            section.row_labels,
            [
                "Calibration\nDataset",
                "Same-Dataset\nGeneralization",
                "Cross-Dataset\nGeneralization",
            ],
        )
        self.assertEqual(section.column_labels, ["VGGT", "π³"])
        self.assertEqual([len(row) for row in section.fixed_rows], [2, 2, 2])
        self.assertTrue(section.row_labels_bold)
        self.assertTrue(section.column_labels_below)
        self.assertTrue(section.column_labels_bold)
        self.assertIsNone(section.row_banners)
        self.assertFalse(section.row_banners_align_content)
        append.assert_called_once()
        positioned_legends = append.call_args.args[1]
        self.assertFalse(append.call_args.kwargs["frameon"])
        self.assertEqual(len(positioned_legends), 2)
        self.assertTrue(all(len(item) == 4 for item in positioned_legends))
        self.assertLess(positioned_legends[0][1], positioned_legends[0][2])
        self.assertLess(positioned_legends[1][1], positioned_legends[1][2])
        self.assertLess(positioned_legends[0][2], positioned_legends[1][1])
        self.assertEqual(result.shape, (60, 120, 3))

    def test_testing_legend_spans_all_rows_without_calibration(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, "VGGT · Same-Dataset\nGeneralization"),
            (tile, "VGGT · Cross-Dataset\nGeneralization"),
            (tile, f"{SHARED_LEGEND_CAPTION}:testing|entries=7"),
        ]
        with (
            mock.patch(
                "analysis.vis.pruning_tab.compose_sections",
                return_value=np.zeros((60, 100, 3), dtype=np.uint8),
            ),
            mock.patch(
                "analysis.vis.pruning_tab.append_shared_legends_right",
                return_value=np.zeros((60, 120, 3), dtype=np.uint8),
            ) as append,
        ):
            _compose_pruning_overview(items)

        positioned_legends = append.call_args.args[1]
        self.assertFalse(append.call_args.kwargs["frameon"])
        self.assertEqual(len(positioned_legends), 1)
        testing_legend = positioned_legends[0]
        self.assertEqual(testing_legend[1], 0.0)
        self.assertGreater(testing_legend[2], testing_legend[1])
        self.assertEqual(testing_legend[3], 7)

    def test_overview_composer_can_replace_dataset_names_with_row_letters(
        self,
    ) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, f"VGGT · {dataset}")
            for dataset in (
                "Calibration\nDataset",
                "Same-Dataset\nGeneralization",
                "Cross-Dataset\nGeneralization",
            )
        ]
        with mock.patch(
            "analysis.vis.pruning_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_pruning_overview(items, simplify_dataset_names=True)

        self.assertEqual(
            compose.call_args.args[0][0].row_labels,
            ["(a)", "(b)", "(c)"],
        )
        self.assertEqual(
            compose.call_args.kwargs["outer_font_size"],
            PRUNING_PLAN_OVERVIEW_OUTER_FONT_SIZE,
        )

    def test_detail_composer_stacks_native_two_by_four_figures(self) -> None:
        items = [
            np.full((20, 30, 3), index, dtype=np.uint8)
            for index in range(2)
        ]
        items.extend([
            (np.full((10, 2400, 3), 50, dtype=np.uint8),
             f"{SHARED_LEGEND_CAPTION}:training"),
            (np.full((10, 2400, 3), 100, dtype=np.uint8),
             f"{SHARED_LEGEND_CAPTION}:testing"),
        ])
        with mock.patch(
            "analysis.vis.pruning_tab.compose_sections",
            return_value=np.zeros((1, 2400, 3), dtype=np.uint8),
        ) as compose:
            image = _compose_pruning_detail(gr.update(value=items, columns=7))
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[1], 2400)
        np.testing.assert_array_equal(image[:10], 50)
        np.testing.assert_array_equal(image[12:22], 100)
        sections = compose.call_args.args[0]
        self.assertEqual(len(sections), 2)
        self.assertTrue(all(section.columns == 1 for section in sections))
        self.assertTrue(all(len(section.cells) == 1 for section in sections))
        self.assertTrue(
            all(section.cells[0].caption == "" for section in sections)
        )

    def _write_fixture(
        self, root: Path, backbone_name: str = "vggt", metric_value: float = 2.0
    ) -> None:
        backbone = root / backbone_name
        backbone.mkdir(parents=True)
        strategies = {
            key: {"display_name": key, "points": {"2": "i1_j2"}}
            for key in (
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
            )
        }
        bundle = {
            "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
            "backbone": backbone_name,
            "dataset": "train_a",
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
            "strategies": strategies,
            "spans": {},
        }
        (backbone / "prune_plan_all.json").write_text(
            json.dumps(bundle), encoding="utf-8"
        )
        metrics = {
            "ape": metric_value,
            "are": metric_value,
            "rpe_trans": metric_value,
            "rpe_rot": metric_value,
            "acc": metric_value,
            "comp": metric_value,
            "chamfer_distance": metric_value,
        }
        for dataset in ("train_a", "test_a"):
            upper = backbone / dataset / "eval" / "upper"
            upper.mkdir(parents=True)
            EvalRecordV3(
                backbone=backbone_name,
                dataset=dataset,
                model_name="upper",
                average={key: 1.0 for key in metrics},
            ).save(upper / "eval_record_v3.json")
            summary = backbone / dataset / "eval" / "pruning" / "i1_j2"
            summary.mkdir(parents=True)
            (summary / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "drop_count": 2,
                        "num_scenes": 1,
                        "metrics_mean": metrics,
                    }
                ),
                encoding="utf-8",
            )

    def test_discovers_summary_scopes_and_resolves_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(discover_pruning_plan_backbones(root), ["vggt"])
            self.assertEqual(
                discover_pruning_plan_scopes(root, "vggt"),
                ["mean", "test_a", "train_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(root, "vggt", "mean"),
                ["train_a", "test_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(root, "vggt", "training"),
                ["train_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(
                    root, "vggt", "same_dataset_testing"
                ),
                ["test_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(
                    root, "vggt", "cross_dataset_testing"
                ),
                [],
            )

    def test_canonical_test_datasets_are_split_by_generalization_type(self) -> None:
        available = [
            "scannet_dev50",
            "nuscenes_train",
            "scannet_test50",
            "nuscenes_val",
            "7scenes_test",
            "waymo_validation",
        ]
        with (
            mock.patch(
                "analysis.vis.pruning_tab._result_datasets",
                return_value=available,
            ),
            mock.patch(
                "analysis.vis.pruning_tab._bundle_training_datasets",
                return_value=["scannet_dev50", "nuscenes_train"],
            ),
        ):
            self.assertEqual(
                resolve_pruning_scope_datasets(
                    Path("/unused"), "vggt", "same_dataset_testing"
                ),
                ["scannet_test50", "nuscenes_val"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(
                    Path("/unused"), "vggt", "cross_dataset_testing"
                ),
                ["7scenes_test", "waymo_validation"],
            )

    def test_renders_one_live_training_overview_plot_without_saved_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root, "vggt", ["training"]
            )
            self.assertEqual(len(gallery), 2, status)
            self.assertTrue(
                gallery[-1][1].startswith(
                    f"{SHARED_LEGEND_CAPTION}:training|entries="
                )
            )
            self.assertIn("plots=1/1", status)
            first_image = gallery[0][0] if isinstance(gallery[0], tuple) else gallery[0]
            self.assertIsInstance(first_image, Image.Image)
            self.assertGreater(first_image.info["axes_center_x"], 0.5)
            self.assertEqual(np.asarray(first_image).ndim, 3)
            self.assertFalse(any(root.rglob("*.png")))

    def test_mean_scope_renders_available_generalization_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(root, "vggt", ["mean"])
            self.assertEqual(len(gallery), 4, status)
            self.assertIn("vggt/training", status)
            self.assertIn("vggt/same_dataset_testing", status)
            self.assertIn("vggt/cross_dataset_testing", status)

    def test_training_average_warns_when_a_planned_dataset_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            (root / "vggt" / "train_a").rename(
                root / "vggt" / "scannet_dev50"
            )
            bundle_path = root / "vggt" / "prune_plan_all.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["dataset"] = "scannet_dev50,nuscenes_train"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

            _, status = render_pruning_plan_gallery(
                root, "vggt", ["training"]
            )

            self.assertIn("⚠️ WARNING", status)
            self.assertIn("1/2 datasets", status)
            self.assertIn("Missing: nuscenes_train", status)
            self.assertIn("Included: scannet_dev50", status)

    def test_detail_renders_two_groups_of_seven_metric_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab._label_pruning_detail_figure",
                wraps=_label_pruning_detail_figure,
            ) as label:
                gallery, status = render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    view="detail",
                )
            self.assertEqual(len(gallery), 4, status)
            self.assertEqual(
                [call.args[1] for call in label.call_args_list],
                ["VGGT", "VGGT"],
            )
            self.assertEqual(
                [item[1] for item in gallery[-2:]],
                [f"{SHARED_LEGEND_CAPTION}:training",
                 f"{SHARED_LEGEND_CAPTION}:testing"],
            )
            self.assertIn("plots=2/2", status)
            self.assertTrue(
                all(
                    isinstance(image[0] if isinstance(image, tuple) else image, np.ndarray)
                    for image in gallery
                )
            )
            self.assertTrue(any(isinstance(image, tuple) for image in gallery))

    def test_live_plots_default_to_drop_step_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.collect_pruning_eval_payload",
                wraps=__import__(
                    "analysis.vis.pruning_tab", fromlist=["collect_pruning_eval_payload"]
                ).collect_pruning_eval_payload,
            ) as collect:
                render_pruning_plan_gallery(root, "vggt", ["training"])
            self.assertEqual(collect.call_args.kwargs["drop_step"], 2)

    def test_renders_multiple_backbones_and_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, "vggt", 2.0)
            self._write_fixture(root, "pi3", 4.0)
            gallery, status = render_pruning_plan_gallery(
                root, ["vggt", "pi3", "mean"], ["training"]
            )
            self.assertEqual(len(gallery), 4, status)
            self.assertIn("mean/training", status)

    def test_mean_backbone_averages_only_shared_points(self) -> None:
        def payload(value: float, include_extra: bool = False) -> dict:
            points = {
                "2": {
                    "available": True,
                    "drop_count": 2,
                    "metrics_mean": {"ape": value},
                    "normalized_metric": {"ape": value / 10.0},
                }
            }
            if include_extra:
                points["4"] = {
                    "available": True,
                    "drop_count": 4,
                    "metrics_mean": {"ape": 9.0},
                }
            return {
                "upper_average": {"ape": 1.0},
                "strategies": {"CKA_double": {"points": points}},
            }

        mean = _mean_backbone_plot_payloads([payload(2.0, True), payload(4.0)])
        points = mean["strategies"]["CKA_double"]["points"]
        self.assertEqual(points["2"]["metrics_mean"]["ape"], 3.0)
        self.assertAlmostEqual(
            points["2"]["normalized_metric"]["ape"], 0.3
        )
        self.assertNotIn("4", points)

    def test_replaceme_uses_one_complete_calibration_winner_on_test_data(self) -> None:
        def point(value: float, marker: str) -> dict:
            return {
                "available": True,
                "drop_count": 2,
                "selector": marker,
                "metrics_mean": {metric: value + 10.0 for metric in SEVEN_METRIC_KEYS},
                "raw_metric": {metric: value + 20.0 for metric in SEVEN_METRIC_KEYS},
                "normalized_metric": {metric: value for metric in SEVEN_METRIC_KEYS},
            }

        calibration = {
            "strategies": {
                "replaceme_single": {"points": {"2": point(0.1, "cal-single")}},
                "replaceme_double": {"points": {"2": point(0.8, "cal-double")}},
            }
        }
        test_payload = {
            "strategies": {
                # Test performance favors Double, but calibration must keep Single.
                "replaceme_single": {"points": {"2": point(0.9, "test-single")}},
                "replaceme_double": {"points": {"2": point(0.2, "test-double")}},
                "shortgpt": {"points": {}},
            }
        }
        selection = _replaceme_selection(calibration)
        merged = _merge_replaceme_strategies(test_payload, selection)
        strategies = merged["strategies"]
        self.assertNotIn("replaceme_single", strategies)
        self.assertNotIn("replaceme_double", strategies)
        self.assertIn("shortgpt", strategies)
        merged_point = strategies["replaceme"]["points"]["2"]
        self.assertEqual(merged_point["selector"], "test-single")
        self.assertEqual(
            merged_point["calibration_selected_prune_strategy"],
            "replaceme_single",
        )
        self.assertEqual(
            set(merged_point["normalized_metric"].values()),
            {0.9},
        )

    def test_testing_baseline_plot_excludes_similarity_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root, "vggt", ["same_dataset_testing"]
                )
            self.assertEqual(render.call_count, 1)
            baseline_keys = render.call_args.kwargs["strategy_keys"]
            self.assertNotIn("CKA_single", baseline_keys)
            self.assertIn("retention_single", baseline_keys)
            self.assertNotIn("CKA_double", baseline_keys)
            self.assertIn("retention_double", baseline_keys)
            self.assertTrue(
                all(
                    pruning_strategy_family(key)
                    not in {"CKA", "cosine", "inv_mse", "cca"}
                    for key in baseline_keys
                    if pruning_strategy_family(key) != "retention"
                )
            )

    def test_overview_method_switch_selects_similarity_or_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    overview_method="similarity",
                )
                similarity = render.call_args.kwargs
                render.reset_mock()
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["same_dataset_testing"],
                    overview_method="baseline",
                )
                baseline = render.call_args.kwargs

            self.assertIn("CKA_single", similarity["strategy_keys"])
            self.assertIn("CKA_double", similarity["strategy_keys"])
            self.assertEqual(similarity["fixed_y_axis_upper"], 0.4)
            self.assertIsNone(baseline["fixed_y_axis_upper"])
            self.assertNotIn("CKA_single", baseline["strategy_keys"])
            self.assertNotIn("CKA_double", baseline["strategy_keys"])
            self.assertEqual(
                similarity["strategy_style_overrides"]["CKA_double"],
                {"linewidth": 3.2, "zorder": 10.0, "marker_zorder": 12.0},
            )
            self.assertNotIn(
                "color", similarity["strategy_style_overrides"]["CKA_double"]
            )
            self.assertEqual(
                similarity["strategy_style_overrides"]["CKA_single"],
                {"linewidth": 3.2, "zorder": 10.0, "marker_zorder": 12.0},
            )
            self.assertGreater(
                similarity["strategy_style_overrides"]["CKA_double"]["zorder"],
                similarity["strategy_style_overrides"]["cosine_double"].get(
                    "zorder", 3.0
                ),
            )
            self.assertEqual(
                similarity["strategy_style_overrides"]["CKA_double"]["linewidth"],
                3.2,
            )
            self.assertEqual(
                similarity["strategy_style_overrides"]["retention_double"]["linewidth"],
                3.2,
            )
            self.assertEqual(
                similarity["strategy_style_overrides"]["retention_single"]["linewidth"],
                3.2,
            )
            self.assertEqual(
                baseline["strategy_style_overrides"]["retention_double"]["linewidth"],
                3.2,
            )
            self.assertTrue(all(
                style["linewidth"] == (
                    3.2
                    if pruning_strategy_family(key) in {"retention", "CKA"}
                    else 2.4
                )
                for kwargs in (similarity, baseline)
                for key, style in kwargs["strategy_style_overrides"].items()
            ))
            expected_first_two_order = (
                "retention",
                "CKA",
                "cosine",
                "cca",
                "inv_mse",
                "replaceme",
                "shortgpt",
                "gardener",
                "simpler",
            )
            self.assertEqual(
                similarity["compact_structure_labels"], ("Single", "Double")
            )
            self.assertEqual(
                baseline["compact_structure_labels"], ("Single", "Multi")
            )
            self.assertEqual(
                similarity["compact_method_order"], expected_first_two_order
            )
            self.assertIsNone(baseline["compact_method_order"])
            for kwargs in (similarity, baseline):
                self.assertEqual(
                    kwargs["legend_fontsize"],
                    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
                )
                self.assertEqual(kwargs["legend_fontweight"], "bold")
                self.assertTrue(kwargs["draw_solid_before_dashed"])
                self.assertEqual(
                    kwargs["compact_method_labels"]["retention"],
                    "Ours",
                )
                self.assertEqual(
                    kwargs["compact_method_labels"]["CKA"],
                    "CKA",
                )
                self.assertTrue(kwargs["combine_compact_legend"])
                self.assertFalse(kwargs["show_figure_title"])
                self.assertFalse(kwargs["show_metric_titles"])
                self.assertEqual(
                    kwargs["shared_ylabel"],
                    "Normalized Degradation",
                )
                self.assertTrue(kwargs["y_tick_percentage"])
                self.assertFalse(kwargs["cap_y_at_normalization_max"])
                self.assertEqual(kwargs["subplot_width"], 7.4)
                self.assertEqual(kwargs["subplot_box_aspect"], 0.55)
                self.assertEqual(kwargs["figure_height_per_row"], 4.8)

    def test_overview_dodges_markers_only_for_training_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    [
                        "training",
                        "same_dataset_testing",
                        "cross_dataset_testing",
                    ],
                )

            self.assertEqual(
                [
                    call.kwargs["dodge_structure_markers"]
                    for call in render.call_args_list
                ],
                [True, False],
            )

    def test_overview_drop_units_is_only_on_the_last_backbone_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, "vggt")
            self._write_fixture(root, "pi3")
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root, ["vggt", "pi3"], ["mean"]
                )

            self.assertEqual(render.call_count, 4)
            self.assertEqual(
                [call.kwargs["shared_xlabel"] for call in render.call_args_list],
                [None] * 4,
            )
            self.assertEqual(
                [call.kwargs["bottom_row_xlabel"] for call in render.call_args_list],
                [None, "Dropped Layers", None, "Dropped Layers"],
            )
            self.assertEqual(
                [call.kwargs["shared_ylabel"] for call in render.call_args_list],
                ["Normalized Degradation"] * 4,
            )
            self.assertTrue(
                all(
                    call.kwargs["y_tick_percentage"]
                    for call in render.call_args_list
                )
            )

    def test_detail_renders_two_native_two_by_four_metric_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    view="detail",
                )
            self.assertEqual(render.call_count, 2)
            for call_index, call in enumerate(render.call_args_list):
                    kwargs = call.kwargs
                    self.assertEqual(
                        kwargs["legend_fontsize"],
                        PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE,
                    )
                    self.assertEqual(kwargs["legend_fontweight"], "bold")
                    self.assertEqual(
                        kwargs["compact_method_labels"],
                        {
                            "retention": "Ours",
                            "CKA": "CKA",
                            "cosine": "Cosine",
                            "inv_mse": "Inv_MSE",
                            "cca": "CCA",
                            "replaceme": "ReplaceMe",
                            "shortgpt": "ShortGPT",
                            "gardener": "Gardener",
                            "simpler": "SIMPLER",
                        },
                    )
                    self.assertTrue(kwargs["combine_compact_legend"])
                    self.assertEqual(
                        kwargs["compact_structure_labels"],
                        (
                            ("Single", "Double")
                            if call_index == 0
                            else ("Single", "Multi")
                        ),
                    )
                    self.assertEqual(
                        kwargs["compact_method_order"],
                        (
                            "retention",
                            "CKA",
                            "cosine",
                            "cca",
                            "inv_mse",
                            "replaceme",
                            "shortgpt",
                            "gardener",
                            "simpler",
                        ),
                    )
                    self.assertFalse(kwargs["show_figure_title"])
                    self.assertFalse(kwargs["show_metric_titles"])
                    self.assertTrue(kwargs["metric_labels_as_ylabels"])
                    self.assertIsNone(kwargs["shared_ylabel"])
                    self.assertFalse(kwargs["y_tick_percentage"])
                    self.assertEqual(kwargs["y_tick_max_decimals"], 2)
                    self.assertFalse(kwargs["cap_y_at_normalization_max"])
                    self.assertEqual(
                        kwargs["upper_anchor_bottom_margin_fraction"],
                        0.10,
                    )
                    self.assertEqual(
                        kwargs["metric_grid"],
                        (SEVEN_METRIC_KEYS[:4], SEVEN_METRIC_KEYS[4:]),
                    )
                    self.assertIsNone(kwargs["shared_xlabel"])
                    self.assertEqual(
                        kwargs["bottom_row_xlabel"], "Dropped Layers"
                    )
                    self.assertTrue(kwargs["all_axes_xlabel"])
                    self.assertEqual(kwargs["x_tick_step"], 4)
                    self.assertEqual(kwargs["subplot_width"], 7.4)
                    self.assertAlmostEqual(
                        kwargs["subplot_box_aspect"], 0.6875
                    )
                    self.assertAlmostEqual(kwargs["figure_height_per_row"], 6.0)
                    self.assertIsNone(kwargs["fixed_layout_margins"])
                    self.assertTrue(
                        all(
                            style["linewidth"] == (
                                3.2
                                if pruning_strategy_family(key)
                                in {"retention", "CKA"}
                                else 2.4
                            )
                            for key, style in kwargs[
                                "strategy_style_overrides"
                            ].items()
                        )
                    )

    def test_detail_method_switch_renders_only_selected_plot_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    view="detail",
                    detail_method="similarity",
                )
                self.assertEqual(render.call_count, 1)
                similarity_keys = render.call_args.kwargs["strategy_keys"]
                render.reset_mock()
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    view="detail",
                    detail_method="baseline",
                )
                self.assertEqual(render.call_count, 1)
                baseline_keys = render.call_args.kwargs["strategy_keys"]

            self.assertTrue(similarity_keys)
            self.assertTrue(baseline_keys)
            self.assertNotEqual(similarity_keys, baseline_keys)

    def test_cka_uses_red_and_replaceme_uses_green(self) -> None:
        self.assertEqual(STRATEGY_PLOT_COLORS["CKA"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["CKA_single"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["CKA_double"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["replaceme"], "#2ca02c")

    def test_builder_uses_single_images_for_overview_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self._write_fixture(root, "pi3")
            with gr.Blocks() as blocks:
                ui = build_pruning_tabs(blocks=blocks, output_root=root)
            self.assertIsInstance(ui["scope_cb"], gr.CheckboxGroup)
            self.assertIsInstance(ui["backbone_dd"], gr.CheckboxGroup)
            self.assertIn(("Mean", "mean"), ui["backbone_dd"].choices)
            self.assertIn(("π³", "pi3"), ui["backbone_dd"].choices)
            self.assertEqual(ui["backbone_dd"].value, ["pi3", "vggt"])
            self.assertEqual(
                ui["scope_cb"].value,
                [
                    "same_dataset_testing",
                    "cross_dataset_testing",
                ],
            )
            self.assertEqual(
                ui["detail"]["scope_cb"].value,
                ["cross_dataset_testing"],
            )
            self.assertNotIn(("Mean", "mean"), ui["scope_cb"].choices)
            self.assertNotIn(("Mean", "mean"), ui["detail"]["scope_cb"].choices)
            self.assertIn(
                ("Calibration\nDataset", "training"), ui["scope_cb"].choices
            )
            self.assertIn(
                ("Same-Dataset\nGeneralization", "same_dataset_testing"),
                ui["scope_cb"].choices,
            )
            self.assertIn(
                ("Cross-Dataset\nGeneralization", "cross_dataset_testing"),
                ui["scope_cb"].choices,
            )
            self.assertTrue(ui["scope_cb"].interactive)
            self.assertEqual(ui["scope_cb"].type, "value")
            self.assertIsInstance(ui["gallery"], gr.Image)
            self.assertEqual(ui["gallery"].type, "numpy")
            self.assertEqual(ui["overview_tab"].label, "Pruning")
            self.assertEqual(ui["detail_tab"].label, "Pruning")
            self.assertIsInstance(ui["detail"]["gallery"], gr.Image)
            self.assertEqual(ui["detail"]["gallery"].type, "numpy")
            self.assertIsInstance(ui["detail"]["detail_method"], gr.Radio)
            self.assertEqual(
                ui["detail"]["detail_method"].choices,
                [("Similarity", "similarity"), ("Baseline", "baseline")],
            )
            self.assertEqual(
                ui["detail"]["detail_method"].value,
                "baseline",
            )
            self.assertIsInstance(ui["overview_method"], gr.Radio)
            self.assertEqual(
                ui["overview_method"].choices,
                [("Similarity", "similarity"), ("Baseline", "baseline")],
            )
            self.assertEqual(ui["overview_method"].value, "baseline")
            self.assertNotIn("normalize_metrics", ui["detail"])
            self.assertNotIn("value_mode", ui)
            self.assertNotIn("normalize_metrics", ui)
            self.assertIsInstance(ui["merge_replaceme"], gr.Checkbox)
            self.assertTrue(ui["merge_replaceme"].value)
            self.assertIsInstance(ui["simplify_dataset_names"], gr.Checkbox)
            self.assertTrue(ui["simplify_dataset_names"].value)
            self.assertNotIn("simplify_dataset_names", ui["detail"])
            self.assertNotIn("overview_mode", ui)
            self.assertNotIn("overview_mode", ui["detail"])

    def test_overview_css_does_not_override_dynamic_columns(self) -> None:
        source = inspect.getsource(analysis_view._build_ui)
        selector = ".pruning-plan-overview-gallery [class*=\"grid\"]"
        self.assertNotIn(selector, source)


if __name__ == "__main__":
    unittest.main()

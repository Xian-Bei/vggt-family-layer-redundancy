from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np

from analysis import view as analysis_view
from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.prune_plan.eval_plots import STRATEGY_PLOT_COLORS
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.pruning_plan_tab import (
    PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
    _compose_pruning_detail,
    _compose_pruning_overview,
    _mean_backbone_plot_payloads,
    _merge_replaceme_strategies,
    build_pruning_plan_tab,
    discover_pruning_plan_backbones,
    discover_pruning_plan_scopes,
    render_pruning_plan_gallery,
    resolve_pruning_scope_datasets,
)


class PruningPlanTabTest(unittest.TestCase):
    def test_overview_composer_uses_backbone_rows_and_shared_scope_headings(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, "VGGT · Training Dataset"),
            tile,
            (tile, "VGGT · Testing Dataset"),
            tile,
            (tile, "π³ · Training Dataset"),
            tile,
            (tile, "π³ · Testing Dataset"),
            tile,
        ]
        with mock.patch(
            "analysis.vis.pruning_plan_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_pruning_overview(items)

        section = compose.call_args.args[0][0]
        self.assertEqual(section.columns, 4)
        self.assertEqual(section.row_labels, ["VGGT", "π³"])
        self.assertEqual(
            section.group_headings,
            [("Training Dataset", 2), ("Testing Dataset", 2)],
        )
        self.assertEqual([len(row) for row in section.fixed_rows], [4, 4])

    def test_detail_composer_preserves_four_by_four_metric_block(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [tile] * 16
        items[7] = (tile, "VGGT · Mean")
        image = _compose_pruning_detail(gr.update(value=items, columns=4))
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[1], 2400)

    def _write_fixture(
        self, root: Path, backbone_name: str = "vggt", metric_value: float = 2.0
    ) -> None:
        backbone = root / backbone_name
        backbone.mkdir(parents=True)
        strategies = {
            key: {"display_name": key, "points": {"2": "i1_j2"}}
            for key in (
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
                resolve_pruning_scope_datasets(root, "vggt", "testing"),
                ["test_a"],
            )

    def test_renders_two_live_overview_plots_without_saved_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root, "vggt", ["training"]
            )
            self.assertEqual(len(gallery), 2, status)
            self.assertIn("plots=2/2", status)
            first_image = gallery[0][0] if isinstance(gallery[0], tuple) else gallery[0]
            self.assertIsInstance(first_image, np.ndarray)
            self.assertEqual(first_image.ndim, 3)
            self.assertFalse(any(root.rglob("*.png")))

    def test_mean_scope_renders_training_and_testing_averages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(root, "vggt", ["mean"])
            self.assertEqual(len(gallery), 4, status)
            self.assertIn("vggt/training", status)
            self.assertIn("vggt/testing", status)

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

    def test_detail_renders_two_groups_of_eight_metric_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root,
                "vggt",
                ["training"],
                view="detail",
            )
            self.assertEqual(len(gallery), 16, status)
            self.assertIn("plots=16/16", status)
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
                "analysis.vis.pruning_plan_tab.collect_pruning_eval_payload",
                wraps=__import__(
                    "analysis.vis.pruning_plan_tab", fromlist=["collect_pruning_eval_payload"]
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
            self.assertEqual(len(gallery), 6, status)
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
                "strategies": {"ours_double": {"points": points}},
            }

        mean = _mean_backbone_plot_payloads([payload(2.0, True), payload(4.0)])
        points = mean["strategies"]["ours_double"]["points"]
        self.assertEqual(points["2"]["metrics_mean"]["ape"], 3.0)
        self.assertAlmostEqual(
            points["2"]["normalized_metric"]["ape"], 0.3
        )
        self.assertNotIn("4", points)

    def test_merges_replaceme_curves_with_pointwise_metric_minimum(self) -> None:
        def point(value: float) -> dict:
            return {
                "available": True,
                "drop_count": 2,
                "metrics_mean": {"ape": value},
                "raw_metric": {"ape": value + 1.0},
                "normalized_metric": {"ape": value + 3.0},
            }

        payload = {
            "strategies": {
                "replaceme_single": {"points": {"2": point(4.0)}},
                "replaceme_double": {"points": {"2": point(2.0)}},
                "shortgpt": {"points": {}},
            }
        }
        merged = _merge_replaceme_strategies(payload)
        strategies = merged["strategies"]
        self.assertNotIn("replaceme_single", strategies)
        self.assertNotIn("replaceme_double", strategies)
        self.assertIn("shortgpt", strategies)
        merged_point = strategies["replaceme"]["points"]["2"]
        self.assertEqual(merged_point["metrics_mean"]["ape"], 2.0)
        self.assertEqual(merged_point["raw_metric"]["ape"], 3.0)
        self.assertEqual(merged_point["normalized_metric"]["ape"], 5.0)

    def test_baseline_plot_excludes_ours_and_gt_single(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_plan_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(root, "vggt", ["training"])
            self.assertEqual(render.call_count, 2)
            baseline_keys = render.call_args_list[1].kwargs["strategy_keys"]
            self.assertNotIn("ours_single", baseline_keys)
            self.assertNotIn("gt_single", baseline_keys)
            self.assertIn("ours_double", baseline_keys)
            self.assertIn("gt_double", baseline_keys)

    def test_overview_shows_double_and_other_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root,
                "vggt",
                ["training"],
            )
            self.assertEqual(len(gallery), 2, status)
            with mock.patch(
                "analysis.vis.pruning_plan_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                )

            double = render.call_args_list[0].kwargs
            others = render.call_args_list[1].kwargs
            self.assertIn("ours_single", double["strategy_keys"])
            self.assertIn("ours_double", double["strategy_keys"])
            self.assertEqual(
                double["strategy_style_overrides"]["ours_single"],
                {
                    "linewidth": 1.6,
                    "alpha": 0.55,
                    "zorder": 1.5,
                    "markeredgewidth": 1.5,
                    "markersize": 49.0,
                    "marker_alpha": 0.55,
                    "marker_zorder": 1.7,
                },
            )
            self.assertNotIn(
                "color", double["strategy_style_overrides"]["ours_single"]
            )
            self.assertNotIn(
                "ours_double", double["strategy_style_overrides"]
            )
            self.assertEqual(others["strategy_style_overrides"], {})
            expected_first_two_order = (
                "gt",
                "ours",
                "cosine",
                "cca",
                "inv_mse",
                "replaceme",
                "shortgpt",
                "simpler",
            )
            self.assertIsNone(double["compact_structure_labels"])
            self.assertIsNone(others["compact_structure_labels"])
            self.assertEqual(
                double["compact_method_order"], expected_first_two_order
            )
            self.assertIsNone(others["compact_method_order"])
            for call_index, call in enumerate(render.call_args_list):
                kwargs = call.kwargs
                self.assertEqual(
                    kwargs["legend_fontsize"],
                    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
                )
                self.assertEqual(kwargs["legend_fontweight"], "bold")
                self.assertEqual(
                    kwargs["compact_method_labels"]["gt"],
                    "Retention",
                )
                self.assertEqual(
                    kwargs["compact_method_labels"]["ours"],
                    "CKA",
                )
                self.assertTrue(kwargs["combine_compact_legend"])
                self.assertFalse(kwargs["show_figure_title"])
                self.assertFalse(kwargs["show_metric_titles"])
                self.assertEqual(
                    kwargs["shared_ylabel"],
                    "Normalized Retention" if call_index == 0 else None,
                )
                self.assertEqual(kwargs["subplot_width"], 7.4)
                self.assertEqual(kwargs["subplot_box_aspect"], 0.704)
                self.assertEqual(kwargs["figure_height_per_row"], 5.8)

    def test_overview_drop_units_is_only_on_the_last_backbone_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, "vggt")
            self._write_fixture(root, "pi3")
            with mock.patch(
                "analysis.vis.pruning_plan_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root, ["vggt", "pi3"], ["mean"]
                )

            self.assertEqual(render.call_count, 8)
            self.assertTrue(
                all(
                    call.kwargs["shared_xlabel"] is None
                    for call in render.call_args_list[:4]
                )
            )
            self.assertTrue(
                all(
                    call.kwargs["shared_xlabel"] == "Dropped units"
                    for call in render.call_args_list[4:]
                )
            )
            self.assertEqual(
                [call.kwargs["shared_ylabel"] for call in render.call_args_list],
                [
                    "Normalized Retention",
                    None,
                    None,
                    None,
                    "Normalized Retention",
                    None,
                    None,
                    None,
                ],
            )

    def test_detail_normalization_toggle_never_disables_metric_mean_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.pruning_plan_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                    view="detail",
                    normalize_individual_metrics=False,
                )
            self.assertEqual(render.call_count, 16)
            expected_metric_labels = (
                "APE (m) ↓",
                "ARE (°) ↓",
                "RPE trans (m) ↓",
                "RPE rot (deg) ↓",
                "Acc (m) ↓",
                "Comp (m) ↓",
                "CD (m) ↓",
                "Normalized 7-metric mean ↓",
            )
            for group_start in (0, 8):
                for index in range(group_start, group_start + 7):
                    self.assertFalse(
                        render.call_args_list[index].kwargs["normalize_metrics"]
                    )
                self.assertTrue(
                    render.call_args_list[group_start + 7].kwargs[
                        "normalize_metrics"
                    ]
                )
                for index, expected_label in enumerate(expected_metric_labels):
                    kwargs = render.call_args_list[group_start + index].kwargs
                    gallery_column_index = index % 4
                    self.assertEqual(
                        kwargs["legend_fontsize"],
                        PRUNING_PLAN_DETAIL_SUBPLOT_FONT_SIZE,
                    )
                    self.assertEqual(kwargs["legend_fontweight"], "bold")
                    self.assertEqual(
                        kwargs["compact_method_labels"],
                        {
                            "gt": "Retention",
                            "ours": "CKA",
                            "cosine": "Cosine",
                            "inv_mse": "Inv_MSE",
                            "cca": "CCA",
                            "replaceme": "ReplaceMe",
                            "shortgpt": "ShortGPT",
                            "simpler": "SIMPLER",
                        },
                    )
                    self.assertTrue(kwargs["combine_compact_legend"])
                    self.assertEqual(
                        kwargs["compact_structure_labels"],
                        ("Multi",) if gallery_column_index == 0 else None,
                    )
                    self.assertEqual(
                        kwargs["compact_method_order"],
                        (
                            "gt",
                            "ours",
                            "cosine",
                            "cca",
                            "inv_mse",
                            "replaceme",
                            "shortgpt",
                            "simpler",
                        )
                        if gallery_column_index < 2
                        else None,
                    )
                    self.assertFalse(kwargs["show_figure_title"])
                    self.assertFalse(kwargs["show_metric_titles"])
                    self.assertEqual(kwargs["shared_ylabel"], expected_label)
                    self.assertEqual(kwargs["shared_xlabel"], "Dropped units")
                    self.assertEqual(kwargs["subplot_width"], 7.4)
                    self.assertEqual(kwargs["subplot_box_aspect"], 0.704)
                    self.assertEqual(kwargs["figure_height_per_row"], 5.8)

    def test_ours_cka_uses_red_and_replaceme_uses_green(self) -> None:
        self.assertEqual(STRATEGY_PLOT_COLORS["ours"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["ours_single"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["ours_double"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["replaceme"], "#2ca02c")

    def test_builder_uses_single_images_for_overview_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self._write_fixture(root, "pi3")
            with gr.Blocks() as blocks:
                ui = build_pruning_plan_tab(blocks=blocks, output_root=root)
            self.assertIsInstance(ui["scope_cb"], gr.CheckboxGroup)
            self.assertIsInstance(ui["backbone_dd"], gr.CheckboxGroup)
            self.assertIn(("Mean", "mean"), ui["backbone_dd"].choices)
            self.assertIn(("π³", "pi3"), ui["backbone_dd"].choices)
            self.assertEqual(ui["backbone_dd"].value, ["pi3", "vggt"])
            self.assertEqual(ui["scope_cb"].value, ["mean"])
            self.assertIn(("Mean", "mean"), ui["scope_cb"].choices)
            self.assertNotIn(
                ("Training Dataset", "training"), ui["scope_cb"].choices
            )
            self.assertNotIn(
                ("Testing Dataset", "testing"), ui["scope_cb"].choices
            )
            self.assertTrue(ui["scope_cb"].interactive)
            self.assertEqual(ui["scope_cb"].type, "value")
            self.assertIsInstance(ui["gallery"], gr.Image)
            self.assertEqual(ui["gallery"].type, "numpy")
            self.assertEqual(ui["overview_tab"].label, "Pruning Plan Overview")
            self.assertEqual(ui["detail_tab"].label, "Pruning Plan Detail")
            self.assertIsInstance(ui["detail"]["gallery"], gr.Image)
            self.assertEqual(ui["detail"]["gallery"].type, "numpy")
            self.assertIsInstance(
                ui["detail"]["normalize_metrics"], gr.Checkbox
            )
            self.assertFalse(ui["detail"]["normalize_metrics"].value)
            self.assertNotIn("value_mode", ui)
            self.assertNotIn("normalize_metrics", ui)
            self.assertIsInstance(ui["merge_replaceme"], gr.Checkbox)
            self.assertTrue(ui["merge_replaceme"].value)
            self.assertNotIn("overview_mode", ui)
            self.assertNotIn("overview_mode", ui["detail"])

    def test_overview_css_does_not_override_dynamic_columns(self) -> None:
        source = inspect.getsource(analysis_view._build_ui)
        selector = ".pruning-plan-overview-gallery [class*=\"grid\"]"
        self.assertNotIn(selector, source)


if __name__ == "__main__":
    unittest.main()

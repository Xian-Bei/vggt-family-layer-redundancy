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
    _mean_backbone_plot_payloads,
    _merge_replaceme_strategies,
    build_pruning_plan_tab,
    discover_pruning_plan_backbones,
    discover_pruning_plan_scopes,
    render_pruning_plan_gallery,
    resolve_pruning_scope_datasets,
)


class PruningPlanTabTest(unittest.TestCase):
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
                ["training", "testing", "test_a", "train_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(root, "vggt", "training"),
                ["train_a"],
            )
            self.assertEqual(
                resolve_pruning_scope_datasets(root, "vggt", "testing"),
                ["test_a"],
            )

    def test_renders_three_live_overview_plots_without_saved_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root, "vggt", ["training"]
            )
            self.assertEqual(len(gallery), 3, status)
            self.assertIn("plots=3/3", status)
            self.assertIsInstance(gallery[0], np.ndarray)
            self.assertEqual(gallery[0].ndim, 3)
            self.assertFalse(any(root.rglob("*.png")))

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
            self.assertTrue(all(isinstance(image, np.ndarray) for image in gallery))

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
            self.assertEqual(len(gallery), 9, status)
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
            self.assertEqual(render.call_count, 3)
            baseline_keys = render.call_args_list[2].kwargs["strategy_keys"]
            self.assertNotIn("ours_single", baseline_keys)
            self.assertNotIn("gt_single", baseline_keys)
            self.assertIn("ours_double", baseline_keys)
            self.assertIn("gt_double", baseline_keys)

    def test_overview_shows_single_double_and_other_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root,
                "vggt",
                ["training"],
            )
            self.assertEqual(len(gallery), 3, status)
            with mock.patch(
                "analysis.vis.pruning_plan_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_pruning_plan_gallery(
                    root,
                    "vggt",
                    ["training"],
                )

            single = render.call_args_list[0].kwargs
            double = render.call_args_list[1].kwargs
            others = render.call_args_list[2].kwargs
            self.assertTrue(single["strategy_keys"])
            self.assertTrue(
                all(key.endswith("_single") for key in single["strategy_keys"])
            )
            self.assertTrue(
                all(
                    style == {"linestyle": "-", "dashes": None}
                    for style in single["strategy_style_overrides"].values()
                )
            )
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

    def test_ours_cka_uses_red_and_replaceme_uses_green(self) -> None:
        self.assertEqual(STRATEGY_PLOT_COLORS["ours"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["ours_single"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["ours_double"], "#d62728")
        self.assertEqual(STRATEGY_PLOT_COLORS["replaceme"], "#2ca02c")

    def test_builder_uses_multi_select_checkboxes_and_auto_height_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self._write_fixture(root, "pi3")
            with gr.Blocks() as blocks:
                ui = build_pruning_plan_tab(blocks=blocks, output_root=root)
            self.assertIsInstance(ui["scope_cb"], gr.CheckboxGroup)
            self.assertIsInstance(ui["backbone_dd"], gr.CheckboxGroup)
            self.assertIn(("mean", "mean"), ui["backbone_dd"].choices)
            self.assertEqual(ui["backbone_dd"].value, ["mean", "pi3", "vggt"])
            self.assertTrue(ui["scope_cb"].interactive)
            self.assertEqual(ui["scope_cb"].type, "value")
            self.assertEqual(ui["gallery"].height, "auto")
            self.assertEqual(ui["gallery"].columns, 3)
            self.assertEqual(ui["overview_tab"].label, "Pruning Plan Overview")
            self.assertEqual(ui["detail_tab"].label, "Pruning Plan Detail")
            self.assertEqual(ui["detail"]["gallery"].columns, 4)
            self.assertIn(
                "pruning-plan-detail-gallery",
                ui["detail"]["gallery"].elem_classes,
            )
            self.assertIsInstance(
                ui["detail"]["normalize_metrics"], gr.Checkbox
            )
            self.assertTrue(ui["detail"]["normalize_metrics"].value)
            self.assertNotIn("value_mode", ui)
            self.assertNotIn("normalize_metrics", ui)
            self.assertIsInstance(ui["merge_replaceme"], gr.Checkbox)
            self.assertTrue(ui["merge_replaceme"].value)
            self.assertNotIn("overview_mode", ui)
            self.assertNotIn("overview_mode", ui["detail"])

    def test_overview_css_uses_three_columns(self) -> None:
        source = inspect.getsource(analysis_view._build_ui)
        selector = ".pruning-plan-overview-gallery [class*=\"grid\"]"
        self.assertIn(selector, source)
        overview_css = source.split(selector, 1)[1].split("}", 1)[0]
        self.assertIn(
            "grid-template-columns: repeat(3, minmax(0, 1fr)) !important;",
            overview_css,
        )


if __name__ == "__main__":
    unittest.main()

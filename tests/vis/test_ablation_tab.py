from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
from PIL import Image

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.schema.v3 import EvalRecordV3
from analysis.prune_plan.eval_plots import (
    PUBLICATION_COMPARISON_AXES_BOX_ASPECT,
    PUBLICATION_COMPARISON_SUBPLOT_HEIGHT,
    PUBLICATION_METRIC_SUBPLOT_WIDTH,
)
from analysis.vis.pruning_tab import (
    PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE,
    PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
    _compose_pruning_overview,
    _pruning_overview_figure_kwargs,
)
from analysis.vis.ablation_tab import (
    MAIN_PRUNE_PLAN_FILENAME,
    TOKEN_ABLATION_VARIANTS,
    TOKEN_ABLATION_Y_AXIS_UPPER,
    _token_strategy_styles,
    build_ablation_tab,
    discover_token_ablation_backbones,
    load_token_ablation_bundles,
    render_token_ablation_gallery,
)


class TokenAblationTabTest(unittest.TestCase):
    def test_shared_pruning_overview_style_has_matching_geometry_and_type(self) -> None:
        style = _pruning_overview_figure_kwargs()

        self.assertEqual(
            style["subplot_box_aspect"],
            PUBLICATION_COMPARISON_AXES_BOX_ASPECT,
        )
        self.assertEqual(
            style["figure_height_per_row"],
            PUBLICATION_COMPARISON_SUBPLOT_HEIGHT,
        )
        self.assertEqual(style["subplot_width"], PUBLICATION_METRIC_SUBPLOT_WIDTH)
        self.assertEqual(
            style["tick_label_fontsize"],
            PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(
            style["axis_label_fontsize"],
            PRUNING_PLAN_OVERVIEW_SUBPLOT_FONT_SIZE,
        )
        self.assertEqual(style["bottom_row_xlabel"], "Dropped Layers")
        self.assertIsNone(style["shared_xlabel"])
        self.assertEqual(PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE, 34)

    def test_ablation_legend_shortens_special_patch_mean(self) -> None:
        from analysis.vis.ablation_tab import (
            _TOKEN_METHOD_COLORS,
            _TOKEN_METHOD_LABELS,
        )

        self.assertEqual(_TOKEN_METHOD_LABELS["token_special_patch_mean"], "s-p")
        self.assertEqual(_TOKEN_METHOD_COLORS["token_patch"], "#ff7f0e")
        self.assertEqual(
            _TOKEN_METHOD_COLORS["token_special_patch_mean"],
            "#d62728",
        )

    def test_single_and_multi_use_the_same_color_intensity(self) -> None:
        styles = _token_strategy_styles()

        for family in (
            "retention",
            "token_all",
            "token_special",
            "token_patch",
            "token_special_patch_mean",
        ):
            self.assertEqual(styles[f"{family}_single"]["alpha"], 1.0)
            self.assertEqual(styles[f"{family}_single"]["marker_alpha"], 1.0)
            self.assertEqual(styles[f"{family}_single"]["linewidth"], 2.4)
            self.assertEqual(styles[f"{family}_double"]["linewidth"], 2.4)
            self.assertEqual(
                styles[f"{family}_single"]["color"],
                styles[f"{family}_double"]["color"],
            )

    def _write_bundle(
        self,
        path: Path,
        *,
        backbone: str,
        selector: str,
        token_region: str,
        include_retention: bool = False,
    ) -> None:
        start = int(selector.split("_", 1)[0][1:])
        end = int(selector.rsplit("j", 1)[1])
        drop = list(range(start, end + 1))
        sets = {}
        strategies = {}
        strategy_names = ["CKA_single", "CKA_double"]
        if include_retention:
            strategy_names = [
                "retention_single",
                "retention_double",
                *strategy_names,
            ]
        for strategy in strategy_names:
            plan_key = f"{strategy}_k{len(drop)}"
            sets[plan_key] = {
                "selector": plan_key,
                "eval_selector": selector,
                "drop": drop,
                "drop_count": len(drop),
            }
            strategies[strategy] = {
                "display_name": strategy,
                "points": {str(len(drop)): plan_key},
            }
        bundle = {
            "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
            "backbone": backbone,
            "dataset": "train_a",
            "num_units": 12,
            "unit_ids": [f"unit_{index}" for index in range(12)],
            "intervals": {},
            "sets": sets,
            "strategies": strategies,
            "spans": {},
            "extra": {"token_region": token_region},
        }
        path.write_text(json.dumps(bundle), encoding="utf-8")

    def _write_fixture(self, root: Path, backbone: str = "vggt_omega") -> None:
        backbone_root = root / backbone
        backbone_root.mkdir(parents=True)
        selectors = {
            "all": "i1_j2",
            "special": "i3_j4",
            "patch": "i5_j6",
            "special_patch_mean": "i7_j8",
        }
        filenames = {
            variant: filename
            for variant, _label, filename, _color in TOKEN_ABLATION_VARIANTS
        }
        regions = {
            "all": "all",
            "special": "special",
            "patch": "patch",
            "special_patch_mean": "special_patch_mean",
        }
        for variant, selector in selectors.items():
            self._write_bundle(
                backbone_root / filenames[variant],
                backbone=backbone,
                selector=selector,
                token_region=regions[variant],
            )
        self._write_bundle(
            backbone_root / MAIN_PRUNE_PLAN_FILENAME,
            backbone=backbone,
            selector=selectors["special_patch_mean"],
            token_region="special_patch_mean",
            include_retention=True,
        )

        metrics = {
            "ape": 2.0,
            "are": 2.0,
            "rpe_trans": 2.0,
            "rpe_rot": 2.0,
            "acc": 2.0,
            "comp": 2.0,
            "chamfer_distance": 2.0,
        }
        for dataset in ("train_a", "test_a"):
            upper = backbone_root / dataset / "eval" / "upper"
            upper.mkdir(parents=True)
            EvalRecordV3(
                backbone=backbone,
                dataset=dataset,
                model_name="upper",
                average={key: 1.0 for key in metrics},
            ).save(upper / "eval_record_v3.json")
            for offset, selector in enumerate(selectors.values()):
                summary = (
                    backbone_root
                    / dataset
                    / "eval"
                    / "pruning"
                    / selector
                )
                summary.mkdir(parents=True)
                (summary / "summary.json").write_text(
                    json.dumps(
                        {
                            "selector": selector,
                            "drop_count": 2,
                            "num_scenes": 1,
                            "metrics_mean": {
                                key: value + offset for key, value in metrics.items()
                            },
                        }
                    ),
                    encoding="utf-8",
                )

    def test_loads_all_dedicated_token_ablation_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            bundles = load_token_ablation_bundles(root, "vggt_omega")
            self.assertEqual(
                list(bundles),
                ["all", "special", "patch", "special_patch_mean"],
            )
            self.assertEqual(
                set(bundles["all"]["strategies"]),
                {"CKA_single", "CKA_double"},
            )
            self.assertEqual(
                bundles["special_patch_mean"]["extra"]["token_region"],
                "special_patch_mean",
            )

    def test_retention_loads_the_canonical_main_bundle_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            bundles = load_token_ablation_bundles(
                root,
                "vggt_omega",
                ["retention", "special_patch_mean"],
            )

            self.assertEqual(list(bundles), ["special_patch_mean", "retention"])
            self.assertEqual(
                bundles["special_patch_mean"]["extra"]["token_region"],
                "special_patch_mean",
            )
            self.assertEqual(
                bundles["retention"]["extra"]["token_region"],
                "special_patch_mean",
            )
            self.assertNotIn(
                "retention_single",
                bundles["special_patch_mean"]["strategies"],
            )
            self.assertIn(
                "retention_single",
                bundles["retention"]["strategies"],
            )

    def test_training_overview_renders_four_tokens_and_both_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_token_ablation_gallery(root)
            self.assertEqual(len(gallery), 2, status)
            image = gallery[0][0]
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(np.asarray(image).ndim, 3)
            legend, legend_caption = gallery[1]
            self.assertGreaterEqual(legend.shape[0], 7 * 50)
            self.assertEqual(
                legend_caption,
                "__shared_overview_legend__:training|entries=7",
            )
            self.assertIn(
                "methods=retention,all,special,patch,special_patch_mean",
                status,
            )
            self.assertIn("retention_single", status)
            self.assertIn("retention_double", status)
            self.assertIn("token_all_single", status)
            self.assertIn("token_special_double", status)
            self.assertIn("token_patch_single", status)
            self.assertIn("token_special_patch_mean_double", status)

    def test_training_overview_caps_y_axis_at_fifty_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with mock.patch(
                "analysis.vis.ablation_tab.render_strategy_metric_figure",
                return_value=None,
            ) as render:
                render_token_ablation_gallery(root)

            self.assertEqual(
                render.call_args.kwargs["fixed_y_axis_upper"],
                TOKEN_ABLATION_Y_AXIS_UPPER,
            )
            self.assertEqual(TOKEN_ABLATION_Y_AXIS_UPPER, 0.5)

    def test_selected_methods_limit_every_backbone_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            _gallery, status = render_token_ablation_gallery(
                root, ["special", "patch"]
            )
            self.assertIn("methods=special,patch", status)
            self.assertIn("token_special_single", status)
            self.assertIn("token_patch_double", status)
            self.assertNotIn("token_all_single", status)

            gallery, status = render_token_ablation_gallery(root, [])
            self.assertEqual(gallery, [])
            self.assertEqual(status, "Select at least one comparison method.")

            _gallery, status = render_token_ablation_gallery(root, ["retention"])
            self.assertIn("methods=retention", status)
            self.assertIn("retention_single", status)
            self.assertIn("retention_double", status)
            self.assertNotIn("token_special_patch_mean_single", status)

    def test_all_backbones_render_training_in_one_composite_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backbone in ("vggt", "pi3", "vggt_omega"):
                self._write_fixture(root, backbone)
            self.assertEqual(
                discover_token_ablation_backbones(root),
                ["vggt_omega", "pi3", "vggt"],
            )
            gallery, status = render_token_ablation_gallery(root)
            self.assertEqual(len(gallery), 4, status)
            self.assertEqual(
                [item[1] for item in gallery[:3]],
                [
                    "VGGT-Ω · Calibration\nDataset · training",
                    "π³ · Calibration\nDataset · training",
                    "VGGT · Calibration\nDataset · training",
                ],
            )
            composite = _compose_pruning_overview(gallery)
            self.assertGreater(composite.shape[1], composite.shape[0])
            self.assertIn("scope=training", status)
            self.assertIn("token_all_single", status)

    def test_composer_can_hide_dataset_names_without_reserving_label_space(
        self,
    ) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        items = [
            (tile, f"{backbone} · Calibration Dataset · training")
            for backbone in ("VGGT-Ω", "π³", "VGGT")
        ]
        with mock.patch(
            "analysis.vis.pruning_tab.compose_sections",
            return_value=np.zeros((20, 90, 3), dtype=np.uint8),
        ) as compose:
            _compose_pruning_overview(items, hide_dataset_names=True)

        section = compose.call_args.args[0][0]
        self.assertIsNone(section.row_labels)
        self.assertEqual(compose.call_args.kwargs["left_label_to_image_gap"], 0)

    def test_builder_creates_dedicated_overview_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with gr.Blocks() as blocks:
                ui = build_ablation_tab(
                    blocks=blocks,
                    output_root=root,
                )
            self.assertEqual(ui["tab"].label, "Ablation Token")
            self.assertNotIn("scope_cb", ui)
            self.assertIsInstance(ui["method_cb"], gr.CheckboxGroup)
            self.assertIsInstance(ui["hide_dataset_names"], gr.Checkbox)
            self.assertTrue(ui["hide_dataset_names"].value)
            self.assertEqual(
                ui["method_cb"].value,
                [
                    "all",
                    "special",
                    "patch",
                    "special_patch_mean",
                ],
            )
            self.assertIsInstance(ui["gallery"], gr.Image)
            self.assertEqual(ui["gallery"].type, "numpy")


if __name__ == "__main__":
    unittest.main()

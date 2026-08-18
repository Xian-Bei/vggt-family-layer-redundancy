from __future__ import annotations

from pathlib import Path
import inspect
import unittest

import gradio as gr

from analysis import view
from analysis.vis import build_heatmaps_tab


class HeatmapsTabTest(unittest.TestCase):
    def test_name_and_default_backbone_filter(self) -> None:
        class Controller:
            dataset = "mean"
            analysis_similarity_dir = Path("similarity")
            analysis_skip_dir = Path("skip")

        with gr.Blocks():
            ui = build_heatmaps_tab(
                controller=Controller(),
                dataset_choices=["min", "mean", "scannet_dev50"],
                backbone_choices=["pi3", "vggt", "vggt_omega"],
                initial_backbone="pi3",
                stage2_modes=["mixed_units"],
                stage2_scene_ids=["mean"],
                initial_stage2_scene="mean",
                stage2_metric_row_pose="pose",
                stage2_metric_row_point="point",
                stage2_token_row_special="special",
                stage2_token_row_camera="camera",
                stage2_token_row_patch="patch",
                stage2_token_row_all="all",
                stage2_token_row_mean="mean",
                heatmaps_resolve_filters_cb=lambda *_args: (
                    "absolute",
                    [],
                    ["all", "min"],
                    [],
                    ["all", "min"],
                    True,
                    True,
                    True,
                ),
                refresh_stage2_full_cb=lambda *_args: (),
            )

        self.assertEqual(ui["tab"].label, "Similarity Metrics")
        self.assertEqual(
            ui["stage2_backbone_filter"].value,
            ["vggt_omega", "pi3", "vggt"],
        )
        self.assertEqual(ui["stage2_dataset_dd"].value, "mean")
        self.assertNotIn("min", ui["stage2_dataset_dd"].choices)
        self.assertEqual(ui["stage2_scene_dd"].value, "mean")
        self.assertFalse(ui["stage2_scene_dd"].interactive)
        self.assertNotIn("min", ui["stage2_token_rows_filter"].choices)
        self.assertNotIn("min", ui["stage2_token_rows_filter"].value)
        self.assertIsInstance(ui["stage2_image"], gr.Image)
        self.assertEqual(ui["stage2_image"].type, "numpy")
        self.assertNotIn("stage2_normalize_skip_metrics", ui)

    def test_token_aggregation_and_blocks_do_not_generate_min(self) -> None:
        packed = [1.0, 0.5, 1.0]
        aggregated = view._similarity_maps_with_token_aggregations(
            {
                "mixed_units": {
                    "camera": {"cka": packed},
                    "patch": {"cka": packed},
                }
            }
        )

        self.assertIn("mean", aggregated["mixed_units"])
        self.assertNotIn("min", aggregated["mixed_units"])
        block_keys = view._stage2_block_keys_for_mode("mixed_units")
        self.assertEqual(len(block_keys), 8)
        self.assertFalse(any(key.endswith("_min") for key in block_keys))

    def test_remaining_labels_every_x_axis_and_only_leftmost_y_axis(self) -> None:
        render_source = inspect.getsource(view._prepare_stage2_render)
        self.assertEqual(render_source.count("show_ylabel=False"), 5)

        ui_source = inspect.getsource(view._build_ui)
        remaining_source = ui_source.split(
            "def _refresh_remaining_tab", 1
        )[1].split("def _refresh_brute_force", 1)[0]
        self.assertIn("show_xlabel=True", remaining_source)
        self.assertIn("show_xlabels=True", remaining_source)
        self.assertNotIn(
            "show_xlabel=backbone_index == len(available_backbones) - 1",
            remaining_source,
        )
        self.assertNotIn(
            "show_xlabels=backbone_index == len(available_backbones) - 1",
            remaining_source,
        )
        self.assertIn("show_ylabel=backbone_index == 0", remaining_source)
        self.assertIn("show_ylabel=False", remaining_source)

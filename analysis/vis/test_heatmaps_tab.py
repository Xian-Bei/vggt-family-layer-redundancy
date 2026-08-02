from __future__ import annotations

from pathlib import Path
import unittest

import gradio as gr

from analysis.vis.heatmaps_tab import build_heatmaps_tab


class HeatmapsTabTest(unittest.TestCase):
    def test_name_and_default_backbone_filter(self) -> None:
        class Controller:
            dataset = "mean"
            analysis_similarity_dir = Path("similarity")
            analysis_skip_dir = Path("skip")

        with gr.Blocks():
            ui = build_heatmaps_tab(
                controller=Controller(),
                dataset_choices=["mean"],
                backbone_choices=["pi3", "vggt", "vggt_omega"],
                initial_backbone="pi3",
                stage2_modes=["mixed_units"],
                stage2_scene_ids=["mean"],
                initial_stage2_scene="mean",
                stage2_metric_row_pose="pose",
                stage2_metric_row_depth="depth",
                stage2_metric_row_point="point",
                stage2_token_row_special="special",
                stage2_token_row_camera="camera",
                stage2_token_row_patch="patch",
                stage2_token_row_all="all",
                stage2_token_row_mean="mean",
                stage2_token_row_min="min",
                heatmaps_resolve_filters_cb=lambda *_args: (
                    "absolute",
                    [],
                    ["all"],
                    [],
                    ["all"],
                    True,
                    True,
                    True,
                ),
                refresh_stage2_full_cb=lambda *_args: (),
            )

        self.assertEqual(ui["tab"].label, "Metrics & Remaining")
        self.assertEqual(ui["stage2_backbone_filter"].value, ["vggt"])

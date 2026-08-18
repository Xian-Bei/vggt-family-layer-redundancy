from __future__ import annotations

import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np

from analysis.vis.brute_force_tab import (
    BRUTE_FORCE_DETAIL_COLUMN_LABELS,
    BRUTE_FORCE_OVERVIEW_COLUMN_LABELS,
    _compose_brute_force_detail,
    build_brute_force_tab,
    filter_brute_force_detail_datasets,
)


class BruteForceTabTest(unittest.TestCase):
    def test_detail_metric_labels_use_canonical_names_and_units(self) -> None:
        self.assertEqual(
            BRUTE_FORCE_DETAIL_COLUMN_LABELS,
            (
                "APE (m) ↓",
                "ARE (°) ↓",
                "RPE trans (m) ↓",
                "RPE rot (deg) ↓",
                "Acc (m) ↓",
                "Comp (m) ↓",
                "CD (m) ↓",
            ),
        )

    def test_detail_composer_uses_one_seven_metric_row_per_scope(self) -> None:
        tile = np.zeros((20, 30, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.brute_force_tab.compose_sections",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ) as compose:
            _compose_brute_force_detail(
                [
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                    gr.update(value=[tile] * 4, columns=4),
                ],
                ["vggt_omega"],
                ["mixed_units"],
                ["scannet_dev50", "nuscenes_train"],
            )

        sections = compose.call_args.args[0]
        self.assertEqual([section.columns for section in sections], [7, 7])
        self.assertEqual(
            [section.heading for section in sections],
            ["VGGT-Omega ScanNet", "VGGT-Omega nuScenes"],
        )
        self.assertEqual(
            sections[1].column_labels,
            list(BRUTE_FORCE_DETAIL_COLUMN_LABELS),
        )
        self.assertIsNone(sections[0].column_labels)
        self.assertTrue(sections[1].column_labels_below)
        self.assertTrue(all(section.row_labels is None for section in sections))
        self.assertEqual([len(section.cells) for section in sections], [7, 7])

    def test_detail_dataset_filter_is_shared_with_refresh_scopes(self) -> None:
        self.assertEqual(
            filter_brute_force_detail_datasets(
                [
                    "7scenes_test",
                    "nuscenes_test",
                    "nuscenes_train",
                    "scannet_dev50",
                    "waymo_validation",
                    "mean",
                ]
            ),
            ["scannet_dev50", "nuscenes_train"],
        )

    def test_overview_and_detail_browser_options(self) -> None:
        with gr.Blocks():
            ui = build_brute_force_tab(
                backbone_choices=["vggt_omega", "pi3", "vggt"],
                dataset_choices=[
                    "7scenes_test",
                    "scannet_dev50",
                    "nuscenes_test",
                    "nuscenes_train",
                    "waymo_validation",
                    "mean",
                ],
                initial_skip_datasets=["scannet_dev50", "nuscenes_train"],
                stage2_modes=["mixed_units"],
                refresh_overview_cb=lambda *_args: ((), (), (), True, True, True, ""),
                refresh_detail_cb=lambda *_args: (),
            )

        self.assertIsInstance(ui["overview_image"], gr.Image)
        self.assertEqual(ui["overview_image"].type, "numpy")
        self.assertIn("Mean metric sources", ui["overview_info"].value)
        self.assertIn("indoor: scannet_dev50", ui["overview_info"].value)
        self.assertIn("outdoor: nuscenes_train", ui["overview_info"].value)
        self.assertEqual(
            ui["detail_datasets"],
            ["scannet_dev50", "nuscenes_train"],
        )
        self.assertIsInstance(ui["detail_image"], gr.Image)
        self.assertEqual(ui["detail_image"].type, "numpy")


if __name__ == "__main__":
    unittest.main()

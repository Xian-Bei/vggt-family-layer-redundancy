from __future__ import annotations

import unittest

import gradio as gr
import numpy as np

from analysis.view import _gallery_item_with_context, _preferred_remaining_mode
from analysis.vis.remaining_tab import build_remaining_tab


class RemainingTabTest(unittest.TestCase):
    def test_one_preferred_skip_mode_is_selected(self) -> None:
        self.assertEqual(
            _preferred_remaining_mode(["default", "mixed_units"]),
            "mixed_units",
        )
        self.assertEqual(_preferred_remaining_mode(["custom"]), "custom")

    def test_existing_score_caption_is_flattened_for_gradio_gallery(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        item = _gallery_item_with_context((image, "V: Dice=0.9"), "vggt")

        self.assertIs(item[0], image)
        self.assertEqual(item[1], "vggt · V: Dice=0.9")
        gr.Gallery().postprocess([item])

    def test_tab_has_exactly_two_heatmap_rows(self) -> None:
        with gr.Blocks():
            ui = build_remaining_tab(
                dataset_choices=["min", "mean", "scannet_dev50"],
                initial_dataset="mean",
                refresh_remaining_cb=lambda *_args: ([], [], ""),
            )

        self.assertEqual(ui["tab"].label, "Remaining")
        self.assertEqual(ui["dataset"].value, "mean")
        self.assertEqual(ui["original_gallery"].label, "7-metric mean")
        self.assertEqual(
            ui["marginal_gallery"].label,
            "Remaining marginal 7-metric mean",
        )


if __name__ == "__main__":
    unittest.main()

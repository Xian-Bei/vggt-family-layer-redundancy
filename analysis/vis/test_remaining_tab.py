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
        item = _gallery_item_with_context((image, "detail"), "vggt")

        self.assertIs(item[0], image)
        self.assertEqual(item[1], "vggt · detail")
        gr.Gallery().postprocess([item])

    def test_tab_has_one_two_column_row_per_backbone(self) -> None:
        with gr.Blocks():
            ui = build_remaining_tab(
                dataset_choices=["min", "mean", "scannet_dev50"],
                backbone_choices=["vggt_omega", "pi3"],
                initial_dataset="mean",
                refresh_remaining_cb=lambda *_args: ([], [], True, True, ""),
            )

        self.assertEqual(ui["tab"].label, "Remaining")
        self.assertEqual(ui["dataset"].value, "mean")
        self.assertEqual(len(ui["galleries"]), 2)
        self.assertEqual(len(ui["backbone_rows"]), 2)
        self.assertTrue(all(gallery.columns == 2 for gallery in ui["galleries"]))
        self.assertTrue(
            all(
                gallery.label == "Original / marginal 7-metric mean"
                for gallery in ui["galleries"]
            )
        )


if __name__ == "__main__":
    unittest.main()

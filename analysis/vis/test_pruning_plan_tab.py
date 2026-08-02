from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import gradio as gr
import numpy as np

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.pruning_plan_tab import (
    build_pruning_plan_tab,
    discover_pruning_plan_backbones,
    discover_pruning_plan_scopes,
    render_pruning_plan_gallery,
    resolve_pruning_scope_datasets,
)


class PruningPlanTabTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        backbone = root / "vggt"
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
                "replaceme_single",
                "replaceme_double",
                "shortgpt",
                "simpler",
            )
        }
        bundle = {
            "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
            "backbone": "vggt",
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
            "ape": 2.0,
            "are": 2.0,
            "rpe_trans": 2.0,
            "rpe_rot": 2.0,
            "acc": 2.0,
            "comp": 2.0,
            "chamfer_distance": 2.0,
        }
        for dataset in ("train_a", "test_a"):
            upper = backbone / dataset / "eval" / "upper"
            upper.mkdir(parents=True)
            EvalRecordV3(
                backbone="vggt",
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

    def test_renders_four_live_plots_without_saved_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            gallery, status = render_pruning_plan_gallery(
                root, "vggt", ["training"]
            )
            self.assertEqual(len(gallery), 4, status)
            self.assertIn("plots=4/4", status)
            self.assertIsInstance(gallery[0][0], np.ndarray)
            self.assertEqual(gallery[0][0].ndim, 3)
            self.assertFalse(any(root.rglob("*.png")))

    def test_builder_uses_multi_select_checkboxes_and_auto_height_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with gr.Blocks() as blocks:
                ui = build_pruning_plan_tab(blocks=blocks, output_root=root)
            self.assertIsInstance(ui["scope_cb"], gr.CheckboxGroup)
            self.assertTrue(ui["scope_cb"].interactive)
            self.assertEqual(ui["scope_cb"].type, "value")
            self.assertEqual(ui["gallery"].height, "auto")
            self.assertEqual(ui["gallery"].columns, 2)


if __name__ == "__main__":
    unittest.main()

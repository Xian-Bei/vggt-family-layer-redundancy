from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np
import torch
from torch import nn

from analysis.vis.backbone_efficiency_tab import (
    _selectors_for_parameter_count,
    _series_from_strategy,
    build_backbone_efficiency_tab,
    count_parameter_series,
    render_backbone_efficiency,
)


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(3))
        self.blocks = nn.ModuleList([
            nn.Linear(1, 2, bias=False),
            nn.Linear(1, 5, bias=False),
            nn.Linear(1, 7, bias=False),
        ])
        self.head = nn.Parameter(torch.zeros(4))


class _FakeAdapter:
    def layer_unit_catalog(self, model):
        return [
            SimpleNamespace(unit_id=f"frame_{index}")
            for index in range(len(model.blocks))
        ]

    def block_for_unit(self, model, unit_id):
        return model.blocks[int(str(unit_id).rsplit("_", 1)[1])]


def _bundle() -> dict:
    return {
        "schema_version": "closed_interval_prune_plan_bundle_v5",
        "backbone": "fake",
        "num_units": 3,
        "intervals": {},
        "sets": {
            "p1": {
                "selector": "p1", "eval_selector": "i1_j1", "drop": [1],
                "dropped_unit_ids": ["frame_1"],
            },
            "p2": {
                "selector": "p2", "eval_selector": "i2_j2", "drop": [2],
                "dropped_unit_ids": ["frame_2"],
            },
        },
        "spans": {},
        "strategies": {
            "CKA_single": {"points": {"1": "p1"}},
        },
    }


class BackboneEfficiencyTabTest(unittest.TestCase):
    def test_parameter_count_is_exact_and_detects_conflicting_plans(self) -> None:
        model = _FakeModel()
        adapter = _FakeAdapter()
        accounting = model.blocks.parameters()
        series, warnings = count_parameter_series(
            model=model, adapter=adapter, accounting_parameters=accounting,
            bundle=_bundle(), selectors_by_drop={1: ["i1_j1"]},
        )
        self.assertEqual(series[0], (21, 14))
        self.assertEqual(series[1], (16, 9))
        self.assertEqual(warnings, [])

        conflict, warnings = count_parameter_series(
            model=model, adapter=adapter,
            accounting_parameters=model.blocks.parameters(), bundle=_bundle(),
            selectors_by_drop={1: ["i1_j1", "i2_j2"]},
        )
        self.assertEqual(conflict, {0: (21, 14)})
        self.assertIn("different parameter counts", warnings[0])

    def test_average_points_resolve_selector_from_bundle(self) -> None:
        payload = {
            "strategies": {
                "recovery_x_CKA_single": {
                    "points": {"1": {"available": True, "metrics_mean": {}}}
                },
                "no_recovery_single": {
                    "points": {"1": {"available": True, "metrics_mean": {}}}
                },
            }
        }
        selectors = _selectors_for_parameter_count(
            payload, _bundle(), method="recovery_x", strategy="CKA",
            structures=["single"],
        )
        self.assertEqual(selectors, {1: ["i1_j1", "i1_j1"]})

    def test_metric_series_reads_summary_fields(self) -> None:
        payload = {
            "strategies": {
                "curve": {
                    "points": {
                        "2": {
                            "available": True, "selector": "i1_j2",
                            "metrics_mean": {
                                "inference_time_ms": 12.5,
                                "infer_peak_vram_gb": 3.25,
                            },
                        }
                    }
                }
            }
        }
        xs, ys, selectors = _series_from_strategy(
            payload, "curve", "inference_time_ms"
        )
        self.assertEqual((xs, ys), ([2], [12.5]))
        self.assertEqual(selectors, {2: "i1_j2"})

    def test_render_uses_expected_line_semantics_and_upper_origin(self) -> None:
        point = {
            "available": True, "selector": "i1_j1",
            "metrics_mean": {
                "inference_time_ms": 80.0, "infer_peak_vram_gb": 4.0,
            },
        }
        payload = {
            "upper_average": {
                "inference_time_ms": 100.0, "infer_peak_vram_gb": 5.0,
            },
            "strategies": {
                "recovery_x_CKA_single": {"points": {"1": point}},
                "no_recovery_single": {
                    "points": {"1": {**point, "metrics_mean": {
                        "inference_time_ms": 75.0, "infer_peak_vram_gb": 3.8,
                    }}}
                },
            },
        }
        observed = {}

        def inspect_figure(fig):
            observed["param_styles"] = [line.get_linestyle() for line in fig.axes[0].lines]
            observed["time_styles"] = [line.get_linestyle() for line in fig.axes[1].lines]
            observed["time_x"] = [list(line.get_xdata()) for line in fig.axes[1].lines]
            return np.zeros((4, 4, 3), dtype=np.uint8)

        with (
            patch(
                "analysis.vis.backbone_efficiency_tab.load_prune_plan_bundle",
                return_value=_bundle(),
            ),
            patch(
                "analysis.vis.backbone_efficiency_tab._load_backbone_efficiency_payload",
                return_value=(payload, []),
            ),
            patch(
                "analysis.vis.backbone_efficiency_tab.compute_parameter_series_for_backbone",
                return_value=({0: (21, 14), 1: (16, 9)}, []),
            ),
            patch(
                "analysis.vis.backbone_efficiency_tab._figure_to_rgb",
                side_effect=inspect_figure,
            ),
        ):
            image, status = render_backbone_efficiency(
                "/unused", ["fake"], "recovery_x", "dataset", "CKA", ["single"]
            )

        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(observed["param_styles"], ["--", "-"])
        self.assertEqual(observed["time_styles"], ["-", "--"])
        self.assertEqual(observed["time_x"], [[0, 1], [0, 1]])
        self.assertIn("parameters were recalculated", status)

    def test_builder_exposes_recovery_remaining_controls(self) -> None:
        with (
            patch(
                "analysis.vis.backbone_efficiency_tab.discover_recovery_overview_options",
                return_value=(["fake"], ["recovery_x"]),
            ),
            patch(
                "analysis.vis.backbone_efficiency_tab.discover_recovery_overview_datasets",
                return_value=["training average", "dataset"],
            ),
        ):
            with gr.Blocks() as blocks:
                ui = build_backbone_efficiency_tab(
                    blocks=blocks, output_root="/unused"
                )
        self.assertEqual(ui["backbone_cb"].value, ["fake"])
        self.assertEqual(ui["prune_strategy_dd"].value, "CKA")
        self.assertEqual(
            ui["prune_strategy_dd"].choices,
            [("retention", "retention"), ("CKA", "CKA")],
        )
        self.assertEqual(ui["prune_structure_cb"].value, ["single", "double"])
        self.assertEqual(ui["method_dd"].value, "recovery_x")

    def test_rejects_cross_family_min_strategy(self) -> None:
        image, status = render_backbone_efficiency(
            "/unused", ["fake"], "recovery_x", "dataset", "min", ["single"]
        )
        self.assertIsNone(image)
        self.assertEqual(status, "Prune strategy must be retention or CKA.")


if __name__ == "__main__":
    unittest.main()

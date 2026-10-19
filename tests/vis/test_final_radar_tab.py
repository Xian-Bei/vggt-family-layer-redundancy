from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from analysis.eval.summary_v3 import METRIC_HEADERS
from analysis.schema.v3 import UNIFIED_METRIC_KEYS
from analysis.vis.display_names import backbone_display_name
from analysis.vis.final_radar_tab import (
    FINAL_RADAR_BUDGETS,
    _AXIS_LABEL_OFFSETS_POINTS,
    _collect_final_radar_points,
    _render_final_radar_figure,
    _shared_radar_bounds,
    build_final_radar_tab,
)
from analysis.vis.summary_tab import (
    _BLOCK_PARAMS_HEADER,
    _INFERENCE_TIME_HEADER,
    _PEAK_VRAM_HEADER,
    SummaryRow,
)


def _row(backbone: str, mode: str, value: float, sources=()) -> SummaryRow:
    metrics = {METRIC_HEADERS[key]: str(value) for key in UNIFIED_METRIC_KEYS}
    metrics.update(
        {
            _BLOCK_PARAMS_HEADER: str(value),
            _INFERENCE_TIME_HEADER: str(value),
            _PEAK_VRAM_HEADER: str(value),
        }
    )
    return SummaryRow(
        dataset="four-dataset testing average",
        backbone=backbone,
        mode=mode,
        metrics=metrics,
        pruning_sources=tuple(sources),
        raw_metrics={METRIC_HEADERS[key]: value for key in UNIFIED_METRIC_KEYS},
    )


class FinalRadarTabTest(unittest.TestCase):
    def test_final_axis_label_offsets_match_requested_adjustments(self) -> None:
        expected_offsets = (
            (0, 0),
            (-50, -25),
            (-60, 0),
            (-60, 0),
            (-50, 25),
            (0, 10),
            (50, 25),
            (60, 0),
            (60, 0),
            (50, -25),
        )
        self.assertEqual(_AXIS_LABEL_OFFSETS_POINTS, expected_offsets)

        points = [
            {
                "backbone": "vggt",
                "state": state,
                "budget": 10,
                "layer_count": 48 if state != "recovery" else 38,
                "raw_metric": {key: value for key in UNIFIED_METRIC_KEYS},
                "inference_time_ms": value,
                "peak_allocated_gib": value,
                "param_count": value,
            }
            for state, value in (("upper", 1.0), ("recovery", 2.0))
        ]
        with patch(
            "analysis.vis.final_radar_tab._AXIS_LABEL_OFFSETS_POINTS",
            ((0, 0),) * 10,
        ):
            baseline = _render_final_radar_figure("vggt", points)
        shifted = _render_final_radar_figure("vggt", points)
        assert baseline is not None and shifted is not None
        baseline.canvas.draw()
        shifted.canvas.draw()
        baseline_renderer = baseline.canvas.get_renderer()
        shifted_renderer = shifted.canvas.get_renderer()
        baseline_centers = np.asarray(
            [
                label.get_window_extent(baseline_renderer).get_points().mean(axis=0)
                for label in baseline.axes[0].get_xticklabels()
            ]
        )
        shifted_centers = np.asarray(
            [
                label.get_window_extent(shifted_renderer).get_points().mean(axis=0)
                for label in shifted.axes[0].get_xticklabels()
            ]
        )
        expected_pixels = np.asarray(expected_offsets) * shifted.dpi / 72.0
        np.testing.assert_allclose(
            shifted_centers - baseline_centers,
            expected_pixels,
            atol=0.5,
        )
        plt.close(baseline)
        plt.close(shifted)

    def test_collects_org_fastvggt_and_budget_recovery_for_each_backbone(self) -> None:
        rows = []
        for backbone, budget in FINAL_RADAR_BUDGETS:
            rows.extend(
                [
                    _row(backbone, "upper", 1.0),
                    _row(backbone, "upper+fastvggt_m0_r0p9", 1.5),
                    _row(
                        backbone,
                        f"i1_j{budget}_recovery_ghost_interval_sequential_autogroup",
                        2.0,
                        ("retention_single",),
                    ),
                    _row(
                        backbone,
                        f"i1_j{budget}_recovery_ghost_interval_sequential_autogroup"
                        "+fastvggt_m0_r0p9",
                        3.0,
                        ("retention_single",),
                    ),
                ]
            )
        rows.append(_row("vggt", "upper+quantvggt_w4a4", 1.75))
        rows.append(
            _row(
                "vggt",
                "i1_j10_recovery_ghost_interval_sequential_autogroup"
                "+quantvggt_w4a4",
                4.0,
                ("retention_single",),
            )
        )
        header = ["Dataset", "Backbone", "Mode", *rows[0].metrics]
        with (
            patch(
                "analysis.vis.final_radar_tab._load_live_eval_rows",
                return_value=(header, rows, [Path("/live")]),
            ),
            patch(
                "analysis.vis.final_radar_tab._add_dataset_average_rows",
                side_effect=lambda source, *_args: source,
            ),
            patch(
                "analysis.vis.final_radar_tab.load_prune_plan_bundle",
                side_effect=lambda path: {
                    "num_units": 36 if Path(path).name == "pi3" else 48
                },
            ),
        ):
            points, messages = _collect_final_radar_points(Path("/unused"))

        self.assertEqual(list(points), [item[0] for item in FINAL_RADAR_BUDGETS])
        for backbone, budget in FINAL_RADAR_BUDGETS:
            expected_states = ["upper", "full_fastvggt"]
            if backbone == "vggt":
                expected_states.append("full_quantvggt")
            expected_states.append("recovery")
            expected_states.append("recovery_fastvggt")
            if backbone == "vggt":
                expected_states.append("recovery_quantvggt")
            self.assertEqual(
                [point["state"] for point in points[backbone]],
                expected_states,
            )
            visible = [
                point for point in points[backbone] if point.get("plot", True)
            ]
            self.assertEqual(visible[-1]["budget"], budget)
            self.assertTrue(
                all(
                    not point["plot"]
                    for point in points[backbone]
                    if point["state"].startswith("recovery_")
                )
            )
            expected_layers = 36 if backbone == "pi3" else 48
            self.assertEqual(visible[0]["layer_count"], expected_layers)
            self.assertEqual(visible[-1]["layer_count"], expected_layers - budget)
        quantvggt = points["vggt"][2]
        self.assertIsNone(quantvggt["inference_time_ms"])
        self.assertIsNone(quantvggt["peak_allocated_gib"])
        bounds = _shared_radar_bounds(points)
        self.assertIsNotNone(bounds)
        assert bounds is not None
        np.testing.assert_allclose(
            bounds[1],
            [4.0] * 7 + [3.0, 3.0, 4.0],
        )
        self.assertEqual(len(messages), 3)

    def test_one_backbone_radar_compares_all_three_results(self) -> None:
        points = [
            {
                "backbone": "vggt",
                "state": state,
                "budget": 10,
                "layer_count": 48 if state != "recovery" else 38,
                "raw_metric": {key: value for key in UNIFIED_METRIC_KEYS},
                "inference_time_ms": value,
                "peak_allocated_gib": value,
                "param_count": value,
            }
            for state, value in (
                ("upper", 1.0),
                ("full_fastvggt", 1.5),
                ("recovery", 2.0),
            )
        ]
        fig = _render_final_radar_figure("vggt", points)
        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_title(), "VGGT")
        data_lines = [
            line
            for line in fig.axes[0].lines
            if line.get_label()
            in {"VGGT (48 layers)", "FastVGGT (48 layers)", "Ours (38 layers)"}
        ]
        self.assertEqual(len(data_lines), 3)
        np.testing.assert_allclose(
            data_lines[0].get_ydata()[:-1],
            [0.5] * 10,
        )
        np.testing.assert_allclose(
            data_lines[1].get_ydata()[:-1],
            [0.75] * 10,
        )
        np.testing.assert_allclose(
            data_lines[2].get_ydata()[:-1],
            [1.0] * 10,
        )
        self.assertEqual(
            [label.get_text() for label in fig.axes[0].get_xticklabels()][-3:],
            ["Latency\n(ms) ↓", "Memory\n(GiB) ↓", "Params.\n(M) ↓"],
        )
        self.assertEqual(
            [text.get_text() for text in fig.legends[1].texts],
            ["Trajectory Errors", "Geometric Errors", "System Efficiency"],
        )
        self.assertTrue(all(line.get_markersize() == 8.0 for line in data_lines))
        self.assertTrue(all(line.get_linewidth() == 5.2 for line in data_lines))
        self.assertTrue(
            all(
                label.get_fontsize() == 19.5
                for label in fig.axes[0].get_xticklabels()
            )
        )
        self.assertEqual(fig.axes[0].title.get_fontsize(), 31.2)
        self.assertEqual(len(fig.axes[0].texts), 30)
        self.assertTrue(
            all(text.get_fontsize() == 14.3 for text in fig.axes[0].texts)
        )
        self.assertEqual(
            [tick.get_pad() for tick in fig.axes[0].xaxis.majorTicks],
            [10, 30, 35, 35, 30, 17, 30, 35, 35, 30],
        )
        self.assertTrue(
            all(
                fig.axes[0].texts[index].get_position()[1] == 1.06
                for index in range(2, 30, 3)
            )
        )
        self.assertEqual(
            [
                label.get_horizontalalignment()
                for label in fig.axes[0].get_xticklabels()
            ],
            [
                "center", "left", "left", "left", "left",
                "center", "right", "right", "right", "right",
            ],
        )
        self.assertTrue(
            all(text.get_fontsize() == 20.15 for text in fig.legends[0].texts)
        )
        self.assertTrue(
            all(text.get_fontsize() == 18.85 for text in fig.legends[1].texts)
        )
        self.assertEqual(
            [text.get_text() for text in fig.axes[0].texts[:3]],
            ["0.67", "1.33", "2.00"],
        )
        self.assertEqual(fig.texts, [])
        plt.close(fig)

    def test_shared_bounds_cover_every_backbone_and_method(self) -> None:
        by_backbone = {
            backbone: [
                {
                    "raw_metric": {
                        key: float(backbone_index * 3 + method_index + 1)
                        for key in UNIFIED_METRIC_KEYS
                    },
                    "inference_time_ms": float(
                        backbone_index * 3 + method_index + 1
                    ),
                    "peak_allocated_gib": float(
                        backbone_index * 3 + method_index + 1
                    ),
                    "param_count": float(backbone_index * 3 + method_index + 1),
                }
                for method_index in range(3)
            ]
            for backbone_index, (backbone, _budget) in enumerate(
                FINAL_RADAR_BUDGETS
            )
        }
        by_backbone["vggt"].append(
            {
                "state": "full_quantvggt",
                "raw_metric": {key: 10.0 for key in UNIFIED_METRIC_KEYS},
                "inference_time_ms": 10.0,
                "peak_allocated_gib": 10.0,
                "param_count": 10.0,
            }
        )

        bounds = _shared_radar_bounds(by_backbone)

        self.assertIsNotNone(bounds)
        assert bounds is not None
        minima, maxima = bounds
        np.testing.assert_allclose(minima, [1.0] * 10)
        np.testing.assert_allclose(
            maxima,
            [10.0] * 7 + [9.0, 9.0, 10.0],
        )

    def test_combined_radar_contains_only_original_and_ours(self) -> None:
        points = []
        expected_labels = []
        for backbone, budget in FINAL_RADAR_BUDGETS:
            states = ["upper", "full_fastvggt", "recovery"]
            if backbone == "vggt":
                states.insert(2, "full_quantvggt")
            for index, state in enumerate(states, start=1):
                value = float(len(points) + index)
                points.append(
                    {
                        "backbone": backbone,
                        "state": state,
                        "budget": budget,
                        "layer_count": (
                            (36 if backbone == "pi3" else 48) - budget
                            if state == "recovery"
                            else (36 if backbone == "pi3" else 48)
                        ),
                        "raw_metric": {
                            key: value for key in UNIFIED_METRIC_KEYS
                        },
                        "inference_time_ms": value,
                        "peak_allocated_gib": value,
                        "param_count": value,
                    }
                )
                method = {
                    "upper": backbone_display_name(backbone),
                    "full_fastvggt": "FastVGGT",
                    "full_quantvggt": "QuantVGGT",
                    "recovery": "Ours",
                }[state]
                layer_count = 48 if backbone != "pi3" else 36
                if state == "recovery":
                    layer_count -= budget
                method = f"{method} ({layer_count} layers)"
                if state in {"upper", "recovery"}:
                    expected_labels.append(
                        method
                        if state == "upper"
                        else f"{backbone_display_name(backbone)} · {method}"
                    )

        fig = _render_final_radar_figure("all", points, combined=True)

        self.assertIsNotNone(fig)
        assert fig is not None
        self.assertEqual(fig.axes[0].get_title(), "All Backbones")
        self.assertEqual(
            [line.get_label() for line in fig.axes[0].lines],
            expected_labels,
        )
        self.assertEqual(len(fig.axes[0].lines), 6)
        ours_fills = [
            patch for patch in fig.axes[0].patches if patch.get_alpha() == 0.12
        ]
        self.assertEqual(len(ours_fills), 3)
        for backbone_index in range(3):
            original = fig.axes[0].lines[backbone_index * 2]
            ours = fig.axes[0].lines[backbone_index * 2 + 1]
            self.assertEqual(original.get_color(), ours.get_color())
        plt.close(fig)

    def test_builder_places_four_images_in_final_radar_tab(self) -> None:
        blank = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch(
            "analysis.vis.final_radar_tab.render_final_radars",
            return_value=(blank, blank, blank, blank, "ready"),
        ):
            with gr.Blocks() as blocks:
                ui = build_final_radar_tab(blocks=blocks, output_root=Path("/unused"))

        self.assertEqual(ui["tab"].id, "final_radar")
        self.assertEqual(len(ui["images"]), 4)
        self.assertEqual(
            [image.label for image in ui["images"]],
            [
                "VGGT · budget 10",
                "π³ · budget 16",
                "VGGT-Ω · budget 18",
                "All backbones · shared scale",
            ],
        )


if __name__ == "__main__":
    unittest.main()

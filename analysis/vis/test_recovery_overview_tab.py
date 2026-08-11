from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np

from analysis.prune_plan import PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
from analysis.schema.v3 import EvalRecordV3
from analysis.vis.recovery_overview_tab import (
    _minimum_strategy_plot_payload,
    build_recovery_overview_tab,
    discover_recovery_overview_datasets,
    discover_recovery_overview_options,
    discover_recovery_overview_prune_strategies,
    render_recovery_overview,
)


class RecoveryOverviewTabTest(unittest.TestCase):
    def test_min_uses_seven_metric_mean_then_copies_the_whole_winning_point(self) -> None:
        metrics = ("ape", "are", "rpe_trans", "rpe_rot", "acc", "comp", "chamfer_distance")

        def candidate(first: float, last: float, marker: str) -> dict:
            normalized = {metric: first for metric in metrics}
            normalized["chamfer_distance"] = last
            return {
                "strategies": {
                    "recovery_linearpatch": {
                        "points": {"2": {
                            "available": True,
                            "normalized_metric": normalized,
                            "raw_metric": {metric: marker for metric in metrics},
                        }}
                    },
                    "no_recovery": {
                        "points": {"2": {
                            "available": True,
                            "normalized_metric": {metric: 0.0 for metric in metrics},
                            "raw_metric": {
                                metric: f"{marker}_baseline" for metric in metrics
                            },
                        }}
                    },
                }
            }

        # GT wins on six individual metrics, but Ours has the lower seven-metric mean.
        merged = _minimum_strategy_plot_payload([
            ("gt_single", candidate(0.4, 1.0, "gt")),
            ("ours_single", candidate(0.5, 0.0, "ours")),
        ])
        point = merged["strategies"]["recovery_linearpatch"]["points"]["2"]
        self.assertEqual(point["min_prune_strategy"], "ours_single")
        self.assertEqual(set(point["raw_metric"].values()), {"ours"})
        baseline = merged["strategies"][
            "no_recovery__recovery_linearpatch"
        ]["points"]["2"]
        self.assertEqual(baseline["min_prune_strategy"], "ours_single")
        self.assertEqual(set(baseline["raw_metric"].values()), {"ours_baseline"})

    def _write_fixture(self, root: Path) -> None:
        backbone = root / "vggt"
        backbone.mkdir(parents=True)
        (backbone / "prune_plan_all.json").write_text(
            json.dumps(
                {
                    "schema_version": PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
                    "backbone": "vggt",
                    "dataset": "scannet_dev50",
                    "num_units": 4,
                    "intervals": {
                        "i1_j2": {
                            "selector": "i1_j2", "i": 1, "j": 2,
                            "start": 1, "end": 2, "drop": [1, 2],
                            "drop_count": 2,
                        }
                    },
                    "strategies": {
                        "ours_double": {
                            "display_name": "Ours double", "points": {"2": "i1_j2"}
                        },
                        "ours_single": {
                            "display_name": "Ours single", "points": {"2": "i1_j2"}
                        },
                        "gt_double": {
                            "display_name": "GT double", "points": {"2": "i1_j2"}
                        },
                        "gt_single": {
                            "display_name": "GT single", "points": {"2": "i1_j2"}
                        },
                        "cosine_double": {
                            "display_name": "Cosine double", "points": {"2": "i1_j2"}
                        },
                    },
                    "spans": {},
                }
            ),
            encoding="utf-8",
        )
        upper = backbone / "scannet_dev50" / "eval" / "upper"
        upper.mkdir(parents=True)
        EvalRecordV3(
            backbone="vggt",
            dataset="scannet_dev50",
            model_name="upper",
            average={"ape": 1.0, "are": 1.0, "rpe_trans": 1.0, "rpe_rot": 1.0,
                     "acc": 1.0, "comp": 1.0, "chamfer_distance": 1.0},
        ).save(upper / "eval_record_v3.json")
        for result_root in (
            backbone / "scannet_dev50" / "eval" / "pruning",
            backbone / "scannet_dev50" / "eval" / "recovery" / "i1_j2",
        ):
            target = (
                result_root / "recovery_linearpatch"
                if result_root.name == "i1_j2" and result_root.parent.name == "recovery"
                else result_root / "i1_j2"
            )
            target.mkdir(parents=True)
            (target / "summary.json").write_text(
                json.dumps(
                    {
                        "selector": "i1_j2",
                        "drop_count": 2,
                        "num_scenes": 1,
                        "metrics_mean": {"ape": 2.0, "are": 2.0, "rpe_trans": 2.0,
                                         "rpe_rot": 2.0, "acc": 2.0, "comp": 2.0,
                                         "chamfer_distance": 2.0},
                    }
                ),
                encoding="utf-8",
            )

    def test_discovers_controls_and_redraws_from_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["vggt"], ["recovery_linearpatch"]),
            )
            self.assertEqual(
                discover_recovery_overview_datasets(root),
                [
                    "training average",
                    "testing average",
                    "scannet_dev50",
                ],
            )
            images, status = render_recovery_overview(
                root, ["vggt"], ["recovery_linearpatch"], "scannet_dev50"
            )
            self.assertEqual(len(images), 1, status)
            self.assertEqual(images[0][0].ndim, 3)
            self.assertGreater(images[0][0].shape[0], 1000)
            self.assertIn("redrawn from current summaries", status)
            self.assertIn("/ scannet_dev50", status)

            _, average_status = render_recovery_overview(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "training average",
            )
            self.assertIn("⚠️ WARNING", average_status)
            self.assertIn("1/2 datasets", average_status)
            self.assertIn("Missing: nuscenes_train", average_status)

    def test_discovers_plan_before_evaluation_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_ghost.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                discover_recovery_overview_options(root),
                (["pi3"], ["recovery_ghost"]),
            )

    def test_discovers_and_renders_shared_selector_for_prune_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self.assertEqual(
                discover_recovery_overview_prune_strategies(root),
                ["ours_double", "ours_single", "gt_double", "gt_single"],
            )
            images, status = render_recovery_overview(
                root,
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
                prune_strategy="ours_double",
            )
            self.assertEqual(len(images), 1, status)
            self.assertIn("ours_double", status)

    def test_discovers_methods_from_recovery_plan_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "pi3"
            backbone.mkdir(parents=True)
            (backbone / "recovery_plan_all.json").write_text(
                json.dumps(
                    {
                        "methods": {
                            "recovery_ghost_interval": {},
                            "recovery_ghost_interval_sequential": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                discover_recovery_overview_options(root),
                (
                    ["pi3"],
                    [
                        "recovery_ghost_interval",
                        "recovery_ghost_interval_sequential",
                    ],
                ),
            )

    def test_builder_exposes_only_normalization_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            with gr.Blocks() as blocks:
                ui = build_recovery_overview_tab(blocks=blocks, output_root=root)
            self.assertNotIn("value_mode", ui)
            self.assertIsInstance(ui["prune_strategy_dd"], gr.Dropdown)
            self.assertEqual(
                ui["prune_strategy_dd"].choices,
                [
                    ("ours_double", "ours_double"),
                    ("ours_single", "ours_single"),
                    ("gt_double", "gt_double"),
                    ("gt_single", "gt_single"),
                    ("min", "min"),
                ],
            )
            self.assertEqual(ui["prune_strategy_dd"].value, "ours_double")
            self.assertEqual(ui["dataset_dd"].value, "training average")
            self.assertTrue(ui["normalize_metrics"].value)
            self.assertIn(("mean", "mean"), ui["backbone_cb"].choices)
            self.assertEqual(ui["backbone_cb"].value, ["mean", "vggt"])

    def test_mean_backbone_averages_shared_recovery_points(self) -> None:
        def plot_payload(value: float) -> dict:
            return {
                "upper_average": {"ape": value},
                "strategies": {
                    "recovery_linearpatch": {
                        "display_name": "Linear patch",
                        "points": {
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "metrics_mean": {"ape": value},
                            }
                        },
                    }
                },
            }

        figure = object()
        with (
            patch(
                "analysis.vis.recovery_overview_tab.discover_recovery_overview_options",
                return_value=(["vggt", "pi3"], ["recovery_linearpatch"]),
            ),
            patch(
                "analysis.vis.recovery_overview_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_overview_tab.collect_recovery_eval_payload",
                side_effect=[
                    {"datasets": {"scannet_dev50": plot_payload(2.0)}},
                    {"datasets": {"scannet_dev50": plot_payload(4.0)}},
                ],
            ),
            patch(
                "analysis.vis.recovery_overview_tab.add_recovery_comparison_strategies",
                side_effect=lambda recovery, *_: recovery,
            ),
            patch(
                "analysis.vis.recovery_overview_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_overview_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_overview_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["mean"],
                ["recovery_linearpatch"],
                "scannet_dev50",
            )

        self.assertEqual(len(images), 1, status)
        averaged = render.call_args.kwargs["payload"]
        self.assertEqual(averaged["num_backbones"], 2)
        self.assertEqual(averaged["upper_average"]["ape"], 3.0)
        self.assertEqual(
            averaged["strategies"]["recovery_linearpatch"]["points"]["2"]
            ["metrics_mean"]["ape"],
            3.0,
        )
        self.assertIn("backbones=vggt,pi3", status)

    def test_recovery_plot_does_not_prepend_upper_anchor_to_strategies(self) -> None:
        payload = {
            "datasets": {
                "scannet_dev50": {
                    "strategies": {
                        "gt_drop": {"points": {}},
                        "recovery_linearpatch": {"points": {}},
                    },
                }
            }
        }
        figure = object()
        with (
            patch(
                "analysis.vis.recovery_overview_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_overview_tab.collect_recovery_eval_payload",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_overview_tab.add_recovery_comparison_strategies",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_overview_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_overview_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_overview_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["vggt"],
                ["recovery_linearpatch"],
                "scannet_dev50",
            )

        self.assertEqual(len(images), 1, status)
        self.assertEqual(
            render.call_args.kwargs["strategy_keys"],
            ["recovery_linearpatch"],
        )
        self.assertFalse(render.call_args.kwargs["include_upper_anchor"])
        self.assertEqual(render.call_args.kwargs["subplot_width"], 7.4)
        self.assertTrue(
            render.call_args.kwargs["always_normalize_metric_mean"]
        )

    def test_training_average_uses_only_training_datasets(self) -> None:
        def dataset_payload(value: float) -> dict:
            return {
                "upper_average": {"ape": value},
                "strategies": {
                    "recovery_linearpatch": {
                        "display_name": "Linear patch",
                        "points": {
                            "1": {
                                "available": True,
                                "drop_count": 1,
                                "metrics_mean": {"ape": value},
                            },
                            "2": {
                                "available": True,
                                "drop_count": 2,
                                "metrics_mean": {"ape": value},
                            },
                            "3": {
                                "available": True,
                                "drop_count": 3,
                                "metrics_mean": {"ape": value},
                            },
                            "4": {
                                "available": True,
                                "drop_count": 4,
                                "metrics_mean": {"ape": value},
                            },
                        },
                    }
                },
            }

        payload = {
            "datasets": {
                "scannet_dev50": dataset_payload(2.0),
                "nuscenes_train": dataset_payload(4.0),
                "scannet_test50": dataset_payload(100.0),
            }
        }
        figure = object()
        with (
            patch(
                "analysis.vis.recovery_overview_tab.collect_pruning_eval_payload",
                return_value={},
            ),
            patch(
                "analysis.vis.recovery_overview_tab.collect_recovery_eval_payload",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_overview_tab.add_recovery_comparison_strategies",
                return_value=payload,
            ),
            patch(
                "analysis.vis.recovery_overview_tab.render_strategy_metric_figure",
                return_value=figure,
            ) as render,
            patch(
                "analysis.vis.recovery_overview_tab._figure_to_rgb",
                return_value=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            patch("analysis.vis.recovery_overview_tab.plt.close"),
        ):
            images, status = render_recovery_overview(
                Path("/unused"),
                ["vggt"],
                ["recovery_linearpatch"],
                "training average",
            )

        self.assertEqual(len(images), 1, status)
        averaged = render.call_args.kwargs["payload"]
        self.assertEqual(averaged["num_datasets"], 2)
        self.assertEqual(averaged["upper_average"]["ape"], 3.0)
        self.assertEqual(
            averaged["strategies"]["recovery_linearpatch"]["points"]["2"]
            ["metrics_mean"]["ape"],
            3.0,
        )

    def test_training_average_no_recovery_uses_even_drop_counts(self) -> None:
        no_recovery = {
            "display_name": "No recovery",
            "points": {
                str(drop): {
                    "available": True,
                    "drop_count": drop,
                    "metrics_mean": {"ape": float(drop)},
                }
                for drop in (*range(1, 6), 26)
            },
        }
        payload = {
            "datasets": {
                "scannet_dev50": {
                    "upper_average": {"ape": 0.0},
                    "strategies": {"no_recovery": no_recovery},
                }
            }
        }

        from analysis.vis.recovery_overview_tab import _dataset_group_average

        average = _dataset_group_average(payload, "training average")
        points = average["strategies"]["no_recovery"]["points"]
        self.assertEqual(list(points), ["2", "4", "26"])


if __name__ == "__main__":
    unittest.main()

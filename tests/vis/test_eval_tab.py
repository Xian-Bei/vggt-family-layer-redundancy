from pathlib import Path
import json
import tempfile
import unittest

from PIL import Image

from analysis.vis.eval_tab import (
    VisualizationMethod,
    _discover_scenes,
    _discover_visualization_methods,
    _gallery_for_scene,
    _sort_scenes_by_replaceme_advantage,
    _visualization_metrics_table,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        Image.new("RGB", (32, 24), "white").save(path)
    else:
        path.touch()


class EvalTabTest(unittest.TestCase):
    def test_discovers_budget_and_original_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            eval_root = tmp_path / "vggt" / "scannet_test50" / "eval"
            recovery = "recovery_ghost_interval_sequential_autogroup"
            selector = "i8_j17"
            roots = [
                eval_root / "recovery" / selector / recovery,
                eval_root / "testtime" / recovery / "fastvggt_m0_r0p9" / selector,
                eval_root / "testtime" / recovery / "quantvggt_w4a4" / selector,
                eval_root / "upper",
                eval_root / "upper_fastvggt_m0_r0p9",
                eval_root / "upper_quantvggt_w4a4",
            ]
            for root in roots:
                _touch(root / "scene0000_00" / "plot.png")

            methods = _discover_visualization_methods(tmp_path, "vggt", "scannet_test50")

            self.assertEqual(
                [method.label for method in methods],
                [
                    "Budget 10 · No test-time",
                    "Budget 10 + FastVGGT",
                    "Budget 10 + QuantVGGT (W4A4)",
                    "Original · No test-time",
                    "Original + FastVGGT",
                    "Original + QuantVGGT (W4A4)",
                ],
            )
            first_row = _gallery_for_scene(methods, "scene0000_00", row=1)
            second_row = _gallery_for_scene(methods, "scene0000_00", row=2)
            self.assertEqual(
                [caption.split(" · ", 1)[0] for _image, caption in first_row],
                ["FastVGGT", "QuantVGGT", "Ours", "Original"],
            )
            self.assertEqual(
                [caption.split(" · ", 1)[0] for _image, caption in second_row],
                ["Ours + FastVGGT", "Ours + QuantVGGT"],
            )

    def test_sorts_scenes_by_replaceme_minus_ours_seven_metric_sum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replaceme_root = root / "replaceme"
            ours_root = root / "ours"
            metric_keys = (
                "ape",
                "are",
                "rpe_trans",
                "rpe_rot",
                "acc",
                "comp",
                "cd",
            )
            # gain_big: 7 * (3 - 1) = 14; gain_small: 7 * (2 - 1) = 7.
            for scene, replaceme_value, our_value in (
                ("gain_small", 2.0, 1.0),
                ("gain_big", 3.0, 1.0),
            ):
                for run_root, value in (
                    (replaceme_root, replaceme_value),
                    (ours_root, our_value),
                ):
                    path = run_root / scene / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({key: value for key in metric_keys}),
                        encoding="utf-8",
                    )
            replaceme = VisualizationMethod(
                "replaceme:recovery_replaceme_ls:selector",
                "ReplaceMe",
                replaceme_root,
            )
            ours = VisualizationMethod(
                "budget:none:recovery:selector", "Ours", ours_root
            )
            self.assertEqual(
                _sort_scenes_by_replaceme_advantage(
                    ["gain_small", "missing", "gain_big"],
                    {"vggt": [ours, replaceme]},
                ),
                ["gain_big", "gain_small", "missing"],
            )

    def test_original_assets_fall_back_to_reference_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_upper = root / "main" / "vggt" / "scannet_test50" / "eval" / "upper"
            reference_upper = (
                root / "reference" / "vggt" / "scannet_test50" / "eval" / "upper"
            )
            main_upper.mkdir(parents=True)
            (main_upper / "eval_record_v3.json").write_text("{}", encoding="utf-8")
            _touch(reference_upper / "scene0000_00" / "plot.png")

            methods = _discover_visualization_methods(
                root / "main",
                "vggt",
                "scannet_test50",
                reference_output_root=root / "reference",
            )
            original = next(method for method in methods if method.key == "original:none")

            self.assertEqual(original.run_root, main_upper.resolve())
            self.assertEqual(original.asset_root, reference_upper.resolve())
            self.assertEqual(len(_gallery_for_scene([original], "scene0000_00", row=1)), 1)

    def test_discovers_selected_replaceme_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone_root = root / "vggt"
            eval_root = backbone_root / "scannet_test50" / "eval"
            selector = "i8_j17"
            backbone_root.mkdir(parents=True)
            (backbone_root / "replaceme_selection_budget10.json").write_text(
                json.dumps(
                    {
                        "budget": 10,
                        "winner": {
                            "strategy": "replaceme_single",
                            "selector": selector,
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_root = (
                eval_root
                / "recovery"
                / selector
                / "recovery_replaceme_ls"
            )
            _touch(run_root / "scene0000_00" / "vis" / "pred" / "viser_points.npz")

            methods = _discover_visualization_methods(
                root, "vggt", "scannet_test50"
            )
            replaceme = [
                method for method in methods
                if method.key.startswith("replaceme:")
            ]

            self.assertEqual(len(replaceme), 1)
            self.assertEqual(
                replaceme[0].label,
                "Budget 10 · ReplaceMe (Single)",
            )
            self.assertEqual(replaceme[0].run_root, run_root.resolve())
            self.assertFalse(replaceme[0].show_in_2d)
            self.assertEqual(
                _gallery_for_scene(methods, "scene0000_00", row=2),
                [],
            )

    def test_nested_scene_and_viser_scene_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            _touch(run_root / "chess" / "seq-03" / "plot.png")
            _touch(run_root / "office" / "seq-02" / "vis" / "pred" / "viser_points.npz")

            self.assertEqual(
                _discover_scenes(run_root),
                ("chess/seq-03", "office/seq-02"),
            )

    def test_formats_all_seven_visualization_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            scene = "scene"
            metrics = {
                "ape": 0.1,
                "are": 0.2,
                "rpe_trans": 0.3,
                "rpe_rot": 0.4,
                "acc": 0.5,
                "comp": 0.6,
                "cd": 0.7,
            }
            path = run_root / scene / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(metrics), encoding="utf-8")
            method = VisualizationMethod(
                "budget:none:recovery:selector", "Ours", run_root
            )
            table = _visualization_metrics_table([method], scene)

            self.assertIn("<th>Ours</th>", table)
            for label, value in (
                ("APE", "0.10"),
                ("ARE", "0.20"),
                ("RPEₜ", "0.30"),
                ("RPEᵣ", "0.40"),
                ("Acc", "0.50"),
                ("Comp", "0.60"),
                ("CD", "0.70"),
            ):
                self.assertIn(f"<tr><th>{label}</th><td>{value}</td></tr>", table)


if __name__ == "__main__":
    unittest.main()

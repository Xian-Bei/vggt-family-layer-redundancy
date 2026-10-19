from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from analysis.datasets.adaptor.scannet import ScanNetAdaptor
from data.extract_nuscenes import load_excluded_scene_names, select_train_val_scenes


class CalibrationPoolTest(unittest.TestCase):
    def test_scannet_manifest_controls_prefix_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scene_id in ("scene0000_00", "scene0001_00", "scene0002_00"):
                color = root / scene_id / "color"
                color.mkdir(parents=True)
                (color / "0.jpg").write_bytes(b"image")
            (root / "selected_scenes.json").write_text(
                json.dumps({"scene_ids": ["scene0002_00", "scene0000_00"]}),
                encoding="utf-8",
            )

            scenes = ScanNetAdaptor._ordered_scenes(root)

            self.assertEqual([scene.scene_id for scene in scenes], ["scene0002_00", "scene0000_00"])

    def test_nuscenes_extra_pool_excludes_main_manifest_and_keeps_rng_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "selected_scenes.json"
            manifest_path.write_text(
                json.dumps({"splits": {"train": ["scene-0001"], "val": ["scene-1001"]}}),
                encoding="utf-8",
            )
            excluded = load_excluded_scene_names([manifest_path])
            nusc = SimpleNamespace(
                scene=[{"name": f"scene-{index:04d}"} for index in range(1, 7)]
            )
            official = {
                "train": [f"scene-{index:04d}" for index in range(1, 7)],
                "val": ["scene-1001"],
            }
            with mock.patch("data.extract_nuscenes.create_splits_scenes", return_value=official):
                selected = select_train_val_scenes(
                    nusc,
                    num_scenes_per_split=50,
                    seed=7,
                    active_splits=("train",),
                    split_counts={"train": 3},
                    excluded_names=excluded,
                    fail_if_insufficient=True,
                    preserve_sample_order=True,
                )

            self.assertEqual(len(selected["train"]), 3)
            self.assertNotIn("scene-0001", selected["train"])
            self.assertEqual(selected["val"], [])
            self.assertEqual(len(selected["train"]), len(set(selected["train"])))

    def test_nuscenes_extra_pool_fails_when_exact_count_is_unavailable(self) -> None:
        nusc = SimpleNamespace(scene=[{"name": "scene-0001"}])
        with mock.patch(
            "data.extract_nuscenes.create_splits_scenes",
            return_value={"train": ["scene-0001"], "val": []},
        ):
            with self.assertRaisesRegex(ValueError, "Not enough eligible"):
                select_train_val_scenes(
                    nusc,
                    num_scenes_per_split=2,
                    seed=0,
                    active_splits=("train",),
                    fail_if_insufficient=True,
                )


if __name__ == "__main__":
    unittest.main()

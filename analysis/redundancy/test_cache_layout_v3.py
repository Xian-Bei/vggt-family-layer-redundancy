from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from analysis.redundancy.cache_layout_v3 import load_scene_unit_v3
from analysis.recovery.activation_store import RowShard, ShardedRowStore
from analysis.redundancy.cache_redundancy_v3 import _save_scene_cache


class DirectUnitLoaderTests(unittest.TestCase):
    def test_direct_pair_opens_only_requested_files_and_matches_halves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "scene" / "layers"
            layers.mkdir(parents=True)
            frame = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
            global_t = frame + 100
            torch.save(frame, layers / "frame_2.pt")
            torch.save(global_t, layers / "global_2.pt")
            item = {
                "scene_id": "scene",
                "dir": "scene",
                "num_frames": 2,
                "layer_files": {
                    "frame_2": "scene/layers/frame_2.pt",
                    "global_2": "scene/layers/global_2.pt",
                },
            }
            original_load = torch.load
            paths = []

            def recording_load(path, *args, **kwargs):
                paths.append(Path(path).name)
                return original_load(path, *args, **kwargs)

            with mock.patch("torch.load", side_effect=recording_load):
                actual_frame = load_scene_unit_v3(root, item, "frame_2")
                actual_global = load_scene_unit_v3(root, item, "global_2")
            self.assertEqual(paths, ["frame_2.pt", "global_2.pt"])
            torch.testing.assert_close(actual_frame, frame, rtol=0, atol=0)
            torch.testing.assert_close(actual_global, global_t, rtol=0, atol=0)

    def test_global_flat_scene_layout_is_normalized_without_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "scene" / "layers"
            layers.mkdir(parents=True)
            flat = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
            torch.save(flat, layers / "global_0.pt")
            item = {
                "dir": "scene",
                "num_frames": 2,
                "layer_files": {"global_0": "scene/layers/global_0.pt"},
            }
            actual = load_scene_unit_v3(root, item, "global_0", mmap=True)
            self.assertEqual(tuple(actual.shape), (1, 2, 3, 4))
            torch.testing.assert_close(actual, flat.reshape(1, 2, 3, 4))

    def test_legacy_concat_fallback_returns_a_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers = root / "scene" / "layers"
            layers.mkdir(parents=True)
            concat = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8)
            torch.save(concat, layers / "layer_1.pt")
            item = {"dir": "scene", "layer_files": {}}
            frame = load_scene_unit_v3(root, item, "frame_1")
            global_t = load_scene_unit_v3(root, item, "global_1")
            torch.testing.assert_close(frame, concat[..., :4])
            torch.testing.assert_close(global_t, concat[..., 4:])

    def test_sharded_global_gather_matches_concat_for_randperm(self) -> None:
        x1 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        x2 = torch.arange(15, dtype=torch.float32).reshape(5, 3) + 100
        y1, y2 = x1 + 7, x2 + 7
        store = ShardedRowStore(
            [RowShard("a", x1, desired=y1), RowShard("b", x2, desired=y2)]
        )
        order = torch.randperm(9, generator=torch.Generator().manual_seed(4))
        actual_x, actual_y = store.gather(order)
        torch.testing.assert_close(actual_x, torch.cat([x1, x2])[order], rtol=0, atol=0)
        torch.testing.assert_close(actual_y, torch.cat([y1, y2])[order], rtol=0, atol=0)

    def test_selected_cache_does_not_write_predictions_or_unrequested_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = torch.ones(1, 2, 3, 4)
            _save_scene_cache(
                cache_root=root,
                scene_id="scene",
                unit_map={"frame_0": selected, "global_0": selected + 1},
                pred=None,
                meta={"patch_start_idx": 0},
                images=torch.zeros(1, 2, 3, 8, 8),
                frame_ids=["0", "1"],
                image_paths=["a", "b"],
                scene_meta={},
                infer_s=0.0,
                adapter=object(),
                model=object(),
                unit_ids_filter={"frame_0"},
                require_predictions=False,
            )
            self.assertTrue((root / "scene/layers/frame_0.pt").is_file())
            self.assertFalse((root / "scene/layers/global_0.pt").exists())
            self.assertFalse((root / "scene/pred_v3.pt").exists())


if __name__ == "__main__":
    unittest.main()

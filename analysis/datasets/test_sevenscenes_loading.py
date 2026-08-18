from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from analysis.backbones.contracts import PredictionBundle
from analysis.datasets import eval_impl
from analysis.datasets.contracts import EvalSampleBatch, PreparedEvalScene
from analysis.datasets.frame_selection import pick_frame_indices
from analysis.datasets.seven_and_n_common import iter_7andn_scenes
from analysis.schema.v3 import DatasetEvalProfile


class _FakeDataset:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return [
            {"label": f"scene/seq/{index}", "instance": f"frame-{index}.png"}
            for index in range(8)
        ]


class _PreparedSceneAdapter:
    @staticmethod
    def load_eval_images(image_paths):
        values = torch.arange(len(image_paths) * 18, dtype=torch.float32)
        return values.reshape(len(image_paths), 3, 2, 3)

    @staticmethod
    def align_7andn_gt_depth(view, *, output_hw):
        del view
        return np.ones(output_hw, dtype=np.float32)


class SevenScenesLoadingTest(unittest.TestCase):
    def test_early_selector_exactly_matches_public_frame_cap(self) -> None:
        # Loading the class also exposes the top-level data module selected by
        # seven_and_n_common's import-path guard.
        from analysis.datasets import seven_and_n_common as common

        dataset_cls = common._load_7andn_dataset_cls("7scenes")
        data_module = sys.modules[dataset_cls.__module__]
        for count in (0, 1, 2, 29, 30, 31, 97, 1000):
            frame_ids = [f"{index:06d}" for index in range(count)]
            for max_frames in (0, 1, 7, 30, 2000):
                expected = [
                    frame_ids[index]
                    for index in pick_frame_indices(count, max_frames)
                ]
                actual = data_module._select_eval_img_idxs(
                    frame_ids, max_frames
                )
                self.assertEqual(actual, expected)

    def test_only_7scenes_receives_early_frame_cap(self) -> None:
        profile = SimpleNamespace(size=518, kf_every=10)
        with mock.patch(
            "analysis.datasets.seven_and_n_common._load_7andn_dataset_cls",
            return_value=_FakeDataset,
        ):
            batch = next(
                iter_7andn_scenes(
                    dataset="7scenes",
                    data_root=Path("unused"),
                    prof=profile,
                    max_scenes=1,
                    max_frames=3,
                    split="test",
                )
            )
            self.assertEqual(_FakeDataset.last_kwargs["eval_max_frames"], 3)
            self.assertEqual(len(batch.images), 3)

            next(
                iter_7andn_scenes(
                    dataset="nrgbd",
                    data_root=Path("unused"),
                    prof=profile,
                    max_scenes=1,
                    max_frames=3,
                )
            )
            self.assertNotIn("eval_max_frames", _FakeDataset.last_kwargs)

    def test_compact_prepare_is_array_equivalent_to_legacy_collate(self) -> None:
        views = []
        for index in range(3):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(index + 1)
            points = np.arange(18, dtype=np.float32).reshape(2, 3, 3) + index
            valid = np.array(
                [[True, False, True], [True, True, False]], dtype=bool
            )
            views.append(
                {
                    "img": torch.full((3, 2, 3), float(index)),
                    "depthmap": np.ones((2, 3), dtype=np.float32),
                    "camera_pose": pose,
                    "pts3d": points,
                    "valid_mask": valid,
                    "ray_map": torch.full((6, 2, 3), float("nan")),
                    "label": f"scene/seq/{index}",
                    "instance": f"frame-{index}.png",
                }
            )
        batch = EvalSampleBatch(scene_id="scene/seq", images=views)
        legacy = eval_impl._collate_7andn_views(batch)
        expected_poses = np.stack(
            [view["camera_pose"].float().numpy() for view in legacy], axis=0
        )
        while expected_poses.ndim > 3 and expected_poses.shape[1] == 1:
            expected_poses = expected_poses[:, 0]
        expected_points = []
        for view in legacy:
            points = view["pts3d"].float().numpy().reshape(-1, 3)
            valid = view["valid_mask"].numpy().astype(bool).reshape(-1)
            expected_points.append(points[valid])

        prepared = eval_impl.prepare_7andn_eval_scene_cpu(
            _PreparedSceneAdapter(), batch, SimpleNamespace()
        )
        self.assertEqual(prepared.state["num_views"], len(views))
        self.assertNotIn("scene_batch", prepared.state)
        self.assertTrue(np.array_equal(prepared.state["gt_poses"], expected_poses))
        for actual, expected in zip(
            prepared.state["gt_points_by_view"], expected_points
        ):
            self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(
            tuple(prepared.images.shape), (1, len(views), 3, 2, 3)
        )

    def test_compact_and_legacy_states_score_identically(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        poses[:, 0, 3] = np.arange(1, 4, dtype=np.float32)
        points = torch.arange(18, dtype=torch.float32).reshape(1, 3, 1, 2, 3)
        prediction = PredictionBundle(
            points=points,
            camera_poses=torch.from_numpy(poses[None]),
        )
        common_state = {
            "gt_poses": poses,
            "pose_valid": np.ones(3, dtype=bool),
            "gt_points_by_view": [
                points[0, index].numpy().reshape(-1, 3) for index in range(3)
            ],
            "pred_valid_masks_by_view": [
                np.ones((1, 2), dtype=bool) for _index in range(3)
            ],
        }
        legacy = PreparedEvalScene(
            scene_id="scene",
            images=None,
            state={**common_state, "scene_batch": [{}, {}, {}]},
        )
        compact = PreparedEvalScene(
            scene_id="scene",
            images=None,
            state={**common_state, "num_views": 3},
        )
        alignment = (
            {"ape": 1.0, "are": 2.0, "rpe_trans": 3.0, "rpe_rot": 4.0},
            np.eye(4, dtype=np.float64),
        )
        point_metrics = {"acc": 5.0, "comp": 6.0, "cd": 5.5}
        profile = DatasetEvalProfile(dataset="7scenes")
        with (
            mock.patch.object(
                eval_impl, "_trajectory_alignment", return_value=alignment
            ),
            mock.patch.object(
                eval_impl, "compute_point_metrics", return_value=point_metrics
            ),
        ):
            legacy_metrics = eval_impl.score_7andn_skip_prediction(
                prediction, legacy, profile, None
            )
            compact_metrics = eval_impl.score_7andn_skip_prediction(
                prediction, compact, profile, None
            )
        self.assertEqual(compact_metrics, legacy_metrics)


if __name__ == "__main__":
    unittest.main()

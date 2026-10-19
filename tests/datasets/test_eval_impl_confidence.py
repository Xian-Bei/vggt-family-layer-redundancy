from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from analysis.backbones.contracts import EvalSceneContext, PredictionBundle
from analysis.datasets import eval_impl
from analysis.datasets.contracts import PreparedSkipEvalScene
from analysis.schema.v3 import DatasetEvalProfile


def _scene_fixture(device: torch.device):
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[:, 0, 3] = np.arange(1, 4, dtype=np.float32)
    points = torch.arange(18, dtype=torch.float32, device=device).reshape(1, 3, 1, 2, 3)
    confidence = torch.full((1, 3, 1, 2, 1), -100.0, device=device)
    bundle = PredictionBundle(
        points=points,
        conf=confidence,
        camera_poses=torch.from_numpy(poses[None]).to(device),
    )
    scene_batch = [
        {
            "camera_pose": torch.from_numpy(poses[index]).to(device),
            "pts3d": points[0, index],
            "valid_mask": torch.ones((1, 2), dtype=torch.bool, device=device),
            "instance": f"frame_{index}.png",
        }
        for index in range(3)
    ]
    return bundle, scene_batch, poses


class ConfidenceIndependentPointEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")
        self.device = torch.device("cuda")
        self.profile = DatasetEvalProfile(dataset="7scenes")
        self.context = EvalSceneContext(
            device=self.device,
            profile=self.profile,
            dataset_name="7scenes",
        )

    @staticmethod
    def _alignment_result(*_args, **_kwargs):
        return (
            {"ape": 0.0, "are": 0.0, "rpe_trans": 0.0, "rpe_rot": 0.0},
            np.eye(4, dtype=np.float64),
        )

    def test_full_eval_keeps_points_below_confidence_threshold(self) -> None:
        bundle, scene_batch, _poses = _scene_fixture(self.device)
        adapter = SimpleNamespace(forward_predictions=lambda _model, _images: bundle)
        captured = {}

        def point_metrics(pred, gt, *_args, **_kwargs):
            captured["pred_count"] = int(pred.shape[0])
            captured["gt_count"] = int(gt.shape[0])
            return {"acc": 0.0, "comp": 0.0, "cd": 0.0}

        with (
            mock.patch.object(
                eval_impl,
                "_prepare_7andn_eval_batch",
                return_value=(
                    scene_batch,
                    torch.zeros((1, 3, 3, 2, 2), device=self.device),
                ),
            ),
            mock.patch.object(eval_impl, "_trajectory_alignment", self._alignment_result),
            mock.patch.object(eval_impl, "compute_point_metrics", side_effect=point_metrics),
        ):
            eval_impl.eval_7andn_points_adapter(
                adapter,
                object(),
                SimpleNamespace(scene_id="scene"),
                self.device,
                self.profile,
                self.context,
            )

        self.assertEqual(captured, {"pred_count": 6, "gt_count": 6})

    def test_skip_eval_keeps_points_below_confidence_threshold(self) -> None:
        bundle, scene_batch, poses = _scene_fixture(self.device)
        prepared = PreparedSkipEvalScene(
            scene_id="scene",
            images=torch.zeros((1, 3, 3, 2, 2), device=self.device),
            state={
                "scene_batch": scene_batch,
                "gt_poses": poses,
                "pose_valid": np.ones(3, dtype=bool),
                "gt_points_by_view": [
                    view["pts3d"].cpu().numpy().reshape(-1, 3) for view in scene_batch
                ],
            },
        )
        captured = {}

        def point_metrics(pred, gt, *_args, **_kwargs):
            captured["pred_count"] = int(pred.shape[0])
            captured["gt_count"] = int(gt.shape[0])
            return {"acc": 0.0, "comp": 0.0, "cd": 0.0}

        with (
            mock.patch.object(eval_impl, "_trajectory_alignment", self._alignment_result),
            mock.patch.object(eval_impl, "compute_point_metrics", side_effect=point_metrics),
        ):
            eval_impl.score_7andn_skip_prediction(
                bundle,
                prepared,
                self.profile,
                self.context,
            )

        self.assertEqual(captured, {"pred_count": 6, "gt_count": 6})

    def test_full_eval_applies_gt_validity_to_predicted_pixels(self) -> None:
        bundle, scene_batch, _poses = _scene_fixture(self.device)
        for view in scene_batch:
            view["valid_mask"] = torch.tensor([[True, False]], device=self.device)
        adapter = SimpleNamespace(forward_predictions=lambda _model, _images: bundle)
        captured = {}

        def point_metrics(pred, gt, *_args, **_kwargs):
            captured["pred_count"] = int(pred.shape[0])
            captured["gt_count"] = int(gt.shape[0])
            return {"acc": 0.0, "comp": 0.0, "cd": 0.0}

        with (
            mock.patch.object(
                eval_impl,
                "_prepare_7andn_eval_batch",
                return_value=(
                    scene_batch,
                    torch.zeros((1, 3, 3, 1, 2), device=self.device),
                ),
            ),
            mock.patch.object(eval_impl, "_trajectory_alignment", self._alignment_result),
            mock.patch.object(eval_impl, "compute_point_metrics", side_effect=point_metrics),
        ):
            eval_impl.eval_7andn_points_adapter(
                adapter,
                object(),
                SimpleNamespace(scene_id="scene"),
                self.device,
                self.profile,
                self.context,
            )

        self.assertEqual(captured, {"pred_count": 3, "gt_count": 3})

    def test_skip_eval_applies_gt_validity_to_predicted_pixels(self) -> None:
        bundle, scene_batch, poses = _scene_fixture(self.device)
        valid = np.array([[True, False]], dtype=bool)
        prepared = PreparedSkipEvalScene(
            scene_id="scene",
            images=torch.zeros((1, 3, 3, 1, 2), device=self.device),
            state={
                "scene_batch": scene_batch,
                "gt_poses": poses,
                "pose_valid": np.ones(3, dtype=bool),
                "gt_points_by_view": [
                    view["pts3d"].cpu().numpy().reshape(-1, 3)[valid.reshape(-1)]
                    for view in scene_batch
                ],
                "pred_valid_masks_by_view": [valid.copy() for _view in scene_batch],
            },
        )
        captured = {}

        def point_metrics(pred, gt, *_args, **_kwargs):
            captured["pred_count"] = int(pred.shape[0])
            captured["gt_count"] = int(gt.shape[0])
            return {"acc": 0.0, "comp": 0.0, "cd": 0.0}

        with (
            mock.patch.object(eval_impl, "_trajectory_alignment", self._alignment_result),
            mock.patch.object(eval_impl, "compute_point_metrics", side_effect=point_metrics),
        ):
            eval_impl.score_7andn_skip_prediction(
                bundle,
                prepared,
                self.profile,
                self.context,
            )

        self.assertEqual(captured, {"pred_count": 3, "gt_count": 3})


class ScanNetVisualAttributesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")
        self.device = torch.device("cuda")

    def test_maps_input_rgb_confidence_and_frame_ids_to_point_order(self) -> None:
        points = torch.zeros((1, 2, 2, 3, 3), device=self.device)
        images = torch.zeros((1, 2, 3, 1, 2), device=self.device)
        images[0, 0, 0] = 1.0
        images[0, 1, 1] = 1.0
        confidence = torch.arange(12, device=self.device).reshape(1, 2, 2, 3)

        colors, conf, frames = eval_impl._scannet_visual_attributes(
            points, images, confidence
        )

        self.assertEqual(colors.shape, (12, 3))
        np.testing.assert_array_equal(colors[:6], np.tile([255, 0, 0], (6, 1)))
        np.testing.assert_array_equal(colors[6:], np.tile([0, 255, 0], (6, 1)))
        np.testing.assert_array_equal(conf, np.arange(12, dtype=np.float32))
        np.testing.assert_array_equal(frames, [0] * 6 + [1] * 6)

    def test_sim3_transforms_camera_centers_into_aligned_world(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        source_centers = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
        poses[:, :3, 3] = -source_centers
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        sim3 = np.eye(4, dtype=np.float64)
        sim3[:3, :3] = 2.5 * rotation
        sim3[:3, 3] = [4.0, -3.0, 1.0]

        aligned = eval_impl._transform_w2c_sim3(poses, sim3)
        self.assertIsNotNone(aligned)
        aligned_centers = -np.einsum(
            "nij,nj->ni",
            np.transpose(aligned[:, :3, :3], (0, 2, 1)),
            aligned[:, :3, 3],
        )
        expected = source_centers @ sim3[:3, :3].T + sim3[:3, 3]
        np.testing.assert_allclose(aligned_centers, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

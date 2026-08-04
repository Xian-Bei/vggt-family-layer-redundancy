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


if __name__ == "__main__":
    unittest.main()

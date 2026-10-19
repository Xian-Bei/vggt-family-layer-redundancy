from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from analysis.backbones.adapters.vggt_adapter import VGGTAdapter
from analysis.backbones.adapters.vggt_omega_adapter import VGGTOmegaAdapter


class _ForbiddenPointHead(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("VGGT point head must not run")


class VGGTHeadPolicyTest(unittest.TestCase):
    def test_load_pretrained_drops_point_head(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")

        class _FakeVGGT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.point_head = nn.Linear(1, 1)

            @classmethod
            def from_pretrained(cls, _name: str):
                return cls()

        adapter = VGGTAdapter()
        with (
            mock.patch.object(adapter, "_assert_vendor_origin"),
            mock.patch(
                "analysis.backbones.adapters._vggt_base.import_vggt_class",
                return_value=_FakeVGGT,
            ),
        ):
            model = adapter.load_pretrained(
                hf_model="unused",
                device="cuda",
                dtype=None,
            )

        self.assertIsNone(model.point_head)

    def test_shared_head_runner_never_runs_point_head(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")
        model = nn.Module()
        model.camera_head = None
        model.depth_head = None
        model.point_head = _ForbiddenPointHead()
        model.track_head = None
        model.eval()
        images = torch.zeros(1, 2, 3, 14, 14, device="cuda")

        predictions = VGGTAdapter().run_heads_from_aggregated_tokens(
            model,
            [],
            0,
            images,
        )

        self.assertNotIn("world_points", predictions)
        self.assertNotIn("world_points_conf", predictions)

    def test_full_eval_forward_can_defer_cpu_unprojection(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")
        adapter = VGGTAdapter()
        model = mock.Mock()
        model.aggregator.return_value = ([], 0)
        images = torch.zeros(1, 2, 3, 2, 2, device="cuda")
        pose_enc = torch.zeros(1, 2, 9, device="cuda")
        depth = torch.ones(1, 2, 2, 2, 1, device="cuda")
        extrinsic = (
            torch.eye(4, device="cuda")[:3]
            .reshape(1, 1, 3, 4)
            .expand(1, 2, 3, 4)
        )
        intrinsic = (
            torch.eye(3, device="cuda")
            .reshape(1, 1, 3, 3)
            .expand(1, 2, 3, 3)
        )

        with (
            mock.patch.object(
                adapter,
                "run_heads_from_aggregated_tokens",
                return_value={"pose_enc": pose_enc, "depth": depth},
            ),
            mock.patch(
                "analysis.backbones.prediction.decode_pose_enc_to_extrinsics_intrinsics",
                return_value=(extrinsic, intrinsic),
            ),
            mock.patch(
                "analysis.backbones.prediction.unproject_depth_payload"
            ) as unproject,
        ):
            prediction = adapter.forward_predictions(
                model,
                images,
                defer_cpu_geometry=True,
            )

        unproject.assert_not_called()
        self.assertIsNone(prediction.points)
        self.assertIs(prediction.depth, depth)
        self.assertEqual(prediction.raw["deferred_unprojection_backbone"], "vggt")
        self.assertIs(prediction.raw["deferred_unprojection_extrinsic"], extrinsic)
        self.assertIs(prediction.raw["deferred_unprojection_intrinsic"], intrinsic)

    def test_omega_full_eval_forward_can_defer_cpu_unprojection(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "GPU test requires CUDA")
        pose_enc = torch.zeros(1, 2, 9, device="cuda")
        depth = torch.ones(1, 2, 2, 2, 1, device="cuda")

        class _FakeOmega(nn.Module):
            def forward(self, _images):
                return {"pose_enc": pose_enc, "depth": depth}

        extrinsic = (
            torch.eye(4, device="cuda")[:3]
            .reshape(1, 1, 3, 4)
            .expand(1, 2, 3, 4)
        )
        intrinsic = (
            torch.eye(3, device="cuda")
            .reshape(1, 1, 3, 3)
            .expand(1, 2, 3, 3)
        )
        with (
            mock.patch(
                "analysis.backbones.prediction.decode_pose_enc_to_extrinsics_intrinsics",
                return_value=(extrinsic, intrinsic),
            ),
            mock.patch(
                "analysis.backbones.prediction.unproject_depth_payload"
            ) as unproject,
        ):
            prediction = VGGTOmegaAdapter().forward_predictions(
                _FakeOmega().cuda().eval(),
                torch.zeros(1, 2, 3, 2, 2, device="cuda"),
                defer_cpu_geometry=True,
            )

        unproject.assert_not_called()
        self.assertIsNone(prediction.points)
        self.assertIs(prediction.depth, depth)
        self.assertEqual(
            prediction.raw["deferred_unprojection_backbone"],
            "vggt_omega",
        )


if __name__ == "__main__":
    unittest.main()

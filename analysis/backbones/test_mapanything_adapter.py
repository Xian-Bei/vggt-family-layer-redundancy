from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from analysis.backbones.adapters.mapanything_adapter import (
    MapAnythingAdapter,
    MapAnythingV1Adapter,
)
from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    UnitLoraSpec,
    UnitPrunePlan,
    plan_unit_id,
    prune_actions_from_plan,
)
from analysis.schema.v3 import PruneAction


class _Attention(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(x).chunk(3, dim=-1)[0])


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()

    def forward(self, x: torch.Tensor, xpos=None) -> torch.Tensor:
        del xpos
        return x + self.attn(x)


def _model(depth: int) -> nn.Module:
    model = nn.Module()
    model.info_sharing = nn.Module()
    model.info_sharing.self_attention_blocks = nn.ModuleList(
        [_Block() for _ in range(depth)]
    )
    return model


class MapAnythingAdapterTest(unittest.TestCase):
    def test_registry_and_checkpoint_defaults(self) -> None:
        self.assertEqual(get_backbone("map-anything").default_hf_model(), "facebook/map-anything")
        self.assertEqual(
            get_backbone("map-anything-v1").default_hf_model(),
            "facebook/map-anything-v1",
        )

    def test_catalog_is_dynamic_and_global_first(self) -> None:
        for adapter, depth in ((MapAnythingAdapter(), 16), (MapAnythingV1Adapter(), 24)):
            ids = [unit.unit_id for unit in adapter.layer_unit_catalog(_model(depth))]
            self.assertEqual(len(ids), depth)
            self.assertEqual(ids[:4], ["global_0", "frame_0", "global_1", "frame_1"])

    def test_capture_excludes_scale_and_reorders_registers(self) -> None:
        adapter = MapAnythingAdapter()
        # Two views, 4 patches + 2 registers per view, then one scale token.
        raw = torch.arange(13 * 3).reshape(1, 13, 3)
        value, registers = adapter._normalize_capture(
            "global_0", raw, batch_size=1, num_frames=2, patch_tokens=4
        )
        self.assertEqual(tuple(value.shape), (1, 2, 6, 3))
        self.assertEqual(registers, 2)
        torch.testing.assert_close(value[0, 0, :2], raw[0, 4:6])
        torch.testing.assert_close(value[0, 0, 2:], raw[0, :4])

    def test_ordered_and_legacy_plan_mapping(self) -> None:
        ordered = ["global_0", "frame_0", "global_1", "frame_1"]
        plan = UnitPrunePlan(
            backbone="mapanything",
            dataset="test",
            num_units=4,
            unit_ids=ordered,
            drop=[1],
            drop_length=1,
            lora={"2": UnitLoraSpec(rank=2, alpha=4)},
        )
        drop, lora = prune_actions_from_plan(plan)
        self.assertEqual(drop[0].unit_id, "frame_0")
        self.assertEqual(lora[0].unit_id, "global_1")
        legacy = UnitPrunePlan(
            backbone="vggt", dataset="test", num_units=4, drop=[], drop_length=0
        )
        self.assertEqual(plan_unit_id(legacy, 0), "frame_0")

    def test_prune_and_lora_share_physical_mapping(self) -> None:
        adapter = MapAnythingAdapter()
        model = _model(4)
        trainable = adapter.inject_lora_plan(
            model,
            [PruneAction(action="lora_unit", unit_id="global_1", rank=2, alpha=4)],
        )
        self.assertTrue(trainable)
        loss = model.info_sharing.self_attention_blocks[2](torch.randn(2, 3, 8)).sum()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in trainable))
        adapter.apply_prune_plan(
            model, [PruneAction(action="drop_unit", unit_id="frame_0")]
        )
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(model.info_sharing.self_attention_blocks[1](x), x)


if __name__ == "__main__":
    unittest.main()

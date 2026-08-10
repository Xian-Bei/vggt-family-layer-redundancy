from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import torch
import torch.nn as nn

from analysis.inference_compression.FastVGGT.adapter import FastVGGTInferenceAdapter
from analysis.inference_compression.LoRAMerge.adapter import LoRAMergeInferenceAdapter
from analysis.inference_compression.QuantVGGT.adapter import QuantVGGTInferenceAdapter
from analysis.inference_compression.contracts import InferenceCompressionConfig
from analysis.inference_compression.lora import merge_lora_linears
from analysis.inference_compression.registry import list_inference_compressions
from analysis.recovery.train.model import LoRALinear, SkipIdentityBlock

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Attention(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 2):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.rope = None

    def forward(self, x, pos=None):
        return x


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()

    def forward(self, x, pos=None):
        return x + self.attn(x, pos=pos)


class _Aggregator(nn.Module):
    patch_size = 2

    def __init__(self):
        super().__init__()
        self.global_blocks = nn.ModuleList([_Block()])

    def forward(self, images):
        batch, frames, _, height, width = images.shape
        tokens_per_frame = (height // self.patch_size) * (width // self.patch_size) + 5
        tokens = torch.randn(batch, frames * tokens_per_frame, 8, device=images.device)
        return self.global_blocks[0](tokens), 5


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = _Aggregator()


class InferenceCompressionTests(unittest.TestCase):
    def test_registry_tags_and_backbone_validation(self):
        fast = FastVGGTInferenceAdapter()
        lora_merge = LoRAMergeInferenceAdapter()
        quant = QuantVGGTInferenceAdapter()
        self.assertEqual(
            list_inference_compressions(),
            ["fastvggt", "lora_merge", "quantvggt"],
        )
        self.assertEqual(
            fast.output_tag(InferenceCompressionConfig(name="fastvggt")),
            "fastvggt_m0_r0p9",
        )
        self.assertEqual(
            quant.output_tag(InferenceCompressionConfig(name="quantvggt", quantvggt_dtype="w6a6")),
            "quantvggt_w6a6",
        )
        self.assertEqual(
            lora_merge.output_tag(InferenceCompressionConfig(name="lora_merge")),
            "lora_merge",
        )
        for backbone in ("vggt", "pi3", "vggt_omega"):
            fast.validate_backbone(backbone)
            quant.validate_backbone(backbone)
        with self.assertRaises(ValueError):
            fast.validate_backbone("unsupported")

    def test_fastvggt_patches_existing_model_and_preserves_shape(self):
        model = _Model().to(_DEVICE).eval()
        config = InferenceCompressionConfig(
            name="fastvggt",
            fastvggt_merge_start=0,
            fastvggt_merge_ratio=0.2,
        )
        FastVGGTInferenceAdapter().apply(model, backbone_name="vggt", config=config)
        output, _ = model.aggregator(torch.zeros(1, 2, 3, 4, 4, device=_DEVICE))
        self.assertEqual(tuple(output.shape), (1, 18, 8))

        skipped = _Model().to(_DEVICE).eval()
        skipped.aggregator.global_blocks[0] = SkipIdentityBlock()
        FastVGGTInferenceAdapter().apply(skipped, backbone_name="vggt", config=config)
        output, _ = skipped.aggregator(torch.zeros(1, 2, 3, 4, 4, device=_DEVICE))
        self.assertEqual(tuple(output.shape), (1, 18, 8))
        attn = model.aggregator.global_blocks[0].attn
        self.assertEqual(attn._fastvggt_patch_height, 2)
        self.assertEqual(attn._fastvggt_patch_width, 2)

    def test_fastvggt_accepts_lora_projection_and_skipped_block(self):
        model = _Model().to(_DEVICE).eval()
        attn = model.aggregator.global_blocks[0].attn
        attn.qkv = LoRALinear(attn.qkv, rank=2, alpha=4.0)
        with torch.no_grad():
            attn.qkv.lora_b.weight.normal_()
        config = InferenceCompressionConfig(
            name="fastvggt",
            fastvggt_merge_start=0,
            fastvggt_merge_ratio=0.2,
        )
        FastVGGTInferenceAdapter().apply(model, backbone_name="vggt", config=config)
        self.assertIsInstance(attn.qkv, nn.Linear)
        self.assertEqual(model._inference_compression["merged_lora_linears"], 1)
        output, _ = model.aggregator(torch.zeros(1, 2, 3, 4, 4, device=_DEVICE))
        self.assertEqual(tuple(output.shape), (1, 18, 8))

    def test_fastvggt_maps_pi3_global_decoder_blocks(self):
        class Pi3Model(nn.Module):
            patch_size = 2

            def __init__(self):
                super().__init__()
                self.decoder = nn.ModuleList([_Block(), _Block(), _Block(), _Block()])

            def forward(self, images):
                batch, frames, _, height, width = images.shape
                per_frame = (height // self.patch_size) * (width // self.patch_size) + 5
                tokens = torch.randn(batch, frames * per_frame, 8, device=images.device)
                return self.decoder[1](tokens)

        model = Pi3Model().to(_DEVICE).eval()
        FastVGGTInferenceAdapter().apply(
            model,
            backbone_name="pi3",
            config=InferenceCompressionConfig(
                name="fastvggt", fastvggt_merge_start=0, fastvggt_merge_ratio=0.2
            ),
        )
        output = model(torch.zeros(1, 2, 3, 4, 4, device=_DEVICE))
        self.assertEqual(tuple(output.shape), (1, 18, 8))
        self.assertFalse(hasattr(model.decoder[0].attn, "_fastvggt_merge_ratio"))
        self.assertTrue(hasattr(model.decoder[1].attn, "_fastvggt_merge_ratio"))

    def test_fastvggt_maps_only_omega_patch_global_blocks(self):
        model = _Model().eval()
        del model.aggregator.global_blocks
        model.aggregator.patch_token_start = 17
        model.aggregator.inter_frame_blocks = nn.ModuleList([_Block(), _Block()])
        model.aggregator.inter_frame_attention_types = ["register", "global"]
        FastVGGTInferenceAdapter().apply(
            model,
            backbone_name="vggt_omega",
            config=InferenceCompressionConfig(
                name="fastvggt", fastvggt_merge_start=0, fastvggt_merge_ratio=0.2
            ),
        )
        self.assertFalse(hasattr(model.aggregator.inter_frame_blocks[0].attn, "_fastvggt_merge_ratio"))
        self.assertTrue(hasattr(model.aggregator.inter_frame_blocks[1].attn, "_fastvggt_merge_ratio"))
        self.assertEqual(
            model.aggregator.inter_frame_blocks[1].attn._fastvggt_num_special_tokens, 17
        )
        attn = model.aggregator.inter_frame_blocks[1].attn.to(_DEVICE)
        attn._fastvggt_patch_height = 2
        attn._fastvggt_patch_width = 2
        output = attn(torch.randn(1, 2 * (17 + 4), 8, device=_DEVICE))
        self.assertEqual(tuple(output.shape), (1, 42, 8))

    def test_quant_block_group_mapping_covers_all_backbones(self):
        vggt = _Model()
        vggt.aggregator.frame_blocks = nn.ModuleList([_Block(), _Block()])
        groups = QuantVGGTInferenceAdapter._block_groups(vggt, "vggt")
        self.assertEqual(groups[1][2], [(0, 0)])

        pi3 = nn.Module()
        pi3.decoder = nn.ModuleList([_Block(), _Block(), _Block(), _Block()])
        groups = QuantVGGTInferenceAdapter._block_groups(pi3, "pi3")
        self.assertEqual(groups[0][2], [(0, 0), (1, 2)])
        self.assertEqual(groups[1][2], [(0, 1), (1, 3)])

        omega = _Model()
        omega.aggregator.frame_blocks = nn.ModuleList([_Block()])
        omega.aggregator.inter_frame_blocks = nn.ModuleList([_Block(), _Block()])
        groups = QuantVGGTInferenceAdapter._block_groups(omega, "vggt_omega")
        self.assertEqual(groups[1][2], [(0, 0), (1, 1)])

    def test_lora_merge_is_eval_equivalent(self):
        torch.manual_seed(7)
        model = nn.Sequential(LoRALinear(nn.Linear(5, 4), rank=2, alpha=4.0)).to(_DEVICE).eval()
        with torch.no_grad():
            model[0].lora_b.weight.normal_()
        inputs = torch.randn(3, 5, device=_DEVICE)
        expected = model(inputs)
        self.assertEqual(merge_lora_linears(model), 1)
        self.assertIsInstance(model[0], nn.Linear)
        torch.testing.assert_close(model(inputs), expected)

    def test_lora_merge_standalone_adapter_is_eval_equivalent(self):
        torch.manual_seed(11)
        model = nn.Sequential(LoRALinear(nn.Linear(5, 4), rank=2, alpha=4.0)).to(_DEVICE).eval()
        with torch.no_grad():
            model[0].lora_b.weight.normal_()
        inputs = torch.randn(3, 5, device=_DEVICE)
        expected = model(inputs)
        LoRAMergeInferenceAdapter().apply(
            model,
            backbone_name="vggt",
            config=InferenceCompressionConfig(name="lora_merge"),
        )
        self.assertIsInstance(model[0], nn.Linear)
        self.assertEqual(model._inference_compression["merged_lora_linears"], 1)
        torch.testing.assert_close(model(inputs), expected)

    def test_quant_requires_parameter_files_before_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = InferenceCompressionConfig(
                name="quantvggt",
                quantvggt_dtype="w4a4",
                quantvggt_params_dir=Path(tmp),
            )
            with self.assertRaises(FileNotFoundError):
                QuantVGGTInferenceAdapter().apply(_Model(), backbone_name="vggt", config=config)

    def test_quant_rejects_manifest_bit_width_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "qs_frame_parameters_total.pth").touch()
            (root / "qs_global_parameters_total.pth").touch()
            (root / "quantized_linear_layers.json").write_text(
                json.dumps([{"w_bits": 8, "a_bits": 8}]),
                encoding="utf-8",
            )
            config = InferenceCompressionConfig(
                name="quantvggt",
                quantvggt_dtype="w4a4",
                quantvggt_params_dir=root,
            )
            with self.assertRaises(ValueError):
                QuantVGGTInferenceAdapter().apply(_Model(), backbone_name="vggt", config=config)


if __name__ == "__main__":
    unittest.main()

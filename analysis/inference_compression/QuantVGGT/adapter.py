"""QuantVGGT fake-quant deployment adapter for a restored LaRe VGGT model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from analysis.inference_compression.contracts import (
    InferenceCompressionAdapter,
    InferenceCompressionConfig,
)
from analysis.inference_compression.lora import merge_lora_linears
from analysis.recovery.train.model import SkipIdentityBlock

_DTYPE_BITS = {"w4a4": (4, 4), "w6a6": (6, 6), "w8a8": (8, 8)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_quant_modules():
    evaluation_root = _repo_root() / "submodules" / "QuantVGGT" / "evaluation"
    if not evaluation_root.is_dir():
        raise FileNotFoundError(
            f"QuantVGGT submodule is missing {evaluation_root}. "
            "Run: git submodule update --init submodules/QuantVGGT"
        )
    root_str = str(evaluation_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from quarot.args_utils import get_config
    from quarot.quarot_linear import SmoothRotQuantizedLinear
    from quarot.utils import quantize_linear

    return get_config, SmoothRotQuantizedLinear, quantize_linear


def _layer_params(raw: Mapping[Any, Any], index: int) -> Mapping[str, torch.Tensor]:
    value = raw.get(index, raw.get(str(index)))
    if not isinstance(value, Mapping):
        raise ValueError(f"QuantVGGT parameters are missing layer {index}")
    return value


def _expected_calibration_keys(block: nn.Module, quantized_type: type[nn.Module]) -> set[str]:
    expected: set[str] = set()
    for name, module in block.named_modules():
        if not isinstance(module, quantized_type):
            continue
        prefix = f"{name}." if name else ""
        expected.add(prefix + "rotation_matrix")
        expected.add(prefix + "channel_wise_scale")
        if getattr(module, "lwc", False):
            expected.update({prefix + "clip_factor_w_max", prefix + "clip_factor_w_min"})
        if getattr(module, "lac", False):
            expected.update(
                {
                    prefix + "act_quantizer.clip_factor_a_max",
                    prefix + "act_quantizer.clip_factor_a_min",
                }
            )
    return expected


class QuantVGGTInferenceAdapter(InferenceCompressionAdapter):
    name = "quantvggt"
    supported_backbones = frozenset({"vggt"})

    def validate_config(self, config: InferenceCompressionConfig) -> None:
        dtype = str(config.quantvggt_dtype).strip().lower()
        if dtype not in _DTYPE_BITS:
            raise ValueError(f"Unsupported QuantVGGT dtype {dtype!r}; expected {sorted(_DTYPE_BITS)}")
        params_dir = config.quantvggt_params_dir
        if params_dir is None:
            raise ValueError("QUANTVGGT_PARAMS_DIR is required when INFERENCE_COMPRESSION=quantvggt")
        params_dir = Path(params_dir)
        required = (
            params_dir / "qs_frame_parameters_total.pth",
            params_dir / "qs_global_parameters_total.pth",
        )
        missing_files = [str(path) for path in required if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing QuantVGGT parameter files: {missing_files}")
        manifest_path = params_dir / "quantized_linear_layers.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            observed = {
                (int(row["w_bits"]), int(row["a_bits"]))
                for row in manifest
                if isinstance(row, dict) and row.get("w_bits") is not None and row.get("a_bits") is not None
            }
            expected_bits = {_DTYPE_BITS[dtype]}
            if observed and observed != expected_bits:
                raise ValueError(
                    f"QuantVGGT parameter manifest uses bit-widths {sorted(observed)}, "
                    f"but --quantvggt-dtype={dtype} requires {sorted(expected_bits)}"
                )

    def apply(
        self,
        model: nn.Module,
        *,
        backbone_name: str,
        config: InferenceCompressionConfig,
    ) -> nn.Module:
        self.validate_backbone(backbone_name)
        self.validate_config(config)
        dtype = str(config.quantvggt_dtype).strip().lower()
        assert config.quantvggt_params_dir is not None
        params_dir = Path(config.quantvggt_params_dir)
        frame_path = params_dir / "qs_frame_parameters_total.pth"
        global_path = params_dir / "qs_global_parameters_total.pth"
        if not torch.cuda.is_available():
            raise RuntimeError("The vendored QuantVGGT implementation requires CUDA")

        aggregator = getattr(model, "aggregator", None)
        frame_blocks = getattr(aggregator, "frame_blocks", None)
        global_blocks = getattr(aggregator, "global_blocks", None)
        if aggregator is None or frame_blocks is None or global_blocks is None:
            raise TypeError("QuantVGGT requires a VGGT model with frame_blocks and global_blocks")

        model.eval()
        merged_count = merge_lora_linears(model)
        get_config, quantized_type, quantize_linear = _load_quant_modules()
        w_bits, a_bits = _DTYPE_BITS[dtype]
        quant_config = get_config()
        quant_config.update_from_args(
            wbit=w_bits,
            abit=a_bits,
            model_id="facebook/VGGT-1B",
            not_smooth=False,
            not_rot=False,
            lwc=True,
            lac=True,
            rv=True,
            exp_name="inference",
        )

        for blocks in (frame_blocks, global_blocks):
            for index, block in enumerate(blocks):
                if isinstance(block, SkipIdentityBlock):
                    continue
                blocks[index] = quantize_linear(block, args=quant_config)

        # The official files are about 1.92 GB each. Memory-map and consume them
        # sequentially so applying QuantVGGT does not materialize both in host RAM.
        for label, blocks, params_path in (
            ("frame", frame_blocks, frame_path),
            ("global", global_blocks, global_path),
        ):
            all_params = torch.load(
                params_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for index, block in enumerate(blocks):
                if isinstance(block, SkipIdentityBlock):
                    continue
                state = _layer_params(all_params, index)
                expected = _expected_calibration_keys(block, quantized_type)
                missing = sorted(expected.difference(state.keys()))
                if missing:
                    raise ValueError(
                        f"QuantVGGT {label} layer {index} parameters do not match {dtype}; "
                        f"missing keys: {missing[:8]}"
                    )
                block.load_state_dict(state, strict=False)
            del all_params

        for module in model.modules():
            if isinstance(module, quantized_type):
                module.smooth_quant_running_stat = False
                module.ori_mode = False
                module.train_mode = False
                module.eval_mode = True
                module.reparameterize()

        record = config.as_record()
        record["merged_lora_linears"] = merged_count
        model._inference_compression = record
        return model

    def output_tag(self, config: InferenceCompressionConfig) -> str:
        return f"quantvggt_{str(config.quantvggt_dtype).strip().lower()}"

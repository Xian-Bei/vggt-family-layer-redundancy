# Inference compression

LaRe treats `vggt`, `pi3`, and `vggt_omega` as backbones. LoRA merge,
FastVGGT, and QuantVGGT are deployment-only transformations registered under
`analysis/inference_compression`; all currently support only the VGGT backbone.

The evaluation order is fixed:

1. load the pretrained backbone;
2. restore the LaRe prune plan and optional LoRA checkpoint;
3. apply one inference compression adapter;
4. run the unchanged dataset evaluation protocol.

## `run_eval.sh` options

```bash
INFERENCE_COMPRESSION=none      # none | lora_merge | fastvggt | quantvggt
FASTVGGT_MERGE_START=0
FASTVGGT_MERGE_RATIO=0.9
QUANTVGGT_DTYPE=w4a4            # w4a4 | w6a6 | w8a8
QUANTVGGT_PARAMS_DIR=           # required only for quantvggt
```

`lora_merge` only folds evaluation-time LoRA projections into their base linear
layers. FastVGGT and QuantVGGT automatically perform the same fold before their
own transformation; they do not need to be combined with `lora_merge`.

FastVGGT then patches the restored model's global attention blocks, preserving
the LaRe topology. QuantVGGT then fake-quantizes the remaining aggregator linear
layers. `QUANTVGGT_PARAMS_DIR` is deliberately empty by default and is consulted
only when `INFERENCE_COMPRESSION=quantvggt`. It is a directory of precomputed
QuantVGGT calibration parameters—not a LaRe checkpoint, dataset, or output
directory—and must contain
`qs_frame_parameters_total.pth` and `qs_global_parameters_total.pth`. The
parameters must match `QUANTVGGT_DTYPE`; parameters recalibrated on the final
deployed LaRe model are preferred. The official W4A4 parameters calibrated on
full VGGT can also be selected explicitly as a transfer baseline, as discussed
below. There is no silent default because these two meanings should not be
confused. The QuantVGGT submodule does not ship the generated `.pth` artifacts,
and `run_eval.sh` intentionally does not perform calibration on the fly.

Download and verify the official W4A4 files with:

```bash
bash data/setup_quantvggt_params.sh
```

The default path can then be passed as:

```bash
INFERENCE_COMPRESSION=quantvggt \
QUANTVGGT_DTYPE=w4a4 \
QUANTVGGT_PARAMS_DIR=data/quantvggt/w4a4_official \
  bash run_eval.sh
```

For example:

```bash
INFERENCE_COMPRESSION=lora_merge CHECKPOINT=/path/to/lare.pt bash run_eval.sh
INFERENCE_COMPRESSION=quantvggt QUANTVGGT_DTYPE=w4a4 \
  QUANTVGGT_PARAMS_DIR=/path/to/matching/w4a4_params bash run_eval.sh
```

Only one inference compression adapter may be selected in a run. Enabling one
for Pi3 or VGGT-Omega is an error. Output model names receive a configuration
suffix automatically, preventing compressed and uncompressed evaluations from
overwriting one another.

## What the QuantVGGT parameter files contain

The official files are **not** packed INT4 VGGT weights. The base weights still
come from the restored VGGT or LaRe checkpoint. Each file is a dictionary with
24 original block indices (`0` through `23`): one file for frame blocks and one
for global blocks. Each block contains calibration state for four Linear layers:
attention `qkv`, attention `proj`, MLP `fc1`, and MLP `fc2`.

Inspection of the downloaded artifacts shows 24 keys per block and 576 tensors
per file: 96 instances of each of the six tensor categories below. Their roughly
1.92 GB per-file size is dominated by FP32 rotation matrices, especially the
`4096 x 4096` matrix associated with each MLP `fc2`. The upstream recipe derives
the learned state from a filtered CO3D calibration cache and the original
`model_tracker_fixed_e20.pt` VGGT weights.

For each Linear, the files store FP32 tensors for:

- `rotation_matrix`: an orthogonal/Hadamard feature rotation;
- `channel_wise_scale`: the SmoothQuant input-channel redistribution scale;
- `clip_factor_w_max/min`: learned per-output-channel weight clipping factors;
- `act_quantizer.clip_factor_a_max/min`: learned activation clipping factors.

They do not contain the original Linear weights, heads, patch embedding, an
integer-packed model, or fixed activation values. At application time the
adapter wraps only active aggregator Linear layers, loads this state, and calls
`reparameterize()`. That operation rotates, rescales, clips, and W4
quantize-dequantizes the current model weight. During forward, activations are
rotated, inversely rescaled, clipped, and dynamically A4
quantize-dequantized per token. The current vendored implementation is therefore
a fake-quant accuracy path; it keeps dequantized weights in floating-point
storage and does not provide an INT4 packing/kernel speedup by itself.

## Applying QuantVGGT after LaRe pruning

LaRe replaces removed blocks with `SkipIdentityBlock` instead of renumbering
the 24 block slots. The current adapter therefore has a well-defined structural
mapping for the official files:

1. restore the LaRe prune topology and checkpoint;
2. merge all LoRA deltas into ordinary Linear weights;
3. leave skipped blocks untouched;
4. wrap each remaining frame/global block's Linear layers;
5. load calibration state using its original block index;
6. reparameterize the current merged weights and run fake-quant inference.

This gives two distinct experimental settings:

1. **Official-parameter transfer.** Use the downloaded original-VGGT W4A4
   parameters. It runs without an index mismatch and is useful as a direct,
   reproducible transfer baseline. However, deleting earlier blocks changes the
   input distribution of later blocks, and LoRA changes their weights, so the
   official SmoothQuant scales and clipping factors are no longer optimal.
2. **LaRe-aware recalibration.** Recommended for the main orthogonality result.
   Starting from the final pruned checkpoint, merge LoRA first, calibrate only
   active blocks in the actual execution order, and save dictionaries keyed by
   their original indices. Skipped indices may be absent because the adapter
   never requests them. Use the same calibration samples and budget for VGGT
   and LaRe comparisons.

The upstream calibration routine currently assumes all 24 frame/global blocks
are present when it trains and merges per-layer files. A proper LaRe-aware
calibration entry point must teach that routine to pass through
`SkipIdentityBlock`, avoid creating parameters for it, and preserve original
indices. Merely deleting entries from the official `.pth` files is not
recalibration: it selects a subset of parameters learned on the original
full-depth activation path.

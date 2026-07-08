# QuantVGGT Deployment Parameters

QuantVGGT deployment uses two precomputed calibration-state files published in
the official Hugging Face repository. They are optional deployment parameters,
not dataset samples and not VGGT/LaRe checkpoints.

Download only the two required W4A4 files, resume partial downloads, and verify
their official byte sizes and SHA-256 digests with:

```bash
bash submodules/setup_quantvggt_params.sh
```

The default destination is:

```text
submodules/quantvggt/w4a4_official
```

To select another location:

```bash
QUANTVGGT_PARAMS_DIR=/path/to/quantvggt/w4a4 \
  bash submodules/setup_quantvggt_params.sh
```

Use the downloaded parameters during evaluation with:

```bash
INFERENCE_COMPRESSION=quantvggt \
QUANTVGGT_DTYPE=w4a4 \
QUANTVGGT_PARAMS_DIR=submodules/quantvggt/w4a4_official \
  bash run_eval.sh
```

The downloader preserves a same-size file that fails verification and stops by
default. Set `QUANTVGGT_FORCE_DOWNLOAD=1` to rename that file with an `.invalid`
suffix and fetch a clean copy.

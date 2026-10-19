# LaRe

Layer pruning and calibrated recovery for visual geometry transformers.

LaRe supports VGGT, Pi3, and VGGT-Omega through a shared analysis, pruning,
recovery, and evaluation pipeline. You can also use the calibrated models
directly, without collecting activations or running calibration.

## Installation

```bash
conda create -n vggt python=3.10
conda activate vggt
pip install -r requirements.txt
python backbones/install_backbone.py vggt pi3 vggt_omega --no-build-extensions
```

Run commands from the repository root. The installer uses the revisions in
[`backbones/backbones.lock.json`](backbones/backbones.lock.json). CUDA is used
for the experiments and the inference examples below. Pi3's optional CUDA
extension can be built through the installer when required by your setup.

## Calibrated models

The model definition is provided by [`lare/model.py`](lare/model.py), using the
pinned upstream backbone definitions. Two weight formats are available:

| Backbone | Dropped units | Selected interval | Standalone weights | Recovery delta |
|---|---:|---|---:|---:|
| VGGT | 10 | `i8_j17` | 4.11 GiB | 8.03 MiB |
| Pi3 | 16 | `i3_j18` | 2.83 GiB | 4.03 MiB |
| VGGT-Omega | 18 | `i1_j18` | 3.43 GiB | 4.03 MiB |

**Standalone weights** contain the complete pruned model and its recovery
operators. Inference does not download the original pretrained weights.
**Recovery deltas** contain the pruning definition and calibrated operators;
they are applied to the original pretrained model automatically.

The artifacts have been prepared locally. Public download URLs have not yet
been published. File names, sizes, and SHA-256 checksums are recorded in
[`models/standalone_manifest.json`](models/standalone_manifest.json) and
[`models/recovery_manifest.json`](models/recovery_manifest.json). Place the
obtained files in this layout:

```text
checkpoints/
├── vggt/         model.pt  recovery.pt
├── pi3/          model.pt  recovery.pt
└── vggt_omega/   model.pt  recovery.pt
```

### Minimal inference

```python
from lare import LaRe

model = LaRe.from_pretrained(
    "vggt",  # also: "pi3", "vggt_omega"
    checkpoint="checkpoints/vggt/model.pt",
    device="cuda",
)
images = model.load_images(["image_a.jpg", "image_b.jpg"])
predictions = model(images)

points = predictions["points"]
camera_poses = predictions["camera_poses"]  # camera-to-world matrices
```

`load_images` applies the appropriate evaluation transform for each backbone
and returns RGB tensors `[S, 3, H, W]` in `[0, 1]`. Forward also accepts
`[B, S, 3, H, W]` and handles inference mode and mixed precision. Keep model
weights in FP32; do not call `.half()` or `.bfloat16()`.

The returned dictionary contains the available geometry fields. VGGT and
VGGT-Omega provide depth and camera predictions, with points reconstructed
from camera and depth. Pi3 provides world points, local points, confidence,
and camera poses; it does not provide a separate `depth` field.

For a directory of images:

```bash
PYTHONPATH=. python examples/infer.py \
  --backbone vggt \
  --checkpoint checkpoints/vggt/model.pt \
  --images path/to/images \
  --output outputs/predictions.pt
```

### Using compact recovery checkpoints

```python
from lare import load_calibrated

model = load_calibrated("pi3", checkpoint="checkpoints/pi3/recovery.pt")
predictions = model(model.load_images(["image_a.jpg", "image_b.jpg"]))
```

For this format, VGGT and Pi3 load `facebook/VGGT-1B` and `yyfz233/Pi3` through
Hugging Face. VGGT-Omega expects its original `vggt_omega_1b_512.pt` at
`backbones/vggt-omega/`, or an explicit `base_model="path/to/base.pt"`.
The original Omega weights require access through the
[upstream model page](https://huggingface.co/facebook/VGGT-Omega).
The standalone format does not require the base file at inference time.

## Reproduce the experiments

Prepare the datasets with [`data/README_data_setup.md`](data/README_data_setup.md).
Keep dataset payloads outside Git; `data/` may contain local dataset symlinks.

```bash
# Full analysis pipeline, including activation caching and recovery.
conda run --no-capture-output -n vggt bash analysis.sh

# Formal experiment pipeline; imports reusable analysis artifacts first.
conda run --no-capture-output -n vggt bash main.sh
```

Both entrypoints execute their complete sequence. They preserve existing
result reuse and resume behavior. `main.sh` requires the analysis results
under `REFERENCE_RESULTS_ROOT` (default `analysis_outputs_v3`); it is not an
independent replacement for the analysis pipeline.

Override parameters through environment variables:

```bash
BACKBONES=pi3 DEVICE=cuda MAX_SCENES=2 \
  conda run --no-capture-output -n vggt bash analysis.sh
```

Analysis plan selection and recovery use `scannet_dev50,nuscenes_train` by
default. The formal experiment calibrates on four training datasets. These
different scopes are intentional. See [`docs/pipelines.md`](docs/pipelines.md)
for the stage map, parameters, result layout, and compression settings.

Additional experiments live in [`experiments/`](experiments/): token,
calibration-sample, and pruning-heuristic ablations, long-sequence evaluation,
and the ReplaceMe baseline. Maintenance runners are under
[`scripts/maintenance/`](scripts/maintenance/).

## Visualization

The local research interface remains available:

```bash
conda run --no-capture-output -n vggt python -m analysis.view
```

[`web/`](web/) contains a static project webpage with result filters, CSV
download, analysis figures, and sampled interactive 3D examples. It uses only
the exported visualization data, without a Python or Gradio server.
See [`docs/webpage.md`](docs/webpage.md) for export and GitHub Pages deployment.

## Release artifacts

The release includes source code, visualization assets, and model/recovery
checkpoints. Full experiment trees, activation caches, and dataset payloads
are not bundled.

```bash
# Compact recovery checkpoints for the three calibrated operating points.
conda run --no-capture-output -n vggt python -m lare.export

# Complete pruned model weights; requires the original weights once at export.
conda run --no-capture-output -n vggt python -m lare.export --standalone

# Only the numerical data and sampled geometry needed by the webpage.
conda run --no-capture-output -n vggt python -m analysis.results.export_web

# Recovery checkpoints referenced by current public analysis/formal plans.
conda run --no-capture-output -n vggt python -m analysis.results.export_recovery_checkpoints
```

Checkpoint selection uses calibration data and the existing shared selection
rule. The current VGGT/Pi3 Double candidates lack complete recovery summaries
at the selected operating points; the complete Single candidates are exported.
Missing candidates are not treated as measured comparisons.

See [`docs/release.md`](docs/release.md) for artifact scope and publication steps.

## Tests

Regression tests are in [`tests/`](tests/). Run them outside the sandbox in
the `vggt` environment, with CUDA available:

```bash
conda run --no-capture-output -n vggt python -m pip install -r requirements-dev.txt
conda run --no-capture-output -n vggt python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
conda run --no-capture-output -n vggt python -m pytest tests -q
```

## Upstream projects

LaRe builds on [VGGT](https://github.com/facebookresearch/vggt),
[Pi3](https://github.com/yyfz/Pi3),
[VGGT-Omega](https://github.com/facebookresearch/vggt-omega),
[FastVGGT](https://github.com/mystorm16/FastVGGT), and
[QuantVGGT](https://github.com/wlfeng0509/QuantVGGT).
See [`NOTICE.md`](NOTICE.md) and the upstream repositories for their licenses
and citation information. The license for LaRe-authored code and the final
paper citation must be specified before public release.

# Does the VGGT Family Need All Its Layers?

Layer pruning and calibrated recovery for visual geometry transformers.

This project supports VGGT, Pi3, and VGGT-Omega through a shared analysis, pruning,
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

Our calibration checkpoints are included in this repository. The
[model interface](models/model.py) combines the official pretrained weights
with our checkpoints, removes the selected layers, and applies the calibrated
recovery operators. No calibration or training is needed to run inference.
To reproduce the full pipeline, see [Reproduce the experiments](#reproduce-the-experiments).

| Backbone | Dropped / total layers | Aggregator params (M), original → calibrated | Calibration checkpoint |
|---|---:|---:|---|
| VGGT | 10/48 | 605 → 479 | [8.03 MiB](checkpoints/vggt/recovery.pt) |
| Pi3 | 16/36 | 454 → 252 | [4.03 MiB](checkpoints/pi3/recovery.pt) |
| VGGT-Omega | 18/48 | 605 → 378 | [4.03 MiB](checkpoints/vggt_omega/recovery.pt) |

Parameter counts cover aggregator blocks only, as in Table 1.
After [installation](#installation), VGGT and Pi3 automatically download their
official weights on first use. For VGGT-Omega, first [prepare its official
weights](#prepare-original-weights). Run from the repository root:

```python
from models import load_calibrated

model = load_calibrated("vggt", device="cuda")  # or "pi3", "vggt_omega"
predictions = model(model.load_images(["image_a.jpg", "image_b.jpg"]))
points = predictions["points"]
camera_poses = predictions["camera_poses"]
```

For a directory of images:

```bash
PYTHONPATH=. python examples/infer.py --backbone vggt --images path/to/images
```

## Reproduce the experiments

### Prepare original weights

- **VGGT:** downloaded automatically from `facebook/VGGT-1B` on first use.
- **Pi3:** downloaded automatically from `yyfz233/Pi3` on first use.
- **VGGT-Omega:** obtain access to `facebook/VGGT-Omega`, authenticate with
  Hugging Face, then download the checkpoint with the repository installer:

  ```bash
  python backbones/install_backbone.py vggt_omega --download-ckpt --no-build-extensions
  ```

  The expected path is `backbones/vggt-omega/vggt_omega_1b_512.pt`.

The inference interface combines these official weights with the included
`checkpoints/<backbone>/recovery.pt`. The pipeline below is only needed to
reproduce pruning and calibration yourself.

### Run the pipeline

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

Maintenance runners are under
[`scripts/maintenance/`](scripts/maintenance/).

## Visualization

The local research interface remains available:

```bash
conda run --no-capture-output -n vggt python -m analysis.view
```

[`web/`](web/) contains a static project webpage with the six analysis/formal
result tabs, their controls, downloadable plots, and filterable tables. It uses only
the exported visualization data, without a Python or Gradio server.
See [`docs/webpage.md`](docs/webpage.md) for export and GitHub Pages deployment.

## Release artifacts

The release includes source code, visualization assets, and model/recovery
checkpoints. Full experiment trees, activation caches, and dataset payloads
are not bundled.

```bash
# Compact recovery checkpoints for the three calibrated operating points.
conda run --no-capture-output -n vggt python -m models.export

# Complete pruned model weights; requires the original weights once at export.
conda run --no-capture-output -n vggt python -m models.export --standalone

# Only the cached matrices, curves and tables needed by the dashboard.
conda run --no-capture-output -n vggt python -m analysis.results.export_dashboard

# Recovery checkpoints referenced by current public analysis/formal plans.
conda run --no-capture-output -n vggt python -m analysis.results.export_recovery_checkpoints
```

Checkpoint selection uses calibration data and the existing shared selection
rule. The current VGGT/Pi3 Double candidates lack complete recovery summaries
at the selected operating points; the complete Single candidates are exported.
Missing candidates are not treated as measured comparisons.

See [`docs/release.md`](docs/release.md) for artifact scope and publication steps.

## Upstream projects and licensing

Third-party sources are pinned in
[`backbones/backbones.lock.json`](backbones/backbones.lock.json) and referenced
by `.gitmodules`. Their original license and attribution files remain
authoritative for those components and their pretrained weights. Refer to
the upstream repositories for their citation information.

| Component | Source |
|---|---|
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| Pi3 | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| VGGT-Omega | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) |
| FastVGGT | [mystorm16/FastVGGT](https://github.com/mystorm16/FastVGGT) |
| QuantVGGT | [wlfeng0509/QuantVGGT](https://github.com/wlfeng0509/QuantVGGT) |

The license for project-authored code and the final paper citation must be
specified before public release. This attribution does not relicense
third-party code, datasets, or model weights.

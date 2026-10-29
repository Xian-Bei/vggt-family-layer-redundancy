# Release contents

The public source consists of the two main shell entrypoints, `analysis/`,
`models/` (including release manifests in `models/metadata/`), `examples/`, `scripts/`, `docs/`,
`web/`, dependency specifications, backbone installer/lock, `.gitmodules`,
dataset setup tools, publication metadata, and the three selected
`checkpoints/{vggt,pi3,vggt_omega}/recovery.pt` files.

Keep these local: `experiments/`, `tests/`, `pytest.ini`, `private_research/`,
`analysis_outputs*/`, `main_results*/`,
dataset payloads and symlinks, `test_length_results/`, pretrained vendor
weights, editor configuration, tool caches, and `paper/` working drafts.
`.gitignore` does not remove files already present in Git history; inspect the
actual release diff rather than relying on ignore rules alone.

## Model files

`python -m models.export` exports three selected recovery checkpoints.
`python -m models.export --standalone` assembles complete inference weights.
Neither command reruns calibration or evaluates held-out data to choose a
winner. The original experiment checkpoints stay in place.

The three compact recovery checkpoints are versioned in `checkpoints/`, so a
normal clone includes them. All other files under that directory, including
multi-GiB standalone weights, remain ignored. Manifests in `models/metadata/`
record file sizes and SHA-256 checksums. `load_calibrated(backbone)` uses the
included checkpoint with the official pretrained base by default; standalone
weights remain an optional local export.

Original upstream weight and code license terms continue to apply to derived
standalone artifacts. Project code licensing remains a project-owner decision.
No license or publication destination is assumed by the export tools.

## Recovery checkpoint collection

`python -m analysis.results.export_recovery_checkpoints` stages the existing
checkpoints referenced by current public pruning strategies and recovery
methods from both result roots. It excludes obsolete selectors, private
research outputs, activation caches, and numerical result trees. The local
directory is `release_artifacts/recovery_checkpoints/`; no compressed archive
or upload is created. The collection manifest in `models/metadata/` records checkpoint
paths, sizes, and checksums for review.

## Visualization files

`python -m analysis.results.export_dashboard` produces the data actually consumed by
the static webpage. Commit `web/data/`, not entire experiment trees. The
manifest records every asset and its checksum.

The full original output directories remain available for continued research.
They are not archived or uploaded as part of this preparation.

## Publication

Review the source changes and the model/visualization manifests, complete local
GPU validation in the `vggt` environment, and commit the three compact recovery
checkpoints with their manifests. Run the manual Pages workflow separately to
update the website. No Git commit, remote push, model
upload, or website deployment is performed by these local export commands.

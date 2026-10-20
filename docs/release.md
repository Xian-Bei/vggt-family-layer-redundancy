# Release contents

The public source consists of the two main shell entrypoints, `analysis/`,
`lare/` (including release manifests in `lare/metadata/`), `examples/`, `scripts/`, `docs/`,
`web/`, dependency specifications, backbone installer/lock, `.gitmodules`,
dataset setup tools, and publication metadata.

Keep these local: `experiments/`, `tests/`, `pytest.ini`, `private_research/`,
`analysis_outputs*/`, `main_results*/`,
dataset payloads and symlinks, `test_length_results/`, pretrained vendor
weights, editor configuration, tool caches, and `paper/` working drafts.
`.gitignore` does not remove files already present in Git history; inspect the
actual release diff rather than relying on ignore rules alone.

## Model files

`python -m lare.export` exports three selected recovery checkpoints.
`python -m lare.export --standalone` assembles complete inference weights.
Neither command reruns calibration or evaluates held-out data to choose a
winner. The original experiment checkpoints stay in place.

The generated files live in ignored `checkpoints/`. Their small manifests are
copied into `lare/metadata/` for review. Standalone checkpoints contain the pinned
backbone state plus pruning/recovery topology; compact checkpoints rely on
the original pretrained base. Each format loads through `LaRe.from_pretrained`.

Original upstream weight and code license terms continue to apply to derived
standalone artifacts. LaRe code licensing, author information, final paper
citation, and artifact hosting URLs still require project-owner decisions.
No license or publication destination is assumed by the export tools.

## Recovery checkpoint collection

`python -m analysis.results.export_recovery_checkpoints` stages the existing
checkpoints referenced by current public pruning strategies and recovery
methods from both result roots. It excludes obsolete selectors, private
research outputs, activation caches, and numerical result trees. The local
directory is `release_artifacts/recovery_checkpoints/`; no compressed archive
or upload is created. The collection manifest in `lare/metadata/` records checkpoint
paths, sizes, and checksums for review.

## Visualization files

`python -m analysis.results.export_web` produces the data actually consumed by
the static webpage. Commit `web/data/`, not entire experiment trees. The
manifest records every asset and its checksum.

The full original output directories remain available for continued research.
They are not archived or uploaded as part of this preparation.

## Publication

Review the source changes and the model/visualization manifests, complete local
GPU validation in the `vggt` environment, choose the project license and citation, and
publish model files to the selected artifact host. Then add the real download
URLs and run the manual Pages workflow. No Git commit, remote push, model
upload, or website deployment is performed by these local export commands.

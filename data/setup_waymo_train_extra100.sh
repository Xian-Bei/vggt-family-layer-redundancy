#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Independent training pool. Never change the main training/validation data.
export WAYMO_RAW_ROOT="${WAYMO_RAW_ROOT:-/media/bb/waymo_raw_extra100}"
export WAYMO_TARGET_ROOT="${WAYMO_TARGET_ROOT:-${REPO_ROOT}/data/waymo_train_extra100}"
export WAYMO_SCENES_PER_SPLIT=100
export WAYMO_SPLITS=training
export WAYMO_EXTRACT_CONDA_ENV="${WAYMO_EXTRACT_CONDA_ENV:-waymo_new}"
exclusions=(
  --exclude-scene-root "${REPO_ROOT}/data/waymo/training"
  --exclude-scene-root "${REPO_ROOT}/data/waymo/validation"
)
if [[ -d /media/cc/TALO/Data/waymo ]]; then
  exclusions+=(--exclude-scene-root /media/cc/TALO/Data/waymo)
fi
python data/download_waymo_subset.py \
  --target-root "${WAYMO_RAW_ROOT}" --splits training --count 100 \
  --seed 20260916 "${exclusions[@]}"
RUN_WAYMO_DOWNLOAD=0 bash data/setup_waymo.sh
python - <<'PY'
import json
import os
from pathlib import Path
raw = Path(os.environ['WAYMO_RAW_ROOT'])
target = Path(os.environ['WAYMO_TARGET_ROOT'])
manifest = json.loads((raw / 'manifests/training_extra_selection.json').read_text())
ids = manifest['scene_ids']
assert len(ids) == len(set(ids)) == 100
assert {p.name for p in (target / 'training').iterdir() if p.is_dir()} == set(ids)
for root in manifest['exclude_scene_roots']:
    assert not set(ids) & {p.name for p in Path(root).iterdir() if p.is_dir()}
(target / 'selected_scenes.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('[extra100] 100 training scenes prepared, sparse depth validated, no existing-scene overlap.')
PY

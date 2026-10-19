#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NUSCENES_RAW_ROOT="${NUSCENES_RAW_ROOT:-/media/bb/nuscenes}"
TARGET_DIR="${TARGET_DIR:-${REPO_ROOT}/data/nuscenes_train_extra100}"
BASE_MANIFEST="${BASE_MANIFEST:-${REPO_ROOT}/data/nuscenes/selected_scenes.json}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
python data/extract_nuscenes.py \
  --data_root "${NUSCENES_RAW_ROOT}" \
  --target_dir "${TARGET_DIR}" \
  --splits train \
  --num_train_scenes "${TRAIN_COUNT:-100}" \
  --seed "${NUSCENES_SEED:-20260915}" \
  --exclude_manifest "${BASE_MANIFEST}" \
  --fail_if_insufficient \
  --preserve_sample_order \
  --skip_existing

#!/usr/bin/env bash
set -euo pipefail

# One-off backfill for already extracted nuScenes and Waymo folders.
# Only depth_points/ and validation reports are written; existing RGB, LiDAR,
# poses and calibration files are never removed or rewritten.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"

RUN_NUSCENES_DEPTH="${RUN_NUSCENES_DEPTH:-1}"
RUN_WAYMO_DEPTH="${RUN_WAYMO_DEPTH:-1}"

NUSCENES_RAW_ROOT="${NUSCENES_RAW_ROOT:-/media/fengyi/bb/nuscenes}"
NUSCENES_TARGET_ROOT="${NUSCENES_TARGET_ROOT:-${DATA_DIR}/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
NUSCENES_CONDA_ENV="${NUSCENES_CONDA_ENV:-vggt}"
NUSCENES_PYTHON="${NUSCENES_PYTHON:-}"

WAYMO_RAW_ROOT="${WAYMO_RAW_ROOT:-${DATA_DIR}/waymo_raw}"
WAYMO_TARGET_ROOT="${WAYMO_TARGET_ROOT:-${DATA_DIR}/waymo}"
WAYMO_SPLITS="${WAYMO_SPLITS:-training validation}"
WAYMO_CONDA_ENV="${WAYMO_CONDA_ENV:-waymo_new}"
WAYMO_PYTHON="${WAYMO_PYTHON:-}"

DEPTH_MIN="${DEPTH_MIN:-0.1}"
DEPTH_MAX="${DEPTH_MAX:-80.0}"
DEPTH_MIN_SAMPLES="${DEPTH_MIN_SAMPLES:-100}"

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

run_nuscenes_python() {
  if [[ -n "${NUSCENES_PYTHON}" ]]; then
    "${NUSCENES_PYTHON}" "$@"
  else
    conda run --no-capture-output -n "${NUSCENES_CONDA_ENV}" python "$@"
  fi
}

run_waymo_python() {
  if [[ -n "${WAYMO_PYTHON}" ]]; then
    "${WAYMO_PYTHON}" "$@"
  else
    conda run --no-capture-output -n "${WAYMO_CONDA_ENV}" python "$@"
  fi
}

echo "== Driving sparse-depth backfill =="

if truthy "${RUN_NUSCENES_DEPTH}"; then
  echo "[nuScenes] official projection -> ${NUSCENES_TARGET_ROOT}/<scene>/depth_points/CAM_FRONT"
  run_nuscenes_python "${DATA_DIR}/extract_nuscenes.py" \
    --data_root "${NUSCENES_RAW_ROOT}" \
    --target_dir "${NUSCENES_TARGET_ROOT}" \
    --version "${NUSCENES_VERSION}" \
    --depth_only \
    --min_depth "${DEPTH_MIN}" \
    --max_depth "${DEPTH_MAX}" \
    --min_depth_samples "${DEPTH_MIN_SAMPLES}"
  python "${DATA_DIR}/validate_sparse_depth.py" "${NUSCENES_TARGET_ROOT}" \
    --min-samples "${DEPTH_MIN_SAMPLES}"
else
  echo "[skip] RUN_NUSCENES_DEPTH=${RUN_NUSCENES_DEPTH}"
fi

if truthy "${RUN_WAYMO_DEPTH}"; then
  for split in ${WAYMO_SPLITS}; do
    raw_split="${WAYMO_RAW_ROOT}/${split}"
    target_split="${WAYMO_TARGET_ROOT}/${split}"
    if [[ ! -d "${raw_split}" ]]; then
      echo "error: Waymo raw split not found: ${raw_split}" >&2
      exit 1
    fi
    if [[ ! -d "${target_split}" ]]; then
      echo "error: Waymo extracted split not found: ${target_split}" >&2
      exit 1
    fi
    echo "[Waymo/${split}] official projection -> ${target_split}/<scene>/depth_points/FRONT"
    run_waymo_python "${DATA_DIR}/extract_waymo.py" \
      --data_root "${raw_split}" \
      --target_dir "${target_split}" \
      --depth_only \
      --min_depth "${DEPTH_MIN}" \
      --max_depth "${DEPTH_MAX}" \
      --min_depth_samples "${DEPTH_MIN_SAMPLES}"
    python "${DATA_DIR}/validate_sparse_depth.py" "${target_split}" \
      --min-samples "${DEPTH_MIN_SAMPLES}"
  done
else
  echo "[skip] RUN_WAYMO_DEPTH=${RUN_WAYMO_DEPTH}"
fi

echo "Driving sparse-depth backfill completed."

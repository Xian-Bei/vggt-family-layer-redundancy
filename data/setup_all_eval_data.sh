#!/usr/bin/env bash
set -euo pipefail

# One-click setup for all evaluation datasets used in this repo:
# 1) ScanNet test50 + dev50
# 2) 7Scenes
# 3) Neural RGB-D
# 4) nuScenes front-camera subset (requires local raw nuScenes)
# 5) Waymo front-camera subset (optional; requires gcloud auth + Waymo terms)
# 6) QuantVGGT official W4A4 deployment parameters (optional; about 3.83 GB)
#
# Usage:
#   bash data/setup_all_eval_data.sh
#
# Optional:
#   CLEAN_DOWNLOADS=1 bash data/setup_all_eval_data.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"
CLEAN_DOWNLOADS="${CLEAN_DOWNLOADS:-0}"
RUN_NUSCENES="${RUN_NUSCENES:-1}"
RUN_WAYMO="${RUN_WAYMO:-0}"
RUN_QUANTVGGT_PARAMS="${RUN_QUANTVGGT_PARAMS:-0}"
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-${DATA_DIR}/quantvggt/w4a4_official}"
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/media/fengyi/bb/nuscenes}"
NUSCENES_TARGET_DIR="${NUSCENES_TARGET_DIR:-${DATA_DIR}/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
NUSCENES_SEED="${NUSCENES_SEED:-0}"
NUSCENES_SCENES_PER_SPLIT="${NUSCENES_SCENES_PER_SPLIT:-50}"
NUSCENES_EXTRACT_CONDA_ENV="${NUSCENES_EXTRACT_CONDA_ENV:-vggt}"
NUSCENES_EXTRACT_PYTHON="${NUSCENES_EXTRACT_PYTHON:-}"
DRIVING_DEPTH_MIN="${DRIVING_DEPTH_MIN:-0.1}"
DRIVING_DEPTH_MAX="${DRIVING_DEPTH_MAX:-80.0}"
DRIVING_DEPTH_MIN_SAMPLES="${DRIVING_DEPTH_MIN_SAMPLES:-100}"
WAYMO_RAW_ROOT="${WAYMO_RAW_ROOT:-${DATA_DIR}/waymo_raw}"
WAYMO_TARGET_ROOT="${WAYMO_TARGET_ROOT:-${DATA_DIR}/waymo}"
WAYMO_SEED="${WAYMO_SEED:-20260616}"
WAYMO_SCENES_PER_SPLIT="${WAYMO_SCENES_PER_SPLIT:-50}"
WAYMO_EXTRACT_CONDA_ENV="${WAYMO_EXTRACT_CONDA_ENV:-waymo_new}"
WAYMO_EXTRACT_PYTHON="${WAYMO_EXTRACT_PYTHON:-}"
RUN_WAYMO_DOWNLOAD="${RUN_WAYMO_DOWNLOAD:-1}"
RUN_WAYMO_EXTRACT="${RUN_WAYMO_EXTRACT:-1}"
WAYMO_FORCE_EXTRACT="${WAYMO_FORCE_EXTRACT:-0}"

run_nuscenes_python() {
  if [[ -n "${NUSCENES_EXTRACT_PYTHON}" ]]; then
    "${NUSCENES_EXTRACT_PYTHON}" "$@"
  else
    conda run --no-capture-output -n "${NUSCENES_EXTRACT_CONDA_ENV}" python "$@"
  fi
}

echo "== Setup all eval datasets =="
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo

echo "[1/6] ScanNet splits + extraction"
bash "${DATA_DIR}/setup_scannet.sh"
echo

echo "[2/6] 7Scenes"
bash "${DATA_DIR}/setup_7scenes.sh"
echo

echo "[3/6] Neural RGB-D"
bash "${DATA_DIR}/setup_nrgbd.sh"
echo

echo "[4/6] nuScenes front-camera subset"
if [[ "${RUN_NUSCENES}" == "0" || "${RUN_NUSCENES}" == "false" ]]; then
  echo "[skip] RUN_NUSCENES=${RUN_NUSCENES}"
elif [[ ! -d "${NUSCENES_DATA_ROOT}" ]]; then
  echo "[skip] nuScenes raw root not found: ${NUSCENES_DATA_ROOT}"
  echo "       Set NUSCENES_DATA_ROOT=/path/to/nuscenes or RUN_NUSCENES=0."
else
  run_nuscenes_python -c "from nuscenes.nuscenes import NuScenes; print('[nuscenes] devkit import ok')"
  run_nuscenes_python "${DATA_DIR}/extract_nuscenes.py" \
    --data_root "${NUSCENES_DATA_ROOT}" \
    --target_dir "${NUSCENES_TARGET_DIR}" \
    --version "${NUSCENES_VERSION}" \
    --seed "${NUSCENES_SEED}" \
    --num_scenes_per_split "${NUSCENES_SCENES_PER_SPLIT}" \
    --min_depth "${DRIVING_DEPTH_MIN}" \
    --max_depth "${DRIVING_DEPTH_MAX}" \
    --min_depth_samples "${DRIVING_DEPTH_MIN_SAMPLES}"
  python "${DATA_DIR}/validate_sparse_depth.py" "${NUSCENES_TARGET_DIR}" \
    --min-samples "${DRIVING_DEPTH_MIN_SAMPLES}"
fi
echo

echo "[5/6] Waymo front-camera subset"
if [[ "${RUN_WAYMO}" == "0" || "${RUN_WAYMO}" == "false" ]]; then
  echo "[skip] RUN_WAYMO=${RUN_WAYMO}"
  echo "       Set RUN_WAYMO=1 after accepting Waymo terms and running gcloud auth login."
else
  WAYMO_RAW_ROOT="${WAYMO_RAW_ROOT}" \
  WAYMO_TARGET_ROOT="${WAYMO_TARGET_ROOT}" \
  WAYMO_SEED="${WAYMO_SEED}" \
  WAYMO_SCENES_PER_SPLIT="${WAYMO_SCENES_PER_SPLIT}" \
  WAYMO_EXTRACT_CONDA_ENV="${WAYMO_EXTRACT_CONDA_ENV}" \
  WAYMO_EXTRACT_PYTHON="${WAYMO_EXTRACT_PYTHON}" \
  RUN_WAYMO_DOWNLOAD="${RUN_WAYMO_DOWNLOAD}" \
  RUN_WAYMO_EXTRACT="${RUN_WAYMO_EXTRACT}" \
  WAYMO_FORCE_EXTRACT="${WAYMO_FORCE_EXTRACT}" \
  DRIVING_DEPTH_MIN="${DRIVING_DEPTH_MIN}" \
  DRIVING_DEPTH_MAX="${DRIVING_DEPTH_MAX}" \
  DRIVING_DEPTH_MIN_SAMPLES="${DRIVING_DEPTH_MIN_SAMPLES}" \
    bash "${DATA_DIR}/setup_waymo.sh"
fi
echo

echo "[6/6] QuantVGGT official W4A4 deployment parameters"
if [[ "${RUN_QUANTVGGT_PARAMS}" == "0" || "${RUN_QUANTVGGT_PARAMS}" == "false" ]]; then
  echo "[skip] RUN_QUANTVGGT_PARAMS=${RUN_QUANTVGGT_PARAMS}"
  echo "       Set RUN_QUANTVGGT_PARAMS=1 to download and verify the 3.83 GB official parameters."
else
  QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR}" \
    bash "${DATA_DIR}/setup_quantvggt_params.sh"
fi
echo

if [[ "${CLEAN_DOWNLOADS}" == "1" ]]; then
  echo "Cleaning download caches..."
  rm -rf "${DATA_DIR}/downloads_7scenes" "${DATA_DIR}/downloads_nrgbd"
fi

echo "All dataset setup steps completed."

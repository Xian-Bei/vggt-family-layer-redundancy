#!/usr/bin/env bash
set -euo pipefail

# Download and extract a reproducible Waymo front-camera subset.
#
# Requirements:
#   1) Accept the Waymo Open Dataset terms in the browser.
#   2) Run `gcloud auth login` with an account that has bucket access.
#   3) Use a Python environment with the Waymo devkit for extraction.
#      On this machine, `waymo_new` is known to work.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"

WAYMO_BUCKET="${WAYMO_BUCKET:-gs://waymo_open_dataset_v_1_4_3/individual_files}"
WAYMO_RAW_ROOT="${WAYMO_RAW_ROOT:-${DATA_DIR}/waymo_raw}"
WAYMO_TARGET_ROOT="${WAYMO_TARGET_ROOT:-${DATA_DIR}/waymo}"
WAYMO_SEED="${WAYMO_SEED:-20260616}"
WAYMO_SCENES_PER_SPLIT="${WAYMO_SCENES_PER_SPLIT:-50}"
WAYMO_SPLITS="${WAYMO_SPLITS:-training validation}"
WAYMO_EXTRACT_CONDA_ENV="${WAYMO_EXTRACT_CONDA_ENV:-waymo_new}"
WAYMO_EXTRACT_PYTHON="${WAYMO_EXTRACT_PYTHON:-}"
RUN_WAYMO_DOWNLOAD="${RUN_WAYMO_DOWNLOAD:-1}"
RUN_WAYMO_EXTRACT="${RUN_WAYMO_EXTRACT:-1}"
WAYMO_FORCE_EXTRACT="${WAYMO_FORCE_EXTRACT:-0}"
DRIVING_DEPTH_MIN="${DRIVING_DEPTH_MIN:-0.1}"
DRIVING_DEPTH_MAX="${DRIVING_DEPTH_MAX:-80.0}"
DRIVING_DEPTH_MIN_SAMPLES="${DRIVING_DEPTH_MIN_SAMPLES:-100}"

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "error: required command not found: ${cmd}" >&2
    return 1
  fi
}

check_gcloud_auth() {
  require_cmd gcloud
  if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
    echo "error: no active gcloud account. Run: gcloud auth login" >&2
    return 1
  fi
}

run_extract_python() {
  if [[ -n "${WAYMO_EXTRACT_PYTHON}" ]]; then
    "${WAYMO_EXTRACT_PYTHON}" "$@"
  else
    conda run --no-capture-output -n "${WAYMO_EXTRACT_CONDA_ENV}" python "$@"
  fi
}

check_extract_python() {
  if [[ -z "${WAYMO_EXTRACT_PYTHON}" ]]; then
    require_cmd conda
  elif [[ ! -x "${WAYMO_EXTRACT_PYTHON}" ]]; then
    echo "error: WAYMO_EXTRACT_PYTHON is not executable: ${WAYMO_EXTRACT_PYTHON}" >&2
    return 1
  fi
  run_extract_python - <<'PY'
from waymo_open_dataset import dataset_pb2  # noqa: F401
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection  # noqa: F401
import tensorflow as tf
print(f"[waymo] extract python ok: tensorflow={tf.__version__}")
PY
}

count_raw_split() {
  local split="$1"
  if [[ ! -d "${WAYMO_RAW_ROOT}/${split}" ]]; then
    echo 0
    return 0
  fi
  find "${WAYMO_RAW_ROOT}/${split}" -maxdepth 1 -type f -name '*.tfrecord' 2>/dev/null | wc -l
}

count_extracted_split() {
  local split="$1"
  if [[ ! -d "${WAYMO_TARGET_ROOT}/${split}" ]]; then
    echo 0
    return 0
  fi
  find "${WAYMO_TARGET_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

count_depth_ready_split() {
  local split="$1"
  local count=0
  local scene
  if [[ ! -d "${WAYMO_TARGET_ROOT}/${split}" ]]; then
    echo 0
    return 0
  fi
  while IFS= read -r scene; do
    [[ -z "${scene}" ]] && continue
    local image_count depth_count
    image_count=0
    depth_count=0
    if [[ -d "${scene}/image/FRONT" ]]; then
      image_count="$(find "${scene}/image/FRONT" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
    fi
    if [[ -d "${scene}/depth_points/FRONT" ]]; then
      depth_count="$(find "${scene}/depth_points/FRONT" -maxdepth 1 -type f -name '*.npz' | wc -l)"
    fi
    if [[ "${image_count}" -gt 0 && "${image_count}" -eq "${depth_count}" ]]; then
      count=$((count + 1))
    fi
  done < <(find "${WAYMO_TARGET_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
  echo "${count}"
}

echo "== Waymo setup =="
echo "bucket=${WAYMO_BUCKET}"
echo "raw_root=${WAYMO_RAW_ROOT}"
echo "target_root=${WAYMO_TARGET_ROOT}"
echo "seed=${WAYMO_SEED}"
echo "scenes_per_split=${WAYMO_SCENES_PER_SPLIT}"
echo "splits=${WAYMO_SPLITS}"
if [[ -n "${WAYMO_EXTRACT_PYTHON}" ]]; then
  echo "extract_python=${WAYMO_EXTRACT_PYTHON}"
else
  echo "extract_conda_env=${WAYMO_EXTRACT_CONDA_ENV}"
fi
echo

if truthy "${RUN_WAYMO_DOWNLOAD}"; then
  check_gcloud_auth
  python "${DATA_DIR}/download_waymo_subset.py" \
    --bucket "${WAYMO_BUCKET}" \
    --target-root "${WAYMO_RAW_ROOT}" \
    --seed "${WAYMO_SEED}" \
    --count "${WAYMO_SCENES_PER_SPLIT}" \
    --splits ${WAYMO_SPLITS}
else
  echo "[skip] RUN_WAYMO_DOWNLOAD=${RUN_WAYMO_DOWNLOAD}"
fi

echo
for split in ${WAYMO_SPLITS}; do
  raw_count="$(count_raw_split "${split}")"
  echo "[waymo] raw ${split}: ${raw_count} tfrecords"
done

if truthy "${RUN_WAYMO_EXTRACT}"; then
  check_extract_python
  for split in ${WAYMO_SPLITS}; do
    raw_dir="${WAYMO_RAW_ROOT}/${split}"
    target_dir="${WAYMO_TARGET_ROOT}/${split}"
    if [[ ! -d "${raw_dir}" ]]; then
      echo "error: Waymo raw split directory not found: ${raw_dir}" >&2
      exit 1
    fi
    raw_count="$(count_raw_split "${split}")"
    if [[ "${raw_count}" -lt "${WAYMO_SCENES_PER_SPLIT}" ]]; then
      echo "error: ${raw_dir} has ${raw_count} tfrecords; expected at least ${WAYMO_SCENES_PER_SPLIT}" >&2
      exit 1
    fi
    extracted_count="$(count_extracted_split "${split}")"
    depth_ready_count="$(count_depth_ready_split "${split}")"
    if [[ "${extracted_count}" -ge "${WAYMO_SCENES_PER_SPLIT}" && "${depth_ready_count}" -ge "${WAYMO_SCENES_PER_SPLIT}" ]] && ! truthy "${WAYMO_FORCE_EXTRACT}"; then
      echo "[skip] extracted ${split}: ${extracted_count} scene dirs, ${depth_ready_count} depth-ready"
      echo "       Set WAYMO_FORCE_EXTRACT=1 to re-extract."
      python "${DATA_DIR}/validate_sparse_depth.py" "${target_dir}" \
        --min-samples "${DRIVING_DEPTH_MIN_SAMPLES}"
      continue
    fi
    if [[ "${extracted_count}" -gt 0 && "${depth_ready_count}" -lt "${WAYMO_SCENES_PER_SPLIT}" ]]; then
      echo "[waymo] existing ${split} extraction lacks official sparse depth; re-extracting"
    fi
    echo "[waymo] extracting ${split}: ${raw_dir} -> ${target_dir}"
    run_extract_python "${DATA_DIR}/extract_waymo.py" \
      --data_root "${raw_dir}" \
      --target_dir "${target_dir}" \
      --min_depth "${DRIVING_DEPTH_MIN}" \
      --max_depth "${DRIVING_DEPTH_MAX}" \
      --min_depth_samples "${DRIVING_DEPTH_MIN_SAMPLES}"
    python "${DATA_DIR}/validate_sparse_depth.py" "${target_dir}" \
      --min-samples "${DRIVING_DEPTH_MIN_SAMPLES}"
    echo "[waymo] extracted ${split}: $(count_extracted_split "${split}") scene dirs"
  done
else
  echo "[skip] RUN_WAYMO_EXTRACT=${RUN_WAYMO_EXTRACT}"
fi

echo "Waymo setup completed."

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BACKBONES="${BACKBONES:-vggt}"

SKIP_GT_EVAL_SCANNET_DEV50="${SKIP_GT_EVAL_SCANNET_DEV50:-0}"
SKIP_GT_EVAL_SCANNET_TEST50="${SKIP_GT_EVAL_SCANNET_TEST50:-0}"
SKIP_GT_EVAL_7SCENES_TEST="${SKIP_GT_EVAL_7SCENES_TEST:-0}"
SKIP_GT_EVAL_7SCENES_TRAIN="${SKIP_GT_EVAL_7SCENES_TRAIN:-0}"
SKIP_GT_EVAL_NRGBD="${SKIP_GT_EVAL_NRGBD:-0}"
SKIP_GT_EVAL_NUSCENES_TRAIN="${SKIP_GT_EVAL_NUSCENES_TRAIN:-0}"
SKIP_GT_EVAL_NUSCENES_VAL="${SKIP_GT_EVAL_NUSCENES_VAL:-0}"
SKIP_GT_EVAL_WAYMO_TRAINING="${SKIP_GT_EVAL_WAYMO_TRAINING:-0}"
SKIP_GT_EVAL_WAYMO_VALIDATION="${SKIP_GT_EVAL_WAYMO_VALIDATION:-0}"

ROOT_SCANNET_DEV50="${ROOT_SCANNET_DEV50:-data/scans_dev50}"
ROOT_SCANNET_TEST50="${ROOT_SCANNET_TEST50:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
ROOT_NUSCENES="${ROOT_NUSCENES:-data/nuscenes}"
ROOT_WAYMO="${ROOT_WAYMO:-data/waymo}"

MAX_SCENES="${MAX_SCENES:-0}"
SKIP_GT_EVAL_MAX_SPAN="${SKIP_GT_EVAL_MAX_SPAN:-8}"
SKIP_GT_EVAL_TIMING="${SKIP_GT_EVAL_TIMING:-0}"
SKIP_GT_EVAL_RESUME="${SKIP_GT_EVAL_RESUME:-1}"

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

run_skip_eval_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local output_dataset="${4:-${dataset}}"
  local split="${5:-}"

  set_backend_pythonpath "${backbone}"

  local extra=(
    --experiment "${backbone}"
    --dataset "${dataset}"
    --data-root "${data_root}"
    --backbone "${backbone}"
    --output-dataset "${output_dataset}"
    --device "${DEVICE}"
    --max-scenes "${MAX_SCENES}"
    --max-frames "${MAX_FRAMES}"
    --max-span "${SKIP_GT_EVAL_MAX_SPAN}"
  )
  if truthy "${SKIP_GT_EVAL_TIMING}"; then
    extra+=(--timing)
  fi
  if ! truthy "${SKIP_GT_EVAL_RESUME}"; then
    extra+=(--no-resume)
  fi
  if [[ -n "${split}" ]]; then
    extra+=(--split "${split}")
  fi

  echo "== skip_gt_eval dataset=${output_dataset} backbone=${backbone} max_span=${SKIP_GT_EVAL_MAX_SPAN} resume=${SKIP_GT_EVAL_RESUME} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.redundancy.skip_gt_eval_v3 \
    "${extra[@]}"
}

run_datasets_for_backbone() {
  local backbone="$1"

  if truthy "${SKIP_GT_EVAL_SCANNET_DEV50}"; then
    run_skip_eval_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "scannet_dev50"
  fi
  if truthy "${SKIP_GT_EVAL_SCANNET_TEST50}"; then
    run_skip_eval_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "scannet_test50"
  fi
  if truthy "${SKIP_GT_EVAL_7SCENES_TEST}"; then
    run_skip_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "7scenes_test" "test"
  fi
  if truthy "${SKIP_GT_EVAL_7SCENES_TRAIN}"; then
    run_skip_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "7scenes_train" "train"
  fi
  if truthy "${SKIP_GT_EVAL_NRGBD}"; then
    run_skip_eval_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "nrgbd"
  fi
  if truthy "${SKIP_GT_EVAL_NUSCENES_VAL}"; then
    run_skip_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "nuscenes_val" "val"
  fi
  if truthy "${SKIP_GT_EVAL_NUSCENES_TRAIN}"; then
    run_skip_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "nuscenes_train" "train"
  fi
  if truthy "${SKIP_GT_EVAL_WAYMO_VALIDATION}"; then
    run_skip_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "waymo_validation" "validation"
  fi
  if truthy "${SKIP_GT_EVAL_WAYMO_TRAINING}"; then
    run_skip_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "waymo_training" "training"
  fi
}

echo "== skip_gt_eval v3 backbones=${BACKBONES} (n=${BB_N}) MAX_FRAMES=${MAX_FRAMES} MAX_SPAN=${SKIP_GT_EVAL_MAX_SPAN} =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  run_datasets_for_backbone "${bb}"
done

echo "== done =="
experiment_output_hint "Skip GT eval"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BACKBONES="${BACKBONES:-vggt_omega}"

RUN_SCANNET_DEV50="${RUN_SCANNET_DEV50:-0}"
RUN_7SCENES_TRAIN="${RUN_7SCENES_TRAIN:-0}"
RUN_7SCENES_TEST="${RUN_7SCENES_TEST:-0}"
RUN_NRGBD="${RUN_NRGBD:-0}"
RUN_NUSCENES_TRAIN="${RUN_NUSCENES_TRAIN:-0}"
RUN_NUSCENES_VAL="${RUN_NUSCENES_VAL:-0}"
RUN_WAYMO_TRAINING="${RUN_WAYMO_TRAINING:-0}"
RUN_WAYMO_VALIDATION="${RUN_WAYMO_VALIDATION:-0}"
CACHE_SCANNET_DEV50="${CACHE_SCANNET_DEV50:-0}"
SKIP_REDUNDANCY="${SKIP_REDUNDANCY:-0}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
TRAINING_PLAN_ROOT="${TRAINING_PLAN_ROOT:-}"
TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME:-}"


#############################################
MAX_SCENES="${MAX_SCENES:-0}"
CACHE_7SCENES_TRAIN="${CACHE_7SCENES_TRAIN:-0}"
CACHE_7SCENES_TEST="${CACHE_7SCENES_TEST:-0}"
CACHE_NRGBD="${CACHE_NRGBD:-0}"
CACHE_NUSCENES_TRAIN="${CACHE_NUSCENES_TRAIN:-0}"
CACHE_NUSCENES_VAL="${CACHE_NUSCENES_VAL:-0}"
CACHE_WAYMO_TRAINING="${CACHE_WAYMO_TRAINING:-0}"
CACHE_WAYMO_VALIDATION="${CACHE_WAYMO_VALIDATION:-0}"
CONDA_ENV="${CONDA_ENV:-vggt}"
SCANNET_ROOT_DEV50="${SCANNET_ROOT_DEV50:-data/scans_dev50}"
SCANNET_ROOT_TEST50="${SCANNET_ROOT_TEST50:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
ROOT_NUSCENES="${ROOT_NUSCENES:-data/nuscenes}"
ROOT_WAYMO="${ROOT_WAYMO:-data/waymo}"
HF_MODEL="${HF_MODEL:-}"


DEVICE="${DEVICE:-cuda}"
MAX_TOKENS="${MAX_TOKENS:-0}"
METRICS="${METRICS:-cka,cosine,neg_mse}"
MAX_SPAN="${MAX_SPAN:-0}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special,patch}"
# Token regions for redundancy similarity: special,camera,patch,all (comma-separated).
# camera = camera token only (subset of special); Pi3 has no camera token and skips this region.
# Used only for redundancy token subsampling (ignored when SKIP_REDUNDANCY=1).
REDUNDANCY_SEED="${REDUNDANCY_SEED:-0}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.90}"
# 1 = forward + save cache only (no similarity matrices / redundancy_graph_<region>.npz).

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local save_cache="$4"
  local split="${5:-}"
  local output_dataset="${6:-}"

  set_backend_pythonpath "${backbone}"
  local save_flag=()
  if truthy "${save_cache}"; then
    save_flag=(--save-cache)
  fi
  local hf_args=()
  if [[ -n "${HF_MODEL}" ]]; then
    hf_args=(--hf-model "${HF_MODEL}")
  fi
  local prune_args=()
  if truthy "${save_cache}" && [[ -n "${PRUNE_PLAN_ROOT}" ]]; then
    prune_args=(--prune-plan-root "${PRUNE_PLAN_ROOT}")
    if [[ -n "${PRUNE_PLAN_NAME}" ]]; then
      prune_args+=(--prune-plan-name "${PRUNE_PLAN_NAME}")
    fi
    if [[ -n "${TRAINING_PLAN_ROOT}" ]]; then
      prune_args+=(--training-plan-root "${TRAINING_PLAN_ROOT}")
    fi
    if [[ -n "${TRAINING_PLAN_NAME}" ]]; then
      prune_args+=(--training-plan-name "${TRAINING_PLAN_NAME}")
    fi
  fi
  local skip_red_args=()
  if truthy "${SKIP_REDUNDANCY}"; then
    skip_red_args=(--skip-redundancy)
  fi
  local split_args=()
  if [[ -n "${split}" ]]; then
    split_args=(--split "${split}")
  fi
  local output_dataset_args=()
  if [[ -n "${output_dataset}" ]]; then
    output_dataset_args=(--output-dataset "${output_dataset}")
  fi

  echo "== cache v3 dataset=${dataset}${split:+_${split}}${output_dataset:+ -> ${output_dataset}} backbone=${backbone} save_cache=${save_cache} skip_redundancy=${SKIP_REDUNDANCY} prune_plan_root=${PRUNE_PLAN_ROOT:-<off>} prune_plan_name=${PRUNE_PLAN_NAME:-<auto>} training_plan_name=${TRAINING_PLAN_NAME:-<auto>} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.redundancy.cache_redundancy_v3 \
    --experiment "${backbone}" \
    --dataset "${dataset}" \
    "${output_dataset_args[@]}" \
    --data-root "${data_root}" \
    --backbone "${backbone}" \
    "${hf_args[@]}" \
    --device "${DEVICE}" \
    --max-scenes "${MAX_SCENES}" \
    --max-frames "${MAX_FRAMES}" \
    --seed "${REDUNDANCY_SEED}" \
    --metrics "${METRICS}" \
    --max-span "${MAX_SPAN}" \
    --max-tokens "${MAX_TOKENS}" \
    --token-regions "${SIM_TOKEN_REGIONS}" \
    --similarity-threshold "${SIM_THRESHOLD}" \
    "${save_flag[@]}" \
    "${skip_red_args[@]}" \
    "${split_args[@]}" \
    "${prune_args[@]}"
}

run_scannet_split() {
  local data_root="$1"
  local backbone="$2"
  local save_cache="$3"
  local target_tag="$4"

  run_one_dataset "scannet" "${data_root}" "${backbone}" "${save_cache}" "" "${target_tag}"
}

echo "== cache+redundancy v3 backbones=${BACKBONES} (n=${BB_N}) MAX_FRAMES=${MAX_FRAMES} =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  if truthy "${RUN_SCANNET_DEV50}"; then
    run_scannet_split "${SCANNET_ROOT_DEV50}" "${bb}" "${CACHE_SCANNET_DEV50}" "scannet_dev50"
  fi
  if truthy "${RUN_7SCENES_TRAIN}"; then
    run_one_dataset "7scenes" "${ROOT_7SCENES}" "${bb}" "${CACHE_7SCENES_TRAIN}" "train"
  fi
  if truthy "${RUN_7SCENES_TEST}"; then
    run_one_dataset "7scenes" "${ROOT_7SCENES}" "${bb}" "${CACHE_7SCENES_TEST}" "test"
  fi
  if truthy "${RUN_NRGBD}"; then
    run_one_dataset "nrgbd" "${ROOT_NRGBD}" "${bb}" "${CACHE_NRGBD}"
  fi
  if truthy "${RUN_NUSCENES_VAL}"; then
    run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${bb}" "${CACHE_NUSCENES_VAL}" "val"
  fi
  if truthy "${RUN_NUSCENES_TRAIN}"; then
    run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${bb}" "${CACHE_NUSCENES_TRAIN}" "train"
  fi
  if truthy "${RUN_WAYMO_VALIDATION}"; then
    run_one_dataset "waymo" "${ROOT_WAYMO}" "${bb}" "${CACHE_WAYMO_VALIDATION}" "validation"
  fi
  if truthy "${RUN_WAYMO_TRAINING}"; then
    run_one_dataset "waymo" "${ROOT_WAYMO}" "${bb}" "${CACHE_WAYMO_TRAINING}" "training"
  fi
done

echo "== done =="
experiment_output_hint "Cache + redundancy"

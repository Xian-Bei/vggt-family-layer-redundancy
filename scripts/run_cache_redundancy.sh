#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN="${RUN:-}"
CACHE="${CACHE:-}"
SKIP_REDUNDANCY="${SKIP_REDUNDANCY:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"


MAX_SCENES="${MAX_SCENES:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_TOKENS="${MAX_TOKENS:-0}"
METRICS="${METRICS:-cka,cosine,angular,inv_mse}"
MAX_SPAN="${MAX_SPAN:-0}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special,patch,all}"
REDUNDANCY_SEED="${REDUNDANCY_SEED:-0}"

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
parse_backbones_list

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

run_progress_command() {
  if truthy "${COMPACT_LOG:-0}"; then
    "$@" >/dev/null
  else
    "$@"
  fi
}

parse_dataset_list "${RUN}" "RUN" "${DATASET_KEYS[@]}"
RUN_LIST=("${PARSED_DATASETS[@]}")
parse_dataset_list "${CACHE}" "CACHE" "${DATASET_KEYS[@]}"
CACHE_LIST=("${PARSED_DATASETS[@]}")

for cache_key in "${CACHE_LIST[@]}"; do
  if ! list_has "${cache_key}" "${RUN_LIST[@]}"; then
    echo "warning: CACHE includes ${cache_key}, but RUN does not; it will be ignored." >&2
  fi
done

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
  local prune_args=()
  if truthy "${save_cache}" && [[ -n "${PRUNE_PLAN_ROOT}" ]]; then
    prune_args=(--prune-plan-root "${PRUNE_PLAN_ROOT}")
    if [[ -n "${PRUNE_PLAN_NAME}" ]]; then
      prune_args+=(--prune-plan-name "${PRUNE_PLAN_NAME}")
    fi
    if [[ -n "${RECOVERY_PLAN_ROOT}" ]]; then
      prune_args+=(--recovery-plan-root "${RECOVERY_PLAN_ROOT}")
    fi
    if [[ -n "${RECOVERY_PLAN_NAME}" ]]; then
      prune_args+=(--recovery-plan-name "${RECOVERY_PLAN_NAME}")
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

  if truthy "${COMPACT_LOG:-0}"; then
    if truthy "${SKIP_REDUNDANCY}"; then
      local compact_spans="${PRUNE_PLAN_NAME//prune_span/}"
      local compact_methods="${RECOVERY_PLAN_NAME//recovery_/}"
      printf '\n  cache\n    model: %s\n    spans: %s\n    recovery methods: %s\n    dataset: %s\n\n' \
        "${backbone}" "${compact_spans:-all}" "${compact_methods:-all}" \
        "${output_dataset:-${dataset}${split:+_${split}}}"
    else
      printf '\n  redundancy\n    model: %s\n    metrics: %s\n    token regions: %s\n    dataset: %s\n\n' \
        "${backbone}" "${METRICS}" "${SIM_TOKEN_REGIONS}" \
        "${output_dataset:-${dataset}${split:+_${split}}}"
    fi
  else
    echo "== cache v3 dataset=${dataset}${split:+_${split}}${output_dataset:+ -> ${output_dataset}} backbone=${backbone} save_cache=${save_cache} skip_redundancy=${SKIP_REDUNDANCY} prune_plan_root=${PRUNE_PLAN_ROOT:-<off>} prune_plan_name=${PRUNE_PLAN_NAME:-<auto>} recovery_plan_name=${RECOVERY_PLAN_NAME:-<auto>} =="
  fi
  run_progress_command python -m analysis.redundancy.cache_redundancy_v3 \
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

run_dataset_key() {
  local key="$1"
  local backbone="$2"
  local save_cache=0
  if list_has "${key}" "${CACHE_LIST[@]}"; then
    save_cache=1
  fi

  case "${key}" in
    SCANNET_DEV50)
      run_scannet_split "${ROOT_SCANNET_DEV50}" "${backbone}" "${save_cache}" "scannet_dev50"
      ;;
    SCANNET_TEST50)
      run_scannet_split "${ROOT_SCANNET_TEST50}" "${backbone}" "${save_cache}" "scannet_test50"
      ;;
    7SCENES_TRAIN)
      run_one_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${save_cache}" "train"
      ;;
    7SCENES_TEST)
      run_one_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${save_cache}" "test"
      ;;
    NRGBD)
      run_one_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "${save_cache}"
      ;;
    NUSCENES_VAL)
      run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${save_cache}" "val"
      ;;
    NUSCENES_TRAIN)
      run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${save_cache}" "train"
      ;;
    WAYMO_VALIDATION)
      run_one_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${save_cache}" "validation"
      ;;
    WAYMO_TRAINING)
      run_one_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${save_cache}" "training"
      ;;
  esac
}

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== cache+redundancy v3 backbones=${BACKBONES} (n=${BB_N}) RUN=${RUN:-<none>} CACHE=${CACHE:-<none>} MAX_FRAMES=${MAX_FRAMES} =="
fi

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for dataset_key in "${RUN_LIST[@]}"; do
    run_dataset_key "${dataset_key}" "${bb}"
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Cache + redundancy"
fi

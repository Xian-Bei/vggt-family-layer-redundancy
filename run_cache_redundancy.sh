#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BACKBONES="${BACKBONES:-vggt_omega}"

RUN_SCANNET_DEV50="${RUN_SCANNET_DEV50:-1}"
RUN_7SCENES="${RUN_7SCENES:-0}"
RUN_NRGBD="${RUN_NRGBD:-0}"
SPLIT_7SCENES="${SPLIT_7SCENES:-train}"
CACHE_SCANNET_DEV50="${CACHE_SCANNET_DEV50:-1}"
SKIP_REDUNDANCY="${SKIP_REDUNDANCY:-0}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
PLAN_NAME="${PLAN_NAME:-}"


#############################################
MAX_SCENES="${MAX_SCENES:-0}"
CACHE_7SCENES="${CACHE_7SCENES:-0}"
CACHE_NRGBD="${CACHE_NRGBD:-0}"
CONDA_ENV="${CONDA_ENV:-vggt}"
SCANNET_ROOT_DEV50="${SCANNET_ROOT_DEV50:-data/scans_dev50}"
SCANNET_ROOT_TEST50="${SCANNET_ROOT_TEST50:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
HF_MODEL="${HF_MODEL:-}"


DEVICE="${DEVICE:-cuda}"
MAX_TOKENS="${MAX_TOKENS:-0}"
METRICS="${METRICS:-cka,cosine,neg_mse}"
MAX_SPAN="${MAX_SPAN:-0}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-all}"
# Used only for redundancy token subsampling (ignored when SKIP_REDUNDANCY=1).
REDUNDANCY_SEED="${REDUNDANCY_SEED:-0}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.90}"
# 1 = forward + save cache only (no similarity matrices / redundancy_graph_v3.npz).

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

validate_split_7scenes() {
  if [[ "${SPLIT_7SCENES}" != "train" && "${SPLIT_7SCENES}" != "test" ]]; then
    echo "error: SPLIT_7SCENES must be train or test, got: ${SPLIT_7SCENES}" >&2
    exit 1
  fi
}
validate_split_7scenes

# cache_redundancy_v3 writes under analysis_outputs_v3/<backbone>/scannet/; move to scannet_dev50|test50.
finalize_scannet_output() {
  local backbone="$1"
  local target_tag="$2"
  local src="${REPO_ROOT}/analysis_outputs_v3/${backbone}/scannet"
  local target="${REPO_ROOT}/analysis_outputs_v3/${backbone}/${target_tag}"

  if [[ ! -d "${src}" ]]; then
    echo "warning: no scannet output to finalize for ${backbone} (expected ${src})" >&2
    return 0
  fi

  if [[ ! -d "${target}" ]]; then
    mv "${src}" "${target}"
    echo "   [scannet] ${backbone}: moved ${src} -> ${target}"
    return 0
  fi

  local sub
  for sub in "${src}"/*; do
    [[ -e "${sub}" ]] || continue
    local name
    name="$(basename "${sub}")"
    if [[ -e "${target}/${name}" ]]; then
      rm -rf "${target:?}/${name}"
    fi
    mv "${sub}" "${target}/"
    echo "   [scannet] ${backbone}: merged ${name} -> ${target}/"
  done
  rmdir "${src}" 2>/dev/null || {
    echo "error: failed to remove staging dir after merge: ${src}" >&2
    return 1
  }
}

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local save_cache="$4"

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
    if [[ -n "${PLAN_NAME}" ]]; then
      prune_args+=(--plan-name "${PLAN_NAME}")
    fi
  fi
  local skip_red_args=()
  if truthy "${SKIP_REDUNDANCY}"; then
    skip_red_args=(--skip-redundancy)
  fi
  local split_args=()
  if [[ "${dataset}" == "7scenes" ]]; then
    split_args=(--split "${SPLIT_7SCENES}")
  fi

  echo "== cache v3 dataset=${dataset} backbone=${backbone} save_cache=${save_cache} skip_redundancy=${SKIP_REDUNDANCY} prune_plan_root=${PRUNE_PLAN_ROOT:-<off>} plan_name=${PLAN_NAME:-<auto>} split_7scenes=${SPLIT_7SCENES} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.redundancy.cache_redundancy_v3 \
    --experiment "${backbone}" \
    --dataset "${dataset}" \
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

  run_one_dataset "scannet" "${data_root}" "${backbone}" "${save_cache}"
  finalize_scannet_output "${backbone}" "${target_tag}"
}

echo "== cache+redundancy v3 backbones=${BACKBONES} (n=${BB_N}) MAX_FRAMES=${MAX_FRAMES} =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  if truthy "${RUN_SCANNET_DEV50}"; then
    run_scannet_split "${SCANNET_ROOT_DEV50}" "${bb}" "${CACHE_SCANNET_DEV50}" "scannet_dev50"
  fi
  if truthy "${RUN_7SCENES}"; then
    run_one_dataset "7scenes" "${ROOT_7SCENES}" "${bb}" "${CACHE_7SCENES}"
  fi
  if truthy "${RUN_NRGBD}"; then
    run_one_dataset "nrgbd" "${ROOT_NRGBD}" "${bb}" "${CACHE_NRGBD}"
  fi
done

echo "== done =="
experiment_output_hint "Cache + redundancy"

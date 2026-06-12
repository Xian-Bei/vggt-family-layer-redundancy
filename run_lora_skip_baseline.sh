#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# v3 outputs are split by dataset tag under analysis_outputs_v3/<backbone>/<dataset_tag>/...
# Default to the dev50 tag (historically called "scannet" but renamed to avoid collisions).
BACKBONES="${BACKBONES:-vggt,fastvggt,vggt-omega}"
DATASET="${DATASET:-scannet_dev50}"
EPOCHS="${EPOCHS:-5}"
MAX_SCENES="${MAX_SCENES:-1}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PLAN_NAME="${PLAN_NAME:-}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-1}"
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${REPO_ROOT}/analysis_outputs_v3}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"
EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-50}"
EVAL_SCANNET_DEV50="${EVAL_SCANNET_DEV50:-0}"
EVAL_SCANNET_TEST50="${EVAL_SCANNET_TEST50:-0}"
EVAL_7SCENES="${EVAL_7SCENES:-1}"
EVAL_NRGBD="${EVAL_NRGBD:-0}"
SPLIT_7SCENES="${SPLIT_7SCENES:-test}"

#############################################
HF_MODEL="${HF_MODEL:-}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
LR="${LR:-5e-5}"
DTYPE="${DTYPE:-bfloat16}"

source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

validate_split_7scenes() {
  if [[ "${SPLIT_7SCENES}" != "train" && "${SPLIT_7SCENES}" != "test" ]]; then
    echo "error: SPLIT_7SCENES must be train or test, got: ${SPLIT_7SCENES}" >&2
    exit 1
  fi
}
validate_split_7scenes

run_one_backbone() {
  local backbone="$1"
  set_backend_pythonpath "${backbone}"
  export BACKEND="${backbone}"

  if [[ -z "${PLAN_NAME}" ]]; then
    echo "error: PLAN_NAME is required (e.g. PLAN_NAME=plan_t2_l1)." >&2
    exit 1
  fi

  local plan_root="${PRUNE_PLAN_ROOT}"
  if [[ -z "${plan_root}" ]]; then
    plan_root="${REPO_ROOT}/analysis_outputs_v3/${backbone}"
  fi

  # Guardrail: `scannet` legacy tag no longer exists after the dev/test split.
  if [[ "${DATASET}" == "scannet" ]]; then
    echo "error: DATASET=scannet is legacy; use DATASET=scannet_dev50 (train/cache) and eval splits via EVAL_SCANNET_* switches." >&2
    exit 1
  fi

  local extra=(
    --prune-plan-root "${plan_root}"
    --plan-name "${PLAN_NAME}"
    --eval-bounds-root "${EVAL_BOUNDS_ROOT}"
    --output-dir "${REPO_ROOT}/analysis_outputs_v3/${backbone}/${DATASET}/train/${PLAN_NAME}"
  )
  if [[ "${EVAL_SCANNET_DEV50}" == "1" ]]; then
    extra+=(--eval-scannet-dev50)
  else
    extra+=(--no-eval-scannet-dev50)
  fi
  if [[ "${EVAL_SCANNET_TEST50}" == "1" ]]; then
    extra+=(--eval-scannet-test50)
  else
    extra+=(--no-eval-scannet-test50)
  fi
  if [[ "${EVAL_7SCENES}" == "1" ]]; then
    extra+=(--eval-7scenes)
  else
    extra+=(--no-eval-7scenes)
  fi
  extra+=(--eval-7scenes-split "${SPLIT_7SCENES}")
  if [[ "${EVAL_NRGBD}" == "1" ]]; then
    extra+=(--eval-nrgbd)
  else
    extra+=(--no-eval-nrgbd)
  fi
  if [[ -n "${HF_MODEL}" ]]; then
    extra+=(--hf-model "${HF_MODEL}")
  fi
  if [[ "${EVAL_AT_START}" == "1" || "${EVAL_AT_START}" == "true" || "${EVAL_AT_START}" == "yes" ]]; then
    extra+=(--eval-at-start)
  else
    extra+=(--no-eval-at-start)
  fi

  echo "== train v3 dataset=${DATASET} backbone=${backbone} plan=${PLAN_NAME} prune_plan_root=${plan_root} eval_at_start=${EVAL_AT_START} save_every_epoch=${SAVE_EVERY_EPOCH} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.train.train_prune_lora_v3 \
    --experiment "${backbone}" \
    --dataset "${DATASET}" \
    --backbone "${backbone}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --max-scenes "${MAX_SCENES}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --dtype "${DTYPE}" \
    --eval-epoch "${EVAL_EPOCH}" \
    --save-every-epoch "${SAVE_EVERY_EPOCH}" \
    --eval-max-scenes "${EVAL_MAX_SCENES}" \
    --max-eval-views "${EVAL_MAX_FRAMES}" \
    "${extra[@]}"
}

echo "== train v3 backbones=${BACKBONES} (n=${BB_N}) =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  run_one_backbone "${bb}"
done

echo "== done =="
experiment_output_hint "Train"

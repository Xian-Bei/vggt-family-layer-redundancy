#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
DATASETS="${DATASETS:-${DATASET:-scannet_dev50}}"
MAX_SCENES="${MAX_SCENES:-1}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-5}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-0}"

source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${EVAL}" "EVAL" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

PRUNE_PLAN_NAME_LIST=()
read -r -a _PRUNE_PLAN_NAME_RAW <<< "${PRUNE_PLAN_NAME//,/ }"
for _prune_plan_name in "${_PRUNE_PLAN_NAME_RAW[@]}"; do
  _prune_plan_name="${_prune_plan_name//[[:space:]]/}"
  [[ -n "${_prune_plan_name}" ]] && PRUNE_PLAN_NAME_LIST+=("${_prune_plan_name}")
done
if [[ "${#PRUNE_PLAN_NAME_LIST[@]}" -eq 0 ]]; then
  echo "error: PRUNE_PLAN_NAME is required (e.g. PRUNE_PLAN_NAME='prune_span5 prune_span10')." >&2
  exit 1
fi

RECOVERY_PLAN_NAME_LIST=()
read -r -a _RECOVERY_PLAN_NAME_RAW <<< "${RECOVERY_PLAN_NAME//,/ }"
for _recovery_plan_name in "${_RECOVERY_PLAN_NAME_RAW[@]}"; do
  _recovery_plan_name="${_recovery_plan_name//[[:space:]]/}"
  if [[ -n "${_recovery_plan_name}" ]]; then
    RECOVERY_PLAN_NAME_LIST+=("${_recovery_plan_name}")
  fi
done
if [[ "${#RECOVERY_PLAN_NAME_LIST[@]}" -eq 0 ]]; then
  echo "error: RECOVERY_PLAN_NAME is required (e.g. RECOVERY_PLAN_NAME='recovery_lora_l1_r32 recovery_ghost')." >&2
  exit 1
fi

resolve_recovery_method() {
  local plan_root="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  conda run --no-capture-output -n "${CONDA_ENV}" python -c "
from pathlib import Path
from analysis.prune_plan.recovery_plan import load_training_plan_bundle
root = Path('${plan_root}')
name = '${recovery_plan_name}'
print(str(load_training_plan_bundle(root, name).get('method', '')).strip())
"
}

run_trainable_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  BACKBONES="${backbone}" \
    DATASETS="${DATASETS}" \
    PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}}" \
    RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-${PRUNE_PLAN_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}}}" \
    RECOVERY_PLAN_NAME="${recovery_plan_name}" \
    PRUNE_PLAN_NAME="${prune_plan_name}" \
    OUTPUT_KIND=recover \
    EPOCHS="${EPOCHS}" \
    MAX_SCENES="${MAX_SCENES}" \
    MAX_STEPS="${MAX_STEPS}" \
    EVAL_EPOCH="${EVAL_EPOCH}" \
    SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
    EVAL_AT_START="${EVAL_AT_START}" \
    EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT}" \
    EVAL_MAX_SCENES="${EVAL_MAX_SCENES}" \
    EVAL="${EVAL}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    bash "run_lora_skip_baseline.sh"
}

run_closed_form_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  local plan_root="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}}"
  local recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  local out_dir="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/recover/${prune_plan_name}/${recovery_plan_name}"
  local hf_args=()
  echo "== recover v3 datasets=${DATASETS} backbone=${backbone} prune_plan_name=${prune_plan_name} recovery_plan_name=${recovery_plan_name} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.recover.recover_prune_v3 \
    --experiment "${backbone}" \
    --dataset "${DATASETS}" \
    --backbone "${backbone}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-scenes "${MAX_SCENES}" \
    --prune-plan-root "${plan_root}" \
    --prune-plan-name "${prune_plan_name}" \
    --recovery-plan-root "${recovery_root}" \
    --recovery-plan-name "${recovery_plan_name}" \
    --output-dir "${out_dir}" \
    "${hf_args[@]}"
  if [[ "${#EVAL_LIST[@]}" -gt 0 ]]; then
    BACKBONES="${backbone}" \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${prune_plan_name}" \
      RECOVERY_PLAN_ROOT="${recovery_root}" \
      RECOVERY_PLAN_NAME="${recovery_plan_name}" \
      CHECKPOINT="${out_dir}/checkpoints/latest.pt" \
      MAX_SCENES="${EVAL_MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      RUN="${EVAL}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      WRITE_SUMMARY=0 \
      bash "run_eval.sh"
  fi
}

echo "== recover v3 backbones=${BACKBONES} (n=${BB_N}) =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  set_backend_pythonpath "${bb}"
  plan_root="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}/${bb}}"
  recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  for prune_plan_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
    for recovery_plan_name in "${RECOVERY_PLAN_NAME_LIST[@]}"; do
      method="$(resolve_recovery_method "${recovery_root}" "${prune_plan_name}" "${recovery_plan_name}")"
      case "${method}" in
        recovery_lora|recovery_train)
          run_trainable_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          ;;
        recovery_diag|recovery_rotate|recovery_ghost)
          run_closed_form_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          ;;
        *)
          echo "error: unsupported recovery method=${method} for ${recovery_plan_name}" >&2
          exit 1
          ;;
      esac
    done
  done
  echo "   [summary] ${OUTPUT_ROOT}/${bb}/summary_comparison.md"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_v3 \
    --experiment "${bb}" \
    --prune-plan-root "${plan_root}"
done

echo "== done =="
experiment_output_hint "Recover"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
DATASETS="${DATASETS:-${DATASET:-scannet_dev50}}"
EVAL="${EVAL:-}"
MAX_SCENES="${MAX_SCENES:-1}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
if [[ "${OUTPUT_ROOT}" = /* ]]; then
  OUTPUT_BASE="${OUTPUT_ROOT}"
else
  OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${OUTPUT_BASE}}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-5}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-0}"
REPLACEME_LS_ALPHA="${REPLACEME_LS_ALPHA:-0}"
REPLACEME_COS_EPOCHS="${REPLACEME_COS_EPOCHS:-10}"
REPLACEME_COS_MORE_EPOCHS="${REPLACEME_COS_MORE_EPOCHS:-30}"
REPLACEME_COS_BATCH_SIZE="${REPLACEME_COS_BATCH_SIZE:-1024}"
REPLACEME_COS_LR="${REPLACEME_COS_LR:-1e-4}"
REPLACEME_SEED="${REPLACEME_SEED:-0}"
PRUNECOMP_EPS="${PRUNECOMP_EPS:-1e-8}"
GHOST_COS_EPOCHS="${GHOST_COS_EPOCHS:-10}"
GHOST_COS_MORE_EPOCHS="${GHOST_COS_MORE_EPOCHS:-30}"
GHOST_COS_BATCH_SIZE="${GHOST_COS_BATCH_SIZE:-1024}"
GHOST_COS_LR="${GHOST_COS_LR:-1e-4}"
GHOST_COS_SEED="${GHOST_COS_SEED:-0}"
GHOST_MORE_COS_WEIGHT="${GHOST_MORE_COS_WEIGHT:-0.04}"
GHOST_MORE_CKA_WEIGHT="${GHOST_MORE_CKA_WEIGHT:-0.10}"
GHOST_MORE_EMA_BETA="${GHOST_MORE_EMA_BETA:-0.95}"
GHOST_MORE_EMA_COS_MIN_WEIGHT="${GHOST_MORE_EMA_COS_MIN_WEIGHT:-0.005}"
GHOST_MORE_EMA_COS_MAX_WEIGHT="${GHOST_MORE_EMA_COS_MAX_WEIGHT:-0.20}"
GHOST_MORE_EMA_CKA_MIN_WEIGHT="${GHOST_MORE_EMA_CKA_MIN_WEIGHT:-0.01}"
GHOST_MORE_EMA_CKA_MAX_WEIGHT="${GHOST_MORE_EMA_CKA_MAX_WEIGHT:-1.0}"
GHOST_MSE_CKA_EPOCHS="${GHOST_MSE_CKA_EPOCHS:-30}"
GHOST_MSE_CKA_LR="${GHOST_MSE_CKA_LR:-1e-5}"
GHOST_MSE_CKA_WEIGHT="${GHOST_MSE_CKA_WEIGHT:-1.0}"
GHOST_MSE_CKA_BUDGET="${GHOST_MSE_CKA_BUDGET:-0.01}"
GHOST_MSE_CKA_MAX_DELTA_RATIO="${GHOST_MSE_CKA_MAX_DELTA_RATIO:-0}"
GHOST_MSE_CKA_GRADIENT_MODE="${GHOST_MSE_CKA_GRADIENT_MODE:-scene}"
GHOST_MSE_CKA_SEED="${GHOST_MSE_CKA_SEED:-0}"
GHOST_CKA_ONLY_EPOCHS="${GHOST_CKA_ONLY_EPOCHS:-60}"
GHOST_CKA_ONLY_LR="${GHOST_CKA_ONLY_LR:-1e-5}"
GHOST_CKA_ONLY_SNAPSHOT_EPOCHS="${GHOST_CKA_ONLY_SNAPSHOT_EPOCHS:-5,20,60}"
GHOST_CKA_ONLY_SEED="${GHOST_CKA_ONLY_SEED:-0}"
GHOST_CKA_RIDGE_GRID="${GHOST_CKA_RIDGE_GRID:-0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2}"

source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${EVAL}" "EVAL" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

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
  local prune_root="$1"
  local recovery_root="$2"
  local prune_plan_name="$3"
  local recovery_plan_name="$4"
  conda run --no-capture-output -n "${CONDA_ENV}" python -c "
from pathlib import Path
from analysis.prune_plan.recovery_plan import load_recovery_plan_bundle, recovery_plan_supports_selector
prune_root = Path('${prune_root}')
recovery_root = Path('${recovery_root}')
name = '${recovery_plan_name}'
selector = '${prune_plan_name}'
bundle = load_recovery_plan_bundle(recovery_root, name)
supported = recovery_plan_supports_selector(
    root=recovery_root, name=name, prune_plan_root=prune_root, prune_selector=selector
)
print(str(bundle.get('method', '')).strip() if supported else 'unsupported')
"
}

run_trainable_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  BACKBONES="${backbone}" \
    DATASETS="${DATASETS}" \
    PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}" \
    RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}}" \
    RECOVERY_PLAN_NAME="${recovery_plan_name}" \
    PRUNE_PLAN_NAME="${prune_plan_name}" \
    OUTPUT_KIND=recovery \
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

run_training_free_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  local plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}"
  local recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  local out_dir="${OUTPUT_BASE}/${backbone}/recovery/${prune_plan_name}/${recovery_plan_name}"
  local hf_args=()
  if truthy "${COMPACT_LOG:-0}"; then
    printf '\n  recovery\n    model: %s\n    span: %s\n    dataset: %s\n    method: %s\n\n' \
      "${backbone}" "${prune_plan_name#prune_span}" "${DATASETS}" \
      "${recovery_plan_name#recovery_}"
  else
    echo "== recovery v3 datasets=${DATASETS} backbone=${backbone} prune_plan_name=${prune_plan_name} recovery_plan_name=${recovery_plan_name} =="
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT}" run_progress_command conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.recovery.recovery_prune_v3 \
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
    --ghost-cos-epochs "${GHOST_COS_EPOCHS}" \
    --ghost-cos-more-epochs "${GHOST_COS_MORE_EPOCHS}" \
    --ghost-cos-batch-size "${GHOST_COS_BATCH_SIZE}" \
    --ghost-cos-lr "${GHOST_COS_LR}" \
    --ghost-cos-seed "${GHOST_COS_SEED}" \
    --ghost-more-cos-weight "${GHOST_MORE_COS_WEIGHT}" \
    --ghost-more-cka-weight "${GHOST_MORE_CKA_WEIGHT}" \
    --ghost-more-ema-beta "${GHOST_MORE_EMA_BETA}" \
    --ghost-more-ema-cos-min-weight "${GHOST_MORE_EMA_COS_MIN_WEIGHT}" \
    --ghost-more-ema-cos-max-weight "${GHOST_MORE_EMA_COS_MAX_WEIGHT}" \
    --ghost-more-ema-cka-min-weight "${GHOST_MORE_EMA_CKA_MIN_WEIGHT}" \
    --ghost-more-ema-cka-max-weight "${GHOST_MORE_EMA_CKA_MAX_WEIGHT}" \
    --ghost-mse-cka-epochs "${GHOST_MSE_CKA_EPOCHS}" \
    --ghost-mse-cka-lr "${GHOST_MSE_CKA_LR}" \
    --ghost-mse-cka-weight "${GHOST_MSE_CKA_WEIGHT}" \
    --ghost-mse-cka-budget "${GHOST_MSE_CKA_BUDGET}" \
    --ghost-mse-cka-max-delta-ratio "${GHOST_MSE_CKA_MAX_DELTA_RATIO}" \
    --ghost-mse-cka-gradient-mode "${GHOST_MSE_CKA_GRADIENT_MODE}" \
    --ghost-mse-cka-seed "${GHOST_MSE_CKA_SEED}" \
    --ghost-cka-only-epochs "${GHOST_CKA_ONLY_EPOCHS}" \
    --ghost-cka-only-lr "${GHOST_CKA_ONLY_LR}" \
    --ghost-cka-only-snapshot-epochs "${GHOST_CKA_ONLY_SNAPSHOT_EPOCHS}" \
    --ghost-cka-only-seed "${GHOST_CKA_ONLY_SEED}" \
    --ghost-cka-ridge-grid "${GHOST_CKA_RIDGE_GRID}" \
    --replaceme-ls-alpha "${REPLACEME_LS_ALPHA}" \
    --replaceme-cos-epochs "${REPLACEME_COS_EPOCHS}" \
    --replaceme-cos-more-epochs "${REPLACEME_COS_MORE_EPOCHS}" \
    --replaceme-cos-batch-size "${REPLACEME_COS_BATCH_SIZE}" \
    --replaceme-cos-lr "${REPLACEME_COS_LR}" \
    --replaceme-seed "${REPLACEME_SEED}" \
    --prunecomp-eps "${PRUNECOMP_EPS}" \
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

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== recovery v3 backbones=${BACKBONES} (n=${BB_N}) =="
fi

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  set_backend_pythonpath "${bb}"
  plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${bb}}"
  recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  for prune_plan_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
    for recovery_plan_name in "${RECOVERY_PLAN_NAME_LIST[@]}"; do
      method="$(resolve_recovery_method "${plan_root}" "${recovery_root}" "${prune_plan_name}" "${recovery_plan_name}")"
      case "${method}" in
        unsupported)
          echo "warning: skipping unsupported ${prune_plan_name}+${recovery_plan_name}" >&2
          continue
          ;;
        recovery_lora|recovery_train)
          run_trainable_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          ;;
        recovery_linearpatch|recovery_ghost|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_ghost_cos|recovery_ghost_cos_more|recovery_ghost_cos_scale|recovery_ghost_mse_more|recovery_ghost_mse_cos_more|recovery_ghost_mse_cka_more|recovery_ghost_mse_cos_cka_more|recovery_ghost_mse_cos_balanced_more|recovery_ghost_mse_cka_balanced_more|recovery_ghost_mse_cos_cka_balanced_more|recovery_ghost_mse_cka_ema_more|recovery_ghost_mse_cos_cka_ema_half_more|recovery_ghost_mse_cos_cka_ema_one_more|recovery_ghost_mse_cka|recovery_ghost_cka_only|recovery_ghost_cka_ridge|recovery_replaceme_ls|recovery_replaceme_cos|recovery_replaceme_cos_more|recovery_replaceme_cos_scale|recovery_prunecomp)
          run_training_free_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          ;;
        *)
          echo "error: unsupported recovery method=${method} for ${recovery_plan_name}" >&2
          exit 1
          ;;
      esac
    done
  done
  if ! truthy "${COMPACT_LOG:-0}"; then
    echo "   [summary] ${OUTPUT_ROOT}/${bb}/summary_comparison.md"
  fi
  run_progress_command conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_v3 \
    --experiment "${bb}" \
    --prune-plan-root "${plan_root}"
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Recover"
fi

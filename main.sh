#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
: "${OUTPUT_ROOT:=main_results}"
: "${REFERENCE_RESULTS_ROOT:=analysis_outputs_v3}"
export OUTPUT_ROOT REFERENCE_RESULTS_ROOT
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/scripts/pipeline_lib.sh"

TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS="${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS:-0}"
TESTTIME_SKIP_FULL_MODEL_ACCURACY="${TESTTIME_SKIP_FULL_MODEL_ACCURACY:-1}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-special_patch_mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-retention_single,retention_double}"
RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY:-retention_single,retention_double}"
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_ghost_interval_sequential_autogroup}"
export COMPACT_LOG="${COMPACT_LOG:-1}"

# --- Stage init: Initialize isolated main results ---
stage_log "init" "Initialize isolated main results"
BACKBONES="${ALL_BACKBONES}" \
    CALIB_DATASETS="${CALIB_DATASETS}" \
    EVAL_DATASETS="${EVAL_DATASETS}" \
    PLAN_MAX_DROP="${PLAN_MAX_DROP}" \
    REFERENCE_RESULTS_ROOT="${REFERENCE_RESULTS_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash "${REPO_ROOT}/scripts/init_main_results.sh"

# --- Stage init.norm: Fix normalization ranges from imported Stage-2 intervals ---
stage_log "init.norm" "Fix normalization ranges from imported Stage-2 intervals"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_command python -m analysis.prune_plan.build_normalization_profiles \
    --plan-root "${OUTPUT_ROOT}/${bb}" \
    --datasets "${CALIB_DATASETS}" \
    --max-drop "${NORMALIZATION_MAX_DROP}"
done

# --- Stage 6/10: Build pruning plans ---
stage_log "6/10" "Build pruning plans"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  if compact_log; then
    printf '\n  pruning plan\n    model: %s\n    strategies: %s\n    metric: cka\n    token region: %s\n    datasets: %s\n\n' \
      "${bb}" "${PRUNE_STRATEGY}" "${PRUNE_TOKEN_REGION}" "${CALIB_DATASETS}"
  fi
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}/prune_plan_all.json"

  run_command python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${CALIB_DATASETS}" \
    --backbone "${bb}" \
    --metric cka \
    --strategies "${PRUNE_STRATEGY}" \
    --max-drop-budget "${PLAN_MAX_DROP}" \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done

PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
# --- Stage 7/10: Evaluate pruning plans ---
stage_log "7/10" "Evaluate pruning plans"

for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_pruning_backbone "${bb}"
done

# --- Stage 8/10: Build recovery plans ---
build_pipeline_recovery_plans

EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
export RECOVERY_CHUNK_ROWS="${RECOVERY_CHUNK_ROWS:-16384}"
export GHOST_AUTOGROUP_THRESHOLD="${GHOST_AUTOGROUP_THRESHOLD:-20}"
# --- Stage 9/10: Cache, recover, and evaluate ---
stage_log "9/10" "Cache, recover, and evaluate"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  recovery_plans="$(
    python -m analysis.recovery.shared_results recovery-plan-names \
      --plan-root "${cache_root}"
  )"
  recovery_prune_plans="$(
    python -m analysis.recovery.shared_results selectors-for-strategies \
      --plan-root "${cache_root}" \
      --strategies "${RECOVERY_PRUNE_STRATEGY}" \
      --selectors "${RECOVERY_PRUNE_PLANS}"
  )"
  for ds in ${CALIB_DATASETS//,/ }; do
    rm -rf "${cache_root}/${ds}/cache"
  done
  BACKBONES="${bb}" \
      RUN="${CALIB_DATASETS}" \
      CACHE="${CALIB_DATASETS}" \
      SKIP_REDUNDANCY=1 \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      run_pipeline_script run_cache_redundancy.sh

  BACKBONES="${bb}" \
      DATASETS="${CALIB_DATASETS}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      EPOCHS="${EPOCHS}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN:-0}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL="${CALIB_DATASETS},${EVAL_DATASETS}" \
      RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY}" \
      run_pipeline_script run_recovery.sh

  for ds in ${CALIB_DATASETS//,/ }; do
    rm -rf "${cache_root}/${ds}/cache"
  done
done

# --- Stage 10/10: Evaluate pruning + recovery + test-time compression ---
stage_log "10/10" "Evaluate pruning + recovery + test-time compression"
testtime_recovery_plan="${TESTTIME_RECOVERY_PLAN_NAME:-recovery_ghost_interval_sequential_autogroup}"
testtime_prune_strategies="retention"
EXPORT_ORIGINAL_VISUALIZATION="${EXPORT_ORIGINAL_VISUALIZATION:-1}"
if [[ "${EXPORT_ORIGINAL_VISUALIZATION}" != "0" && "${EXPORT_ORIGINAL_VISUALIZATION}" != "1" ]]; then
  echo "error: EXPORT_ORIGINAL_VISUALIZATION must be 0 or 1" >&2
  exit 1
fi
if [[ "${EXPORT_ORIGINAL_VISUALIZATION}" == "1" ]]; then
  BACKBONES="${ALL_BACKBONES}" \
      RUN="${EVAL_DATASETS}" \
      EXPORT_VIS_2D=1 \
      EXPORT_VIS_3D=1 \
      PRUNE=0 \
      CHECKPOINT="" \
      RECOVERY_PLAN_NAME="" \
      MODEL_NAME=upper \
      INFERENCE_COMPRESSION=none \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      run_pipeline_script run_eval.sh
fi
if [[ "${TESTTIME_SKIP_FULL_MODEL_ACCURACY}" != "0" && "${TESTTIME_SKIP_FULL_MODEL_ACCURACY}" != "1" ]]; then
  echo "error: TESTTIME_SKIP_FULL_MODEL_ACCURACY must be 0 or 1" >&2
  exit 1
fi
if [[ "${TESTTIME_SKIP_FULL_MODEL_ACCURACY}" == "1" ]]; then
  echo "[testtime] skip migrated full-model FastVGGT/QuantVGGT accuracy evaluation"
else
  full_fastvggt_backbones=()
  for bb in "${ALL_BACKBONES_LIST[@]}"; do
    case "${bb}" in
      vggt|pi3|vggt_omega) full_fastvggt_backbones+=("${bb}") ;;
    esac
  done
  if [[ "${#full_fastvggt_backbones[@]}" -gt 0 ]]; then
    BACKBONES="$(join_by_comma "${full_fastvggt_backbones[@]}")" \
        RUN="${EVAL_DATASETS}" \
        EXPORT_VIS_2D=1 \
        EXPORT_VIS_3D=1 \
        PRUNE=0 \
        CHECKPOINT="" \
        RECOVERY_PLAN_NAME="" \
        MODEL_NAME=upper \
        INFERENCE_COMPRESSION=fastvggt \
        FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START}" \
        FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        run_pipeline_script run_eval.sh
  fi

  if list_has "vggt" "${ALL_BACKBONES_LIST[@]}"; then
    full_quantvggt_params_dir="${QUANTVGGT_PARAMS_DIR}"
    if [[ -f "${QUANTVGGT_PARAMS_DIR}/vggt/qs_frame_parameters_total.pth" ]]; then
      full_quantvggt_params_dir="${QUANTVGGT_PARAMS_DIR}/vggt"
    fi
    BACKBONES=vggt \
        RUN="${EVAL_DATASETS}" \
        EXPORT_VIS_2D=1 \
        EXPORT_VIS_3D=1 \
        PRUNE=0 \
        CHECKPOINT="" \
        RECOVERY_PLAN_NAME="" \
        MODEL_NAME=upper \
        INFERENCE_COMPRESSION=quantvggt \
        QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE}" \
        QUANTVGGT_PARAMS_DIR="${full_quantvggt_params_dir}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        run_pipeline_script run_eval.sh
  fi
fi

for bb in "${ALL_BACKBONES_LIST[@]}"; do
  case "${bb}" in
    pi3)
      testtime_prune_plans="prune_span16"
      ;;
    vggt_omega)
      testtime_prune_plans="prune_span18"
      ;;
    vggt)
      testtime_prune_plans="prune_span10"
      ;;
    *)
      echo "error: no test-time pruning configuration for backbone=${bb@Q}" >&2
      exit 1
      ;;
  esac

  OUTPUT_ROOT="${OUTPUT_ROOT}" \
      TESTTIME_BACKBONES="${bb}" \
      TESTTIME_PRUNE_PLAN_NAME="${testtime_prune_plans}" \
      TESTTIME_PRUNE_STRATEGY="${testtime_prune_strategies}" \
      TESTTIME_RECOVERY_PLAN_NAME="${testtime_recovery_plan}" \
      TESTTIME_SELECTION_DATASETS="${CALIB_DATASETS}" \
      EXPORT_VIS_2D=1 \
      EXPORT_VIS_3D=1 \
      TESTTIME_COMPRESSIONS=none,fastvggt,quantvggt \
      TESTTIME_RUN="${EVAL_DATASETS}" \
      run_pipeline_script run_testtime_eval.sh

  OUTPUT_ROOT="${OUTPUT_ROOT}" \
      TESTTIME_BACKBONES="${bb}" \
      TESTTIME_PRUNE_PLAN_NAME="${testtime_prune_plans}" \
      TESTTIME_PRUNE_STRATEGY="${testtime_prune_strategies}" \
      TESTTIME_RECOVERY_PLAN_NAME="${testtime_recovery_plan}" \
      TESTTIME_SELECTION_DATASETS="${CALIB_DATASETS}" \
      TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS="${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS}" \
      TESTTIME_RUN="${EVAL_DATASETS}" \
      run_pipeline_script run_testtime_benchmark.sh
done

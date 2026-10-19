#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/scripts/pipeline_lib.sh"
install_error_diagnostics "${BASH_SOURCE[0]}"

ANALYSIS_DATASETS="${ANALYSIS_DATASETS:-scannet_dev50,nuscenes_train}"
PRUNE_TOKEN_REGIONS="${PRUNE_TOKEN_REGIONS:-all,special_patch_mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-retention_single,retention_double,CKA_single,CKA_double,cosine_single,cosine_double,inv_mse_single,inv_mse_double,cca_single,cca_double,replaceme_single,replaceme_double,shortgpt,gardener,simpler}"
RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY:-retention_single,retention_double,CKA_single,CKA_double}"
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_linearpatch,recovery_ghost,recovery_ghost_interval_sequential,recovery_ghost_interval_sequential_autogroup,recovery_replaceme_ls,recovery_replaceme_cos,recovery_streamline_ffn}"
VGGT_POINT="${VGGT_POINT-12,16}"
PI3_POINT="${PI3_POINT-16,20}"
VGGT_OMEGA_POINT="${VGGT_OMEGA_POINT-16,20}"
for point_var in VGGT_POINT PI3_POINT VGGT_OMEGA_POINT; do
  raw_points="${!point_var//,/ }"
  if [[ -z "${raw_points//[[:space:]]/}" ]]; then
    continue
  fi
  read -r -a point_values <<< "${raw_points}"
  for point in "${point_values[@]}"; do
    if [[ ! "${point}" =~ ^[1-9][0-9]*$ ]]; then
      echo "error: ${point_var} must be a comma- or space-separated list of positive integers; got '${!point_var}'" >&2
      exit 1
    fi
  done
done

parse_prune_token_regions
parse_dual_token_strategies
export OUTPUT_ROOT
export COMPACT_LOG="${COMPACT_LOG:-1}"

# --- Stage 1/10: Evaluate pretrained models ---
stage_log "1/10" "Evaluate pretrained models"
BACKBONES="${ALL_BACKBONES}" \
    RUN="${CALIB_DATASETS},${EVAL_DATASETS}" \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=1 \
    PRUNE=0 \
    CHECKPOINT="" \
    RECOVERY_PLAN_NAME="" \
    INFERENCE_COMPRESSION=none \
    run_pipeline_script run_eval.sh

# --- Stage 2/10: Evaluate skip intervals ---
stage_log "2/10" "Evaluate skip intervals"
BACKBONES="${ALL_BACKBONES}" \
    EVAL_DATASET="${CALIB_DATASETS}" \
    bash "${REPO_ROOT}/scripts/run_skip_eval.sh"
# --- Stage 2.1/10: Fix normalization ranges from brute-force intervals ---
stage_log "2.1/10" "Fix normalization ranges from brute-force intervals"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_command python -m analysis.prune_plan.build_normalization_profiles \
    --plan-root "${OUTPUT_ROOT}/${bb}" \
    --datasets "${CALIB_DATASETS}" \
    --max-drop "${NORMALIZATION_MAX_DROP}"
done

# --- Stage 3/10: Compute redundancy ---
stage_log "3/10" "Compute redundancy"
BACKBONES="${ALL_BACKBONES}" \
    RUN="${ANALYSIS_DATASETS}" \
    CACHE="" \
    METRICS=cka,cosine,angular,inv_mse,cca \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    run_pipeline_script run_cache_redundancy.sh

# --- Stage 4/10: Evaluate skip intervals after pruning ---
stage_log "4/10" "Evaluate skip intervals after pruning"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  case "${bb}" in
    vggt) configured_points="${VGGT_POINT}" ;;
    pi3) configured_points="${PI3_POINT}" ;;
    vggt_omega) configured_points="${VGGT_OMEGA_POINT}" ;;
    *)
      echo "error: no stage-4 point configured for backbone '${bb}'" >&2
      exit 1
      ;;
  esac
  configured_points="${configured_points//,/ }"
  if [[ -z "${configured_points//[[:space:]]/}" ]]; then
    continue
  fi
  read -r -a point_values <<< "${configured_points}"
  for point in "${point_values[@]}"; do
    read -r base_selector drop_count < <(
      python -m analysis.prune_plan.retention_single_interval_at_point \
          --root "${OUTPUT_ROOT}/${bb}" \
          --datasets "${CALIB_DATASETS}" \
          --point "${point}"
    )
    BACKBONES="${bb}" \
        RUN="${CALIB_DATASETS}" \
        PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
        PRUNE_PLAN_NAME="${base_selector}" \
        EXPECTED_DROP_COUNT="${drop_count}" \
        run_pipeline_script run_remaining_skip_eval.sh
  done
done

# --- Stage 6/10: Build token-region pruning candidates ---
stage_log "6/10" "Build token-region pruning candidates"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  candidate_root="${plan_root}/.prune_token_region_candidates"
  mkdir -p "${plan_root}"
  find "${plan_root}" -maxdepth 1 -type f -name 'prune*.json' \
    ! -name 'prune_plan_all_token_*.json' -delete
  find "${plan_root}" -maxdepth 1 -type f -name 'prune*.png' \
    ! -name 'prune_plan_*_drop_grid_token_*.png' -delete
  find "${plan_root}" -type f \
    \( -name '*_gt_components.png' -o -name '*_paths.png' -o -name '*_gap_heatmaps.png' \) \
    -delete
  for token_region in "${PRUNE_TOKEN_REGION_LIST[@]}"; do
    candidate_plan_root="${candidate_root}/${token_region}"
    candidate_strategies="${PRUNE_STRATEGY}"
    if [[ "${token_region}" == "special_patch_mean" ]]; then
      candidate_strategies="${DUAL_TOKEN_STRATEGY}"
      [[ -z "${candidate_strategies}" ]] && continue
    fi
    mkdir -p "${candidate_plan_root}"
    rm -f "${candidate_plan_root}/prune_plan_all.json"
    rm -f "${candidate_plan_root}"/prune_plan_*_drop_grid.png
    for ds in ${ANALYSIS_DATASETS//,/ } ${EVAL_DATASETS//,/ }; do
      ensure_directory_symlink "${plan_root}/${ds}" "${candidate_plan_root}/${ds}"
    done
    if compact_log; then
      printf '\n  pruning candidate\n    model: %s\n    strategies: %s\n    metric: cka\n    token region: %s\n    datasets: %s\n\n' \
        "${bb}" "${candidate_strategies}" "${token_region}" "${ANALYSIS_DATASETS}"
    fi
    run_command python -m analysis.prune_plan.build_prune_plan_v3 \
      --experiment "${bb}" \
      --dataset "${ANALYSIS_DATASETS}" \
      --backbone "${bb}" \
      --metric cka \
      --strategies "${candidate_strategies}" \
      --max-drop-budget "${PLAN_MAX_DROP}" \
      --token-region "${token_region}" \
      --prune-plan-root "${plan_root}" \
      --output "${candidate_plan_root}/prune_plan_all.json" \
      --viz-out-dir "${candidate_plan_root}"
  done
done

PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
if [[ -z "${PRUNE_STRATEGY//[[:space:],]/}" || -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ]]; then
  echo "error: PRUNE_STRATEGY, PLAN_MIN_DROP, PLAN_MAX_DROP, and PLAN_DROP_STEP are required" >&2
  exit 1
fi
# --- Stage 7/10: Evaluate token-region candidates on training datasets ---
stage_log "7/10" "Evaluate token-region candidates on training datasets"

for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_pruning_evals \
    "${bb}" \
    "${OUTPUT_ROOT}/${bb}/.prune_token_region_candidates/all" \
    "${PRUNE_STRATEGY}" \
    "${ANALYSIS_DATASETS}" \
    1 \
    "train/all"
  if [[ -n "${DUAL_TOKEN_STRATEGY}" ]]; then
    run_pruning_evals \
      "${bb}" \
      "${OUTPUT_ROOT}/${bb}/.prune_token_region_candidates/special_patch_mean" \
      "${DUAL_TOKEN_STRATEGY}" \
      "${ANALYSIS_DATASETS}" \
      1 \
      "train/special_patch_mean"
  fi
done

# --- Stage 7.5/10: Select pointwise token-region winners ---
stage_log "7.5/10" "Select pointwise token-region winners"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  plan_root="${OUTPUT_ROOT}/${bb}"
  candidate_root="${plan_root}/.prune_token_region_candidates"
  finalize_args=(
    --experiment-root "${plan_root}"
    --all-bundle "${candidate_root}/all/prune_plan_all.json"
    --selection-datasets "${ANALYSIS_DATASETS}"
    --min-drop "${PLAN_MIN_DROP}"
    --max-drop "${PLAN_MAX_DROP}"
    --drop-step "${PLAN_DROP_STEP}"
    --output "${plan_root}/prune_plan_all.json"
    --viz-out-dir "${plan_root}"
  )
  if [[ -n "${DUAL_TOKEN_STRATEGY}" ]]; then
    finalize_args+=(
      --special-patch-mean-bundle "${candidate_root}/special_patch_mean/prune_plan_all.json"
    )
  fi
  run_command python -m analysis.prune_plan.finalize_token_region_plan "${finalize_args[@]}"
done

# --- Stage 7.6/10: Evaluate finalized pruning plans on held-out datasets ---
stage_log "7.6/10" "Evaluate finalized pruning plans on held-out datasets"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_pruning_evals \
    "${bb}" \
    "${OUTPUT_ROOT}/${bb}" \
    "${PRUNE_STRATEGY}" \
    "${EVAL_DATASETS}" \
    0 \
    "held-out/final"
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
  for ds in ${ANALYSIS_DATASETS//,/ } 7scenes_train waymo_training; do
    rm -rf "${cache_root}/${ds}/cache"
  done
  BACKBONES="${bb}" \
      RUN="${ANALYSIS_DATASETS}" \
      CACHE="${ANALYSIS_DATASETS}" \
      SKIP_REDUNDANCY=1 \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      run_pipeline_script run_cache_redundancy.sh

  BACKBONES="${bb}" \
      DATASETS="${ANALYSIS_DATASETS}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      EPOCHS="${EPOCHS}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN:-0}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL="${ANALYSIS_DATASETS},${EVAL_DATASETS}" \
      RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY}" \
      run_pipeline_script run_recovery.sh

  for ds in ${ANALYSIS_DATASETS//,/ } 7scenes_train waymo_training; do
    rm -rf "${cache_root}/${ds}/cache"
  done
done

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
BACKBONES="${ALL_BACKBONES}"
parse_backbones_list
ALL_BACKBONES_LIST=("${BB_LIST[@]}")
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
# Plan-building / train datasets used by skip-eval, redundancy, prune, remaining-*, and recovery.
TRAINING_DATASETS="${TRAINING_DATASETS:-scannet_dev50,nuscenes_train}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special,patch,all}"
POINT="${POINT:-}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-gt_single,gt_double,ours_single,ours_double,cosine_single,cosine_double,inv_mse_single,inv_mse_double,replaceme_single,replaceme_double,shortgpt,simpler}"
RECOVERY_PRUNE_PLANS="${RECOVERY_PRUNE_PLANS:-prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24}"
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_linearpatch,recovery_ghost,recovery_ghost_sequential,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"


export OUTPUT_ROOT
export COMPACT_LOG="${COMPACT_LOG:-1}"

compact_log() {
  [[ "${COMPACT_LOG}" == "1" || "${COMPACT_LOG}" == "true" || "${COMPACT_LOG}" == "yes" ]]
}

stage_log() {
  if compact_log; then
    printf '\n[%s] %s\n' "$1" "$2"
  fi
}

run_command() {
  if compact_log; then
    "$@" >/dev/null
  else
    "$@"
  fi
}


# 7. Complete and evaluate selected physical multi-interval prune plans.
#    Use explicit PLAN_GROUP="i9_j19,i8_j20", or select an inclusive strategy
#    drop-count range with PRUNE_STRATEGY + PLAN_MIN_DROP + PLAN_MAX_DROP + PLAN_DROP_STEP.
#    Example: PLAN_MIN_DROP=0 PLAN_MAX_DROP=24 PLAN_DROP_STEP=2 evaluates
#    the unique plan selectors selected by all enabled Single/Double variants,
#    ShortGPT, and SIMPLER at drops 2,4,...,24.
#    Missing results are discovered in order: two training datasets, then four
#    held-out datasets. Evaluation is grouped by physical selector so one model
#    load serves every dataset missing that selector. Existing summary.json files
#    are always reused. Missing training single-interval summaries are fatal
#    because step 2 must have produced them.
#    Per-plan lower eval results use the canonical physical selector:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/<selector>/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/<selector>/summary.json
#    This stage writes evaluation summaries only. The Pruning Plan tab reloads
#    those summaries and renders all four views interactively without saved PNGs.
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
if [[ -z "${PLAN_GROUP//[[:space:],]/}" && -z "${PRUNE_STRATEGY//[[:space:],]/}" && ( -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ) ]]; then
  echo "error: PLAN_GROUP is required (e.g. PLAN_GROUP='i9_j19,i8_j20') or use PRUNE_STRATEGY+PLAN_MIN_DROP+PLAN_MAX_DROP+PLAN_DROP_STEP." >&2
  exit 1
fi
stage_log "7/10" "Evaluate pruning plans"

run_pruning_backbone() {
  local bb="$1"
  local plan_root="${OUTPUT_ROOT}/${bb}"
  local schedule
  local -a dataset_specs=()
  local ds
  for ds in ${TRAINING_DATASETS//,/ }; do
    dataset_specs+=(--dataset-spec "${ds}:${ds}:1")
  done
  for ds in ${EVAL_DATASETS//,/ }; do
    dataset_specs+=(--dataset-spec "${ds}:${ds}:0")
  done
  schedule="$(
    python -m analysis.prune_plan.stage7 \
        --prune-plan-root "${plan_root}" \
        --strategies "${PRUNE_STRATEGY}" \
        --selectors "${PLAN_GROUP}" \
        --min-drop "${PLAN_MIN_DROP}" \
        --max-drop "${PLAN_MAX_DROP}" \
        --drop-step "${PLAN_DROP_STEP}" \
        "${dataset_specs[@]}"
  )"

  if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" ]]; then
    return
  fi

  local selector dataset_keys
  while IFS=$'\t' read -r selector dataset_keys; do
    [[ -z "${selector}" ]] && continue
    echo "   [evaluate] ${bb}/${selector}: ${dataset_keys}"
    BACKBONES="${bb}" \
      RUN="${dataset_keys}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PLAN_GROUP="${selector}" \
      PLAN_STRATEGY="" \
      PLAN_MIN_DROP="" \
      PLAN_MAX_DROP="" \
      PLAN_DROP_STEP="" \
      WRITE_SUMMARY=0 \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
  done <<< "${schedule}"
}

for bb in "${ALL_BACKBONES_LIST[@]}"; do
  run_pruning_backbone "${bb}"
done


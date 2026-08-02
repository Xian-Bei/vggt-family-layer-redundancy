#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
ALL_BACKBONES_SPACE_SEPARATED="${ALL_BACKBONES//,/ }"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
MAX_SCENES="${MAX_SCENES:-0}"
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






# 6. Build the prune-only multi-strategy bundle:
#    GT, Ours, and ReplaceMe each expose Single and Double variants. GT/Ours
#    Double select the better one/two-interval plan with the 0.01 threshold;
#    ReplaceMe follows the official angular-distance greedy scan: num_A=2 uses
#    two equal-length non-overlapping blocks, allows adjacency, and therefore
#    only defines even total-drop budgets. Other two-interval methods leave at
#    least one retained unit between components.
#    ShortGPT follows its official unconstrained global BI pruning order and may
#    form any number of intervals; SIMPLER remains a suffix interval.
#    Ours/GT Single/Double paths require each interval at budget k to overlap
#    an interval at budget k+1. Double-to-Double continuity is one-to-one:
#    both prior intervals must retain distinct overlapping successors.
#    GT/Ours paths start at the largest available budget <= 24 and scan downward.
#    No strategy, including any baseline, generates a plan above budget 24.
#    Independently, Adaptive Double may split from one interval to two only
#    once; after returning to one interval it stays one.
#    The selected plans are visualized only in the combined structural drop grid.
#    Bundle and plan-level figures are saved under:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    <OUTPUT_ROOT>/<backbone>/prune_plan_*.png
#    Metric plots are generated only after step 7 has completed the exact plans.
PRUNE_DATASET="${PRUNE_DATASET:-scannet_dev50,nuscenes_train}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-gt_single,gt_double,ours_single,ours_double,cosine_single,cosine_double,inv_mse_single,inv_mse_double,replaceme_single,replaceme_double,shortgpt,simpler}"
stage_log "6/9" "Build pruning plans"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  if compact_log; then
    printf '\n  pruning plan\n    model: %s\n    strategies: %s\n    metric: cka\n    token region: %s\n    datasets: %s\n\n' \
      "${bb}" "${PRUNE_STRATEGY}" "${PRUNE_TOKEN_REGION}" "${PRUNE_DATASET}"
  fi
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/prune*.json
  rm -f "${plan_root}"/prune*.png
  find "${plan_root}" -type f \
    \( -name '*_gt_components.png' -o -name '*_paths.png' -o -name '*_gap_heatmaps.png' \) \
    -delete

  run_command conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${PRUNE_DATASET}" \
    --backbone "${bb}" \
    --metric cka \
    --strategies "${PRUNE_STRATEGY}" \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done



# 7. Complete and evaluate selected physical multi-interval prune plans.
#    Use explicit PLAN_GROUP="i9_j19,i8_j20", or select an inclusive strategy
#    drop-count range with PLAN_STRATEGY + PLAN_MIN_DROP + PLAN_MAX_DROP + PLAN_DROP_STEP.
#    Example: PLAN_MIN_DROP=0 PLAN_MAX_DROP=24 PLAN_DROP_STEP=2 evaluates
#    the unique plan selectors selected by all enabled Single/Double variants,
#    ShortGPT, and SIMPLER at drops 2,4,...,24.
#    Results are completed in order: two training datasets, then four held-out
#    datasets. Existing summary.json files are always reused. Missing training
#    single-interval summaries are fatal because step 2 must have produced them.
#    Per-plan lower eval results use the canonical physical selector:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/<selector>/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/<selector>/summary.json
#    This stage writes evaluation summaries only. The Pruning Plan tab reloads
#    those summaries and renders all four views interactively without saved PNGs.
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_STRATEGY="${PLAN_STRATEGY:-gt_single,gt_double,ours_single,ours_double,cosine_single,cosine_double,inv_mse_single,inv_mse_double,replaceme_single,replaceme_double,shortgpt,simpler}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
if [[ -z "${PLAN_GROUP//[[:space:],]/}" && -z "${PLAN_STRATEGY//[[:space:],]/}" && ( -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ) ]]; then
  echo "error: PLAN_GROUP is required (e.g. PLAN_GROUP='i9_j19,i8_j20') or use PLAN_STRATEGY+PLAN_MIN_DROP+PLAN_MAX_DROP+PLAN_DROP_STEP." >&2
  exit 1
fi
stage_log "7/9" "Evaluate pruning plans"

run_pruning_dataset() {
  local bb="$1"
  local dataset_key="$2"
  local dataset_dir="$3"
  local training_flag="$4"
  local plan_root="${OUTPUT_ROOT}/${bb}"
  local result_root="${plan_root}/${dataset_dir}/eval/pruning"
  local training_arg=()
  [[ "${training_flag}" == "1" ]] && training_arg=(--training)
  local missing=()
  local missing_output
  missing_output="$(
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m analysis.prune_plan.stage7 \
        --prune-plan-root "${plan_root}" \
        --result-root "${result_root}" \
        --dataset "${dataset_dir}" \
        --strategies "${PLAN_STRATEGY}" \
        --selectors "${PLAN_GROUP}" \
        --min-drop "${PLAN_MIN_DROP}" \
        --max-drop "${PLAN_MAX_DROP}" \
        --drop-step "${PLAN_DROP_STEP}" \
        "${training_arg[@]}"
  )"
  if [[ -n "${missing_output}" ]]; then
    mapfile -t missing <<< "${missing_output}"
  fi
  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "   [reuse] ${bb}/${dataset_dir}: all selected plans already complete"
  else
    local missing_csv
    missing_csv="$(IFS=,; echo "${missing[*]}")"
    echo "   [missing] ${bb}/${dataset_dir}: ${missing_csv}"
    if [[ "${PRUNING_EVAL_DRY_RUN}" != "1" ]]; then
      BACKBONES="${bb}" \
        RUN="${dataset_key}" \
        EXPORT_VIS_2D=0 \
        EXPORT_VIS_3D=0 \
        PRUNE=1 \
        PRUNE_PLAN_ROOT="${plan_root}" \
        PLAN_GROUP="${missing_csv}" \
        PLAN_STRATEGY="" \
        PLAN_MIN_DROP="" \
        PLAN_MAX_DROP="" \
        PLAN_DROP_STEP="" \
        WRITE_SUMMARY=0 \
        MAX_SCENES="${MAX_SCENES}" \
        CONDA_ENV="${CONDA_ENV}" \
        DEVICE="${DEVICE}" \
        bash "run_eval.sh"
    fi
  fi
}

for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  run_pruning_dataset "${bb}" SCANNET_DEV50 scannet_dev50 1
  run_pruning_dataset "${bb}" NUSCENES_TRAIN nuscenes_train 1

  run_pruning_dataset "${bb}" SCANNET_TEST50 scannet_test50 0
  run_pruning_dataset "${bb}" 7SCENES_TEST 7scenes_test 0
  run_pruning_dataset "${bb}" NUSCENES_VAL nuscenes_val 0
  run_pruning_dataset "${bb}" WAYMO_VALIDATION waymo_validation 0
done

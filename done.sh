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

# 1. Evaluate the full pretrained model (upper bound, no pruning).
#    Eval records are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/upper/eval_record_v3.json
#    Optional per-scene visualizations live under the same eval/upper directory.
stage_log "1/9" "Evaluate pretrained models"
BACKBONES="${ALL_BACKBONES}" \
    RUN=SCANNET_DEV50,7SCENES_TRAIN,NUSCENES_TRAIN,WAYMO_TRAINING,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=1 \
    PRUNE=0 \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    bash "run_eval.sh"



# 2. Brute-force skip-GT sanity check on plan-building datasets.
#    Skip-eval interval summaries are saved in i_j layout:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE/summary.json
#    The interval index is:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/skip_eval_interval_index_v3.json
stage_log "2/9" "Evaluate skip intervals"
BACKBONES="${ALL_BACKBONES}" \
    SKIP_GT_EVAL_SCANNET_DEV50=1 \
    SKIP_GT_EVAL_7SCENES_TRAIN=0 \
    SKIP_GT_EVAL_NUSCENES_TRAIN=1 \
    SKIP_GT_EVAL_WAYMO_TRAINING=0 \
    SKIP_GT_EVAL_MAX_SPAN=0 \
    bash "run_skip_eval.sh"



# 3. Compute redundancy/similarity graphs for plan-building datasets.
#    No activation cache is saved here. Redundancy graphs are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/redundancy/redundancy_graph_<region>.npz
stage_log "3/9" "Compute redundancy"
BACKBONES="${ALL_BACKBONES}" \
    RUN=SCANNET_DEV50,NUSCENES_TRAIN \
    CACHE="" \
    METRICS=cka,cosine,angular,inv_mse,cca \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    bash "run_cache_redundancy.sh"


# 4. Brute-force skip-GT evaluation on the GT-knee single-interval-pruned model.
#    The base prune plan is structurally applied before dataset iteration. Skip
#    candidates whose combined original drop set is contiguous reuse the existing
#    eval/pruning/iS_jE result. Only true two-interval candidates are evaluated:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE_iU_jV/summary.json
#    Checkpoints are isolated by the base interval under:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/_remaining_skip/iS_jE/
#    The reuse/evaluation mapping is:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/remaining_skip_eval_index_v3.json
#    The base interval is the GT single-interval knee selected from the first-pass
#    skip-eval results. Its auditable selection record and curve plot are:
#    <OUTPUT_ROOT>/<backbone>/remaining_skip_gt_knee.json
#    <OUTPUT_ROOT>/<backbone>/remaining_skip_gt_knee.png
REMAINING_GT_DATASETS="${REMAINING_GT_DATASETS:-${PRUNE_DATASET:-scannet_dev50,nuscenes_train}}"
REMAINING_GT_KNEE_THRESHOLD="${REMAINING_GT_KNEE_THRESHOLD:-0.05}"
REMAINING_GT_KNEE_DROP_STEP="${REMAINING_GT_KNEE_DROP_STEP:-1}"
declare -A REMAINING_BASE_SELECTOR_BY_BACKBONE=()
declare -A REMAINING_BASE_DROP_COUNT_BY_BACKBONE=()
declare -A REMAINING_BASE_SELECTION_PATH_BY_BACKBONE=()
stage_log "4/9" "Evaluate skip intervals after pruning"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  base_selection_path="${OUTPUT_ROOT}/${bb}/remaining_skip_gt_knee.json"
  read -r base_selector knee_drop_count < <(
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m analysis.prune_plan.gt_single_interval_knee \
        --root "${OUTPUT_ROOT}/${bb}" \
        --datasets "${REMAINING_GT_DATASETS}" \
        --threshold "${REMAINING_GT_KNEE_THRESHOLD}" \
        --drop-step "${REMAINING_GT_KNEE_DROP_STEP}" \
        --output "${base_selection_path}"
  )
  REMAINING_BASE_SELECTOR_BY_BACKBONE["${bb}"]="${base_selector}"
  REMAINING_BASE_DROP_COUNT_BY_BACKBONE["${bb}"]="${knee_drop_count}"
  REMAINING_BASE_SELECTION_PATH_BY_BACKBONE["${bb}"]="${base_selection_path}"
  BACKBONES="${bb}" \
      RUN=SCANNET_DEV50,NUSCENES_TRAIN \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PRUNE_PLAN_NAME="${base_selector}" \
      EXPECTED_DROP_COUNT="${knee_drop_count}" \
      BASE_SELECTION_PATH="${base_selection_path}" \
      REMAINING_SKIP_EVAL_MAX_SPAN=0 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      bash "run_remaining_skip_eval.sh"
done

# 5. Re-run inference with the step-4 GT-knee-pruned model and compute
#    remaining-layer CKA.
#    The prune plan is applied before any dataset scene is read. Surviving-layer
#    activations exist only in memory for the current scene and are released
#    immediately after CKA accumulation. Only graphs and provenance are saved:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/remaining_redundancy/redundancy_graph_<region>.npz
#    <OUTPUT_ROOT>/<backbone>/<dataset>/remaining_redundancy/manifest.json
#    The exact base interval, expected drop count, and auditable selection record
#    are reconstructed from step-4 outputs so this stage can run independently.
REMAINING_GT_DATASETS="${REMAINING_GT_DATASETS:-${PRUNE_DATASET:-scannet_dev50,nuscenes_train}}"
REMAINING_GT_KNEE_THRESHOLD="${REMAINING_GT_KNEE_THRESHOLD:-0.05}"
REMAINING_GT_KNEE_DROP_STEP="${REMAINING_GT_KNEE_DROP_STEP:-1}"
REMAINING_SIM_TOKEN_REGIONS="${REMAINING_SIM_TOKEN_REGIONS:-special,patch,all}"
stage_log "5/9" "Compute remaining redundancy after pruning"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  base_selection_path="${OUTPUT_ROOT}/${bb}/remaining_skip_gt_knee.json"
  read -r base_selector knee_drop_count < <(
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m analysis.prune_plan.gt_single_interval_knee \
        --root "${OUTPUT_ROOT}/${bb}" \
        --datasets "${REMAINING_GT_DATASETS}" \
        --threshold "${REMAINING_GT_KNEE_THRESHOLD}" \
        --drop-step "${REMAINING_GT_KNEE_DROP_STEP}" \
        --output "${base_selection_path}"
  )
  BACKBONES="${bb}" \
      RUN=SCANNET_DEV50,NUSCENES_TRAIN \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PRUNE_PLAN_NAME="${base_selector}" \
      EXPECTED_DROP_COUNT="${knee_drop_count}" \
      BASE_SELECTION_PATH="${base_selection_path}" \
      METRICS=cosine,angular,inv_mse,cca \
      SIM_TOKEN_REGIONS="${REMAINING_SIM_TOKEN_REGIONS}" \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      bash "run_remaining_redundancy.sh"
done





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


# 8. Build recovery-plan bundles from prune_plan_all.json.
#    Recovery plans decide cache units and either trainable units or closed-form boundary operators.
#    Supported methods: recovery_lora, recovery_train, recovery_linearpatch, recovery_ghost,
#    recovery_ghost_interval, recovery_ghost_interval_sequential,
#    recovery_ghost_cos, recovery_ghost_cos_more, recovery_ghost_cos_scale,
#    recovery_replaceme_ls, recovery_replaceme_cos, recovery_replaceme_cos_more,
#    recovery_replaceme_cos_scale. Prune&Comp is temporarily excluded.
#    All recovery methods are saved in:
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_all.json
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_linearpatch,recovery_ghost,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
stage_log "8/9" "Build recovery plans"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}/recovery_plan_all.json"

  for method in ${RECOVERY_METHODS//,/ }; do
    if compact_log; then
      printf '\n  recovery plan\n    model: %s\n    method: %s\n    spans: all\n\n' \
        "${bb}" "${method#recovery_}"
    fi
    max_prefix_for_method="${MAX_LORA_ANCHORS}"
    case "${method}" in
      recovery_linearpatch|recovery_ghost|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_ghost_cos_more|recovery_replaceme_ls|recovery_replaceme_cos_more|recovery_prunecomp)
        max_prefix_for_method=1
        ;;
    esac
    for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
      run_command conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_recovery_plan \
        --experiment "${bb}" \
        --backbone "${bb}" \
        --prune-plan-root "${plan_root}" \
        --recovery-plan-root "${plan_root}" \
        --method "${method}" \
        --lora-prefix "${lora_prefix}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}"
    done
  done
done




RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_linearpatch,recovery_ghost,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"
EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
# 9. Save minimal activation caches, run recovery, and evaluate recovered models.
#    Minimal cache is saved only for the selected prune/recovery plan union:
#    <OUTPUT_ROOT>/<backbone>/<train_dataset>/cache/cache_manifest_v3.json
#    Recovery checkpoints/artifacts are saved to:
#    <OUTPUT_ROOT>/<backbone>/recovery/<prune_spanK>/<recovery_plan>/checkpoints/latest.pt
#    Recovered eval results are saved in interval-result layout:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/summary.json
#    Recover metric plots are written to:
#    run_recovery.sh automatically calls summary_v3 after each backbone and writes:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery_plan_<eval_dataset>_recovery_drop.png
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_<backbone>_recovery_drop.png
#    Summary table:
#    <OUTPUT_ROOT>/<backbone>/summary_comparison.md

stage_log "9/9" "Cache, recover, and evaluate"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
  BACKBONES="${bb}" \
      RUN=SCANNET_DEV50,NUSCENES_TRAIN \
      CACHE=SCANNET_DEV50,NUSCENES_TRAIN \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      bash "run_cache_redundancy.sh"
  
  BACKBONES="${bb}" \
      DATASETS=scannet_dev50,nuscenes_train \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      EPOCHS="${EPOCHS}" \
      MAX_SCENES="${MAX_SCENES}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL=SCANNET_DEV50,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
      bash "run_recovery.sh"

  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
done

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
PLAN_MAX_DROP="${PLAN_MAX_DROP:-28}"
# Plan-building / train datasets used by skip-eval, redundancy, prune, remaining-*, and recovery.
TRAINING_DATASETS="${TRAINING_DATASETS:-scannet_dev50,nuscenes_train}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special,patch,all}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-gt_single,gt_double,ours_single,ours_double,cosine_single,cosine_double,inv_mse_single,inv_mse_double,cca_single,cca_double,replaceme_single,replaceme_double,shortgpt,simpler}"
# Strategy whose plans are used by recovery at the drop lengths below.
RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY:-gt_single,gt_double,ours_single,ours_double}"
RECOVERY_PRUNE_PLANS="${RECOVERY_PRUNE_PLANS:-prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24,prune_span26,prune_span28}"
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_linearpatch,recovery_ghost,recovery_ghost_sequential,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"
RECOVERY_PRUNE_STRATEGY_LIST=()
read -r -a _RECOVERY_PRUNE_STRATEGY_RAW <<< "${RECOVERY_PRUNE_STRATEGY//,/ }"
for _recovery_prune_strategy in "${_RECOVERY_PRUNE_STRATEGY_RAW[@]}"; do
  _recovery_prune_strategy="${_recovery_prune_strategy//[[:space:]]/}"
  [[ -z "${_recovery_prune_strategy}" ]] && continue
  case "${_recovery_prune_strategy}" in
    gt_single|gt_double|ours_single|ours_double|cosine_single|cosine_double|inv_mse_single|inv_mse_double|cca_single|cca_double|replaceme_single|replaceme_double|shortgpt|simpler)
      ;;
    *)
      echo "error: unsupported RECOVERY_PRUNE_STRATEGY entry=${_recovery_prune_strategy@Q}" >&2
      exit 1
      ;;
  esac
  if [[ ! " ${RECOVERY_PRUNE_STRATEGY_LIST[*]} " =~ " ${_recovery_prune_strategy} " ]]; then
    RECOVERY_PRUNE_STRATEGY_LIST+=("${_recovery_prune_strategy}")
  fi
done
if [[ "${#RECOVERY_PRUNE_STRATEGY_LIST[@]}" -eq 0 ]]; then
  echo "error: RECOVERY_PRUNE_STRATEGY must contain at least one strategy" >&2
  exit 1
fi
POINT="${POINT:-13}"
if [[ ! "${POINT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: POINT must be provided as a positive integer (number of layers to delete)" >&2
  exit 1
fi
export OUTPUT_ROOT
export COMPACT_LOG="${COMPACT_LOG:-1}"

# 1. Evaluate the full pretrained model (upper bound, no pruning).
#    Eval records are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/upper/eval_record_v3.json
#    Optional per-scene visualizations live under the same eval/upper directory.
stage_log "1/10" "Evaluate pretrained models"
BACKBONES="${ALL_BACKBONES}" \
    RUN="${TRAINING_DATASETS},${EVAL_DATASETS}" \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=1 \
    PRUNE=0 \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    bash "${REPO_ROOT}/scripts/run_eval.sh"



# 2. Brute-force skip-GT sanity check on plan-building datasets.
#    Skip-eval interval summaries are saved in i_j layout:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE/summary.json
#    The interval index is:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/skip_eval_interval_index_v3.json
stage_log "2/10" "Evaluate skip intervals"
BACKBONES="${ALL_BACKBONES}" \
    EVAL_DATASET="${TRAINING_DATASETS}" \
    SKIP_GT_EVAL_MAX_SPAN=0 \
    bash "${REPO_ROOT}/scripts/run_skip_eval.sh"



# 3. Compute redundancy/similarity graphs for plan-building datasets.
#    No activation cache is saved here. Redundancy graphs are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/redundancy/redundancy_graph_<region>.npz
stage_log "3/10" "Compute redundancy"
BACKBONES="${ALL_BACKBONES}" \
    RUN="${TRAINING_DATASETS}" \
    CACHE="" \
    METRICS=cka,cosine,angular,inv_mse,cca \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"


# 4. Brute-force skip-GT evaluation after manually choosing the number of layers
#    to delete with POINT. The canonical GT single-interval path maps POINT to
#    the concrete iS_jE interval before pruning.
#    The base prune plan is structurally applied before dataset iteration. Skip
#    candidates whose combined original drop set is contiguous reuse the existing
#    eval/pruning/iS_jE result. Only true two-interval candidates are evaluated:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE_iU_jV/summary.json
#    Checkpoints are isolated by the base interval under:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/_remaining_skip/iS_jE/
#    The reuse/evaluation mapping is:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/remaining_skip_eval_index_v3.json

stage_log "4/10" "Evaluate skip intervals after pruning"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  read -r base_selector drop_count < <(
    python -m analysis.prune_plan.gt_single_interval_at_point \
        --root "${OUTPUT_ROOT}/${bb}" \
        --datasets "${TRAINING_DATASETS}" \
        --point "${POINT}"
  )
  BACKBONES="${bb}" \
      RUN="${TRAINING_DATASETS}" \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PRUNE_PLAN_NAME="${base_selector}" \
      EXPECTED_DROP_COUNT="${drop_count}" \
      REMAINING_SKIP_EVAL_MAX_SPAN=0 \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_remaining_skip_eval.sh"
done

# 5. Re-run inference with the step-4 manually selected model and compute
#    remaining-layer CKA.
#    The prune plan is applied before any dataset scene is read. Surviving-layer
#    activations exist only in memory for the current scene and are released
#    immediately after CKA accumulation. Only graphs and provenance are saved:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/remaining_redundancy/redundancy_graph_<region>.npz
#    <OUTPUT_ROOT>/<backbone>/<dataset>/remaining_redundancy/manifest.json
#    The exact base interval is reconstructed from POINT so this stage can run
#    independently.
stage_log "5/10" "Compute remaining redundancy after pruning"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  read -r base_selector drop_count < <(
    python -m analysis.prune_plan.gt_single_interval_at_point \
        --root "${OUTPUT_ROOT}/${bb}" \
        --datasets "${TRAINING_DATASETS}" \
        --point "${POINT}"
  )
  BACKBONES="${bb}" \
      RUN="${TRAINING_DATASETS}" \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PRUNE_PLAN_NAME="${base_selector}" \
      EXPECTED_DROP_COUNT="${drop_count}" \
      METRICS=cosine,angular,inv_mse,cca \
      SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS}" \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_remaining_redundancy.sh"
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
#    GT/Ours paths start at the largest available budget <= PLAN_MAX_DROP and
#    scan downward. No strategy, including any baseline, generates a plan above
#    PLAN_MAX_DROP. The same limit controls plan generation here and evaluation
#    in step 7.
#    Independently, Adaptive Double may split from one interval to two only
#    once; after returning to one interval it stays one.
#    The selected plans are visualized only in the combined structural drop grid.
#    Bundle and plan-level figures are saved under:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    <OUTPUT_ROOT>/<backbone>/prune_plan_*.png
#    Metric plots are generated only after step 7 has completed the exact plans.
stage_log "6/10" "Build pruning plans"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  if compact_log; then
    printf '\n  pruning plan\n    model: %s\n    strategies: %s\n    metric: cka\n    token region: %s\n    datasets: %s\n\n' \
      "${bb}" "${PRUNE_STRATEGY}" "${PRUNE_TOKEN_REGION}" "${TRAINING_DATASETS}"
  fi
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/prune*.json
  rm -f "${plan_root}"/prune*.png
  find "${plan_root}" -type f \
    \( -name '*_gt_components.png' -o -name '*_paths.png' -o -name '*_gap_heatmaps.png' \) \
    -delete

  run_command python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${TRAINING_DATASETS}" \
    --backbone "${bb}" \
    --metric cka \
    --strategies "${PRUNE_STRATEGY}" \
    --max-drop-budget "${PLAN_MAX_DROP}" \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done



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


# 8. Build recovery-plan bundles from prune_plan_all.json.
#    Recovery plans decide cache units and either trainable units or closed-form boundary operators.
#    Supported methods: recovery_lora, recovery_train, recovery_linearpatch, recovery_ghost,
#    recovery_ghost_sequential,
#    recovery_ghost_interval, recovery_ghost_interval_sequential,
#    recovery_replaceme_ls, recovery_replaceme_cos.
#    All recovery methods are saved in:
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_all.json
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
stage_log "8/10" "Build recovery plans"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
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
      recovery_linearpatch|recovery_ghost|recovery_ghost_sequential|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_replaceme_ls|recovery_replaceme_cos)
        max_prefix_for_method=1
        ;;
    esac
    for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
      run_command python -m analysis.prune_plan.build_recovery_plan \
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




EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
export RECOVERY_CHUNK_ROWS="${RECOVERY_CHUNK_ROWS:-16384}"
# 9. Save minimal activation caches, run recovery, and evaluate recovered models.
#    Minimal cache is saved only for the selected prune/recovery plan union:
#    <OUTPUT_ROOT>/<backbone>/<train_dataset>/cache/cache_manifest_v3.json
#    Recovery checkpoints/artifacts are saved to:
#    <OUTPUT_ROOT>/<backbone>/recovery/<physical_selector>/<recovery_plan>/checkpoints/latest.pt
#    Recovered eval results are saved in interval-result layout:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<physical_selector>/<recovery_plan>/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<physical_selector>/<recovery_plan>/summary.json
#    Recover metric plots are written to:
#    scripts/run_recovery.sh automatically calls summary_v3 after each backbone and writes:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery_plan_<eval_dataset>_recovery_drop.png
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_<backbone>_recovery_drop.png
#    Summary table:
#    <OUTPUT_ROOT>/<backbone>/summary_comparison_<prune_strategy>.md
stage_log "9/10" "Cache, recover, and evaluate"
for bb in "${ALL_BACKBONES_LIST[@]}"; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  recovery_plans="$(
    python -c "import sys; from pathlib import Path; from analysis.prune_plan.recovery_plan import list_recovery_plan_names; print(','.join(list_recovery_plan_names(Path(sys.argv[1]))))" \
      "${cache_root}"
  )"
  recovery_prune_plans="$(
    python - "${cache_root}" "${RECOVERY_PRUNE_STRATEGY}" "${RECOVERY_PRUNE_PLANS}" <<'PY'
import re
import sys
from pathlib import Path

from analysis.recovery.shared_results import selectors_for_strategies

root = Path(sys.argv[1])
strategies = [value for value in re.split(r"[\s,]+", sys.argv[2]) if value]
requested = [value for value in re.split(r"[\s,]+", sys.argv[3]) if value]
try:
    selected = selectors_for_strategies(root, strategies, requested)
except ValueError as exc:
    raise SystemExit(f"error: {exc}") from exc
print(",".join(selected))
PY
  )"
  for ds in ${TRAINING_DATASETS//,/ } 7scenes_train waymo_training; do
    rm -rf "${cache_root}/${ds}/cache"
  done
  BACKBONES="${bb}" \
      RUN="${TRAINING_DATASETS}" \
      CACHE="${TRAINING_DATASETS}" \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
  
  BACKBONES="${bb}" \
      DATASETS="${TRAINING_DATASETS}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" \
      RECOVERY_PLAN_NAME="${recovery_plans}" \
      EPOCHS="${EPOCHS}" \
      MAX_SCENES="${MAX_SCENES}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      DEVICE="${DEVICE}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL="${TRAINING_DATASETS},${EVAL_DATASETS}" \
      RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY}" \
      bash "${REPO_ROOT}/scripts/run_recovery.sh"

  for ds in ${TRAINING_DATASETS//,/ } 7scenes_train waymo_training; do
    rm -rf "${cache_root}/${ds}/cache"
  done
done


# 10. Apply deployment-time compression after the best pruning + recovery
#     models for all selected backbones and evaluate with the unchanged protocol. FastVGGT token
#     merging and QuantVGGT fake quantization are evaluated as two alternatives.
#     No cache or recovery training is repeated.  Results are isolated under:
#     <OUTPUT_ROOT>/vggt/<dataset>/eval/testtime/
#       recovery_ghost_interval_sequential/<compression>/<selector>/
#     QuantVGGT defaults to the explicitly named official W4A4 transfer
#     parameters; set QUANTVGGT_PARAMS_DIR to a LaRe-aware calibration directory
#     for the main result.
stage_log "10/10" "Evaluate pruning + recovery + test-time compression"
OUTPUT_ROOT="${OUTPUT_ROOT}" \
    TESTTIME_BACKBONES="${ALL_BACKBONES}" \
    DEVICE="${DEVICE}" \
    MAX_SCENES="${MAX_SCENES}" \
    TESTTIME_RUN="${TRAINING_DATASETS},${EVAL_DATASETS}" \
    bash "${REPO_ROOT}/scripts/run_testtime_eval.sh"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
ALL_BACKBONES_SPACE_SEPARATED="${ALL_BACKBONES//,/ }"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
export OUTPUT_ROOT



# 1. Evaluate the full pretrained model (upper bound, no pruning).
#    Eval records are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/upper/eval_record_v3.json
#    Optional per-scene visualizations live under the same eval/upper directory.
BACKBONES="${ALL_BACKBONES}" \
    RUN=SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=1 \
    PRUNE=0 \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    bash "${REPO_ROOT}/scripts/run_eval.sh"


# 2. Compute redundancy/similarity graphs for plan-building datasets.
#    No activation cache is saved here. Redundancy graphs are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/redundancy/redundancy_graph_<region>.npz
BACKBONES="${ALL_BACKBONES}" \
    RUN=SCANNET_DEV50,7SCENES_TRAIN,NUSCENES_TRAIN,WAYMO_TRAINING \
    CACHE="" \
    METRICS=cka,cosine,inv_mse,cca \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"


# 3. Brute-force skip-GT sanity check on plan-building datasets.
#    Skip-eval interval summaries are saved in i_j layout:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE/summary.json
#    The interval index is:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/skip_eval_interval_index_v3.json
BACKBONES="${ALL_BACKBONES}" \
    EVAL_DATASET=scannet_dev50,nuscenes_train \
    SKIP_GT_EVAL_MAX_SPAN=0 \
    bash "${REPO_ROOT}/scripts/run_skip_eval.sh"



# 4. Compute additional CKA-only redundancy graphs for recovery/cache datasets.
#    No activation cache is saved here. Redundancy graphs are saved to:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/redundancy/redundancy_graph_<region>.npz
BACKBONES="${ALL_BACKBONES}" \
    RUN=7SCENES_TRAIN,WAYMO_TRAINING \
    CACHE="" \
    METRICS=cka \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"


# 5. Build the prune-only multi-strategy bundle:
#    It stores physical i_j intervals, set selectors, and strategy -> drop_count mappings.
#    Bundle and plan-level figures are saved under:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    <OUTPUT_ROOT>/<backbone>/prune_plan_*.png
#    Brute-force-supported source-dataset pruning plots are written to:
#    <OUTPUT_ROOT>/<backbone>/scannet_dev50/eval/scannet_dev50_pruning_drop.png
#    <OUTPUT_ROOT>/<backbone>/nuscenes_train/eval/nuscenes_train_pruning_drop.png
#    <OUTPUT_ROOT>/<backbone>/prune_plan_<backbone>_pruning_drop.png
#    GT-only 7-metric degradation plots are additionally written to:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_<backbone>_gt_degradation.png
#    Maximum CKA/Cosine/INV_MSE/CCA similarity at each exact j-i, overlaid with
#    the GT 7-metric mean degradation curve, is written to:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_<backbone>_max_similarity_by_drop_count.png
PRUNE_DATASET="${PRUNE_DATASET:-scannet_dev50,nuscenes_train}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-ours,gt,replaceme,inv_mse,shortgpt,simpler}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/prune*.json
  rm -f "${plan_root}"/prune*.png

  python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${PRUNE_DATASET}" \
    --backbone "${bb}" \
    --metric cka \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done


# 6. Evaluate selected prune plans on held-out datasets.
#    Use explicit PLAN_GROUP="i9_j19,i8_j20", or select an inclusive strategy
#    drop-count range with PLAN_STRATEGY + PLAN_MIN_DROP + PLAN_MAX_DROP + PLAN_DROP_STEP.
#    Example: PLAN_MIN_DROP=0 PLAN_MAX_DROP=24 PLAN_DROP_STEP=2 evaluates
#    the unique plan selectors selected by Ours/GT/ReplaceMe/INV_MSE/ShortGPT/SIMPLER
#    at drops 2,4,...,24.
#    Per-plan lower eval results are saved in the same interval-result layout as skip eval:
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<dataset>/eval/pruning/iS_jE/summary.json
#    Summary table:
#    <OUTPUT_ROOT>/<backbone>/summary_comparison.md
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_STRATEGY="${PLAN_STRATEGY:-ours}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
if [[ -z "${PLAN_GROUP//[[:space:],]/}" && -z "${PLAN_STRATEGY//[[:space:],]/}" && ( -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ) ]]; then
  echo "error: PLAN_GROUP is required (e.g. PLAN_GROUP='i9_j19,i8_j20') or use PLAN_STRATEGY+PLAN_MIN_DROP+PLAN_MAX_DROP+PLAN_DROP_STEP." >&2
  exit 1
fi
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  BACKBONES="${bb}" \
      RUN=SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PLAN_GROUP="${PLAN_GROUP}" \
      PLAN_STRATEGY="${PLAN_STRATEGY}" \
      PLAN_MIN_DROP="${PLAN_MIN_DROP}" \
      PLAN_MAX_DROP="${PLAN_MAX_DROP}" \
      PLAN_DROP_STEP="${PLAN_DROP_STEP}" \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
done


# 7. Build recovery-plan bundles from prune_plan_all.json.
#    Recovery plans decide cache units and either trainable units or closed-form boundary operators.
#    Supported methods: recovery_lora, recovery_train, recovery_diag, recovery_rotate, recovery_ghost.
#    Recovery-plan bundles are saved under:
#    <OUTPUT_ROOT>/<backbone>/recovery_*.json
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_lora,recovery_train,recovery_diag,recovery_rotate,recovery_ghost}"
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/recovery_*.json

  for method in ${RECOVERY_METHODS//,/ }; do
    max_prefix_for_method="${MAX_LORA_ANCHORS}"
    case "${method}" in
      recovery_diag|recovery_rotate|recovery_ghost)
        max_prefix_for_method=1
        ;;
    esac
    for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
      python -m analysis.prune_plan.build_recovery_plan \
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




RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_diag,recovery_rotate,recovery_ghost}"
EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
# 8. Save minimal activation caches, run recovery, and evaluate recovered models.
#    Minimal cache is saved only for the selected prune/recovery plan union:
#    <OUTPUT_ROOT>/<backbone>/<train_dataset>/cache/cache_manifest_v3.json
#    Recovery checkpoints/artifacts are saved to:
#    <OUTPUT_ROOT>/<backbone>/recovery/<prune_spanK>/<recovery_plan>/checkpoints/latest.pt
#    Recovered eval results are saved in interval-result layout:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/summary.json
#    Recover metric plots are written to:
#    scripts/run_recovery.sh automatically calls summary_v3 after each backbone and writes:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery_plan_<eval_dataset>_recovery_drop.png
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_<backbone>_recovery_drop.png
#    Summary table:
#    <OUTPUT_ROOT>/<backbone>/summary_comparison.md

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
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
  
  BACKBONES="${bb}" \
      DATASETS=scannet_dev50,nuscenes_train \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      EPOCHS="${EPOCHS}" \
      MAX_SCENES="${MAX_SCENES}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      DEVICE="${DEVICE}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL=SCANNET_DEV50,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
      bash "${REPO_ROOT}/scripts/run_recovery.sh"

  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
done

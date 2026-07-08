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



# 1. Evaluate the full pretrained model (upper bound, no pruning).
BACKBONES="${ALL_BACKBONES}" \
    RUN_SCANNET_DEV50=1 \
    RUN_SCANNET_TEST50=1 \
    RUN_7SCENES_TEST=1 \
    RUN_NUSCENES_VAL=1 \
    RUN_WAYMO_VALIDATION=1 \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=1 \
    PRUNE=0 \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    bash "run_eval.sh"


# 2. Compute redundancy/similarity graphs only; no activation cache is saved here.
#    These graphs drive prune_plan_all.json in stage 4.
BACKBONES="${ALL_BACKBONES}" \
    RUN_WAYMO_TRAINING=0 \
    RUN_SCANNET_DEV50=1 \
    RUN_7SCENES_TRAIN=1 \
    RUN_NUSCENES_TRAIN=1 \
    CACHE_SCANNET_DEV50=0 \
    CACHE_NUSCENES_TRAIN=0 \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=camera,patch,all \
    bash "run_cache_redundancy.sh"




# 3. Optional skip-GT sanity check: compare direct skip quality against CKA-based pruning choices.
BACKBONES="${ALL_BACKBONES}" \
    SKIP_GT_EVAL_SCANNET_DEV50=1 \
    SKIP_GT_EVAL_MAX_SPAN=0 \
    bash "run_skip_eval.sh"


# 4. Build the prune-only span bundle:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    It stores only span -> dropped unit indices, plus prune-only visualizations.
PRUNE_DATASET="${PRUNE_DATASET:-scannet_dev50}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/plan_budget*.json "${plan_root}"/plan_t*.json "${plan_root}"/plan_*.png
  rm -f "${plan_root}"/prune_plan_all.json "${plan_root}"/prune_plan_*.png

  conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${PRUNE_DATASET}" \
    --backbone "${bb}" \
    --metric cka \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done


# 5. Evaluate prune-only lower models.
#    Use explicit PLAN_GROUP="prune_span5,prune_span10", or select an inclusive
#    span range with PLAN_MIN_DROP + PLAN_MAX_DROP + PLAN_DROP_STEP.
#    Example: PLAN_MIN_DROP=0 PLAN_MAX_DROP=24 PLAN_DROP_STEP=2 evaluates
#    prune_span2, prune_span4, ..., prune_span24. prune_span0 is skipped.
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
if [[ -z "${PLAN_GROUP//[[:space:],]/}" && ( -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ) ]]; then
  echo "error: PLAN_GROUP is required (e.g. PLAN_GROUP='prune_span5,prune_span10') or use PLAN_MIN_DROP+PLAN_MAX_DROP+PLAN_DROP_STEP." >&2
  exit 1
fi
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  BACKBONES="${bb}" \
      RUN_SCANNET_DEV50=1 \
      RUN_SCANNET_TEST50=1 \
      RUN_7SCENES_TEST=1 \
      RUN_NUSCENES_VAL=1 \
      RUN_WAYMO_VALIDATION=1 \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
      PLAN_GROUP="${PLAN_GROUP}" \
      PLAN_MIN_DROP="${PLAN_MIN_DROP}" \
      PLAN_MAX_DROP="${PLAN_MAX_DROP}" \
      PLAN_DROP_STEP="${PLAN_DROP_STEP}" \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      bash "run_eval.sh"
done


# 6. Build recovery-plan bundles from prune_plan_all.json.
#    Recovery plans decide cache units and either trainable units or closed-form boundary operators.
#    Supported methods: recovery_lora, recovery_train, recovery_diag, recovery_rotate, recovery_ghost.
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_lora,recovery_train,recovery_diag,recovery_rotate,recovery_ghost}"
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/recovery_*.json "${plan_root}"/recovery_plan_*.png

  for method in ${RECOVERY_METHODS//,/ }; do
    max_prefix_for_method="${MAX_LORA_ANCHORS}"
    case "${method}" in
      recovery_diag|recovery_rotate|recovery_ghost)
        max_prefix_for_method=1
        ;;
    esac
    for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
      conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_recovery_plan \
        --experiment "${bb}" \
        --backbone "${bb}" \
        --prune-plan-root "${plan_root}" \
        --recovery-plan-root "${plan_root}" \
        --method "${method}" \
        --lora-prefix "${lora_prefix}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --viz-out-dir "${plan_root}"
    done
  done
done




RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_diag,recovery_rotate,recovery_ghost}"
EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
# 7. Save the minimal activation cache required by each backbone's selected
#    prune/recovery plan pair. Set PRUNE_PLAN_NAME or RECOVERY_PLAN_NAME to
#    override the per-backbone defaults for all backbones.
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
  BACKBONES="${bb}" \
      RUN_SCANNET_DEV50=1 \
      CACHE_SCANNET_DEV50=1 \
      RUN_7SCENES_TRAIN=1 \
      CACHE_7SCENES_TRAIN=1 \
      RUN_NUSCENES_TRAIN=1 \
      CACHE_NUSCENES_TRAIN=1 \
      RUN_WAYMO_TRAINING=1 \
      CACHE_WAYMO_TRAINING=1 \
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
      DATASETS=scannet_dev50,7scenes_train,nuscenes_train,waymo_training \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      EPOCHS="${EPOCHS}" \
      MAX_SCENES="${MAX_SCENES}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL_SCANNET_DEV50=1 \
      EVAL_SCANNET_TEST50=1 \
      EVAL_7SCENES_TEST=1 \
      EVAL_NUSCENES_VAL=1 \
      EVAL_WAYMO_VALIDATION=1 \
      bash "run_recover.sh"

  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
done

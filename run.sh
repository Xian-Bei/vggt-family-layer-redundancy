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
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
export OUTPUT_ROOT

plan_names_for_backbone() {
  local bb="$1"
  local default_prune_plan_name
  local default_training_plan_name
  case "${bb}" in
    pi3)
      default_prune_plan_name="prune_span19"
      default_training_plan_name="training_lora_l1_r32"
      ;;
    vggt)
      default_prune_plan_name="prune_span12"
      default_training_plan_name="training_lora_l1_r32"
      ;;
    vggt_omega)
      default_prune_plan_name="prune_span12"
      default_training_plan_name="training_original_l1"
      ;;
    *)
      echo "error: unknown backbone=${bb}" >&2
      exit 1
      ;;
  esac
  PRUNE_PLAN_NAME_FOR_BB="${PRUNE_PLAN_NAME:-${default_prune_plan_name}}"
  TRAINING_PLAN_NAME_FOR_BB="${TRAINING_PLAN_NAME:-${default_training_plan_name}}"
}


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
PRUNE_DATASET="${PRUNE_DATASET:-mean}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-min}"
PRUNE_EXCLUDE_TOKEN_REGIONS="${PRUNE_EXCLUDE_TOKEN_REGIONS:-}"
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
    --exclude-token-regions "${PRUNE_EXCLUDE_TOKEN_REGIONS}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done


# 5. Evaluate prune-only lower models.
#    Use explicit PLAN_GROUP="prune_span5,prune_span10", or auto-sample with
#    PLAN_MAX_DROP_RATIO + PLAN_TEST_COUNT.
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_MAX_DROP_RATIO="${PLAN_MAX_DROP_RATIO:-0.5}"
PLAN_TEST_COUNT="${PLAN_TEST_COUNT:-6}"
if [[ -z "${PLAN_GROUP//[[:space:],]/}" && ( -z "${PLAN_MAX_DROP_RATIO}" || -z "${PLAN_TEST_COUNT}" ) ]]; then
  echo "error: PLAN_GROUP is required (e.g. PLAN_GROUP='prune_span5,prune_span10') or use PLAN_MAX_DROP_RATIO+PLAN_TEST_COUNT." >&2
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
      PLAN_MAX_DROP_RATIO="${PLAN_MAX_DROP_RATIO}" \
      PLAN_TEST_COUNT="${PLAN_TEST_COUNT}" \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      bash "run_eval.sh"
done


# 6. Build training-plan bundles from prune_plan_all.json.
#    Training plans decide cache units and trainable units, independent of the prune bundle.
#    Supported methods: lora_prune and train_original.
TRAINING_METHODS="${TRAINING_METHODS:-lora_prune,train_original}"
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/training_*.json "${plan_root}"/training_plan_*.png

  for method in ${TRAINING_METHODS//,/ }; do
    for lora_prefix in $(seq 1 "${MAX_LORA_ANCHORS}"); do
      conda run -n "${CONDA_ENV}" python -m analysis.training_plan.build_training_plan \
        --experiment "${bb}" \
        --backbone "${bb}" \
        --prune-plan-root "${plan_root}" \
        --training-plan-root "${plan_root}" \
        --method "${method}" \
        --lora-prefix "${lora_prefix}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --viz-out-dir "${plan_root}"
    done
  done
done



# 7. Save the minimal activation cache required by each backbone's selected
#    prune/training plan pair. Set PRUNE_PLAN_NAME or TRAINING_PLAN_NAME to
#    override the per-backbone defaults for all backbones.
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  rm -rf "${cache_root}/scannet_dev50/cache" "${cache_root}/scannet/cache"
  plan_names_for_backbone "${bb}"
  BACKBONES="${bb}" \
      RUN_SCANNET_DEV50=1 \
      CACHE_SCANNET_DEV50=1 \
      RUN_7SCENES_TRAIN=0 \
      CACHE_7SCENES_TRAIN=0 \
      RUN_NUSCENES_TRAIN=1 \
      CACHE_NUSCENES_TRAIN=1 \
      RUN_WAYMO_TRAINING=0 \
      CACHE_WAYMO_TRAINING=0 \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      TRAINING_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME_FOR_BB}" \
      TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME_FOR_BB}" \
      bash "run_cache_redundancy.sh"
done



# 8. Train each backbone with its selected prune span and training plan.
EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  plan_names_for_backbone "${bb}"
  BACKBONES="${bb}" \
      DATASET=scannet_dev50 \
      PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME_FOR_BB}" \
      TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME_FOR_BB}" \
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
      bash "run_lora_skip_baseline.sh"
done


# # 9. Evaluate finetuned model checkpoints produced in stage 8.
# for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
#   for prune_plan_name in ${PRUNE_PLAN_NAME//,/ }; do
#     prune_plan_name="${prune_plan_name//[[:space:]]/}"
#     [[ -z "${prune_plan_name}" ]] && continue
#     ckpt="${OUTPUT_ROOT}/${bb}/scannet_dev50/train/${prune_plan_name}/${TRAINING_PLAN_NAME}/checkpoints/latest.pt"
#     if [[ ! -f "${ckpt}" ]]; then
#       echo "warning: skip model eval; missing checkpoint: ${ckpt}" >&2
#       continue
#     fi
#     BACKBONES="${bb}" \
#         RUN_SCANNET_DEV50=1 \
#         RUN_SCANNET_TEST50=1 \
#         RUN_7SCENES_TEST=1 \
#         RUN_NUSCENES_VAL=1 \
#         RUN_WAYMO_VALIDATION=1 \
#         EXPORT_VIS_2D=1 \
#         EXPORT_VIS_3D=0 \
#         PRUNE=1 \
#         PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${bb}" \
#         PRUNE_PLAN_NAME="${prune_plan_name}" \
#         TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME}" \
#         CHECKPOINT="${ckpt}" \
#         MAX_SCENES="${MAX_SCENES}" \
#         CONDA_ENV="${CONDA_ENV}" \
#         DEVICE="${DEVICE}" \
#         bash "run_eval.sh"
#   done
# done

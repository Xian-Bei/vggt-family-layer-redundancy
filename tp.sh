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


# 4. Build the prune-only span bundle:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    It stores only span -> dropped unit indices, plus prune-only visualizations.
PRUNE_DATASET="${PRUNE_DATASET:-mean}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_EXCLUDE_TOKEN_REGIONS="${PRUNE_EXCLUDE_TOKEN_REGIONS:-camera}"
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

exit 0
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

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



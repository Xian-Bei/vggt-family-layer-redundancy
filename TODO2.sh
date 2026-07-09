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


# 4. Build the prune-only multi-strategy interval bundle:
#    <OUTPUT_ROOT>/<backbone>/prune_plan_all.json
#    It stores physical i_j intervals plus strategy -> span/drop_count -> i_j mappings.
PRUNE_DATASET="${PRUNE_DATASET:-scannet_dev50,nuscenes_train}"
PRUNE_TOKEN_REGION="${PRUNE_TOKEN_REGION:-mean}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-ours,gt,cosine}"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}"/prune*.json

  conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${PRUNE_DATASET}" \
    --backbone "${bb}" \
    --metric cka \
    --token-region "${PRUNE_TOKEN_REGION}" \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done

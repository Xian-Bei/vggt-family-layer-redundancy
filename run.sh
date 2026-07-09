#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-pi3}"
ALL_BACKBONES_SPACE_SEPARATED="${ALL_BACKBONES//,/ }"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
MAX_SCENES="${MAX_SCENES:-0}"
export OUTPUT_ROOT




# 1. Evaluate the full pretrained model (upper bound, no pruning).
BACKBONES="${ALL_BACKBONES}" \
    RUN=NUSCENES_TRAIN \
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
    RUN=SCANNET_DEV50,NUSCENES_TRAIN \
    CACHE="" \
    METRICS=cca \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    SIM_TOKEN_REGIONS=special,patch,all \
    bash "run_cache_redundancy.sh"


# 3. Optional skip-GT sanity check: compare direct skip quality against CKA-based pruning choices.
BACKBONES="${ALL_BACKBONES}" \
    SKIP_GT_EVAL_SCANNET_DEV50=0 \
    SKIP_GT_EVAL_NUSCENES_TRAIN=1 \
    SKIP_GT_EVAL_MAX_SPAN=0 \
    bash "run_skip_eval.sh"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

REFERENCE_RESULTS_ROOT="${REFERENCE_RESULTS_ROOT:-analysis_outputs_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-main_results}"
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
CALIB_DATASETS="${CALIB_DATASETS:-scannet_dev50,7scenes_train,nuscenes_train,waymo_training}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"

python -m analysis.results.init_main_results \
  --source-root "${REFERENCE_RESULTS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --backbones "${BACKBONES}" \
  --calib-datasets "${CALIB_DATASETS}" \
  --eval-datasets "${EVAL_DATASETS}" \
  --max-drop-budget "${PLAN_MAX_DROP}"

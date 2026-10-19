#!/usr/bin/env bash
set -euo pipefail

# Temporary visualization re-export runner.
#
# It evaluates exactly three variants for each requested backbone:
#   1. original model
#   2. original model + FastVGGT
#   3. recovered LaRe model at the requested budget (without test-time compression)
#
# Defaults reproduce the budgets used by main.sh:
#   VGGT=10, Pi3=16, VGGT-Omega=18.
#
# Example:
#   conda run -n vggt bash scripts/maintenance/rerun_visualization_exports.sh
#   BACKBONES=vggt VGGT_BUDGET=10 EVAL_DATASETS=scannet_test50 \
#     MAX_SCENES=2 conda run -n vggt bash scripts/maintenance/rerun_visualization_exports.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-main_results}"
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
EVAL_DATASETS="${EVAL_DATASETS:-7scenes_test,nuscenes_val,waymo_validation}"
CALIB_DATASETS="${CALIB_DATASETS:-scannet_dev50,7scenes_train,nuscenes_train,waymo_training}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_FRAMES="${MAX_FRAMES:-30}"
VIS_MAX_POINTS="${VIS_MAX_POINTS:-0}"

VGGT_BUDGET="${VGGT_BUDGET:-10}"
PI3_BUDGET="${PI3_BUDGET:-16}"
VGGT_OMEGA_BUDGET="${VGGT_OMEGA_BUDGET:-18}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_ghost_interval_sequential_autogroup}"
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"

source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
install_error_diagnostics "${BASH_SOURCE[0]}"
parse_backbones_list

if [[ "${DEVICE}" != cuda* ]]; then
  echo "error: this runner requires a CUDA device; got DEVICE=${DEVICE@Q}" >&2
  exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

budget_for_backbone() {
  case "$1" in
    vggt) echo "${VGGT_BUDGET}" ;;
    pi3) echo "${PI3_BUDGET}" ;;
    vggt_omega) echo "${VGGT_OMEGA_BUDGET}" ;;
    *)
      echo "error: no visualization budget configured for backbone=$1" >&2
      return 1
      ;;
  esac
}

# Fail before loading any model if a requested recovery checkpoint is absent.
for bb in "${BB_LIST[@]}"; do
  budget="$(budget_for_backbone "${bb}")"
  if [[ ! "${budget}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: invalid budget for ${bb}: ${budget@Q}" >&2
    exit 1
  fi
  selection="$(
    python -m analysis.recovery.shared_results best-recovery-result \
      --plan-root "${OUTPUT_ROOT}/${bb}" \
      --drop-count "${budget}" \
      --datasets "${CALIB_DATASETS}" \
      --recovery-plan-name "${RECOVERY_PLAN_NAME}" \
      --prune-family retention
  )"
  IFS=$'\t' read -r _strategy selector _score <<< "${selection}"
  checkpoint="${OUTPUT_ROOT}/${bb}/recovery/${selector}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "error: missing recovered checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  printf '[preflight] backbone=%s budget=%s selector=%s checkpoint=%s\n' \
    "${bb}" "${budget}" "${selector}" "${checkpoint}"
done

echo
echo "== 1/3 Original models =="
BACKBONES="${BACKBONES}" \
RUN="${EVAL_DATASETS}" \
PRUNE=0 \
PRUNE_PLAN_NAME="" \
PLAN_STRATEGY="" \
CHECKPOINT="" \
RECOVERY_PLAN_NAME="" \
MODEL_NAME=upper \
INFERENCE_COMPRESSION=none \
EXPORT_VIS_2D=1 \
EXPORT_VIS_3D=1 \
VIS_MAX_POINTS="${VIS_MAX_POINTS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MAX_SCENES="${MAX_SCENES}" \
MAX_FRAMES="${MAX_FRAMES}" \
DEVICE="${DEVICE}" \
bash "${REPO_ROOT}/scripts/run_eval.sh"

echo
echo "== 2/3 Original models + FastVGGT =="
BACKBONES="${BACKBONES}" \
RUN="${EVAL_DATASETS}" \
PRUNE=0 \
PRUNE_PLAN_NAME="" \
PLAN_STRATEGY="" \
CHECKPOINT="" \
RECOVERY_PLAN_NAME="" \
MODEL_NAME=upper \
INFERENCE_COMPRESSION=fastvggt \
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START}" \
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO}" \
EXPORT_VIS_2D=1 \
EXPORT_VIS_3D=1 \
VIS_MAX_POINTS="${VIS_MAX_POINTS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MAX_SCENES="${MAX_SCENES}" \
MAX_FRAMES="${MAX_FRAMES}" \
DEVICE="${DEVICE}" \
bash "${REPO_ROOT}/scripts/run_eval.sh"

echo
echo "== 3/3 Recovered LaRe models at requested budgets =="
for bb in "${BB_LIST[@]}"; do
  budget="$(budget_for_backbone "${bb}")"
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TESTTIME_BACKBONES="${bb}" \
  TESTTIME_PRUNE_PLAN_NAME="prune_span${budget}" \
  TESTTIME_PRUNE_STRATEGY=retention \
  TESTTIME_RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
  TESTTIME_SELECTION_DATASETS="${CALIB_DATASETS}" \
  TESTTIME_COMPRESSIONS=none \
  TESTTIME_RUN="${EVAL_DATASETS}" \
  EXPORT_VIS_2D=1 \
  EXPORT_VIS_3D=1 \
  VIS_MAX_POINTS="${VIS_MAX_POINTS}" \
  MAX_SCENES="${MAX_SCENES}" \
  MAX_FRAMES="${MAX_FRAMES}" \
  DEVICE="${DEVICE}" \
  bash "${REPO_ROOT}/scripts/run_testtime_eval.sh"
done

echo
echo "Visualization re-export completed under ${OUTPUT_ROOT}."

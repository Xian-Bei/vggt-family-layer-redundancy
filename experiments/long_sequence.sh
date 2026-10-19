#!/usr/bin/env bash
set -euo pipefail

# Long-sequence generalization evaluation for the frozen operating points chosen
# by the original 30-frame experiments.  For each backbone this runs exactly:
#   1. Original
#   2. Original + FastVGGT
#   3. Ours (pruned + recovered LaRe)
# It evaluates accuracy and then measures the sequence-dependent efficiency
# metrics: end-to-end latency and peak GPU memory.  No visualization assets are
# exported.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# MAX_FRAMES accepts one value or a comma/space-separated sweep.  Multi-value
# runs are dispatched as isolated single-length invocations so all existing
# accuracy/benchmark resume semantics remain unchanged.
max_frames_raw="${MAX_FRAMES:-200}"
read -r -a max_frames_candidates <<< "${max_frames_raw//,/ }"
max_frames_list=()
for frame_count in "${max_frames_candidates[@]}"; do
  frame_count="${frame_count//[[:space:]]/}"
  [[ -z "${frame_count}" ]] && continue
  if [[ ! "${frame_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: every MAX_FRAMES entry must be a positive integer; got ${frame_count@Q}" >&2
    exit 1
  fi
  duplicate=0
  for existing_frame_count in "${max_frames_list[@]}"; do
    if [[ "${existing_frame_count}" == "${frame_count}" ]]; then
      duplicate=1
      break
    fi
  done
  [[ "${duplicate}" == "0" ]] && max_frames_list+=("${frame_count}")
done
if [[ "${#max_frames_list[@]}" -eq 0 ]]; then
  echo "error: MAX_FRAMES must contain at least one positive integer" >&2
  exit 1
fi

if [[ "${#max_frames_list[@]}" -gt 1 && "${TEST_LENGTH_SINGLE_RUN:-0}" != "1" ]]; then
  output_base="${OUTPUT_ROOT:-test_length_results}"
  printf 'Long-sequence sweep: MAX_FRAMES=%s, output base=%s\n' \
    "$(IFS=,; echo "${max_frames_list[*]}")" "${output_base}"
  for frame_count in "${max_frames_list[@]}"; do
    echo
    echo "######## MAX_FRAMES=${frame_count} ########"
    MAX_FRAMES="${frame_count}" \
    OUTPUT_ROOT="${output_base}/frames_${frame_count}" \
    TEST_LENGTH_SINGLE_RUN=1 \
    bash "${BASH_SOURCE[0]}"
  done
  exit 0
fi

MAX_FRAMES="${max_frames_list[0]}"
MODEL_ROOT="${MODEL_ROOT:-main_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-test_length_results/frames_${MAX_FRAMES}}"
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
CALIB_DATASETS="${CALIB_DATASETS:-scannet_dev50,7scenes_train,nuscenes_train,waymo_training}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"

VGGT_BUDGET="${VGGT_BUDGET:-10}"
PI3_BUDGET="${PI3_BUDGET:-16}"
VGGT_OMEGA_BUDGET="${VGGT_OMEGA_BUDGET:-18}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_ghost_interval_sequential_autogroup}"
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
TESTTIME_BENCHMARK_SCENES="${TESTTIME_BENCHMARK_SCENES:-10}"
SKIP_ACCURACY="${SKIP_ACCURACY:-0}"

source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
install_error_diagnostics "${BASH_SOURCE[0]}"
parse_backbones_list

if [[ "${DEVICE}" != cuda* ]]; then
  echo "error: long-sequence evaluation requires CUDA; got DEVICE=${DEVICE@Q}" >&2
  exit 1
fi
if [[ ! "${MAX_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MAX_FRAMES must be a positive integer; got ${MAX_FRAMES@Q}" >&2
  exit 1
fi
if [[ "${SKIP_ACCURACY}" != "0" && "${SKIP_ACCURACY}" != "1" ]]; then
  echo "error: SKIP_ACCURACY must be 0 or 1; got ${SKIP_ACCURACY@Q}" >&2
  exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

budget_for_backbone() {
  case "$1" in
    vggt) echo "${VGGT_BUDGET}" ;;
    pi3) echo "${PI3_BUDGET}" ;;
    vggt_omega) echo "${VGGT_OMEGA_BUDGET}" ;;
    *)
      echo "error: no long-sequence budget configured for backbone=$1" >&2
      return 1
      ;;
  esac
}

# Resolve all selectors from the frozen 30-frame results and fail before model
# loading if any required recovery checkpoint is unavailable.
for bb in "${BB_LIST[@]}"; do
  budget="$(budget_for_backbone "${bb}")"
  if [[ ! "${budget}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: invalid budget for ${bb}: ${budget@Q}" >&2
    exit 1
  fi
  selection="$(
    python -m analysis.recovery.shared_results best-recovery-result \
      --plan-root "${MODEL_ROOT}/${bb}" \
      --drop-count "${budget}" \
      --datasets "${CALIB_DATASETS}" \
      --recovery-plan-name "${RECOVERY_PLAN_NAME}" \
      --prune-family retention
  )"
  IFS=$'\t' read -r _strategy selector _score <<< "${selection}"
  checkpoint="${MODEL_ROOT}/${bb}/recovery/${selector}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "error: missing recovered checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  printf '[preflight] backbone=%s frames=%s budget=%s selector=%s\n' \
    "${bb}" "${MAX_FRAMES}" "${budget}" "${selector}"
done

if [[ "${SKIP_ACCURACY}" == "1" ]]; then
  echo
  echo "== Accuracy: skipped as requested (${MAX_FRAMES} frames/scene) =="
else
  echo
  echo "== Accuracy 1/3: Original models (${MAX_FRAMES} frames/scene) =="
BACKBONES="${BACKBONES}" \
RUN="${EVAL_DATASETS}" \
PRUNE=0 \
PRUNE_PLAN_NAME="" \
PLAN_STRATEGY="" \
CHECKPOINT="" \
RECOVERY_PLAN_NAME="" \
MODEL_NAME=upper \
INFERENCE_COMPRESSION=none \
EXPORT_VIS=0 \
EXPORT_VIS_2D=0 \
EXPORT_VIS_3D=0 \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MAX_SCENES="${MAX_SCENES}" \
MAX_FRAMES="${MAX_FRAMES}" \
DEVICE="${DEVICE}" \
bash "${REPO_ROOT}/scripts/run_eval.sh" || exit $?

echo
echo "== Accuracy 2/3: Original models + FastVGGT (${MAX_FRAMES} frames/scene) =="
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
EXPORT_VIS=0 \
EXPORT_VIS_2D=0 \
EXPORT_VIS_3D=0 \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MAX_SCENES="${MAX_SCENES}" \
MAX_FRAMES="${MAX_FRAMES}" \
DEVICE="${DEVICE}" \
bash "${REPO_ROOT}/scripts/run_eval.sh" || exit $?

echo
echo "== Accuracy 3/3: Ours, frozen 30-frame operating points (${MAX_FRAMES} frames/scene) =="
for bb in "${BB_LIST[@]}"; do
  budget="$(budget_for_backbone "${bb}")"
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TESTTIME_MODEL_ROOT="${MODEL_ROOT}" \
  TESTTIME_BACKBONES="${bb}" \
  TESTTIME_PRUNE_PLAN_NAME="prune_span${budget}" \
  TESTTIME_PRUNE_STRATEGY=retention \
  TESTTIME_RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
  TESTTIME_SELECTION_DATASETS="${CALIB_DATASETS}" \
  TESTTIME_COMPRESSIONS=none \
  TESTTIME_RUN="${EVAL_DATASETS}" \
  EXPORT_VIS=0 \
  EXPORT_VIS_2D=0 \
  EXPORT_VIS_3D=0 \
  MAX_SCENES="${MAX_SCENES}" \
  MAX_FRAMES="${MAX_FRAMES}" \
  DEVICE="${DEVICE}" \
  bash "${REPO_ROOT}/scripts/run_testtime_eval.sh" || exit $?
done
fi

echo
echo "== Efficiency: end-to-end latency and peak GPU memory =="
for bb in "${BB_LIST[@]}"; do
  budget="$(budget_for_backbone "${bb}")"
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TESTTIME_MODEL_ROOT="${MODEL_ROOT}" \
  TESTTIME_BACKBONES="${bb}" \
  TESTTIME_PRUNE_PLAN_NAME="prune_span${budget}" \
  TESTTIME_PRUNE_STRATEGY=retention \
  TESTTIME_RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
  TESTTIME_SELECTION_DATASETS="${CALIB_DATASETS}" \
  TESTTIME_BENCHMARK_VARIANTS=full,full_fastvggt,recovery \
  TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES=1 \
  TESTTIME_BENCHMARK_SCENES="${TESTTIME_BENCHMARK_SCENES}" \
  TESTTIME_RUN="${EVAL_DATASETS}" \
  MAX_SCENES="${MAX_SCENES}" \
  MAX_FRAMES="${MAX_FRAMES}" \
  DEVICE="${DEVICE}" \
  bash "${REPO_ROOT}/scripts/run_testtime_benchmark.sh" || exit $?
done

echo
echo "Long-sequence accuracy and efficiency results: ${OUTPUT_ROOT}"

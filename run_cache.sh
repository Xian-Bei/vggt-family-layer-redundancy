#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Dataset roots (run_eval.sh aligned for 7scenes/nrgbd)
SCANNET_ROOT="${SCANNET_ROOT:-data/scans_dev50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
RUN_SCANNET="${RUN_SCANNET:-1}"
RUN_7SCENES="${RUN_7SCENES:-0}"
RUN_NRGBD="${RUN_NRGBD:-0}"

# Shared knobs (kept identical across datasets)
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
# Cache storage dtype only (Stage1 autocast is fixed to bf16 on CUDA).
DTYPE="${DTYPE:-float32}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_FRAMES="${MAX_FRAMES:-50}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-0}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"
GT_PLY_DIR="${GT_PLY_DIR:-}"
# Per-scene timing in logs (1/true=on, 0/false=off). Python defaults to on if unset.
TIMING="${TIMING:-1}"

# run_eval.sh alignment for 7scenes/nrgbd input path
KF="${KF:-10}"
SIZE="${SIZE:-518}"

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local out_dir="analysis_outputs/cache/${dataset}"
  local skip_gt_eval="0"

  if [[ "${dataset}" != "scannet" ]]; then
    # Non-ScanNet currently reuses run_eval.sh 7andN preprocessing path and skips ScanNet-specific GT eval.
    skip_gt_eval="1"
  fi

  echo "== cache dataset=${dataset} =="
  echo "   data_root=${data_root}"
  echo "   output_dir=${out_dir}"
  echo "   timing=${TIMING}"

  extra_args=()
  if [[ -n "${GT_PLY_DIR}" ]]; then
    extra_args+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  if [[ "${skip_gt_eval}" == "1" ]]; then
    extra_args+=(--skip-gt-eval)
  fi
  if [[ "${TIMING}" != "1" && "${TIMING}" != "true" ]]; then
    extra_args+=(--no-timing)
  fi

  python -m analysis.redundancy.cache_dataset \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --output-dir "${out_dir}" \
    --hf-model "${HF_MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-scenes "${MAX_SCENES}" \
    --max-frames "${MAX_FRAMES}" \
    --max-tokens-per-layer "${MAX_TOKENS_PER_LAYER}" \
    --kf "${KF}" \
    --size "${SIZE}" \
    --chamfer-max-dist "${CHAMFER_MAX_DIST}" \
    --depth-conf-thresh "${DEPTH_CONF_THRESH}" \
    "${extra_args[@]}"
}

if [[ "${RUN_SCANNET}" == "1" || "${RUN_SCANNET}" == "true" ]]; then
  run_one_dataset "scannet" "${SCANNET_ROOT}"
fi
if [[ "${RUN_7SCENES}" == "1" || "${RUN_7SCENES}" == "true" ]]; then
  run_one_dataset "7scenes" "${ROOT_7SCENES}"
fi
if [[ "${RUN_NRGBD}" == "1" || "${RUN_NRGBD}" == "true" ]]; then
  run_one_dataset "nrgbd" "${ROOT_NRGBD}"
fi

echo "== done =="
echo "   Caches: analysis_outputs/cache/<dataset>/"

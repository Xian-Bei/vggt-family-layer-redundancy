#!/usr/bin/env bash
# =============================================================================
#   Skip eval + similarity analysis.
#   For each (i,j) and each skip mode, replay from layer i frame cache, skip
#   global blocks i..j, compare j-layer global outputs (new vs cache),
#   then continue to heads and compute GT deltas vs cache.
# =============================================================================

set -euo pipefail

# Repo root = directory containing this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Python must resolve the local package `vggt` and `analysis`
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# [Shared data/model]
DATA_ROOT="${DATA_ROOT:-data/scans_dev50/}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

# [Shared eval knobs]
GT_PLY_DIR="${GT_PLY_DIR:-}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"

CACHE_DIR="${CACHE_DIR:-analysis_outputs/scannet_cache}"
MAX_FRAMES="${MAX_FRAMES:-30}"
SIMILARITY_DIR="${SIMILARITY_DIR:-analysis_outputs/scannet_redundancy_similarity}"
SKIP_DIR="${SKIP_DIR:-analysis_outputs/scannet_redundancy_skip}"
MAX_SPAN="${MAX_SPAN:-24}"
MODES="${MODES:-both,identity}"
SIMILARITY_METRICS="${SIMILARITY_METRICS:-cka,cosine,neg_mse}"
TIMING="${TIMING:-1}"
HEATMAP_FILL="${HEATMAP_FILL:-nan}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-${DEVICE}}"
SAVE_PER_SCENE="${SAVE_PER_SCENE:-1}"
SAVE_HEATMAP_PNG="${SAVE_HEATMAP_PNG:-0}"
NUM_PROCS="${NUM_PROCS:-1}"
EXPORT_VIS="${EXPORT_VIS:-0}"
VIS_DIR="${VIS_DIR:-${SIMILARITY_DIR}/visualization}"
ENABLE_SIMILARITY="${ENABLE_SIMILARITY:-1}"
ENABLE_SKIP_GT_EVAL="${ENABLE_SKIP_GT_EVAL:-0}"
CACHE_SANITY_CHECK="${CACHE_SANITY_CHECK:-1}"
CACHE_SANITY_ATOL="${CACHE_SANITY_ATOL:-1e-6}"
CACHE_SANITY_RTOL="${CACHE_SANITY_RTOL:-1e-6}"

echo "== skip eval + similarity analysis =="
echo "    CACHE_DIR=${CACHE_DIR}"
echo "    SIMILARITY_DIR=${SIMILARITY_DIR}"
echo "    SKIP_DIR=${SKIP_DIR}"
echo "    ANALYSIS_DEVICE=${ANALYSIS_DEVICE}"
echo "    MODES=${MODES}"
echo "    SIMILARITY_METRICS=${SIMILARITY_METRICS}"
echo "    ENABLE_SIMILARITY=${ENABLE_SIMILARITY}"
echo "    ENABLE_SKIP_GT_EVAL=${ENABLE_SKIP_GT_EVAL}"

TIMING_ARGS=()
if [[ "${TIMING}" == "1" || "${TIMING}" == "true" ]]; then
  TIMING_ARGS=(--timing)
fi

EXTRA_ARGS=()
if [[ -n "${GT_PLY_DIR}" ]]; then
  EXTRA_ARGS+=(--gt-ply-dir "${GT_PLY_DIR}")
fi
if [[ "${ENABLE_SKIP_GT_EVAL}" != "1" && "${ENABLE_SKIP_GT_EVAL}" != "true" ]]; then
  EXTRA_ARGS+=(--skip-gt-eval)
fi
if [[ "${ENABLE_SIMILARITY}" != "1" && "${ENABLE_SIMILARITY}" != "true" ]]; then
  EXTRA_ARGS+=(--no-fusion-compute-similarity)
fi
if [[ "${EXPORT_VIS}" == "1" || "${EXPORT_VIS}" == "true" ]]; then
  EXTRA_ARGS+=(--export-visualization --visualization-dir "${VIS_DIR}")
fi
if [[ "${SAVE_HEATMAP_PNG}" == "1" || "${SAVE_HEATMAP_PNG}" == "true" ]]; then
  EXTRA_ARGS+=(--save-heatmap-png)
fi
if [[ "${CACHE_SANITY_CHECK}" != "1" && "${CACHE_SANITY_CHECK}" != "true" ]]; then
  EXTRA_ARGS+=(--disable-cache-sanity-check)
else
  EXTRA_ARGS+=(--cache-sanity-atol "${CACHE_SANITY_ATOL}" --cache-sanity-rtol "${CACHE_SANITY_RTOL}")
fi

MODES_RAW="${MODES#[}"
MODES_RAW="${MODES_RAW%]}"
IFS=',' read -r -a MODES_ARR <<< "${MODES_RAW}"
MODE_ARGS=()
for mode in "${MODES_ARR[@]}"; do
  mode="${mode// /}"
  if [[ -n "${mode}" ]]; then
    MODE_ARGS+=("${mode}")
  fi
done
if [[ "${#MODE_ARGS[@]}" -eq 0 ]]; then
  echo "    [error] MODES must contain at least one mode." >&2
  exit 1
fi

METRICS_RAW="${SIMILARITY_METRICS#[}"
METRICS_RAW="${METRICS_RAW%]}"
IFS=',' read -r -a METRICS_ARR <<< "${METRICS_RAW}"
METRIC_ARGS=()
for metric in "${METRICS_ARR[@]}"; do
  metric="${metric// /}"
  if [[ -n "${metric}" ]]; then
    METRIC_ARGS+=("${metric}")
  fi
done
if [[ ("${ENABLE_SIMILARITY}" == "1" || "${ENABLE_SIMILARITY}" == "true") && "${#METRIC_ARGS[@]}" -eq 0 ]]; then
  echo "    [error] SIMILARITY_METRICS must contain at least one metric." >&2
  exit 1
fi

SAVE_ARGS=()
if [[ "${SAVE_PER_SCENE}" == "1" || "${SAVE_PER_SCENE}" == "true" ]]; then
  SAVE_ARGS+=(--fusion-save-per-scene)
else
  SAVE_ARGS+=(--no-fusion-save-per-scene)
fi

if ! [[ "${NUM_PROCS}" =~ ^[0-9]+$ ]]; then
  echo "    [error] NUM_PROCS must be a positive integer, got: ${NUM_PROCS}" >&2
  exit 1
fi
PROC_COUNT="${NUM_PROCS}"
if [[ "${PROC_COUNT}" -lt 1 ]]; then
  PROC_COUNT=1
fi
MODE_CLI_ARGS=(--fusion-modes "${MODE_ARGS[@]}")
METRIC_CLI_ARGS=()
if [[ "${#METRIC_ARGS[@]}" -gt 0 ]]; then
  METRIC_CLI_ARGS=(--fusion-similarity-metrics "${METRIC_ARGS[@]}")
fi

COMMON_ARGS=(
  --cache-dir "${CACHE_DIR}"
  --similarity-output-dir "${SIMILARITY_DIR}"
  --skip-output-dir "${SKIP_DIR}"
  --data-root "${DATA_ROOT}"
  --hf-model "${HF_MODEL}"
  --max-frames "${MAX_FRAMES}"
  --device "${ANALYSIS_DEVICE}"
  --skip-loss-device "${DEVICE}"
  --skip-loss-dtype "${DTYPE}"
  --heatmap-fill "${HEATMAP_FILL}"
  --chamfer-max-dist "${CHAMFER_MAX_DIST}"
  --depth-conf-thresh "${DEPTH_CONF_THRESH}"
  --fusion-max-span "${MAX_SPAN}"
  --fusion-num-procs "${NUM_PROCS}"
  "${MODE_CLI_ARGS[@]}"
  "${METRIC_CLI_ARGS[@]}"
  "${SAVE_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
  "${TIMING_ARGS[@]}"
)

if [[ "${PROC_COUNT}" -le 1 ]]; then
  python -m analysis.redundancy.analyze_scannet_cache "${COMMON_ARGS[@]}"
else
  echo "    Running with ${PROC_COUNT} scene-sharded worker processes."
  SHARD_SCENE_ROOT="${SKIP_DIR}/.scene_shards"
  rm -rf "${SHARD_SCENE_ROOT}"
  mkdir -p "${SHARD_SCENE_ROOT}"
  mkdir -p "${SIMILARITY_DIR}/per_scene" "${SKIP_DIR}/per_scene"

  mapfile -t SCENE_IDS < <(CACHE_DIR_ENV="${CACHE_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
manifest_path = Path(os.environ["CACHE_DIR_ENV"]) / "manifest.json"
try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    data = {}
scenes = data.get("scenes", []) if isinstance(data, dict) else []
for item in scenes:
    if isinstance(item, dict):
        sid = str(item.get("scene_id", "")).strip()
        if sid:
            print(sid)
PY
)
  if [[ "${#SCENE_IDS[@]}" -eq 0 ]]; then
    echo "    [error] No scene_id found in ${CACHE_DIR}/manifest.json" >&2
    exit 1
  fi
  if [[ "${PROC_COUNT}" -gt "${#SCENE_IDS[@]}" ]]; then
    PROC_COUNT="${#SCENE_IDS[@]}"
  fi

  for ((i=0; i<PROC_COUNT; i++)); do
    : > "${SHARD_SCENE_ROOT}/worker_${i}.txt"
  done
  for idx in "${!SCENE_IDS[@]}"; do
    shard_idx=$((idx % PROC_COUNT))
    echo "${SCENE_IDS[idx]}" >> "${SHARD_SCENE_ROOT}/worker_${shard_idx}.txt"
  done

  GPU_IDS=()
  if [[ "${DEVICE}" != "cpu" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  fi
  GPU_COUNT="${#GPU_IDS[@]}"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    echo "    Detected ${GPU_COUNT} GPUs; assigning workers in round-robin."
  fi

  PIDS=()
  WORKER_IDS=()
  for ((worker_idx=0; worker_idx<PROC_COUNT; worker_idx++)); do
    scene_file="${SHARD_SCENE_ROOT}/worker_${worker_idx}.txt"
    if [[ ! -s "${scene_file}" ]]; then
      continue
    fi
    gpu_msg="cpu/default"
    if [[ "${GPU_COUNT}" -gt 0 ]]; then
      gpu_id="${GPU_IDS[$((worker_idx % GPU_COUNT))]}"
      gpu_msg="gpu:${gpu_id}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python -m analysis.redundancy.analyze_scannet_cache \
        "${COMMON_ARGS[@]}" \
        --scene-ids-file "${scene_file}" \
        --no-fusion-write-aggregate &
    else
      python -m analysis.redundancy.analyze_scannet_cache \
        "${COMMON_ARGS[@]}" \
        --scene-ids-file "${scene_file}" \
        --no-fusion-write-aggregate &
    fi
    echo "      worker ${worker_idx}: scenes=$(wc -l < "${scene_file}") ${gpu_msg}"
    PIDS+=("$!")
    WORKER_IDS+=("${worker_idx}")
  done

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[i]}"; then
      echo "    [error] worker ${WORKER_IDS[i]} failed." >&2
      exit 1
    fi
  done

  python -m analysis.redundancy.analyze_scannet_cache \
    "${COMMON_ARGS[@]}" \
    --fusion-aggregate-from-per-scene
  rm -rf "${SHARD_SCENE_ROOT}"
fi

echo "== done =="
if [[ "${ENABLE_SIMILARITY}" == "1" || "${ENABLE_SIMILARITY}" == "true" ]]; then
  echo "    Similarity summary:   ${SIMILARITY_DIR}/summary_metrics_similarity.csv"
  echo "    Similarity scene CSV: ${SIMILARITY_DIR}/per_scene_scores_similarity.csv"
  echo "    Similarity report:    ${SIMILARITY_DIR}/full_report_similarity.json"
fi
if [[ "${ENABLE_SKIP_GT_EVAL}" == "1" || "${ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
  echo "    Skip summary:         ${SKIP_DIR}/summary_metrics_skip.csv"
  echo "    Skip scene CSV:       ${SKIP_DIR}/per_scene_scores_skip.csv"
  echo "    Skip report:          ${SKIP_DIR}/full_report_skip.json"
fi
if [[ "${SAVE_HEATMAP_PNG}" == "1" || "${SAVE_HEATMAP_PNG}" == "true" ]]; then
  if [[ "${ENABLE_SIMILARITY}" == "1" || "${ENABLE_SIMILARITY}" == "true" ]]; then
    echo "    Similarity figures:   ${SIMILARITY_DIR}/*.png"
  fi
  if [[ "${ENABLE_SKIP_GT_EVAL}" == "1" || "${ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
    echo "    Skip figures:         ${SKIP_DIR}/*.png"
  fi
fi
if [[ "${SAVE_PER_SCENE}" == "1" || "${SAVE_PER_SCENE}" == "true" ]]; then
  if [[ "${ENABLE_SIMILARITY}" == "1" || "${ENABLE_SIMILARITY}" == "true" ]]; then
    echo "    Similarity per-scene: ${SIMILARITY_DIR}/per_scene/*_similarity.json"
  fi
  if [[ "${ENABLE_SKIP_GT_EVAL}" == "1" || "${ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
    echo "    Skip per-scene:       ${SKIP_DIR}/per_scene/*_skip.json"
  fi
  if [[ "${SAVE_HEATMAP_PNG}" == "1" || "${SAVE_HEATMAP_PNG}" == "true" ]]; then
    if [[ "${ENABLE_SIMILARITY}" == "1" || "${ENABLE_SIMILARITY}" == "true" ]]; then
      echo "    Similarity scene figs: ${SIMILARITY_DIR}/per_scene/*.png"
    fi
    if [[ "${ENABLE_SKIP_GT_EVAL}" == "1" || "${ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
      echo "    Skip scene figs:       ${SKIP_DIR}/per_scene/*.png"
    fi
  fi
fi

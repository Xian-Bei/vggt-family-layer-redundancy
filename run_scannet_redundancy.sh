#!/usr/bin/env bash
# =============================================================================
# ScanNet v2 (or compatible) redundancy pipeline — stages 0–4
# =============================================================================
# Stage 0: Weight-only analysis (no images / no activation cache). Writes
#          WEIGHT_ANALYSIS_DIR/weight_report.json + bar charts.
# Stage 1: For every scene under DATA_ROOT, run VGGT inference and save
#          activation caches; GT evaluation and visualization are done directly
#          from in-memory predictions (no per-scene prediction-file round-trip).
#          Stage1 GT metrics now include:
#            - pose / geometry: ate, are, rpe_rot, rpe_trans, chamfer_distance
#            - depth: depth_abs_rel, depth_delta1
#          ScanNet depth GT is aligned to the model input via the same RGB crop-mode
#          preprocessing used by VGGT, clipped to the FastVGGT-valid indoor range
#          (1e-3m, 10m),
#          and aggregated across scenes by direct mean (FastVGGT-style).
# Stage 2: skip eval + similarity analysis:
#            For each (i,j) and each skip mode, replay from layer i frame cache,
#            skip global blocks i..j, compare j-layer global outputs (new vs Stage1 cached),
#            then continue to heads and compute GT deltas vs Stage1.
#
# Usage (from repo root):
#   chmod +x run_scannet_redundancy.sh
#   ./run_scannet_redundancy.sh
#
# Or override paths / limits:
#   DATA_ROOT=data/scannetv2 HF_MODEL=facebook/VGGT-1B ./run_scannet_redundancy.sh
#
# Expected data layouts (discover_scannet_scenes):
#   - DATA_ROOT/scans/sceneXXXX_XX/color/*.jpg   (official-style)
#   - DATA_ROOT/sceneXXXX_XX/color/*.jpg         (scans content directly under root)
#   - Fallback: scene/images/* or scene/* images
# =============================================================================

# Set 0 to skip 
STAGE0="${STAGE0:-0}"
STAGE1="${STAGE1:-1}"
STAGE2="${STAGE2:-0}"

set -euo pipefail

# Repo root = directory containing this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Python must resolve the local package `vggt` and `analysis`
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# User-tunable variables (override from environment before calling this script)
# -----------------------------------------------------------------------------

# [Shared data/model]
# DATA_ROOT: ScanNet root (supports DATA_ROOT/scans/... and DATA_ROOT/scene...).
# HF_MODEL: HuggingFace VGGT model id.
# DEVICE/DTYPE: default inference device and autocast dtype.
DATA_ROOT="${DATA_ROOT:-data/scans_dev50/}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

# [Shared eval knobs]
# GT_PLY_DIR optionally overrides where *_vh_clean_2.ply are searched.
# CHAMFER_MAX_DIST / DEPTH_CONF_THRESH are FastVGGT-aligned reconstruction settings.
# Depth eval itself additionally uses ScanNet GT depth aligned with RGB crop-mode preprocessing,
# valid GT mask derived from depth values, and the FastVGGT-style 10m depth cap.
GT_PLY_DIR="${GT_PLY_DIR:-}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"

# [Stage 0: weight-only analysis]
# WEIGHT_ANALYSIS_DIR: output root for checkpoint-level weight stats and plots.
WEIGHT_ANALYSIS_DIR="${WEIGHT_ANALYSIS_DIR:-analysis_outputs/scannet_weight_analysis}"

# [Stage 1: cache + baseline evaluation]
# CACHE_DIR: stage1 outputs (manifest, activation cache, eval_scannet.json, optional visualization).
# MAX_SCENES/MAX_FRAMES control scene/frame sampling.
# MAX_TOKENS_PER_LAYER is kept only for compatibility; Stage1 now saves only
# full aggregated_concat_per_layer and no longer writes separate flattened caches.
# EXPORT_VIS_STAGE1 controls whether stage1 visualization assets are exported (default OFF).
CACHE_DIR="${CACHE_DIR:-analysis_outputs/scannet_cache}"
MAX_SCENES="${MAX_SCENES:-50}"
MAX_FRAMES="${MAX_FRAMES:-30}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-0}"
EXPORT_VIS_STAGE1="${EXPORT_VIS_STAGE1:-0}"
STAGE1_VIS_DIR="${STAGE1_VIS_DIR:-${CACHE_DIR}/visualization}"

# [Stage 2: skip eval + similarity analysis]
# STAGE2_SIMILARITY_DIR / STAGE2_SKIP_DIR: separated Stage2 outputs.
# STAGE2_MAX_SPAN caps replay span for (i,j) pairs.
# STAGE2_MODES controls skipped-global approximations, comma-separated.
# STAGE2_SIMILARITY_METRICS controls metrics for comparing Stage1 vs replayed j-layer global outputs.
# STAGE2_ENABLE_SIMILARITY controls whether Stage2 computes similarity.
# STAGE2_ENABLE_SKIP_GT_EVAL controls whether Stage2 computes skip-layer GT deltas.
# SAVE_PER_SCENE_STAGE2 controls whether per-scene Stage2 JSON is written (default ON).
# SAVE_HEATMAP_PNG_STAGE2 controls PNG heatmap generation (default OFF).
# STAGE2_NUM_PROCS controls Stage2 multiprocessing (scene-sharded workers).
STAGE2_SIMILARITY_DIR="${STAGE2_SIMILARITY_DIR:-analysis_outputs/scannet_redundancy_similarity}"
STAGE2_SKIP_DIR="${STAGE2_SKIP_DIR:-analysis_outputs/scannet_redundancy_skip}"
STAGE2_MAX_SPAN="${STAGE2_MAX_SPAN:-24}"
STAGE2_MODES="${STAGE2_MODES:-both}"
STAGE2_SIMILARITY_METRICS="${STAGE2_SIMILARITY_METRICS:-cka,cosine,neg_mse}"
TIMING="${TIMING:-1}"
HEATMAP_FILL="${HEATMAP_FILL:-nan}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-${DEVICE}}"
SAVE_PER_SCENE_STAGE2="${SAVE_PER_SCENE_STAGE2:-1}"
SAVE_HEATMAP_PNG_STAGE2="${SAVE_HEATMAP_PNG_STAGE2:-0}"
STAGE2_NUM_PROCS="${STAGE2_NUM_PROCS:-1}"
EXPORT_VIS_STAGE2="${EXPORT_VIS_STAGE2:-0}"
STAGE2_VIS_DIR="${STAGE2_VIS_DIR:-${STAGE2_SIMILARITY_DIR}/visualization}"
STAGE2_ENABLE_SIMILARITY="${STAGE2_ENABLE_SIMILARITY:-1}"
STAGE2_ENABLE_SKIP_GT_EVAL="${STAGE2_ENABLE_SKIP_GT_EVAL:-0}"
STAGE1_SANITY_CHECK="${STAGE1_SANITY_CHECK:-1}"
STAGE1_SANITY_ATOL="${STAGE1_SANITY_ATOL:-1e-6}"
STAGE1_SANITY_RTOL="${STAGE1_SANITY_RTOL:-1e-6}"

if ! [[ "${STAGE1_CACHE_DIR+set}" == "set" ]]; then
  STAGE1_CACHE_DIR="${CACHE_DIR}"
fi

# -----------------------------------------------------------------------------
# Stage 0 — checkpoint weight stats (no cache / no images)
# -----------------------------------------------------------------------------
if [[ "${STAGE0}" == "1" || "${STAGE0}" == "true" ]]; then
  echo "== Stage 0: weight analysis (checkpoint only) =="
  echo "    WEIGHT_ANALYSIS_DIR=${WEIGHT_ANALYSIS_DIR}"
  echo "    HF_MODEL=${HF_MODEL}"

  python -m analysis.redundancy.analyze_vggt_weights \
    --hf-model "${HF_MODEL}" \
    --output-dir "${WEIGHT_ANALYSIS_DIR}"

fi

# -----------------------------------------------------------------------------
# Stage 1 — inference + cache global activations for every scene
# -----------------------------------------------------------------------------
if [[ "${STAGE1}" == "1" || "${STAGE1}" == "true" ]]; then
  echo "== Stage 1: caching global-block activations + direct GT eval =="
  echo "    DATA_ROOT=${DATA_ROOT}"
  echo "    CACHE_DIR=${CACHE_DIR}"

  CACHE_EVAL_ARGS=()
  if [[ -n "${GT_PLY_DIR}" ]]; then
    CACHE_EVAL_ARGS+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  CACHE_EVAL_ARGS+=(--chamfer-max-dist "${CHAMFER_MAX_DIST}" --depth-conf-thresh "${DEPTH_CONF_THRESH}")
  if [[ "${EXPORT_VIS_STAGE1}" == "1" || "${EXPORT_VIS_STAGE1}" == "true" ]]; then
    CACHE_EVAL_ARGS+=(--export-visualization --visualization-dir "${STAGE1_VIS_DIR}")
  fi
  python -m analysis.redundancy.cache_scannet \
    --data-root "${DATA_ROOT}" \
    --output-dir "${CACHE_DIR}" \
    --hf-model "${HF_MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-scenes "${MAX_SCENES}" \
    --max-frames "${MAX_FRAMES}" \
    --max-tokens-per-layer "${MAX_TOKENS_PER_LAYER}" \
    "${CACHE_EVAL_ARGS[@]}"
fi

# python -m analysis.redundancy.view_scene_viser --vis-root analysis_outputs/scannet_cache/visualization --scene-id scene0000_00 --port 8080
# python -m analysis.redundancy.view_results_gradio


# -----------------------------------------------------------------------------
# Stage 2 — skip eval + similarity analysis
# -----------------------------------------------------------------------------
if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
  echo "== Stage 2: skip eval + similarity analysis =="
  echo "    CACHE_DIR=${CACHE_DIR}"
  echo "    STAGE2_SIMILARITY_DIR=${STAGE2_SIMILARITY_DIR}"
  echo "    STAGE2_SKIP_DIR=${STAGE2_SKIP_DIR}"
  echo "    ANALYSIS_DEVICE=${ANALYSIS_DEVICE}"
  echo "    STAGE2_MODES=${STAGE2_MODES}"
  echo "    STAGE2_SIMILARITY_METRICS=${STAGE2_SIMILARITY_METRICS}"
  echo "    STAGE2_ENABLE_SIMILARITY=${STAGE2_ENABLE_SIMILARITY}"
  echo "    STAGE2_ENABLE_SKIP_GT_EVAL=${STAGE2_ENABLE_SKIP_GT_EVAL}"

  TIMING_ARGS=()
  if [[ "${TIMING}" == "1" || "${TIMING}" == "true" ]]; then
    TIMING_ARGS=(--timing)
  fi

  STAGE2_EXTRA_ARGS=()
  if [[ -n "${GT_PLY_DIR}" ]]; then
    STAGE2_EXTRA_ARGS+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  if [[ "${STAGE2_ENABLE_SKIP_GT_EVAL}" != "1" && "${STAGE2_ENABLE_SKIP_GT_EVAL}" != "true" ]]; then
    STAGE2_EXTRA_ARGS+=(--skip-gt-eval)
  fi
  if [[ "${STAGE2_ENABLE_SIMILARITY}" != "1" && "${STAGE2_ENABLE_SIMILARITY}" != "true" ]]; then
    STAGE2_EXTRA_ARGS+=(--no-fusion-compute-similarity)
  fi
  if [[ "${EXPORT_VIS_STAGE2}" == "1" || "${EXPORT_VIS_STAGE2}" == "true" ]]; then
    STAGE2_EXTRA_ARGS+=(--export-visualization --visualization-dir "${STAGE2_VIS_DIR}")
  fi
  if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
    STAGE2_EXTRA_ARGS+=(--save-heatmap-png)
  fi
  if [[ ! "${STAGE1_SANITY_CHECK}" == "1" && ! "${STAGE1_SANITY_CHECK}" == "true" ]]; then
    STAGE2_EXTRA_ARGS+=(--disable-stage1-sanity-check)
  else
    STAGE2_EXTRA_ARGS+=(--stage1-sanity-atol "${STAGE1_SANITY_ATOL}" --stage1-sanity-rtol "${STAGE1_SANITY_RTOL}")
  fi

  STAGE2_MODES_RAW="${STAGE2_MODES#[}"
  STAGE2_MODES_RAW="${STAGE2_MODES_RAW%]}"
  IFS=',' read -r -a STAGE2_MODES_ARR <<< "${STAGE2_MODES_RAW}"
  STAGE2_MODE_ARGS=()
  for mode in "${STAGE2_MODES_ARR[@]}"; do
    mode="${mode// /}"
    if [[ -n "${mode}" ]]; then
      STAGE2_MODE_ARGS+=("${mode}")
    fi
  done
  if [[ "${#STAGE2_MODE_ARGS[@]}" -eq 0 ]]; then
    echo "    [error] STAGE2_MODES must contain at least one mode." >&2
    exit 1
  fi

  STAGE2_METRICS_RAW="${STAGE2_SIMILARITY_METRICS#[}"
  STAGE2_METRICS_RAW="${STAGE2_METRICS_RAW%]}"
  IFS=',' read -r -a STAGE2_METRICS_ARR <<< "${STAGE2_METRICS_RAW}"
  STAGE2_METRIC_ARGS=()
  for metric in "${STAGE2_METRICS_ARR[@]}"; do
    metric="${metric// /}"
    if [[ -n "${metric}" ]]; then
      STAGE2_METRIC_ARGS+=("${metric}")
    fi
  done
  if [[ ("${STAGE2_ENABLE_SIMILARITY}" == "1" || "${STAGE2_ENABLE_SIMILARITY}" == "true") && "${#STAGE2_METRIC_ARGS[@]}" -eq 0 ]]; then
    echo "    [error] STAGE2_SIMILARITY_METRICS must contain at least one metric." >&2
    exit 1
  fi

  STAGE2_SAVE_ARGS=()
  if [[ "${SAVE_PER_SCENE_STAGE2}" == "1" || "${SAVE_PER_SCENE_STAGE2}" == "true" ]]; then
    STAGE2_SAVE_ARGS+=(--fusion-save-per-scene)
  else
    STAGE2_SAVE_ARGS+=(--no-fusion-save-per-scene)
  fi

  if ! [[ "${STAGE2_NUM_PROCS}" =~ ^[0-9]+$ ]]; then
    echo "    [error] STAGE2_NUM_PROCS must be a positive integer, got: ${STAGE2_NUM_PROCS}" >&2
    exit 1
  fi
  STAGE2_PROC_COUNT="${STAGE2_NUM_PROCS}"
  if [[ "${STAGE2_PROC_COUNT}" -lt 1 ]]; then
    STAGE2_PROC_COUNT=1
  fi
  STAGE2_MODE_CLI_ARGS=(--fusion-modes "${STAGE2_MODE_ARGS[@]}")
  STAGE2_METRIC_CLI_ARGS=()
  if [[ "${#STAGE2_METRIC_ARGS[@]}" -gt 0 ]]; then
    STAGE2_METRIC_CLI_ARGS=(--fusion-similarity-metrics "${STAGE2_METRIC_ARGS[@]}")
  fi

  COMMON_STAGE2_ARGS=(
    --cache-dir "${CACHE_DIR}"
    --similarity-output-dir "${STAGE2_SIMILARITY_DIR}"
    --skip-output-dir "${STAGE2_SKIP_DIR}"
    --data-root "${DATA_ROOT}"
    --hf-model "${HF_MODEL}"
    --max-frames "${MAX_FRAMES}"
    --device "${ANALYSIS_DEVICE}"
    --skip-loss-device "${DEVICE}"
    --skip-loss-dtype "${DTYPE}"
    --heatmap-fill "${HEATMAP_FILL}"
    --chamfer-max-dist "${CHAMFER_MAX_DIST}"
    --depth-conf-thresh "${DEPTH_CONF_THRESH}"
    --fusion-max-span "${STAGE2_MAX_SPAN}"
    --fusion-num-procs "${STAGE2_NUM_PROCS}"
    "${STAGE2_MODE_CLI_ARGS[@]}"
    "${STAGE2_METRIC_CLI_ARGS[@]}"
    "${STAGE2_SAVE_ARGS[@]}"
    "${STAGE2_EXTRA_ARGS[@]}"
    "${TIMING_ARGS[@]}"
  )

  if [[ "${STAGE2_PROC_COUNT}" -le 1 ]]; then
    python -m analysis.redundancy.analyze_scannet_cache "${COMMON_STAGE2_ARGS[@]}"
  else
    echo "    Running Stage2 with ${STAGE2_PROC_COUNT} scene-sharded worker processes."
    SHARD_SCENE_ROOT="${STAGE2_SKIP_DIR}/.scene_shards"
    rm -rf "${SHARD_SCENE_ROOT}"
    mkdir -p "${SHARD_SCENE_ROOT}"
    mkdir -p "${STAGE2_SIMILARITY_DIR}/per_scene" "${STAGE2_SKIP_DIR}/per_scene"

    mapfile -t STAGE2_SCENE_IDS < <(CACHE_DIR_ENV="${CACHE_DIR}" python - <<'PY'
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
    if [[ "${#STAGE2_SCENE_IDS[@]}" -eq 0 ]]; then
      echo "    [error] No scene_id found in ${CACHE_DIR}/manifest.json" >&2
      exit 1
    fi
    if [[ "${STAGE2_PROC_COUNT}" -gt "${#STAGE2_SCENE_IDS[@]}" ]]; then
      STAGE2_PROC_COUNT="${#STAGE2_SCENE_IDS[@]}"
    fi

    for ((i=0; i<STAGE2_PROC_COUNT; i++)); do
      : > "${SHARD_SCENE_ROOT}/worker_${i}.txt"
    done
    for idx in "${!STAGE2_SCENE_IDS[@]}"; do
      shard_idx=$((idx % STAGE2_PROC_COUNT))
      echo "${STAGE2_SCENE_IDS[idx]}" >> "${SHARD_SCENE_ROOT}/worker_${shard_idx}.txt"
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
    for ((worker_idx=0; worker_idx<STAGE2_PROC_COUNT; worker_idx++)); do
      scene_file="${SHARD_SCENE_ROOT}/worker_${worker_idx}.txt"
      if [[ ! -s "${scene_file}" ]]; then
        continue
      fi
      gpu_msg="cpu/default"
      if [[ "${GPU_COUNT}" -gt 0 ]]; then
        gpu_id="${GPU_IDS[$((worker_idx % GPU_COUNT))]}"
        gpu_msg="gpu:${gpu_id}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python -m analysis.redundancy.analyze_scannet_cache \
          "${COMMON_STAGE2_ARGS[@]}" \
          --scene-ids-file "${scene_file}" \
          --no-fusion-write-aggregate &
      else
        python -m analysis.redundancy.analyze_scannet_cache \
          "${COMMON_STAGE2_ARGS[@]}" \
          --scene-ids-file "${scene_file}" \
          --no-fusion-write-aggregate &
      fi
      echo "      worker ${worker_idx}: scenes=$(wc -l < "${scene_file}") ${gpu_msg}"
      PIDS+=("$!")
      WORKER_IDS+=("${worker_idx}")
    done

    for i in "${!PIDS[@]}"; do
      if ! wait "${PIDS[i]}"; then
        echo "    [error] Stage2 worker ${WORKER_IDS[i]} failed." >&2
        exit 1
      fi
    done

    python -m analysis.redundancy.analyze_scannet_cache \
      "${COMMON_STAGE2_ARGS[@]}" \
      --fusion-aggregate-from-per-scene
    rm -rf "${SHARD_SCENE_ROOT}"
  fi
fi

echo "== Done =="
echo "    Weight report:        ${WEIGHT_ANALYSIS_DIR}/weight_report.json"
echo "    Per-scene caches:     ${CACHE_DIR}/<scene_id>/ (metadata.json, layers/layer_*.pt, pred.pt)"
echo "    Manifest:             ${CACHE_DIR}/manifest.json"
if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
  if [[ "${STAGE2_ENABLE_SIMILARITY}" == "1" || "${STAGE2_ENABLE_SIMILARITY}" == "true" ]]; then
    echo "    Similarity summary:    ${STAGE2_SIMILARITY_DIR}/summary_metrics_similarity.csv"
    echo "    Similarity scene CSV:  ${STAGE2_SIMILARITY_DIR}/per_scene_scores_similarity.csv"
    echo "    Similarity report:     ${STAGE2_SIMILARITY_DIR}/full_report_similarity.json"
  fi
  if [[ "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "1" || "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
    echo "    Skip summary:          ${STAGE2_SKIP_DIR}/summary_metrics_skip.csv"
    echo "    Skip scene CSV:        ${STAGE2_SKIP_DIR}/per_scene_scores_skip.csv"
    echo "    Skip report:           ${STAGE2_SKIP_DIR}/full_report_skip.json"
  fi
fi
if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
  if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
    if [[ "${STAGE2_ENABLE_SIMILARITY}" == "1" || "${STAGE2_ENABLE_SIMILARITY}" == "true" ]]; then
      echo "    Similarity figures:    ${STAGE2_SIMILARITY_DIR}/*.png"
    fi
    if [[ "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "1" || "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
      echo "    Skip figures:          ${STAGE2_SKIP_DIR}/*.png"
    fi
  fi
fi
if [[ "${SAVE_PER_SCENE_STAGE2}" == "1" || "${SAVE_PER_SCENE_STAGE2}" == "true" ]]; then
  if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
    if [[ "${STAGE2_ENABLE_SIMILARITY}" == "1" || "${STAGE2_ENABLE_SIMILARITY}" == "true" ]]; then
      echo "    Similarity per-scene:  ${STAGE2_SIMILARITY_DIR}/per_scene/*_similarity.json"
    fi
    if [[ "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "1" || "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
      echo "    Skip per-scene:        ${STAGE2_SKIP_DIR}/per_scene/*_skip.json"
    fi
  fi
  if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
    if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
      if [[ "${STAGE2_ENABLE_SIMILARITY}" == "1" || "${STAGE2_ENABLE_SIMILARITY}" == "true" ]]; then
        echo "    Similarity per-scene figs: ${STAGE2_SIMILARITY_DIR}/per_scene/*.png"
      fi
      if [[ "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "1" || "${STAGE2_ENABLE_SKIP_GT_EVAL}" == "true" ]]; then
        echo "    Skip per-scene figs:       ${STAGE2_SKIP_DIR}/per_scene/*.png"
      fi
    fi
  fi
fi
if [[ "${STAGE1}" == "1" || "${STAGE1}" == "true" ]]; then
  echo "    GT eval (Stage1):      ${CACHE_DIR}/eval_scannet.json"
fi

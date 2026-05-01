#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SCANNET_ROOT="${SCANNET_ROOT:-data/scans_dev50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
RUN_SCANNET="${RUN_SCANNET:-1}"
RUN_7SCENES="${RUN_7SCENES:-0}"
RUN_NRGBD="${RUN_NRGBD:-0}"

# Shared knobs
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
# Replay cache load dtype only (Stage2 autocast is fixed via --skip-loss-dtype bfloat16).
DTYPE="${DTYPE:-float32}"
GT_PLY_DIR="${GT_PLY_DIR:-}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"
MAX_SPAN="${MAX_SPAN:-24}"
MODES="${MODES:-skip_global}"
SIMILARITY_METRICS="${SIMILARITY_METRICS:-cka,cosine,neg_mse}"
TIMING="${TIMING:-1}"
HEATMAP_FILL="${HEATMAP_FILL:-nan}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-${DEVICE}}"
SAVE_PER_SCENE="${SAVE_PER_SCENE:-1}"
SAVE_HEATMAP_PNG="${SAVE_HEATMAP_PNG:-0}"
NUM_PROCS="${NUM_PROCS:-1}"
MAX_SCENES="${MAX_SCENES:-0}"
EXPORT_VIS="${EXPORT_VIS:-0}"
ENABLE_SIMILARITY="${ENABLE_SIMILARITY:-0}"
LAYER_GT_EVAL="${LAYER_GT_EVAL:-1}"
CACHE_SANITY_CHECK="${CACHE_SANITY_CHECK:-1}"
CACHE_SANITY_ATOL="${CACHE_SANITY_ATOL:-1e-6}"
CACHE_SANITY_RTOL="${CACHE_SANITY_RTOL:-1e-6}"

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local cache_dir="analysis_outputs/cache/${dataset}"
  local sim_dir="analysis_outputs/similarity/${dataset}"
  local skip_dir="analysis_outputs/skip/${dataset}"
  local vis_dir="${sim_dir}/visualization"

  echo "== redundancy dataset=${dataset} =="
  echo "   cache_dir=${cache_dir}"
  echo "   sim_dir=${sim_dir}"
  echo "   skip_dir=${skip_dir}"
  echo "   max_scenes=${MAX_SCENES}"

  extra_args=()
  if [[ -n "${GT_PLY_DIR}" ]]; then
    extra_args+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  if [[ "${LAYER_GT_EVAL}" == "1" || "${LAYER_GT_EVAL}" == "true" ]]; then
    extra_args+=(--layer-gt-eval)
  fi
  extra_args+=(--cache-load-dtype "${DTYPE}")
  if [[ "${ENABLE_SIMILARITY}" != "1" && "${ENABLE_SIMILARITY}" != "true" ]]; then
    extra_args+=(--no-fusion-compute-similarity)
  fi
  if [[ "${EXPORT_VIS}" == "1" || "${EXPORT_VIS}" == "true" ]]; then
    extra_args+=(--export-visualization --visualization-dir "${vis_dir}")
  fi
  if [[ "${SAVE_HEATMAP_PNG}" == "1" || "${SAVE_HEATMAP_PNG}" == "true" ]]; then
    extra_args+=(--save-heatmap-png)
  fi
  if [[ "${CACHE_SANITY_CHECK}" != "1" && "${CACHE_SANITY_CHECK}" != "true" ]]; then
    extra_args+=(--disable-cache-sanity-check)
  else
    extra_args+=(--cache-sanity-atol "${CACHE_SANITY_ATOL}" --cache-sanity-rtol "${CACHE_SANITY_RTOL}")
  fi
  if [[ "${TIMING}" == "1" || "${TIMING}" == "true" ]]; then
    extra_args+=(--timing)
  fi
  if ! [[ "${MAX_SCENES}" =~ ^[0-9]+$ ]]; then
    echo "    [error] MAX_SCENES must be a non-negative integer, got: ${MAX_SCENES}" >&2
    exit 1
  fi

  MODES_RAW="${MODES#[}"
  MODES_RAW="${MODES_RAW%]}"
  IFS=',' read -r -a modes_arr <<< "${MODES_RAW}"
  METRICS_RAW="${SIMILARITY_METRICS#[}"
  METRICS_RAW="${METRICS_RAW%]}"
  IFS=',' read -r -a metrics_arr <<< "${METRICS_RAW}"

  mode_args=()
  for mode in "${modes_arr[@]}"; do
    mode="${mode// /}"
    [[ -n "${mode}" ]] && mode_args+=("${mode}")
  done
  metric_args=()
  for metric in "${metrics_arr[@]}"; do
    metric="${metric// /}"
    [[ -n "${metric}" ]] && metric_args+=("${metric}")
  done

  save_args=(--fusion-save-per-scene)
  if [[ "${SAVE_PER_SCENE}" != "1" && "${SAVE_PER_SCENE}" != "true" ]]; then
    save_args=(--no-fusion-save-per-scene)
  fi

  scene_file_args=()
  if [[ "${MAX_SCENES}" -gt 0 ]]; then
    scene_limit_file="${skip_dir}/.scene_ids_max_${MAX_SCENES}.txt"
    mkdir -p "${skip_dir}"
    CACHE_DIR_ENV="${cache_dir}" MAX_SCENES_ENV="${MAX_SCENES}" SCENE_FILE_ENV="${scene_limit_file}" python - <<'PY'
import json
import os
from pathlib import Path

cache_dir = Path(os.environ["CACHE_DIR_ENV"])
max_scenes = int(os.environ["MAX_SCENES_ENV"])
scene_file = Path(os.environ["SCENE_FILE_ENV"])
manifest_path = cache_dir / "manifest.json"
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
scene_ids = sorted(
    {
        str(item.get("scene_id", "")).strip()
        for item in scenes
        if isinstance(item, dict) and str(item.get("scene_id", "")).strip()
    }
)
selected = scene_ids[:max_scenes]
scene_file.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
PY
    scene_file_args=(--scene-ids-file "${scene_limit_file}")
  fi

  python -m analysis.redundancy.analyze_dataset_cache \
    --dataset "${dataset}" \
    --cache-dir "${cache_dir}" \
    --similarity-output-dir "${sim_dir}" \
    --skip-output-dir "${skip_dir}" \
    --data-root "${data_root}" \
    --hf-model "${HF_MODEL}" \
    --device "${ANALYSIS_DEVICE}" \
    --skip-loss-device "${DEVICE}" \
    --skip-loss-dtype "bfloat16" \
    --heatmap-fill "${HEATMAP_FILL}" \
    --chamfer-max-dist "${CHAMFER_MAX_DIST}" \
    --depth-conf-thresh "${DEPTH_CONF_THRESH}" \
    --fusion-max-span "${MAX_SPAN}" \
    --fusion-num-procs "${NUM_PROCS}" \
    --fusion-modes "${mode_args[@]}" \
    --fusion-similarity-metrics "${metric_args[@]}" \
    "${scene_file_args[@]}" \
    "${save_args[@]}" \
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
echo "   Similarity: analysis_outputs/similarity/<dataset>/"
echo "   Skip:       analysis_outputs/skip/<dataset>/"

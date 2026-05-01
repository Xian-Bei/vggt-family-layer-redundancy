#!/usr/bin/env bash
# =============================================================================
#   For every scene under DATA_ROOT, run VGGT inference and save activation
#   caches; GT evaluation and visualization are done directly from in-memory
#   predictions (no per-scene prediction-file round-trip).
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

# [cache + baseline evaluation]
CACHE_DIR="${CACHE_DIR:-analysis_outputs/scannet_cache}"
MAX_SCENES="${MAX_SCENES:-50}"
MAX_FRAMES="${MAX_FRAMES:-30}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-0}"

echo "== caching global-block activations + direct GT eval =="
echo "    DATA_ROOT=${DATA_ROOT}"
echo "    CACHE_DIR=${CACHE_DIR}"

CACHE_EVAL_ARGS=()
if [[ -n "${GT_PLY_DIR}" ]]; then
  CACHE_EVAL_ARGS+=(--gt-ply-dir "${GT_PLY_DIR}")
fi
CACHE_EVAL_ARGS+=(--chamfer-max-dist "${CHAMFER_MAX_DIST}" --depth-conf-thresh "${DEPTH_CONF_THRESH}")

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

echo "== done =="
echo "    Per-scene caches: ${CACHE_DIR}/<scene_id>/ (metadata.json, layers/layer_*.pt, pred.pt)"
echo "    Manifest:         ${CACHE_DIR}/manifest.json"
echo "    GT eval:          ${CACHE_DIR}/eval_scannet.json"

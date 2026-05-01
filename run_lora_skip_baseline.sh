#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Standalone multi-interval skip+LoRA training entrypoint
# Example:
#   CACHE_DIR=analysis_outputs/scannet_cache_with_pred \
#   INTERVAL_CONFIG_FILE=analysis/redundancy/interval_config.template.json \
#   ./run_lora_skip_baseline.sh

CACHE_DIR="${CACHE_DIR:-analysis_outputs/scannet_cache}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
OUT_DIR="${OUT_DIR:-analysis_outputs/lora_skip_baseline}"
# Backend switching (match run_eval.sh semantics)
BACKEND="${BACKEND:-vggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"
# Hyperparam search config (enable one at a time, in recommended order):
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/10_lr.json}"
SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/01_loss_weight.json}"
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/03_lora_dropout.json}"
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/02_rank_alpha_tuple.json}"
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/10_align_mse_cos_tuple.json}"
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/10_depth_point_grad_tuple.json}"
# SEARCH_CONFIG_FILE="${SEARCH_CONFIG_FILE:-analysis/train/hp_search/10_lr.json}"
INTERVAL_CONFIG_FILE="${INTERVAL_CONFIG_FILE:-analysis/redundancy/interval_config.template.json}"
VARIANT="${VARIANT:-multi_interval_config}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"               # float32 | bfloat16 | float16 (default follows official training)
CONDA_ENV="${CONDA_ENV:-vggt}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR="${MIN_LR:-1e-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
GRAD_CLIP_MAX_NORM="${GRAD_CLIP_MAX_NORM:-1.0}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
RESUME_FROM="${RESUME_FROM:-}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

W_ALIGN="${W_ALIGN:-1.0}"
W_ALIGN_MSE="${W_ALIGN_MSE:-1.0}"
W_ALIGN_COS="${W_ALIGN_COS:-1.0}"
W_ALIGN_CKA="${W_ALIGN_CKA:-1.0}"
W_PRED="${W_PRED:-4.0}"
W_CAMERA="${W_CAMERA:-5.0}"
W_DEPTH="${W_DEPTH:-1.0}"
DEPTH_VALID_RANGE="${DEPTH_VALID_RANGE:-0.98}"
DEPTH_GRADIENT_LOSS_FN="${DEPTH_GRADIENT_LOSS_FN:-grad}"
CAMERA_LOSS_TYPE="${CAMERA_LOSS_TYPE:-l1}"
TARGET_SOURCE="${TARGET_SOURCE:-teacher}"   # teacher | dataset
GT_DATA_ROOT="${GT_DATA_ROOT:-data/scans_dev50}"
W_REG="${W_REG:-0.0}"
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
PRINT_STEP_TIMING="${PRINT_STEP_TIMING:-0}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-data/scans_dev50}"
TEST_DATA_DIR="${TEST_DATA_DIR:-data/scans_test50}"
TEST_GT_PLY_DIR="${TEST_GT_PLY_DIR:-data/scans_test50}"
TEST_SANITY_AVG_FILE="${TEST_SANITY_AVG_FILE:-}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"
EVAL_INPUT_FRAME="${EVAL_INPUT_FRAME:-30}"
TEST_SANITY_ATOL="${TEST_SANITY_ATOL:-5e-3}"
TEST_SANITY_RTOL="${TEST_SANITY_RTOL:-1e-2}"
BOUNDS_CACHE_FILE="${BOUNDS_CACHE_FILE:-${OUT_DIR}/bounds_summary.json}"

echo "== Train LoRA Skip Baseline =="
echo "  CACHE_DIR=${CACHE_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  INTERVAL_CONFIG_FILE=${INTERVAL_CONFIG_FILE}"
echo "  VARIANT=${VARIANT}"
echo "  EPOCHS=${EPOCHS} MAX_SCENES=${MAX_SCENES} MAX_STEPS=${MAX_STEPS}"
echo "  EVAL_EPOCH=${EVAL_EPOCH} EVAL_MAX_SCENES=${EVAL_MAX_SCENES} EVAL_INPUT_FRAME=${EVAL_INPUT_FRAME}"
echo "  SEARCH_CONFIG_FILE=${SEARCH_CONFIG_FILE:-<none>}"
echo "  CONDA_ENV=${CONDA_ENV}"
echo "  BACKEND=${BACKEND}"

# Avoid buffered/captured output so progress logs stream in real time.
export PYTHONUNBUFFERED=1

set_backend_pythonpath() {
  local backend="$1"
  if [[ "${backend}" == "fastvggt" ]]; then
    export PYTHONPATH="${FASTVGGT_VENDOR_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
  else
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  fi
}

ensure_bounds_cache() {
  local trial_out_dir="$1"
  local eval_root="${EVAL_ROOT:-analysis_outputs/evaluation}"
  local dev_set
  dev_set="$(basename "${DEV_DATA_ROOT}")"
  local test_set
  test_set="$(basename "${TEST_DATA_DIR}")"

  echo "== Bounds cache =="
  echo "  eval_root=${eval_root}"
  echo "  dev_set=${dev_set} test_set=${test_set} input_frame=${EVAL_INPUT_FRAME} backend=${BACKEND}"
  local p_upper_dev="${eval_root}/${dev_set}_input_frame_${EVAL_INPUT_FRAME}/Baseline_${BACKEND}/average_metrics.json"
  local p_lower_dev="${eval_root}/${dev_set}_input_frame_${EVAL_INPUT_FRAME}/Ours_${BACKEND}/average_metrics.json"
  local p_upper_test="${eval_root}/${test_set}_input_frame_${EVAL_INPUT_FRAME}/Baseline_${BACKEND}/average_metrics.json"
  local p_lower_test="${eval_root}/${test_set}_input_frame_${EVAL_INPUT_FRAME}/Ours_${BACKEND}/average_metrics.json"
  for p in "${p_upper_dev}" "${p_lower_dev}" "${p_upper_test}" "${p_lower_test}"; do
    if [[ ! -f "${p}" ]]; then
      echo "[error] Missing bounds file: ${p}" >&2
      exit 1
    fi
  done
}

run_one_trial() {
  local trial_out_dir="$1"
  echo "== Trial =="
  echo "  OUT_DIR=${trial_out_dir}"
  ensure_bounds_cache "${trial_out_dir}"
  set_backend_pythonpath "${BACKEND}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.train.train_lora_skip_baseline \
    --cache-dir "${CACHE_DIR}" \
    --hf-model "${HF_MODEL}" \
    --output-dir "${trial_out_dir}" \
    --interval-config-file "${INTERVAL_CONFIG_FILE}" \
    --variant "${VARIANT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --max-scenes "${MAX_SCENES}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --min-lr "${MIN_LR}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --grad-clip-max-norm "${GRAD_CLIP_MAX_NORM}" \
    --save-every-epoch "${SAVE_EVERY_EPOCH}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --w-align "${W_ALIGN}" \
    --w-align-mse "${W_ALIGN_MSE}" \
    --w-align-cos "${W_ALIGN_COS}" \
    --w-align-cka "${W_ALIGN_CKA}" \
    --w-pred "${W_PRED}" \
    --w-camera "${W_CAMERA}" \
    --w-depth "${W_DEPTH}" \
    --depth-valid-range "${DEPTH_VALID_RANGE}" \
    --depth-gradient-loss-fn "${DEPTH_GRADIENT_LOSS_FN}" \
    --camera-loss-type "${CAMERA_LOSS_TYPE}" \
    --target-source "${TARGET_SOURCE}" \
    --gt-data-root "${GT_DATA_ROOT}" \
    --w-reg "${W_REG}" \
    --eval-epoch "${EVAL_EPOCH}" \
    --dev-data-root "${DEV_DATA_ROOT}" \
    --test-data-dir "${TEST_DATA_DIR}" \
    --test-gt-ply-dir "${TEST_GT_PLY_DIR}" \
    --bounds-cache-file "${BOUNDS_CACHE_FILE}" \
    $( [[ -n "${TEST_SANITY_AVG_FILE}" ]] && printf '%s %s' --test-sanity-average-metrics-file "${TEST_SANITY_AVG_FILE}" ) \
    --eval-max-scenes "${EVAL_MAX_SCENES}" \
    --eval-input-frame "${EVAL_INPUT_FRAME}" \
    --test-sanity-atol "${TEST_SANITY_ATOL}" \
    --test-sanity-rtol "${TEST_SANITY_RTOL}" \
    $( [[ -n "${RESUME_FROM}" ]] && printf '%s %s' --resume-from "${RESUME_FROM}" ) \
    $( [[ "${USE_CHECKPOINT}" == "1" || "${USE_CHECKPOINT}" == "true" ]] && echo "--use-checkpoint" || echo "--no-use-checkpoint" ) \
    $( [[ "${PRINT_STEP_TIMING}" == "1" || "${PRINT_STEP_TIMING}" == "true" ]] && echo "--print-step-timing" )
}

if [[ -z "${SEARCH_CONFIG_FILE}" ]]; then
  run_one_trial "${OUT_DIR}"
  exit 0
fi

if [[ ! -f "${SEARCH_CONFIG_FILE}" ]]; then
  echo "[error] SEARCH_CONFIG_FILE not found: ${SEARCH_CONFIG_FILE}" >&2
  exit 1
fi

mapfile -t SEARCH_ROWS < <(python - "${SEARCH_CONFIG_FILE}" <<'PY'
import json
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding="utf-8"))
if not isinstance(obj, dict) or len(obj) != 1:
    raise SystemExit("SEARCH_CONFIG_FILE must be a dict with exactly one key.")
(raw_key, values), = obj.items()
if not isinstance(values, list) or not values:
    raise SystemExit("Search values must be a non-empty list.")

def sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-zA-Z0-9._=+\-]", "_", s)
    return s

if raw_key.startswith("(") and raw_key.endswith(")"):
    keys = [x.strip() for x in raw_key[1:-1].split(",") if x.strip()]
    if len(keys) < 2:
        raise SystemExit(f"Invalid tuple key: {raw_key}")
    for i, v in enumerate(values):
        if not isinstance(v, (list, tuple)) or len(v) != len(keys):
            raise SystemExit(f"Tuple value at idx={i} must have len={len(keys)}.")
        vals = [str(x) for x in v]
        label = sanitize(f"{raw_key}={tuple(vals)}")
        assignments = ";".join([f"{k}={vv}" for k, vv in zip(keys, vals)])
        print("\t".join([label, assignments]))
else:
    key = raw_key.strip()
    for v in values:
        v_str = str(v)
        label = sanitize(f"{key}={v_str}")
        print("\t".join([label, f"{key}={v_str}"]))
PY
)

if [[ "${#SEARCH_ROWS[@]}" -eq 0 ]]; then
  echo "[error] No trials parsed from SEARCH_CONFIG_FILE=${SEARCH_CONFIG_FILE}" >&2
  exit 1
fi

for row in "${SEARCH_ROWS[@]}"; do
  IFS=$'\t' read -r TRIAL_LABEL ASSIGNMENTS <<< "${row}"
  if [[ -z "${TRIAL_LABEL}" || -z "${ASSIGNMENTS}" ]]; then
    echo "[error] Invalid parsed trial row: ${row}" >&2
    exit 1
  fi
  IFS=';' read -r -a KV_ARR <<< "${ASSIGNMENTS}"
  for kv in "${KV_ARR[@]}"; do
    [[ -z "${kv}" ]] && continue
    export "${kv}"
  done
  run_one_trial "${OUT_DIR}/${TRIAL_LABEL}"
done


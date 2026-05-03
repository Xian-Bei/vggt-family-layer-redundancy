#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Standalone multi-interval skip+LoRA training entrypoint
# Loads CONFIG_FILE (JSON): "intervals" for model structure (see train_lora_skip_baseline),
# "name" for the save subdirectory under OUT_DIR, "params" optional flat dict to override
# training env defaults (keys omitted in JSON keep shell defaults above).
# Example (single config, default):
#   CACHE_DIR=analysis_outputs/scannet_cache_with_pred \
#   CONFIG_FILE=analysis/train/config_90.json \
#   ./run_lora_skip_baseline.sh
#
# Multiple configs (positional args take precedence):
#   ./run_lora_skip_baseline.sh analysis/train/config_85.json analysis/train/config_90.json
# Or space-separated env (paths must not contain spaces):
#   CONFIG_FILES="analysis/train/config_85.json analysis/train/config_90.json" ./run_lora_skip_baseline.sh
# Same for CONFIG_FILE: multiple paths are split on whitespace (one path must not contain spaces).

CACHE_DIR="${CACHE_DIR:-analysis_outputs/cache/scannet}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
OUT_DIR="${OUT_DIR:-analysis_outputs/lora_skip}"
# Backend switching (match run_eval.sh semantics)
BACKEND="${BACKEND:-vggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"
CONFIG_FILE="${CONFIG_FILE:-analysis/train/config_90_vggt.json analysis/train/config_90_fastvggt.json}"
VARIANT="${VARIANT:-multi_interval_config}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-10}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-50}"
# Per-dataset periodic-eval toggles (1/true = enabled, 0/false = disabled).
EVAL_SCANNET_DEV50="${EVAL_SCANNET_DEV50:-1}"
EVAL_SCANNET_TEST50="${EVAL_SCANNET_TEST50:-1}"
EVAL_7SCENES="${EVAL_7SCENES:-1}"
EVAL_NRGBD="${EVAL_NRGBD:-0}"


W_CAMERA="${W_CAMERA:-5.0}"
W_DEPTH="${W_DEPTH:-1.0}"
DEPTH_VALID_RANGE="${DEPTH_VALID_RANGE:-0.98}"
DEPTH_GRADIENT_LOSS_FN="${DEPTH_GRADIENT_LOSS_FN:-grad}"
CAMERA_LOSS_TYPE="${CAMERA_LOSS_TYPE:-l1}"
TARGET_SOURCE="${TARGET_SOURCE:-teacher}"   # teacher | dataset
GT_DATA_ROOT="${GT_DATA_ROOT:-data/scans_dev50}"
W_REG="${W_REG:-0.0}"
USE_CHECKPOINT="${USE_CHECKPOINT:-1}"
RESUME_FROM="${RESUME_FROM:-}"
PRINT_STEP_TIMING="${PRINT_STEP_TIMING:-0}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR="${MIN_LR:-1e-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
GRAD_CLIP_MAX_NORM="${GRAD_CLIP_MAX_NORM:-1.0}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
W_ALIGN="${W_ALIGN:-1.0}"
W_ALIGN_CKA="${W_ALIGN_CKA:-1.0}"
W_ALIGN_MSE="${W_ALIGN_MSE:-0.0}"
W_ALIGN_COS="${W_ALIGN_COS:-1.0}"
W_PRED="${W_PRED:-1.0}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-data/scans_dev50}"
TEST_DATA_DIR="${TEST_DATA_DIR:-data/scans_test50}"
TEST_GT_PLY_DIR="${TEST_GT_PLY_DIR:-data/scans_test50}"
TEST_SANITY_AVG_FILE="${TEST_SANITY_AVG_FILE:-}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"
EVAL_INPUT_FRAME="${EVAL_INPUT_FRAME:-50}"
TEST_SANITY_ATOL="${TEST_SANITY_ATOL:-5e-3}"
TEST_SANITY_RTOL="${TEST_SANITY_RTOL:-5e-3}"
# 7scenes / NRGBD periodic eval (defaults match run_eval.sh KF=10).
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
EVAL_7ANDN_KF="${EVAL_7ANDN_KF:-10}"
EVAL_7ANDN_MAX_SCENES_PER_DATASET="${EVAL_7ANDN_MAX_SCENES_PER_DATASET:-0}"

# Resolve config list: CLI args > CONFIG_FILES > CONFIG_FILE
declare -a CONFIG_PATHS=()
if [[ "$#" -gt 0 ]]; then
  CONFIG_PATHS=("$@")
elif [[ -n "${CONFIG_FILES:-}" ]]; then
  read -r -a CONFIG_PATHS <<< "${CONFIG_FILES}"
else
  read -r -a CONFIG_PATHS <<< "${CONFIG_FILE}"
fi

if [[ "${#CONFIG_PATHS[@]}" -eq 0 ]]; then
  echo "[error] No config files: pass paths as arguments, set CONFIG_FILES, or set CONFIG_FILE." >&2
  exit 1
fi

for cfg in "${CONFIG_PATHS[@]}"; do
  if [[ -z "${cfg}" ]]; then
    echo "[error] Empty entry in config list." >&2
    exit 1
  fi
  if [[ ! -f "${cfg}" ]]; then
    echo "[error] Config not found: ${cfg}" >&2
    exit 1
  fi
done

# Apply CONFIG_FILE: TRIAL_OUT_DIR = OUT_DIR / name; export params keys over defaults.
apply_config_and_run_trial() {
  local CONFIG_FILE="$1"

eval "$(python - "${CONFIG_FILE}" "${OUT_DIR}" <<'PY'
import json
import re
import shlex
import sys
from pathlib import Path

_cfg_path = Path(sys.argv[1])
_out_parent = Path(sys.argv[2])
obj = json.loads(_cfg_path.read_text(encoding="utf-8"))
if not isinstance(obj, dict):
    raise SystemExit("CONFIG_FILE must be a JSON object.")

intervals = obj.get("intervals")
if intervals is None:
    raise SystemExit('CONFIG_FILE must contain a non-null "intervals" array.')
if not isinstance(intervals, list) or not intervals:
    raise SystemExit('"intervals" must be a non-empty list.')

name = obj.get("name")
if name is None or (isinstance(name, str) and not name.strip()):
    name = _cfg_path.stem
else:
    name = str(name).strip()

_env_key = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
params = obj.get("params")
if params is None:
    params = {}
elif not isinstance(params, dict):
    raise SystemExit('"params" must be a JSON object if present.')

lines = [f"export TRIAL_OUT_DIR={shlex.quote(str(_out_parent / name))}"]
for k, v in params.items():
    if not isinstance(k, str) or not _env_key.match(k):
        raise SystemExit(f"Invalid params key (use env-style names like LR): {k!r}")
    if isinstance(v, bool):
        val = "1" if v else "0"
    elif v is None:
        continue
    else:
        val = str(v)
    lines.append(f"export {k}={shlex.quote(val)}")

print("\n".join(lines))
PY
)"

echo "== Train LoRA Skip Baseline =="
echo "  CACHE_DIR=${CACHE_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  TRIAL_OUT_DIR=${TRIAL_OUT_DIR}"
echo "  CONFIG_FILE=${CONFIG_FILE}"
echo "  VARIANT=${VARIANT}"
echo "  EPOCHS=${EPOCHS} MAX_SCENES=${MAX_SCENES} MAX_STEPS=${MAX_STEPS}"
echo "  EVAL_EPOCH=${EVAL_EPOCH} EVAL_MAX_SCENES=${EVAL_MAX_SCENES} EVAL_INPUT_FRAME=${EVAL_INPUT_FRAME}"
echo "  EVAL_SCANNET_DEV50=${EVAL_SCANNET_DEV50} EVAL_SCANNET_TEST50=${EVAL_SCANNET_TEST50} EVAL_7SCENES=${EVAL_7SCENES} EVAL_NRGBD=${EVAL_NRGBD}"
echo "  ROOT_7SCENES=${ROOT_7SCENES} ROOT_NRGBD=${ROOT_NRGBD}"
echo "  EVAL_7ANDN_KF=${EVAL_7ANDN_KF} EVAL_7ANDN_MAX_SCENES_PER_DATASET=${EVAL_7ANDN_MAX_SCENES_PER_DATASET}"
echo "  BACKEND=${BACKEND}"

run_one_trial "${TRIAL_OUT_DIR}"
}

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
  echo "  7andn kf=${EVAL_7ANDN_KF}"
  # ScanNet bounds (4 files: dev/test * Baseline/Ours).
  local p_upper_dev="${eval_root}/${dev_set}_input_frame_${EVAL_INPUT_FRAME}/Baseline_${BACKEND}/average_metrics.json"
  local p_lower_dev="${eval_root}/${dev_set}_input_frame_${EVAL_INPUT_FRAME}/Ours_${BACKEND}/average_metrics.json"
  local p_upper_test="${eval_root}/${test_set}_input_frame_${EVAL_INPUT_FRAME}/Baseline_${BACKEND}/average_metrics.json"
  local p_lower_test="${eval_root}/${test_set}_input_frame_${EVAL_INPUT_FRAME}/Ours_${BACKEND}/average_metrics.json"
  # 7Scenes / NRGBD bounds (4 files: 7scenes/NRGBD * Baseline/Ours).
  local p_upper_7s="${eval_root}/7andn/Baseline_${BACKEND}/${EVAL_7ANDN_KF}/7scenes/average_metrics.json"
  local p_lower_7s="${eval_root}/7andn/Ours_${BACKEND}/${EVAL_7ANDN_KF}/7scenes/average_metrics.json"
  local p_upper_nr="${eval_root}/7andn/Baseline_${BACKEND}/${EVAL_7ANDN_KF}/NRGBD/average_metrics.json"
  local p_lower_nr="${eval_root}/7andn/Ours_${BACKEND}/${EVAL_7ANDN_KF}/NRGBD/average_metrics.json"
  local -a required_bounds=()
  if [[ "${EVAL_SCANNET_DEV50}" == "1" || "${EVAL_SCANNET_DEV50}" == "true" ]]; then
    required_bounds+=("${p_upper_dev}" "${p_lower_dev}")
  fi
  if [[ "${EVAL_SCANNET_TEST50}" == "1" || "${EVAL_SCANNET_TEST50}" == "true" ]]; then
    required_bounds+=("${p_upper_test}" "${p_lower_test}")
  fi
  if [[ "${EVAL_7SCENES}" == "1" || "${EVAL_7SCENES}" == "true" ]]; then
    required_bounds+=("${p_upper_7s}" "${p_lower_7s}")
  fi
  if [[ "${EVAL_NRGBD}" == "1" || "${EVAL_NRGBD}" == "true" ]]; then
    required_bounds+=("${p_upper_nr}" "${p_lower_nr}")
  fi
  for p in "${required_bounds[@]}"; do
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
  conda run --no-capture-output -n vggt python -m analysis.train.train_lora_skip_baseline \
    --cache-dir "${CACHE_DIR}" \
    --hf-model "${HF_MODEL}" \
    --output-dir "${trial_out_dir}" \
    --interval-config-file "${CONFIG_FILE}" \
    --variant "${VARIANT}" \
    --device "${DEVICE}" \
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
    $( [[ -n "${TEST_SANITY_AVG_FILE}" ]] && printf '%s %s' --test-sanity-average-metrics-file "${TEST_SANITY_AVG_FILE}" ) \
    --eval-max-scenes "${EVAL_MAX_SCENES}" \
    --eval-input-frame "${EVAL_INPUT_FRAME}" \
    $( [[ "${EVAL_SCANNET_DEV50}" == "1" || "${EVAL_SCANNET_DEV50}" == "true" ]] && echo "--eval-scannet-dev50" || echo "--no-eval-scannet-dev50" ) \
    $( [[ "${EVAL_SCANNET_TEST50}" == "1" || "${EVAL_SCANNET_TEST50}" == "true" ]] && echo "--eval-scannet-test50" || echo "--no-eval-scannet-test50" ) \
    $( [[ "${EVAL_7SCENES}" == "1" || "${EVAL_7SCENES}" == "true" ]] && echo "--eval-7scenes" || echo "--no-eval-7scenes" ) \
    $( [[ "${EVAL_NRGBD}" == "1" || "${EVAL_NRGBD}" == "true" ]] && echo "--eval-nrgbd" || echo "--no-eval-nrgbd" ) \
    --root-7scenes "${ROOT_7SCENES}" \
    --root-nrgbd "${ROOT_NRGBD}" \
    --eval-7andn-kf "${EVAL_7ANDN_KF}" \
    --eval-7andn-max-scenes-per-dataset "${EVAL_7ANDN_MAX_SCENES_PER_DATASET}" \
    --test-sanity-atol "${TEST_SANITY_ATOL}" \
    --test-sanity-rtol "${TEST_SANITY_RTOL}" \
    $( [[ -n "${RESUME_FROM}" ]] && printf '%s %s' --resume-from "${RESUME_FROM}" ) \
    $( [[ "${USE_CHECKPOINT}" == "1" || "${USE_CHECKPOINT}" == "true" ]] && echo "--use-checkpoint" || echo "--no-use-checkpoint" ) \
    $( [[ "${PRINT_STEP_TIMING}" == "1" || "${PRINT_STEP_TIMING}" == "true" ]] && echo "--print-step-timing" )
}

total_configs="${#CONFIG_PATHS[@]}"
idx=0
for CONFIG_FILE in "${CONFIG_PATHS[@]}"; do
  idx=$((idx + 1))
  echo ""
  echo "======== Config ${idx}/${total_configs}: ${CONFIG_FILE} ========"
  # Subshell so JSON "params" exports do not leak into the next config iteration.
  ( apply_config_and_run_trial "${CONFIG_FILE}" )
done

echo ""
echo "== All ${total_configs} config(s) finished =="


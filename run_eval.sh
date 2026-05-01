#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# FastVGGT-aligned CLI knobs
DATA_DIR="${DATA_DIR:-data/scans_test50,data/scans_dev50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_outputs/evaluation}"
CKPT_PATH="${CKPT_PATH:-}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
INTERVAL_CONFIG_FILE="${INTERVAL_CONFIG_FILE:-analysis/redundancy/interval_config.template.json}"
DEVICE="${DEVICE:-cuda:0}"
INPUT_FRAME="${INPUT_FRAME:-30}"
NUM_SCENES="${NUM_SCENES:-0}"
KF="${KF:-10}"
SIZE="${SIZE:-518}"
MERGING="${MERGING:-0}"
MERGE_RATIO="${MERGE_RATIO:-0.9}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
# Set to 1 to pass --skip_depth_eval (no depth_abs_rel / depth_delta1 vs ScanNet GT).
SKIP_DEPTH_EVAL="${SKIP_DEPTH_EVAL:-0}"
# Export per-scene visualization assets for the Gradio+Viser panel.
EXPORT_VIS="${EXPORT_VIS:-1}"
VIS_MAX_POINTS="${VIS_MAX_POINTS:-0}"
BASELINE_BACKEND="${BASELINE_BACKEND:-vggt}"
FAST_BACKEND="${FAST_BACKEND:-fastvggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"
SCANNET_MODELS="${SCANNET_MODELS:-Baseline(fastvggt),Ours(vggt),Ours(fastvggt)}"
MODELS_7ANDN="${MODELS_7ANDN:-Baseline(fastvggt)}"
# Tag used for ScanNet-style output directories. Defaults to basename of DATA_DIR
# (e.g. scans_dev50 / scans_test50), but you can override explicitly.
SCANNET_TAG="${SCANNET_TAG:-$(basename "${DATA_DIR}")}"
# Ours checkpoint resolution (used when model spec is Ours(<backend>))
OURS_TRAIN_DIR="${OURS_TRAIN_DIR:-}"
OURS_EPOCH="${OURS_EPOCH:-}"
RUN_SCANNET="${RUN_SCANNET:-1}"
RUN_7ANDN="${RUN_7ANDN:-1}"

trim_spaces() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

parse_model_spec() {
  local spec
  spec="$(trim_spaces "$1")"
  # Accept optional ckpt suffix, e.g. Ours(vggt)ckpt
  if [[ "${spec}" =~ ^([^()]+)\(([^()]+)\)(ckpt)?$ ]]; then
    MODEL_NAME_PARSED="$(trim_spaces "${BASH_REMATCH[1]}")"
    MODEL_BACKEND_PARSED="$(trim_spaces "${BASH_REMATCH[2]}")"
    MODEL_CKPT_PARSED="$(trim_spaces "${BASH_REMATCH[3]:-}")" # "" or "ckpt"
    if [[ -z "${MODEL_NAME_PARSED}" || -z "${MODEL_BACKEND_PARSED}" ]]; then
      echo "[error] Invalid model spec: '${spec}'" >&2
      exit 1
    fi
  else
    echo "[error] Invalid model spec: '${spec}', expected <model>(<backend>)[ckpt]" >&2
    exit 1
  fi
}

set_backend_pythonpath() {
  local backend="$1"
  if [[ "${backend}" == "fastvggt" ]]; then
    export PYTHONPATH="${FASTVGGT_VENDOR_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
  else
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  fi
}

run_scannet() {
  local model_name="$1"
  local backend="$2"
  local ckpt_flag="${3:-}" # "" or "ckpt"
  local model_dir="${model_name}_${backend}"
  if [[ "${ckpt_flag}" == "ckpt" ]]; then
    model_dir="${model_dir}_ckpt"
  fi
  local out_dir="${OUTPUT_DIR}/${SCANNET_TAG}_input_frame_${INPUT_FRAME}/${model_dir}"
  local num_scenes_arg="${NUM_SCENES}"
  local input_frame_arg="${INPUT_FRAME}"

  args=(
    --data_dir "${DATA_DIR}"
    --output_path "${out_dir}"
    --input_frame "${input_frame_arg}"
    --depth_conf_thresh "${DEPTH_CONF_THRESH}"
    --chamfer_max_dist "${CHAMFER_MAX_DIST}"
    --model-name "${model_name}"
    --backend "${backend}"
  )
  # Always pass num_scenes so eval_scannet can interpret 0 as \"all scenes\".
  args+=(--num_scenes "${num_scenes_arg}")
  # Checkpoint routing:
  # - If CKPT_PATH is set globally, evaluate that checkpoint (legacy mode).
  # - Else if evaluating Ours(<backend>)ckpt, resolve ckpt from OURS_TRAIN_DIR/OURS_EPOCH (required).
  # - Else if evaluating Ours(<backend>) (no ckpt suffix), evaluate lowerbound (skip-only; requires INTERVAL_CONFIG_FILE).
  if [[ -n "${CKPT_PATH}" ]]; then
    args+=(--ckpt_path "${CKPT_PATH}" --hf_model "${HF_MODEL}")
  elif [[ "${model_name}" == "Ours" && "${ckpt_flag}" == "ckpt" ]]; then
    if [[ -z "${OURS_TRAIN_DIR}" || -z "${OURS_EPOCH}" ]]; then
      echo "[error] ${model_name}(${backend})ckpt requires OURS_TRAIN_DIR and OURS_EPOCH" >&2
      exit 1
    fi
    args+=(--hf_model "${HF_MODEL}" --train-dir "${OURS_TRAIN_DIR}" --epoch "${OURS_EPOCH}")
  elif [[ "${model_name}" == "Ours" ]]; then
    args+=(--hf_model "${HF_MODEL}" --interval-config-file "${INTERVAL_CONFIG_FILE}")
  else
    args+=(--hf_model "${HF_MODEL}")
  fi
  if [[ "${backend}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi
  if [[ "${SKIP_DEPTH_EVAL}" == "1" || "${SKIP_DEPTH_EVAL}" == "true" ]]; then
    args+=(--skip_depth_eval)
  fi
  if [[ "${EXPORT_VIS}" == "1" || "${EXPORT_VIS}" == "true" ]]; then
    args+=(--export-visualization --vis-max-points "${VIS_MAX_POINTS}")
  fi

  set_backend_pythonpath "${backend}"
  rm -rf "${out_dir}"
  echo "   [scannet/${model_name}] backend=${backend} ckpt=${ckpt_flag:-none}"
  python "analysis/eval/eval_scannet.py" "${args[@]}"
}

run_7andn() {
  local model_name="$1"
  local backend="$2"
  local ckpt_flag="${3:-}" # "" or "ckpt"
  local model_dir="${model_name}_${backend}"
  if [[ "${ckpt_flag}" == "ckpt" ]]; then
    model_dir="${model_dir}_ckpt"
  fi
  local out_dir="${OUTPUT_DIR}/7andn/${model_dir}"
  local max_scenes=0
  local max_samples=0
  local size_arg="${SIZE}"
  local kf_arg="${KF}"

  args=(
    --device "${DEVICE}"
    --output_dir "${out_dir}"
    --size "${size_arg}"
    --kf "${kf_arg}"
    --root_7scenes "${ROOT_7SCENES}"
    --root_nrgbd "${ROOT_NRGBD}"
    --max_scenes_per_dataset "${max_scenes}"
    --max_samples "${max_samples}"
    --model-name "${model_name}"
    --backend "${backend}"
  )
  if [[ -n "${CKPT_PATH}" ]]; then
    args+=(--ckpt_path "${CKPT_PATH}" --hf_model "${HF_MODEL}")
  elif [[ "${model_name}" == "Ours" && "${ckpt_flag}" == "ckpt" ]]; then
    if [[ -z "${OURS_TRAIN_DIR}" || -z "${OURS_EPOCH}" ]]; then
      echo "[error] ${model_name}(${backend})ckpt requires OURS_TRAIN_DIR and OURS_EPOCH" >&2
      exit 1
    fi
    args+=(--hf_model "${HF_MODEL}" --train-dir "${OURS_TRAIN_DIR}" --epoch "${OURS_EPOCH}")
  elif [[ "${model_name}" == "Ours" ]]; then
    args+=(--hf_model "${HF_MODEL}" --interval-config-file "${INTERVAL_CONFIG_FILE}")
  else
    args+=(--hf_model "${HF_MODEL}")
  fi
  if [[ "${backend}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi
  if [[ "${EXPORT_VIS}" == "1" || "${EXPORT_VIS}" == "true" ]]; then
    args+=(--export-visualization --vis-max-points "${VIS_MAX_POINTS}")
  fi

  set_backend_pythonpath "${backend}"
  rm -rf "${out_dir}"
  echo "   [7andn/${model_name}] backend=${backend} ckpt=${ckpt_flag:-none}"
  python "analysis/eval/eval_7andN.py" "${args[@]}"
}

echo "== FastVGGT-aligned evaluation =="
echo "   output: ${OUTPUT_DIR}"
echo "   skip_depth_eval (ScanNet depth vs GT): ${SKIP_DEPTH_EVAL}"
echo "   scannet_models: ${SCANNET_MODELS}"
echo "   models_7andn: ${MODELS_7ANDN}"
echo "   run_scannet: ${RUN_SCANNET}"
echo "   run_7andn: ${RUN_7ANDN}"

if [[ "${RUN_SCANNET}" == "1" || "${RUN_SCANNET}" == "true" ]]; then
  IFS=',' read -r -a DATA_DIR_ARR <<< "${DATA_DIR}"
  IFS=',' read -r -a SCANNET_MODEL_ARR <<< "${SCANNET_MODELS}"
  for dd in "${DATA_DIR_ARR[@]}"; do
    dd="$(trim_spaces "${dd}")"
    [[ -z "${dd}" ]] && continue
    DATA_DIR="${dd}"
    SCANNET_TAG="$(basename "${DATA_DIR}")"
    for spec in "${SCANNET_MODEL_ARR[@]}"; do
      parse_model_spec "${spec}"
      run_scannet "${MODEL_NAME_PARSED}" "${MODEL_BACKEND_PARSED}" "${MODEL_CKPT_PARSED}"
    done
  done
else
  echo "   [skip] scannet evaluation disabled"
fi

if [[ "${RUN_7ANDN}" == "1" || "${RUN_7ANDN}" == "true" ]]; then
  IFS=',' read -r -a MODEL_7ANDN_ARR <<< "${MODELS_7ANDN}"
  for spec in "${MODEL_7ANDN_ARR[@]}"; do
    parse_model_spec "${spec}"
    run_7andn "${MODEL_NAME_PARSED}" "${MODEL_BACKEND_PARSED}" "${MODEL_CKPT_PARSED}"
  done
else
  echo "   [skip] 7scenes + NRGBD evaluation disabled"
fi

python -m analysis.eval.summary --output-dir "${OUTPUT_DIR}"

echo "== Done =="
echo "   ScanNet outputs:  ${OUTPUT_DIR}/<scannet_tag>_input_frame_${INPUT_FRAME}/<model>_<backend>[_ckpt]"
echo "   7N outputs:       ${OUTPUT_DIR}/7andn/<model>_<backend>[_ckpt]"


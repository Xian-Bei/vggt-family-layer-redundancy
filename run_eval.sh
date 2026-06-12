#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BACKBONES="${BACKBONES:-vggt}"
RUN_SCANNET_DEV50="${RUN_SCANNET_DEV50:-0}"
RUN_SCANNET_TEST50="${RUN_SCANNET_TEST50:-0}"
RUN_7SCENES="${RUN_7SCENES:-1}"
RUN_NRGBD="${RUN_NRGBD:-0}"
SPLIT_7SCENES="${SPLIT_7SCENES:-test}"

EXPORT_VIS_2D="${EXPORT_VIS_2D:-1}"
EXPORT_VIS_3D="${EXPORT_VIS_3D:-0}"

PRUNE="${PRUNE:-0,1}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
PLAN_NAME="${PLAN_NAME:-}"
MAX_SCENES="${MAX_SCENES:-0}"

#############################################

ROOT_SCANNET_DEV50="${ROOT_SCANNET_DEV50:-data/scans_dev50}"
ROOT_SCANNET_TEST50="${ROOT_SCANNET_TEST50:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
CHECKPOINT="${CHECKPOINT:-}"
HF_MODEL="${HF_MODEL:-}"

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

PRUNE_N=0
parse_prune_list() {
  local raw="${PRUNE//,/ }"
  read -r -a PRUNE_LIST <<< "${raw}"
  PRUNE_N=0
  local p
  for p in "${PRUNE_LIST[@]}"; do
    p="${p//[[:space:]]/}"
    [[ -z "${p}" ]] && continue
    if [[ "${p}" != "0" && "${p}" != "1" ]]; then
      echo "error: PRUNE entries must be 0 or 1, got: ${p}" >&2
      exit 1
    fi
    PRUNE_N=$((PRUNE_N + 1))
  done
  if [[ "${PRUNE_N}" -eq 0 ]]; then
    echo "error: PRUNE is empty" >&2
    exit 1
  fi
}
parse_prune_list

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

validate_split_7scenes() {
  if [[ "${SPLIT_7SCENES}" != "train" && "${SPLIT_7SCENES}" != "test" ]]; then
    echo "error: SPLIT_7SCENES must be train or test, got: ${SPLIT_7SCENES}" >&2
    exit 1
  fi
}
validate_split_7scenes

# upper: no prune; lower: prune only; model: prune + finetuned checkpoint
resolve_eval_mode() {
  local prune_flag="$1"
  if truthy "${prune_flag}"; then
    if [[ -n "${CHECKPOINT}" ]]; then
      echo "model"
    else
      echo "lower"
    fi
  else
    echo "upper"
  fi
}

checkpoint_for_model_mode() {
  local backbone="$1"
  local scannet_tag="${2:-scannet_dev50}"
  if [[ -n "${CHECKPOINT}" ]]; then
    echo "${CHECKPOINT}"
    return
  fi
  echo "${REPO_ROOT}/analysis_outputs_v3/${backbone}/${scannet_tag}/train/checkpoints/latest.pt"
}

run_eval_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local mode="$4"
  local model_name="$5"
  local output_dataset="${6:-${dataset}}"
  local plan_name="${7:-}"
  set_backend_pythonpath "${backbone}"

  local out_dir="${REPO_ROOT}/analysis_outputs_v3/${backbone}/${output_dataset}/eval/${model_name}"
  if [[ -n "${plan_name}" && "${mode}" == "lower" ]]; then
    out_dir="${out_dir}/${plan_name}"
  fi
  local extra=(
    --eval-mode "${mode}"
    --prune-plan-root "${PRUNE_PLAN_ROOT}"
    --plan-name "${plan_name}"
    --model-name "${model_name}"
    --output-dir "${out_dir}"
  )
  if [[ -n "${HF_MODEL}" ]]; then
    extra+=(--hf-model "${HF_MODEL}")
  fi
  extra+=(--max-frames "${MAX_FRAMES}")
  if [[ "${dataset}" == "7scenes" ]]; then
    extra+=(--split "${SPLIT_7SCENES}")
  fi

  if truthy "${EXPORT_VIS:-0}"; then
    EXPORT_VIS_2D=1
    EXPORT_VIS_3D=1
  fi
  if truthy "${EXPORT_VIS_2D}"; then
    extra+=(--export-vis-2d)
  fi
  if truthy "${EXPORT_VIS_3D}"; then
    extra+=(--export-vis-3d)
  fi
  if [[ "${VIS_MAX_POINTS:-0}" -gt 0 ]]; then
    extra+=(--vis-max-points "${VIS_MAX_POINTS}")
  fi

  case "${mode}" in
    upper)
      if [[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}" ]]; then
        extra+=(--checkpoint "${CHECKPOINT}")
      elif [[ -n "${CHECKPOINT}" ]]; then
        echo "warning: CHECKPOINT set but missing for upper mode: ${CHECKPOINT}" >&2
        extra+=(--checkpoint "${CHECKPOINT}")
      fi
      ;;
    lower)
      ;;
    model)
      local ckpt
      local ckpt_tag="${output_dataset}"
      if [[ "${dataset}" == "scannet" ]]; then
        ckpt_tag="scannet_dev50"
      fi
      ckpt="$(checkpoint_for_model_mode "${backbone}" "${ckpt_tag}")"
      if [[ -f "${ckpt}" ]]; then
        extra+=(--checkpoint "${ckpt}")
      else
        echo "error: model mode requires checkpoint (set CHECKPOINT or train first): ${ckpt}" >&2
        return 1
      fi
      ;;
  esac

  echo "   [eval/${output_dataset}] ${model_name}/${backbone} mode=${mode}"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.eval_v3 \
    --experiment "${backbone}" \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --backbone "${backbone}" \
    --device "${DEVICE}" \
    --max-scenes "${MAX_SCENES}" \
    "${extra[@]}"
}

run_datasets_for_backbone() {
  local backbone="$1"
  local prune_flag="$2"
  local plan_name="${3:-}"
  local mode
  mode="$(resolve_eval_mode "${prune_flag}")"
  local model_name="${MODEL_NAME:-${mode}}"

  echo "   [eval] ${backbone} PRUNE=${prune_flag} -> ${mode}/${model_name}${plan_name:+ plan=${plan_name}}"
  if truthy "${prune_flag}"; then
    if [[ -n "${plan_name}" ]]; then
      echo "   [prune] ${PRUNE_PLAN_ROOT}/${plan_name}.json"
    else
      echo "   [prune] ${PRUNE_PLAN_ROOT}/${backbone}.json"
    fi
  fi

  if truthy "${RUN_SCANNET_DEV50}"; then
    run_eval_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "${mode}" "${model_name}" "scannet_dev50" "${plan_name}"
  fi
  if truthy "${RUN_SCANNET_TEST50}"; then
    run_eval_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "${mode}" "${model_name}" "scannet_test50" "${plan_name}"
  fi
  if [[ "${RUN_7SCENES}" == "1" || "${RUN_7SCENES}" == "true" ]]; then
    run_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${mode}" "${model_name}" "7scenes_${SPLIT_7SCENES}" "${plan_name}"
  fi
  if [[ "${RUN_NRGBD}" == "1" || "${RUN_NRGBD}" == "true" ]]; then
    run_eval_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "${mode}" "${model_name}" "nrgbd" "${plan_name}"
  fi
}

run_eval_for_prune_variants() {
  local bb="$1"
  if [[ -n "${PLAN_NAME}" ]]; then
    run_datasets_for_backbone "${bb}" "1" "${PLAN_NAME}"
    return
  fi
  local root="${PRUNE_PLAN_ROOT}"
  if [[ "${root}" == "${REPO_ROOT}/analysis/prune_plan" || "${root}" == "analysis/prune_plan" ]]; then
    run_datasets_for_backbone "${bb}" "1" ""
    return
  fi
  if [[ ! -d "${root}" ]]; then
    echo "error: PRUNE_PLAN_ROOT not found: ${root}" >&2
    exit 1
  fi
  local plans=()
  local f
  for f in "${root}"/plan_t*_l*.json; do
    [[ -e "${f}" ]] || continue
    plans+=("$(basename "${f}" .json)")
  done
  IFS=$'\n' plans=($(printf '%s\n' "${plans[@]}" | sort)); unset IFS
  if [[ "${#plans[@]}" -eq 0 ]]; then
    echo "error: no plan files found under ${root} (expected plan_t*_l*.json)" >&2
    exit 1
  fi
  local p
  for p in "${plans[@]}"; do
    run_datasets_for_backbone "${bb}" "1" "${p}"
  done
}

echo "== eval v3 backbones=${BACKBONES} (n=${BB_N}) PRUNE=${PRUNE} (n=${PRUNE_N}) MAX_FRAMES=${MAX_FRAMES} VIS_2D=${EXPORT_VIS_2D} VIS_3D=${EXPORT_VIS_3D} =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for prune_flag in "${PRUNE_LIST[@]}"; do
    prune_flag="${prune_flag//[[:space:]]/}"
    [[ -z "${prune_flag}" ]] && continue
    if truthy "${prune_flag}"; then
      run_eval_for_prune_variants "${bb}"
    else
      run_datasets_for_backbone "${bb}" "${prune_flag}" ""
    fi
  done
  echo "   [summary] analysis_outputs_v3/${bb}/summary_comparison.md"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_v3 \
    --experiment "${bb}" \
    --prune-plan-root "${PRUNE_PLAN_ROOT}" \
    --plan-name "${PLAN_NAME}"
done

echo "== done =="
experiment_output_hint "Eval outputs"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BACKBONES="${BACKBONES:-vggt}"
RUN_SCANNET_DEV50="${RUN_SCANNET_DEV50:-0}"
RUN_SCANNET_TEST50="${RUN_SCANNET_TEST50:-0}"
RUN_7SCENES_TEST="${RUN_7SCENES_TEST:-0}"
RUN_7SCENES_TRAIN="${RUN_7SCENES_TRAIN:-0}"
RUN_NRGBD="${RUN_NRGBD:-0}"
RUN_NUSCENES_TRAIN="${RUN_NUSCENES_TRAIN:-0}"
RUN_NUSCENES_VAL="${RUN_NUSCENES_VAL:-0}"
RUN_WAYMO_TRAINING="${RUN_WAYMO_TRAINING:-0}"
RUN_WAYMO_VALIDATION="${RUN_WAYMO_VALIDATION:-0}"

EXPORT_VIS_2D="${EXPORT_VIS_2D:-1}"
EXPORT_VIS_3D="${EXPORT_VIS_3D:-0}"

PRUNE="${PRUNE:-0,1}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
# Pruned eval selectors use the split plan layout:
#   PLAN_GROUP / PRUNE_PLAN_NAME: explicit prune selectors such as "prune_span5 prune_span10"
#   PLAN_MAX_DROP_RATIO + PLAN_TEST_COUNT: auto-sample prune_spanN selectors from prune_plan_all.json
# For model-mode eval, TRAINING_PLAN_NAME optionally composes the prune selector with
# a training plan bundle such as "training_lora_l1_r32".
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_MAX_DROP_RATIO="${PLAN_MAX_DROP_RATIO:-}"
PLAN_TEST_COUNT="${PLAN_TEST_COUNT:-}"
TRAINING_PLAN_ROOT="${TRAINING_PLAN_ROOT:-}"
TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME:-}"
MAX_SCENES="${MAX_SCENES:-0}"

#############################################

ROOT_SCANNET_DEV50="${ROOT_SCANNET_DEV50:-data/scans_dev50}"
ROOT_SCANNET_TEST50="${ROOT_SCANNET_TEST50:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
ROOT_NUSCENES="${ROOT_NUSCENES:-data/nuscenes}"
ROOT_WAYMO="${ROOT_WAYMO:-data/waymo}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
CHECKPOINT="${CHECKPOINT:-}"
HF_MODEL="${HF_MODEL:-}"
INFERENCE_COMPRESSION="${INFERENCE_COMPRESSION:-none}"
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
# No universal default: quantvggt requires calibration files matching the final LaRe topology/dtype.
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-}"

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

PLAN_GROUP_LIST=()
PRUNE_PLAN_NAME_LIST=()
read -r -a _PLAN_GROUP_RAW <<< "${PLAN_GROUP//,/ }"
for _value in "${_PLAN_GROUP_RAW[@]}"; do
  _value="${_value//[[:space:]]/}"
  [[ -n "${_value}" ]] && PLAN_GROUP_LIST+=("${_value}")
done
read -r -a _PRUNE_PLAN_NAME_RAW <<< "${PRUNE_PLAN_NAME//,/ }"
for _value in "${_PRUNE_PLAN_NAME_RAW[@]}"; do
  _value="${_value//[[:space:]]/}"
  [[ -n "${_value}" ]] && PRUNE_PLAN_NAME_LIST+=("${_value}")
done
if [[ "${#PLAN_GROUP_LIST[@]}" -gt 0 && "${#PRUNE_PLAN_NAME_LIST[@]}" -gt 0 ]]; then
  echo "error: set only one of PLAN_GROUP or PRUNE_PLAN_NAME for prune-span eval." >&2
  exit 1
fi
for _prune_value in "${PRUNE_LIST[@]}"; do
  _prune_value="${_prune_value//[[:space:]]/}"
  if [[ "${_prune_value}" == "1" && "${#PLAN_GROUP_LIST[@]}" -eq 0 && "${#PRUNE_PLAN_NAME_LIST[@]}" -eq 0 && ( -z "${PLAN_MAX_DROP_RATIO}" || -z "${PLAN_TEST_COUNT}" ) ]]; then
    echo "error: PLAN_GROUP/PRUNE_PLAN_NAME or PLAN_MAX_DROP_RATIO+PLAN_TEST_COUNT is required when PRUNE includes 1." >&2
    exit 1
  fi
done

case "${INFERENCE_COMPRESSION}" in
  none|lora_merge|fastvggt|quantvggt) ;;
  *)
    echo "error: INFERENCE_COMPRESSION must be none, lora_merge, fastvggt, or quantvggt; got ${INFERENCE_COMPRESSION}" >&2
    exit 1
    ;;
esac
case "${QUANTVGGT_DTYPE}" in
  w4a4|w6a6|w8a8) ;;
  *)
    echo "error: QUANTVGGT_DTYPE must be w4a4, w6a6, or w8a8; got ${QUANTVGGT_DTYPE}" >&2
    exit 1
    ;;
esac
if [[ ! "${FASTVGGT_MERGE_START}" =~ ^[0-9]+$ ]]; then
  echo "error: FASTVGGT_MERGE_START must be a non-negative integer; got ${FASTVGGT_MERGE_START}" >&2
  exit 1
fi
if [[ ! "${FASTVGGT_MERGE_RATIO}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "error: FASTVGGT_MERGE_RATIO must be numeric; got ${FASTVGGT_MERGE_RATIO}" >&2
  exit 1
fi
if [[ "${INFERENCE_COMPRESSION}" == "quantvggt" && -z "${QUANTVGGT_PARAMS_DIR}" ]]; then
  echo "error: QUANTVGGT_PARAMS_DIR is required for INFERENCE_COMPRESSION=quantvggt" >&2
  exit 1
fi
if [[ "${INFERENCE_COMPRESSION}" != "none" ]]; then
  for bb in "${BB_LIST[@]}"; do
    bb="${bb//[[:space:]]/}"
    if [[ -n "${bb}" && "${bb}" != "vggt" ]]; then
      echo "error: ${INFERENCE_COMPRESSION} inference compression only supports backbone vggt; got ${bb}" >&2
      exit 1
    fi
  done
fi

compression_output_tag() {
  case "${INFERENCE_COMPRESSION}" in
    none) echo "" ;;
    lora_merge) echo "lora_merge" ;;
    fastvggt)
      local ratio_normalized
      printf -v ratio_normalized '%g' "${FASTVGGT_MERGE_RATIO}"
      local ratio_tag="${ratio_normalized//./p}"
      echo "fastvggt_m$((FASTVGGT_MERGE_START + 0))_r${ratio_tag}"
      ;;
    quantvggt) echo "quantvggt_${QUANTVGGT_DTYPE}" ;;
  esac
}

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

python_plan_group_from_name() {
  local name="$1"
  PRUNE_PLAN_NAME_FOR_PY="${name}" conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
from analysis.prune_plan import plan_group_from_name
print(plan_group_from_name(os.environ['PRUNE_PLAN_NAME_FOR_PY']))
"
}

python_resolve_canonical_plan() {
  local root="$1"
  local group="$2"
  PRUNE_PLAN_ROOT_FOR_PY="${root}" PLAN_GROUP_FOR_PY="${group}" conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
from pathlib import Path
from analysis.prune_plan import canonical_plan_name_for_group
try:
    print(canonical_plan_name_for_group(Path(os.environ['PRUNE_PLAN_ROOT_FOR_PY']), os.environ['PLAN_GROUP_FOR_PY']))
except FileNotFoundError:
    pass
"
}

python_list_plan_groups() {
  local root="$1"
  PRUNE_PLAN_ROOT_FOR_PY="${root}" conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
from pathlib import Path
from analysis.prune_plan import list_plan_groups
for g in list_plan_groups(Path(os.environ['PRUNE_PLAN_ROOT_FOR_PY'])):
    print(g)
"
}

python_select_prune_spans() {
  local root="$1"
  local ratio="$2"
  local count="$3"
  PRUNE_PLAN_ROOT_FOR_PY="${root}" PLAN_MAX_DROP_RATIO_FOR_PY="${ratio}" PLAN_TEST_COUNT_FOR_PY="${count}" conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
from pathlib import Path
from analysis.prune_plan import load_prune_plan_bundle
root = Path(os.environ['PRUNE_PLAN_ROOT_FOR_PY'])
ratio = float(os.environ['PLAN_MAX_DROP_RATIO_FOR_PY'])
count = int(os.environ['PLAN_TEST_COUNT_FOR_PY'])
if ratio <= 0 or ratio > 1:
    raise SystemExit('PLAN_MAX_DROP_RATIO must be in (0,1]')
if count <= 0:
    raise SystemExit('PLAN_TEST_COUNT must be > 0')
b = load_prune_plan_bundle(root)
n = int(b.get('num_units', 0))
spans = sorted(int(k) for k in b.get('spans', {}).keys() if str(k).isdigit())
max_span = min(max(spans, default=0), max(1, int(round(n * ratio))))
eligible = [s for s in spans if 1 <= s <= max_span]
if not eligible:
    raise SystemExit(f'no prune spans <= {max_span} under {root}')
targets = [int(round(i * max_span / count)) for i in range(1, count + 1)]
selected = []
for target in targets:
    target = max(1, min(max_span, int(target)))
    if target in eligible:
        pick = target
    else:
        pick = min(eligible, key=lambda s: (abs(s - target), s))
    if pick not in selected:
        selected.append(pick)
for s in selected:
    print(f'prune_span{s}')
"
}

# upper: full pretrained; lower: apply prune_spanN only; model: apply prune_spanN plus finetuned checkpoint.
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
  local prune_plan_name="${3:-}"
  if [[ -n "${CHECKPOINT}" ]]; then
    echo "${CHECKPOINT}"
    return
  fi
  if [[ -n "${prune_plan_name}" && -n "${TRAINING_PLAN_NAME}" ]]; then
    echo "${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${scannet_tag}/train/${prune_plan_name}/${TRAINING_PLAN_NAME}/checkpoints/latest.pt"
    return
  fi
  echo "${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${scannet_tag}/train/checkpoints/latest.pt"
}

run_eval_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local mode="$4"
  local model_name="$5"
  local output_dataset="${6:-${dataset}}"
  local prune_plan_name="${7:-}"
  local plan_group="${8:-}"
  local split="${9:-}"
  set_backend_pythonpath "${backbone}"

  local compression_tag
  compression_tag="$(compression_output_tag)"
  local effective_model_name="${model_name}"
  if [[ -n "${compression_tag}" ]]; then
    effective_model_name="${model_name}_${compression_tag}"
  fi
  local out_dir="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${output_dataset}/eval/${effective_model_name}"
  if [[ "${mode}" == "lower" && -n "${plan_group}" ]]; then
    out_dir="${out_dir}/${plan_group}"
  elif [[ "${mode}" == "model" && -n "${prune_plan_name}" ]]; then
    out_dir="${out_dir}/${prune_plan_name}"
  fi
  local extra=(
    --eval-mode "${mode}"
    --prune-plan-root "${PRUNE_PLAN_ROOT}"
    --prune-plan-name "${prune_plan_name}"
    --model-name "${model_name}"
    --output-dir "${out_dir}"
    --inference-compression "${INFERENCE_COMPRESSION}"
    --fastvggt-merge-start "${FASTVGGT_MERGE_START}"
    --fastvggt-merge-ratio "${FASTVGGT_MERGE_RATIO}"
    --quantvggt-dtype "${QUANTVGGT_DTYPE}"
  )
  local training_root="${TRAINING_PLAN_ROOT:-${PRUNE_PLAN_ROOT}}"
  if [[ -n "${TRAINING_PLAN_NAME}" ]]; then
    extra+=(--training-plan-root "${training_root}")
    extra+=(--training-plan-name "${TRAINING_PLAN_NAME}")
  fi
  if [[ -n "${QUANTVGGT_PARAMS_DIR}" ]]; then
    extra+=(--quantvggt-params-dir "${QUANTVGGT_PARAMS_DIR}")
  fi
  if [[ -n "${HF_MODEL}" ]]; then
    extra+=(--hf-model "${HF_MODEL}")
  fi
  extra+=(--max-frames "${MAX_FRAMES}")
  if [[ -n "${split}" ]]; then
    extra+=(--split "${split}")
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
      ckpt="$(checkpoint_for_model_mode "${backbone}" "${ckpt_tag}" "${prune_plan_name}")"
      if [[ -f "${ckpt}" ]]; then
        extra+=(--checkpoint "${ckpt}")
      else
        echo "error: model mode requires checkpoint (set CHECKPOINT or train first): ${ckpt}" >&2
        return 1
      fi
      ;;
  esac

  echo "   [eval/${output_dataset}] ${effective_model_name}/${backbone} mode=${mode} compression=${INFERENCE_COMPRESSION}"
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
  local prune_plan_name="${3:-}"
  local plan_group="${4:-}"
  local mode
  mode="$(resolve_eval_mode "${prune_flag}")"
  local model_name="${MODEL_NAME:-${mode}}"

  echo "   [eval] ${backbone} PRUNE=${prune_flag} -> ${mode}/${model_name}${plan_group:+ group=${plan_group}}${prune_plan_name:+ prune_plan_name=${prune_plan_name}}"
  if truthy "${prune_flag}"; then
    if [[ -n "${prune_plan_name}" ]]; then
      echo "   [prune] ${prune_plan_name} from ${PRUNE_PLAN_ROOT}/prune_plan_all.json"
    else
      echo "   [prune] ${PRUNE_PLAN_ROOT}/${backbone}.json"
    fi
  fi

  if truthy "${RUN_SCANNET_DEV50}"; then
    run_eval_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "${mode}" "${model_name}" "scannet_dev50" "${prune_plan_name}" "${plan_group}"
  fi
  if truthy "${RUN_SCANNET_TEST50}"; then
    run_eval_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "${mode}" "${model_name}" "scannet_test50" "${prune_plan_name}" "${plan_group}"
  fi
  if truthy "${RUN_7SCENES_TEST}"; then
    run_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${mode}" "${model_name}" "7scenes_test" "${prune_plan_name}" "${plan_group}" "test"
  fi
  if truthy "${RUN_7SCENES_TRAIN}"; then
    run_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${mode}" "${model_name}" "7scenes_train" "${prune_plan_name}" "${plan_group}" "train"
  fi
  if [[ "${RUN_NRGBD}" == "1" || "${RUN_NRGBD}" == "true" ]]; then
    run_eval_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "${mode}" "${model_name}" "nrgbd" "${prune_plan_name}" "${plan_group}"
  fi
  if truthy "${RUN_NUSCENES_VAL}"; then
    run_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${mode}" "${model_name}" "nuscenes_val" "${prune_plan_name}" "${plan_group}" "val"
  fi
  if truthy "${RUN_NUSCENES_TRAIN}"; then
    run_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${mode}" "${model_name}" "nuscenes_train" "${prune_plan_name}" "${plan_group}" "train"
  fi
  if truthy "${RUN_WAYMO_VALIDATION}"; then
    run_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${mode}" "${model_name}" "waymo_validation" "${prune_plan_name}" "${plan_group}" "validation"
  fi
  if truthy "${RUN_WAYMO_TRAINING}"; then
    run_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${mode}" "${model_name}" "waymo_training" "${prune_plan_name}" "${plan_group}" "training"
  fi
}

run_eval_group_variant() {
  local bb="$1"
  local group="$2"
  local canonical
  canonical="$(python_resolve_canonical_plan "${PRUNE_PLAN_ROOT}" "${group}")"
  canonical="${canonical//[[:space:]]/}"
  if [[ -z "${canonical}" ]]; then
    echo "warning: skip missing prune-span selector for ${bb}: ${group} under ${PRUNE_PLAN_ROOT}" >&2
    return 0
  fi
  run_datasets_for_backbone "${bb}" "1" "${canonical}" "${group}"
}

run_eval_for_prune_variants() {
  local bb="$1"
  if [[ "${#PLAN_GROUP_LIST[@]}" -gt 0 ]]; then
    local requested_group
    for requested_group in "${PLAN_GROUP_LIST[@]}"; do
      run_eval_group_variant "${bb}" "${requested_group}"
    done
    return
  fi
  if [[ "${#PRUNE_PLAN_NAME_LIST[@]}" -gt 0 ]]; then
    local requested_name
    if [[ -n "${CHECKPOINT}" ]]; then
      for requested_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
        run_datasets_for_backbone "${bb}" "1" "${requested_name}" ""
      done
      return
    fi
    local group
    for requested_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
      group="$(python_plan_group_from_name "${requested_name}")"
      run_eval_group_variant "${bb}" "${group}"
    done
    return
  fi
  if [[ -n "${PLAN_MAX_DROP_RATIO}" || -n "${PLAN_TEST_COUNT}" ]]; then
    if [[ -z "${PLAN_MAX_DROP_RATIO}" || -z "${PLAN_TEST_COUNT}" ]]; then
      echo "error: PLAN_MAX_DROP_RATIO and PLAN_TEST_COUNT must be set together." >&2
      exit 1
    fi
    local selected_groups=()
    local sg
    while IFS= read -r sg; do
      [[ -n "${sg}" ]] && selected_groups+=("${sg}")
    done < <(python_select_prune_spans "${PRUNE_PLAN_ROOT}" "${PLAN_MAX_DROP_RATIO}" "${PLAN_TEST_COUNT}")
    for sg in "${selected_groups[@]}"; do
      run_eval_group_variant "${bb}" "${sg}"
    done
    return
  fi
  local root="${PRUNE_PLAN_ROOT}"
  if [[ "${root}" == "${REPO_ROOT}/analysis/prune_plan" || "${root}" == "analysis/prune_plan" ]]; then
    run_datasets_for_backbone "${bb}" "1" "" ""
    return
  fi
  if [[ ! -d "${root}" ]]; then
    echo "error: PRUNE_PLAN_ROOT not found: ${root}" >&2
    exit 1
  fi
  local groups=()
  local g
  while IFS= read -r g; do
    [[ -n "${g}" ]] && groups+=("${g}")
  done < <(python_list_plan_groups "${root}")
  if [[ "${#groups[@]}" -eq 0 ]]; then
    echo "error: no prune-span selectors found under ${root} (expected prune_plan_all.json)" >&2
    exit 1
  fi
  for g in "${groups[@]}"; do
    run_eval_group_variant "${bb}" "${g}"
  done
}

echo "== eval v3 backbones=${BACKBONES} (n=${BB_N}) PRUNE=${PRUNE} (n=${PRUNE_N}) MAX_FRAMES=${MAX_FRAMES} COMPRESSION=${INFERENCE_COMPRESSION} VIS_2D=${EXPORT_VIS_2D} VIS_3D=${EXPORT_VIS_3D} =="

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
  echo "   [summary] ${OUTPUT_ROOT}/${bb}/summary_comparison.md"
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_v3 \
    --experiment "${bb}" \
    --prune-plan-root "${PRUNE_PLAN_ROOT}"
done

echo "== done =="
experiment_output_hint "Eval outputs"

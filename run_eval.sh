#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN="${RUN:-}"

EXPORT_VIS_2D="${EXPORT_VIS_2D:-1}"
EXPORT_VIS_3D="${EXPORT_VIS_3D:-0}"

PRUNE="${PRUNE:-0,1}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${REPO_ROOT}/analysis/prune_plan}"
# Pruned eval selectors use the split plan layout:
#   PLAN_GROUP / PRUNE_PLAN_NAME: explicit physical interval selectors such as "i9_j19 i8_j20".
#   PLAN_STRATEGY: strategy keys from prune_plan_all.json, such as "ours gt cosine".
#   PLAN_MIN_DROP + PLAN_MAX_DROP + PLAN_DROP_STEP: select interval union from strategies by range.
# For model-mode eval, RECOVERY_PLAN_NAME optionally composes
# the prune selector with a recovery plan bundle such as "recovery_lora_l1_r32".
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
PLAN_GROUP="${PLAN_GROUP:-}"
PLAN_STRATEGY="${PLAN_STRATEGY:-}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"
MAX_SCENES="${MAX_SCENES:-0}"
WRITE_SUMMARY="${WRITE_SUMMARY:-1}"

#############################################

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
CHECKPOINT="${CHECKPOINT:-}"
INFERENCE_COMPRESSION="${INFERENCE_COMPRESSION:-none}"
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
# No universal default: quantvggt requires calibration files matching the final LaRe topology/dtype.
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-}"

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${RUN}" "RUN" "${DATASET_KEYS[@]}"
RUN_LIST=("${PARSED_DATASETS[@]}")

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
PLAN_STRATEGY_LIST=()
read -r -a _PLAN_GROUP_RAW <<< "${PLAN_GROUP//,/ }"
for _value in "${_PLAN_GROUP_RAW[@]}"; do
  _value="${_value//[[:space:]]/}"
  [[ -n "${_value}" ]] && PLAN_GROUP_LIST+=("${_value}")
done
read -r -a _PLAN_STRATEGY_RAW <<< "${PLAN_STRATEGY//,/ }"
for _value in "${_PLAN_STRATEGY_RAW[@]}"; do
  _value="${_value//[[:space:]]/}"
  [[ -n "${_value}" ]] && PLAN_STRATEGY_LIST+=("${_value}")
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
if [[ "${#PLAN_STRATEGY_LIST[@]}" -gt 0 && ( "${#PLAN_GROUP_LIST[@]}" -gt 0 || "${#PRUNE_PLAN_NAME_LIST[@]}" -gt 0 ) ]]; then
  echo "error: set PLAN_STRATEGY with range selectors, or explicit PLAN_GROUP/PRUNE_PLAN_NAME, not both." >&2
  exit 1
fi
for _prune_value in "${PRUNE_LIST[@]}"; do
  _prune_value="${_prune_value//[[:space:]]/}"
  if [[ "${_prune_value}" == "1" && "${#PLAN_GROUP_LIST[@]}" -eq 0 && "${#PRUNE_PLAN_NAME_LIST[@]}" -eq 0 && "${#PLAN_STRATEGY_LIST[@]}" -eq 0 && ( -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ) ]]; then
    echo "error: PLAN_GROUP/PRUNE_PLAN_NAME, PLAN_STRATEGY, or PLAN_MIN_DROP+PLAN_MAX_DROP+PLAN_DROP_STEP is required when PRUNE includes 1." >&2
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

python_select_interval_range() {
  local root="$1"
  local min_drop="$2"
  local max_drop="$3"
  local step="$4"
  local strategies="$5"
  PRUNE_PLAN_ROOT_FOR_PY="${root}" PLAN_MIN_DROP_FOR_PY="${min_drop}" PLAN_MAX_DROP_FOR_PY="${max_drop}" PLAN_DROP_STEP_FOR_PY="${step}" PLAN_STRATEGY_FOR_PY="${strategies}" conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
from pathlib import Path
from analysis.prune_plan import strategy_interval_selectors
root = Path(os.environ['PRUNE_PLAN_ROOT_FOR_PY'])
try:
    min_drop = int(os.environ['PLAN_MIN_DROP_FOR_PY'])
    max_drop = int(os.environ['PLAN_MAX_DROP_FOR_PY'])
    step = int(os.environ['PLAN_DROP_STEP_FOR_PY'])
except ValueError as exc:
    raise SystemExit('PLAN_MIN_DROP, PLAN_MAX_DROP, and PLAN_DROP_STEP must be integers') from exc
if min_drop < 0:
    raise SystemExit('PLAN_MIN_DROP must be >= 0')
if max_drop < min_drop:
    raise SystemExit('PLAN_MAX_DROP must be >= PLAN_MIN_DROP')
if step <= 0:
    raise SystemExit('PLAN_DROP_STEP must be > 0')
strategies = [x for x in os.environ.get('PLAN_STRATEGY_FOR_PY', '').replace(',', ' ').split() if x.strip()]
selected = strategy_interval_selectors(
    root,
    strategies=strategies,
    min_drop=min_drop,
    max_drop=max_drop,
    drop_step=step,
)
if not selected:
    raise SystemExit(f'no available prune intervals in [{min_drop}, {max_drop}] step {step} under {root}')
for selector in selected:
    print(selector)
"
}

# upper: full pretrained; lower: apply one physical interval only; model: apply prune plan plus finetuned checkpoint.
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
  if [[ -n "${prune_plan_name}" && -n "${RECOVERY_PLAN_NAME}" ]]; then
    local recover_ckpt="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/recover/${prune_plan_name}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
    local new_ckpt="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/train/${prune_plan_name}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
    local legacy_ckpt="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${scannet_tag}/train/${prune_plan_name}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
    if [[ -f "${recover_ckpt}" ]]; then
      echo "${recover_ckpt}"
    elif [[ -f "${new_ckpt}" || ! -f "${legacy_ckpt}" ]]; then
      echo "${new_ckpt}"
    else
      echo "${legacy_ckpt}"
    fi
    return
  fi
  local new_ckpt="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/train/checkpoints/latest.pt"
  local legacy_ckpt="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${scannet_tag}/train/checkpoints/latest.pt"
  if [[ -f "${new_ckpt}" || ! -f "${legacy_ckpt}" ]]; then
    echo "${new_ckpt}"
  else
    echo "${legacy_ckpt}"
  fi
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
    if [[ -n "${RECOVERY_PLAN_NAME}" ]]; then
      out_dir="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${output_dataset}/eval/recover/${RECOVERY_PLAN_NAME}/${prune_plan_name}"
    else
      out_dir="${out_dir}/${prune_plan_name}"
    fi
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
  local recovery_root="${RECOVERY_PLAN_ROOT:-${PRUNE_PLAN_ROOT}}"
  if [[ -n "${RECOVERY_PLAN_NAME}" ]]; then
    extra+=(--recovery-plan-root "${recovery_root}")
    extra+=(--recovery-plan-name "${RECOVERY_PLAN_NAME}")
  fi
  if [[ -n "${QUANTVGGT_PARAMS_DIR}" ]]; then
    extra+=(--quantvggt-params-dir "${QUANTVGGT_PARAMS_DIR}")
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
        echo "error: model mode requires checkpoint (set CHECKPOINT or recover first): ${ckpt}" >&2
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

run_eval_dataset_key() {
  local key="$1"
  local backbone="$2"
  local mode="$3"
  local model_name="$4"
  local prune_plan_name="${5:-}"
  local plan_group="${6:-}"

  case "${key}" in
    SCANNET_DEV50)
      run_eval_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "${mode}" "${model_name}" "scannet_dev50" "${prune_plan_name}" "${plan_group}"
      ;;
    SCANNET_TEST50)
      run_eval_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "${mode}" "${model_name}" "scannet_test50" "${prune_plan_name}" "${plan_group}"
      ;;
    7SCENES_TEST)
      run_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${mode}" "${model_name}" "7scenes_test" "${prune_plan_name}" "${plan_group}" "test"
      ;;
    7SCENES_TRAIN)
      run_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "${mode}" "${model_name}" "7scenes_train" "${prune_plan_name}" "${plan_group}" "train"
      ;;
    NRGBD)
      run_eval_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "${mode}" "${model_name}" "nrgbd" "${prune_plan_name}" "${plan_group}"
      ;;
    NUSCENES_VAL)
      run_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${mode}" "${model_name}" "nuscenes_val" "${prune_plan_name}" "${plan_group}" "val"
      ;;
    NUSCENES_TRAIN)
      run_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "${mode}" "${model_name}" "nuscenes_train" "${prune_plan_name}" "${plan_group}" "train"
      ;;
    WAYMO_VALIDATION)
      run_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${mode}" "${model_name}" "waymo_validation" "${prune_plan_name}" "${plan_group}" "validation"
      ;;
    WAYMO_TRAINING)
      run_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "${mode}" "${model_name}" "waymo_training" "${prune_plan_name}" "${plan_group}" "training"
      ;;
  esac
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

  local dataset_key
  for dataset_key in "${RUN_LIST[@]}"; do
    run_eval_dataset_key "${dataset_key}" "${backbone}" "${mode}" "${model_name}" "${prune_plan_name}" "${plan_group}"
  done
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
  run_datasets_for_backbone "${bb}" "1" "${canonical}" "${canonical}"
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
  if [[ "${#PLAN_STRATEGY_LIST[@]}" -gt 0 && -z "${PLAN_MIN_DROP}${PLAN_MAX_DROP}${PLAN_DROP_STEP}" ]]; then
    local strategy_selectors=()
    local strategy_csv="${PLAN_STRATEGY_LIST[*]}"
    local selector
    while IFS= read -r selector; do
      [[ -n "${selector}" ]] && strategy_selectors+=("${selector}")
    done < <(python_select_interval_range "${PRUNE_PLAN_ROOT}" "1" "9999" "1" "${strategy_csv}")
    for selector in "${strategy_selectors[@]}"; do
      run_eval_group_variant "${bb}" "${selector}"
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
  if [[ -n "${PLAN_MIN_DROP}" || -n "${PLAN_MAX_DROP}" || -n "${PLAN_DROP_STEP}" ]]; then
    if [[ -z "${PLAN_MIN_DROP}" || -z "${PLAN_MAX_DROP}" || -z "${PLAN_DROP_STEP}" ]]; then
      echo "error: PLAN_MIN_DROP, PLAN_MAX_DROP, and PLAN_DROP_STEP must be set together." >&2
      exit 1
    fi
    local selected_groups=()
    local sg
    local strategy_csv="${PLAN_STRATEGY_LIST[*]}"
    while IFS= read -r sg; do
      [[ -n "${sg}" ]] && selected_groups+=("${sg}")
    done < <(python_select_interval_range "${PRUNE_PLAN_ROOT}" "${PLAN_MIN_DROP}" "${PLAN_MAX_DROP}" "${PLAN_DROP_STEP}" "${strategy_csv}")
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

echo "== eval v3 backbones=${BACKBONES} (n=${BB_N}) RUN=${RUN:-<none>} PRUNE=${PRUNE} (n=${PRUNE_N}) MAX_FRAMES=${MAX_FRAMES} COMPRESSION=${INFERENCE_COMPRESSION} VIS_2D=${EXPORT_VIS_2D} VIS_3D=${EXPORT_VIS_3D} =="

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
  if truthy "${WRITE_SUMMARY}"; then
    echo "   [summary] ${OUTPUT_ROOT}/${bb}/summary_comparison.md"
    conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_v3 \
      --experiment "${bb}" \
      --prune-plan-root "${PRUNE_PLAN_ROOT}"
  fi
done

echo "== done =="
experiment_output_hint "Eval outputs"

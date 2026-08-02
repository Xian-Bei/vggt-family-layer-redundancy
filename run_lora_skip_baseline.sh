#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# v3 cache/eval outputs are split by dataset tag under ${OUTPUT_ROOT}/<backbone>/<dataset_tag>/...
# Trainable recovery can consume one or more cache dataset tags and writes under
# ${OUTPUT_ROOT}/<backbone>/<OUTPUT_KIND>/<prune_plan>/<recovery_plan>/.
DATASETS="${DATASETS:-${DATASET:-scannet_dev50}}"
EPOCHS="${EPOCHS:-5}"
MAX_SCENES="${MAX_SCENES:-1}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
OUTPUT_KIND="${OUTPUT_KIND:-train}"
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"

#############################################
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
DTYPE="${DTYPE:-bfloat16}"

source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${EVAL}" "EVAL" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

run_progress_command() {
  if truthy "${COMPACT_LOG:-0}"; then
    "$@" >/dev/null
  else
    "$@"
  fi
}

PRUNE_PLAN_NAME_LIST=()
read -r -a _PRUNE_PLAN_NAME_RAW <<< "${PRUNE_PLAN_NAME//,/ }"
for _prune_plan_name in "${_PRUNE_PLAN_NAME_RAW[@]}"; do
  _prune_plan_name="${_prune_plan_name//[[:space:]]/}"
  [[ -n "${_prune_plan_name}" ]] && PRUNE_PLAN_NAME_LIST+=("${_prune_plan_name}")
done
if [[ "${#PRUNE_PLAN_NAME_LIST[@]}" -eq 0 ]]; then
  echo "error: PRUNE_PLAN_NAME is required (e.g. PRUNE_PLAN_NAME='prune_span5 prune_span10')." >&2
  exit 1
fi

RECOVERY_PLAN_NAME_LIST=()
read -r -a _RECOVERY_PLAN_NAME_RAW <<< "${RECOVERY_PLAN_NAME//,/ }"
for _recovery_plan_name in "${_RECOVERY_PLAN_NAME_RAW[@]}"; do
  _recovery_plan_name="${_recovery_plan_name//[[:space:]]/}"
  if [[ -n "${_recovery_plan_name}" ]]; then
    RECOVERY_PLAN_NAME_LIST+=("${_recovery_plan_name}")
  fi
done
if [[ "${#RECOVERY_PLAN_NAME_LIST[@]}" -eq 0 ]]; then
  echo "error: RECOVERY_PLAN_NAME is required (e.g. RECOVERY_PLAN_NAME='recovery_lora_l1_r32')." >&2
  exit 1
fi

DATASET_LIST=()
read -r -a _DATASET_RAW <<< "${DATASETS//,/ }"
for _dataset_name in "${_DATASET_RAW[@]}"; do
  _dataset_name="${_dataset_name//[[:space:]]/}"
  [[ -n "${_dataset_name}" ]] && DATASET_LIST+=("${_dataset_name}")
done
if [[ "${#DATASET_LIST[@]}" -eq 0 ]]; then
  echo "error: DATASETS is empty (e.g. DATASETS='scannet_dev50 nuscenes_train')." >&2
  exit 1
fi
for _dataset_name in "${DATASET_LIST[@]}"; do
  if [[ "${_dataset_name}" == "scannet" ]]; then
    echo "error: DATASET(S)=scannet is legacy; use scannet_dev50 (train/cache) and eval splits via EVAL=SCANNET_DEV50,SCANNET_TEST50." >&2
    exit 1
  fi
done
DATASETS_FOR_PY="${DATASET_LIST[*]}"

eval_enabled() {
  list_has "$1" "${EVAL_LIST[@]}"
}

resolve_eval_7scenes_split() {
  if eval_enabled "7SCENES_TEST" && eval_enabled "7SCENES_TRAIN"; then
    echo "error: run_lora_skip_baseline.sh supports one 7scenes eval split per run; include only 7SCENES_TEST or 7SCENES_TRAIN in EVAL." >&2
    exit 1
  fi
  if eval_enabled "7SCENES_TRAIN"; then
    echo "train"
  elif eval_enabled "7SCENES_TEST"; then
    echo "test"
  else
    echo ""
  fi
}

resolve_eval_nuscenes_split() {
  if eval_enabled "NUSCENES_VAL" && eval_enabled "NUSCENES_TRAIN"; then
    echo "error: run_lora_skip_baseline.sh supports one nuscenes eval split per run; include only NUSCENES_VAL or NUSCENES_TRAIN in EVAL." >&2
    exit 1
  fi
  if eval_enabled "NUSCENES_TRAIN"; then
    echo "train"
  elif eval_enabled "NUSCENES_VAL"; then
    echo "val"
  else
    echo ""
  fi
}

resolve_eval_waymo_split() {
  if eval_enabled "WAYMO_VALIDATION" && eval_enabled "WAYMO_TRAINING"; then
    echo "error: run_lora_skip_baseline.sh supports one waymo eval split per run; include only WAYMO_VALIDATION or WAYMO_TRAINING in EVAL." >&2
    exit 1
  fi
  if eval_enabled "WAYMO_TRAINING"; then
    echo "training"
  elif eval_enabled "WAYMO_VALIDATION"; then
    echo "validation"
  else
    echo ""
  fi
}

run_one_backbone() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  set_backend_pythonpath "${backbone}"
  export BACKEND="${backbone}"

  local plan_root="${PRUNE_PLAN_ROOT}"
  if [[ -z "${plan_root}" ]]; then
    plan_root="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}"
  fi
  local recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"

  local extra=(
    --prune-plan-root "${plan_root}"
    --prune-plan-name "${prune_plan_name}"
    --recovery-plan-root "${recovery_root}"
    --recovery-plan-name "${recovery_plan_name}"
    --eval-bounds-root "${EVAL_BOUNDS_ROOT}"
    --output-dir "${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${OUTPUT_KIND}/${prune_plan_name}/${recovery_plan_name}"
  )
  if eval_enabled "SCANNET_DEV50"; then
    extra+=(--eval-scannet-dev50)
  else
    extra+=(--no-eval-scannet-dev50)
  fi
  if eval_enabled "SCANNET_TEST50"; then
    extra+=(--eval-scannet-test50)
  else
    extra+=(--no-eval-scannet-test50)
  fi
  local eval_7scenes_split
  eval_7scenes_split="$(resolve_eval_7scenes_split)"
  if [[ -n "${eval_7scenes_split}" ]]; then
    extra+=(--eval-7scenes)
    extra+=(--eval-7scenes-split "${eval_7scenes_split}")
  else
    extra+=(--no-eval-7scenes)
  fi
  if eval_enabled "NRGBD"; then
    extra+=(--eval-nrgbd)
  else
    extra+=(--no-eval-nrgbd)
  fi
  local eval_nuscenes_split
  eval_nuscenes_split="$(resolve_eval_nuscenes_split)"
  if [[ -n "${eval_nuscenes_split}" ]]; then
    extra+=(--eval-nuscenes)
    extra+=(--eval-nuscenes-split "${eval_nuscenes_split}")
  else
    extra+=(--no-eval-nuscenes)
  fi
  local eval_waymo_split
  eval_waymo_split="$(resolve_eval_waymo_split)"
  if [[ -n "${eval_waymo_split}" ]]; then
    extra+=(--eval-waymo)
    extra+=(--eval-waymo-split "${eval_waymo_split}")
  else
    extra+=(--no-eval-waymo)
  fi
  extra+=(--root-7scenes "${ROOT_7SCENES}")
  extra+=(--root-nrgbd "${ROOT_NRGBD}")
  extra+=(--root-nuscenes "${ROOT_NUSCENES}")
  extra+=(--root-waymo "${ROOT_WAYMO}")
  if [[ "${EVAL_AT_START}" == "1" || "${EVAL_AT_START}" == "true" || "${EVAL_AT_START}" == "yes" ]]; then
    extra+=(--eval-at-start)
  else
    extra+=(--no-eval-at-start)
  fi

  if truthy "${COMPACT_LOG:-0}"; then
    printf '\n  recovery training\n    model: %s\n    span: %s\n    dataset: %s\n    method: %s\n\n' \
      "${backbone}" "${prune_plan_name#prune_span}" "${DATASETS_FOR_PY}" \
      "${recovery_plan_name#recovery_}"
  else
    echo "== ${OUTPUT_KIND} v3 datasets=${DATASETS_FOR_PY} backbone=${backbone} prune_plan_name=${prune_plan_name} recovery_plan_name=${recovery_plan_name} prune_plan_root=${plan_root} eval_at_start=${EVAL_AT_START} save_every_epoch=${SAVE_EVERY_EPOCH} =="
  fi
  run_progress_command conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.recovery.train.train_prune_lora_v3 \
    --experiment "${backbone}" \
    --dataset "${DATASETS_FOR_PY}" \
    --backbone "${backbone}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --max-scenes "${MAX_SCENES}" \
    --max-steps "${MAX_STEPS}" \
    --dtype "${DTYPE}" \
    --eval-epoch "${EVAL_EPOCH}" \
    --save-every-epoch "${SAVE_EVERY_EPOCH}" \
    --eval-max-scenes "${EVAL_MAX_SCENES}" \
    --max-eval-views "${MAX_FRAMES}" \
    "${extra[@]}"
}

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== ${OUTPUT_KIND} v3 backbones=${BACKBONES} (n=${BB_N}) =="
fi

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for prune_plan_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
    for recovery_plan_name in "${RECOVERY_PLAN_NAME_LIST[@]}"; do
      run_one_backbone "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
    done
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  if [[ "${OUTPUT_KIND}" == "recovery" ]]; then
    experiment_output_hint "Recover"
  else
    experiment_output_hint "Train"
  fi
fi

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# v3 outputs are split by dataset tag under ${OUTPUT_ROOT}/<backbone>/<dataset_tag>/...
# Default to the dev50 tag (historically called "scannet" but renamed to avoid collisions).
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
DATASET="${DATASET:-scannet_dev50}"
EPOCHS="${EPOCHS:-5}"
MAX_SCENES="${MAX_SCENES:-1}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
TRAINING_PLAN_ROOT="${TRAINING_PLAN_ROOT:-}"
TRAINING_PLAN_NAME="${TRAINING_PLAN_NAME:-}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${REPO_ROOT}/${OUTPUT_ROOT}}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"
EVAL_SCANNET_DEV50="${EVAL_SCANNET_DEV50:-0}"
EVAL_SCANNET_TEST50="${EVAL_SCANNET_TEST50:-0}"
EVAL_7SCENES_TRAIN="${EVAL_7SCENES_TRAIN:-0}"
if [[ -z "${EVAL_7SCENES_TEST+x}" ]]; then
  if [[ "${EVAL_7SCENES_TRAIN}" == "1" || "${EVAL_7SCENES_TRAIN}" == "true" || "${EVAL_7SCENES_TRAIN}" == "yes" ]]; then
    EVAL_7SCENES_TEST=0
  else
    EVAL_7SCENES_TEST=1
  fi
fi
EVAL_NRGBD="${EVAL_NRGBD:-0}"
EVAL_NUSCENES_TRAIN="${EVAL_NUSCENES_TRAIN:-0}"
EVAL_NUSCENES_VAL="${EVAL_NUSCENES_VAL:-0}"
EVAL_WAYMO_TRAINING="${EVAL_WAYMO_TRAINING:-0}"
EVAL_WAYMO_VALIDATION="${EVAL_WAYMO_VALIDATION:-0}"

#############################################
HF_MODEL="${HF_MODEL:-}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
DTYPE="${DTYPE:-bfloat16}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
ROOT_NUSCENES="${ROOT_NUSCENES:-data/nuscenes}"
ROOT_WAYMO="${ROOT_WAYMO:-data/waymo}"

source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list

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

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

resolve_eval_7scenes_split() {
  if truthy "${EVAL_7SCENES_TEST}" && truthy "${EVAL_7SCENES_TRAIN}"; then
    echo "error: run_lora_skip_baseline.sh supports one 7scenes eval split per run; enable only EVAL_7SCENES_TEST or EVAL_7SCENES_TRAIN." >&2
    exit 1
  fi
  if truthy "${EVAL_7SCENES_TRAIN}"; then
    echo "train"
  elif truthy "${EVAL_7SCENES_TEST}"; then
    echo "test"
  else
    echo ""
  fi
}

resolve_eval_nuscenes_split() {
  if truthy "${EVAL_NUSCENES_VAL}" && truthy "${EVAL_NUSCENES_TRAIN}"; then
    echo "error: run_lora_skip_baseline.sh supports one nuscenes eval split per run; enable only EVAL_NUSCENES_VAL or EVAL_NUSCENES_TRAIN." >&2
    exit 1
  fi
  if truthy "${EVAL_NUSCENES_TRAIN}"; then
    echo "train"
  elif truthy "${EVAL_NUSCENES_VAL}"; then
    echo "val"
  else
    echo ""
  fi
}

resolve_eval_waymo_split() {
  if truthy "${EVAL_WAYMO_VALIDATION}" && truthy "${EVAL_WAYMO_TRAINING}"; then
    echo "error: run_lora_skip_baseline.sh supports one waymo eval split per run; enable only EVAL_WAYMO_VALIDATION or EVAL_WAYMO_TRAINING." >&2
    exit 1
  fi
  if truthy "${EVAL_WAYMO_TRAINING}"; then
    echo "training"
  elif truthy "${EVAL_WAYMO_VALIDATION}"; then
    echo "validation"
  else
    echo ""
  fi
}

run_one_backbone() {
  local backbone="$1"
  local prune_plan_name="$2"
  set_backend_pythonpath "${backbone}"
  export BACKEND="${backbone}"

  local plan_root="${PRUNE_PLAN_ROOT}"
  if [[ -z "${plan_root}" ]]; then
    plan_root="${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}"
  fi
  local training_root="${TRAINING_PLAN_ROOT:-${plan_root}}"
  if [[ -z "${TRAINING_PLAN_NAME}" ]]; then
    echo "error: TRAINING_PLAN_NAME is required (e.g. TRAINING_PLAN_NAME='training_lora_l1_r32')." >&2
    exit 1
  fi

  # Guardrail: `scannet` legacy tag no longer exists after the dev/test split.
  if [[ "${DATASET}" == "scannet" ]]; then
    echo "error: DATASET=scannet is legacy; use DATASET=scannet_dev50 (train/cache) and eval splits via EVAL_SCANNET_* switches." >&2
    exit 1
  fi

  local extra=(
    --prune-plan-root "${plan_root}"
    --prune-plan-name "${prune_plan_name}"
    --training-plan-root "${training_root}"
    --training-plan-name "${TRAINING_PLAN_NAME}"
    --eval-bounds-root "${EVAL_BOUNDS_ROOT}"
    --output-dir "${REPO_ROOT}/${OUTPUT_ROOT}/${backbone}/${DATASET}/train/${prune_plan_name}/${TRAINING_PLAN_NAME}"
  )
  if [[ "${EVAL_SCANNET_DEV50}" == "1" ]]; then
    extra+=(--eval-scannet-dev50)
  else
    extra+=(--no-eval-scannet-dev50)
  fi
  if [[ "${EVAL_SCANNET_TEST50}" == "1" ]]; then
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
  if [[ "${EVAL_NRGBD}" == "1" ]]; then
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
  if [[ -n "${HF_MODEL}" ]]; then
    extra+=(--hf-model "${HF_MODEL}")
  fi
  if [[ "${EVAL_AT_START}" == "1" || "${EVAL_AT_START}" == "true" || "${EVAL_AT_START}" == "yes" ]]; then
    extra+=(--eval-at-start)
  else
    extra+=(--no-eval-at-start)
  fi

  echo "== train v3 dataset=${DATASET} backbone=${backbone} prune_plan_name=${prune_plan_name} prune_plan_root=${plan_root} eval_at_start=${EVAL_AT_START} save_every_epoch=${SAVE_EVERY_EPOCH} =="
  conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.train.train_prune_lora_v3 \
    --experiment "${backbone}" \
    --dataset "${DATASET}" \
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

echo "== train v3 backbones=${BACKBONES} (n=${BB_N}) =="

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for prune_plan_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
    run_one_backbone "${bb}" "${prune_plan_name}"
  done
done

echo "== done =="
experiment_output_hint "Train"

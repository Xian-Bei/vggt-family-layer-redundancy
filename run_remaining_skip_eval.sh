#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN="${RUN:-}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
EXPECTED_DROP_COUNT="${EXPECTED_DROP_COUNT:-}"
BASE_SELECTION_PATH="${BASE_SELECTION_PATH:-}"
MAX_SCENES="${MAX_SCENES:-0}"
REMAINING_SKIP_EVAL_MAX_SPAN="${REMAINING_SKIP_EVAL_MAX_SPAN:-0}"
REMAINING_SKIP_EVAL_TIMING="${REMAINING_SKIP_EVAL_TIMING:-0}"
REMAINING_SKIP_EVAL_RESUME="${REMAINING_SKIP_EVAL_RESUME:-1}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
export OUTPUT_ROOT

# shellcheck source=run_backbones_lib.sh
source "${REPO_ROOT}/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${RUN}" "RUN" "${DATASET_KEYS[@]}"
RUN_LIST=("${PARSED_DATASETS[@]}")

if [[ "${#RUN_LIST[@]}" -eq 0 ]]; then
  echo "error: RUN is empty" >&2
  exit 1
fi
if [[ -z "${PRUNE_PLAN_ROOT//[[:space:]]/}" ]]; then
  echo "error: PRUNE_PLAN_ROOT is required" >&2
  exit 1
fi
if [[ -z "${PRUNE_PLAN_NAME//[[:space:]]/}" && -z "${EXPECTED_DROP_COUNT//[[:space:]]/}" ]]; then
  read -r PRUNE_PLAN_NAME EXPECTED_DROP_COUNT < <(
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m analysis.prune_plan.knee_point --root "${PRUNE_PLAN_ROOT}"
  )
elif [[ -z "${PRUNE_PLAN_NAME//[[:space:]]/}" || -z "${EXPECTED_DROP_COUNT//[[:space:]]/}" ]]; then
  echo "error: set both PRUNE_PLAN_NAME and EXPECTED_DROP_COUNT, or leave both empty to use prune_plan_all.json knee_points.ours" >&2
  exit 1
fi
if [[ -z "${PRUNE_PLAN_NAME//[[:space:]]/}" ]]; then
  echo "error: PRUNE_PLAN_NAME is required" >&2
  exit 1
fi
if [[ "${PRUNE_PLAN_NAME}" == *","* || "${PRUNE_PLAN_NAME}" =~ [[:space:]] ]]; then
  echo "error: PRUNE_PLAN_NAME must contain exactly one selector" >&2
  exit 1
fi
if [[ ! "${EXPECTED_DROP_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: EXPECTED_DROP_COUNT must be a positive integer" >&2
  exit 1
fi
if [[ "${DEVICE}" != cuda && "${DEVICE}" != cuda:* ]]; then
  echo "error: remaining skip evaluation requires DEVICE=cuda or DEVICE=cuda:<index>" >&2
  exit 1
fi

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

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local split="${4:-}"
  local output_dataset="${5:-}"
  set_backend_pythonpath "${backbone}"

  local extra=(
    --experiment "${backbone}"
    --dataset "${dataset}"
    --data-root "${data_root}"
    --backbone "${backbone}"
    --device "${DEVICE}"
    --max-scenes "${MAX_SCENES}"
    --max-frames "${MAX_FRAMES}"
    --max-span "${REMAINING_SKIP_EVAL_MAX_SPAN}"
    --prune-plan-root "${PRUNE_PLAN_ROOT}"
    --prune-plan-name "${PRUNE_PLAN_NAME}"
    --expected-drop-count "${EXPECTED_DROP_COUNT}"
  )
  if [[ -n "${BASE_SELECTION_PATH//[[:space:]]/}" ]]; then
    extra+=(--base-selection-path "${BASE_SELECTION_PATH}")
  fi
  if [[ -n "${split}" ]]; then
    extra+=(--split "${split}")
  fi
  if [[ -n "${output_dataset}" ]]; then
    extra+=(--output-dataset "${output_dataset}")
  fi
  if truthy "${REMAINING_SKIP_EVAL_TIMING}"; then
    extra+=(--timing)
  fi
  if ! truthy "${REMAINING_SKIP_EVAL_RESUME}"; then
    extra+=(--no-resume)
  fi

  if truthy "${COMPACT_LOG:-0}"; then
    local span_scope="1-${REMAINING_SKIP_EVAL_MAX_SPAN}"
    if [[ "${REMAINING_SKIP_EVAL_MAX_SPAN}" == "0" ]]; then
      span_scope="all"
    fi
    printf '\n  remaining skip evaluation\n    model: %s\n    prune plan: %s\n    dropped units: %s\n    spans over remaining units: %s\n    dataset: %s\n\n' \
      "${backbone}" "${PRUNE_PLAN_NAME}" "${EXPECTED_DROP_COUNT}" "${span_scope}" \
      "${output_dataset:-${dataset}${split:+_${split}}}"
  else
    echo "== remaining_skip_eval dataset=${output_dataset:-${dataset}${split:+_${split}}} backbone=${backbone} prune_plan=${PRUNE_PLAN_NAME} max_span=${REMAINING_SKIP_EVAL_MAX_SPAN} resume=${REMAINING_SKIP_EVAL_RESUME} =="
  fi

  run_progress_command conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m analysis.redundancy.remaining_skip_gt_eval_v3 "${extra[@]}"
}

run_dataset_key() {
  local key="$1"
  local backbone="$2"
  case "${key}" in
    SCANNET_DEV50)
      run_one_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "" "scannet_dev50"
      ;;
    SCANNET_TEST50)
      run_one_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "" "scannet_test50"
      ;;
    7SCENES_TRAIN)
      run_one_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "train"
      ;;
    7SCENES_TEST)
      run_one_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "test"
      ;;
    NRGBD)
      run_one_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}"
      ;;
    NUSCENES_VAL)
      run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "val"
      ;;
    NUSCENES_TRAIN)
      run_one_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "train"
      ;;
    WAYMO_VALIDATION)
      run_one_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "validation"
      ;;
    WAYMO_TRAINING)
      run_one_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "training"
      ;;
  esac
}

for backbone in "${BB_LIST[@]}"; do
  backbone="${backbone//[[:space:]]/}"
  [[ -z "${backbone}" ]] && continue
  for dataset_key in "${RUN_LIST[@]}"; do
    run_dataset_key "${dataset_key}" "${backbone}"
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Remaining skip evaluation"
fi

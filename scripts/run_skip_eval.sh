#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BACKBONES="${BACKBONES:-vggt}"
EVAL_DATASET="${EVAL_DATASET:-}"

MAX_SCENES="${MAX_SCENES:-0}"
SKIP_GT_EVAL_MAX_SPAN="${SKIP_GT_EVAL_MAX_SPAN:-0}"
SKIP_GT_EVAL_TIMING="${SKIP_GT_EVAL_TIMING:-0}"
SKIP_GT_EVAL_RESUME="${SKIP_GT_EVAL_RESUME:-1}"

DEVICE="${DEVICE:-cuda}"

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
parse_backbones_list
parse_dataset_list "${EVAL_DATASET}" "EVAL_DATASET" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

if [[ "${#EVAL_LIST[@]}" -eq 0 ]]; then
  echo "error: EVAL_DATASET is empty; expected one of: $(dataset_keys_help "${DATASET_KEYS[@]}")" >&2
  exit 1
fi

run_skip_eval_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local output_dataset="${4:-${dataset}}"
  local split="${5:-}"

  set_backend_pythonpath "${backbone}"

  local extra=(
    --experiment "${backbone}"
    --dataset "${dataset}"
    --data-root "${data_root}"
    --backbone "${backbone}"
    --output-dataset "${output_dataset}"
    --device "${DEVICE}"
    --max-scenes "${MAX_SCENES}"
    --max-frames "${MAX_FRAMES}"
    --max-span "${SKIP_GT_EVAL_MAX_SPAN}"
    --score-workers "${ASYNC_SCORE_WORKERS}"
    --score-max-pending "${ASYNC_SCORE_MAX_PENDING}"
  )
  if truthy "${SKIP_GT_EVAL_TIMING}"; then
    extra+=(--timing)
  fi
  if ! truthy "${SKIP_GT_EVAL_RESUME}"; then
    extra+=(--no-resume)
  fi
  if [[ -n "${split}" ]]; then
    extra+=(--split "${split}")
  fi

  if truthy "${COMPACT_LOG:-0}"; then
    local span_scope="1-${SKIP_GT_EVAL_MAX_SPAN}"
    if [[ "${SKIP_GT_EVAL_MAX_SPAN}" == "0" ]]; then
      span_scope="all"
    fi
    printf '\n  skip evaluation\n    model: %s\n    spans: %s\n    score workers: %s\n    max pending: %s\n    dataset: %s\n\n' \
      "${backbone}" "${span_scope}" "${ASYNC_SCORE_WORKERS}" \
      "${ASYNC_SCORE_MAX_PENDING}" "${output_dataset}"
  else
    echo "== skip_gt_eval dataset=${output_dataset} backbone=${backbone} max_span=${SKIP_GT_EVAL_MAX_SPAN} resume=${SKIP_GT_EVAL_RESUME} score_workers=${ASYNC_SCORE_WORKERS} max_pending=${ASYNC_SCORE_MAX_PENDING} =="
  fi
  run_command python -m analysis.redundancy.skip_gt_eval_v3 \
    "${extra[@]}"
}

run_dataset_key() {
  local key="$1"
  local backbone="$2"
  case "${key}" in
    SCANNET_DEV50)
      run_skip_eval_dataset "scannet" "${ROOT_SCANNET_DEV50}" "${backbone}" "scannet_dev50"
      ;;
    SCANNET_TEST50)
      run_skip_eval_dataset "scannet" "${ROOT_SCANNET_TEST50}" "${backbone}" "scannet_test50"
      ;;
    7SCENES_TEST)
      run_skip_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "7scenes_test" "test"
      ;;
    7SCENES_TRAIN)
      run_skip_eval_dataset "7scenes" "${ROOT_7SCENES}" "${backbone}" "7scenes_train" "train"
      ;;
    NRGBD)
      run_skip_eval_dataset "nrgbd" "${ROOT_NRGBD}" "${backbone}" "nrgbd"
      ;;
    NUSCENES_VAL)
      run_skip_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "nuscenes_val" "val"
      ;;
    NUSCENES_TRAIN)
      run_skip_eval_dataset "nuscenes" "${ROOT_NUSCENES}" "${backbone}" "nuscenes_train" "train"
      ;;
    WAYMO_VALIDATION)
      run_skip_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "waymo_validation" "validation"
      ;;
    WAYMO_TRAINING)
      run_skip_eval_dataset "waymo" "${ROOT_WAYMO}" "${backbone}" "waymo_training" "training"
      ;;
  esac
}

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== skip_gt_eval v3 backbones=${BACKBONES} (n=${BB_N}) datasets=$(join_by_comma "${EVAL_LIST[@]}") MAX_FRAMES=${MAX_FRAMES} MAX_SPAN=${SKIP_GT_EVAL_MAX_SPAN} SCORE_WORKERS=${ASYNC_SCORE_WORKERS} MAX_PENDING=${ASYNC_SCORE_MAX_PENDING} =="
fi

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for dataset_key in "${EVAL_LIST[@]}"; do
    run_dataset_key "${dataset_key}" "${bb}"
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Skip GT eval"
fi

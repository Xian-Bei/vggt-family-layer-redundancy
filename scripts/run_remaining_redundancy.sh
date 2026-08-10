#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN="${RUN:-}"
PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
EXPECTED_DROP_COUNT="${EXPECTED_DROP_COUNT:-}"
BASE_SELECTION_PATH="${BASE_SELECTION_PATH:-}"
MAX_SCENES="${MAX_SCENES:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_TOKENS="${MAX_TOKENS:-0}"
METRICS="${METRICS:-cka}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special}"
REDUNDANCY_SEED="${REDUNDANCY_SEED:-0}"
REUSE_COMPLETE_REMAINING_REDUNDANCY="${REUSE_COMPLETE_REMAINING_REDUNDANCY:-1}"
export OUTPUT_ROOT

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
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
if [[ -z "${PRUNE_PLAN_NAME//[[:space:]]/}" || -z "${EXPECTED_DROP_COUNT//[[:space:]]/}" ]]; then
  echo "error: PRUNE_PLAN_NAME and EXPECTED_DROP_COUNT are required" >&2
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
  echo "error: remaining redundancy requires DEVICE=cuda or DEVICE=cuda:<index>" >&2
  exit 1
fi

remaining_output_complete() {
  local backbone="$1"
  local output_dataset="$2"
  local manifest_path="${OUTPUT_ROOT}/${backbone}/${output_dataset}/remaining_redundancy/manifest.json"
  [[ -f "${manifest_path}" ]] || return 1
  python -c '
import hashlib, json, pathlib, sys
manifest_path, selector, count, metrics, regions, max_tokens, seed, max_frames, selection_path = sys.argv[1:]
root = pathlib.Path(manifest_path).parent
payload = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
expected_metrics = [item for item in metrics.split(",") if item]
expected_regions = [item for item in regions.split(",") if item]
available_metrics = payload.get("metrics", [])
valid = (
    payload.get("prune_plan_name") == selector
    and int(payload.get("expected_drop_count", -1)) == int(count)
    and all(metric in available_metrics for metric in expected_metrics)
    and payload.get("token_regions") == expected_regions
    and int(payload.get("max_tokens", -1)) == int(max_tokens)
    and int(payload.get("seed", -1)) == int(seed)
    and int(payload.get("max_frames", -1)) == int(max_frames)
    and bool(payload.get("scene_ids"))
    and len(payload.get("scenes", [])) == len(payload.get("scene_ids", []))
    and all((root / f"redundancy_graph_{region}.npz").is_file() for region in expected_regions)
)
if selection_path:
    raw = pathlib.Path(selection_path).read_bytes()
    valid = valid and payload.get("selection_sha256") == hashlib.sha256(raw).hexdigest()
raise SystemExit(0 if valid else 1)
' \
      "${manifest_path}" "${PRUNE_PLAN_NAME}" "${EXPECTED_DROP_COUNT}" \
      "${METRICS}" "${SIM_TOKEN_REGIONS}" "${MAX_TOKENS}" \
      "${REDUNDANCY_SEED}" "${MAX_FRAMES}" "${BASE_SELECTION_PATH}" \
      >/dev/null 2>&1
}

run_one_dataset() {
  local dataset="$1"
  local data_root="$2"
  local backbone="$3"
  local split="${4:-}"
  local output_dataset="${5:-}"
  set_backend_pythonpath "${backbone}"

  local split_args=()
  if [[ -n "${split}" ]]; then
    split_args=(--split "${split}")
  fi
  local output_dataset_args=()
  if [[ -n "${output_dataset}" ]]; then
    output_dataset_args=(--output-dataset "${output_dataset}")
  fi
  local selection_args=()
  if [[ -n "${BASE_SELECTION_PATH}" ]]; then
    selection_args=(--base-selection-path "${BASE_SELECTION_PATH}")
  fi

  if truthy "${REUSE_COMPLETE_REMAINING_REDUNDANCY}" \
      && remaining_output_complete "${backbone}" "${output_dataset:-${dataset}${split:+_${split}}}"; then
    printf '  [reuse] %s/%s: remaining redundancy is already complete\n' \
      "${backbone}" "${output_dataset:-${dataset}${split:+_${split}}}"
    return
  fi

  if truthy "${COMPACT_LOG:-0}"; then
    printf '\n  remaining redundancy\n    model: %s\n    prune plan: %s\n    dropped units: %s\n    metrics: %s\n    token regions: %s\n    dataset: %s\n\n' \
      "${backbone}" "${PRUNE_PLAN_NAME}" "${EXPECTED_DROP_COUNT}" "${METRICS}" \
      "${SIM_TOKEN_REGIONS}" "${output_dataset:-${dataset}${split:+_${split}}}"
  else
    echo "== remaining redundancy dataset=${dataset}${split:+_${split}}${output_dataset:+ -> ${output_dataset}} backbone=${backbone} prune_plan=${PRUNE_PLAN_NAME} expected_drop_count=${EXPECTED_DROP_COUNT} =="
  fi

  run_command python -m analysis.redundancy.remaining_redundancy_v3 \
      --experiment "${backbone}" \
      --dataset "${dataset}" \
      "${output_dataset_args[@]}" \
      --data-root "${data_root}" \
      --backbone "${backbone}" \
      --device "${DEVICE}" \
      --max-scenes "${MAX_SCENES}" \
      --max-frames "${MAX_FRAMES}" \
      --seed "${REDUNDANCY_SEED}" \
      --metrics "${METRICS}" \
      --max-tokens "${MAX_TOKENS}" \
      --token-regions "${SIM_TOKEN_REGIONS}" \
      --prune-plan-root "${PRUNE_PLAN_ROOT}" \
      --prune-plan-name "${PRUNE_PLAN_NAME}" \
      --expected-drop-count "${EXPECTED_DROP_COUNT}" \
      "${selection_args[@]}" \
      "${split_args[@]}"
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

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  for dataset_key in "${RUN_LIST[@]}"; do
    run_dataset_key "${dataset_key}" "${bb}"
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Remaining redundancy"
fi

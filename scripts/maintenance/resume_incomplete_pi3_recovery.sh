#!/usr/bin/env bash
set -euo pipefail

# Temporary repair runner for incomplete Pi3 recovery fits/evaluations.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "vggt" ]]; then
  echo "error: activate the vggt conda environment before running this script" >&2
  echo "  conda activate vggt" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"

BACKBONE="pi3"
FIT_DATASETS="scannet_dev50,nuscenes_train"
EVAL_DATASETS="scannet_dev50,nuscenes_train,scannet_test50,7scenes_test,nuscenes_val,waymo_validation"
RECOVERY_PRUNE_STRATEGIES="retention_single,retention_double,CKA_single,CKA_double"
RECOVERY_PRUNE_BUDGETS="prune_span10,prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24"
MIN_SCRATCH_GIB="${MIN_SCRATCH_GIB:-45}"
DEVICE="${DEVICE:-cuda}"

if [[ "${OUTPUT_ROOT}" = /* ]]; then
  OUTPUT_BASE="${OUTPUT_ROOT}"
else
  OUTPUT_BASE="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
PLAN_ROOT="${OUTPUT_BASE}/${BACKBONE}"
SCRATCH_ROOT="${TMPDIR:-/tmp}"
CACHE_DIRS=(
  "${PLAN_ROOT}/scannet_dev50/cache"
  "${PLAN_ROOT}/nuscenes_train/cache"
)

if [[ ! -d "${SCRATCH_ROOT}" || ! -w "${SCRATCH_ROOT}" ]]; then
  echo "error: scratch directory is not writable: ${SCRATCH_ROOT}" >&2
  exit 1
fi

cleanup_cache() {
  local cache_dir
  for cache_dir in "${CACHE_DIRS[@]}"; do
    case "${cache_dir}" in
      "${PLAN_ROOT}/scannet_dev50/cache"|"${PLAN_ROOT}/nuscenes_train/cache")
        rm -rf -- "${cache_dir}"
        ;;
      *)
        echo "error: refusing to remove unexpected cache path: ${cache_dir}" >&2
        exit 1
        ;;
    esac
  done
}

available_gib() {
  df -Pk "${SCRATCH_ROOT}" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'
}

checkpoint_path() {
  local selector="$1"
  local method="$2"
  printf '%s/recovery/%s/%s/checkpoints/latest.pt\n' \
    "${PLAN_ROOT}" "${selector}" "${method}"
}

eval_path() {
  local selector="$1"
  local method="$2"
  local dataset="$3"
  printf '%s/%s/eval/recovery/%s/%s/eval_record_v3.json\n' \
    "${PLAN_ROOT}" "${dataset}" "${selector}" "${method}"
}

method_eval_complete() {
  local selector="$1"
  local method="$2"
  local dataset
  for dataset in ${EVAL_DATASETS//,/ }; do
    [[ -f "$(eval_path "${selector}" "${method}" "${dataset}")" ]] || return 1
  done
}

validate_selector() {
  local selector="$1"
  local method dataset missing=0
  for method in "${RECOVERY_METHODS[@]}"; do
    if [[ ! -f "$(checkpoint_path "${selector}" "${method}")" ]]; then
      echo "error: missing checkpoint: $(checkpoint_path "${selector}" "${method}")" >&2
      missing=1
    fi
    for dataset in ${EVAL_DATASETS//,/ }; do
      if [[ ! -f "$(eval_path "${selector}" "${method}" "${dataset}")" ]]; then
        echo "error: missing eval: ${dataset}/${selector}/${method}" >&2
        missing=1
      fi
    done
  done
  [[ "${missing}" -eq 0 ]]
}

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is not available in the vggt conda environment")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY

if [[ ! -f "${PLAN_ROOT}/prune_plan_all.json" || ! -f "${PLAN_ROOT}/recovery_plan_all.json" ]]; then
  echo "error: recovery/prune plan bundles are missing below ${PLAN_ROOT}" >&2
  exit 1
fi

mapfile -t SELECTORS < <(
  python -m analysis.recovery.shared_results selectors-for-strategies \
    --plan-root "${PLAN_ROOT}" \
    --strategies "${RECOVERY_PRUNE_STRATEGIES}" \
    --selectors "${RECOVERY_PRUNE_BUDGETS}" | tr ',' '\n'
)
mapfile -t RECOVERY_METHODS < <(
  python -m analysis.recovery.shared_results recovery-plan-names \
    --plan-root "${PLAN_ROOT}" | tr ',' '\n'
)

if [[ "${#SELECTORS[@]}" -eq 0 || "${#RECOVERY_METHODS[@]}" -eq 0 ]]; then
  echo "error: no Pi3 selectors or recovery methods were discovered" >&2
  exit 1
fi

echo "Checking ${#SELECTORS[@]} Pi3 selectors and ${#RECOVERY_METHODS[@]} recovery methods."
echo "Scratch filesystem currently has $(available_gib) GiB available."

for selector in "${SELECTORS[@]}"; do
  needed_methods=()
  fit_methods=()
  for method in "${RECOVERY_METHODS[@]}"; do
    checkpoint="$(checkpoint_path "${selector}" "${method}")"
    if [[ ! -f "${checkpoint}" ]]; then
      needed_methods+=("${method}")
      fit_methods+=("${method}")
    elif ! method_eval_complete "${selector}" "${method}"; then
      needed_methods+=("${method}")
    fi
  done

  echo
  echo "===== selector ${selector} ====="
  if [[ "${#needed_methods[@]}" -eq 0 ]]; then
    echo "Already complete; skipping ${selector}."
    continue
  fi

  needed_csv="$(IFS=,; echo "${needed_methods[*]}")"
  cache_created=0

  if [[ "${#fit_methods[@]}" -gt 0 ]]; then
    fit_csv="$(IFS=,; echo "${fit_methods[*]}")"
    echo "Missing fits: ${fit_csv}"

    # Cache one selector at a time. This avoids materializing the union cache
    # for all Pi3 selectors and keeps enough room for sequential scratch data.
    cleanup_cache
    BACKBONES="${BACKBONE}" \
      RUN="${FIT_DATASETS}" \
      CACHE="${FIT_DATASETS}" \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES=0 \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${PLAN_ROOT}" \
      RECOVERY_PLAN_ROOT="${PLAN_ROOT}" \
      PRUNE_PLAN_NAME="${selector}" \
      RECOVERY_PLAN_NAME="${fit_csv}" \
      bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
    cache_created=1

    free_gib="$(available_gib)"
    echo "Scratch filesystem has ${free_gib} GiB available after caching."
    if (( free_gib < MIN_SCRATCH_GIB )); then
      echo "error: need at least ${MIN_SCRATCH_GIB} GiB free after caching; cache is preserved" >&2
      exit 1
    fi
  else
    echo "All checkpoints exist; only missing evaluations will be run."
  fi

  BACKBONES="${BACKBONE}" \
    DATASETS="${FIT_DATASETS}" \
    PRUNE_PLAN_ROOT="${PLAN_ROOT}" \
    RECOVERY_PLAN_ROOT="${PLAN_ROOT}" \
    PRUNE_PLAN_NAME="${selector}" \
    RECOVERY_PLAN_NAME="${needed_csv}" \
    RECOVERY_FORCE_RERUN=0 \
    EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
    EVAL="${EVAL_DATASETS}" \
    EVAL_MAX_SCENES=0 \
    MAX_SCENES=0 \
    DEVICE="${DEVICE}" \
    bash "${REPO_ROOT}/scripts/run_recovery.sh"

  # The recovery driver can proceed after a batched fit failure, so validate
  # every configured Pi3 method and dataset before deleting the cache.
  if ! validate_selector "${selector}"; then
    echo "error: ${selector} is incomplete; any newly built cache is preserved" >&2
    exit 1
  fi

  if [[ "${cache_created}" -eq 1 ]]; then
    cleanup_cache
    echo "Completed ${selector}; temporary activation cache removed."
  else
    echo "Completed missing evaluations for ${selector}."
  fi
done

echo
echo "All configured Pi3 recovery fits and evaluations are complete."

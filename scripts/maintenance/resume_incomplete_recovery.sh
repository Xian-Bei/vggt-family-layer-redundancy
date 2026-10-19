#!/usr/bin/env bash
set -euo pipefail

# Temporary repair runner for the recovery jobs that were skipped after the
# sequential ReplaceMe scratch-space failure on 2026-09-25.

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

BACKBONE="vggt_omega"
FIT_DATASETS="scannet_dev50,nuscenes_train"
EVAL_DATASETS="scannet_dev50,nuscenes_train,scannet_test50,7scenes_test,nuscenes_val,waymo_validation"
RECOVERY_METHODS="recovery_replaceme_cos,recovery_replaceme_ls,recovery_streamline_ffn"
MIN_SCRATCH_GIB="${MIN_SCRATCH_GIB:-45}"
DEVICE="${DEVICE:-cuda}"

SELECTORS=(
  i1_j17_i41_j45
  i1_j17_i39_j45
  i6_j18_i45_j45
  i6_j18_i43_j45
  i6_j18_i41_j45
  i6_j18_i39_j45
  i3_j17_i39_j45
  i2_j18_i39_j45
)

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

selector_complete() {
  local selector="$1"
  local method dataset
  for method in ${RECOVERY_METHODS//,/ }; do
    [[ -f "$(checkpoint_path "${selector}" "${method}")" ]] || return 1
    for dataset in ${EVAL_DATASETS//,/ }; do
      [[ -f "${PLAN_ROOT}/${dataset}/eval/recovery/${selector}/${method}/eval_record_v3.json" ]] || return 1
    done
  done
}

validate_selector() {
  local selector="$1"
  local method dataset missing=0
  for method in ${RECOVERY_METHODS//,/ }; do
    if [[ ! -f "$(checkpoint_path "${selector}" "${method}")" ]]; then
      echo "error: missing checkpoint: $(checkpoint_path "${selector}" "${method}")" >&2
      missing=1
    fi
    for dataset in ${EVAL_DATASETS//,/ }; do
      if [[ ! -f "${PLAN_ROOT}/${dataset}/eval/recovery/${selector}/${method}/eval_record_v3.json" ]]; then
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

echo "Repairing only the 8 incomplete selectors and 3 missing recovery methods."
echo "Scratch filesystem currently has $(available_gib) GiB available."

for selector in "${SELECTORS[@]}"; do
  echo
  echo "===== selector ${selector} ====="

  if selector_complete "${selector}"; then
    echo "Already complete; skipping ${selector}."
    continue
  fi

  # Cache one selector at a time to avoid recreating the previous ~full-disk
  # union cache. Remove only the two known calibration-cache directories.
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
    RECOVERY_PLAN_NAME="${RECOVERY_METHODS}" \
    bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"

  free_gib="$(available_gib)"
  echo "Scratch filesystem has ${free_gib} GiB available after caching."
  if (( free_gib < MIN_SCRATCH_GIB )); then
    echo "error: need at least ${MIN_SCRATCH_GIB} GiB free after caching; cache is preserved" >&2
    exit 1
  fi

  BACKBONES="${BACKBONE}" \
    DATASETS="${FIT_DATASETS}" \
    PRUNE_PLAN_ROOT="${PLAN_ROOT}" \
    RECOVERY_PLAN_ROOT="${PLAN_ROOT}" \
    PRUNE_PLAN_NAME="${selector}" \
    RECOVERY_PLAN_NAME="${RECOVERY_METHODS}" \
    RECOVERY_FORCE_RERUN=0 \
    EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
    EVAL="${EVAL_DATASETS}" \
    EVAL_MAX_SCENES=0 \
    MAX_SCENES=0 \
    DEVICE="${DEVICE}" \
    bash "${REPO_ROOT}/scripts/run_recovery.sh"

  # run_recovery.sh may continue to evaluation after a batched fit failure, so
  # explicitly verify every expected artifact before deleting the cache.
  if ! validate_selector "${selector}"; then
    echo "error: ${selector} is incomplete; cache is preserved for diagnosis/retry" >&2
    exit 1
  fi

  cleanup_cache
  echo "Completed ${selector}; temporary activation cache removed."
done

echo
echo "All targeted recovery fits and evaluations are complete."

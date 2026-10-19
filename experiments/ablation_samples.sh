#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# N is the final calibration count *per domain*. The fixed 50-scene main pool
# is used first (only its first N scenes when N < 50); then use the extra pool.
SAMPLE_COUNTS="${SAMPLE_COUNTS:-5}"
BASE_SCENES_PER_DOMAIN="${BASE_SCENES_PER_DOMAIN:-50}"
SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-analysis_outputs_v3}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-analysis_outputs_v3_samples_ablation}"
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY:-retention_single,retention_double}"
RECOVERY_PRUNE_PLANS="${RECOVERY_PRUNE_PLANS:-prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24}"
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_ghost_interval_sequential_autogroup}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_FRAMES="${MAX_FRAMES:-30}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"
RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
CLEAN_CACHE_BEFORE="${CLEAN_CACHE_BEFORE:-1}"
CLEAN_CACHE_AFTER="${CLEAN_CACHE_AFTER:-1}"
DRY_RUN="${DRY_RUN:-0}"
CACHE_ONLY="${CACHE_ONLY:-0}"

SCANNET_BASE_LIST="${SCANNET_BASE_LIST:-data/scannet_dev50.txt}"
SCANNET_TEST_LIST="${SCANNET_TEST_LIST:-data/scannet_test50.txt}"
ROOT_SCANNET_DEV_EXTRA100="${ROOT_SCANNET_DEV_EXTRA100:-data/scans_dev_extra100}"
NUSCENES_BASE_MANIFEST="${NUSCENES_BASE_MANIFEST:-data/nuscenes/selected_scenes.json}"
ROOT_NUSCENES_TRAIN_EXTRA100="${ROOT_NUSCENES_TRAIN_EXTRA100:-data/nuscenes_train_extra100}"

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
parse_backbones_list
ABLATION_BACKBONES=("${BB_LIST[@]}")

truthy_local() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

run_step() {
  if truthy_local "${DRY_RUN}"; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

sample_values=()
max_samples=0
for raw in ${SAMPLE_COUNTS//,/ }; do
  value="${raw//[[:space:]]/}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: SAMPLE_COUNTS must contain positive integers; got '${raw}'" >&2
    exit 1
  fi
  if (( value > BASE_SCENES_PER_DOMAIN + 100 )); then
    echo "error: sample count ${value} is outside [1, $((BASE_SCENES_PER_DOMAIN + 100))]" >&2
    exit 1
  fi
  if [[ ! " ${sample_values[*]} " =~ " ${value} " ]]; then
    sample_values+=("${value}")
  fi
  (( value > max_samples )) && max_samples="${value}"
done
if [[ "${#sample_values[@]}" -eq 0 ]]; then
  echo "error: SAMPLE_COUNTS is empty" >&2
  exit 1
fi
max_extra=$((max_samples - BASE_SCENES_PER_DOMAIN))
if (( max_extra < 0 )); then max_extra=0; fi
cache_root="${ABLATION_OUTPUT_ROOT}/cache"

# Only remove this ablation's selected backbone caches, never result folders.
# CACHE_ONLY callers should set CLEAN_CACHE_AFTER=0 to retain the built cache.
cleanup_cache() {
  local phase="$1" backbone target resolved_root resolved_cache
  resolved_root="$(realpath -m -- "${ABLATION_OUTPUT_ROOT}")" || return
  resolved_cache="$(realpath -m -- "${cache_root}")" || return
  if [[ "${resolved_root}" == / || "${resolved_root}" == "${REPO_ROOT}" ||
        "${resolved_root}" == "$(realpath -m -- "${SOURCE_OUTPUT_ROOT}")" ||
        "${resolved_cache}" != "${resolved_root}/cache" ]]; then
    echo "error: refusing unsafe cache cleanup path: ${cache_root}" >&2
    return 1
  fi
  for backbone in "${ABLATION_BACKBONES[@]}"; do
    [[ "${backbone}" =~ ^[a-zA-Z0-9_]+$ ]] || return 1
    target="${resolved_cache}/${backbone}"
    if [[ -L "${target}" ]]; then
      echo "error: refusing symlink cache cleanup: ${target}" >&2
      return 1
    fi
    echo "[cache] cleanup ${phase}: ${target}"
    run_step rm -rf --one-file-system -- "${target}" || return
  done
}

on_exit() {
  local status=$?
  trap - EXIT
  if truthy_local "${CLEAN_CACHE_AFTER}"; then
    if ! cleanup_cache after; then
      echo "error: exit cache cleanup failed" >&2
      if (( status == 0 )); then status=1; fi
    fi
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if truthy_local "${CLEAN_CACHE_BEFORE}"; then
  cleanup_cache before
fi

echo "== calibration-sample ablation =="
echo "samples per domain: ${sample_values[*]}"
echo "base per domain: ${BASE_SCENES_PER_DOMAIN}; cached extra per domain: ${max_extra}"
echo "source plans: ${SOURCE_OUTPUT_ROOT}"
echo "outputs: ${ABLATION_OUTPUT_ROOT}"

if ! truthy_local "${DRY_RUN}"; then
  python -m analysis.results.samples_ablation validate-data \
    --base-count "${BASE_SCENES_PER_DOMAIN}" \
    --extra-needed "${max_extra}" \
    --scannet-base-list "${SCANNET_BASE_LIST}" \
    --scannet-test-list "${SCANNET_TEST_LIST}" \
    --scannet-extra-root "${ROOT_SCANNET_DEV_EXTRA100}" \
    --nuscenes-base-manifest "${NUSCENES_BASE_MANIFEST}" \
    --nuscenes-extra-root "${ROOT_NUSCENES_TRAIN_EXTRA100}"
  python -m analysis.results.samples_ablation init-manifest \
    --output-root "${ABLATION_OUTPUT_ROOT}" \
    --source-output-root "${SOURCE_OUTPUT_ROOT}" \
    --samples "$(IFS=,; echo "${sample_values[*]}")" \
    --base-count "${BASE_SCENES_PER_DOMAIN}" \
    --max-extra "${max_extra}" \
    --backbones "${BACKBONES}" \
    --methods "${RECOVERY_METHODS}" \
    --eval-datasets "${EVAL_DATASETS}"
fi

for bb in "${ABLATION_BACKBONES[@]}"; do
  plan_root="${SOURCE_OUTPUT_ROOT}/${bb}"
  if truthy_local "${DRY_RUN}"; then
    recovery_plans="<filtered recovery plans from ${plan_root}>"
    recovery_prune_plans="<selectors from ${plan_root}>"
  else
    recovery_plans="$(python -m analysis.results.samples_ablation recovery-names \
      --plan-root "${plan_root}" --methods "${RECOVERY_METHODS}")"
    recovery_prune_plans="$(python -m analysis.recovery.shared_results selectors-for-strategies \
      --plan-root "${plan_root}" \
      --strategies "${RECOVERY_PRUNE_STRATEGY}" \
      --selectors "${RECOVERY_PRUNE_PLANS}")"
  fi

  expected_cache=(
    --expect "scannet_dev50=${BASE_SCENES_PER_DOMAIN}"
    --expect "nuscenes_train=${BASE_SCENES_PER_DOMAIN}"
  )
  if (( max_extra > 0 )); then
    expected_cache+=(
      --expect "scannet_dev_extra100=${max_extra}"
      --expect "nuscenes_train_extra100=${max_extra}"
    )
  fi

  cache_valid=0
  if ! truthy_local "${DRY_RUN}" && ! truthy_local "${REBUILD_CACHE}"; then
    if python -m analysis.results.samples_ablation validate-cache \
      --cache-root "${cache_root}" --backbone "${bb}" "${expected_cache[@]}" 2>/dev/null; then
      cache_valid=1
      echo "[cache] reuse ${cache_root}/${bb}"
    fi
  fi
  if (( cache_valid == 0 )); then
    if truthy_local "${REBUILD_CACHE}" && [[ -d "${cache_root}/${bb}" ]]; then
      run_step rm -rf --one-file-system -- "${cache_root:?}/${bb}"
    fi
    run_step env \
      OUTPUT_ROOT="${cache_root}" BACKBONES="${bb}" \
      RUN="scannet_dev50,nuscenes_train" CACHE="scannet_dev50,nuscenes_train" \
      SKIP_REDUNDANCY=1 MAX_SCENES="${BASE_SCENES_PER_DOMAIN}" MAX_FRAMES="${MAX_FRAMES}" \
      DEVICE="${DEVICE}" PRUNE_PLAN_ROOT="${plan_root}" RECOVERY_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" RECOVERY_PLAN_NAME="${recovery_plans}" \
      bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
    if (( max_extra > 0 )); then
      run_step env \
        OUTPUT_ROOT="${cache_root}" BACKBONES="${bb}" \
        RUN="scannet_dev_extra100,nuscenes_train_extra100" \
        CACHE="scannet_dev_extra100,nuscenes_train_extra100" \
        SKIP_REDUNDANCY=1 MAX_SCENES="${max_extra}" MAX_FRAMES="${MAX_FRAMES}" \
        DEVICE="${DEVICE}" PRUNE_PLAN_ROOT="${plan_root}" RECOVERY_PLAN_ROOT="${plan_root}" \
        PRUNE_PLAN_NAME="${recovery_prune_plans}" RECOVERY_PLAN_NAME="${recovery_plans}" \
        bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
    fi
    if ! truthy_local "${DRY_RUN}"; then
      python -m analysis.results.samples_ablation validate-cache \
        --cache-root "${cache_root}" --backbone "${bb}" "${expected_cache[@]}"
    fi
  fi

  if truthy_local "${CACHE_ONLY}"; then
    continue
  fi
  for sample_count in "${sample_values[@]}"; do
    base_count="${BASE_SCENES_PER_DOMAIN}"
    if (( sample_count < base_count )); then base_count="${sample_count}"; fi
    extra_count=$((sample_count - base_count))
    datasets="scannet_dev50,nuscenes_train"
    scene_limits="scannet_dev50=${base_count},nuscenes_train=${base_count}"
    if (( extra_count > 0 )); then
      datasets="scannet_dev50,scannet_dev_extra100,nuscenes_train,nuscenes_train_extra100"
      scene_limits="scannet_dev50=${BASE_SCENES_PER_DOMAIN},scannet_dev_extra100=${extra_count},nuscenes_train=${BASE_SCENES_PER_DOMAIN},nuscenes_train_extra100=${extra_count}"
    fi
    sample_root="${ABLATION_OUTPUT_ROOT}/samples_$(printf '%03d' "${sample_count}")"
    run_step env \
      OUTPUT_ROOT="${sample_root}" BACKBONES="${bb}" DATASETS="${datasets}" \
      PRUNE_PLAN_ROOT="${plan_root}" RECOVERY_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${recovery_prune_plans}" RECOVERY_PLAN_NAME="${recovery_plans}" \
      RECOVERY_CACHE_ROOT="${cache_root}" RECOVERY_SCENE_LIMITS="${scene_limits}" \
      MAX_SCENES=0 MAX_FRAMES="${MAX_FRAMES}" EVAL_MAX_SCENES="${EVAL_MAX_SCENES}" \
      DEVICE="${DEVICE}" DTYPE="${DTYPE}" RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN}" \
      EVAL_BOUNDS_ROOT="${SOURCE_OUTPUT_ROOT}" EVAL="${EVAL_DATASETS}" \
      RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY}" \
      bash "${REPO_ROOT}/scripts/run_recovery.sh"
  done
done

if ! truthy_local "${DRY_RUN}" && ! truthy_local "${CACHE_ONLY}"; then
  python -m analysis.results.samples_ablation summarize --output-root "${ABLATION_OUTPUT_ROOT}"
fi

echo "== sample ablation done: ${ABLATION_OUTPUT_ROOT} =="

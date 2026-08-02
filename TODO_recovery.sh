#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
ALL_BACKBONES_SPACE_SEPARATED="${ALL_BACKBONES//,/ }"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-vggt}"
MAX_SCENES="${MAX_SCENES:-0}"
export OUTPUT_ROOT
export COMPACT_LOG="${COMPACT_LOG:-1}"

compact_log() {
  [[ "${COMPACT_LOG}" == "1" || "${COMPACT_LOG}" == "true" || "${COMPACT_LOG}" == "yes" ]]
}

stage_log() {
  if compact_log; then
    printf '\n[%s] %s\n' "$1" "$2"
  fi
}

run_command() {
  if compact_log; then
    "$@" >/dev/null
  else
    "$@"
  fi
}



# 8. Build recovery-plan bundles from prune_plan_all.json.
#    Recovery plans decide cache units and either trainable units or closed-form boundary operators.
#    Supported methods: recovery_lora, recovery_train, recovery_linearpatch, recovery_ghost,
#    recovery_ghost_interval, recovery_ghost_interval_sequential,
#    recovery_ghost_cos, recovery_ghost_cos_more, recovery_ghost_cos_scale,
#    recovery_replaceme_ls, recovery_replaceme_cos, recovery_replaceme_cos_more,
#    recovery_replaceme_cos_scale. Prune&Comp is temporarily excluded.
#    All recovery methods are saved in:
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_all.json
RECOVERY_METHODS="${RECOVERY_METHODS:-recovery_linearpatch,recovery_ghost,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"
MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
stage_log "8/9" "Build recovery plans"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  set_backend_pythonpath "${bb}"
  plan_root="${OUTPUT_ROOT}/${bb}"
  mkdir -p "${plan_root}"
  rm -f "${plan_root}/recovery_plan_all.json"

  for method in ${RECOVERY_METHODS//,/ }; do
    if compact_log; then
      printf '\n  recovery plan\n    model: %s\n    method: %s\n    spans: all\n\n' \
        "${bb}" "${method#recovery_}"
    fi
    max_prefix_for_method="${MAX_LORA_ANCHORS}"
    case "${method}" in
      recovery_linearpatch|recovery_ghost|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_ghost_cos_more|recovery_replaceme_ls|recovery_replaceme_cos_more|recovery_prunecomp)
        max_prefix_for_method=1
        ;;
    esac
    for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
      run_command conda run -n "${CONDA_ENV}" python -m analysis.prune_plan.build_recovery_plan \
        --experiment "${bb}" \
        --backbone "${bb}" \
        --prune-plan-root "${plan_root}" \
        --recovery-plan-root "${plan_root}" \
        --method "${method}" \
        --lora-prefix "${lora_prefix}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}"
    done
  done
done




RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_linearpatch,recovery_ghost,recovery_ghost_interval,recovery_ghost_interval_sequential,recovery_replaceme_ls,recovery_replaceme_cos}"
EPOCHS="${EPOCHS:-25}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-25}"
# 9. Save minimal activation caches, run recovery, and evaluate recovered models.
#    Minimal cache is saved only for the selected prune/recovery plan union:
#    <OUTPUT_ROOT>/<backbone>/<train_dataset>/cache/cache_manifest_v3.json
#    Recovery checkpoints/artifacts are saved to:
#    <OUTPUT_ROOT>/<backbone>/recovery/<prune_spanK>/<recovery_plan>/checkpoints/latest.pt
#    Recovered eval results are saved in interval-result layout:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/eval_record_v3.json
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery/<recovery_plan>/iS_jE/summary.json
#    Recover metric plots are written to:
#    run_recovery.sh automatically calls summary_v3 after each backbone and writes:
#    <OUTPUT_ROOT>/<backbone>/<eval_dataset>/eval/recovery_plan_<eval_dataset>_recovery_drop.png
#    <OUTPUT_ROOT>/<backbone>/recovery_plan_<backbone>_recovery_drop.png
#    Summary table:
#    <OUTPUT_ROOT>/<backbone>/summary_comparison.md

stage_log "9/9" "Cache, recover, and evaluate"
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  cache_root="${OUTPUT_ROOT}/${bb}"
  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
  BACKBONES="${bb}" \
      RUN=SCANNET_DEV50,NUSCENES_TRAIN \
      CACHE=SCANNET_DEV50,NUSCENES_TRAIN \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      RECOVERY_PLAN_ROOT="${cache_root}" \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      bash "run_cache_redundancy.sh"
  
  BACKBONES="${bb}" \
      DATASETS=scannet_dev50,nuscenes_train \
      PRUNE_PLAN_NAME="prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      EPOCHS="${EPOCHS}" \
      MAX_SCENES="${MAX_SCENES}" \
      EVAL_EPOCH="${EVAL_EPOCH}" \
      SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      EVAL_BOUNDS_ROOT="${OUTPUT_ROOT}" \
      EVAL=SCANNET_DEV50,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION \
      bash "run_recovery.sh"

  rm -rf \
      "${cache_root}/scannet_dev50/cache" \
      "${cache_root}/nuscenes_train/cache" \
      "${cache_root}/7scenes_train/cache" \
      "${cache_root}/waymo_training/cache"
done

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
DATASETS="${DATASETS:-${DATASET:-}}"
EVAL="${EVAL:-}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"

PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-}"
PRUNE_PLAN_NAME="${PRUNE_PLAN_NAME:-}"
RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-}"
RECOVERY_CACHE_ROOT="${RECOVERY_CACHE_ROOT:-}"
RECOVERY_SCENE_LIMITS="${RECOVERY_SCENE_LIMITS:-}"
RECOVERY_FIT_SIGNATURE="${RECOVERY_FIT_SIGNATURE:-}"
RECOVERY_PRUNE_STRATEGY="${RECOVERY_PRUNE_STRATEGY:-}"
RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
OUTPUT_BASE="${OUTPUT_ROOT}"
EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT:-${OUTPUT_BASE}}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-5}"
EVAL_EPOCH="${EVAL_EPOCH:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
EVAL_AT_START="${EVAL_AT_START:-0}"
REPLACEME_LS_ALPHA="${REPLACEME_LS_ALPHA:-0}"
REPLACEME_COS_EPOCHS="${REPLACEME_COS_EPOCHS:-10}"
REPLACEME_COS_BATCH_SIZE="${REPLACEME_COS_BATCH_SIZE:-1024}"
REPLACEME_COS_LR="${REPLACEME_COS_LR:-1e-4}"
REPLACEME_SEED="${REPLACEME_SEED:-0}"
STREAMLINE_FFN_EPOCHS="${STREAMLINE_FFN_EPOCHS:-1}"
STREAMLINE_FFN_BATCH_SIZE="${STREAMLINE_FFN_BATCH_SIZE:-16384}"
STREAMLINE_FFN_LR="${STREAMLINE_FFN_LR:-2e-4}"
STREAMLINE_FFN_MIN_LR="${STREAMLINE_FFN_MIN_LR:-5e-6}"
STREAMLINE_FFN_WEIGHT_DECAY="${STREAMLINE_FFN_WEIGHT_DECAY:-1e-3}"
STREAMLINE_FFN_EXPANSION="${STREAMLINE_FFN_EXPANSION:-4}"
STREAMLINE_FFN_SEED="${STREAMLINE_FFN_SEED:-0}"
RECOVERY_CHUNK_ROWS="${RECOVERY_CHUNK_ROWS:-16384}"
GHOST_STABILITY_PROFILE="${GHOST_STABILITY_PROFILE:-legacy}"
GHOST_SPECIAL_RIDGE="${GHOST_SPECIAL_RIDGE:-0}"
GHOST_CAMERA_RIDGE="${GHOST_CAMERA_RIDGE:-0}"
GHOST_REGISTER_RIDGE="${GHOST_REGISTER_RIDGE:-0}"
GHOST_COUPLING_PROFILE="${GHOST_COUPLING_PROFILE:-none}"
GHOST_RESIDUAL_RIDGE="${GHOST_RESIDUAL_RIDGE:-0}"
GHOST_CONTEXT_RIDGE="${GHOST_CONTEXT_RIDGE:-0}"
GHOST_AUTOGROUP_THRESHOLD="${GHOST_AUTOGROUP_THRESHOLD:-20}"
GHOST_BOUNDARY_FIT_MODE="${GHOST_BOUNDARY_FIT_MODE:-sequential_current_target}"
REUSE_BOUNDARY_PREFIX_CHECKPOINT="${REUSE_BOUNDARY_PREFIX_CHECKPOINT:-}"

source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
install_error_diagnostics "${BASH_SOURCE[0]}"
parse_backbones_list
parse_dataset_list "${EVAL}" "EVAL" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

if [[ -z "${DATASETS//[[:space:],]/}" ]]; then
  echo "error: DATASETS is required" >&2
  exit 1
fi

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
  echo "error: RECOVERY_PLAN_NAME is required (e.g. RECOVERY_PLAN_NAME='recovery_lora_l1_r32 recovery_ghost')." >&2
  exit 1
fi

resolve_recovery_method() {
  local prune_root="$1"
  local recovery_root="$2"
  local prune_plan_name="$3"
  local recovery_plan_name="$4"
  python -c "
from pathlib import Path
from analysis.prune_plan.recovery_plan import load_recovery_plan_bundle, recovery_plan_supports_selector
prune_root = Path('${prune_root}')
recovery_root = Path('${recovery_root}')
name = '${recovery_plan_name}'
selector = '${prune_plan_name}'
bundle = load_recovery_plan_bundle(recovery_root, name)
supported = recovery_plan_supports_selector(
    root=recovery_root, name=name, prune_plan_root=prune_root, prune_selector=selector
)
print(str(bundle.get('method', '')).strip() if supported else 'unsupported')
"
}

recovery_output_complete() {
  local backbone="$1"
  local plan_root="$2"
  local prune_plan_name="$3"
  local recovery_plan_name="$4"
  local out_dir="${OUTPUT_BASE}/${backbone}/recovery/${prune_plan_name}/${recovery_plan_name}"
  if truthy "${RECOVERY_FORCE_RERUN}"; then
    return 1
  fi
  local signature_args=()
  if [[ -n "${RECOVERY_FIT_SIGNATURE}" ]]; then
    signature_args+=(--fit-signature "${RECOVERY_FIT_SIGNATURE}")
  fi
  python -m analysis.recovery.shared_results recovery-complete \
    --output-dir "${out_dir}" \
    --plan-root "${plan_root}" \
    --backbone "${backbone}" \
    --selector "${prune_plan_name}" \
    --recovery-plan-name "${recovery_plan_name}" \
    "${signature_args[@]}"
}

run_trainable_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  BACKBONES="${backbone}" \
    DATASETS="${DATASETS}" \
    PRUNE_PLAN_ROOT="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}" \
    RECOVERY_PLAN_ROOT="${RECOVERY_PLAN_ROOT:-${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}}" \
    RECOVERY_PLAN_NAME="${recovery_plan_name}" \
    PRUNE_PLAN_NAME="${prune_plan_name}" \
    OUTPUT_KIND=recovery \
    EPOCHS="${EPOCHS}" \
    MAX_SCENES="${MAX_SCENES}" \
    MAX_STEPS="${MAX_STEPS}" \
    EVAL_EPOCH="${EVAL_EPOCH}" \
    SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
    EVAL_AT_START="${EVAL_AT_START}" \
    EVAL_BOUNDS_ROOT="${EVAL_BOUNDS_ROOT}" \
    EVAL_MAX_SCENES="${EVAL_MAX_SCENES}" \
    EVAL="${EVAL}" \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    bash "${REPO_ROOT}/scripts/run_lora_skip_baseline.sh"
}

run_cache_fit_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  local plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}"
  local recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  local out_dir="${OUTPUT_BASE}/${backbone}/recovery/${prune_plan_name}/${recovery_plan_name}"
  local hf_args=()
  local cache_args=()
  local boundary_fit_args=(--ghost-boundary-fit-mode "${GHOST_BOUNDARY_FIT_MODE}")
  if [[ -n "${REUSE_BOUNDARY_PREFIX_CHECKPOINT}" ]]; then
    boundary_fit_args+=(--reuse-boundary-prefix-checkpoint "${REUSE_BOUNDARY_PREFIX_CHECKPOINT}")
  fi
  if [[ -n "${RECOVERY_CACHE_ROOT}" ]]; then
    cache_args+=(--cache-root "${RECOVERY_CACHE_ROOT}")
  fi
  if [[ -n "${RECOVERY_SCENE_LIMITS}" ]]; then
    cache_args+=(--scene-limits "${RECOVERY_SCENE_LIMITS}")
  fi
  if [[ -n "${RECOVERY_JOBS_MANIFEST:-}" && "${RECOVERY_SESSION_EXEC:-0}" != "1" ]]; then
    RECOVERY_JOB_FILE="${RECOVERY_JOBS_MANIFEST}" \
    RECOVERY_JOB_PRUNE="${prune_plan_name}" RECOVERY_JOB_NAME="${recovery_plan_name}" \
    RECOVERY_JOB_OUTPUT="${out_dir}" python -c '
import json, os
from pathlib import Path
p = Path(os.environ["RECOVERY_JOB_FILE"])
jobs = json.loads(p.read_text())
jobs.append({
    "prune_plan_name": os.environ["RECOVERY_JOB_PRUNE"],
    "recovery_plan_name": os.environ["RECOVERY_JOB_NAME"],
    "output_dir": os.environ["RECOVERY_JOB_OUTPUT"],
})
p.write_text(json.dumps(jobs, indent=2))
'
    return 0
  fi
  if truthy "${COMPACT_LOG:-0}"; then
    printf '\n  recovery\n    model: %s\n    span: %s\n    dataset: %s\n    method: %s\n\n' \
      "${backbone}" "${prune_plan_name#prune_span}" "${DATASETS}" \
      "${recovery_plan_name#recovery_}"
  else
    echo "== recovery v3 datasets=${DATASETS} backbone=${backbone} prune_plan_name=${prune_plan_name} recovery_plan_name=${recovery_plan_name} =="
  fi
  local session_args=()
  if [[ -n "${RECOVERY_JOBS_MANIFEST:-}" ]]; then
    session_args+=(--jobs-manifest "${RECOVERY_JOBS_MANIFEST}")
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT}" run_command python -m analysis.recovery.recovery_prune_v3 \
    --experiment "${backbone}" \
    --dataset "${DATASETS}" \
    --backbone "${backbone}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-scenes "${MAX_SCENES}" \
    --prune-plan-root "${plan_root}" \
    --prune-plan-name "${prune_plan_name}" \
    --recovery-plan-root "${recovery_root}" \
    --recovery-plan-name "${recovery_plan_name}" \
    --output-dir "${out_dir}" \
    "${cache_args[@]}" \
    --ghost-stability-profile "${GHOST_STABILITY_PROFILE}" \
    --ghost-special-ridge "${GHOST_SPECIAL_RIDGE}" \
    --ghost-camera-ridge "${GHOST_CAMERA_RIDGE}" \
    --ghost-register-ridge "${GHOST_REGISTER_RIDGE}" \
    --ghost-coupling-profile "${GHOST_COUPLING_PROFILE}" \
    --ghost-residual-ridge "${GHOST_RESIDUAL_RIDGE}" \
    --ghost-context-ridge "${GHOST_CONTEXT_RIDGE}" \
    --ghost-autogroup-threshold "${GHOST_AUTOGROUP_THRESHOLD}" \
    "${boundary_fit_args[@]}" \
    --recovery-chunk-rows "${RECOVERY_CHUNK_ROWS}" \
    --replaceme-ls-alpha "${REPLACEME_LS_ALPHA}" \
    --replaceme-cos-epochs "${REPLACEME_COS_EPOCHS}" \
    --replaceme-cos-batch-size "${REPLACEME_COS_BATCH_SIZE}" \
    --replaceme-cos-lr "${REPLACEME_COS_LR}" \
    --replaceme-seed "${REPLACEME_SEED}" \
    --streamline-ffn-epochs "${STREAMLINE_FFN_EPOCHS}" \
    --streamline-ffn-batch-size "${STREAMLINE_FFN_BATCH_SIZE}" \
    --streamline-ffn-lr "${STREAMLINE_FFN_LR}" \
    --streamline-ffn-min-lr "${STREAMLINE_FFN_MIN_LR}" \
    --streamline-ffn-weight-decay "${STREAMLINE_FFN_WEIGHT_DECAY}" \
    --streamline-ffn-expansion "${STREAMLINE_FFN_EXPANSION}" \
    --streamline-ffn-seed "${STREAMLINE_FFN_SEED}" \
    "${session_args[@]}" \
    "${hf_args[@]}"
  if [[ "${RECOVERY_SESSION_EXEC:-0}" == "1" ]]; then
    return 0
  fi
  eval_recovery "${backbone}" "${prune_plan_name}" "${recovery_plan_name}"
}

eval_recovery() {
  local backbone="$1"
  local prune_plan_name="$2"
  local recovery_plan_name="$3"
  local plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${backbone}}"
  local recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  local out_dir="${OUTPUT_BASE}/${backbone}/recovery/${prune_plan_name}/${recovery_plan_name}"
  if [[ "${#EVAL_LIST[@]}" -gt 0 ]]; then
    printf '\n  recovery eval\n    model: %s\n    span: %s\n    recovery: %s\n    datasets: %s\n\n' \
      "${backbone}" "${prune_plan_name#prune_span}" \
      "${recovery_plan_name#recovery_}" "${EVAL}"
    BACKBONES="${backbone}" \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${prune_plan_name}" \
      RECOVERY_PLAN_ROOT="${recovery_root}" \
      RECOVERY_PLAN_NAME="${recovery_plan_name}" \
      RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN}" \
      CHECKPOINT="${out_dir}/checkpoints/latest.pt" \
      MAX_SCENES="${EVAL_MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      RUN="${EVAL}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
  fi
}

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== recovery v3 backbones=${BACKBONES} (n=${BB_N}) =="
fi

for bb in "${BB_LIST[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  set_backend_pythonpath "${bb}"
  recovery_plan_names_for_backbone=("${RECOVERY_PLAN_NAME_LIST[@]}")
  if [[ "${bb}" == "pi3" ]]; then
    recovery_plan_names_for_backbone=()
    for requested_recovery_name in "${RECOVERY_PLAN_NAME_LIST[@]}"; do
      dependency_name=""
      case "${requested_recovery_name}" in
        recovery_ghost_interval_sequential_grouped)
          dependency_name="recovery_ghost_interval_sequential"
          ;;
      esac
      for candidate_name in "${dependency_name}" "${requested_recovery_name}"; do
        [[ -z "${candidate_name}" ]] && continue
        if [[ ! " ${recovery_plan_names_for_backbone[*]} " =~ " ${candidate_name} " ]]; then
          recovery_plan_names_for_backbone+=("${candidate_name}")
        fi
      done
    done
  fi
  plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_BASE}/${bb}}"
  recovery_root="${RECOVERY_PLAN_ROOT:-${plan_root}}"
  canonical_prune_plan_names=()
  for requested_prune_plan_name in "${PRUNE_PLAN_NAME_LIST[@]}"; do
    prune_plan_name="$(
      python -m analysis.recovery.shared_results canonical-selector \
        --plan-root "${plan_root}" --selector "${requested_prune_plan_name}"
    )"
    if [[ " ${canonical_prune_plan_names[*]} " =~ " ${prune_plan_name} " ]]; then
      continue
    fi
    canonical_prune_plan_names+=("${prune_plan_name}")
    recovery_jobs_manifest="$(mktemp "${TMPDIR:-/tmp}/lare_recovery_jobs.XXXXXX.json")"
    printf '[]\n' > "${recovery_jobs_manifest}"
    cache_fit_names=()
    pending_cache_fit_names=()
    for recovery_plan_name in "${recovery_plan_names_for_backbone[@]}"; do
      method="$(resolve_recovery_method "${plan_root}" "${recovery_root}" "${prune_plan_name}" "${recovery_plan_name}")"
      case "${method}" in
        unsupported)
          echo "warning: skipping unsupported ${prune_plan_name}+${recovery_plan_name}" >&2
          continue
          ;;
        recovery_lora|recovery_train)
          if recovery_output_complete "${bb}" "${plan_root}" "${prune_plan_name}" "${recovery_plan_name}"; then
            echo "   [reuse] ${bb}/${prune_plan_name}/${recovery_plan_name}"
          else
            run_trainable_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          fi
          eval_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
          ;;
        recovery_linearpatch|recovery_ghost|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_ghost_interval_sequential_grouped|recovery_ghost_interval_sequential_autogroup|recovery_replaceme_ls|recovery_replaceme_cos|recovery_streamline_ffn)
          cache_fit_names+=("${recovery_plan_name}")
          if recovery_output_complete "${bb}" "${plan_root}" "${prune_plan_name}" "${recovery_plan_name}"; then
            echo "   [reuse] ${bb}/${prune_plan_name}/${recovery_plan_name}"
          else
            RECOVERY_JOBS_MANIFEST="${recovery_jobs_manifest}" \
              run_cache_fit_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
            pending_cache_fit_names+=("${recovery_plan_name}")
          fi
          ;;
        *)
          echo "error: unsupported recovery method=${method} for ${recovery_plan_name}" >&2
          exit 1
          ;;
      esac
    done
    if [[ "${#pending_cache_fit_names[@]}" -gt 0 ]]; then
      RECOVERY_JOBS_MANIFEST="${recovery_jobs_manifest}" RECOVERY_SESSION_EXEC=1 \
        run_cache_fit_recovery "${bb}" "${prune_plan_name}" "${pending_cache_fit_names[0]}"
    fi
    for recovery_plan_name in "${cache_fit_names[@]}"; do
      eval_recovery "${bb}" "${prune_plan_name}" "${recovery_plan_name}"
    done
    rm -f "${recovery_jobs_manifest}"
  done
done

if ! truthy "${COMPACT_LOG:-0}"; then
  echo "== done =="
  experiment_output_hint "Recover"
fi

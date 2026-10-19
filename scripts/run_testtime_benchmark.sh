#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
TESTTIME_MODEL_ROOT="${TESTTIME_MODEL_ROOT:-${OUTPUT_ROOT}}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
TESTTIME_BACKBONES="${TESTTIME_BACKBONES:-}"
TESTTIME_RUN="${TESTTIME_RUN:-}"
TESTTIME_PRUNE_PLAN_NAME="${TESTTIME_PRUNE_PLAN_NAME:-}"
TESTTIME_PRUNE_STRATEGY="${TESTTIME_PRUNE_STRATEGY:-}"
TESTTIME_RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME:-}"
TESTTIME_SELECTION_DATASETS="${TESTTIME_SELECTION_DATASETS:-}"
TESTTIME_BENCHMARK_SCENES="${TESTTIME_BENCHMARK_SCENES:-10}"
TESTTIME_BENCHMARK_GPU_POLL_MS="${TESTTIME_BENCHMARK_GPU_POLL_MS:-200}"
TESTTIME_BENCHMARK_DRY_RUN="${TESTTIME_BENCHMARK_DRY_RUN:-0}"
TESTTIME_BENCHMARK_FORCE_RERUN="${TESTTIME_BENCHMARK_FORCE_RERUN:-0}"
TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS="${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS:-0}"
TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS="${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS:-0}"
TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES="${TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES:-0}"
# Comma-separated subset of the variants normally benchmarked by this runner.
# The empty default preserves the original complete schedule.
TESTTIME_BENCHMARK_VARIANTS="${TESTTIME_BENCHMARK_VARIANTS:-}"

FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-submodules/quantvggt/w4a4_official}"

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"

if [[ -z "${TESTTIME_BACKBONES//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_BACKBONES is required" >&2
  exit 1
fi
if [[ -z "${TESTTIME_RUN//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_RUN is required" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" != "0" && "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" != "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS must be 0 or 1" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS}" != "0" && "${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS}" != "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS must be 0 or 1" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" == "1" && "${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS}" == "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS and TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS cannot both be 1" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" != "1" ]]; then
  if [[ -z "${TESTTIME_PRUNE_PLAN_NAME//[[:space:],]/}" ]]; then
    echo "error: TESTTIME_PRUNE_PLAN_NAME is required" >&2
    exit 1
  fi
  if [[ -z "${TESTTIME_PRUNE_STRATEGY//[[:space:],]/}" ]]; then
    echo "error: TESTTIME_PRUNE_STRATEGY must contain retention and/or CKA" >&2
    exit 1
  fi
  if [[ -z "${TESTTIME_RECOVERY_PLAN_NAME//[[:space:],]/}" ]]; then
    echo "error: TESTTIME_RECOVERY_PLAN_NAME is required" >&2
    exit 1
  fi
  if [[ -z "${TESTTIME_SELECTION_DATASETS//[[:space:],]/}" ]]; then
    echo "error: TESTTIME_SELECTION_DATASETS is required" >&2
    exit 1
  fi
fi
if [[ "${TESTTIME_BENCHMARK_SCENES}" != "10" ]]; then
  echo "error: TESTTIME_BENCHMARK_SCENES must be 10 for the fixed benchmark protocol" >&2
  exit 1
fi
if [[ ! "${MAX_SCENES}" =~ ^[0-9]+$ ]]; then
  echo "error: MAX_SCENES must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "${TESTTIME_BENCHMARK_GPU_POLL_MS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: TESTTIME_BENCHMARK_GPU_POLL_MS must be a positive integer" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_DRY_RUN}" != "0" && "${TESTTIME_BENCHMARK_DRY_RUN}" != "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_DRY_RUN must be 0 or 1" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_FORCE_RERUN}" != "0" && "${TESTTIME_BENCHMARK_FORCE_RERUN}" != "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_FORCE_RERUN must be 0 or 1" >&2
  exit 1
fi
if [[ "${TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES}" != "0" && "${TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES}" != "1" ]]; then
  echo "error: TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES must be 0 or 1" >&2
  exit 1
fi

benchmark_variant_enabled() {
  local requested="$1"
  [[ -z "${TESTTIME_BENCHMARK_VARIANTS//[[:space:],]/}" ]] && return 0
  local candidate
  for candidate in ${TESTTIME_BENCHMARK_VARIANTS//,/ }; do
    [[ "${candidate}" == "${requested}" ]] && return 0
  done
  return 1
}
for candidate in ${TESTTIME_BENCHMARK_VARIANTS//,/ }; do
  case "${candidate}" in
    full|full_fastvggt|full_quantvggt|pruning|recovery|recovery_fastvggt|recovery_quantvggt) ;;
    *)
      echo "error: unsupported TESTTIME_BENCHMARK_VARIANTS entry: ${candidate@Q}" >&2
      exit 1
      ;;
  esac
done

BACKBONES="${TESTTIME_BACKBONES}"
parse_backbones_list
if [[ "${BB_N}" -ne 1 ]]; then
  echo "error: benchmark requires exactly one TESTTIME_BACKBONES entry per invocation" >&2
  exit 1
fi
backbone="${BB_LIST[0]}"
set_backend_pythonpath "${backbone}"

parse_dataset_list "${TESTTIME_RUN}" "TESTTIME_RUN" "${DATASET_KEYS[@]}"
RUN_LIST=("${PARSED_DATASETS[@]}")
if [[ "${#RUN_LIST[@]}" -eq 0 ]]; then
  echo "error: TESTTIME_RUN is empty" >&2
  exit 1
fi

testtime_prune_families=()
if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" != "1" ]]; then
  for prune_family in ${TESTTIME_PRUNE_STRATEGY//,/ }; do
    case "${prune_family,,}" in
      retention) canonical_prune_family="retention" ;;
      cka) canonical_prune_family="CKA" ;;
      *)
        echo "error: TESTTIME_PRUNE_STRATEGY entries must be retention or CKA; got ${prune_family@Q}" >&2
        exit 1
        ;;
    esac
    if [[ ! " ${testtime_prune_families[*]} " =~ " ${canonical_prune_family} " ]]; then
      testtime_prune_families+=("${canonical_prune_family}")
    fi
  done
fi

job_args=()
for dataset_key in "${RUN_LIST[@]}"; do
  case "${dataset_key}" in
    SCANNET_DEV50)
      job_args+=(--job "scannet|${ROOT_SCANNET_DEV50}|scannet_dev50|")
      ;;
    SCANNET_TEST50)
      job_args+=(--job "scannet|${ROOT_SCANNET_TEST50}|scannet_test50|")
      ;;
    7SCENES_TEST)
      job_args+=(--job "7scenes|${ROOT_7SCENES}|7scenes_test|test")
      ;;
    7SCENES_TRAIN)
      job_args+=(--job "7scenes|${ROOT_7SCENES}|7scenes_train|train")
      ;;
    NRGBD)
      job_args+=(--job "nrgbd|${ROOT_NRGBD}|nrgbd|")
      ;;
    NUSCENES_VAL)
      job_args+=(--job "nuscenes|${ROOT_NUSCENES}|nuscenes_val|val")
      ;;
    NUSCENES_TRAIN)
      job_args+=(--job "nuscenes|${ROOT_NUSCENES}|nuscenes_train|train")
      ;;
    WAYMO_VALIDATION)
      job_args+=(--job "waymo|${ROOT_WAYMO}|waymo_validation|validation")
      ;;
    WAYMO_TRAINING)
      job_args+=(--job "waymo|${ROOT_WAYMO}|waymo_training|training")
      ;;
  esac
done

selection_map=""
cleanup() {
  if [[ -n "${selection_map}" ]]; then
    rm -f "${selection_map}"
  fi
}
trap cleanup EXIT

selectors=()
declare -A scheduled_selectors=()
if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" != "1" ]]; then
  selection_map="$(mktemp "${TMPDIR:-/tmp}/lare_testtime_benchmark_selection.XXXXXX.tsv")"
  for prune_plan_name in ${TESTTIME_PRUNE_PLAN_NAME//,/ }; do
    drop_count="${prune_plan_name#prune_span}"
    if [[ "${prune_plan_name}" != "prune_span${drop_count}" || ! "${drop_count}" =~ ^[1-9][0-9]*$ ]]; then
      echo "error: expected prune_spanN, got ${prune_plan_name@Q}" >&2
      exit 1
    fi
    for prune_family in "${testtime_prune_families[@]}"; do
      selection="$(
        python -m analysis.recovery.shared_results best-recovery-result \
          --plan-root "${TESTTIME_MODEL_ROOT}/${backbone}" \
          --drop-count "${drop_count}" \
          --datasets "${TESTTIME_SELECTION_DATASETS}" \
          --recovery-plan-name "${TESTTIME_RECOVERY_PLAN_NAME}" \
          --prune-family "${prune_family}"
      )"
      IFS=$'\t' read -r selected_strategy physical_selector selection_score <<< "${selection}"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${physical_selector}" "${drop_count}" "${prune_family}" \
        "${selected_strategy}" "${selection_score}" >> "${selection_map}"
      if [[ -z "${scheduled_selectors[${physical_selector}]:-}" ]]; then
        scheduled_selectors["${physical_selector}"]=1
        selectors+=("${physical_selector}")
      fi
    done
  done
fi

quant_params_dir="${QUANTVGGT_PARAMS_DIR}"
if [[ "${backbone}" == "vggt" && -f "${QUANTVGGT_PARAMS_DIR}/${backbone}/qs_frame_parameters_total.pth" ]]; then
  quant_params_dir="${QUANTVGGT_PARAMS_DIR}/${backbone}"
fi
benchmark_needs_quant=0
if benchmark_variant_enabled full_quantvggt || benchmark_variant_enabled recovery_quantvggt; then
  benchmark_needs_quant=1
fi
if [[ "${backbone}" == "vggt" && "${benchmark_needs_quant}" == "1" && "${TESTTIME_BENCHMARK_DRY_RUN}" != "1" ]]; then
  for filename in qs_frame_parameters_total.pth qs_global_parameters_total.pth; do
    if [[ ! -f "${quant_params_dir}/${filename}" ]]; then
      echo "error: missing QuantVGGT calibration file: ${quant_params_dir}/${filename}" >&2
      exit 1
    fi
  done
fi

run_benchmark_variant() {
  local variant="$1"
  local selector="$2"
  local compression="$3"
  local checkpoint="${4:-}"
  local command=(
    python -m analysis.eval.inference_benchmark
    --backbone "${backbone}"
    --device "${DEVICE}"
    --variant "${variant}"
    --inference-compression "${compression}"
    --fastvggt-merge-start "${FASTVGGT_MERGE_START}"
    --fastvggt-merge-ratio "${FASTVGGT_MERGE_RATIO}"
    --quantvggt-dtype "${QUANTVGGT_DTYPE}"
    --output-root "${OUTPUT_ROOT}"
    --sample-count "${TESTTIME_BENCHMARK_SCENES}"
    --max-scenes "${MAX_SCENES}"
    --max-frames "${MAX_FRAMES}"
    --gpu-poll-interval-ms "${TESTTIME_BENCHMARK_GPU_POLL_MS}"
    "${job_args[@]}"
  )
  if [[ "${variant}" != full* ]]; then
    command+=(
      --prune-plan-root "${TESTTIME_MODEL_ROOT}/${backbone}"
      --prune-plan-name "${selector}"
      --selection-map "${selection_map}"
      --selection-datasets "${TESTTIME_SELECTION_DATASETS}"
    )
  fi
  if [[ "${variant}" == recovery* ]]; then
    command+=(
      --recovery-plan-root "${TESTTIME_MODEL_ROOT}/${backbone}"
      --recovery-plan-name "${TESTTIME_RECOVERY_PLAN_NAME}"
      --checkpoint "${checkpoint}"
    )
  fi
  if [[ "${compression}" == "quantvggt" ]]; then
    command+=(--quantvggt-params-dir "${quant_params_dir}")
  fi
  if [[ "${TESTTIME_BENCHMARK_FORCE_RERUN}" == "1" ]]; then
    command+=(--force-rerun)
  fi
  if [[ "${TESTTIME_BENCHMARK_ALLOW_VARIABLE_FRAMES}" == "1" ]]; then
    command+=(--allow-variable-frames)
  fi

  printf '\n[testtime/benchmark] backbone=%s selector=%s variant=%s compression=%s datasets=%s\n' \
    "${backbone}" "${selector}" "${variant}" "${compression}" "${TESTTIME_RUN}"
  if [[ "${TESTTIME_BENCHMARK_DRY_RUN}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  "${command[@]}"
}

if [[ "${TESTTIME_BENCHMARK_ONLY_FULL_COMPRESSIONS}" == "1" ]]; then
  echo "[testtime/benchmark] standalone full-model compression backfill only"
  benchmark_variant_enabled full_fastvggt && run_benchmark_variant full_fastvggt full fastvggt
  if [[ "${backbone}" == "vggt" ]] && benchmark_variant_enabled full_quantvggt; then
    run_benchmark_variant full_quantvggt full quantvggt
  elif benchmark_variant_enabled full_quantvggt; then
    echo "[testtime/benchmark] skip unsupported variant=full_quantvggt backbone=${backbone}"
  fi
  exit 0
fi

if [[ "${TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS}" == "1" ]]; then
  echo "[testtime/benchmark] skip migrated full/full_fastvggt/full_quantvggt variants"
else
  benchmark_variant_enabled full && run_benchmark_variant full full none
  benchmark_variant_enabled full_fastvggt && run_benchmark_variant full_fastvggt full fastvggt
  if [[ "${backbone}" == "vggt" ]] && benchmark_variant_enabled full_quantvggt; then
    run_benchmark_variant full_quantvggt full quantvggt
  elif benchmark_variant_enabled full_quantvggt; then
    echo "[testtime/benchmark] skip unsupported variant=full_quantvggt backbone=${backbone}"
  fi
fi

for selector in "${selectors[@]}"; do
  checkpoint="${TESTTIME_MODEL_ROOT}/${backbone}/recovery/${selector}/${TESTTIME_RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
  if [[ "${TESTTIME_BENCHMARK_DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    echo "error: missing recovered checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  benchmark_variant_enabled pruning && run_benchmark_variant pruning "${selector}" none
  benchmark_variant_enabled recovery && run_benchmark_variant recovery "${selector}" none "${checkpoint}"
  benchmark_variant_enabled recovery_fastvggt && run_benchmark_variant recovery_fastvggt "${selector}" fastvggt "${checkpoint}"
  if [[ "${backbone}" == "vggt" ]] && benchmark_variant_enabled recovery_quantvggt; then
    run_benchmark_variant recovery_quantvggt "${selector}" quantvggt "${checkpoint}"
  elif benchmark_variant_enabled recovery_quantvggt; then
    echo "[testtime/benchmark] skip unsupported variant=recovery_quantvggt backbone=${backbone}"
  fi
done

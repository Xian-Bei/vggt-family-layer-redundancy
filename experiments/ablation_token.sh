#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"

# Default backbones for all stages. Override with BACKBONES="vggt,pi3,...".
ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
BACKBONES="${ALL_BACKBONES}"
parse_backbones_list
ALL_BACKBONES_LIST=("${BB_LIST[@]}")
PRUNE_STRATEGY="CKA_single,CKA_double"

DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
ANALYSIS_DATASETS="${ANALYSIS_DATASETS:-scannet_dev50,nuscenes_train}"
EVAL_DATASETS="${EVAL_DATASETS:-}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-0}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
TOKEN_VARIANTS="${TOKEN_VARIANTS:-all,special,patch,special_patch_mean}"
export OUTPUT_ROOT
export COMPACT_LOG="${COMPACT_LOG:-1}"

VIZ_WORK_DIR=""

cleanup_viz_work_dir() {
  if [[ -n "${VIZ_WORK_DIR}" && -d "${VIZ_WORK_DIR}" && "$(basename "${VIZ_WORK_DIR}")" == lare_token_ablation_viz.* ]]; then
    rm -rf -- "${VIZ_WORK_DIR}"
  fi
}
trap cleanup_viz_work_dir EXIT

if [[ ! "${PLAN_MAX_DROP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: PLAN_MAX_DROP must be a positive integer" >&2
  exit 1
fi
if [[ ! "${PLAN_MIN_DROP}" =~ ^[0-9]+$ ]]; then
  echo "error: PLAN_MIN_DROP must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "${PLAN_DROP_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: PLAN_DROP_STEP must be a positive integer" >&2
  exit 1
fi
if (( PLAN_MIN_DROP > PLAN_MAX_DROP )); then
  echo "error: PLAN_MIN_DROP must not exceed PLAN_MAX_DROP" >&2
  exit 1
fi

parse_token_variants() {
  local raw variant
  local -a requested=()
  TOKEN_VARIANT_LIST=()
  raw="${TOKEN_VARIANTS//,/ }"
  read -r -a requested <<< "${raw}"
  for variant in "${requested[@]}"; do
    variant="${variant//[[:space:]]/}"
    [[ -z "${variant}" ]] && continue
    case "${variant}" in
      all|special|patch|special_patch_mean) ;;
      *)
        echo "error: unsupported TOKEN_VARIANTS entry '${variant}'; expected all, special, patch, or special_patch_mean" >&2
        exit 1
        ;;
    esac
    if ! list_has "${variant}" "${TOKEN_VARIANT_LIST[@]}"; then
      TOKEN_VARIANT_LIST+=("${variant}")
    fi
  done
  if [[ "${#TOKEN_VARIANT_LIST[@]}" -eq 0 ]]; then
    echo "error: TOKEN_VARIANTS is empty" >&2
    exit 1
  fi
}

ensure_symlink() {
  local source_path="$1"
  local link_path="$2"
  local resolved_source resolved_link relative_source
  if [[ ! -e "${source_path}" ]]; then
    echo "error: required source does not exist: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${link_path}")"
  resolved_source="$(realpath "${source_path}")"
  if [[ -L "${link_path}" ]]; then
    resolved_link="$(realpath "${link_path}")"
    if [[ "${resolved_link}" != "${resolved_source}" ]]; then
      echo "error: existing symlink points to a different source: ${link_path}" >&2
      exit 1
    fi
    return
  fi
  if [[ -e "${link_path}" ]]; then
    echo "error: refusing to replace existing non-symlink path: ${link_path}" >&2
    exit 1
  fi
  relative_source="$(realpath --relative-to="$(dirname "${link_path}")" "${resolved_source}")"
  ln -s "${relative_source}" "${link_path}"
}

bundle_path_for_variant() {
  local variant="$1"
  echo "${PLAN_ROOT}/prune_plan_all_token_${variant}.json"
}

plan_view_for_variant() {
  local variant="$1"
  echo "${PLAN_VIEW_ROOT}/${variant}"
}

prepare_plan_view() {
  local variant="$1"
  local bundle_path plan_view ds
  bundle_path="$(bundle_path_for_variant "${variant}")"
  plan_view="$(plan_view_for_variant "${variant}")"
  ensure_symlink "${bundle_path}" "${plan_view}/prune_plan_all.json"
  for ds in ${ANALYSIS_DATASETS//,/ } ${EVAL_DATASETS//,/ }; do
    ensure_symlink "${PLAN_ROOT}/${ds}" "${plan_view}/${ds}"
  done
}

parse_token_variants

stage_log "1/2" "Build CKA token-region ablation plans"
for BACKBONE in "${ALL_BACKBONES_LIST[@]}"; do
  PLAN_ROOT="${OUTPUT_ROOT}/${BACKBONE}"
  # Stage 7 and run_eval.sh require a directory containing prune_plan_all.json.
  # These compatibility views point at the suffixed bundles and at the shared
  # dataset result directories, so existing physical intervals are reused.
  PLAN_VIEW_ROOT="${PLAN_ROOT}/.token_ablation_plan_roots"

  mkdir -p "${PLAN_ROOT}"
  set_backend_pythonpath "${BACKBONE}"

  for variant in "${TOKEN_VARIANT_LIST[@]}"; do
    token_region="${variant}"

    bundle_path="$(bundle_path_for_variant "${variant}")"
    VIZ_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lare_token_ablation_viz.XXXXXX")"
    printf '\n  pruning plan\n    model: %s\n    strategies: %s\n    token variant: %s\n    token selector: %s\n    datasets: %s\n    output: %s\n\n' \
      "${BACKBONE}" "${PRUNE_STRATEGY}" "${variant}" "${token_region}" \
      "${ANALYSIS_DATASETS}" "${bundle_path}"

    python -m analysis.prune_plan.build_prune_plan_v3 \
      --experiment "${BACKBONE}" \
      --dataset "${ANALYSIS_DATASETS}" \
      --backbone "${BACKBONE}" \
      --metric cka \
      --strategies "${PRUNE_STRATEGY}" \
      --max-drop-budget "${PLAN_MAX_DROP}" \
      --token-region "${token_region}" \
      --prune-plan-root "${PLAN_ROOT}" \
      --output "${bundle_path}" \
      --viz-out-dir "${VIZ_WORK_DIR}"

    generated_grid="${VIZ_WORK_DIR}/prune_plan_${BACKBONE}_drop_grid.png"
    tagged_grid="${PLAN_ROOT}/prune_plan_${BACKBONE}_drop_grid_token_${variant}.png"
    if [[ -f "${generated_grid}" ]]; then
      mv -f "${generated_grid}" "${tagged_grid}"
    fi
    rmdir "${VIZ_WORK_DIR}"
    VIZ_WORK_DIR=""
    prepare_plan_view "${variant}"
  done
done

# Evaluate each suffixed bundle against the shared main result tree. Stage 7
# checks summary.json by physical selector, so main-experiment intervals and
# intervals completed by earlier token variants are skipped automatically.
stage_log "2/2" "Evaluate missing CKA token-ablation intervals"
for BACKBONE in "${ALL_BACKBONES_LIST[@]}"; do
  PLAN_ROOT="${OUTPUT_ROOT}/${BACKBONE}"
  PLAN_VIEW_ROOT="${PLAN_ROOT}/.token_ablation_plan_roots"

  for variant in "${TOKEN_VARIANT_LIST[@]}"; do
    plan_view="$(plan_view_for_variant "${variant}")"
    dataset_specs=()
    for ds in ${ANALYSIS_DATASETS//,/ }; do
      dataset_specs+=(--dataset-spec "${ds}:${ds}:1")
    done
    for ds in ${EVAL_DATASETS//,/ }; do
      dataset_specs+=(--dataset-spec "${ds}:${ds}:0")
    done

    printf '\n  pruning eval\n    model: %s\n    token variant: %s\n    result root: %s\n\n' \
      "${BACKBONE}" "${variant}" "${PLAN_ROOT}"
    schedule="$(
      python -m analysis.prune_plan.stage7 \
        --prune-plan-root "${plan_view}" \
        --strategies "${PRUNE_STRATEGY}" \
        --min-drop "${PLAN_MIN_DROP}" \
        --max-drop "${PLAN_MAX_DROP}" \
        --drop-step "${PLAN_DROP_STEP}" \
        "${dataset_specs[@]}"
    )"

    if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" ]]; then
      continue
    fi

    while IFS=$'\t' read -r selector dataset_keys; do
      [[ -z "${selector}" ]] && continue
      echo "   [evaluate] ${BACKBONE}/${variant}/${selector}: ${dataset_keys}"
      BACKBONES="${BACKBONE}" \
        RUN="${dataset_keys}" \
        EXPORT_VIS_2D=0 \
        EXPORT_VIS_3D=0 \
        PRUNE=1 \
        PRUNE_PLAN_ROOT="${plan_view}" \
        PRUNE_PLAN_NAME="${selector}" \
        MAX_SCENES="${MAX_SCENES}" \
        DEVICE="${DEVICE}" \
        bash "${REPO_ROOT}/scripts/run_eval.sh"
    done <<< "${schedule}"
  done
done

echo
echo "== CKA token ablation complete =="
echo "   plans: ${OUTPUT_ROOT}/<backbone>/prune_plan_all_token_<variant>.json"
echo "   evaluations: ${OUTPUT_ROOT}/<backbone>/<dataset>/eval/pruning/<physical-selector>/"

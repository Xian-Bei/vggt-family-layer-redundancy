#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
install_error_diagnostics "${BASH_SOURCE[0]}"

ALL_BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
BACKBONES="${ALL_BACKBONES}"
parse_backbones_list
ALL_BACKBONES_LIST=("${BB_LIST[@]}")
SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-analysis_outputs_v3}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-analysis_outputs_v3_heuristic_ablation}"
ANALYSIS_DATASETS="${ANALYSIS_DATASETS:-scannet_dev50,nuscenes_train}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
PLAN_MIN_DROP="${PLAN_MIN_DROP:-2}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
PLAN_DROP_STEP="${PLAN_DROP_STEP:-2}"
MAX_SCENES="${MAX_SCENES:-0}"
DEVICE="${DEVICE:-cuda}"
PRUNING_EVAL_DRY_RUN="${PRUNING_EVAL_DRY_RUN:-0}"
COMPACT_LOG="${COMPACT_LOG:-1}"
PRUNE_STRATEGIES="${PRUNE_STRATEGIES:-retention_single,retention_double,CKA_single,CKA_double}"
CKA_STRATEGIES="${CKA_STRATEGIES:-CKA_single,CKA_double}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-baseline,no_single_double}"
REPORT_PLOT_MODE="${REPORT_PLOT_MODE:-heldout-generalization}"
EVAL_PARALLELISM="${EVAL_PARALLELISM:-1}"
IFS=',' read -r -a VARIANTS <<< "${ABLATION_VARIANTS}"
for variant in "${VARIANTS[@]}"; do
  case "${variant}" in
    baseline|no_single_double) ;;
    *) echo "error: unknown ablation variant ${variant}" >&2; exit 1 ;;
  esac
done
[[ "${#VARIANTS[@]}" -gt 0 ]] || { echo "error: no ablation variants" >&2; exit 1; }
DRY_RUN_HAS_MISSING=0

if [[ "${SOURCE_OUTPUT_ROOT}" = /* ]]; then
  echo "error: SOURCE_OUTPUT_ROOT must be repository-relative for run_eval.sh" >&2
  exit 1
fi
if [[ "$(realpath -m "${SOURCE_OUTPUT_ROOT}")" == "$(realpath -m "${ABLATION_OUTPUT_ROOT}")" ]]; then
  echo "error: source and ablation output roots must differ" >&2
  exit 1
fi
if [[ ! "${PLAN_MIN_DROP}" =~ ^[0-9]+$ || ! "${PLAN_MAX_DROP}" =~ ^[1-9][0-9]*$ || ! "${PLAN_DROP_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: invalid drop range" >&2
  exit 1
fi
if (( PLAN_MIN_DROP > PLAN_MAX_DROP || PLAN_MIN_DROP % PLAN_DROP_STEP != 0 )); then
  echo "error: PLAN_MIN_DROP must not exceed max and must align with step" >&2
  exit 1
fi
if [[ ! "${EVAL_PARALLELISM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: EVAL_PARALLELISM must be a positive integer" >&2
  exit 1
fi

ensure_symlink() {
  local source_path="$1" link_path="$2" resolved_source resolved_link relative_source
  [[ -e "${source_path}" ]] || { echo "error: missing source ${source_path}" >&2; exit 1; }
  mkdir -p "$(dirname "${link_path}")"
  resolved_source="$(realpath "${source_path}")"
  if [[ -L "${link_path}" ]]; then
    resolved_link="$(realpath "${link_path}")"
    [[ "${resolved_link}" == "${resolved_source}" ]] || {
      echo "error: symlink points elsewhere: ${link_path}" >&2; exit 1;
    }
  elif [[ -e "${link_path}" ]]; then
    echo "error: refusing to replace non-symlink ${link_path}" >&2
    exit 1
  else
    relative_source="$(realpath --relative-to="$(dirname "${link_path}")" "${resolved_source}")"
    ln -s "${relative_source}" "${link_path}"
  fi
}

prepare_plan_view() {
  local plan_root="$1" backbone="$2" dataset
  mkdir -p "${plan_root}"
  for dataset in ${ANALYSIS_DATASETS//,/ } ${EVAL_DATASETS//,/ }; do
    ensure_symlink "${SOURCE_OUTPUT_ROOT}/${backbone}/${dataset}" "${plan_root}/${dataset}"
  done
}

variant_tolerance() {
  case "$1" in
    baseline) echo "0.005" ;;
    no_single_double) echo "0" ;;
    *) return 1 ;;
  esac
}

run_pruning_evals() {
  local variant="$1" backbone="$2" plan_root="$3" strategies="$4"
  local datasets="$5" training_flag="$6" phase="$7"
  local schedule selector dataset_keys dataset
  local -a dataset_specs=()
  for dataset in ${datasets//,/ }; do
    dataset_specs+=(--dataset-spec "${dataset}:${dataset}:${training_flag}")
  done
  [[ "${#dataset_specs[@]}" -gt 0 ]] || return
  schedule="$(python -m analysis.prune_plan.stage7 \
    --prune-plan-root "${plan_root}" --strategies "${strategies}" \
    --min-drop "${PLAN_MIN_DROP}" --max-drop "${PLAN_MAX_DROP}" \
    --drop-step "${PLAN_DROP_STEP}" "${dataset_specs[@]}")"
  if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" ]]; then
    if [[ -n "${schedule}" ]]; then
      DRY_RUN_HAS_MISSING=1
    fi
    return
  fi
  local -a selectors=() dataset_groups=() pids=()
  while IFS=$'\t' read -r selector dataset_keys; do
    [[ -n "${selector}" ]] || continue
    selectors+=("${selector}")
    dataset_groups+=("${dataset_keys}")
  done <<< "${schedule}"
  local batch_start batch_end index pid failed
  for ((batch_start = 0; batch_start < ${#selectors[@]}; batch_start += EVAL_PARALLELISM)); do
    batch_end=$((batch_start + EVAL_PARALLELISM))
    (( batch_end > ${#selectors[@]} )) && batch_end=${#selectors[@]}
    pids=()
    for ((index = batch_start; index < batch_end; index++)); do
      selector="${selectors[index]}"
      dataset_keys="${dataset_groups[index]}"
      echo "   [evaluate] ${variant}/${backbone}/${phase}/${selector}: ${dataset_keys}"
      (
        OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT}" BACKBONES="${backbone}" RUN="${dataset_keys}" \
          EXPORT_VIS_2D=0 EXPORT_VIS_3D=0 PRUNE=1 \
          PRUNE_PLAN_ROOT="${plan_root}" PRUNE_PLAN_NAME="${selector}" \
          MAX_SCENES="${MAX_SCENES}" DEVICE="${DEVICE}" \
          bash "${REPO_ROOT}/scripts/run_eval.sh"
      ) &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "${pid}" || failed=1
    done
    (( failed == 0 )) || return 1
    for ((index = batch_start; index < batch_end; index++)); do
      python -m analysis.results.pruning_heuristic_ablation record-eval \
        --output-root "${ABLATION_OUTPUT_ROOT}" --variant "${variant}" \
        --backbone "${backbone}" --selector "${selectors[index]}" \
        --datasets "${dataset_groups[index]}" --phase "${phase}"
    done
  done
}

mkdir -p "${ABLATION_OUTPUT_ROOT}"
python -m analysis.results.pruning_heuristic_ablation init-manifest \
  --output-root "${ABLATION_OUTPUT_ROOT}" --source-output-root "${SOURCE_OUTPUT_ROOT}" \
  --backbones "${ALL_BACKBONES}" --training-datasets "${ANALYSIS_DATASETS}" \
  --heldout-datasets "${EVAL_DATASETS}" --min-drop "${PLAN_MIN_DROP}" \
  --max-drop "${PLAN_MAX_DROP}" --drop-step "${PLAN_DROP_STEP}" \
  --variants "${ABLATION_VARIANTS}" --strategies "${PRUNE_STRATEGIES}"

stage_log "1/4" "Build similarity single-double tolerance variants"
for variant in "${VARIANTS[@]}"; do
  single_double_tolerance="$(variant_tolerance "${variant}")"
  for backbone in "${ALL_BACKBONES_LIST[@]}"; do
    set_backend_pythonpath "${backbone}"
    source_plan_root="${SOURCE_OUTPUT_ROOT}/${backbone}"
    variant_plan_root="${ABLATION_OUTPUT_ROOT}/${variant}/${backbone}"
    candidate_root="${variant_plan_root}/.prune_token_region_candidates"
    all_candidate="${candidate_root}/all"
    special_candidate="${candidate_root}/special_patch_mean"
    prepare_plan_view "${variant_plan_root}" "${backbone}"
    prepare_plan_view "${all_candidate}" "${backbone}"
    prepare_plan_view "${special_candidate}" "${backbone}"
    printf '\n  plan: %s/%s (similarity single-double=%s)\n' \
      "${variant}" "${backbone}" "${single_double_tolerance}"
    OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT}" python -m analysis.prune_plan.build_prune_plan_v3 \
      --experiment "${backbone}" --dataset "${ANALYSIS_DATASETS}" \
      --backbone "${backbone}" --metric cka --strategies "${PRUNE_STRATEGIES}" \
      --max-drop-budget "${PLAN_MAX_DROP}" --token-region all \
      --single-double-comparison-tolerance "${single_double_tolerance}" \
      --prune-plan-root "${source_plan_root}" \
      --output "${all_candidate}/prune_plan_all.json" --viz-out-dir "${all_candidate}"
    OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT}" python -m analysis.prune_plan.build_prune_plan_v3 \
      --experiment "${backbone}" --dataset "${ANALYSIS_DATASETS}" \
      --backbone "${backbone}" --metric cka --strategies "${CKA_STRATEGIES}" \
      --max-drop-budget "${PLAN_MAX_DROP}" --token-region special_patch_mean \
      --single-double-comparison-tolerance "${single_double_tolerance}" \
      --prune-plan-root "${source_plan_root}" \
      --output "${special_candidate}/prune_plan_all.json" --viz-out-dir "${special_candidate}"
  done
done

stage_log "2/4" "Evaluate token-region candidates on training datasets"
for variant in "${VARIANTS[@]}"; do
  for backbone in "${ALL_BACKBONES_LIST[@]}"; do
    candidate_root="${ABLATION_OUTPUT_ROOT}/${variant}/${backbone}/.prune_token_region_candidates"
    run_pruning_evals "${variant}" "${backbone}" "${candidate_root}/all" \
      "${PRUNE_STRATEGIES}" "${ANALYSIS_DATASETS}" 1 "training/all"
    run_pruning_evals "${variant}" "${backbone}" "${candidate_root}/special_patch_mean" \
      "${CKA_STRATEGIES}" "${ANALYSIS_DATASETS}" 1 "training/special_patch_mean"
  done
done

if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" && "${DRY_RUN_HAS_MISSING}" == "1" ]]; then
  echo "Dry run found missing training candidate evals."
  echo "Final token-region selection and held-out scheduling require those evals first."
  echo "Plans and the complete training schedule were generated successfully."
  exit 0
fi

stage_log "3/4" "Finalize token-region winners and evaluate held-out datasets"
for variant in "${VARIANTS[@]}"; do
  for backbone in "${ALL_BACKBONES_LIST[@]}"; do
    set_backend_pythonpath "${backbone}"
    source_plan_root="${SOURCE_OUTPUT_ROOT}/${backbone}"
    variant_plan_root="${ABLATION_OUTPUT_ROOT}/${variant}/${backbone}"
    candidate_root="${variant_plan_root}/.prune_token_region_candidates"
    python -m analysis.prune_plan.finalize_token_region_plan \
      --experiment-root "${source_plan_root}" \
      --all-bundle "${candidate_root}/all/prune_plan_all.json" \
      --special-patch-mean-bundle "${candidate_root}/special_patch_mean/prune_plan_all.json" \
      --selection-datasets "${ANALYSIS_DATASETS}" --min-drop "${PLAN_MIN_DROP}" \
      --max-drop "${PLAN_MAX_DROP}" --drop-step "${PLAN_DROP_STEP}" \
      --output "${variant_plan_root}/prune_plan_all.json" --viz-out-dir "${variant_plan_root}"
    run_pruning_evals "${variant}" "${backbone}" "${variant_plan_root}" \
      "${PRUNE_STRATEGIES}" "${EVAL_DATASETS}" 0 "heldout/final"
  done
done

stage_log "4/4" "Summarize degradation and render comparison curves"
if [[ "${PRUNING_EVAL_DRY_RUN}" != "1" ]]; then
  python -m analysis.results.pruning_heuristic_ablation summarize \
    --output-root "${ABLATION_OUTPUT_ROOT}" \
    --plot-mode "${REPORT_PLOT_MODE}"
else
  echo "Dry run complete; summary generation skipped."
fi
echo "== Pruning heuristic ablation complete =="
echo "plans/reports: ${ABLATION_OUTPUT_ROOT}"
echo "eval results: ${SOURCE_OUTPUT_ROOT}/<backbone>/<dataset>/eval/pruning/"

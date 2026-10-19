# Shared defaults and orchestration helpers for analysis.sh and main.sh.
# Source after REPO_ROOT and pipeline-specific OUTPUT_ROOT defaults are set.
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"

ALL_BACKBONES="${BACKBONES:-vggt_omega,pi3,vggt}"
BACKBONES="${ALL_BACKBONES}"
parse_backbones_list
ALL_BACKBONES_LIST=("${BB_LIST[@]}")
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
NORMALIZATION_MAX_DROP="${NORMALIZATION_MAX_DROP:-48}"
CALIB_DATASETS="${CALIB_DATASETS:-scannet_dev50,7scenes_train,nuscenes_train,waymo_training}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-submodules/quantvggt/w4a4_official}"
SIM_TOKEN_REGIONS="${SIM_TOKEN_REGIONS:-special,patch,all}"
RECOVERY_PRUNE_PLANS="${RECOVERY_PRUNE_PLANS:-prune_span10,prune_span12,prune_span14,prune_span16,prune_span18,prune_span20,prune_span22,prune_span24}"

run_pipeline_script() {
  local script="$1"
  shift
  MAX_SCENES="${MAX_SCENES}" DEVICE="${DEVICE}" bash "${REPO_ROOT}/scripts/${script}" "$@"
}

parse_prune_token_regions() {
  local raw region
  local -a requested=()
  PRUNE_TOKEN_REGION_LIST=()
  raw="${PRUNE_TOKEN_REGIONS//,/ }"
  read -r -a requested <<< "${raw}"
  for region in "${requested[@]}"; do
    region="${region//[[:space:]]/}"
    [[ -z "${region}" ]] && continue
    case "${region}" in
      all|special_patch_mean) ;;
      *)
        echo "error: unsupported PRUNE_TOKEN_REGIONS entry '${region}'; expected all and special_patch_mean" >&2
        exit 1
        ;;
    esac
    if ! list_has "${region}" "${PRUNE_TOKEN_REGION_LIST[@]}"; then
      PRUNE_TOKEN_REGION_LIST+=("${region}")
    fi
  done
  if ! list_has all "${PRUNE_TOKEN_REGION_LIST[@]}" || ! list_has special_patch_mean "${PRUNE_TOKEN_REGION_LIST[@]}"; then
    echo "error: PRUNE_TOKEN_REGIONS must contain both all and special_patch_mean" >&2
    exit 1
  fi
}

append_dual_token_strategy() {
  local strategy="$1"
  if ! list_has "${strategy}" "${DUAL_TOKEN_STRATEGY_LIST[@]}"; then
    DUAL_TOKEN_STRATEGY_LIST+=("${strategy}")
  fi
}

parse_dual_token_strategies() {
  local raw strategy key
  local -a requested=()
  DUAL_TOKEN_STRATEGY_LIST=()
  raw="${PRUNE_STRATEGY//,/ }"
  read -r -a requested <<< "${raw}"
  for strategy in "${requested[@]}"; do
    key="${strategy,,}"
    case "${key}" in
      cka)
        append_dual_token_strategy CKA_single
        append_dual_token_strategy CKA_double
        ;;
      cka_single) append_dual_token_strategy CKA_single ;;
      cka_double) append_dual_token_strategy CKA_double ;;
      cosine)
        append_dual_token_strategy cosine_single
        append_dual_token_strategy cosine_double
        ;;
      cosine_single|cosine_double) append_dual_token_strategy "${key}" ;;
      inv_mse)
        append_dual_token_strategy inv_mse_single
        append_dual_token_strategy inv_mse_double
        ;;
      inv_mse_single|inv_mse_double) append_dual_token_strategy "${key}" ;;
      cca)
        append_dual_token_strategy cca_single
        append_dual_token_strategy cca_double
        ;;
      cca_single|cca_double) append_dual_token_strategy "${key}" ;;
    esac
  done
  DUAL_TOKEN_STRATEGY="$(join_by_comma "${DUAL_TOKEN_STRATEGY_LIST[@]}")"
}

ensure_directory_symlink() {
  local source_path="$1"
  local link_path="$2"
  local resolved_source resolved_link relative_source
  if [[ ! -d "${source_path}" ]]; then
    echo "error: candidate plan source directory is missing: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${link_path}")"
  resolved_source="$(realpath "${source_path}")"
  if [[ -L "${link_path}" ]]; then
    resolved_link="$(realpath "${link_path}")"
    if [[ "${resolved_link}" != "${resolved_source}" ]]; then
      echo "error: candidate plan symlink points to a different source: ${link_path}" >&2
      exit 1
    fi
    return
  fi
  if [[ -e "${link_path}" ]]; then
    echo "error: refusing to replace candidate plan path: ${link_path}" >&2
    exit 1
  fi
  relative_source="$(realpath --relative-to="$(dirname "${link_path}")" "${resolved_source}")"
  ln -s "${relative_source}" "${link_path}"
}

run_pruning_evals() {
  local bb="$1"
  local plan_root="$2"
  local candidate_strategies="$3"
  local datasets="$4"
  local training_flag="$5"
  local scope="$6"
  local schedule
  local -a dataset_specs=()
  local ds
  for ds in ${datasets//,/ }; do
    dataset_specs+=(--dataset-spec "${ds}:${ds}:${training_flag}")
  done
  [[ "${#dataset_specs[@]}" -eq 0 ]] && return
  schedule="$(
    python -m analysis.prune_plan.stage7 \
        --prune-plan-root "${plan_root}" \
        --strategies "${candidate_strategies}" \
        --min-drop "${PLAN_MIN_DROP}" \
        --max-drop "${PLAN_MAX_DROP}" \
        --drop-step "${PLAN_DROP_STEP}" \
        "${dataset_specs[@]}"
  )"

  if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" ]]; then
    return
  fi
  local selector dataset_keys
  while IFS=$'\t' read -r selector dataset_keys; do
    [[ -z "${selector}" ]] && continue
    echo "   [evaluate] ${scope}/${bb}/${selector}: ${dataset_keys}"
    BACKBONES="${bb}" \
      RUN="${dataset_keys}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${selector}" \
      run_pipeline_script run_eval.sh
  done <<< "${schedule}"
}

run_pruning_backbone() {
  local bb="$1"
  local plan_root="${OUTPUT_ROOT}/${bb}"
  local schedule
  local -a dataset_specs=()
  local ds
  for ds in ${CALIB_DATASETS//,/ }; do
    dataset_specs+=(--dataset-spec "${ds}:${ds}:1")
  done
  for ds in ${EVAL_DATASETS//,/ }; do
    dataset_specs+=(--dataset-spec "${ds}:${ds}:0")
  done
  schedule="$(
    python -m analysis.prune_plan.stage7 \
      --prune-plan-root "${plan_root}" \
      --strategies "${PRUNE_STRATEGY}" \
      --min-drop "${PLAN_MIN_DROP}" \
      --max-drop "${PLAN_MAX_DROP}" \
      --drop-step "${PLAN_DROP_STEP}" \
      "${dataset_specs[@]}"
  )"

  if [[ "${PRUNING_EVAL_DRY_RUN}" == "1" ]]; then
    return
  fi

  local selector dataset_keys
  while IFS=$'\t' read -r selector dataset_keys; do
    [[ -z "${selector}" ]] && continue
    echo "   [evaluate] ${bb}/${selector}: ${dataset_keys}"
    BACKBONES="${bb}" \
      RUN="${dataset_keys}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${selector}" \
      run_pipeline_script run_eval.sh
  done <<< "${schedule}"
}

build_pipeline_recovery_plans() {
  MAX_LORA_ANCHORS="${MAX_LORA_ANCHORS:-1}"
  LORA_RANK="${LORA_RANK:-32}"
  LORA_ALPHA="${LORA_ALPHA:-64}"
  stage_log "8/10" "Build recovery plans"
  for bb in "${ALL_BACKBONES_LIST[@]}"; do
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
        recovery_linearpatch|recovery_ghost|recovery_ghost_interval|recovery_ghost_interval_sequential|recovery_ghost_interval_sequential_grouped|recovery_ghost_interval_sequential_autogroup|recovery_replaceme_ls|recovery_replaceme_cos|recovery_streamline_ffn)
          max_prefix_for_method=1
          ;;
      esac
      for lora_prefix in $(seq 1 "${max_prefix_for_method}"); do
        run_command python -m analysis.prune_plan.build_recovery_plan \
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
}

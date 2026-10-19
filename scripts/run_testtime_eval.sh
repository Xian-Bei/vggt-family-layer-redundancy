#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Evaluate already recovered LaRe models for VGGT, Pi3, and VGGT-Omega, with
# optional deployment-time compression. The prune_span aliases are recorded in
# prune_plan_all.json; no recovery training or activation cache is repeated.
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
# Plans and recovered checkpoints may live in a different tree from newly
# generated evaluation results (for example, long-sequence generalization).
TESTTIME_MODEL_ROOT="${TESTTIME_MODEL_ROOT:-${OUTPUT_ROOT}}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
MAX_FRAMES="${MAX_FRAMES:-30}"
TESTTIME_BACKBONES="${TESTTIME_BACKBONES:-}"
TESTTIME_RUN="${TESTTIME_RUN:-}"
TESTTIME_PRUNE_PLAN_NAME="${TESTTIME_PRUNE_PLAN_NAME:-}"
TESTTIME_PRUNE_STRATEGY="${TESTTIME_PRUNE_STRATEGY:-}"
TESTTIME_RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME:-}"
TESTTIME_SELECTION_DATASETS="${TESTTIME_SELECTION_DATASETS:-}"
TESTTIME_COMPRESSIONS="${TESTTIME_COMPRESSIONS:-fastvggt,quantvggt}"
TESTTIME_DRY_RUN="${TESTTIME_DRY_RUN:-0}"
EXPORT_VIS_2D="${EXPORT_VIS_2D:-0}"
EXPORT_VIS_3D="${EXPORT_VIS_3D:-0}"

FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
# This is the explicitly named official-parameter transfer baseline.  Override
# it with parameters recalibrated on the final pruned+recovered model when
# available.
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-submodules/quantvggt/w4a4_official}"

if [[ -z "${TESTTIME_BACKBONES//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_BACKBONES is required" >&2
  exit 1
fi
if [[ -z "${TESTTIME_RUN//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_RUN is required" >&2
  exit 1
fi
if [[ -z "${TESTTIME_PRUNE_PLAN_NAME//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_PRUNE_PLAN_NAME is required" >&2
  exit 1
fi
if [[ -z "${TESTTIME_PRUNE_STRATEGY//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_PRUNE_STRATEGY must contain retention and/or CKA" >&2
  exit 1
fi
if [[ -z "${TESTTIME_SELECTION_DATASETS//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_SELECTION_DATASETS is required for single/double selection" >&2
  exit 1
fi
if [[ -z "${TESTTIME_RECOVERY_PLAN_NAME//[[:space:],]/}" ]]; then
  echo "error: TESTTIME_RECOVERY_PLAN_NAME is required" >&2
  exit 1
fi
if [[ "${TESTTIME_DRY_RUN}" != "0" && "${TESTTIME_DRY_RUN}" != "1" ]]; then
  echo "error: TESTTIME_DRY_RUN must be 0 or 1" >&2
  exit 1
fi
testtime_log_suffix=""
if [[ "${TESTTIME_DRY_RUN}" == "1" ]]; then
  testtime_log_suffix="/dry-run"
fi
testtime_prune_families=()
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

for compression in ${TESTTIME_COMPRESSIONS//,/ }; do
  case "${compression}" in
    none|fastvggt|quantvggt) ;;
    *)
      echo "error: TESTTIME_COMPRESSIONS entries must be none, fastvggt, or quantvggt; got ${compression}" >&2
      exit 1
      ;;
  esac

  for backbone in ${TESTTIME_BACKBONES//,/ }; do
    if [[ "${compression}" == "quantvggt" && "${backbone}" != "vggt" ]]; then
      echo "[testtime${testtime_log_suffix}] skip unsupported compression=${compression} backbone=${backbone}; supported=vggt"
      continue
    fi

    declare -A scheduled_selectors=()
    quant_params_dir="${QUANTVGGT_PARAMS_DIR}"
    if [[ "${compression}" == "quantvggt" && "${TESTTIME_DRY_RUN}" != "1" ]]; then
      # A root containing per-backbone calibration directories is accepted;
      # retain compatibility with the original single-directory VGGT layout.
      if [[ -f "${QUANTVGGT_PARAMS_DIR}/${backbone}/qs_frame_parameters_total.pth" ]]; then
        quant_params_dir="${QUANTVGGT_PARAMS_DIR}/${backbone}"
      fi
      for filename in qs_frame_parameters_total.pth qs_global_parameters_total.pth; do
        if [[ ! -f "${quant_params_dir}/${filename}" ]]; then
          echo "error: missing QuantVGGT calibration file for ${backbone}: ${quant_params_dir}/${filename}" >&2
          echo "       set QUANTVGGT_PARAMS_DIR to a matching directory or a root with <backbone>/ subdirectories" >&2
          exit 1
        fi
      done
    fi

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

        printf '\n[testtime%s] backbone=%s span=%s family=%s strategy=%s selector=%s score=%s recovery=%s compression=%s selection_datasets=%s eval_datasets=%s\n' \
          "${testtime_log_suffix}" \
          "${backbone}" "${drop_count}" "${prune_family}" "${selected_strategy}" \
          "${physical_selector}" "${selection_score}" "${TESTTIME_RECOVERY_PLAN_NAME}" \
          "${compression}" "${TESTTIME_SELECTION_DATASETS}" "${TESTTIME_RUN}"
        if [[ -n "${scheduled_selectors[${physical_selector}]:-}" ]]; then
          echo "   [deduplicate] selector=${physical_selector} already selected by ${scheduled_selectors[${physical_selector}]}; skip duplicate ${prune_family} schedule"
          continue
        fi
        scheduled_selectors["${physical_selector}"]="${prune_family}"
        if [[ "${TESTTIME_DRY_RUN}" == "1" ]]; then
          continue
        fi

        checkpoint="${TESTTIME_MODEL_ROOT}/${backbone}/recovery/${physical_selector}/${TESTTIME_RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
        if [[ ! -f "${checkpoint}" ]]; then
          echo "error: missing recovered checkpoint: ${checkpoint}" >&2
          echo "       finish step 9 for ${backbone}/${prune_plan_name}/${TESTTIME_RECOVERY_PLAN_NAME} first" >&2
          exit 1
        fi

        recovery_result_kind=testtime
        recovery_force_rerun="${RECOVERY_FORCE_RERUN:-1}"
        if [[ "${compression}" == "none" ]]; then
          # The uncompressed baseline belongs to the canonical recovery tree.
          # Force evaluation so an existing metrics-only result cannot suppress
          # the requested per-scene visualizations.
          recovery_result_kind=recovery
          recovery_force_rerun=1
        fi

        BACKBONES="${backbone}" \
          RUN="${TESTTIME_RUN}" \
          PRUNE=1 \
          PRUNE_PLAN_ROOT="${TESTTIME_MODEL_ROOT}/${backbone}" \
          RECOVERY_PLAN_ROOT="${TESTTIME_MODEL_ROOT}/${backbone}" \
          PRUNE_PLAN_NAME="${physical_selector}" \
          RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
          RECOVERY_OUTPUT_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
          RECOVERY_RESULT_KIND="${recovery_result_kind}" \
          RECOVERY_FORCE_RERUN="${recovery_force_rerun}" \
          CHECKPOINT="${checkpoint}" \
          INFERENCE_COMPRESSION="${compression}" \
          FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START}" \
          FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO}" \
          QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE}" \
          QUANTVGGT_PARAMS_DIR="${quant_params_dir}" \
          EXPORT_VIS_2D="${EXPORT_VIS_2D}" \
          EXPORT_VIS_3D="${EXPORT_VIS_3D}" \
          OUTPUT_ROOT="${OUTPUT_ROOT}" \
          MAX_SCENES="${MAX_SCENES}" \
          MAX_FRAMES="${MAX_FRAMES}" \
          DEVICE="${DEVICE}" \
          bash "${REPO_ROOT}/scripts/run_eval.sh"
      done
    done
    unset scheduled_selectors
  done
done

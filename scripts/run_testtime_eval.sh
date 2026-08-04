#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Evaluate deployment-time compression on the already recovered best LaRe
# models.  The default prune_span aliases are the Ours/CKA plans recorded in
# prune_plan_all.json; no recovery training or activation cache is repeated.
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
TESTTIME_BACKBONES="${TESTTIME_BACKBONES:-vggt}"
TESTTIME_RUN="${TESTTIME_RUN:-SCANNET_DEV50,NUSCENES_TRAIN,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION}"
TESTTIME_PRUNE_PLAN_NAME="${TESTTIME_PRUNE_PLAN_NAME:-prune_span12,prune_span16,prune_span20,prune_span24}"
TESTTIME_RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME:-recovery_ghost_interval_sequential}"
TESTTIME_COMPRESSIONS="${TESTTIME_COMPRESSIONS:-fastvggt}"

FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
# This is the explicitly named official-parameter transfer baseline.  Override
# it with parameters recalibrated on the final pruned+recovered model when
# available.
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-submodules/quantvggt/w4a4_official}"

if [[ "${TESTTIME_BACKBONES//[[:space:],]/}" != "vggt" ]]; then
  echo "error: FastVGGT and QuantVGGT test-time adapters currently support only TESTTIME_BACKBONES=vggt" >&2
  exit 1
fi

for compression in ${TESTTIME_COMPRESSIONS//,/ }; do
  case "${compression}" in
    fastvggt|quantvggt) ;;
    *)
      echo "error: TESTTIME_COMPRESSIONS entries must be fastvggt or quantvggt; got ${compression}" >&2
      exit 1
      ;;
  esac

  if [[ "${compression}" == "quantvggt" ]]; then
    for filename in qs_frame_parameters_total.pth qs_global_parameters_total.pth; do
      if [[ ! -f "${QUANTVGGT_PARAMS_DIR}/${filename}" ]]; then
        echo "error: missing QuantVGGT calibration file: ${QUANTVGGT_PARAMS_DIR}/${filename}" >&2
        echo "       run 'bash submodules/setup_quantvggt_params.sh' or set QUANTVGGT_PARAMS_DIR" >&2
        exit 1
      fi
    done
  fi

  for prune_plan_name in ${TESTTIME_PRUNE_PLAN_NAME//,/ }; do
    checkpoint="${OUTPUT_ROOT}/vggt/recovery/${prune_plan_name}/${TESTTIME_RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "error: missing recovered checkpoint: ${checkpoint}" >&2
      echo "       finish step 9 for ${prune_plan_name}/${TESTTIME_RECOVERY_PLAN_NAME} first" >&2
      exit 1
    fi

    printf '\n[testtime] span=%s recovery=%s compression=%s datasets=%s\n' \
      "${prune_plan_name#prune_span}" "${TESTTIME_RECOVERY_PLAN_NAME}" "${compression}" "${TESTTIME_RUN}"
    BACKBONES="${TESTTIME_BACKBONES}" \
      RUN="${TESTTIME_RUN}" \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/vggt" \
      RECOVERY_PLAN_ROOT="${OUTPUT_ROOT}/vggt" \
      PRUNE_PLAN_NAME="${prune_plan_name}" \
      RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
      RECOVERY_OUTPUT_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
      RECOVERY_RESULT_KIND=testtime \
      CHECKPOINT="${checkpoint}" \
      INFERENCE_COMPRESSION="${compression}" \
      FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START}" \
      FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO}" \
      QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE}" \
      QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      WRITE_SUMMARY=0 \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
  done
done

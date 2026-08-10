#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Evaluate deployment-time compression on already recovered LaRe models for
# VGGT, Pi3, and VGGT-Omega. The default prune_span aliases are recorded in
# prune_plan_all.json; no recovery training or activation cache is repeated.
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"
TESTTIME_BACKBONES="${TESTTIME_BACKBONES:-vggt}"
TESTTIME_RUN="${TESTTIME_RUN:-SCANNET_DEV50,NUSCENES_TRAIN,SCANNET_TEST50,7SCENES_TEST,NUSCENES_VAL,WAYMO_VALIDATION}"
TESTTIME_PRUNE_PLAN_NAME="${TESTTIME_PRUNE_PLAN_NAME:-prune_span12,prune_span16,prune_span20,prune_span24}"
TESTTIME_RECOVERY_PRUNE_STRATEGY="${TESTTIME_RECOVERY_PRUNE_STRATEGY:-ours_double}"
TESTTIME_RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME:-recovery_ghost_interval_sequential}"
TESTTIME_COMPRESSIONS="${TESTTIME_COMPRESSIONS:-fastvggt}"

FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START:-0}"
FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO:-0.9}"
QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE:-w4a4}"
# This is the explicitly named official-parameter transfer baseline.  Override
# it with parameters recalibrated on the final pruned+recovered model when
# available.
QUANTVGGT_PARAMS_DIR="${QUANTVGGT_PARAMS_DIR:-submodules/quantvggt/w4a4_official}"

for compression in ${TESTTIME_COMPRESSIONS//,/ }; do
  case "${compression}" in
    fastvggt|quantvggt) ;;
    *)
      echo "error: TESTTIME_COMPRESSIONS entries must be fastvggt or quantvggt; got ${compression}" >&2
      exit 1
      ;;
  esac

  for backbone in ${TESTTIME_BACKBONES//,/ }; do
    quant_params_dir="${QUANTVGGT_PARAMS_DIR}"
    if [[ "${compression}" == "quantvggt" ]]; then
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
      physical_selector="$(python - "${OUTPUT_ROOT}/${backbone}" "${TESTTIME_RECOVERY_PRUNE_STRATEGY}" "${prune_plan_name}" <<'PY'
import re
import sys
from pathlib import Path
from analysis.prune_plan import strategy_interval_selectors

match = re.fullmatch(r"prune_span(\d+)", sys.argv[3])
if match is None:
    raise SystemExit(f"error: expected prune_spanN, got {sys.argv[3]!r}")
drop = int(match.group(1))
selectors = strategy_interval_selectors(
    Path(sys.argv[1]), strategies=[sys.argv[2]],
    min_drop=drop, max_drop=drop, drop_step=1,
)
if len(selectors) != 1:
    raise SystemExit(f"error: expected one selector, got {selectors}")
print(selectors[0])
PY
    )"
      checkpoint="${OUTPUT_ROOT}/${backbone}/recovery/${physical_selector}/${TESTTIME_RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "error: missing recovered checkpoint: ${checkpoint}" >&2
        echo "       finish step 9 for ${backbone}/${prune_plan_name}/${TESTTIME_RECOVERY_PLAN_NAME} first" >&2
        exit 1
      fi

      printf '\n[testtime] backbone=%s span=%s recovery=%s compression=%s datasets=%s\n' \
        "${backbone}" "${prune_plan_name#prune_span}" "${TESTTIME_RECOVERY_PLAN_NAME}" "${compression}" "${TESTTIME_RUN}"
      BACKBONES="${backbone}" \
        RUN="${TESTTIME_RUN}" \
        PRUNE=1 \
        PRUNE_PLAN_ROOT="${OUTPUT_ROOT}/${backbone}" \
        RECOVERY_PLAN_ROOT="${OUTPUT_ROOT}/${backbone}" \
        PRUNE_PLAN_NAME="${physical_selector}" \
        RECOVERY_PLAN_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
        RECOVERY_OUTPUT_NAME="${TESTTIME_RECOVERY_PLAN_NAME}" \
        RECOVERY_RESULT_KIND=testtime \
        CHECKPOINT="${checkpoint}" \
        INFERENCE_COMPRESSION="${compression}" \
        FASTVGGT_MERGE_START="${FASTVGGT_MERGE_START}" \
        FASTVGGT_MERGE_RATIO="${FASTVGGT_MERGE_RATIO}" \
        QUANTVGGT_DTYPE="${QUANTVGGT_DTYPE}" \
        QUANTVGGT_PARAMS_DIR="${quant_params_dir}" \
        EXPORT_VIS_2D=0 \
        EXPORT_VIS_3D=0 \
        WRITE_SUMMARY=0 \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        MAX_SCENES="${MAX_SCENES}" \
        DEVICE="${DEVICE}" \
        bash "${REPO_ROOT}/scripts/run_eval.sh"
    done
  done
done

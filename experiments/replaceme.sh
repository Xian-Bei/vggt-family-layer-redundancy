#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Build both ReplaceMe num_A=1/2 structures from all-token angular distance,
# select Single/Double using their four-calibration pure-pruning pose errors,
# then fit recovery for the selected physical pruning plan only.
BACKBONES="${BACKBONES:-vggt,pi3,vggt_omega}"
CALIB_DATASETS="${CALIB_DATASETS:-scannet_dev50,7scenes_train,nuscenes_train,waymo_training}"
EVAL_DATASETS="${EVAL_DATASETS:-scannet_test50,7scenes_test,nuscenes_val,waymo_validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-main_results}"
REDUNDANCY_ROOT="${REDUNDANCY_ROOT:-analysis_outputs_v3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SCENES="${MAX_SCENES:-0}"
PLAN_MAX_DROP="${PLAN_MAX_DROP:-24}"
RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME:-recovery_replaceme_ls}"
REPLACEME_LS_ALPHA="${REPLACEME_LS_ALPHA:-0}"
RECOVERY_CHUNK_ROWS="${RECOVERY_CHUNK_ROWS:-16384}"
RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN:-1}"
BENCHMARK_MAX_SCENES="${BENCHMARK_MAX_SCENES:-0}"
BENCHMARK_MAX_FRAMES="${BENCHMARK_MAX_FRAMES:-30}"
BENCHMARK_GPU_POLL_MS="${BENCHMARK_GPU_POLL_MS:-200}"
BENCHMARK_FORCE_RERUN="${BENCHMARK_FORCE_RERUN:-0}"
BENCHMARK_WAIT_FOR_GPU="${BENCHMARK_WAIT_FOR_GPU:-1}"
SKIP_EFFICIENCY="${SKIP_EFFICIENCY:-0}"
export COMPACT_LOG="${COMPACT_LOG:-1}"

if [[ "${OUTPUT_ROOT}" = /* || "${REDUNDANCY_ROOT}" = /* ]]; then
  echo "error: OUTPUT_ROOT and REDUNDANCY_ROOT must be repository-relative paths" >&2
  exit 1
fi
if [[ "${RECOVERY_PLAN_NAME}" != "recovery_replaceme_ls" ]]; then
  echo "error: this runner is intentionally fixed to recovery_replaceme_ls" >&2
  exit 1
fi
if [[ "${DEVICE}" != cuda* ]]; then
  echo "error: ReplaceMe evaluation must run on CUDA; got DEVICE=${DEVICE@Q}" >&2
  exit 1
fi
for integer_name in MAX_SCENES PLAN_MAX_DROP RECOVERY_CHUNK_ROWS BENCHMARK_MAX_SCENES BENCHMARK_MAX_FRAMES BENCHMARK_GPU_POLL_MS; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "error: ${integer_name} must be a non-negative integer" >&2
    exit 1
  fi
done
if [[ "${BENCHMARK_MAX_FRAMES}" -eq 0 || "${BENCHMARK_GPU_POLL_MS}" -eq 0 ]]; then
  echo "error: BENCHMARK_MAX_FRAMES and BENCHMARK_GPU_POLL_MS must be positive" >&2
  exit 1
fi
if [[ "${RECOVERY_FORCE_RERUN}" != 0 && "${RECOVERY_FORCE_RERUN}" != 1 ]]; then
  echo "error: RECOVERY_FORCE_RERUN must be 0 or 1" >&2
  exit 1
fi
if [[ "${BENCHMARK_FORCE_RERUN}" != 0 && "${BENCHMARK_FORCE_RERUN}" != 1 ]]; then
  echo "error: BENCHMARK_FORCE_RERUN must be 0 or 1" >&2
  exit 1
fi
if [[ "${BENCHMARK_WAIT_FOR_GPU}" != 0 && "${BENCHMARK_WAIT_FOR_GPU}" != 1 ]]; then
  echo "error: BENCHMARK_WAIT_FOR_GPU must be 0 or 1" >&2
  exit 1
fi
if [[ "${SKIP_EFFICIENCY}" != 0 && "${SKIP_EFFICIENCY}" != 1 ]]; then
  echo "error: SKIP_EFFICIENCY must be 0 or 1" >&2
  exit 1
fi

# shellcheck source=scripts/run_backbones_lib.sh
source "${REPO_ROOT}/scripts/run_backbones_lib.sh"
parse_backbones_list
REPLACEME_BACKBONES=("${BB_LIST[@]}")
parse_dataset_list "${CALIB_DATASETS}" "CALIB_DATASETS" "${DATASET_KEYS[@]}"
CALIB_LIST=("${PARSED_DATASETS[@]}")
parse_dataset_list "${EVAL_DATASETS}" "EVAL_DATASETS" "${DATASET_KEYS[@]}"
EVAL_LIST=("${PARSED_DATASETS[@]}")

expected_calib=(SCANNET_DEV50 7SCENES_TRAIN NUSCENES_TRAIN WAYMO_TRAINING)
expected_eval=(SCANNET_TEST50 7SCENES_TEST NUSCENES_VAL WAYMO_VALIDATION)
if [[ "${CALIB_LIST[*]}" != "${expected_calib[*]}" ]]; then
  echo "error: CALIB_DATASETS must be exactly ${expected_calib[*]}; got ${CALIB_LIST[*]}" >&2
  exit 1
fi
if [[ "${EVAL_LIST[*]}" != "${expected_eval[*]}" ]]; then
  echo "error: EVAL_DATASETS must be exactly ${expected_eval[*]}; got ${EVAL_LIST[*]}" >&2
  exit 1
fi

work_root="${OUTPUT_ROOT}/.replaceme_work/max_scenes_${MAX_SCENES}"
mkdir -p "${work_root}"
cleanup() {
  local status=$?
  if [[ "${status}" -eq 0 && "${SKIP_EFFICIENCY}" -eq 0 ]]; then
    rm -rf -- "${work_root}"
    rmdir --ignore-fail-on-non-empty "${OUTPUT_ROOT}/.replaceme_work" 2>/dev/null || true
  elif [[ "${status}" -eq 0 ]]; then
    echo "[ReplaceMe] non-efficiency stages complete; temporary data retained for benchmark backfill at ${work_root}"
  else
    echo "[ReplaceMe] failed; resumable temporary data retained at ${work_root}" >&2
  fi
}
trap cleanup EXIT

budget_for_backbone() {
  case "$1" in
    vggt) echo 10 ;;
    pi3) echo 16 ;;
    vggt_omega) echo 18 ;;
    *) echo "error: no ReplaceMe budget for backbone=$1" >&2; return 1 ;;
  esac
}

graph_has_angular() {
  local graph="$1"
  GRAPH_PATH="${graph}" python -c '
import os
import numpy as np
with np.load(os.environ["GRAPH_PATH"], allow_pickle=False) as payload:
    raise SystemExit(0 if "angular_mean" in payload.files else 1)
'
}

prepare_redundancy_inputs() {
  local backbone="$1"
  local prepared_root="$2"
  local dataset
  local -a missing_datasets=()
  for dataset in scannet_dev50 7scenes_train nuscenes_train waymo_training; do
    local prepared_graph="${prepared_root}/${backbone}/${dataset}/redundancy/redundancy_graph_all.npz"
    if [[ -f "${prepared_graph}" ]] && graph_has_angular "${prepared_graph}"; then
      continue
    fi
    local graph="${REDUNDANCY_ROOT}/${backbone}/${dataset}/redundancy/redundancy_graph_all.npz"
    if [[ ! -f "${graph}" ]]; then
      missing_datasets+=("${dataset}")
      continue
    fi
    if graph_has_angular "${graph}"; then
      local destination="${prepared_root}/${backbone}/${dataset}/redundancy"
      mkdir -p "${destination}"
      cp -- "${graph}" "${destination}/redundancy_graph_all.npz"
    else
      missing_datasets+=("${dataset}")
    fi
  done
  if [[ "${#missing_datasets[@]}" -gt 0 ]]; then
    local missing_csv
    missing_csv="$(IFS=,; echo "${missing_datasets[*]}")"
    printf '[ReplaceMe] angular redundancy is missing for %s/%s; computing it now\n' \
      "${backbone}" "${missing_csv}"
    OUTPUT_ROOT="${prepared_root}" \
      BACKBONES="${backbone}" \
      RUN="${missing_csv}" \
      CACHE="" \
      SKIP_REDUNDANCY=0 \
      METRICS=angular \
      SIM_TOKEN_REGIONS=all \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"
  fi
  for dataset in scannet_dev50 7scenes_train nuscenes_train waymo_training; do
    local prepared_graph="${prepared_root}/${backbone}/${dataset}/redundancy/redundancy_graph_all.npz"
    if [[ ! -f "${prepared_graph}" ]] || ! graph_has_angular "${prepared_graph}"; then
      echo "error: angular redundancy preparation failed: ${prepared_graph}" >&2
      return 1
    fi
  done
}

merge_replaceme_plans() {
  local source_bundle="$1"
  local target_bundle="$2"
  SOURCE_BUNDLE="${source_bundle}" TARGET_BUNDLE="${target_bundle}" python - <<'PY'
import json
import os
from pathlib import Path

source_path = Path(os.environ["SOURCE_BUNDLE"])
target_path = Path(os.environ["TARGET_BUNDLE"])
source = json.loads(source_path.read_text(encoding="utf-8"))
target = json.loads(target_path.read_text(encoding="utf-8"))
for field in ("schema_version", "backbone", "num_units", "unit_ids"):
    if source.get(field) != target.get(field):
        raise SystemExit(
            f"cannot merge ReplaceMe plan: {field} mismatch "
            f"({source.get(field)!r} != {target.get(field)!r})"
        )

source_sets = source.get("sets", {})
target_sets = target.setdefault("sets", {})
if not isinstance(source_sets, dict) or not isinstance(target_sets, dict):
    raise SystemExit("invalid pruning bundle sets object")
strategies = target.setdefault("strategies", {})
if not isinstance(strategies, dict):
    raise SystemExit("invalid pruning bundle strategies object")
merged_points = 0
for strategy_name in ("replaceme_single", "replaceme_double"):
    strategy = source.get("strategies", {}).get(strategy_name)
    if not isinstance(strategy, dict):
        raise SystemExit(f"staged bundle has no {strategy_name} strategy")
    points = strategy.get("points", {})
    if not isinstance(points, dict) or not points:
        raise SystemExit(f"staged {strategy_name} strategy has no points")
    for alias in points.values():
        alias = str(alias)
        record = source_sets.get(alias)
        if not isinstance(record, dict):
            raise SystemExit(f"staged plan is missing set {alias}")
        existing = target_sets.get(alias)
        if existing is not None and existing != record:
            existing_sources = {
                str(existing.get("source_strategy", "")),
                *(str(item) for item in existing.get("source_strategies", [])),
            }
            if strategy_name not in existing_sources:
                raise SystemExit(
                    f"refusing to replace non-ReplaceMe set collision {alias}: "
                    f"sources={sorted(existing_sources)}"
                )
        target_sets[alias] = record
    strategies[strategy_name] = strategy
    merged_points += len(points)

# Parameter accounting is architecture-only.  Keep the main bundle's record,
# but fill it from the staged bundle if the initialized bundle lacks one.
if not bool(target.get("param_accounting", {}).get("available", False)):
    target["param_accounting"] = source.get("param_accounting", {})

temporary = target_path.with_suffix(target_path.suffix + ".replaceme.tmp")
temporary.write_text(json.dumps(target, indent=2), encoding="utf-8")
temporary.replace(target_path)
print(f"Merged ReplaceMe Single/Double ({merged_points} points) into {target_path}")
PY
}

candidate_selectors() {
  local plan_root="$1"
  local budget="$2"
  PLAN_ROOT="${plan_root}" BUDGET="${budget}" python - <<'PY'
import json
import os
from pathlib import Path
from analysis.prune_plan import physical_pruning_selector_from_record

root = Path(os.environ["PLAN_ROOT"])
budget = str(int(os.environ["BUDGET"]))
bundle = json.loads((root / "prune_plan_all.json").read_text(encoding="utf-8"))
for strategy_name in ("replaceme_single", "replaceme_double"):
    strategy = bundle["strategies"][strategy_name]
    alias = str(strategy["points"][budget])
    record = bundle["sets"][alias]
    selector = physical_pruning_selector_from_record(record)
    angular_score = float(strategy.get("scores", {}).get(budget, 0.0))
    print(f"{strategy_name}\t{selector}\t{angular_score:.12g}")
PY
}

select_pruning_candidate() {
  local plan_root="$1"
  local budget="$2"
  local selection_path="$3"
  PLAN_ROOT="${plan_root}" BUDGET="${budget}" SELECTION_PATH="${selection_path}" python - <<'PY'
import json
import math
import os
from pathlib import Path

from analysis.prune_plan import physical_pruning_selector_from_record
from analysis.prune_plan.eval_plots import pose_selection_score

root = Path(os.environ["PLAN_ROOT"])
budget = str(int(os.environ["BUDGET"]))
selection_path = Path(os.environ["SELECTION_PATH"])
bundle = json.loads((root / "prune_plan_all.json").read_text(encoding="utf-8"))
datasets = ("scannet_dev50", "7scenes_train", "nuscenes_train", "waymo_training")
candidates = []
for strategy_name in ("replaceme_single", "replaceme_double"):
    strategy = bundle["strategies"][strategy_name]
    alias = str(strategy["points"][budget])
    selector = physical_pruning_selector_from_record(bundle["sets"][alias])
    dataset_scores = {}
    for dataset in datasets:
        record_path = root / dataset / "eval" / "pruning" / selector / "eval_record_v3.json"
        if not record_path.is_file():
            raise SystemExit(f"missing calibration pruning result: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        average = record.get("average", {})
        score = pose_selection_score(average) if isinstance(average, dict) else None
        if score is None or not math.isfinite(float(score)):
            raise SystemExit(f"incomplete four-pose pruning metrics: {record_path}")
        dataset_scores[dataset] = float(score)
    mean_score = sum(dataset_scores.values()) / len(dataset_scores)
    candidates.append(
        {
            "strategy": strategy_name,
            "alias": alias,
            "selector": selector,
            "dataset_pose_sums": dataset_scores,
            "mean_pose_sum": mean_score,
            # Match the existing pruning-view deterministic tuple ordering:
            # replaceme_double sorts before replaceme_single on an exact tie.
            "tie_preference": 0 if strategy_name == "replaceme_double" else 1,
        }
    )

# Lower calibration pose-error sum wins; exact ties follow the existing view.
winner = min(candidates, key=lambda item: (item["mean_pose_sum"], item["tie_preference"]))
payload = {
    "schema_version": 1,
    "selection": "minimum mean of APE+ARE+RPE_t+RPE_r over four calibration sets",
    "token_region": "all",
    "budget": int(budget),
    "candidates": candidates,
    "winner": {
        "strategy": winner["strategy"],
        "alias": winner["alias"],
        "selector": winner["selector"],
        "mean_pose_sum": winner["mean_pose_sum"],
    },
}
selection_path.parent.mkdir(parents=True, exist_ok=True)
temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
temporary.replace(selection_path)
print(
    f'{winner["strategy"]}\t{winner["selector"]}\t'
    f'{winner["mean_pose_sum"]:.12g}'
)
PY
}

pruning_candidate_complete() {
  local plan_root="$1"
  local selector="$2"
  PLAN_ROOT="${plan_root}" SELECTOR="${selector}" python -c '
import json, math, os
from pathlib import Path
root = Path(os.environ["PLAN_ROOT"])
selector = os.environ["SELECTOR"]
for dataset in ("scannet_dev50", "7scenes_train", "nuscenes_train", "waymo_training"):
    path = root / dataset / "eval" / "pruning" / selector / "eval_record_v3.json"
    if not path.is_file():
        raise SystemExit(1)
    average = json.loads(path.read_text(encoding="utf-8")).get("average", {})
    values = [average.get(key) for key in ("ape", "are", "rpe_trans", "rpe_rot")]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise SystemExit(1)
'
}

recovery_visualization_complete() {
  local plan_root="$1"
  local selector="$2"
  local method="$3"
  local root="${plan_root}/scannet_test50/eval/recovery/${selector}/${method}"
  [[ -f "${root}/eval_record_v3.json" ]] \
    && find "${root}" -type f -name plot.png -print -quit | grep -q . \
    && find "${root}" -type f -name viser_points.npz -print -quit | grep -q . \
    && find "${root}" -type f -name viser_points_gt.npz -print -quit | grep -q .
}

wait_for_exclusive_gpu() {
  if [[ "${BENCHMARK_WAIT_FOR_GPU}" != 1 ]]; then
    return 0
  fi
  local waited=0
  local processes=""
  while true; do
    processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "${processes//[[:space:]]/}" ]]; then
      if [[ "${waited}" -gt 0 ]]; then
        echo "[ReplaceMe] GPU is idle; starting exclusive benchmark"
      fi
      return 0
    fi
    if (( waited % 1800 == 0 )); then
      printf '[ReplaceMe] benchmark waiting for exclusive GPU (%ss elapsed):\n%s\n' \
        "${waited}" "${processes}"
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

benchmark_jobs=()
for dataset_key in "${EVAL_LIST[@]}"; do
  case "${dataset_key}" in
    SCANNET_TEST50)
      benchmark_jobs+=(--job "scannet|${ROOT_SCANNET_TEST50}|scannet_test50|") ;;
    7SCENES_TEST)
      benchmark_jobs+=(--job "7scenes|${ROOT_7SCENES}|7scenes_test|test") ;;
    NUSCENES_VAL)
      benchmark_jobs+=(--job "nuscenes|${ROOT_NUSCENES}|nuscenes_val|val") ;;
    WAYMO_VALIDATION)
      benchmark_jobs+=(--job "waymo|${ROOT_WAYMO}|waymo_validation|validation") ;;
  esac
done

for bb in "${REPLACEME_BACKBONES[@]}"; do
  bb="${bb//[[:space:]]/}"
  [[ -z "${bb}" ]] && continue
  budget="$(budget_for_backbone "${bb}")"
  plan_root="${OUTPUT_ROOT}/${bb}"
  staged_plan_root="${work_root}/plans/${bb}"
  cache_root="${work_root}/cache"
  # Missing angular graphs are an intermediate only.  Existing graphs are
  # copied and missing ones are computed under mktemp, then all are removed.
  prepared_redundancy_root="${work_root}/redundancy"
  selection_map="${work_root}/selection_${bb}.tsv"
  mkdir -p "${staged_plan_root}"

  if [[ ! -f "${plan_root}/prune_plan_all.json" ]]; then
    echo "error: initialize ${plan_root}/prune_plan_all.json with main.sh first" >&2
    exit 1
  fi
  set_backend_pythonpath "${bb}"

  printf '\n[ReplaceMe 1/10] prepare all-token angular redundancy for four calibration sets\n'
  prepare_redundancy_inputs "${bb}" "${prepared_redundancy_root}"

  printf '\n[ReplaceMe 2/10] build Single/Double pruning plans: backbone=%s budget=%s\n' "${bb}" "${budget}"
  OUTPUT_ROOT="${prepared_redundancy_root}" python -m analysis.prune_plan.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset "${CALIB_DATASETS}" \
    --backbone "${bb}" \
    --metric angular \
    --strategies replaceme_single,replaceme_double \
    --max-drop-budget "${PLAN_MAX_DROP}" \
    --token-region all \
    --prune-plan-root "${staged_plan_root}" \
    --viz-out-dir "${staged_plan_root}"
  merge_replaceme_plans \
    "${staged_plan_root}/prune_plan_all.json" \
    "${plan_root}/prune_plan_all.json"

  printf '\n[ReplaceMe 3/10] evaluate Single/Double pure pruning on four calibration sets\n'
  candidate_file="${work_root}/candidates_${bb}.tsv"
  candidate_selectors "${plan_root}" "${budget}" > "${candidate_file}"
  declare -A evaluated_selectors=()
  while IFS=$'\t' read -r candidate_strategy candidate_selector angular_score; do
    if [[ -z "${candidate_selector}" || -n "${evaluated_selectors[${candidate_selector}]:-}" ]]; then
      continue
    fi
    evaluated_selectors["${candidate_selector}"]=1
    printf '[ReplaceMe] pruning candidate: %s selector=%s angular=%s\n' \
      "${candidate_strategy}" "${candidate_selector}" "${angular_score}"
    if pruning_candidate_complete "${plan_root}" "${candidate_selector}"; then
      echo "[ReplaceMe] reuse complete four-calibration pruning result: ${candidate_selector}"
      continue
    fi
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
      BACKBONES="${bb}" \
      RUN="${CALIB_DATASETS}" \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${candidate_selector}" \
      RECOVERY_PLAN_NAME="" \
      CHECKPOINT="" \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      EXPORT_VIS_2D=0 \
      EXPORT_VIS_3D=0 \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
  done < "${candidate_file}"

  printf '\n[ReplaceMe 4/10] select Single/Double from calibration pruning errors\n'
  selection_record="${plan_root}/replaceme_selection_budget${budget}.json"
  IFS=$'\t' read -r selected_strategy selector selection_score < <(
    select_pruning_candidate "${plan_root}" "${budget}" "${selection_record}"
  )
  if [[ -z "${selected_strategy}" || -z "${selector}" ]]; then
    echo "error: failed to select ReplaceMe pruning plan at budget ${budget}" >&2
    exit 1
  fi
  printf '[ReplaceMe] selected %s: selector=%s calibration_pose_sum=%s\n' \
    "${selected_strategy}" "${selector}" "${selection_score}"

  printf '\n[ReplaceMe 5/10] merge ReplaceMe-LS recovery plan\n'
  python -m analysis.prune_plan.build_recovery_plan \
    --experiment "${bb}" \
    --backbone "${bb}" \
    --prune-plan-root "${plan_root}" \
    --recovery-plan-root "${plan_root}" \
    --method "${RECOVERY_PLAN_NAME}" \
    --lora-prefix 1

  printf '\n[ReplaceMe 6/10] cache only the selected pruning plan\n'
  OUTPUT_ROOT="${cache_root}" \
    BACKBONES="${bb}" \
    RUN="${CALIB_DATASETS}" \
    CACHE="${CALIB_DATASETS}" \
    SKIP_REDUNDANCY=1 \
    MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    PRUNE_PLAN_ROOT="${plan_root}" \
    RECOVERY_PLAN_ROOT="${plan_root}" \
    PRUNE_PLAN_NAME="${selector}" \
    RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
    bash "${REPO_ROOT}/scripts/run_cache_redundancy.sh"

  printf '\n[ReplaceMe 7/10] fit selected plan; evaluate three non-visual test sets\n'
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
    BACKBONES="${bb}" \
    DATASETS="${CALIB_DATASETS}" \
    EVAL="7scenes_test,nuscenes_val,waymo_validation" \
    PRUNE_PLAN_ROOT="${plan_root}" \
    RECOVERY_PLAN_ROOT="${plan_root}" \
    RECOVERY_CACHE_ROOT="${cache_root}" \
    PRUNE_PLAN_NAME="${selector}" \
    RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
    REPLACEME_LS_ALPHA="${REPLACEME_LS_ALPHA}" \
    RECOVERY_CHUNK_ROWS="${RECOVERY_CHUNK_ROWS}" \
    RECOVERY_FORCE_RERUN="${RECOVERY_FORCE_RERUN}" \
    MAX_SCENES="${MAX_SCENES}" \
    EVAL_MAX_SCENES="${MAX_SCENES}" \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    bash "${REPO_ROOT}/scripts/run_recovery.sh"

  checkpoint="${plan_root}/recovery/${selector}/${RECOVERY_PLAN_NAME}/checkpoints/latest.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "error: recovery did not produce ${checkpoint}" >&2
    exit 1
  fi

  printf '\n[ReplaceMe 8/10] evaluate ScanNet test50 and export 2D/3D visualization\n'
  if [[ "${RECOVERY_FORCE_RERUN}" == 0 ]] \
      && recovery_visualization_complete "${plan_root}" "${selector}" "${RECOVERY_PLAN_NAME}"; then
    echo "[ReplaceMe] reuse complete ScanNet test50 metrics and 2D/3D visualization"
  else
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
      BACKBONES="${bb}" \
      RUN="scannet_test50" \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="${plan_root}" \
      PRUNE_PLAN_NAME="${selector}" \
      RECOVERY_PLAN_ROOT="${plan_root}" \
      RECOVERY_PLAN_NAME="${RECOVERY_PLAN_NAME}" \
      CHECKPOINT="${checkpoint}" \
      RECOVERY_FORCE_RERUN=1 \
      MAX_SCENES="${MAX_SCENES}" \
      DEVICE="${DEVICE}" \
      EXPORT_VIS_2D=1 \
      EXPORT_VIS_3D=1 \
      bash "${REPO_ROOT}/scripts/run_eval.sh"
  fi

  printf '%s\t%s\treplaceme\t%s\t%s\n' \
    "${selector}" "${budget}" "${selected_strategy}" "${selection_score}" > "${selection_map}"

  if [[ "${SKIP_EFFICIENCY}" == 1 ]]; then
    printf '\n[ReplaceMe 9/10] skip efficiency benchmark as requested\n'
  else
    printf '\n[ReplaceMe 9/10] benchmark latency and peak VRAM on four test sets\n'
    benchmark_force_args=()
    if [[ "${BENCHMARK_FORCE_RERUN}" == 1 ]]; then
      benchmark_force_args+=(--force-rerun)
    fi
    wait_for_exclusive_gpu
    python -m analysis.eval.inference_benchmark \
      --backbone "${bb}" \
      --device "${DEVICE}" \
      --variant recovery \
      --inference-compression none \
      --prune-plan-root "${plan_root}" \
      --prune-plan-name "${selector}" \
      --recovery-plan-root "${plan_root}" \
      --recovery-plan-name "${RECOVERY_PLAN_NAME}" \
      --checkpoint "${checkpoint}" \
      --selection-map "${selection_map}" \
      --selection-datasets "${CALIB_DATASETS}" \
      --output-root "${OUTPUT_ROOT}" \
      --sample-count 10 \
      --max-scenes "${BENCHMARK_MAX_SCENES}" \
      --max-frames "${BENCHMARK_MAX_FRAMES}" \
      --gpu-poll-interval-ms "${BENCHMARK_GPU_POLL_MS}" \
      "${benchmark_jobs[@]}" \
      "${benchmark_force_args[@]}"
  fi

  printf '\n[ReplaceMe 10/10] validate canonical outputs\n'
  BACKBONE="${bb}" PLAN_ROOT="${plan_root}" SELECTOR="${selector}" \
    RECOVERY_NAME="${RECOVERY_PLAN_NAME}" SELECTION_RECORD="${selection_record}" \
    SKIP_EFFICIENCY="${SKIP_EFFICIENCY}" python - <<'PY'
import json
import os
from pathlib import Path

from analysis.schema.v3 import UNIFIED_METRIC_KEYS

backbone = os.environ["BACKBONE"]
root = Path(os.environ["PLAN_ROOT"])
selector = os.environ["SELECTOR"]
method = os.environ["RECOVERY_NAME"]
selection_record = Path(os.environ["SELECTION_RECORD"])
skip_efficiency = os.environ["SKIP_EFFICIENCY"] == "1"
missing = []
if not selection_record.is_file():
    missing.append(str(selection_record))
else:
    selection = json.loads(selection_record.read_text(encoding="utf-8"))
    if selection.get("winner", {}).get("selector") != selector:
        missing.append(f"winner mismatch in {selection_record}")
for dataset in ("scannet_test50", "7scenes_test", "nuscenes_val", "waymo_validation"):
    base = root / dataset
    record = base / "eval" / "recovery" / selector / method / "eval_record_v3.json"
    benchmark = base / "eval" / "testtime" / "inference_benchmark_v3.json"
    if not record.is_file():
        missing.append(str(record))
    else:
        average = json.loads(record.read_text(encoding="utf-8")).get("average", {})
        absent_metrics = [key for key in UNIFIED_METRIC_KEYS if key not in average]
        if absent_metrics:
            missing.append(f"{record}: missing metrics {absent_metrics}")
    if not skip_efficiency and not benchmark.is_file():
        missing.append(str(benchmark))
vis_root = root / "scannet_test50" / "eval" / "recovery" / selector / method
if not any(vis_root.rglob("plot.png")):
    missing.append(f"{vis_root}/**/plot.png")
if not any(vis_root.rglob("viser_points.npz")):
    missing.append(f"{vis_root}/**/vis/pred/viser_points.npz")
if not any(vis_root.rglob("viser_points_gt.npz")):
    missing.append(f"{vis_root}/**/vis/gt/viser_points_gt.npz")
if missing:
    raise SystemExit("missing expected ReplaceMe outputs:\n  " + "\n  ".join(missing))
print(f"Validated ReplaceMe outputs for {backbone}: selector={selector}")
PY

  # These directories contain only this script's mktemp-backed staging data.
  rm -rf -- \
    "${work_root}/cache/${bb}" \
    "${work_root}/plans/${bb}"
done

echo "ReplaceMe main-result run complete. Final Table can scan ${OUTPUT_ROOT}."

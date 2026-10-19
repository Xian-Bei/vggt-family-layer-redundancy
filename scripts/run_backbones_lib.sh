# Shared backbone list helpers for run_*.sh scripts.
# Source after REPO_ROOT is set.
#
#   BACKBONES=vggt,pi3,vggt_omega,mapanything,mapanything_v1
#   MAX_FRAMES=50   # same cap for cache, redundancy, eval (dataset adaptors apply it)
#   EXPORT_VIS_2D=0 # per-scene plot.png (trajectory)
#   EXPORT_VIS_3D=0 # vis/pred viser NPZ for analysis.view
#   EXPORT_VIS=1    # shorthand: enable both 2D and 3D
#   VIS_MAX_POINTS=0
#   ASYNC_SCORE_WORKERS=12
#   ASYNC_SCORE_MAX_PENDING=36
# Outputs use ${OUTPUT_ROOT}/<backbone>/<dataset>/...

: "${MAX_FRAMES:=30}"
: "${OUTPUT_ROOT:=analysis_outputs_v3}"
: "${ASYNC_SCORE_WORKERS:=12}"
: "${ASYNC_SCORE_MAX_PENDING:=36}"
: "${ROOT_SCANNET_DEV50:=${SCANNET_ROOT_DEV50:-data/scans_dev50}}"
: "${ROOT_SCANNET_TEST50:=${SCANNET_ROOT_TEST50:-data/scans_test50}}"
: "${ROOT_SCANNET_DEV_EXTRA100:=${SCANNET_ROOT_DEV_EXTRA100:-data/scans_dev_extra100}}"
: "${ROOT_7SCENES:=data/7scenes}"
: "${ROOT_NRGBD:=data/neural_rgbd_data}"
: "${ROOT_NUSCENES:=data/nuscenes}"
: "${ROOT_NUSCENES_TRAIN_EXTRA100:=data/nuscenes_train_extra100}"
: "${ROOT_WAYMO:=data/waymo}"

_lare_error_context() {
  local context=""
  local name value
  for name in bb backbone prune_plan_name recovery_plan_name dataset_key output_dataset; do
    value="${!name:-}"
    if [[ -n "${value}" ]]; then
      context+=" ${name}=${value}"
    fi
  done
  [[ -n "${context}" ]] && printf '  context:%s\n' "${context}" >&2
}

_lare_report_error() {
  local status="$1"
  local command="$2"
  local source="$3"
  local line="$4"
  local signal=""
  local signal_number

  # Diagnostics must never recursively trigger ERR or hide the original status.
  trap - ERR
  set +e
  if [[ "${status}" =~ ^[0-9]+$ ]] && (( status > 128 )); then
    signal_number=$((status - 128))
    signal="$(kill -l "${signal_number}" 2>/dev/null || true)"
  fi

  printf '\n[command-error]\n' >&2
  printf '  timestamp: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" >&2
  printf '  pid: %s\n' "${BASHPID:-$$}" >&2
  printf '  script: %s\n' "${LARE_DIAGNOSTIC_SCRIPT:-${source}}" >&2
  printf '  location: %s:%s\n' "${source}" "${line}" >&2
  printf '  exit: %s' "${status}" >&2
  [[ -n "${signal}" ]] && printf ' (signal %s)' "${signal}" >&2
  printf '\n  command: %s\n' "${command}" >&2
  _lare_error_context
  printf '  note: native crashes such as SIGSEGV may not produce a Python traceback; check the kernel log for the same timestamp.\n' >&2
  return "${status}"
}

_lare_report_signal() {
  local signal="$1"
  local status="$2"
  trap - ERR INT TERM HUP
  set +e
  printf '\n[script-signal]\n  timestamp: %s\n  pid: %s\n  script: %s\n  signal: %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${BASHPID:-$$}" \
    "${LARE_DIAGNOSTIC_SCRIPT:-${BASH_SOURCE[1]:-$0}}" "${signal}" >&2
  _lare_error_context
  exit "${status}"
}

install_error_diagnostics() {
  LARE_DIAGNOSTIC_SCRIPT="${1:-${BASH_SOURCE[1]:-$0}}"
  # ERR inheritance is needed for failures inside functions and command substitutions.
  set -E
  trap '_lare_report_error "$?" "$BASH_COMMAND" "${BASH_SOURCE[0]:-${LARE_DIAGNOSTIC_SCRIPT:-$0}}" "${LINENO:-unknown}"' ERR
  trap '_lare_report_signal INT 130' INT
  trap '_lare_report_signal TERM 143' TERM
  trap '_lare_report_signal HUP 129' HUP
}

truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

compact_log() {
  truthy "${COMPACT_LOG:-0}"
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

DATASET_KEYS=(
  SCANNET_DEV50
  SCANNET_TEST50
  SCANNET_DEV_EXTRA100
  7SCENES_TEST
  7SCENES_TRAIN
  NRGBD
  NUSCENES_VAL
  NUSCENES_TRAIN
  NUSCENES_TRAIN_EXTRA100
  WAYMO_VALIDATION
  WAYMO_TRAINING
)

parse_backbones_list() {
  bb_raw="${BACKBONES//,/ }"
  read -r -a BB_LIST <<< "${bb_raw}"
  BB_N=0
  for bb in "${BB_LIST[@]}"; do
    bb="${bb//[[:space:]]/}"
    [[ -z "${bb}" ]] && continue
    case "${bb}" in
      map-anything) bb="mapanything" ;;
      map-anything-v1) bb="mapanything_v1" ;;
    esac
    case "${bb}" in
      vggt|pi3|vggt_omega|mapanything|mapanything_v1) ;;
      *)
        echo "error: unsupported backbone ${bb}; expected vggt, pi3, vggt_omega, mapanything, or mapanything_v1" >&2
        exit 1
        ;;
    esac
    BB_LIST[$((BB_N))]="${bb}"
    BB_N=$((BB_N + 1))
  done
  if [[ "${BB_N}" -eq 0 ]]; then
    echo "error: BACKBONES is empty" >&2
    exit 1
  fi
}

list_has() {
  local needle="$1"
  shift || true
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

join_by_comma() {
  local out=""
  local item
  for item in "$@"; do
    [[ -z "${item}" ]] && continue
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${item}"
  done
  echo "${out}"
}

dataset_keys_help() {
  join_by_comma "$@"
}

parse_dataset_list() {
  local raw_list="$1"
  local label="${2:-DATASETS}"
  shift 2 || true
  local allowed_keys=("$@")
  local raw key
  PARSED_DATASETS=()
  raw_list="${raw_list//,/ }"
  for raw in ${raw_list}; do
    key="${raw//[[:space:]]/}"
    [[ -z "${key}" ]] && continue
    key="${key//-/_}"
    key="${key^^}"
    if ! list_has "${key}" "${allowed_keys[@]}"; then
      echo "error: unsupported dataset key '${raw}' in ${label}; expected one of: $(dataset_keys_help "${allowed_keys[@]}")" >&2
      exit 1
    fi
    if ! list_has "${key}" "${PARSED_DATASETS[@]}"; then
      PARSED_DATASETS+=("${key}")
    fi
  done
}

set_backend_pythonpath() {
  local backbone="$1"
  export PYTHONPATH="$(python -m analysis.backbones.bootstrap --backbone "${backbone}"):${PYTHONPATH:-}"
}

experiment_output_hint() {
  local suffix="$1"
  echo "   ${suffix}: ${OUTPUT_ROOT}/<backbone>/..."
}

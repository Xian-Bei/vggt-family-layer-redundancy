# Shared backbone list helpers for run_*.sh scripts.
# Source after REPO_ROOT is set.
#
#   BACKBONES=vggt,pi3,vggt_omega
#   MAX_FRAMES=50   # same cap for cache, redundancy, eval (dataset adaptors apply it)
#   EXPORT_VIS_2D=0 # per-scene plot.png (trajectory)
#   EXPORT_VIS_3D=0 # vis/pred viser NPZ for analysis.view
#   EXPORT_VIS=1    # shorthand: enable both 2D and 3D
#   VIS_MAX_POINTS=0
# Outputs use ${OUTPUT_ROOT}/<backbone>/<dataset>/...

: "${CONDA_ENV:=vggt}"
: "${MAX_FRAMES:=30}"
: "${OUTPUT_ROOT:=analysis_outputs_v3}"
: "${ROOT_SCANNET_DEV50:=${SCANNET_ROOT_DEV50:-data/scans_dev50}}"
: "${ROOT_SCANNET_TEST50:=${SCANNET_ROOT_TEST50:-data/scans_test50}}"
: "${ROOT_7SCENES:=data/7scenes}"
: "${ROOT_NRGBD:=data/neural_rgbd_data}"
: "${ROOT_NUSCENES:=data/nuscenes}"
: "${ROOT_WAYMO:=data/waymo}"

DATASET_KEYS=(
  SCANNET_DEV50
  SCANNET_TEST50
  7SCENES_TEST
  7SCENES_TRAIN
  NRGBD
  NUSCENES_VAL
  NUSCENES_TRAIN
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
      vggt|pi3|vggt_omega) ;;
      *)
        echo "error: unsupported backbone ${bb}; expected vggt, pi3, or vggt_omega" >&2
        exit 1
        ;;
    esac
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
  export PYTHONPATH="$(conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.backbones.bootstrap --backbone "${backbone}"):${PYTHONPATH:-}"
}

experiment_output_hint() {
  local suffix="$1"
  echo "   ${suffix}: ${OUTPUT_ROOT}/<backbone>/..."
}

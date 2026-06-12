# Shared backbone list helpers for run_*.sh scripts.
# Source after REPO_ROOT is set.
#
#   BACKBONES=vggt,fastvggt,pi3
#   MAX_FRAMES=50   # same cap for cache, redundancy, eval (dataset plugins apply it)
#   EXPORT_VIS_2D=0 # per-scene plot.png (trajectory)
#   EXPORT_VIS_3D=0 # vis/pred viser NPZ for analysis.view
#   EXPORT_VIS=1    # shorthand: enable both 2D and 3D
#   VIS_MAX_POINTS=0
# Outputs use analysis_outputs_v3/<backbone>/<dataset>/...

: "${CONDA_ENV:=vggt}"
: "${MAX_FRAMES:=50}"

parse_backbones_list() {
  bb_raw="${BACKBONES//,/ }"
  read -r -a BB_LIST <<< "${bb_raw}"
  BB_N=0
  for bb in "${BB_LIST[@]}"; do
    bb="${bb//[[:space:]]/}"
    [[ -n "${bb}" ]] && BB_N=$((BB_N + 1))
  done
  if [[ "${BB_N}" -eq 0 ]]; then
    echo "error: BACKBONES is empty" >&2
    exit 1
  fi
}

set_backend_pythonpath() {
  local backbone="$1"
  export PYTHONPATH="$(conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.backbones.bootstrap --backbone "${backbone}"):${PYTHONPATH:-}"
}

experiment_output_hint() {
  local suffix="$1"
  echo "   ${suffix}: analysis_outputs_v3/<backbone>/..."
}

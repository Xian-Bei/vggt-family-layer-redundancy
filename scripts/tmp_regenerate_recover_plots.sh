#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs_v3}"
CONDA_ENV="${CONDA_ENV:-vggt}"
BACKBONES="${BACKBONES:-}"

BACKBONE_LIST=()
if [[ -n "${BACKBONES//[[:space:],]/}" ]]; then
  read -r -a _BACKBONES_RAW <<< "${BACKBONES//,/ }"
  for bb in "${_BACKBONES_RAW[@]}"; do
    bb="${bb//[[:space:]]/}"
    [[ -n "${bb}" ]] && BACKBONE_LIST+=("${bb}")
  done
else
  shopt -s nullglob
  for d in "${OUTPUT_ROOT}"/*; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/prune_plan_all.json" || -d "${d}/scannet_dev50" ]] || continue
    BACKBONE_LIST+=("$(basename "${d}")")
  done
  shopt -u nullglob
fi

if [[ "${#BACKBONE_LIST[@]}" -eq 0 ]]; then
  echo "error: no backbone outputs found under ${OUTPUT_ROOT}; set BACKBONES=vggt,pi3 or OUTPUT_ROOT=..." >&2
  exit 1
fi

echo "== regenerate summary/recover plots output_root=${OUTPUT_ROOT} backbones=${BACKBONE_LIST[*]} =="
for bb in "${BACKBONE_LIST[@]}"; do
  plan_root="${PRUNE_PLAN_ROOT:-${OUTPUT_ROOT}/${bb}}"
  echo "   [summary] ${OUTPUT_ROOT}/${bb}/summary_comparison.md"
  OUTPUT_ROOT="${OUTPUT_ROOT}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m analysis.eval.summary_v3 \
      --experiment "${bb}" \
      --prune-plan-root "${plan_root}"
done
echo "== done =="

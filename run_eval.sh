#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# FastVGGT-aligned CLI knobs
DATA_DIR="${DATA_DIR:-data/scans_test50}"
GT_PLY_DIR="${GT_PLY_DIR:-data/scans_test50}"
ROOT_7SCENES="${ROOT_7SCENES:-data/7scenes}"
ROOT_NRGBD="${ROOT_NRGBD:-data/neural_rgbd_data}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_outputs/evaluation}"
CKPT_PATH="${CKPT_PATH:-}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda:0}"
INPUT_FRAME="${INPUT_FRAME:-30}"
NUM_SCENES="${NUM_SCENES:-0}"
KF="${KF:-10}"
SIZE="${SIZE:-518}"
MERGING="${MERGING:-0}"
MERGE_RATIO="${MERGE_RATIO:-0.9}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
BASELINE_BACKEND="${BASELINE_BACKEND:-vggt}"
FAST_BACKEND="${FAST_BACKEND:-fastvggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"
SCANNET_MODELS="${SCANNET_MODELS:-baseline(vggt),fastvggt(fastvggt)}"
MODELS_7ANDN="${MODELS_7ANDN:-baseline(vggt),fastvggt(fastvggt)}"
RUN_SCANNET="${RUN_SCANNET:-1}"
RUN_7ANDN="${RUN_7ANDN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

trim_spaces() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

parse_model_spec() {
  local spec
  spec="$(trim_spaces "$1")"
  if [[ "${spec}" =~ ^([^()]+)\(([^()]+)\)$ ]]; then
    MODEL_NAME_PARSED="$(trim_spaces "${BASH_REMATCH[1]}")"
    MODEL_BACKEND_PARSED="$(trim_spaces "${BASH_REMATCH[2]}")"
    if [[ -z "${MODEL_NAME_PARSED}" || -z "${MODEL_BACKEND_PARSED}" ]]; then
      echo "[error] Invalid model spec: '${spec}'" >&2
      exit 1
    fi
  else
    echo "[error] Invalid model spec: '${spec}', expected <model>(<backend>)" >&2
    exit 1
  fi
}

set_backend_pythonpath() {
  local backend="$1"
  if [[ "${backend}" == "fastvggt" ]]; then
    export PYTHONPATH="${FASTVGGT_VENDOR_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
  else
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  fi
}

run_scannet() {
  local model_name="$1"
  local backend="$2"
  local out_dir="${OUTPUT_DIR}/scannet/${model_name}"
  local num_scenes_arg="${NUM_SCENES}"
  local input_frame_arg="${INPUT_FRAME}"

  args=(
    --data_dir "${DATA_DIR}"
    --gt_ply_dir "${GT_PLY_DIR}"
    --output_path "${out_dir}"
    --input_frame "${input_frame_arg}"
    --depth_conf_thresh "${DEPTH_CONF_THRESH}"
    --chamfer_max_dist "${CHAMFER_MAX_DIST}"
  )
  if [[ "${num_scenes_arg}" -gt 0 ]]; then
    args+=(--num_scenes "${num_scenes_arg}")
  fi
  if [[ -n "${CKPT_PATH}" ]]; then
    args+=(--ckpt_path "${CKPT_PATH}")
  else
    args+=(--hf_model "${HF_MODEL}")
  fi
  if [[ "${model_name}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi

  set_backend_pythonpath "${backend}"
  rm -rf "${out_dir}"
  echo "   [scannet/${model_name}] backend=${backend}"
  python "analysis/eval/eval_scannet.py" "${args[@]}"
}

run_7andn() {
  local model_name="$1"
  local backend="$2"
  local out_dir="${OUTPUT_DIR}/7andn/${model_name}"
  local max_scenes=0
  local max_samples=0
  local size_arg="${SIZE}"
  local kf_arg="${KF}"

  args=(
    --device "${DEVICE}"
    --output_dir "${out_dir}"
    --size "${size_arg}"
    --kf "${kf_arg}"
    --root_7scenes "${ROOT_7SCENES}"
    --root_nrgbd "${ROOT_NRGBD}"
    --max_scenes_per_dataset "${max_scenes}"
    --max_samples "${max_samples}"
  )
  if [[ -n "${CKPT_PATH}" ]]; then
    args+=(--ckpt_path "${CKPT_PATH}")
  else
    args+=(--hf_model "${HF_MODEL}")
  fi
  if [[ "${model_name}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi

  set_backend_pythonpath "${backend}"
  rm -rf "${out_dir}"
  echo "   [7andn/${model_name}] backend=${backend}"
  python "analysis/eval/eval_7andN.py" "${args[@]}"
}

summarize_results() {
  local summary_path="${OUTPUT_DIR}/summary_comparison.md"
  OUTPUT_DIR_ENV="${OUTPUT_DIR}" SCANNET_MODELS_ENV="${SCANNET_MODELS}" MODELS_7ANDN_ENV="${MODELS_7ANDN}" python - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR_ENV"])
scannet_models = [s.strip() for s in os.environ.get("SCANNET_MODELS_ENV", "").split(",") if s.strip()]
models_7andn = [s.strip() for s in os.environ.get("MODELS_7ANDN_ENV", "").split(",") if s.strip()]

def parse_model_name(spec: str) -> str:
    m = re.match(r"^\s*([^()]+)\(([^()]+)\)\s*$", spec)
    if not m:
        return spec
    return m.group(1).strip()

def latest_input_frame_dir(base: Path):
    candidates = [p for p in base.glob("input_frame_*") if p.is_dir()]
    if not candidates:
        return None
    def key_fn(p: Path):
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return -1
    return sorted(candidates, key=key_fn)[-1]

def read_scannet(mode: str):
    p = latest_input_frame_dir(root / "scannet" / mode)
    if p is None:
        return {}
    avg = p / "average_metrics.json"
    if not avg.exists():
        return {}
    return json.loads(avg.read_text(encoding="utf-8"))

def parse_7n_logs(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    mean_line = None
    for line in text.splitlines():
        if line.strip().startswith("mean"):
            mean_line = line
            break
    if mean_line is None:
        return {}
    pairs = re.findall(r"([a-zA-Z0-9_]+):\s*([0-9.+-eE]+)", mean_line)
    out = {}
    for k, v in pairs:
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

rows = []
scannet_metrics = ["chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms"]
for spec in scannet_models:
    mode = parse_model_name(spec)
    d = read_scannet(mode)
    rows.append(["scannet", mode] + [d.get(k, float("nan")) for k in scannet_metrics])

for spec in models_7andn:
    mode = parse_model_name(spec)
    for ds in ["7scenes", "NRGBD"]:
        log_path = root / "7andn" / mode
        kf_dirs = sorted([p for p in log_path.glob("*") if p.is_dir()])
        d = {}
        if kf_dirs:
            d = parse_7n_logs(kf_dirs[-1] / ds / "logs_all.txt")
        rows.append([ds.lower(), mode, d.get("acc", float("nan")), d.get("comp", float("nan")), d.get("nc", float("nan")), d.get("acc_med", float("nan")), d.get("comp_med", float("nan")), d.get("nc_med", float("nan"))])

summary_lines = []
summary_lines.append("| dataset | mode | m1 | m2 | m3 | m4 | m5 | m6 |")
summary_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows[:2]:
    summary_lines.append(f"| {r[0]} | {r[1]} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:.6f} | {r[6]:.6f} | {r[7]:.6f} |")
for r in rows[2:]:
    summary_lines.append(f"| {r[0]} | {r[1]} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:.6f} | {r[6]:.6f} | {r[7]:.6f} |")

legend = [
    "",
    "ScanNet: m1=chamfer_distance, m2=ate, m3=are, m4=rpe_rot, m5=rpe_trans, m6=inference_time_ms",
    "7Scenes/NRGBD: m1=acc, m2=comp, m3=nc, m4=acc_med, m5=comp_med, m6=nc_med",
]
summary = "\n".join(summary_lines + legend) + "\n"

out = root / "summary_comparison.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(summary, encoding="utf-8")
print(summary)
print(f"[saved] {out}")
PY
  echo "   Summary table:    ${summary_path}"
}

echo "== FastVGGT-aligned evaluation =="
echo "   output: ${OUTPUT_DIR}"
echo "   scannet_models: ${SCANNET_MODELS}"
echo "   models_7andn: ${MODELS_7ANDN}"
echo "   run_scannet: ${RUN_SCANNET}"
echo "   run_7andn: ${RUN_7ANDN}"
echo "   run_summary: ${RUN_SUMMARY}"

if [[ "${RUN_SCANNET}" == "1" || "${RUN_SCANNET}" == "true" ]]; then
  IFS=',' read -r -a SCANNET_MODEL_ARR <<< "${SCANNET_MODELS}"
  for spec in "${SCANNET_MODEL_ARR[@]}"; do
    parse_model_spec "${spec}"
    run_scannet "${MODEL_NAME_PARSED}" "${MODEL_BACKEND_PARSED}"
  done
else
  echo "   [skip] scannet evaluation disabled"
fi

if [[ "${RUN_7ANDN}" == "1" || "${RUN_7ANDN}" == "true" ]]; then
  IFS=',' read -r -a MODEL_7ANDN_ARR <<< "${MODELS_7ANDN}"
  for spec in "${MODEL_7ANDN_ARR[@]}"; do
    parse_model_spec "${spec}"
    run_7andn "${MODEL_NAME_PARSED}" "${MODEL_BACKEND_PARSED}"
  done
else
  echo "   [skip] 7scenes + NRGBD evaluation disabled"
fi

if [[ "${RUN_SUMMARY}" == "1" || "${RUN_SUMMARY}" == "true" ]]; then
  summarize_results
else
  echo "   [skip] summary generation disabled"
fi

echo "== Done =="
echo "   ScanNet outputs:  ${OUTPUT_DIR}/scannet/<model>"
echo "   7N outputs:       ${OUTPUT_DIR}/7andn/<model>"


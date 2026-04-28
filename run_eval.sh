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
# Set to 1 to pass --skip_depth_eval (no depth_abs_rel / depth_delta1 vs ScanNet GT).
SKIP_DEPTH_EVAL="${SKIP_DEPTH_EVAL:-0}"
BASELINE_BACKEND="${BASELINE_BACKEND:-vggt}"
FAST_BACKEND="${FAST_BACKEND:-fastvggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"
SCANNET_MODELS="${SCANNET_MODELS:-baseline(vggt),fastvggt(fastvggt)}"
MODELS_7ANDN="${MODELS_7ANDN:-fastvggt(fastvggt)}"
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
  local out_dir="${OUTPUT_DIR}/scans_test50/${model_name}"
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
  if [[ "${SKIP_DEPTH_EVAL}" == "1" || "${SKIP_DEPTH_EVAL}" == "true" ]]; then
    args+=(--skip_depth_eval)
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
    # Must match run_scannet() output: ${OUTPUT_DIR}/scans_test50/<model>
    p = latest_input_frame_dir(root / "scans_test50" / mode)
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

scannet_metric_keys = [
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "depth_abs_rel",
    "depth_delta1",
]
scannet_rows = []
for spec in scannet_models:
    mode = parse_model_name(spec)
    d = read_scannet(mode)
    scannet_rows.append((mode, d))

seven_metric_keys = ["acc", "comp", "nc", "acc_med", "comp_med", "nc_med"]
seven_rows = []
nrgbd_rows = []
for spec in models_7andn:
    mode = parse_model_name(spec)
    log_path = root / "7andn" / mode
    kf_dirs = sorted([p for p in log_path.glob("*") if p.is_dir()])
    d7 = {}
    dn = {}
    if kf_dirs:
        d7 = parse_7n_logs(kf_dirs[-1] / "7scenes" / "logs_all.txt")
        dn = parse_7n_logs(kf_dirs[-1] / "NRGBD" / "logs_all.txt")
    seven_rows.append((mode, d7))
    nrgbd_rows.append((mode, dn))


def fmt_cell(v: float) -> str:
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{float(v):.6f}"


def md_metric_table(title: str, mode_col: str, metric_keys: list, rows: list) -> list:
    """Return markdown lines for one block (title + table), or empty if rows is empty."""
    if not rows:
        return []
    lines = [f"### {title}", ""]
    header = "| " + " | ".join([mode_col] + metric_keys) + " |"
    sep = "| " + " | ".join([":---"] + ["---:"] * len(metric_keys)) + " |"
    lines.extend([header, sep])
    for mode, d in rows:
        vals = [fmt_cell(d.get(k, float("nan"))) for k in metric_keys]
        lines.append("| " + " | ".join([mode] + vals) + " |")
    lines.append("")
    return lines


summary_lines = ["# Evaluation summary", ""]
summary_lines.extend(
    md_metric_table("ScanNet", "mode", scannet_metric_keys, scannet_rows)
)
summary_lines.extend(
    md_metric_table("7Scenes", "mode", seven_metric_keys, seven_rows)
)
summary_lines.extend(
    md_metric_table("NRGBD", "mode", seven_metric_keys, nrgbd_rows)
)
while summary_lines and summary_lines[-1] == "":
    summary_lines.pop()
summary = "\n".join(summary_lines) + "\n"

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
echo "   skip_depth_eval (ScanNet depth vs GT): ${SKIP_DEPTH_EVAL}"
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
echo "   ScanNet outputs:  ${OUTPUT_DIR}/scans_test50/<model>"
echo "   7N outputs:       ${OUTPUT_DIR}/7andn/<model>"


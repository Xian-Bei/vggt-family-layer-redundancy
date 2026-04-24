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
OUTPUT_DIR="${OUTPUT_DIR:-analysis_outputs/fastvggt_eval}"
CKPT_PATH="${CKPT_PATH:-}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda:0}"
INPUT_FRAME="${INPUT_FRAME:-1000}"
NUM_SCENES="${NUM_SCENES:-0}"
KF="${KF:-10}"
SIZE="${SIZE:-518}"
MERGING="${MERGING:-0}"
MERGE_RATIO="${MERGE_RATIO:-0.9}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
VALIDATE_ONE_SCENE="${VALIDATE_ONE_SCENE:-0}"
BASELINE_BACKEND="${BASELINE_BACKEND:-ours}"
FAST_BACKEND="${FAST_BACKEND:-fastvggt}"
FASTVGGT_VENDOR_PATH="${FASTVGGT_VENDOR_PATH:-${REPO_ROOT}/third_party/fastvggt}"

set_backend_pythonpath() {
  local backend="$1"
  if [[ "${backend}" == "fastvggt" ]]; then
    export PYTHONPATH="${FASTVGGT_VENDOR_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
  else
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  fi
}

run_scannet() {
  local mode="$1"
  local backend="$2"
  local out_dir="${OUTPUT_DIR}/scannet/${mode}"
  local num_scenes_arg="${NUM_SCENES}"
  local input_frame_arg="${INPUT_FRAME}"
  if [[ "${VALIDATE_ONE_SCENE}" == "1" || "${VALIDATE_ONE_SCENE}" == "true" ]]; then
    num_scenes_arg=1
    input_frame_arg=30
  fi

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
  if [[ "${mode}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi

  set_backend_pythonpath "${backend}"
  echo "   [scannet/${mode}] backend=${backend}"
  python "analysis/eval/eval_scannet.py" "${args[@]}"
}

run_7andn() {
  local mode="$1"
  local backend="$2"
  local out_dir="${OUTPUT_DIR}/7andn/${mode}"
  local max_scenes=0
  local max_samples=0
  local size_arg="${SIZE}"
  local kf_arg="${KF}"
  if [[ "${VALIDATE_ONE_SCENE}" == "1" || "${VALIDATE_ONE_SCENE}" == "true" ]]; then
    max_scenes=1
    max_samples=1
    size_arg=224
    kf_arg=100
  fi

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
  if [[ "${mode}" == "fastvggt" ]]; then
    args+=(--merging "${MERGING}" --merge_ratio "${MERGE_RATIO}")
  fi

  set_backend_pythonpath "${backend}"
  echo "   [7andn/${mode}] backend=${backend}"
  python "analysis/eval/eval_7andN.py" "${args[@]}"
}

summarize_results() {
  local summary_path="${OUTPUT_DIR}/summary_comparison.md"
  OUTPUT_DIR_ENV="${OUTPUT_DIR}" python - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR_ENV"])

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
for mode in ["baseline", "fastvggt"]:
    d = read_scannet(mode)
    rows.append(["scannet", mode] + [d.get(k, float("nan")) for k in scannet_metrics])

for mode in ["baseline", "fastvggt"]:
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
echo "   validate_one_scene: ${VALIDATE_ONE_SCENE}"
echo "   baseline_backend: ${BASELINE_BACKEND}"
echo "   fast_backend: ${FAST_BACKEND}"

run_scannet "baseline" "${BASELINE_BACKEND}"
run_7andn "baseline" "${BASELINE_BACKEND}"
run_scannet "fastvggt" "${FAST_BACKEND}"
run_7andn "fastvggt" "${FAST_BACKEND}"
summarize_results

echo "== Done =="
echo "   ScanNet baseline: ${OUTPUT_DIR}/scannet/baseline"
echo "   ScanNet fast:     ${OUTPUT_DIR}/scannet/fastvggt"
echo "   7N baseline:      ${OUTPUT_DIR}/7andn/baseline"
echo "   7N fast:          ${OUTPUT_DIR}/7andn/fastvggt"


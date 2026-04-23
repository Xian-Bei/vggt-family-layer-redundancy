#!/usr/bin/env bash
# =============================================================================
# ScanNet v2 (or compatible) redundancy pipeline — stages 0–4
# =============================================================================
# Stage 0: Weight-only analysis (no images / no activation cache). Writes
#          WEIGHT_ANALYSIS_DIR/weight_report.json + bar charts.
# Stage 1: For every scene under DATA_ROOT, run VGGT inference and save
#          activation caches; GT evaluation and visualization are done directly
#          from in-memory predictions (no per-scene prediction-file round-trip).
#          Stage1 GT metrics now include:
#            - pose / geometry: ate, are, rpe_rot, rpe_trans, chamfer_distance
#            - depth: depth_abs_rel, depth_delta1
#          ScanNet depth GT is aligned to the model input via the same RGB crop-mode
#          preprocessing used by VGGT, clipped to the FastVGGT-valid indoor range
#          (1e-3m, 10m),
#          and aggregated across scenes by direct mean (FastVGGT-style).
# Stage 2: activation similarity from caches only:
#            - default/mainline: intra-layer sim(frame_i, global_i), i.e. how much each global block changes its input;
#            - legacy when PAIR_MAX_SPAN>1: inter-layer global-output similarity (mixes frame/global effects).
# Stage 3: skip-range GT loss experiments only:
#            (i,j) skips global blocks i..j (span <= SKIP_RANGE_MAX_SPAN), score_k = -(pruned_metric_k - stage1_metric_k).
#            The GT metric set includes both pose/geometry and depth metrics.
# Stage 4: Reserved for future baseline implementations (currently placeholder).
#
# Usage (from repo root):
#   chmod +x run_scannet_redundancy.sh
#   ./run_scannet_redundancy.sh
#
# Or override paths / limits:
#   DATA_ROOT=data/scannetv2 HF_MODEL=facebook/VGGT-1B ./run_scannet_redundancy.sh
#
# Expected data layouts (discover_scannet_scenes):
#   - DATA_ROOT/scans/sceneXXXX_XX/color/*.jpg   (official-style)
#   - DATA_ROOT/sceneXXXX_XX/color/*.jpg         (scans content directly under root)
#   - Fallback: scene/images/* or scene/* images
# =============================================================================

# Set 0 to skip 
STAGE0="${STAGE0:-0}"
STAGE1="${STAGE1:-1}"
STAGE2="${STAGE2:-0}"
STAGE3="${STAGE3:-1}"
STAGE4="${STAGE4:-0}"

set -euo pipefail

# Repo root = directory containing this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Python must resolve the local package `vggt` and `analysis`
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# User-tunable variables (override from environment before calling this script)
# -----------------------------------------------------------------------------

# [Shared data/model]
# DATA_ROOT: ScanNet root (supports DATA_ROOT/scans/... and DATA_ROOT/scene...).
# HF_MODEL: HuggingFace VGGT model id.
# DEVICE/DTYPE: default inference device and autocast dtype.
DATA_ROOT="${DATA_ROOT:-data/scans_dev50/}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

# [Shared eval knobs]
# SKIP_GT_EVAL=1 skips GT metrics (Stage2 skip-range GT drop will fail in this mode).
# GT_PLY_DIR optionally overrides where *_vh_clean_2.ply are searched.
# CHAMFER_MAX_DIST / DEPTH_CONF_THRESH are FastVGGT-aligned reconstruction settings.
# Depth eval itself additionally uses ScanNet GT depth aligned with RGB crop-mode preprocessing,
# valid GT mask derived from depth values, and the FastVGGT-style 10m depth cap.
SKIP_GT_EVAL="${SKIP_GT_EVAL:-0}"
GT_PLY_DIR="${GT_PLY_DIR:-}"
CHAMFER_MAX_DIST="${CHAMFER_MAX_DIST:-0.5}"
DEPTH_CONF_THRESH="${DEPTH_CONF_THRESH:-1.0}"

# [Stage 0: weight-only analysis]
# WEIGHT_ANALYSIS_DIR: output root for checkpoint-level weight stats and plots.
WEIGHT_ANALYSIS_DIR="${WEIGHT_ANALYSIS_DIR:-analysis_outputs/scannet_weight_analysis}"

# [Stage 1: cache + baseline evaluation]
# CACHE_DIR: stage1 outputs (manifest, activation cache, eval_scannet.json, optional visualization).
# MAX_SCENES/MAX_FRAMES control scene/frame sampling.
# MAX_TOKENS_PER_LAYER is kept only for compatibility; Stage1 now saves only
# full aggregated_concat_per_layer and no longer writes separate flattened caches.
# EXPORT_VIS_STAGE1 controls whether stage1 visualization assets are exported (default OFF).
CACHE_DIR="${CACHE_DIR:-analysis_outputs/scannet_cache}"
MAX_SCENES="${MAX_SCENES:-50}"
MAX_FRAMES="${MAX_FRAMES:-30}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-0}"
EXPORT_VIS_STAGE1="${EXPORT_VIS_STAGE1:-0}"
STAGE1_VIS_DIR="${STAGE1_VIS_DIR:-${CACHE_DIR}/visualization}"

# [Stage 2: similarity-only analysis]
# STAGE2_ANALYSIS_DIR: stage2 similarity outputs.
# MAX_CONCAT_TOKENS caps tokens before similarity metrics.
# PAIR_MAX_SPAN default 1 = intra-layer sim(frame_i, global_i). Values >1 enable the legacy
# inter-layer global-output similarity and cap its pair span (j-i+1).
# ENABLE_PWCCA enables PWCCA (slow); TIMING enables timing prints.
# SAVE_PER_SCENE_STAGE2 controls per-scene activation JSON emission under STAGE2_ANALYSIS_DIR/per_scene (default ON),
# which is also used by the Gradio panel for per-scene and mean-pct rendering.
# SAVE_HEATMAP_PNG_STAGE2 controls whether Stage2 writes PNG heatmaps; the Gradio panel can render from JSON,
# so this now defaults OFF to reduce clutter.
STAGE2_ANALYSIS_DIR="${STAGE2_ANALYSIS_DIR:-analysis_outputs/scannet_redundancy_similarity}"
MAX_CONCAT_TOKENS="${MAX_CONCAT_TOKENS:-0}"
PAIR_MAX_SPAN="${PAIR_MAX_SPAN:-1}"
ENABLE_PWCCA="${ENABLE_PWCCA:-0}"
TIMING="${TIMING:-1}"
HEATMAP_FILL="${HEATMAP_FILL:-nan}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-${DEVICE}}"
SAVE_PER_SCENE_STAGE2="${SAVE_PER_SCENE_STAGE2:-1}"
SAVE_HEATMAP_PNG_STAGE2="${SAVE_HEATMAP_PNG_STAGE2:-0}"

# [Stage 3: skip-range GT-loss experiments]
# STAGE3_ANALYSIS_DIR: stage3 skip outputs.
# SKIP_RANGE_MAX_SPAN caps skip-range span for GT-drop analysis.
# SKIPPED_GLOBAL_MODES controls how skipped global layers are approximated during Stage3 replay.
# Use a comma-separated list such as "special_only,identity" to run multiple modes for comparison,
# or a single value such as "identity" / "special_only" / "special_only_nomlp" /
# "patch_special" / "patch_special_nomlp"
# to keep current single-mode behavior.
# - identity: current behavior, fully skip the global block
# - special_only: still update camera/register tokens with asymmetric global attention
# - special_only_nomlp: same as special_only but skip post-attention norm2/MLP for special tokens
# - patch_special: keep special->all attention and patch->special attention (no patch->patch),
#   both branches keep normal post-attention norm2/MLP
# - patch_special_nomlp: same bipartite attention pattern, but skip post-attention norm2/MLP on both branches
# EXPORT_VIS_STAGE3 controls stage3 per-(i,j) visualization export (default OFF).
# STAGE3_NUM_PROCS controls scene-sharded parallel Stage3 workers (default 1 = single process).
# STAGE1_SANITY_CHECK verifies Stage3 uses the exact same frame selection as Stage1 and
# that a no-op forward reproduces Stage1 GT metrics within tight tolerances.
STAGE3_ANALYSIS_DIR="${STAGE3_ANALYSIS_DIR:-analysis_outputs/scannet_redundancy_skip}"
SKIP_RANGE_MAX_SPAN="${SKIP_RANGE_MAX_SPAN:-24}"
SKIPPED_GLOBAL_MODES="${SKIPPED_GLOBAL_MODES:-identity,special_only,special_only_nomlp,patch_special,patch_special_nomlp}"
EXPORT_VIS_STAGE3="${EXPORT_VIS_STAGE3:-0}"
STAGE3_VIS_DIR="${STAGE3_VIS_DIR:-${STAGE3_ANALYSIS_DIR}/visualization}"
STAGE3_NUM_PROCS="${STAGE3_NUM_PROCS:-3}"
STAGE1_SANITY_CHECK="${STAGE1_SANITY_CHECK:-1}"
STAGE1_SANITY_ATOL="${STAGE1_SANITY_ATOL:-1e-6}"
STAGE1_SANITY_RTOL="${STAGE1_SANITY_RTOL:-1e-6}"

# [Stage 4: reserved placeholder]
# Parameters kept for compatibility with older baseline flow; Stage4 is currently no-op.
BASELINE_DIR="${BASELINE_DIR:-analysis_outputs/scannet_baseline}"
SCHEMES_JSON="${SCHEMES_JSON:-analysis/redundancy/schemes.example.json}"
SCHEME_NAME="${SCHEME_NAME:-}"
BASELINE_RECOVERY_STEPS="${BASELINE_RECOVERY_STEPS:-0}"
BASELINE_RECOVERY_LR="${BASELINE_RECOVERY_LR:-1e-5}"
if ! [[ "${STAGE1_CACHE_DIR+set}" == "set" ]]; then
  STAGE1_CACHE_DIR="${CACHE_DIR}"
fi

# -----------------------------------------------------------------------------
# Stage 0 — checkpoint weight stats (no cache / no images)
# -----------------------------------------------------------------------------
if [[ "${STAGE0}" == "1" || "${STAGE0}" == "true" ]]; then
  echo "== Stage 0: weight analysis (checkpoint only) =="
  echo "    WEIGHT_ANALYSIS_DIR=${WEIGHT_ANALYSIS_DIR}"
  echo "    HF_MODEL=${HF_MODEL}"

  python -m analysis.redundancy.analyze_vggt_weights \
    --hf-model "${HF_MODEL}" \
    --output-dir "${WEIGHT_ANALYSIS_DIR}"

fi

# -----------------------------------------------------------------------------
# Stage 1 — inference + cache global activations for every scene
# -----------------------------------------------------------------------------
if [[ "${STAGE1}" == "1" || "${STAGE1}" == "true" ]]; then
  echo "== Stage 1: caching global-block activations + direct GT eval =="
  echo "    DATA_ROOT=${DATA_ROOT}"
  echo "    CACHE_DIR=${CACHE_DIR}"

  CACHE_EVAL_ARGS=()
  if [[ "${SKIP_GT_EVAL}" == "1" || "${SKIP_GT_EVAL}" == "true" ]]; then
    CACHE_EVAL_ARGS=(--skip-gt-eval)
  fi
  if [[ -n "${GT_PLY_DIR}" ]]; then
    CACHE_EVAL_ARGS+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  CACHE_EVAL_ARGS+=(--chamfer-max-dist "${CHAMFER_MAX_DIST}" --depth-conf-thresh "${DEPTH_CONF_THRESH}")
  if [[ "${EXPORT_VIS_STAGE1}" == "1" || "${EXPORT_VIS_STAGE1}" == "true" ]]; then
    CACHE_EVAL_ARGS+=(--export-visualization --visualization-dir "${STAGE1_VIS_DIR}")
  fi
  python -m analysis.redundancy.cache_scannet \
    --data-root "${DATA_ROOT}" \
    --output-dir "${CACHE_DIR}" \
    --hf-model "${HF_MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-scenes "${MAX_SCENES}" \
    --max-frames "${MAX_FRAMES}" \
    --max-tokens-per-layer "${MAX_TOKENS_PER_LAYER}" \
    "${CACHE_EVAL_ARGS[@]}"
fi

# python -m analysis.redundancy.view_scene_viser --vis-root analysis_outputs/scannet_cache/visualization --scene-id scene0000_00 --port 8080
# python -m analysis.redundancy.view_results_gradio


# -----------------------------------------------------------------------------
# Stage 2 — activation similarity only
# -----------------------------------------------------------------------------
if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
  echo "== Stage 2: analyzing caches (similarity only) =="
  echo "    CACHE_DIR=${CACHE_DIR}"
  echo "    STAGE2_ANALYSIS_DIR=${STAGE2_ANALYSIS_DIR}"
  echo "    ANALYSIS_DEVICE=${ANALYSIS_DEVICE}  ENABLE_PWCCA=${ENABLE_PWCCA}"

  TIMING_ARGS=()
  if [[ "${TIMING}" == "1" || "${TIMING}" == "true" ]]; then
    TIMING_ARGS=(--timing)
  fi

  PWCCA_ARGS=()
  if [[ "${ENABLE_PWCCA}" == "1" || "${ENABLE_PWCCA}" == "true" ]]; then
    PWCCA_ARGS=(--enable-pwcca)
  fi
  STAGE2_EXTRA=()
  if [[ "${SAVE_PER_SCENE_STAGE2}" == "1" || "${SAVE_PER_SCENE_STAGE2}" == "true" ]]; then
    STAGE2_EXTRA+=(--save-per-scene)
  fi
  if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
    STAGE2_EXTRA+=(--save-heatmap-png)
  fi
  if [[ ! "${STAGE1_SANITY_CHECK}" == "1" && ! "${STAGE1_SANITY_CHECK}" == "true" ]]; then
    STAGE2_EXTRA+=(--disable-stage1-sanity-check)
  else
    STAGE2_EXTRA+=(--stage1-sanity-atol "${STAGE1_SANITY_ATOL}" --stage1-sanity-rtol "${STAGE1_SANITY_RTOL}")
  fi

  python -m analysis.redundancy.analyze_scannet_similarity \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${STAGE2_ANALYSIS_DIR}" \
    --max-concat-tokens "${MAX_CONCAT_TOKENS}" \
    --pair-max-span "${PAIR_MAX_SPAN}" \
    --heatmap-fill "${HEATMAP_FILL}" \
    --device "${ANALYSIS_DEVICE}" \
    --disable-skip-range-loss \
    "${STAGE2_EXTRA[@]}" \
    "${PWCCA_ARGS[@]}" \
    "${TIMING_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Stage 3 — skip-range GT-loss experiments only
# -----------------------------------------------------------------------------
if [[ "${STAGE3}" == "1" || "${STAGE3}" == "true" ]]; then
  echo "== Stage 3: skip-range GT-loss only =="
  echo "    CACHE_DIR=${CACHE_DIR}"
  echo "    STAGE3_ANALYSIS_DIR=${STAGE3_ANALYSIS_DIR}"
  echo "    SKIPPED_GLOBAL_MODES=${SKIPPED_GLOBAL_MODES}"
  echo "    STAGE3_NUM_PROCS=${STAGE3_NUM_PROCS}"

  TIMING_ARGS=()
  if [[ "${TIMING}" == "1" || "${TIMING}" == "true" ]]; then
    TIMING_ARGS=(--timing)
  fi

  STAGE3_EXTRA=()
  if [[ -n "${GT_PLY_DIR}" ]]; then
    STAGE3_EXTRA+=(--gt-ply-dir "${GT_PLY_DIR}")
  fi
  if [[ "${SKIP_GT_EVAL}" == "1" || "${SKIP_GT_EVAL}" == "true" ]]; then
    STAGE3_EXTRA+=(--skip-gt-eval)
    echo "    [warn] Stage3 skip-range loss uses GT drops vs Stage1; SKIP_GT_EVAL=1 will fail."
  fi
  if [[ "${EXPORT_VIS_STAGE3}" == "1" || "${EXPORT_VIS_STAGE3}" == "true" ]]; then
    STAGE3_EXTRA+=(--export-visualization --visualization-dir "${STAGE3_VIS_DIR}")
    if [[ -n "${STAGE1_VIS_DIR}" ]]; then
      STAGE3_EXTRA+=(--stage1-visualization-dir "${STAGE1_VIS_DIR}")
    fi
  fi
  if [[ "${SAVE_PER_SCENE_STAGE2}" == "1" || "${SAVE_PER_SCENE_STAGE2}" == "true" ]]; then
    STAGE3_EXTRA+=(--save-per-scene)
  fi
  if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
    STAGE3_EXTRA+=(--save-heatmap-png)
  fi
  if [[ ! "${STAGE1_SANITY_CHECK}" == "1" && ! "${STAGE1_SANITY_CHECK}" == "true" ]]; then
    STAGE3_EXTRA+=(--disable-stage1-sanity-check)
  else
    STAGE3_EXTRA+=(--stage1-sanity-atol "${STAGE1_SANITY_ATOL}" --stage1-sanity-rtol "${STAGE1_SANITY_RTOL}")
  fi

  SKIPPED_GLOBAL_MODES_RAW="${SKIPPED_GLOBAL_MODES#[}"
  SKIPPED_GLOBAL_MODES_RAW="${SKIPPED_GLOBAL_MODES_RAW%]}"
  IFS=',' read -r -a SKIPPED_GLOBAL_MODES_ARR <<< "${SKIPPED_GLOBAL_MODES_RAW}"
  SKIPPED_GLOBAL_MODE_ARGS=()
  for mode in "${SKIPPED_GLOBAL_MODES_ARR[@]}"; do
    mode="${mode// /}"
    if [[ -n "${mode}" ]]; then
      SKIPPED_GLOBAL_MODE_ARGS+=("${mode}")
    fi
  done
  if [[ "${#SKIPPED_GLOBAL_MODE_ARGS[@]}" -eq 0 ]]; then
    echo "    [error] SKIPPED_GLOBAL_MODES must contain at least one mode." >&2
    exit 1
  fi

  STAGE3_COMMON_ARGS=(
    --cache-dir "${CACHE_DIR}"
    --data-root "${DATA_ROOT}"
    --hf-model "${HF_MODEL}"
    --max-frames "${MAX_FRAMES}"
    --heatmap-fill "${HEATMAP_FILL}"
    --skip-loss-device "${DEVICE}"
    --skip-loss-dtype "${DTYPE}"
    --chamfer-max-dist "${CHAMFER_MAX_DIST}"
    --depth-conf-thresh "${DEPTH_CONF_THRESH}"
    --skip-range-max-span "${SKIP_RANGE_MAX_SPAN}"
    --skipped-global-modes "${SKIPPED_GLOBAL_MODE_ARGS[@]}"
    "${STAGE3_EXTRA[@]}"
    "${TIMING_ARGS[@]}"
  )

  STAGE3_NUM_PROCS_INT=$((STAGE3_NUM_PROCS))
  if [[ "${STAGE3_NUM_PROCS_INT}" -le 1 ]]; then
    python -m analysis.redundancy.analyze_scannet_skip \
      --output-dir "${STAGE3_ANALYSIS_DIR}" \
      "${STAGE3_COMMON_ARGS[@]}"
  else
    SHARDS_DIR="${STAGE3_ANALYSIS_DIR}/_stage3_shards"
    SCENE_LIST_DIR="${SHARDS_DIR}/scene_lists"
    SHARD_OUTPUT_ROOT="${SHARDS_DIR}/outputs"
    mkdir -p "${SCENE_LIST_DIR}" "${SHARD_OUTPUT_ROOT}" "${STAGE3_ANALYSIS_DIR}/per_scene"

    python - <<PY
import json
from pathlib import Path

cache_dir = Path("${CACHE_DIR}")
out_dir = Path("${SCENE_LIST_DIR}")
n = max(1, int("${STAGE3_NUM_PROCS_INT}"))
manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
scene_ids = [str(item.get("scene_id", "")).strip() for item in manifest.get("scenes", [])]
scene_ids = [sid for sid in scene_ids if sid]
if not scene_ids:
    raise SystemExit("No scene_id found in manifest.json")
shards = [[] for _ in range(n)]
for idx, sid in enumerate(scene_ids):
    shards[idx % n].append(sid)
for idx, shard in enumerate(shards):
    p = out_dir / f"scene_ids_{idx:03d}.txt"
    p.write_text("\\n".join(shard) + ("\\n" if shard else ""), encoding="utf-8")
PY

    SHARD_PIDS=()
    shopt -s nullglob
    for scene_file in "${SCENE_LIST_DIR}"/scene_ids_*.txt; do
      if [[ ! -s "${scene_file}" ]]; then
        continue
      fi
      shard_name="$(basename "${scene_file}" .txt)"
      shard_out="${SHARD_OUTPUT_ROOT}/${shard_name}"
      mkdir -p "${shard_out}"
      python -m analysis.redundancy.analyze_scannet_skip \
        --output-dir "${shard_out}" \
        --scene-ids-file "${scene_file}" \
        "${STAGE3_COMMON_ARGS[@]}" &
      SHARD_PIDS+=("$!")
    done
    shopt -u nullglob

    if [[ "${#SHARD_PIDS[@]}" -eq 0 ]]; then
      echo "    [error] No non-empty Stage3 shard scene lists generated." >&2
      exit 1
    fi

    for pid in "${SHARD_PIDS[@]}"; do
      wait "${pid}"
    done

    python - <<PY
import json
import math
from pathlib import Path

target_dir = Path("${STAGE3_ANALYSIS_DIR}")
shard_root = Path("${SHARD_OUTPUT_ROOT}")
base_report_path = target_dir / "full_report_skip.json"
base_report = json.loads(base_report_path.read_text(encoding="utf-8")) if base_report_path.exists() else {}

mode_acc = {}
scene_ids_union = []
scene_ids_seen = set()

for shard_report_path in sorted(shard_root.glob("*/full_report_skip.json")):
    shard = json.loads(shard_report_path.read_text(encoding="utf-8"))
    shard_modes = shard.get("global_skip_range_loss_by_mode")
    if not isinstance(shard_modes, dict):
        single = shard.get("global_skip_range_loss")
        if isinstance(single, dict):
            mode = str(single.get("skipped_global_mode", "")).strip()
            if mode:
                shard_modes = {mode: single}
    if not isinstance(shard_modes, dict):
        continue
    for sid in shard.get("scene_ids", []):
        sid = str(sid)
        if sid and sid not in scene_ids_seen:
            scene_ids_seen.add(sid)
            scene_ids_union.append(sid)
    for mode, payload in shard_modes.items():
        if not isinstance(payload, dict):
            continue
        w = int(payload.get("num_scenes_used", 0))
        metric_map = payload.get("skip_range_loss_upper_by_metric")
        if w <= 0 or not isinstance(metric_map, dict):
            continue
        acc = mode_acc.setdefault(mode, {"meta": payload, "sum": {}, "cnt": {}})
        for mk, packed in metric_map.items():
            if not isinstance(packed, list):
                continue
            s = acc["sum"].setdefault(mk, [0.0] * len(packed))
            c = acc["cnt"].setdefault(mk, [0] * len(packed))
            if len(s) != len(packed):
                raise SystemExit(f"Packed length mismatch mode={mode} metric={mk}")
            for i, v in enumerate(packed):
                fv = float(v)
                if math.isnan(fv):
                    continue
                s[i] += fv * w
                c[i] += w
    shard_per_scene = shard_report_path.parent / "per_scene"
    if shard_per_scene.exists():
        for p in shard_per_scene.glob("*_skip.json"):
            (target_dir / "per_scene" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

merged_modes = {}
for mode, acc in mode_acc.items():
    meta = dict(acc["meta"])
    merged_metric = {}
    for mk, sums in acc["sum"].items():
        cnts = acc["cnt"][mk]
        merged_metric[mk] = [float("nan") if cnts[i] <= 0 else sums[i] / cnts[i] for i in range(len(sums))]
    meta["skip_range_loss_upper_by_metric"] = merged_metric
    if scene_ids_union:
        meta["num_scenes_used"] = len(scene_ids_union)
    merged_modes[mode] = meta

if merged_modes:
    mode_order = list(merged_modes.keys())
    base_report["skipped_global_modes"] = mode_order
    if len(mode_order) > 1:
        base_report["global_skip_range_loss_by_mode"] = merged_modes
        base_report.pop("global_skip_range_loss", None)
    else:
        only = mode_order[0]
        base_report["global_skip_range_loss"] = merged_modes[only]
        base_report.pop("global_skip_range_loss_by_mode", None)
if scene_ids_union:
    base_report["scene_ids"] = scene_ids_union
    base_report["num_scenes"] = len(scene_ids_union)
base_report_path.write_text(json.dumps(base_report, indent=2), encoding="utf-8")
PY
  fi
fi

# -----------------------------------------------------------------------------
# Stage 4 — placeholder (future baseline implementations)
# -----------------------------------------------------------------------------
if [[ "${STAGE4}" == "1" || "${STAGE4}" == "true" ]]; then
  echo "== Stage 4: placeholder (no-op for now) =="
  echo "    Stage 4 baseline runner is intentionally disabled."
fi

echo "== Done =="
echo "    Weight report:        ${WEIGHT_ANALYSIS_DIR}/weight_report.json"
echo "    Per-scene caches:     ${CACHE_DIR}/*.pt"
echo "    Manifest:             ${CACHE_DIR}/manifest.json"
if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
  echo "    Stage2 summary CSV:    ${STAGE2_ANALYSIS_DIR}/summary_metrics.csv"
  echo "    Stage2 scene CSV:      ${STAGE2_ANALYSIS_DIR}/per_scene_scores.csv"
  echo "    Stage2 report:         ${STAGE2_ANALYSIS_DIR}/full_report_similarity.json"
fi
if [[ "${STAGE3}" == "1" || "${STAGE3}" == "true" ]]; then
  echo "    Stage3 summary CSV:    ${STAGE3_ANALYSIS_DIR}/summary_metrics.csv"
  echo "    Stage3 scene CSV:      ${STAGE3_ANALYSIS_DIR}/per_scene_scores.csv"
  echo "    Stage3 report:         ${STAGE3_ANALYSIS_DIR}/full_report_skip.json"
fi
if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
  echo "    Stage2 figures:        ${STAGE2_ANALYSIS_DIR}/*.png"
  if [[ "${STAGE3}" == "1" || "${STAGE3}" == "true" ]]; then
    echo "    Stage3 figures:        ${STAGE3_ANALYSIS_DIR}/*.png"
  fi
fi
if [[ "${SAVE_PER_SCENE_STAGE2}" == "1" || "${SAVE_PER_SCENE_STAGE2}" == "true" ]]; then
  if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
    echo "    Stage2 per-scene:      ${STAGE2_ANALYSIS_DIR}/per_scene/*_similarity.json"
  fi
  if [[ "${STAGE3}" == "1" || "${STAGE3}" == "true" ]]; then
    echo "    Stage3 per-scene:      ${STAGE3_ANALYSIS_DIR}/per_scene/*_skip.json"
  fi
  if [[ "${SAVE_HEATMAP_PNG_STAGE2}" == "1" || "${SAVE_HEATMAP_PNG_STAGE2}" == "true" ]]; then
    if [[ "${STAGE2}" == "1" || "${STAGE2}" == "true" ]]; then
      echo "    Stage2 per-scene figs: ${STAGE2_ANALYSIS_DIR}/per_scene/*.png"
    fi
    if [[ "${STAGE3}" == "1" || "${STAGE3}" == "true" ]]; then
      echo "    Stage3 per-scene figs: ${STAGE3_ANALYSIS_DIR}/per_scene/*.png"
    fi
  fi
fi
if [[ "${STAGE4}" == "1" || "${STAGE4}" == "true" ]]; then
  echo "    Stage4:                placeholder only (no outputs yet)"
fi
if [[ "${STAGE1}" == "1" || "${STAGE1}" == "true" ]]; then
  echo "    GT eval (Stage1):      ${CACHE_DIR}/eval_scannet.json"
fi

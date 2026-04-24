"""
Step 2: unified redundancy analysis.

Part A (from cached activations):
- default/mainline similarity mode: within-layer ``sim(frame_i, global_i)`` to measure how much
  each global block changes its own input (stored in the same upper-triangle format, so only the
  diagonal is populated and off-diagonal entries are ``NaN``);
- legacy mode (when ``pair_max_span > 1``): inter-layer similarity between Stage1 cached global
  outputs. This mixes frame/global effects because adjacent global outputs are separated by a frame
  block, so it should not be interpreted as "global-only" redundancy.

Part B (skip-range loss): for each layer range [i..j] with span ``j-i+1 <= max_span``,
reuse Stage1 cached prefix activations and replay only the affected suffix.
Compare to Stage1 GT metrics loaded from Stage1 ``eval_scannet.json``:
for ``ate``, ``are``, ``rpe_rot``, ``rpe_trans``, ``chamfer_distance``, and ``depth_abs_rel``,
score_k(i,j) = stage1_metric_k - pruned_metric_k; for ``depth_delta1`` the delta is negated,
i.e. score_k = pruned_metric_k - stage1_metric_k. Values are reported per GT metric key.
Wider spans are stored as NaN in the packed upper triangle (heatmap masked like the lower triangle).

The GT metric set used for skip-range loss currently includes both pose/geometry and
dense-depth metrics:
- ``ate``, ``are``, ``rpe_rot``, ``rpe_trans``, ``chamfer_distance``
- ``depth_abs_rel``, ``depth_delta1``

Depth GT follows the same ScanNet protocol as Stage1: align GT depth to the RGB
``crop`` preprocessing used for model input, derive valid pixels from GT depth values,
and keep only depths in ``(1e-3m, 10m)`` before applying the StreamVGGT-style
scale-only depth alignment.

Both parts are exported as upper-triangle JSON + heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT

from .collect import predictions_to_cpu_float
from .metrics import (
    paired_layer_similarity_matrices,
    pairwise_similarity_matrices,
    upper_triangle_pack,
    upper_triangle_unpack_for_heatmap,
)
from .prediction_eval import (
    build_fastvggt_scene_eval_context,
    build_scannet_eval_row_from_payload_with_context,
    metrics_float_dict,
    read_eval_scannet_rows_by_scene,
)
from .scannet_eval import SceneData, discover_scannet_scenes, load_scene_images, select_scene_image_paths_fastvggt
from .prediction_save import build_scene_prediction_payload
from .visualization_export import export_scene_visualization_assets


def _read_manifest(cache_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run cache_scannet first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes", [])
    if not scenes:
        raise SystemExit("Manifest has empty scenes list.")
    return scenes


def _split_concat_branch_tuples(
    row: Dict[str, Any],
    *,
    patch_start_idx: int,
    branch: str,
    max_tokens_per_layer: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    replay_layers = row.get("aggregated_concat_per_layer")
    if not isinstance(replay_layers, list) or not replay_layers:
        raise SystemExit(
            "Stage2 similarity now requires Stage1 replay cache field "
            "'aggregated_concat_per_layer'. Please rerun Stage1 with the updated code."
        )

    frame_layers: List[torch.Tensor] = []
    global_layers: List[torch.Tensor] = []
    expected_n: Optional[int] = None
    for li, t in enumerate(replay_layers):
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            raise SystemExit(f"Invalid replay cache tensor at layer {li}: expected 4D tensor.")
        two_c = int(t.shape[-1])
        if two_c % 2 != 0:
            raise SystemExit(f"Invalid replay cache tensor at layer {li}: expected even channel dim, got {two_c}.")
        c = two_c // 2
        frame_t = t[..., :c]
        global_t = t[..., c:]
        if branch == "patch":
            frame_t = frame_t[:, :, patch_start_idx:, :]
            global_t = global_t[:, :, patch_start_idx:, :]
        elif branch == "special":
            frame_t = frame_t[:, :, :patch_start_idx, :]
            global_t = global_t[:, :, :patch_start_idx, :]
        else:
            raise ValueError(f"Unknown branch: {branch}")

        frame_flat = frame_t.reshape(-1, frame_t.shape[-1]).float()
        global_flat = global_t.reshape(-1, global_t.shape[-1]).float()
        if frame_flat.shape != global_flat.shape:
            raise SystemExit(
                f"Replay cache frame/global mismatch at layer {li}: "
                f"{tuple(frame_flat.shape)} vs {tuple(global_flat.shape)}."
            )
        if expected_n is None:
            expected_n = int(frame_flat.shape[0])
        elif int(frame_flat.shape[0]) != expected_n:
            raise SystemExit(
                f"Inconsistent flattened token count across replay layers for branch {branch}: "
                f"expected {expected_n}, got {tuple(frame_flat.shape)} at layer {li}."
            )
        frame_layers.append(frame_flat)
        global_layers.append(global_flat)

    assert expected_n is not None
    if max_tokens_per_layer > 0 and expected_n > max_tokens_per_layer:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) & 0x7FFFFFFF)
        idx = torch.randperm(expected_n, generator=g)[:max_tokens_per_layer]
        frame_layers = [t[idx] for t in frame_layers]
        global_layers = [t[idx] for t in global_layers]

    return tuple(frame_layers), tuple(global_layers)


def _similarity_mode_from_pair_span(pair_max_span: int) -> str:
    return "intra_layer_frame_vs_global" if pair_max_span <= 1 else "legacy_inter_layer_global_outputs"


def _finite_mean_upper(sum_t: torch.Tensor, cnt_t: torch.Tensor) -> List[float]:
    mean = torch.full_like(sum_t, float("nan"), dtype=torch.float64)
    valid = cnt_t > 0
    mean[valid] = sum_t[valid] / cnt_t[valid]
    return mean.tolist()


def _mean_diag(packed: List[float], n: int) -> float:
    vals: List[float] = []
    idx = 0
    for i in range(n):
        for j in range(i, n):
            v = float(packed[idx])
            idx += 1
            if i != j or math.isnan(v):
                continue
            vals.append(v)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _parse_heatmap_fill(s: str) -> float:
    sl = s.strip().lower()
    if sl in ("nan", "none"):
        return float("nan")
    return float(s)


def _heatmap_grid(
    mat: List[List[float]],
    out_path: Path,
    title: str,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    a = np.asarray(mat, dtype=np.float64)
    finite_vals = a[np.isfinite(a)]
    if np.any(np.isnan(a)):
        a = np.ma.masked_invalid(a)

    fig, ax = plt.subplots(figsize=(8, 7))
    imshow_kw: Dict[str, Any] = {"cmap": "viridis"}
    if vmin is not None:
        imshow_kw["vmin"] = vmin
    if vmax is not None:
        imshow_kw["vmax"] = vmax
    im = ax.imshow(a, **imshow_kw)
    ax.set_title(title)
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    data_min = float(vmin) if vmin is not None else (float(finite_vals.min()) if finite_vals.size else None)
    data_max = float(vmax) if vmax is not None else (float(finite_vals.max()) if finite_vals.size else None)
    if data_min is not None and data_max is not None:
        if abs(data_max - data_min) < 1e-12:
            cbar.set_ticks([data_min])
        else:
            cbar.set_ticks(np.linspace(data_min, data_max, num=5).tolist())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _heatmap_from_upper(
    packed: List[float],
    L: int,
    out_path: Path,
    title: str,
    fill_value: float,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    grid = upper_triangle_unpack_for_heatmap(packed, L, fill_value)
    _heatmap_grid(grid, out_path, title, vmin, vmax)


def _write_redundancy_heatmaps(
    patch_upper: Dict[str, List[float]],
    special_upper: Dict[str, List[float]],
    out_dir: Path,
    num_layers: int,
    fill_value: float,
    enable_pwcca: bool,
    *,
    similarity_mode: str,
    title_suffix: str,
    fname_prefix: str,
) -> None:
    n = num_layers
    vm_default: tuple[Optional[float], Optional[float]] = (0.0, 1.0)
    vm_mse: tuple[Optional[float], Optional[float]] = (None, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = title_suffix

    def _h(packed: List[float], fname: str, title: str, vm: tuple[Optional[float], Optional[float]]) -> None:
        _heatmap_from_upper(packed, n, out_dir / fname, title, fill_value, vm[0], vm[1])

    pfx = fname_prefix
    if similarity_mode == "intra_layer_frame_vs_global":
        patch_label = "Global patch intra-layer (frame_i vs global_i)"
        special_label = "Global special intra-layer (frame_i vs global_i)"
    else:
        patch_label = "Legacy global patch inter-layer"
        special_label = "Legacy global special inter-layer"

    _h(patch_upper["cka"], f"{pfx}heatmap_global_patch_cka.png", f"{patch_label} CKA{ts}", vm_default)
    _h(patch_upper["cosine"], f"{pfx}heatmap_global_patch_cosine.png", f"{patch_label} cosine{ts}", vm_default)
    if enable_pwcca:
        _h(patch_upper["pwcca"], f"{pfx}heatmap_global_patch_pwcca.png", f"{patch_label} PWCCA{ts}", vm_default)
    _h(patch_upper["neg_mse"], f"{pfx}heatmap_global_patch_neg_mse.png", f"{patch_label} -MSE{ts}", vm_mse)

    _h(special_upper["cka"], f"{pfx}heatmap_global_special_cka.png", f"{special_label} CKA{ts}", vm_default)
    _h(special_upper["cosine"], f"{pfx}heatmap_global_special_cosine.png", f"{special_label} cosine{ts}", vm_default)
    if enable_pwcca:
        _h(special_upper["pwcca"], f"{pfx}heatmap_global_special_pwcca.png", f"{special_label} PWCCA{ts}", vm_default)
    _h(special_upper["neg_mse"], f"{pfx}heatmap_global_special_neg_mse.png", f"{special_label} -MSE{ts}", vm_mse)


def _write_skip_loss_heatmap(
    packed: List[float],
    out_dir: Path,
    num_layers: int,
    fill_value: float,
    *,
    metric_key: str,
    title_suffix: str,
    fname_prefix: str,
    mode_suffix: str = "",
) -> None:
    _heatmap_from_upper(
        packed,
        num_layers,
        out_dir / f"{fname_prefix}heatmap_global_skip_range_loss_{metric_key}{mode_suffix}.png",
        f"Global skip-range loss ({metric_key}, higher means less drop){title_suffix}",
        fill_value,
        None,
        None,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fusion redundancy analysis from Stage1 cache + replay + GT.")
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/scannet_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/scannet_redundancy_fusion"))
    p.add_argument(
        "--similarity-output-dir",
        type=Path,
        default=None,
        help="Output directory for similarity reports. Default: analysis_outputs/scannet_redundancy_similarity.",
    )
    p.add_argument(
        "--skip-output-dir",
        type=Path,
        default=None,
        help="Output directory for skip GT reports. Default: analysis_outputs/scannet_redundancy_skip.",
    )
    p.add_argument(
        "--full-report-name",
        type=str,
        default="full_report_fusion.json",
        help="(Deprecated) compatibility-only; split reports now use dedicated names.",
    )
    p.add_argument(
        "--per-scene-report-suffix",
        type=str,
        default="_fusion.json",
        help="(Deprecated) compatibility-only; split per-scene reports use dedicated suffixes.",
    )
    p.add_argument(
        "--similarity-full-report-name",
        type=str,
        default="full_report_similarity.json",
        help="Filename for aggregated similarity report under similarity-output-dir.",
    )
    p.add_argument(
        "--skip-full-report-name",
        type=str,
        default="full_report_skip.json",
        help="Filename for aggregated skip report under skip-output-dir.",
    )
    p.add_argument(
        "--similarity-per-scene-report-suffix",
        type=str,
        default="_similarity.json",
        help="Suffix for per-scene similarity report files under similarity-output-dir/per_scene.",
    )
    p.add_argument(
        "--skip-per-scene-report-suffix",
        type=str,
        default="_skip.json",
        help="Suffix for per-scene skip report files under skip-output-dir/per_scene.",
    )
    p.add_argument(
        "--scene-ids-file",
        type=Path,
        default=None,
        help="Optional newline-separated scene_id list; when set, only these scenes are processed.",
    )
    p.add_argument(
        "--heatmap-fill",
        type=str,
        default="nan",
        help="Value for strict lower triangle (j<i) in heatmaps; use 'nan' to mask.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", help="Device for pairwise activations metrics.")
    p.add_argument("--timing", action="store_true", help="Print timing details.")
    p.add_argument(
        "--fusion-max-span",
        type=int,
        default=24,
        help=(
            "Fusion span cap for (i,j), where (i,j) means skip global blocks i..j and compare new j-layer "
            "global output to Stage1 cached j-layer global output. Larger spans are stored as NaN."
        ),
    )
    p.add_argument(
        "--fusion-modes",
        nargs="+",
        default=("identity", "special_only", "special_only_nomlp", "patch_special", "patch_special_nomlp"),
        choices=(
            "identity",
            "special_only",
            "special_only_nomlp",
            "patch_special",
            "patch_special_nomlp",
            "patch_special_bipartite",
        ),
        help="Skipped-global approximations to evaluate for each (i,j).",
    )
    p.add_argument(
        "--fusion-similarity-metrics",
        nargs="+",
        default=("cosine", "cka"),
        choices=("cosine", "cka", "pwcca", "neg_mse"),
        help="Similarity metrics between Stage1 and replayed j-layer global outputs.",
    )
    p.add_argument(
        "--fusion-compute-similarity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute Stage2 fusion similarity matrices (default: true).",
    )
    p.add_argument(
        "--fusion-num-procs",
        type=int,
        default=1,
        help="External sharding hint for wrappers/scripts; reported in full report metadata.",
    )
    p.add_argument(
        "--fusion-save-per-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-scene fusion JSON files (default: true).",
    )
    p.add_argument(
        "--fusion-write-aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write aggregated full reports and CSV summaries (default: true).",
    )
    p.add_argument(
        "--fusion-aggregate-from-per-scene",
        action="store_true",
        help="Skip model replay and aggregate outputs directly from existing per-scene JSON files.",
    )

    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--skip-loss-device", type=str, default="cuda")
    p.add_argument(
        "--skip-loss-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--skip-gt-eval", action="store_true")
    p.add_argument("--gt-ply-dir", type=Path, default=None)
    p.add_argument("--depth-conf-thresh", type=float, default=1.0)
    p.add_argument("--chamfer-max-dist", type=float, default=0.5)
    p.add_argument(
        "--disable-stage1-sanity-check",
        action="store_true",
        help="Disable Stage2 sanity check that verifies Stage1 and Stage2 use identical frame selection per scene.",
    )
    p.add_argument(
        "--stage1-sanity-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for Stage1 no-op sanity check metric comparison.",
    )
    p.add_argument(
        "--stage1-sanity-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for Stage1 no-op sanity check metric comparison.",
    )
    p.add_argument(
        "--export-visualization",
        action="store_true",
        help="Export/reuse per-scene visualization assets for Stage2 under similarity-output-dir/visualization.",
    )
    p.add_argument(
        "--save-heatmap-png",
        action="store_true",
        help="Write fusion heatmap PNG files alongside JSON outputs (disabled by default).",
    )
    p.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Stage2 visualization root. Default: <similarity-output-dir>/visualization",
    )
    p.add_argument(
        "--stage1-visualization-dir",
        type=Path,
        default=None,
        help="Reserved for compatibility; Stage2 now exports one visualization per skip pair.",
    )
    return p


def _metric_names(enable_pwcca: bool) -> Tuple[str, ...]:
    names: List[str] = ["cosine", "cka"]
    if enable_pwcca:
        names.append("pwcca")
    names.append("neg_mse")
    return tuple(names)


def _pack_all(sim: Dict[str, torch.Tensor], metric_names: Sequence[str]) -> Dict[str, List[float]]:
    return {k: upper_triangle_pack(sim[k]) for k in metric_names}


def _apply_span_mask_upper(packed: List[float], num_layers: int, max_span: int) -> List[float]:
    if max_span <= 0:
        return packed
    out: List[float] = []
    idx = 0
    for i in range(num_layers):
        for j in range(i, num_layers):
            v = packed[idx]
            out.append(v if (j - i + 1) <= max_span else float("nan"))
            idx += 1
    return out


def _resolve_analysis_device(device_arg: str) -> torch.device:
    want = device_arg.strip().lower()
    if want in ("cpu",):
        return torch.device("cpu")
    if want.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        print(f"[analyze_scannet_cache] requested {device_arg!r} but CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def _safe_scene_id(scene_id: str) -> str:
    return scene_id.replace("/", "_")


def _export_stage2_skip_pair_visualization(
    *,
    stage2_vis_root: Path,
    scene_eff: SceneData,
    pair_name: str,
    images: torch.Tensor,
    prediction_payload: Dict[str, Any],
    depth_conf_thresh: float,
) -> None:
    """
    Export one visualization bundle for one Stage2 skip pair (i,j):
    <stage2_vis_root>/<scene_id>/<pair_name>/{pointcloud_rgb.ply,trajectory_line.ply,cameras.json}
    """
    scene_root = stage2_vis_root / _safe_scene_id(scene_eff.scene_id)
    export_scene_visualization_assets(
        output_root=scene_root,
        scene_id=pair_name,
        image_paths=scene_eff.image_paths,
        images_tensor=images,
        prediction_payload=prediction_payload,
        depth_conf_thresh=depth_conf_thresh,
    )


GT_KEYS_FOR_SKIP = (
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "chamfer_distance",
    "depth_abs_rel",
    "depth_delta1",
)

# Skip-range per-key scores come from ``_aligned_change_per_key``: ``stage1 - pruned`` for
# ``ate``/``are``/``rpe_rot``/``rpe_trans``/``chamfer_distance``/``depth_abs_rel``, and
# ``pruned - stage1`` (negated form) for ``depth_delta1``.


def _nanmean_finite(xs: List[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _canonical_skipped_global_mode(mode: str) -> str:
    m = str(mode).strip()
    if m == "patch_special_bipartite":
        return "patch_special"
    return m


def _normalize_skipped_global_modes(modes: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for mode in modes:
        m = _canonical_skipped_global_mode(mode)
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    if not out:
        raise SystemExit("Expected at least one skipped global mode.")
    return out


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _finite_mean_packed(rows: Sequence[List[float]]) -> List[float]:
    if not rows:
        return []
    n = len(rows[0])
    sums = [0.0] * n
    cnts = [0] * n
    for row in rows:
        if len(row) != n:
            raise SystemExit("Packed length mismatch while aggregating per-scene rows.")
        for i, v in enumerate(row):
            fv = float(v)
            if math.isnan(fv):
                continue
            sums[i] += fv
            cnts[i] += 1
    out: List[float] = []
    for i in range(n):
        out.append(float("nan") if cnts[i] <= 0 else float(sums[i] / cnts[i]))
    return out


def _aggregate_from_per_scene_split_outputs(
    similarity_out_dir: Path,
    skip_out_dir: Path,
    *,
    similarity_report_name: str,
    skip_report_name: str,
    fusion_modes: Sequence[str],
    sim_metrics: Sequence[str],
    save_heatmap_png: bool,
    fill_value: float,
    compute_pwcca: bool,
    fusion_num_procs: int,
    analysis_device: torch.device,
    skip_gt_eval: bool,
) -> None:
    sim_per_scene_dir = similarity_out_dir / "per_scene"
    skip_per_scene_dir = skip_out_dir / "per_scene"
    if not sim_per_scene_dir.exists() or not skip_per_scene_dir.exists():
        raise SystemExit(
            "Per-scene directories are missing. Expected both "
            f"{sim_per_scene_dir} and {skip_per_scene_dir}."
        )
    sim_files = sorted(sim_per_scene_dir.glob("*.json"))
    skip_files = sorted(skip_per_scene_dir.glob("*.json"))
    if not sim_files and not skip_files:
        raise SystemExit("No per-scene JSON files found for aggregation.")

    sim_rows_by_scene: Dict[str, Dict[str, Any]] = {}
    skip_rows_by_scene: Dict[str, Dict[str, Any]] = {}
    for p in sim_files:
        row = _read_json_if_exists(p)
        sid = str(row.get("scene_id", "")).strip()
        if sid:
            sim_rows_by_scene[sid] = row
    for p in skip_files:
        row = _read_json_if_exists(p)
        sid = str(row.get("scene_id", "")).strip()
        if sid:
            skip_rows_by_scene[sid] = row

    scene_ids = sorted(set(sim_rows_by_scene.keys()) | set(skip_rows_by_scene.keys()))
    if not scene_ids:
        raise SystemExit("No valid scene_id found in per-scene JSON files.")

    first_row = None
    for sid in scene_ids:
        first_row = sim_rows_by_scene.get(sid) or skip_rows_by_scene.get(sid)
        if first_row is not None:
            break
    if first_row is None:
        raise SystemExit("Failed to infer metadata from per-scene JSON files.")
    num_layers = int(first_row.get("num_layers", 0))
    if num_layers <= 0:
        raise SystemExit("Invalid or missing num_layers in per-scene files.")

    mode_list = list(fusion_modes) if fusion_modes else []
    if not mode_list:
        mode_seen: set[str] = set()
        for sid in scene_ids:
            for row in (sim_rows_by_scene.get(sid), skip_rows_by_scene.get(sid)):
                if not isinstance(row, dict):
                    continue
                by_mode = row.get("fusion_by_mode", {})
                if not isinstance(by_mode, dict):
                    continue
                for m in by_mode.keys():
                    ms = str(m)
                    if ms and ms not in mode_seen:
                        mode_seen.add(ms)
                        mode_list.append(ms)
    if not mode_list:
        raise SystemExit("Failed to infer fusion_modes from per-scene files.")

    sim_fusion_by_mode: Dict[str, Dict[str, Any]] = {}
    skip_fusion_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in mode_list:
        mode_sim_patch: Dict[str, List[float]] = {}
        mode_sim_special: Dict[str, List[float]] = {}
        mode_skip_gt: Dict[str, List[float]] = {}

        for mk in sim_metrics:
            patch_rows: List[List[float]] = []
            special_rows: List[List[float]] = []
            for sid in scene_ids:
                row = sim_rows_by_scene.get(sid, {})
                by_mode = row.get("fusion_by_mode", {}) if isinstance(row, dict) else {}
                mode_map = by_mode.get(mode, {}) if isinstance(by_mode, dict) else {}
                if not isinstance(mode_map, dict):
                    continue
                patch = mode_map.get("global_patch_similarity_upper_by_metric", {})
                special = mode_map.get("global_special_similarity_upper_by_metric", {})
                if isinstance(patch, dict) and isinstance(patch.get(mk), list):
                    patch_rows.append([float(v) for v in patch[mk]])
                if isinstance(special, dict) and isinstance(special.get(mk), list):
                    special_rows.append([float(v) for v in special[mk]])
            if patch_rows:
                mode_sim_patch[mk] = _finite_mean_packed(patch_rows)
            if special_rows:
                mode_sim_special[mk] = _finite_mean_packed(special_rows)

        for gk in GT_KEYS_FOR_SKIP:
            gt_rows: List[List[float]] = []
            for sid in scene_ids:
                row = skip_rows_by_scene.get(sid, {})
                by_mode = row.get("fusion_by_mode", {}) if isinstance(row, dict) else {}
                mode_map = by_mode.get(mode, {}) if isinstance(by_mode, dict) else {}
                if not isinstance(mode_map, dict):
                    continue
                gt_map = mode_map.get("gt_delta_upper_by_metric", {})
                if isinstance(gt_map, dict) and isinstance(gt_map.get(gk), list):
                    gt_rows.append([float(v) for v in gt_map[gk]])
            if gt_rows:
                mode_skip_gt[gk] = _finite_mean_packed(gt_rows)

        sim_fusion_by_mode[mode] = {
            "num_scenes_used": len(scene_ids),
            "global_patch_similarity_upper_by_metric": mode_sim_patch,
            "global_special_similarity_upper_by_metric": mode_sim_special,
        }
        skip_fusion_by_mode[mode] = {
            "num_scenes_used": len(scene_ids),
            "gt_delta_upper_by_metric": mode_skip_gt,
        }

    fill_json = None if math.isnan(fill_value) else fill_value
    report_common: Dict[str, Any] = {
        "num_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "aggregation": "finite_mean_upper_packed_per_scene",
        "num_layers": num_layers,
        "fusion_storage": "upper_triangle_row_major_including_diagonal",
        "fusion_max_span": int(first_row.get("fusion_max_span", 0)),
        "fusion_modes": mode_list,
        "fusion_similarity_metrics": list(sim_metrics),
        "fusion_compute_similarity": bool(sim_metrics),
        "fusion_num_procs": fusion_num_procs,
        "analysis_device": str(analysis_device),
        "skip_gt_eval": skip_gt_eval,
        "fusion_save_per_scene": True,
        "save_heatmap_png": bool(save_heatmap_png),
        "heatmap_lower_triangle_fill": fill_json if fill_json is not None else "nan",
    }

    sim_report = dict(report_common)
    sim_report["report_kind"] = "similarity"
    sim_report["fusion_by_mode"] = sim_fusion_by_mode
    skip_report = dict(report_common)
    skip_report["report_kind"] = "skip"
    skip_report["fusion_by_mode"] = skip_fusion_by_mode

    similarity_out_dir.mkdir(parents=True, exist_ok=True)
    skip_out_dir.mkdir(parents=True, exist_ok=True)
    (similarity_out_dir / similarity_report_name).write_text(json.dumps(sim_report, indent=2), encoding="utf-8")
    (skip_out_dir / skip_report_name).write_text(json.dumps(skip_report, indent=2), encoding="utf-8")

    similarity_summary_rows: List[Dict[str, Any]] = []
    skip_summary_rows: List[Dict[str, Any]] = []
    for mode in mode_list:
        mode_sim = sim_fusion_by_mode[mode]
        mode_skip = skip_fusion_by_mode[mode]
        for mk in sim_metrics:
            if mk in mode_sim["global_patch_similarity_upper_by_metric"]:
                similarity_summary_rows.append(
                    {
                        "metric": f"fusion_global_patch_{mk}_{mode}_mean_finite",
                        "value": _nanmean_finite(mode_sim["global_patch_similarity_upper_by_metric"][mk]),
                    }
                )
            if mk in mode_sim["global_special_similarity_upper_by_metric"]:
                similarity_summary_rows.append(
                    {
                        "metric": f"fusion_global_special_{mk}_{mode}_mean_finite",
                        "value": _nanmean_finite(mode_sim["global_special_similarity_upper_by_metric"][mk]),
                    }
                )
        for gk in GT_KEYS_FOR_SKIP:
            if gk in mode_skip["gt_delta_upper_by_metric"]:
                skip_summary_rows.append(
                    {
                        "metric": f"fusion_gt_delta_{gk}_{mode}_mean_finite",
                        "value": _nanmean_finite(mode_skip["gt_delta_upper_by_metric"][gk]),
                    }
                )

    scene_scores_map: Dict[str, Dict[str, Any]] = {}
    score_keys_set: set[str] = set()
    for sid, row in sim_rows_by_scene.items():
        ss = row.get("scene_scores", {})
        if isinstance(ss, dict):
            scene_scores_map[sid] = dict(ss)
            score_keys_set.update(ss.keys())
    for sid, row in skip_rows_by_scene.items():
        if sid in scene_scores_map:
            continue
        ss = row.get("scene_scores", {})
        if isinstance(ss, dict):
            scene_scores_map[sid] = dict(ss)
            score_keys_set.update(ss.keys())

    score_keys = sorted(score_keys_set)
    per_scene_rows_final: List[Dict[str, Any]] = [
        {"scene_id": sid, **{k: scene_scores_map.get(sid, {}).get(k, "") for k in score_keys}} for sid in scene_ids
    ]
    for k in score_keys:
        vals = [float(r[k]) for r in per_scene_rows_final if r.get(k) not in ("", None) and str(r[k]).strip() != ""]
        if vals:
            mean_v = sum(vals) / len(vals)
            similarity_summary_rows.append({"metric": f"scene_avg_{k}", "value": mean_v})
            skip_summary_rows.append({"metric": f"scene_avg_{k}", "value": mean_v})

    with (similarity_out_dir / "summary_metrics_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for row in similarity_summary_rows:
            w.writerow(row)
    with (skip_out_dir / "summary_metrics_skip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for row in skip_summary_rows:
            w.writerow(row)

    with (similarity_out_dir / "per_scene_scores_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scene_id"] + score_keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_scene_rows_final)
    with (skip_out_dir / "per_scene_scores_skip.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scene_id"] + score_keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_scene_rows_final)

    if save_heatmap_png:
        for mode in mode_list:
            mode_sim = sim_fusion_by_mode[mode]
            mode_skip = skip_fusion_by_mode[mode]
            if sim_metrics:
                _write_redundancy_heatmaps(
                    mode_sim["global_patch_similarity_upper_by_metric"],
                    mode_sim["global_special_similarity_upper_by_metric"],
                    similarity_out_dir,
                    num_layers,
                    fill_value,
                    compute_pwcca,
                    similarity_mode="intra_layer_frame_vs_global",
                    title_suffix=f" (mean, {mode})",
                    fname_prefix=f"{mode}_",
                )
            for gk in GT_KEYS_FOR_SKIP:
                if gk not in mode_skip["gt_delta_upper_by_metric"]:
                    continue
                _write_skip_loss_heatmap(
                    mode_skip["gt_delta_upper_by_metric"][gk],
                    skip_out_dir,
                    num_layers,
                    fill_value,
                    metric_key=gk,
                    title_suffix=f" (mean, {mode})",
                    fname_prefix="",
                    mode_suffix=f"_{mode}",
                )

def _extract_skip_loss_by_mode(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_mode = report.get("global_skip_range_loss_by_mode")
    if isinstance(by_mode, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in by_mode.items():
            if isinstance(v, dict):
                out[_canonical_skipped_global_mode(str(k))] = v
        return out

    single = report.get("global_skip_range_loss")
    if isinstance(single, dict):
        mode = single.get("skipped_global_mode")
        if isinstance(mode, str) and mode:
            return {_canonical_skipped_global_mode(mode): single}
    return {}


def _load_existing_per_scene_skip_by_mode(
    per_scene_dir: Optional[Path],
    scene_ids: Sequence[str],
    *,
    per_scene_report_suffix: str,
) -> Dict[str, Dict[str, Any]]:
    if per_scene_dir is None:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for sid in scene_ids:
        path = per_scene_dir / f"{sid.replace('/', '_')}{per_scene_report_suffix}"
        row = _read_json_if_exists(path)
        if not row:
            continue

        by_mode = row.get("global_skip_range_loss_upper_by_metric_by_mode")
        if isinstance(by_mode, dict):
            clean = {_canonical_skipped_global_mode(str(k)): v for k, v in by_mode.items() if isinstance(v, dict)}
            if clean:
                out[sid] = clean
                continue

        single = row.get("global_skip_range_loss_upper_by_metric")
        single_mode = row.get("skipped_global_mode")
        if isinstance(single, dict) and isinstance(single_mode, str) and single_mode:
            out[sid] = {_canonical_skipped_global_mode(single_mode): single}
    return out


def _extract_scene_skip_by_mode(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_mode = row.get("global_skip_range_loss_upper_by_metric_by_mode")
    if isinstance(by_mode, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in by_mode.items():
            if isinstance(v, dict):
                out[_canonical_skipped_global_mode(str(k))] = v
        if out:
            return out
    single = row.get("global_skip_range_loss_upper_by_metric")
    single_mode = row.get("skipped_global_mode")
    if isinstance(single, dict) and isinstance(single_mode, str) and single_mode.strip():
        return {_canonical_skipped_global_mode(single_mode): single}
    return {}


def _aggregate_skip_modes_from_per_scene_files(
    per_scene_dir: Path,
    scene_ids: Sequence[str],
    *,
    per_scene_report_suffix: str,
) -> Dict[str, Dict[str, Any]]:
    mode_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    mode_scene_ids: Dict[str, set[str]] = {}
    for sid in scene_ids:
        path = per_scene_dir / f"{sid.replace('/', '_')}{per_scene_report_suffix}"
        row = _read_json_if_exists(path)
        if not row:
            continue
        by_mode = _extract_scene_skip_by_mode(row)
        for mode, metric_map in by_mode.items():
            if not isinstance(metric_map, dict):
                continue
            mode_scene_ids.setdefault(mode, set()).add(sid)
            mode_bucket = mode_acc.setdefault(mode, {"sum": {}, "cnt": {}})
            for mk, packed in metric_map.items():
                if not isinstance(packed, list):
                    continue
                sum_map = mode_bucket["sum"].setdefault(mk, [0.0] * len(packed))
                cnt_map = mode_bucket["cnt"].setdefault(mk, [0.0] * len(packed))
                if len(sum_map) != len(packed):
                    raise SystemExit(f"Packed length mismatch when aggregating per-scene skip rows: mode={mode}, metric={mk}")
                for idx, v in enumerate(packed):
                    fv = float(v)
                    if math.isnan(fv):
                        continue
                    sum_map[idx] += fv
                    cnt_map[idx] += 1.0

    out: Dict[str, Dict[str, Any]] = {}
    for mode, bucket in mode_acc.items():
        metric_out: Dict[str, List[float]] = {}
        for mk, sums in bucket["sum"].items():
            cnts = bucket["cnt"].get(mk, [])
            metric_out[mk] = [
                float("nan") if i >= len(cnts) or cnts[i] <= 0 else float(sums[i] / cnts[i]) for i in range(len(sums))
            ]
        out[mode] = {
            "metric_map": metric_out,
            "num_scenes_used": len(mode_scene_ids.get(mode, set())),
        }
    return out



def _aligned_change_per_key(stage1_gt: Dict[str, float], pruned_gt: Dict[str, float]) -> Dict[str, float]:
    """Per-key skip delta: ``stage1 - pruned`` except ``depth_delta1`` uses ``pruned - stage1``."""
    out: Dict[str, float] = {}
    for k in GT_KEYS_FOR_SKIP:
        if k not in stage1_gt or k not in pruned_gt:
            continue
        s1 = float(stage1_gt[k])
        pr = float(pruned_gt[k])
        if k == "depth_delta1":
            out[k] = pr - s1
        else:
            out[k] = s1 - pr
    return out


def _manifest_frame_ids(item: Dict[str, Any]) -> Optional[List[int]]:
    vals = item.get("frame_ids")
    if not isinstance(vals, list):
        return None
    out: List[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            return None
    return out


def _check_stage1_selection_match(scene_id: str, item: Dict[str, Any], scene_eff: SceneData) -> None:
    expected = _manifest_frame_ids(item)
    actual = [int(p.stem) for p in scene_eff.image_paths]
    if expected is None:
        raise SystemExit(
            f"Stage1 sanity check failed for {scene_id}: manifest.json lacks frame_ids. "
            "Please rerun Stage1 with the updated code, or disable the sanity check explicitly."
        )
    if expected != actual:
        raise SystemExit(
            f"Stage1 sanity check failed for {scene_id}: frame selection mismatch.\n"
            f"Stage1 frame_ids={expected}\n"
            f"Stage2 frame_ids={actual}"
        )
    expected_num_frames = item.get("num_frames")
    if expected_num_frames is not None and int(expected_num_frames) != len(actual):
        raise SystemExit(
            f"Stage1 sanity check failed for {scene_id}: num_frames mismatch "
            f"(manifest={expected_num_frames}, current={len(actual)})."
        )


def _check_stage1_noop_metrics_match(
    scene_id: str,
    stage1_gt: Dict[str, float],
    noop_gt: Dict[str, float],
    *,
    atol: float,
    rtol: float,
) -> None:
    mismatches: List[str] = []
    for k in GT_KEYS_FOR_SKIP:
        stage1_has = k in stage1_gt
        noop_has = k in noop_gt
        if not stage1_has and not noop_has:
            continue
        if not stage1_has or not noop_has:
            mismatches.append(f"{k}: missing (stage1 has {k in stage1_gt}, noop has {k in noop_gt})")
            continue
        a = float(stage1_gt[k])
        b = float(noop_gt[k])
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            mismatches.append(f"{k}: stage1={a:.10f}, noop={b:.10f}, delta={b - a:+.10f}")
    if mismatches:
        msg = "\n".join(mismatches)
        raise SystemExit(
            f"Stage1 sanity check failed for {scene_id}: no-op forward does not match Stage1 metrics "
            f"(atol={atol}, rtol={rtol}).\n{msg}"
        )


def _load_scene_replay_cache(cache_dir: Path, item: Dict[str, Any], num_layers: int) -> List[torch.Tensor]:
    path = cache_dir / str(item["file"])
    try:
        row = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        row = torch.load(path, map_location="cpu")
    replay_layers = row.get("aggregated_concat_per_layer")
    if not isinstance(replay_layers, list) or len(replay_layers) != num_layers:
        raise SystemExit(
            f"Skip replay requires Stage1 cache field 'aggregated_concat_per_layer' "
            f"with {num_layers} layers in {path}. Please rerun Stage1 with the updated code."
        )
    out: List[torch.Tensor] = []
    for idx, t in enumerate(replay_layers):
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            raise SystemExit(
                f"Invalid replay cache tensor at layer {idx} in {path}: expected 4D tensor, got {type(t).__name__}."
            )
        out.append(t.contiguous())
    return out


def _load_scene_cache_row(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    path = cache_dir / str(item["file"])
    try:
        row = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        row = torch.load(path, map_location="cpu")
    if not isinstance(row, dict):
        raise SystemExit(f"Invalid cache row in {path}: expected dict, got {type(row).__name__}.")
    return row


def _infer_image_hw_for_replay(
    cache_row: Dict[str, Any],
    replay_concat_layers: Sequence[torch.Tensor],
    *,
    patch_start_idx: int,
    patch_size: int,
) -> Tuple[int, int]:
    image_hw = cache_row.get("image_hw")
    if isinstance(image_hw, (list, tuple)) and len(image_hw) == 2:
        try:
            h = int(image_hw[0])
            w = int(image_hw[1])
            if h > 0 and w > 0:
                return h, w
        except (TypeError, ValueError):
            pass

    if not replay_concat_layers:
        raise SystemExit("Cannot infer image_hw for replay: replay cache is empty.")
    tokens_per_frame = int(replay_concat_layers[0].shape[2])
    patch_tokens = tokens_per_frame - int(patch_start_idx)
    if patch_tokens <= 0:
        raise SystemExit(
            f"Cannot infer image_hw for replay: invalid patch_tokens={patch_tokens}, "
            f"tokens_per_frame={tokens_per_frame}, patch_start_idx={patch_start_idx}."
        )
    side = int(round(math.sqrt(float(patch_tokens))))
    if side * side != patch_tokens:
        raise SystemExit(
            f"Cannot infer image_hw for replay: patch token grid is not square (patch_tokens={patch_tokens})."
        )
    return side * int(patch_size), side * int(patch_size)


def _infer_num_layers_from_manifest(cache_dir: Path, scenes_meta: Sequence[Dict[str, Any]]) -> int:
    for item in scenes_meta:
        scene_file = item.get("file")
        if not isinstance(scene_file, str) or not scene_file.strip():
            continue
        path = cache_dir / scene_file
        try:
            row = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            row = torch.load(path, map_location="cpu")
        replay_layers = row.get("aggregated_concat_per_layer")
        if isinstance(replay_layers, list) and replay_layers:
            return len(replay_layers)
    raise SystemExit(
        "Could not infer num_layers from cache manifest: missing valid aggregated_concat_per_layer in all scenes."
    )


def _build_aggregator_positions(model: VGGT, images: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    agg = model.aggregator
    if agg.rope is None:
        return None, None
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if images.dim() != 5:
        raise ValueError(f"Expected images with 4 or 5 dims, got shape {tuple(images.shape)}")
    bsz, seq_len, _, height, width = images.shape
    pos_frame = agg.position_getter(
        bsz * seq_len,
        height // agg.patch_size,
        width // agg.patch_size,
        device=images.device,
    )
    if agg.patch_start_idx > 0:
        pos_frame = pos_frame + 1
        pos_special = torch.zeros(
            bsz * seq_len,
            agg.patch_start_idx,
            2,
            device=images.device,
            dtype=pos_frame.dtype,
        )
        pos_frame = torch.cat([pos_special, pos_frame], dim=1)
    _, p, _ = pos_frame.shape
    pos_global = pos_frame.view(bsz, seq_len, p, 2).view(bsz, seq_len * p, 2)
    return pos_frame, pos_global


def _special_only_global_block_forward(
    block: torch.nn.Module,
    global_in: torch.Tensor,
    *,
    bsz: int,
    seq_len: int,
    tokens_per_frame: int,
    patch_start_idx: int,
    pos_global: Optional[torch.Tensor],
    apply_mlp: bool,
) -> torch.Tensor:
    if not (0 < patch_start_idx < tokens_per_frame):
        return global_in

    total_tokens = seq_len * tokens_per_frame
    if global_in.shape != (bsz, total_tokens, global_in.shape[-1]):
        raise ValueError(
            f"Expected global_in shape {(bsz, total_tokens, global_in.shape[-1])}, got {tuple(global_in.shape)}"
        )

    x = global_in.view(bsz, seq_len, tokens_per_frame, -1)
    c = x.shape[-1]
    num_special = patch_start_idx
    num_heads = block.attn.num_heads
    head_dim = block.attn.head_dim

    special = x[:, :, :num_special, :].contiguous().view(bsz, seq_len * num_special, c)
    all_tokens = x.view(bsz, total_tokens, c)
    patch = x[:, :, num_special:, :]

    norm_special = block.norm1(special)
    norm_all = block.norm1(all_tokens)

    qkv_weight = block.attn.qkv.weight
    qkv_bias = block.attn.qkv.bias
    q_bias = qkv_bias[:c] if qkv_bias is not None else None
    k_bias = qkv_bias[c : 2 * c] if qkv_bias is not None else None
    v_bias = qkv_bias[2 * c :] if qkv_bias is not None else None

    q = F.linear(norm_special, qkv_weight[:c], q_bias)
    k = F.linear(norm_all, qkv_weight[c : 2 * c], k_bias)
    v = F.linear(norm_all, qkv_weight[2 * c :], v_bias)

    q = q.view(bsz, seq_len * num_special, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(bsz, total_tokens, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(bsz, total_tokens, num_heads, head_dim).permute(0, 2, 1, 3)

    q = block.attn.q_norm(q)
    k = block.attn.k_norm(k)

    special_pos = None
    if pos_global is not None:
        pos_view = pos_global.view(bsz, seq_len, tokens_per_frame, 2)
        special_pos = pos_view[:, :, :num_special, :].contiguous().view(bsz, seq_len * num_special, 2)

    if block.attn.rope is not None:
        q = block.attn.rope(q, special_pos)
        k = block.attn.rope(k, pos_global)

    if block.attn.fused_attn:
        special_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=block.attn.attn_drop.p if block.training else 0.0,
        )
    else:
        q = q * block.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = block.attn.attn_drop(attn)
        special_out = attn @ v

    special_out = special_out.transpose(1, 2).reshape(bsz, seq_len * num_special, c)
    special_out = block.attn.proj(special_out)
    special_out = block.attn.proj_drop(special_out)

    special = special + block.ls1(special_out)
    if apply_mlp:
        special = special + block.ls2(block.mlp(block.norm2(special)))

    special = special.view(bsz, seq_len, num_special, c)
    return torch.cat([special, patch], dim=2).reshape(bsz, total_tokens, c)


def _split_global_token_groups(
    global_in: torch.Tensor,
    *,
    bsz: int,
    seq_len: int,
    tokens_per_frame: int,
    patch_start_idx: int,
    pos_global: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not (0 < patch_start_idx < tokens_per_frame):
        raise ValueError(
            f"Expected 0 < patch_start_idx < tokens_per_frame, got patch_start_idx={patch_start_idx}, "
            f"tokens_per_frame={tokens_per_frame}"
        )

    total_tokens = seq_len * tokens_per_frame
    if global_in.shape != (bsz, total_tokens, global_in.shape[-1]):
        raise ValueError(
            f"Expected global_in shape {(bsz, total_tokens, global_in.shape[-1])}, got {tuple(global_in.shape)}"
        )

    x = global_in.view(bsz, seq_len, tokens_per_frame, -1)
    c = x.shape[-1]
    num_special = patch_start_idx
    special = x[:, :, :num_special, :].contiguous().view(bsz, seq_len * num_special, c)
    patch = x[:, :, num_special:, :].contiguous().view(bsz, seq_len * (tokens_per_frame - num_special), c)

    special_pos = None
    patch_pos = None
    if pos_global is not None:
        pos_view = pos_global.view(bsz, seq_len, tokens_per_frame, 2)
        special_pos = pos_view[:, :, :num_special, :].contiguous().view(bsz, seq_len * num_special, 2)
        patch_pos = pos_view[:, :, num_special:, :].contiguous().view(bsz, seq_len * (tokens_per_frame - num_special), 2)

    return special, patch, global_in, special_pos, patch_pos


def _run_asymmetric_attention(
    block: torch.nn.Module,
    *,
    q_tokens: torch.Tensor,
    kv_tokens: torch.Tensor,
    q_pos: Optional[torch.Tensor],
    kv_pos: Optional[torch.Tensor],
) -> torch.Tensor:
    c = q_tokens.shape[-1]
    bsz = q_tokens.shape[0]
    if kv_tokens.shape[0] != bsz:
        raise ValueError(f"Query/KV batch mismatch: {q_tokens.shape[0]} vs {kv_tokens.shape[0]}")

    num_heads = block.attn.num_heads
    head_dim = block.attn.head_dim

    norm_q = block.norm1(q_tokens)
    norm_kv = block.norm1(kv_tokens)

    qkv_weight = block.attn.qkv.weight
    qkv_bias = block.attn.qkv.bias
    q_bias = qkv_bias[:c] if qkv_bias is not None else None
    k_bias = qkv_bias[c : 2 * c] if qkv_bias is not None else None
    v_bias = qkv_bias[2 * c :] if qkv_bias is not None else None

    q = F.linear(norm_q, qkv_weight[:c], q_bias)
    k = F.linear(norm_kv, qkv_weight[c : 2 * c], k_bias)
    v = F.linear(norm_kv, qkv_weight[2 * c :], v_bias)

    q = q.view(bsz, q_tokens.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(bsz, kv_tokens.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(bsz, kv_tokens.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)

    q = block.attn.q_norm(q)
    k = block.attn.k_norm(k)

    if block.attn.rope is not None:
        q = block.attn.rope(q, q_pos)
        k = block.attn.rope(k, kv_pos)

    if block.attn.fused_attn:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=block.attn.attn_drop.p if block.training else 0.0,
        )
    else:
        q = q * block.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = block.attn.attn_drop(attn)
        out = attn @ v

    out = out.transpose(1, 2).reshape(bsz, q_tokens.shape[1], c)
    out = block.attn.proj(out)
    out = block.attn.proj_drop(out)
    return out


def _patch_special_global_block_forward(
    block: torch.nn.Module,
    global_in: torch.Tensor,
    *,
    bsz: int,
    seq_len: int,
    tokens_per_frame: int,
    patch_start_idx: int,
    pos_global: Optional[torch.Tensor],
    apply_mlp: bool,
) -> torch.Tensor:
    if not (0 < patch_start_idx < tokens_per_frame):
        return global_in

    special, patch, all_tokens, special_pos, patch_pos = _split_global_token_groups(
        global_in,
        bsz=bsz,
        seq_len=seq_len,
        tokens_per_frame=tokens_per_frame,
        patch_start_idx=patch_start_idx,
        pos_global=pos_global,
    )

    special_attn = _run_asymmetric_attention(
        block,
        q_tokens=special,
        kv_tokens=all_tokens,
        q_pos=special_pos,
        kv_pos=pos_global,
    )
    patch_attn = _run_asymmetric_attention(
        block,
        q_tokens=patch,
        kv_tokens=special,
        q_pos=patch_pos,
        kv_pos=special_pos,
    )

    special = special + block.ls1(special_attn)
    if apply_mlp:
        special = special + block.ls2(block.mlp(block.norm2(special)))

    patch = patch + block.ls1(patch_attn)
    if apply_mlp:
        patch = patch + block.ls2(block.mlp(block.norm2(patch)))

    num_special = patch_start_idx
    num_patch = tokens_per_frame - num_special
    c = global_in.shape[-1]
    special = special.view(bsz, seq_len, num_special, c)
    patch = patch.view(bsz, seq_len, num_patch, c)
    return torch.cat([special, patch], dim=2).reshape(bsz, seq_len * tokens_per_frame, c)


def _apply_skipped_global_mode(
    model: VGGT,
    global_in: torch.Tensor,
    *,
    layer_idx: int,
    skipped_global_mode: str,
    bsz: int,
    seq_len: int,
    tokens_per_frame: int,
    pos_global: Optional[torch.Tensor],
) -> torch.Tensor:
    if skipped_global_mode == "identity":
        return global_in
    if skipped_global_mode == "special_only":
        return _special_only_global_block_forward(
            model.aggregator.global_blocks[layer_idx],
            global_in,
            bsz=bsz,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            patch_start_idx=model.aggregator.patch_start_idx,
            pos_global=pos_global,
            apply_mlp=True,
        )
    if skipped_global_mode == "special_only_nomlp":
        return _special_only_global_block_forward(
            model.aggregator.global_blocks[layer_idx],
            global_in,
            bsz=bsz,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            patch_start_idx=model.aggregator.patch_start_idx,
            pos_global=pos_global,
            apply_mlp=False,
        )
    if skipped_global_mode == "patch_special":
        return _patch_special_global_block_forward(
            model.aggregator.global_blocks[layer_idx],
            global_in,
            bsz=bsz,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            patch_start_idx=model.aggregator.patch_start_idx,
            pos_global=pos_global,
            apply_mlp=True,
        )
    if skipped_global_mode == "patch_special_nomlp":
        return _patch_special_global_block_forward(
            model.aggregator.global_blocks[layer_idx],
            global_in,
            bsz=bsz,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            patch_start_idx=model.aggregator.patch_start_idx,
            pos_global=pos_global,
            apply_mlp=False,
        )
    raise ValueError(f"Unknown skipped_global_mode: {skipped_global_mode}")


def _forward_with_suffix_replay(
    model: VGGT,
    images: torch.Tensor,
    replay_concat_layers: Sequence[torch.Tensor],
    *,
    start_layer: int,
    skip_start: Optional[int],
    skip_end: Optional[int],
    skipped_global_mode: str,
    pos_frame: Optional[torch.Tensor],
    pos_global: Optional[torch.Tensor],
    capture_global_layer: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    num_layers = len(replay_concat_layers)
    if not (0 <= start_layer < num_layers):
        raise ValueError(f"Invalid start_layer={start_layer} for num_layers={num_layers}")
    if images.dim() == 4:
        images = images.unsqueeze(0)
    if images.dim() != 5:
        raise ValueError(f"Expected images with 4 or 5 dims, got shape {tuple(images.shape)}")
    def _required_head_layers() -> List[int]:
        keep: set[int] = set()
        if model.camera_head is not None and num_layers > 0:
            keep.add(num_layers - 1)
        if model.depth_head is not None:
            keep.update(int(i) for i in getattr(model.depth_head, "intermediate_layer_idx", []))
        if model.point_head is not None:
            keep.update(int(i) for i in getattr(model.point_head, "intermediate_layer_idx", []))
        return sorted(i for i in keep if 0 <= i < num_layers)

    required_layers = _required_head_layers()
    required_set = set(required_layers)
    layer_cache: Dict[int, torch.Tensor] = {}
    captured_global: Optional[torch.Tensor] = None

    def _get_layer(idx: int) -> torch.Tensor:
        cached = layer_cache.get(idx)
        if cached is not None:
            return cached
        t = replay_concat_layers[idx]
        if images.device.type == "cuda":
            t = t.to(device=images.device, non_blocking=True)
        else:
            t = t.to(device=images.device)
        layer_cache[idx] = t
        return t

    bsz, seq_len, p, two_c = _get_layer(start_layer).shape
    if two_c % 2 != 0:
        raise ValueError(f"Expected even channel count in replay cache, got {two_c}")
    c = two_c // 2
    selected_layers: Dict[int, torch.Tensor] = {}
    for idx in required_layers:
        if idx < start_layer:
            selected_layers[idx] = _get_layer(idx)
    frame_view = _get_layer(start_layer)[..., :c]
    for layer_idx in range(start_layer, num_layers):
        global_in = frame_view.view(bsz, seq_len * p, c)
        if skip_start is not None and skip_end is not None and skip_start <= layer_idx <= skip_end:
            global_out = _apply_skipped_global_mode(
                model,
                global_in,
                layer_idx=layer_idx,
                skipped_global_mode=skipped_global_mode,
                bsz=bsz,
                seq_len=seq_len,
                tokens_per_frame=p,
                pos_global=pos_global,
            )
        else:
            global_out = model.aggregator.global_blocks[layer_idx](global_in, pos=pos_global)
        global_view = global_out.view(bsz, seq_len, p, c)
        if capture_global_layer is not None and layer_idx == capture_global_layer:
            captured_global = global_view
        concat_tokens = torch.cat([frame_view, global_view], dim=-1)
        if layer_idx in required_set:
            selected_layers[layer_idx] = concat_tokens
        if layer_idx + 1 >= num_layers:
            break
        frame_in = global_out.view(bsz * seq_len, p, c)
        frame_out = model.aggregator.frame_blocks[layer_idx + 1](frame_in, pos=pos_frame)
        frame_view = frame_out.view(bsz, seq_len, p, c)
        del global_in, global_out, global_view, concat_tokens, frame_in, frame_out
    if model.camera_head is not None and (num_layers - 1) not in selected_layers:
        raise RuntimeError("Replay sparse layers missing final layer required by camera head.")
    aggregated_tokens_list: List[torch.Tensor] = [None] * num_layers  # type: ignore[list-item]
    for idx, t in selected_layers.items():
        aggregated_tokens_list[idx] = t
    pred = model.forward_from_aggregated_tokens(
        aggregated_tokens_list,
        model.aggregator.patch_start_idx,
        images,
    )
    if capture_global_layer is not None:
        if captured_global is None:
            raise RuntimeError(
                f"Failed to capture replayed global output at layer {capture_global_layer} "
                f"(start_layer={start_layer}, num_layers={num_layers})."
            )
        pred["_captured_global"] = captured_global
    return pred


def _replay_global_output_at_layer(
    model: VGGT,
    replay_concat_layers: Sequence[torch.Tensor],
    *,
    device: torch.device,
    start_layer: int,
    target_layer: int,
    skip_start: int,
    skip_end: int,
    skipped_global_mode: str,
    pos_frame: Optional[torch.Tensor],
    pos_global: Optional[torch.Tensor],
) -> torch.Tensor:
    num_layers = len(replay_concat_layers)
    if not (0 <= start_layer < num_layers):
        raise ValueError(f"Invalid start_layer={start_layer} for num_layers={num_layers}")
    if not (start_layer <= target_layer < num_layers):
        raise ValueError(
            f"Invalid target_layer={target_layer}; expected {start_layer} <= target_layer < {num_layers}"
        )

    start_tokens = replay_concat_layers[start_layer].to(device=device, non_blocking=(device.type == "cuda"))
    bsz, seq_len, p, two_c = start_tokens.shape
    if two_c % 2 != 0:
        raise ValueError(f"Expected even channel count in replay cache, got {two_c}")
    c = two_c // 2
    frame_view = start_tokens[..., :c]

    for layer_idx in range(start_layer, target_layer + 1):
        global_in = frame_view.view(bsz, seq_len * p, c)
        if skip_start <= layer_idx <= skip_end:
            global_out = _apply_skipped_global_mode(
                model,
                global_in,
                layer_idx=layer_idx,
                skipped_global_mode=skipped_global_mode,
                bsz=bsz,
                seq_len=seq_len,
                tokens_per_frame=p,
                pos_global=pos_global,
            )
        else:
            global_out = model.aggregator.global_blocks[layer_idx](global_in, pos=pos_global)
        global_view = global_out.view(bsz, seq_len, p, c)
        if layer_idx == target_layer:
            return global_view
        frame_in = global_out.view(bsz * seq_len, p, c)
        frame_out = model.aggregator.frame_blocks[layer_idx + 1](frame_in, pos=pos_frame)
        frame_view = frame_out.view(bsz, seq_len, p, c)

    raise RuntimeError("Unexpected replay flow when capturing target global output.")


def _collect_skip_range_loss(
    scenes_meta: List[Dict[str, Any]],
    cache_dir: Path,
    data_root: Path,
    hf_model: str,
    max_frames: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_layers: int,
    timing: bool,
    output_dir: Path,
    gt_ply_dir: Optional[Path],
    depth_conf_thresh: float,
    chamfer_max_dist: float,
    skip_gt_eval: bool,
    skip_max_span: int,
    export_visualization: bool,
    stage2_vis_root: Optional[Path],
    skipped_global_mode: str,
    vis_pair_suffix: str,
    stage1_sanity_check: bool,
    stage1_sanity_atol: float,
    stage1_sanity_rtol: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not hf_model.strip():
        raise SystemExit("--hf-model must be non-empty when skip-range loss is enabled.")

    discovered = discover_scannet_scenes(data_root)
    scene_map: Dict[str, SceneData] = {s.scene_id: s for s in discovered}
    eval_path = cache_dir / "eval_scannet.json"
    if not eval_path.exists():
        raise SystemExit(
            f"Missing {eval_path}; Stage2 skip-range GT comparison now requires Stage1 eval results."
        )
    try:
        stage1_eval_map = read_eval_scannet_rows_by_scene(eval_path)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    selected: List[Tuple[Dict[str, Any], SceneData, Dict[str, Any]]] = []
    for item in scenes_meta:
        sid = str(item["scene_id"])
        if sid not in scene_map:
            raise SystemExit(
                f"Stage2 skip-loss requires complete data, but scene_id {sid} was not found under {data_root}."
            )
        if sid not in stage1_eval_map:
            raise SystemExit(
                f"Stage2 skip-loss requires complete Stage1 eval rows, but {sid} is missing in "
                f"{cache_dir / 'eval_scannet.json'}."
            )
        selected.append((item, scene_map[sid], stage1_eval_map[sid]))

    if not selected:
        raise SystemExit("Skip-range loss enabled but no valid scene with Stage1 predictions was found.")

    model = VGGT.from_pretrained(hf_model).to(device).eval()
    use_amp = device.type == "cuda"
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    sum_loss_by_metric = {k: [[0.0] * num_layers for _ in range(num_layers)] for k in GT_KEYS_FOR_SKIP}
    cnt_loss_by_metric = {k: [[0] * num_layers for _ in range(num_layers)] for k in GT_KEYS_FOR_SKIP}
    per_scene_rows: List[Dict[str, Any]] = []
    global_breakdown_totals: Dict[str, float] = {
        "replay_prep": 0.0,
        "replay_forward": 0.0,
        "to_cpu_payload": 0.0,
        "export_vis": 0.0,
        "gt_eval": 0.0,
        "postprocess": 0.0,
        "pair_total": 0.0,
    }
    global_breakdown_pairs = 0

    for si, (_item, scene, stage1_eval_row) in enumerate(selected):
        t0_scene = time.perf_counter()
        scene_eff = select_scene_image_paths_fastvggt(scene, max_frames)
        if not scene_eff.image_paths:
            raise SystemExit(
                f"Stage2 skip-loss requires complete data, but scene {scene.scene_id} has no valid frames "
                "after FastVGGT selection."
            )
        if stage1_sanity_check:
            _check_stage1_selection_match(scene.scene_id, _item, scene_eff)
        images = load_scene_images(scene_eff, mode="crop").to(device)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if skip_gt_eval:
            raise SystemExit("GT drop loss requires GT eval. Please set SKIP_GT_EVAL=0 for Stage2.")
        eval_ctx = build_fastvggt_scene_eval_context(
            scene_eff,
            data_root,
            gt_ply_dir,
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
        )
        stage1_gt = metrics_float_dict(stage1_eval_row)
        hw = (int(images.shape[-2]), int(images.shape[-1]))
        replay_concat_layers = _load_scene_replay_cache(cache_dir, _item, num_layers)
        pos_frame, pos_global = _build_aggregator_positions(model, images)
        if stage1_sanity_check:
            with torch.no_grad(), torch.autocast(
                device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp
            ):
                noop_pred = _forward_with_suffix_replay(
                    model,
                    images,
                    replay_concat_layers,
                    start_layer=0,
                    skip_start=None,
                    skip_end=None,
                    skipped_global_mode=skipped_global_mode,
                    pos_frame=pos_frame,
                    pos_global=pos_global,
                )
            noop_pred_cpu = predictions_to_cpu_float(noop_pred)
            noop_payload = build_scene_prediction_payload(noop_pred_cpu, hw)
            noop_eval_row = build_scannet_eval_row_from_payload_with_context(
                noop_payload,
                scene_eff,
                eval_ctx,
                depth_conf_thresh=depth_conf_thresh,
                chamfer_max_dist=chamfer_max_dist,
                inference_time_ms=0.0,
            )
            noop_gt = metrics_float_dict(noop_eval_row)
            _check_stage1_noop_metrics_match(
                scene.scene_id,
                stage1_gt,
                noop_gt,
                atol=stage1_sanity_atol,
                rtol=stage1_sanity_rtol,
            )
            del noop_pred, noop_pred_cpu
            if timing:
                print(
                    f"[timing] stage1 sanity check passed for {scene.scene_id} "
                    f"(mode={skipped_global_mode}, atol={stage1_sanity_atol}, rtol={stage1_sanity_rtol})",
                    flush=True,
                )

        total_pairs = 0
        for ii in range(num_layers):
            total_pairs += max(0, min(num_layers - 1, ii + skip_max_span - 1) - ii + 1)
        done_pairs = 0
        scene_breakdown_totals: Dict[str, float] = {
            "replay_prep": 0.0,
            "replay_forward": 0.0,
            "to_cpu_payload": 0.0,
            "export_vis": 0.0,
            "gt_eval": 0.0,
            "postprocess": 0.0,
            "pair_total": 0.0,
        }
        scene_breakdown_pairs = 0

        scene_loss_upper_by_metric = {k: [] for k in GT_KEYS_FOR_SKIP}
        for i in range(num_layers):
            for j in range(i, num_layers):
                if j - i + 1 > skip_max_span:
                    for k in GT_KEYS_FOR_SKIP:
                        scene_loss_upper_by_metric[k].append(float("nan"))
                    continue
                t_pair0 = time.perf_counter()
                t_replay_prep = 0.0
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast(
                    device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp
                ):
                    pred = _forward_with_suffix_replay(
                        model,
                        images,
                        replay_concat_layers,
                        start_layer=i,
                        skip_start=i,
                        skip_end=j,
                        skipped_global_mode=skipped_global_mode,
                        pos_frame=pos_frame,
                        pos_global=pos_global,
                    )
                t_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred_cpu = predictions_to_cpu_float(pred)
                payload = build_scene_prediction_payload(pred_cpu, hw)
                t_payload = time.perf_counter() - t0
                t_vis = 0.0
                if export_visualization and stage2_vis_root is not None:
                    tv0 = time.perf_counter()
                    _export_stage2_skip_pair_visualization(
                        stage2_vis_root=stage2_vis_root,
                        scene_eff=scene_eff,
                        pair_name=f"skip_l{i:02d}_l{j:02d}{vis_pair_suffix}",
                        images=images,
                        prediction_payload=payload,
                        depth_conf_thresh=depth_conf_thresh,
                    )
                    t_vis = time.perf_counter() - tv0
                t0 = time.perf_counter()
                pruned_eval_row = build_scannet_eval_row_from_payload_with_context(
                    payload,
                    scene_eff,
                    eval_ctx,
                    depth_conf_thresh=depth_conf_thresh,
                    chamfer_max_dist=chamfer_max_dist,
                    inference_time_ms=0.0,
                )
                t_eval = time.perf_counter() - t0
                t0 = time.perf_counter()
                pruned_gt = metrics_float_dict(pruned_eval_row)
                per_key = _aligned_change_per_key(stage1_gt, pruned_gt)
                for k in GT_KEYS_FOR_SKIP:
                    if k not in per_key:
                        scene_loss_upper_by_metric[k].append(float("nan"))
                    else:
                        v = float(per_key[k])
                        scene_loss_upper_by_metric[k].append(v)
                        sum_loss_by_metric[k][i][j] += v
                        cnt_loss_by_metric[k][i][j] += 1
                t_post = time.perf_counter() - t0
                del pred, pred_cpu
                done_pairs += 1
                pair_total = time.perf_counter() - t_pair0
                scene_breakdown_totals["replay_prep"] += t_replay_prep
                scene_breakdown_totals["replay_forward"] += t_forward
                scene_breakdown_totals["to_cpu_payload"] += t_payload
                scene_breakdown_totals["export_vis"] += t_vis
                scene_breakdown_totals["gt_eval"] += t_eval
                scene_breakdown_totals["postprocess"] += t_post
                scene_breakdown_totals["pair_total"] += pair_total
                scene_breakdown_pairs += 1
                global_breakdown_totals["replay_prep"] += t_replay_prep
                global_breakdown_totals["replay_forward"] += t_forward
                global_breakdown_totals["to_cpu_payload"] += t_payload
                global_breakdown_totals["export_vis"] += t_vis
                global_breakdown_totals["gt_eval"] += t_eval
                global_breakdown_totals["postprocess"] += t_post
                global_breakdown_totals["pair_total"] += pair_total
                global_breakdown_pairs += 1
                if timing:
                    span = j - i + 1
                    # Print per (i,j) timing so long runs show progress.
                    print(
                        f"[timing] skip-loss pair {done_pairs}/{total_pairs} "
                        f"(i={i}, j={j}, span={span}): total={pair_total:.2f}s "
                        f"[prep={t_replay_prep:.2f}s, replay={t_forward:.2f}s, payload={t_payload:.2f}s, "
                        f"vis={t_vis:.2f}s, eval={t_eval:.2f}s, post={t_post:.2f}s]",
                        flush=True,
                    )

        per_scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "skipped_global_mode": skipped_global_mode,
                "skip_range_loss_upper_by_metric": scene_loss_upper_by_metric,
            }
        )
        if timing:
            if scene_breakdown_pairs > 0:
                scene_total = max(scene_breakdown_totals["pair_total"], 1e-12)
                print(
                    "[timing] skip-loss scene breakdown "
                    f"{scene.scene_id}: "
                    f"prep={scene_breakdown_totals['replay_prep']/scene_total*100:.1f}%, "
                    f"replay={scene_breakdown_totals['replay_forward']/scene_total*100:.1f}%, "
                    f"payload={scene_breakdown_totals['to_cpu_payload']/scene_total*100:.1f}%, "
                    f"vis={scene_breakdown_totals['export_vis']/scene_total*100:.1f}%, "
                    f"eval={scene_breakdown_totals['gt_eval']/scene_total*100:.1f}%, "
                    f"post={scene_breakdown_totals['postprocess']/scene_total*100:.1f}% "
                    f"(pairs={scene_breakdown_pairs}, pair_total={scene_breakdown_totals['pair_total']:.2f}s)",
                    flush=True,
                )
            print(
                f"[timing] skip-loss scene {si + 1}/{len(selected)} {scene.scene_id}: "
                f"{time.perf_counter() - t0_scene:.2f}s",
                flush=True,
            )

    mean_upper_by_metric: Dict[str, List[float]] = {}
    for k in GT_KEYS_FOR_SKIP:
        mean_upper: List[float] = []
        for i in range(num_layers):
            for j in range(i, num_layers):
                if cnt_loss_by_metric[k][i][j] <= 0:
                    mean_upper.append(float("nan"))
                else:
                    mean_upper.append(sum_loss_by_metric[k][i][j] / cnt_loss_by_metric[k][i][j])
        mean_upper_by_metric[k] = mean_upper

    return (
        {
            "enabled": True,
            "metric": "neg_direction_aligned_change_vs_stage1_per_metric",
            "gt_keys_used": list(GT_KEYS_FOR_SKIP),
            "skip_max_span": int(skip_max_span),
            "skipped_global_mode": skipped_global_mode,
            "num_scenes_used": len(selected),
            "stage1_sanity_check": bool(stage1_sanity_check),
            "stage1_sanity_atol": float(stage1_sanity_atol),
            "stage1_sanity_rtol": float(stage1_sanity_rtol),
            "skip_range_loss_upper_by_metric": mean_upper_by_metric,
            "timing_breakdown_total_seconds": global_breakdown_totals if global_breakdown_pairs > 0 else {},
            "timing_breakdown_pairs": int(global_breakdown_pairs),
        },
        per_scene_rows,
    )


def main() -> None:
    args = _build_parser().parse_args()
    t_main_start = time.perf_counter()
    torch.manual_seed(args.seed)
    similarity_report_name = str(args.similarity_full_report_name).strip() or "full_report_similarity.json"
    skip_report_name = str(args.skip_full_report_name).strip() or "full_report_skip.json"
    similarity_per_scene_suffix = str(args.similarity_per_scene_report_suffix).strip() or "_similarity.json"
    skip_per_scene_suffix = str(args.skip_per_scene_report_suffix).strip() or "_skip.json"
    if not similarity_per_scene_suffix.endswith(".json"):
        raise SystemExit(
            f"similarity_per_scene_report_suffix must end with .json, got: {similarity_per_scene_suffix}"
        )
    if not skip_per_scene_suffix.endswith(".json"):
        raise SystemExit(f"skip_per_scene_report_suffix must end with .json, got: {skip_per_scene_suffix}")
    similarity_out_dir = (
        args.similarity_output_dir
        if args.similarity_output_dir is not None
        else Path("analysis_outputs/scannet_redundancy_similarity")
    )
    skip_out_dir = args.skip_output_dir if args.skip_output_dir is not None else Path("analysis_outputs/scannet_redundancy_skip")
    similarity_report_path = similarity_out_dir / similarity_report_name
    skip_report_path = skip_out_dir / skip_report_name
    fill_value = _parse_heatmap_fill(args.heatmap_fill)
    analysis_device = _resolve_analysis_device(args.device)
    fusion_modes = _normalize_skipped_global_modes(args.fusion_modes)
    fusion_max_span = max(1, int(args.fusion_max_span))
    fusion_num_procs = max(1, int(args.fusion_num_procs))
    vis_root_stage2 = args.visualization_dir if args.visualization_dir is not None else (similarity_out_dir / "visualization")

    compute_similarity = bool(args.fusion_compute_similarity)
    sim_metrics: List[str] = []
    if compute_similarity:
        seen_metrics: set[str] = set()
        for m in args.fusion_similarity_metrics:
            mk = str(m).strip().lower()
            if not mk or mk in seen_metrics:
                continue
            seen_metrics.add(mk)
            sim_metrics.append(mk)
        if not sim_metrics:
            raise SystemExit("fusion_similarity_metrics is empty after normalization.")
    compute_pwcca = compute_similarity and ("pwcca" in sim_metrics)
    write_aggregate = bool(args.fusion_write_aggregate)
    skip_gt_eval = bool(args.skip_gt_eval)

    if bool(args.fusion_aggregate_from_per_scene):
        _aggregate_from_per_scene_split_outputs(
            similarity_out_dir,
            skip_out_dir,
            similarity_report_name=similarity_report_name,
            skip_report_name=skip_report_name,
            fusion_modes=fusion_modes,
            sim_metrics=sim_metrics,
            save_heatmap_png=bool(args.save_heatmap_png),
            fill_value=fill_value,
            compute_pwcca=compute_pwcca,
            fusion_num_procs=fusion_num_procs,
            analysis_device=analysis_device,
            skip_gt_eval=skip_gt_eval,
        )
        print(
            f"Aggregated split fusion reports from per-scene files into {similarity_report_path} and {skip_report_path}"
        )
        return

    scenes_meta = _read_manifest(args.cache_dir)
    if args.scene_ids_file is not None:
        if not args.scene_ids_file.exists():
            raise SystemExit(f"scene_ids_file not found: {args.scene_ids_file}")
        wanted = {
            line.strip()
            for line in args.scene_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not wanted:
            raise SystemExit(f"scene_ids_file is empty: {args.scene_ids_file}")
        scenes_meta = [item for item in scenes_meta if str(item.get("scene_id", "")).strip() in wanted]
        if not scenes_meta:
            raise SystemExit(f"No manifest scenes matched scene_ids_file: {args.scene_ids_file}")
    similarity_out_dir.mkdir(parents=True, exist_ok=True)
    skip_out_dir.mkdir(parents=True, exist_ok=True)
    save_per_scene = bool(args.fusion_save_per_scene)
    per_scene_similarity_dir: Optional[Path] = similarity_out_dir / "per_scene" if save_per_scene else None
    per_scene_skip_dir: Optional[Path] = skip_out_dir / "per_scene" if save_per_scene else None
    if per_scene_similarity_dir is not None:
        per_scene_similarity_dir.mkdir(parents=True, exist_ok=True)
    if per_scene_skip_dir is not None:
        per_scene_skip_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    skip_loss_dtype = dtype_map[args.skip_loss_dtype]
    skip_loss_device = torch.device(
        args.skip_loss_device if torch.cuda.is_available() or args.skip_loss_device == "cpu" else "cpu"
    )
    use_amp = skip_loss_device.type == "cuda"
    autocast_device_type = "cuda" if skip_loss_device.type == "cuda" else "cpu"

    scene_map: Dict[str, SceneData] = {}
    stage1_eval_map: Dict[str, Dict[str, Any]] = {}
    selected: List[Tuple[Dict[str, Any], Optional[SceneData], Optional[Dict[str, Any]]]] = []

    if skip_gt_eval:
        selected = [(item, None, None) for item in scenes_meta]
        if not selected:
            raise SystemExit("No scenes were selected from manifest for fusion analysis.")
    else:
        discovered = discover_scannet_scenes(args.data_root)
        scene_map = {s.scene_id: s for s in discovered}
        eval_path = args.cache_dir / "eval_scannet.json"
        if not eval_path.exists():
            raise SystemExit(
                f"Missing {eval_path}; fusion analysis requires Stage1 eval results for GT delta computation."
            )
        try:
            stage1_eval_map = read_eval_scannet_rows_by_scene(eval_path)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e

        for item in scenes_meta:
            sid = str(item["scene_id"])
            if sid not in scene_map:
                raise SystemExit(
                    f"Fusion analysis requires complete data, but scene_id {sid} was not found under {args.data_root}."
                )
            if sid not in stage1_eval_map:
                raise SystemExit(
                    f"Fusion analysis requires complete Stage1 eval rows, but {sid} is missing in "
                    f"{args.cache_dir / 'eval_scannet.json'}."
                )
            selected.append((item, scene_map[sid], stage1_eval_map[sid]))
        if not selected:
            raise SystemExit("No valid scene with Stage1 predictions was found for fusion analysis.")

    num_layers = _infer_num_layers_from_manifest(args.cache_dir, scenes_meta)
    model = VGGT.from_pretrained(args.hf_model).to(skip_loss_device).eval()

    num_pairs = num_layers * (num_layers + 1) // 2
    sim_sum_patch: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {m: None for m in sim_metrics} for mode in fusion_modes
    }
    sim_cnt_patch: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {m: None for m in sim_metrics} for mode in fusion_modes
    }
    sim_sum_special: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {m: None for m in sim_metrics} for mode in fusion_modes
    }
    sim_cnt_special: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {m: None for m in sim_metrics} for mode in fusion_modes
    }
    gt_sum: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {k: None for k in GT_KEYS_FOR_SKIP} for mode in fusion_modes
    }
    gt_cnt: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        mode: {k: None for k in GT_KEYS_FOR_SKIP} for mode in fusion_modes
    }

    scene_ids: List[str] = []
    scene_score_dicts: List[Dict[str, Any]] = []
    score_keys_set: set[str] = set()

    for si, (item, scene, stage1_eval_row) in enumerate(selected):
        sid = str(item.get("scene_id", ""))
        print(f"[fusion] scene {si + 1}/{len(selected)}: {sid}", flush=True)
        scene_ids.append(sid)
        scene_scores_raw = item.get("scene_scores", {})
        scene_scores = dict(scene_scores_raw) if isinstance(scene_scores_raw, dict) else {}
        scene_score_dicts.append(scene_scores)
        score_keys_set.update(scene_scores.keys())

        replay_concat_layers = _load_scene_replay_cache(args.cache_dir, item, num_layers)
        patch_start_idx = int(item.get("patch_start_idx", model.aggregator.patch_start_idx))

        eval_ctx = None
        scene_eff: Optional[SceneData] = None
        stage1_gt: Dict[str, float] = {}
        if skip_gt_eval:
            cache_row = _load_scene_cache_row(args.cache_dir, item)
            h, w = _infer_image_hw_for_replay(
                cache_row,
                replay_concat_layers,
                patch_start_idx=patch_start_idx,
                patch_size=int(model.aggregator.patch_size),
            )
            seq_len_cache = int(cache_row.get("num_frames", replay_concat_layers[0].shape[1]))
            if seq_len_cache <= 0:
                seq_len_cache = int(replay_concat_layers[0].shape[1])
            images = torch.zeros(
                (1, seq_len_cache, 3, h, w),
                device=skip_loss_device,
                dtype=torch.float32,
            )
            hw = (h, w)
        else:
            assert scene is not None
            assert stage1_eval_row is not None

            scene_eff = select_scene_image_paths_fastvggt(scene, args.max_frames)
            if not scene_eff.image_paths:
                raise SystemExit(
                    f"Fusion analysis requires complete data, but scene {scene.scene_id} has no valid frames "
                    "after FastVGGT selection."
                )
            if not bool(args.disable_stage1_sanity_check):
                _check_stage1_selection_match(scene.scene_id, item, scene_eff)

            images = load_scene_images(scene_eff, mode="crop").to(skip_loss_device)
            if images.dim() == 4:
                images = images.unsqueeze(0)
            eval_ctx = build_fastvggt_scene_eval_context(
                scene_eff,
                args.data_root,
                args.gt_ply_dir,
                image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            )
            stage1_gt = metrics_float_dict(stage1_eval_row)
            hw = (int(images.shape[-2]), int(images.shape[-1]))

        pos_frame, pos_global = _build_aggregator_positions(model, images)

        if not skip_gt_eval and not bool(args.disable_stage1_sanity_check):
            with torch.no_grad(), torch.autocast(
                device_type=autocast_device_type, dtype=skip_loss_dtype, enabled=use_amp
            ):
                noop_pred = _forward_with_suffix_replay(
                    model,
                    images,
                    replay_concat_layers,
                    start_layer=0,
                    skip_start=None,
                    skip_end=None,
                    skipped_global_mode=fusion_modes[0],
                    pos_frame=pos_frame,
                    pos_global=pos_global,
                )
            noop_pred_cpu = predictions_to_cpu_float(noop_pred)
            noop_payload = build_scene_prediction_payload(noop_pred_cpu, hw)
            noop_eval_row = build_scannet_eval_row_from_payload_with_context(
                noop_payload,
                scene_eff,
                eval_ctx,
                depth_conf_thresh=args.depth_conf_thresh,
                chamfer_max_dist=args.chamfer_max_dist,
                inference_time_ms=0.0,
            )
            noop_gt = metrics_float_dict(noop_eval_row)
            _check_stage1_noop_metrics_match(
                scene.scene_id,
                stage1_gt,
                noop_gt,
                atol=float(args.stage1_sanity_atol),
                rtol=float(args.stage1_sanity_rtol),
            )
            del noop_pred, noop_pred_cpu

        original_global_cache: Dict[int, torch.Tensor] = {}

        def _orig_global(layer_idx: int) -> torch.Tensor:
            cached = original_global_cache.get(layer_idx)
            if cached is not None:
                return cached
            concat = replay_concat_layers[layer_idx].to(
                device=skip_loss_device,
                non_blocking=(skip_loss_device.type == "cuda"),
            )
            c_local = concat.shape[-1] // 2
            g = concat[..., c_local:]
            original_global_cache[layer_idx] = g
            return g

        scene_fusion_by_mode: Dict[str, Dict[str, Any]] = {}
        for mi, mode in enumerate(fusion_modes):
            if args.timing:
                print(
                    f"[timing] fusion scene {si + 1}/{len(selected)} {sid} | mode {mi + 1}/{len(fusion_modes)}: {mode}",
                    flush=True,
                )
            scene_patch_upper_by_metric = {m: [] for m in sim_metrics}
            scene_special_upper_by_metric = {m: [] for m in sim_metrics}
            scene_gt_upper_by_metric = {k: [] for k in GT_KEYS_FOR_SKIP}

            for i in range(num_layers):
                for j in range(i, num_layers):
                    if j - i + 1 > fusion_max_span:
                        for m in sim_metrics:
                            scene_patch_upper_by_metric[m].append(float("nan"))
                            scene_special_upper_by_metric[m].append(float("nan"))
                        for k in GT_KEYS_FOR_SKIP:
                            scene_gt_upper_by_metric[k].append(float("nan"))
                        continue

                    t_pair0 = time.perf_counter()
                    pred: Optional[Dict[str, torch.Tensor]] = None
                    new_global: Optional[torch.Tensor] = None
                    with torch.no_grad(), torch.autocast(
                        device_type=autocast_device_type,
                        dtype=skip_loss_dtype,
                        enabled=use_amp,
                    ):
                        if compute_similarity and not skip_gt_eval:
                            pred = _forward_with_suffix_replay(
                                model,
                                images,
                                replay_concat_layers,
                                start_layer=i,
                                skip_start=i,
                                skip_end=j,
                                skipped_global_mode=mode,
                                pos_frame=pos_frame,
                                pos_global=pos_global,
                                capture_global_layer=j,
                            )
                            captured_global = pred.get("_captured_global")
                            if isinstance(captured_global, torch.Tensor):
                                new_global = captured_global
                            pred.pop("_captured_global", None)
                        elif compute_similarity:
                            new_global = _replay_global_output_at_layer(
                                model,
                                replay_concat_layers,
                                device=skip_loss_device,
                                start_layer=i,
                                target_layer=j,
                                skip_start=i,
                                skip_end=j,
                                skipped_global_mode=mode,
                                pos_frame=pos_frame,
                                pos_global=pos_global,
                            )
                        elif not skip_gt_eval:
                            pred = _forward_with_suffix_replay(
                                model,
                                images,
                                replay_concat_layers,
                                start_layer=i,
                                skip_start=i,
                                skip_end=j,
                                skipped_global_mode=mode,
                                pos_frame=pos_frame,
                                pos_global=pos_global,
                            )

                    if compute_similarity:
                        assert new_global is not None
                        orig_global = _orig_global(j)
                        c = int(orig_global.shape[-1])
                        orig_patch = orig_global[:, :, patch_start_idx:, :].reshape(-1, c).float()
                        new_patch = new_global[:, :, patch_start_idx:, :].reshape(-1, c).float()
                        orig_special = orig_global[:, :, :patch_start_idx, :].reshape(-1, c).float()
                        new_special = new_global[:, :, :patch_start_idx, :].reshape(-1, c).float()

                        patch_sim = paired_layer_similarity_matrices(
                            (orig_patch,),
                            (new_patch,),
                            compute_pwcca=compute_pwcca,
                            device=analysis_device,
                        )
                        special_sim = paired_layer_similarity_matrices(
                            (orig_special,),
                            (new_special,),
                            compute_pwcca=compute_pwcca,
                            device=analysis_device,
                        )
                        for m in sim_metrics:
                            pv = patch_sim.get(m)
                            sv = special_sim.get(m)
                            scene_patch_upper_by_metric[m].append(
                                float(pv[0, 0].item()) if pv is not None else float("nan")
                            )
                            scene_special_upper_by_metric[m].append(
                                float(sv[0, 0].item()) if sv is not None else float("nan")
                            )

                    if skip_gt_eval:
                        for k in GT_KEYS_FOR_SKIP:
                            scene_gt_upper_by_metric[k].append(float("nan"))
                    else:
                        assert pred is not None
                        assert eval_ctx is not None
                        assert scene_eff is not None
                        pred_cpu = predictions_to_cpu_float(pred)
                        payload = build_scene_prediction_payload(pred_cpu, hw)
                        pruned_eval_row = build_scannet_eval_row_from_payload_with_context(
                            payload,
                            scene_eff,
                            eval_ctx,
                            depth_conf_thresh=args.depth_conf_thresh,
                            chamfer_max_dist=args.chamfer_max_dist,
                            inference_time_ms=0.0,
                        )
                        pruned_gt = metrics_float_dict(pruned_eval_row)
                        gt_delta = _aligned_change_per_key(stage1_gt, pruned_gt)
                        for k in GT_KEYS_FOR_SKIP:
                            scene_gt_upper_by_metric[k].append(float(gt_delta[k]) if k in gt_delta else float("nan"))

                    if args.timing:
                        print(
                            f"[timing] fusion scene {si + 1}/{len(selected)} {sid} | mode={mode} | "
                            f"(i={i},j={j},span={j - i + 1}) total={time.perf_counter() - t_pair0:.2f}s",
                            flush=True,
                        )

            scene_fusion_by_mode[mode] = {
                "global_patch_similarity_upper_by_metric": scene_patch_upper_by_metric,
                "global_special_similarity_upper_by_metric": scene_special_upper_by_metric,
                "gt_delta_upper_by_metric": scene_gt_upper_by_metric,
            }

            for m in sim_metrics:
                pt = torch.tensor(scene_patch_upper_by_metric[m], dtype=torch.float64)
                st = torch.tensor(scene_special_upper_by_metric[m], dtype=torch.float64)
                pt_mask = torch.isfinite(pt).to(torch.float64)
                st_mask = torch.isfinite(st).to(torch.float64)
                pt_clean = torch.nan_to_num(pt, nan=0.0)
                st_clean = torch.nan_to_num(st, nan=0.0)
                if sim_sum_patch[mode][m] is None:
                    sim_sum_patch[mode][m] = pt_clean
                    sim_cnt_patch[mode][m] = pt_mask
                else:
                    sim_sum_patch[mode][m] = sim_sum_patch[mode][m] + pt_clean  # type: ignore[operator]
                    sim_cnt_patch[mode][m] = sim_cnt_patch[mode][m] + pt_mask  # type: ignore[operator]
                if sim_sum_special[mode][m] is None:
                    sim_sum_special[mode][m] = st_clean
                    sim_cnt_special[mode][m] = st_mask
                else:
                    sim_sum_special[mode][m] = sim_sum_special[mode][m] + st_clean  # type: ignore[operator]
                    sim_cnt_special[mode][m] = sim_cnt_special[mode][m] + st_mask  # type: ignore[operator]

            for k in GT_KEYS_FOR_SKIP:
                gt_t = torch.tensor(scene_gt_upper_by_metric[k], dtype=torch.float64)
                gt_mask = torch.isfinite(gt_t).to(torch.float64)
                gt_clean = torch.nan_to_num(gt_t, nan=0.0)
                if gt_sum[mode][k] is None:
                    gt_sum[mode][k] = gt_clean
                    gt_cnt[mode][k] = gt_mask
                else:
                    gt_sum[mode][k] = gt_sum[mode][k] + gt_clean  # type: ignore[operator]
                    gt_cnt[mode][k] = gt_cnt[mode][k] + gt_mask  # type: ignore[operator]

        if save_per_scene:
            assert per_scene_similarity_dir is not None
            assert per_scene_skip_dir is not None
            scene_report = {
                "scene_id": sid,
                "num_layers": num_layers,
                "fusion_storage": "upper_triangle_row_major_including_diagonal",
                "fusion_max_span": fusion_max_span,
                "fusion_modes": fusion_modes,
                "fusion_similarity_metrics": sim_metrics,
                "scene_scores": scene_scores,
                "fusion_by_mode": scene_fusion_by_mode,
            }
            scene_modes = scene_report.get("fusion_by_mode", {})
            similarity_scene_report = {
                "scene_id": sid,
                "num_layers": scene_report["num_layers"],
                "fusion_storage": scene_report["fusion_storage"],
                "fusion_max_span": scene_report["fusion_max_span"],
                "fusion_modes": scene_report["fusion_modes"],
                "fusion_similarity_metrics": scene_report["fusion_similarity_metrics"],
                "scene_scores": scene_report["scene_scores"],
                "report_kind": "similarity",
                "fusion_by_mode": {
                    mode: {
                        "global_patch_similarity_upper_by_metric": mode_map["global_patch_similarity_upper_by_metric"],
                        "global_special_similarity_upper_by_metric": mode_map["global_special_similarity_upper_by_metric"],
                    }
                    for mode, mode_map in scene_modes.items()
                    if isinstance(mode_map, dict)
                },
            }
            skip_scene_report = {
                "scene_id": sid,
                "num_layers": scene_report["num_layers"],
                "fusion_storage": scene_report["fusion_storage"],
                "fusion_max_span": scene_report["fusion_max_span"],
                "fusion_modes": scene_report["fusion_modes"],
                "scene_scores": scene_report["scene_scores"],
                "report_kind": "skip",
                "fusion_by_mode": {
                    mode: {"gt_delta_upper_by_metric": mode_map["gt_delta_upper_by_metric"]}
                    for mode, mode_map in scene_modes.items()
                    if isinstance(mode_map, dict)
                },
            }
            safe_id = sid.replace("/", "_")
            (per_scene_similarity_dir / f"{safe_id}{similarity_per_scene_suffix}").write_text(
                json.dumps(similarity_scene_report, indent=2), encoding="utf-8"
            )
            (per_scene_skip_dir / f"{safe_id}{skip_per_scene_suffix}").write_text(
                json.dumps(skip_scene_report, indent=2), encoding="utf-8"
            )
            if args.save_heatmap_png and isinstance(scene_modes, dict):
                for mode, mode_map in scene_modes.items():
                    if not isinstance(mode_map, dict):
                        continue
                    if compute_similarity:
                        _write_redundancy_heatmaps(
                            mode_map["global_patch_similarity_upper_by_metric"],
                            mode_map["global_special_similarity_upper_by_metric"],
                            per_scene_similarity_dir,
                            num_layers,
                            fill_value,
                            compute_pwcca,
                            similarity_mode="intra_layer_frame_vs_global",
                            title_suffix=f" ({sid}, {mode})",
                            fname_prefix=f"{safe_id}_{mode}_",
                        )
                    for k in GT_KEYS_FOR_SKIP:
                        _write_skip_loss_heatmap(
                            mode_map["gt_delta_upper_by_metric"][k],
                            per_scene_skip_dir,
                            num_layers,
                            fill_value,
                            metric_key=k,
                            title_suffix=f" ({sid}, {mode})",
                            fname_prefix=f"{safe_id}_{mode}_",
                            mode_suffix=f"_{mode}",
                        )

    n_scenes = len(selected)
    if n_scenes == 0:
        raise SystemExit("No scenes processed.")
    if not write_aggregate:
        print(
            f"Finished Stage2 scene processing only (no aggregate write). "
            f"Per-scene outputs are under {per_scene_similarity_dir} and {per_scene_skip_dir}."
        )
        return

    fusion_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in fusion_modes:
        mode_patch_avg: Dict[str, List[float]] = {}
        mode_special_avg: Dict[str, List[float]] = {}
        mode_gt_avg: Dict[str, List[float]] = {}
        for m in sim_metrics:
            spt = sim_sum_patch[mode][m]
            cpt = sim_cnt_patch[mode][m]
            sst = sim_sum_special[mode][m]
            cst = sim_cnt_special[mode][m]
            mode_patch_avg[m] = _finite_mean_upper(spt, cpt) if spt is not None and cpt is not None else [float("nan")] * num_pairs
            mode_special_avg[m] = _finite_mean_upper(sst, cst) if sst is not None and cst is not None else [float("nan")] * num_pairs
        for k in GT_KEYS_FOR_SKIP:
            sgt = gt_sum[mode][k]
            cgt = gt_cnt[mode][k]
            mode_gt_avg[k] = _finite_mean_upper(sgt, cgt) if sgt is not None and cgt is not None else [float("nan")] * num_pairs
        fusion_by_mode[mode] = {
            "num_scenes_used": n_scenes,
            "global_patch_similarity_upper_by_metric": mode_patch_avg,
            "global_special_similarity_upper_by_metric": mode_special_avg,
            "gt_delta_upper_by_metric": mode_gt_avg,
        }

    fill_json = None if math.isnan(fill_value) else fill_value
    report_common: Dict[str, Any] = {
        "num_scenes": n_scenes,
        "scene_ids": scene_ids,
        "aggregation": "finite_mean_upper_packed_per_scene",
        "num_layers": num_layers,
        "fusion_storage": "upper_triangle_row_major_including_diagonal",
        "fusion_max_span": fusion_max_span,
        "fusion_modes": fusion_modes,
        "fusion_similarity_metrics": sim_metrics,
        "fusion_compute_similarity": compute_similarity,
        "fusion_num_procs": fusion_num_procs,
        "analysis_device": str(analysis_device),
        "skip_gt_eval": skip_gt_eval,
        "fusion_save_per_scene": save_per_scene,
        "save_heatmap_png": bool(args.save_heatmap_png),
        "heatmap_lower_triangle_fill": fill_json if fill_json is not None else "nan",
    }
    similarity_fusion_by_mode: Dict[str, Dict[str, Any]] = {}
    skip_fusion_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in fusion_modes:
        mode_report = fusion_by_mode[mode]
        similarity_fusion_by_mode[mode] = {
            "num_scenes_used": mode_report["num_scenes_used"],
            "global_patch_similarity_upper_by_metric": mode_report["global_patch_similarity_upper_by_metric"],
            "global_special_similarity_upper_by_metric": mode_report["global_special_similarity_upper_by_metric"],
        }
        skip_fusion_by_mode[mode] = {
            "num_scenes_used": mode_report["num_scenes_used"],
            "gt_delta_upper_by_metric": mode_report["gt_delta_upper_by_metric"],
        }

    similarity_report: Dict[str, Any] = dict(report_common)
    similarity_report["report_kind"] = "similarity"
    similarity_report["fusion_by_mode"] = similarity_fusion_by_mode
    skip_report: Dict[str, Any] = dict(report_common)
    skip_report["report_kind"] = "skip"
    skip_report["fusion_by_mode"] = skip_fusion_by_mode

    n = num_layers
    if args.save_heatmap_png:
        for mode in fusion_modes:
            mode_report = fusion_by_mode[mode]
            if compute_similarity:
                _write_redundancy_heatmaps(
                    mode_report["global_patch_similarity_upper_by_metric"],
                    mode_report["global_special_similarity_upper_by_metric"],
                    similarity_out_dir,
                    n,
                    fill_value,
                    compute_pwcca,
                    similarity_mode="intra_layer_frame_vs_global",
                    title_suffix=f" (mean, {mode})",
                    fname_prefix=f"{mode}_",
                )
            for k in GT_KEYS_FOR_SKIP:
                _write_skip_loss_heatmap(
                    mode_report["gt_delta_upper_by_metric"][k],
                    skip_out_dir,
                    n,
                    fill_value,
                    metric_key=k,
                    title_suffix=f" (mean, {mode})",
                    fname_prefix="",
                    mode_suffix=f"_{mode}",
                )

    similarity_summary_rows: List[Dict[str, Any]] = []
    skip_summary_rows: List[Dict[str, Any]] = []
    for mode in fusion_modes:
        mode_report = fusion_by_mode[mode]
        for m in sim_metrics:
            similarity_summary_rows.append(
                {
                    "metric": f"fusion_global_patch_{m}_{mode}_mean_finite",
                    "value": _nanmean_finite(mode_report["global_patch_similarity_upper_by_metric"][m]),
                }
            )
            similarity_summary_rows.append(
                {
                    "metric": f"fusion_global_special_{m}_{mode}_mean_finite",
                    "value": _nanmean_finite(mode_report["global_special_similarity_upper_by_metric"][m]),
                }
            )
        for k in GT_KEYS_FOR_SKIP:
            skip_summary_rows.append(
                {
                    "metric": f"fusion_gt_delta_{k}_{mode}_mean_finite",
                    "value": _nanmean_finite(mode_report["gt_delta_upper_by_metric"][k]),
                }
            )

    score_keys = sorted(score_keys_set)
    per_scene_rows_final: List[Dict[str, Any]] = [
        {"scene_id": scene_ids[i], **{k: scene_score_dicts[i].get(k, "") for k in score_keys}}
        for i in range(n_scenes)
    ]
    for k in score_keys:
        vals = [float(r[k]) for r in per_scene_rows_final if r.get(k) not in ("", None) and str(r[k]).strip() != ""]
        if vals:
            mean_v = sum(vals) / len(vals)
            similarity_summary_rows.append({"metric": f"scene_avg_{k}", "value": mean_v})
            skip_summary_rows.append({"metric": f"scene_avg_{k}", "value": mean_v})

    with (similarity_out_dir / "summary_metrics_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for row in similarity_summary_rows:
            w.writerow(row)
    with (skip_out_dir / "summary_metrics_skip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for row in skip_summary_rows:
            w.writerow(row)

    with (similarity_out_dir / "per_scene_scores_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scene_id"] + score_keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_scene_rows_final)
    with (skip_out_dir / "per_scene_scores_skip.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scene_id"] + score_keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_scene_rows_final)

    similarity_report_path.write_text(json.dumps(similarity_report, indent=2), encoding="utf-8")
    skip_report_path.write_text(json.dumps(skip_report, indent=2), encoding="utf-8")
    if args.timing:
        print(
            f"[timing] fusion total: {time.perf_counter() - t_main_start:.2f}s",
            flush=True,
        )
    print(
        f"Wrote split fusion reports to {similarity_report_path} and {skip_report_path} "
        f"(heatmap PNGs {'enabled' if args.save_heatmap_png else 'disabled'})"
    )


if __name__ == "__main__":
    main()


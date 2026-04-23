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
    p = argparse.ArgumentParser(description="Analyze cached ScanNet activations + skip-range loss.")
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_outputs/scannet_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/scannet_redundancy_analysis"))
    p.add_argument(
        "--full-report-name",
        type=str,
        default="full_report.json",
        help="Filename for aggregated report under output-dir.",
    )
    p.add_argument(
        "--per-scene-report-suffix",
        type=str,
        default="_activation.json",
        help="Suffix for per-scene report files under output-dir/per_scene.",
    )
    p.add_argument(
        "--scene-ids-file",
        type=Path,
        default=None,
        help="Optional newline-separated scene_id list; when set, only these scenes are processed.",
    )
    p.add_argument(
        "--max-concat-tokens",
        type=int,
        default=0,
        help="Max token rows per layer per scene before pairwise metrics (0 = no subsampling).",
    )
    p.add_argument(
        "--heatmap-fill",
        type=str,
        default="nan",
        help="Value for strict lower triangle (j<i) in heatmaps; use 'nan' to mask.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-pwcca", action="store_true", help="Include PWCCA (slow on large N).")
    p.add_argument("--device", type=str, default="cuda", help="Device for pairwise activations metrics.")
    p.add_argument("--timing", action="store_true", help="Print timing details.")
    p.add_argument(
        "--pair-max-span",
        type=int,
        default=1,
        help=(
            "Similarity span cap. Default 1 = mainline intra-layer mode that measures sim(frame_i, global_i). "
            "Values >1 enable legacy inter-layer global-output similarity (mixes frame/global effects); "
            "larger spans are stored as NaN."
        ),
    )

    p.add_argument(
        "--disable-skip-range-loss",
        action="store_true",
        help="Disable range-skip loss analysis (enabled by default).",
    )
    p.add_argument(
        "--disable-similarity-analysis",
        action="store_true",
        help="Disable activation similarity analysis and reuse previously saved similarity fields from full_report.json.",
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
        help="Export/reuse per-scene visualization assets for Stage2 under <output-dir>/visualization.",
    )
    p.add_argument(
        "--save-per-scene",
        action="store_true",
        help="Save per-scene activation JSON under <output-dir>/per_scene.",
    )
    p.add_argument(
        "--save-heatmap-png",
        action="store_true",
        help="Write Stage2 heatmap PNG files alongside JSON outputs (disabled by default).",
    )
    p.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Stage2 visualization root. Default: <output-dir>/visualization",
    )
    p.add_argument(
        "--stage1-visualization-dir",
        type=Path,
        default=None,
        help="Reserved for compatibility; Stage2 now exports one visualization per skip pair.",
    )
    p.add_argument(
        "--skip-range-max-span",
        type=int,
        default=6,
        help="Max number of consecutive global blocks to skip (inclusive span j-i+1). Larger (i,j) stored as NaN.",
    )
    p.add_argument(
        "--skipped-global-modes",
        nargs="+",
        default=("identity",),
        choices=(
            "identity",
            "special_only",
            "special_only_nomlp",
            "patch_special",
            "patch_special_nomlp",
            # backward-compatible alias; normalized internally to patch_special
            "patch_special_bipartite",
        ),
        help=(
            "One or more approximations for skipped global layers during Stage2 replay. "
            "A single value preserves current single-mode behavior. Multiple values run skip-loss "
            "once per mode and suffix outputs for comparison. "
            "'special_only' keeps special->all attention and special MLP. "
            "'special_only_nomlp' removes the post-attention norm2/MLP branch for special tokens. "
            "'patch_special' keeps special->all plus patch->special attention with normal MLP on both branches. "
            "'patch_special_nomlp' removes the post-attention norm2/MLP branch on both branches. "
            "'patch_special_bipartite' is a backward-compatible alias of 'patch_special'."
        ),
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
    return model.forward_from_aggregated_tokens(
        aggregated_tokens_list,
        model.aggregator.patch_start_idx,
        images,
    )


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
    report_name = str(args.full_report_name).strip() or "full_report.json"
    per_scene_suffix = str(args.per_scene_report_suffix).strip() or "_activation.json"
    if not per_scene_suffix.endswith(".json"):
        raise SystemExit(f"per_scene_report_suffix must end with .json, got: {per_scene_suffix}")
    report_path = args.output_dir / report_name
    fill_value = _parse_heatmap_fill(args.heatmap_fill)
    analysis_device = _resolve_analysis_device(args.device)
    metric_names = _metric_names(args.enable_pwcca)
    pair_max_span = max(0, int(args.pair_max_span))
    similarity_mode = _similarity_mode_from_pair_span(pair_max_span)
    similarity_enabled = not args.disable_similarity_analysis
    skip_loss_enabled = not args.disable_skip_range_loss
    skipped_global_modes = _normalize_skipped_global_modes(args.skipped_global_modes)
    vis_root_stage2 = args.visualization_dir if args.visualization_dir is not None else (args.output_dir / "visualization")

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
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_per_scene = bool(args.save_per_scene)
    per_scene_dir: Optional[Path] = out_dir / "per_scene" if save_per_scene else None
    if per_scene_dir is not None:
        per_scene_dir.mkdir(parents=True, exist_ok=True)

    sum_patch: Dict[str, Optional[torch.Tensor]] = {k: None for k in metric_names}
    sum_special: Dict[str, Optional[torch.Tensor]] = {k: None for k in metric_names}
    cnt_patch: Dict[str, Optional[torch.Tensor]] = {k: None for k in metric_names}
    cnt_special: Dict[str, Optional[torch.Tensor]] = {k: None for k in metric_names}
    scene_ids: List[str] = []
    scene_score_dicts: List[Dict[str, Any]] = []
    score_keys_set: set[str] = set()
    num_layers: Optional[int] = None
    per_scene_reports: Dict[str, Dict[str, Any]] = {}
    t_loop_start = time.perf_counter()
    existing_report_for_similarity: Dict[str, Any] = {}
    if not similarity_enabled:
        existing_report_for_similarity = _read_json_if_exists(report_path)

    if not similarity_enabled:
        for si, item in enumerate(scenes_meta):
            sid = str(item.get("scene_id", f"scene_{si}"))
            scene_ids.append(sid)
            sc_raw = item.get("scene_scores", {})
            sc = dict(sc_raw) if isinstance(sc_raw, dict) else {}
            scene_score_dicts.append(sc)
            score_keys_set.update(sc.keys())
            if save_per_scene:
                safe_id = sid.replace("/", "_")
                assert per_scene_dir is not None
                existing_scene = _read_json_if_exists(per_scene_dir / f"{safe_id}{per_scene_suffix}")
                if existing_scene:
                    per_scene_reports[sid] = existing_scene
                else:
                    per_scene_reports[sid] = {
                        "scene_id": sid,
                        "num_layers": None,
                        "scene_scores": sc,
                    }
    else:
        for si, item in enumerate(scenes_meta):
            print(f"Processing scene {si} of {len(scenes_meta)}: {item['scene_id']}")
            path = args.cache_dir / item["file"]
            t0 = time.perf_counter()
            try:
                row = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                row = torch.load(path, map_location="cpu")
            t_load = time.perf_counter() - t0

            sid = row.get("scene_id", item.get("scene_id", path.stem))
            scene_ids.append(str(sid))
            sc = dict(row.get("scene_scores", {}))
            scene_score_dicts.append(sc)
            score_keys_set.update(sc.keys())
            if num_layers is None:
                replay_layers = row.get("aggregated_concat_per_layer")
                if isinstance(replay_layers, list) and replay_layers:
                    num_layers = len(replay_layers)

            seed = args.seed + si * 10007
            t0 = time.perf_counter()
            if similarity_mode == "intra_layer_frame_vs_global":
                patch_frame_tuple, patch_global_tuple = _split_concat_branch_tuples(
                    row,
                    patch_start_idx=int(row["patch_start_idx"]),
                    branch="patch",
                    max_tokens_per_layer=args.max_concat_tokens,
                    seed=seed,
                )
                special_frame_tuple, special_global_tuple = _split_concat_branch_tuples(
                    row,
                    patch_start_idx=int(row["patch_start_idx"]),
                    branch="special",
                    max_tokens_per_layer=args.max_concat_tokens,
                    seed=seed + 1,
                )
            else:
                # Legacy mode: inter-layer global-output similarity mixes frame/global effects
                # because adjacent global outputs are separated by frame blocks.
                _, patch_tuple = _split_concat_branch_tuples(
                    row,
                    patch_start_idx=int(row["patch_start_idx"]),
                    branch="patch",
                    max_tokens_per_layer=args.max_concat_tokens,
                    seed=seed,
                )
                _, special_tuple = _split_concat_branch_tuples(
                    row,
                    patch_start_idx=int(row["patch_start_idx"]),
                    branch="special",
                    max_tokens_per_layer=args.max_concat_tokens,
                    seed=seed + 1,
                )
            t_tuple = time.perf_counter() - t0

            if num_layers is None:
                if similarity_mode == "intra_layer_frame_vs_global":
                    num_layers = len(patch_frame_tuple)
                else:
                    num_layers = len(patch_tuple)
            elif (
                similarity_mode == "intra_layer_frame_vs_global"
                and (len(patch_frame_tuple) != num_layers or len(special_frame_tuple) != num_layers)
            ) or (
                similarity_mode != "intra_layer_frame_vs_global"
                and (len(patch_tuple) != num_layers or len(special_tuple) != num_layers)
            ):
                raise SystemExit(f"Inconsistent num layers at scene {sid}: expected {num_layers}")

            if args.timing and si == 0:
                sample_tuple = patch_frame_tuple if similarity_mode == "intra_layer_frame_vs_global" else patch_tuple
                n0, c0 = sample_tuple[0].shape[0], sample_tuple[0].shape[-1]
                npairs = num_layers if similarity_mode == "intra_layer_frame_vs_global" else num_layers * (num_layers + 1) // 2
                print(
                    f"[timing] first scene token shape per layer: N={n0}, C={c0} "
                    f"(comparisons per branch: {npairs}); max_concat_tokens={args.max_concat_tokens}; "
                    f"similarity_mode={similarity_mode}",
                    flush=True,
                )

            prof_patch: Optional[Dict[str, float]] = None
            prof_spec: Optional[Dict[str, float]] = None
            if args.timing and si == 0:
                prof_patch = {k: 0.0 for k in metric_names}
                prof_spec = {k: 0.0 for k in metric_names}

            t0 = time.perf_counter()
            if similarity_mode == "intra_layer_frame_vs_global":
                patch_sim = paired_layer_similarity_matrices(
                    patch_frame_tuple,
                    patch_global_tuple,
                    metric_seconds=prof_patch,
                    compute_pwcca=args.enable_pwcca,
                    device=analysis_device,
                )
            else:
                patch_sim = pairwise_similarity_matrices(
                    patch_tuple,
                    metric_seconds=prof_patch,
                    compute_pwcca=args.enable_pwcca,
                    device=analysis_device,
                )
            t_patch = time.perf_counter() - t0

            t0 = time.perf_counter()
            if similarity_mode == "intra_layer_frame_vs_global":
                special_sim = paired_layer_similarity_matrices(
                    special_frame_tuple,
                    special_global_tuple,
                    metric_seconds=prof_spec,
                    compute_pwcca=args.enable_pwcca,
                    device=analysis_device,
                )
            else:
                special_sim = pairwise_similarity_matrices(
                    special_tuple,
                    metric_seconds=prof_spec,
                    compute_pwcca=args.enable_pwcca,
                    device=analysis_device,
                )
            t_spec = time.perf_counter() - t0

            t0 = time.perf_counter()
            patch_upper = _pack_all(patch_sim, metric_names)
            special_upper = _pack_all(special_sim, metric_names)
            if similarity_mode != "intra_layer_frame_vs_global":
                patch_upper = {k: _apply_span_mask_upper(v, num_layers, pair_max_span) for k, v in patch_upper.items()}
                special_upper = {k: _apply_span_mask_upper(v, num_layers, pair_max_span) for k, v in special_upper.items()}
            safe_id = str(sid).replace("/", "_")
            if save_per_scene:
                per_scene_report = {
                    "scene_id": str(sid),
                    "num_layers": num_layers,
                    "similarity_storage": "upper_triangle_row_major_including_diagonal",
                    "similarity_mode": similarity_mode,
                    "enable_pwcca": bool(args.enable_pwcca),
                    "analysis_device": str(analysis_device),
                    "pair_max_span": pair_max_span,
                    "global_patch_similarity_upper": patch_upper,
                    "global_special_similarity_upper": special_upper,
                    "scene_scores": sc,
                }
                per_scene_reports[str(sid)] = per_scene_report
                if args.save_heatmap_png:
                    assert per_scene_dir is not None
                    _write_redundancy_heatmaps(
                        patch_upper,
                        special_upper,
                        per_scene_dir,
                        num_layers,
                        fill_value,
                        args.enable_pwcca,
                        similarity_mode=similarity_mode,
                        title_suffix=f" ({sid})",
                        fname_prefix=f"{safe_id}_",
                    )
            t_json = time.perf_counter() - t0

            if args.timing:
                print(
                    f"[timing] scene {sid}: load={t_load:.2f}s tuple={t_tuple:.2f}s "
                    f"pairwise_patch={t_patch:.2f}s pairwise_special={t_spec:.2f}s pack+heatmap={t_json:.2f}s",
                    flush=True,
                )
                if si == 0 and prof_patch is not None and prof_spec is not None:

                    def _fmt(d: Dict[str, float]) -> str:
                        return ", ".join(f"{k}={v:.2f}s" for k, v in sorted(d.items(), key=lambda x: -x[1]))

                    print(f"[timing] first scene patch metric totals: {_fmt(prof_patch)}", flush=True)
                    print(f"[timing] first scene special metric totals: {_fmt(prof_spec)}", flush=True)

            for name in metric_names:
                pt = torch.tensor(patch_upper[name], dtype=torch.float64)
                st = torch.tensor(special_upper[name], dtype=torch.float64)
                pt_mask = torch.isfinite(pt).to(torch.float64)
                st_mask = torch.isfinite(st).to(torch.float64)
                pt_clean = torch.nan_to_num(pt, nan=0.0)
                st_clean = torch.nan_to_num(st, nan=0.0)
                if sum_patch[name] is None:
                    sum_patch[name] = pt_clean
                    sum_special[name] = st_clean
                    cnt_patch[name] = pt_mask
                    cnt_special[name] = st_mask
                else:
                    sum_patch[name] = sum_patch[name] + pt_clean  # type: ignore[operator]
                    sum_special[name] = sum_special[name] + st_clean  # type: ignore[operator]
                    cnt_patch[name] = cnt_patch[name] + pt_mask  # type: ignore[operator]
                    cnt_special[name] = cnt_special[name] + st_mask  # type: ignore[operator]

            del row, patch_sim, special_sim

    n_scenes = len(scenes_meta)
    if n_scenes == 0:
        raise SystemExit("No scenes processed.")

    if similarity_enabled:
        patch_avg_upper = {
            k: _finite_mean_upper(sum_patch[k], cnt_patch[k])  # type: ignore[arg-type]
            for k in metric_names
            if sum_patch[k] is not None and cnt_patch[k] is not None
        }
        special_avg_upper = {
            k: _finite_mean_upper(sum_special[k], cnt_special[k])  # type: ignore[arg-type]
            for k in metric_names
            if sum_special[k] is not None and cnt_special[k] is not None
        }
    else:
        patch_loaded = existing_report_for_similarity.get("global_patch_similarity_upper")
        special_loaded = existing_report_for_similarity.get("global_special_similarity_upper")
        patch_avg_upper = patch_loaded if isinstance(patch_loaded, dict) else {}
        special_avg_upper = special_loaded if isinstance(special_loaded, dict) else {}
        existing_layers = int(existing_report_for_similarity.get("num_layers", 0))
        if num_layers is None or num_layers <= 0:
            num_layers = existing_layers
        if existing_layers > 0 and num_layers > 0 and num_layers != existing_layers:
            raise SystemExit(
                f"Similarity-disabled mode mismatch: cache num_layers={num_layers} but existing full_report has {existing_layers}."
            )
        similarity_mode = str(existing_report_for_similarity.get("similarity_mode", similarity_mode))
        pair_max_span = int(existing_report_for_similarity.get("pair_max_span", pair_max_span))

    if (num_layers is None or num_layers <= 0) and skip_loss_enabled:
        num_layers = _infer_num_layers_from_manifest(args.cache_dir, scenes_meta)
    assert num_layers is not None
    if save_per_scene and not similarity_enabled:
        for sid in scene_ids:
            if sid in per_scene_reports and per_scene_reports[sid].get("num_layers") is None:
                per_scene_reports[sid]["num_layers"] = num_layers

    fill_json = None if math.isnan(fill_value) else fill_value
    report: Dict[str, Any] = {
        "num_scenes": n_scenes,
        "scene_ids": scene_ids,
        "aggregation": "finite_mean_upper_packed_per_scene",
        "num_layers": num_layers,
        "similarity_storage": "upper_triangle_row_major_including_diagonal",
        "similarity_mode": similarity_mode,
        "similarity_mode_note": (
            "mainline: diagonal-only intra-layer sim(frame_i, global_i)"
            if similarity_mode == "intra_layer_frame_vs_global"
            else "legacy: inter-layer global-output similarity mixes frame/global effects"
        ),
        "enable_pwcca": bool(args.enable_pwcca),
        "analysis_device": str(analysis_device),
        "pair_max_span": pair_max_span,
        "save_per_scene": save_per_scene,
        "save_heatmap_png": bool(args.save_heatmap_png),
        "heatmap_lower_triangle_fill": fill_json if fill_json is not None else "nan",
        "max_tokens_per_layer_per_scene": args.max_concat_tokens,
        "skipped_global_modes": skipped_global_modes,
        "global_patch_similarity_upper": patch_avg_upper,
        "global_special_similarity_upper": special_avg_upper,
    }

    skip_loss_report: Optional[Dict[str, Any]] = None
    merged_skip_loss_by_mode: Dict[str, Dict[str, Any]] = {}
    current_skip_loss_by_mode: Dict[str, Dict[str, Any]] = {}
    merged_skip_modes_order: List[str] = []
    per_scene_written_during_skip_stream = False
    if skip_loss_enabled:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        skip_loss_dtype = dtype_map[args.skip_loss_dtype]
        skip_loss_device = torch.device(
            args.skip_loss_device if torch.cuda.is_available() or args.skip_loss_device == "cpu" else "cpu"
        )
        existing_report = _read_json_if_exists(report_path)
        existing_skip_loss_by_mode = _extract_skip_loss_by_mode(existing_report)
        existing_per_scene_skip_by_mode = _load_existing_per_scene_skip_by_mode(
            per_scene_dir, scene_ids, per_scene_report_suffix=per_scene_suffix
        )
        multi_mode = len(skipped_global_modes) > 1
        t0_skip_all = time.perf_counter()
        if not similarity_enabled:
            stream_per_scene_dir = per_scene_dir if per_scene_dir is not None else (out_dir / "_tmp_skip_stream")
            stream_per_scene_dir.mkdir(parents=True, exist_ok=True)
            discovered = discover_scannet_scenes(args.data_root)
            scene_map: Dict[str, SceneData] = {s.scene_id: s for s in discovered}
            eval_path = args.cache_dir / "eval_scannet.json"
            if not eval_path.exists():
                raise SystemExit(
                    f"Missing {eval_path}; Stage3 skip-range GT comparison requires Stage1 eval results."
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
                        f"Stage3 skip-loss requires complete data, but scene_id {sid} was not found under {args.data_root}."
                    )
                if sid not in stage1_eval_map:
                    raise SystemExit(
                        f"Stage3 skip-loss requires complete Stage1 eval rows, but {sid} is missing in "
                        f"{args.cache_dir / 'eval_scannet.json'}."
                    )
                selected.append((item, scene_map[sid], stage1_eval_map[sid]))
            if not selected:
                raise SystemExit("Skip-range loss enabled but no valid scene with Stage1 predictions was found.")

            model = VGGT.from_pretrained(args.hf_model).to(skip_loss_device).eval()
            use_amp = skip_loss_device.type == "cuda"
            autocast_device_type = "cuda" if skip_loss_device.type == "cuda" else "cpu"
            mode_timing_totals: Dict[str, Dict[str, float]] = {
                m: {
                    "replay_prep": 0.0,
                    "replay_forward": 0.0,
                    "to_cpu_payload": 0.0,
                    "export_vis": 0.0,
                    "gt_eval": 0.0,
                    "postprocess": 0.0,
                    "pair_total": 0.0,
                }
                for m in skipped_global_modes
            }
            mode_timing_pairs: Dict[str, int] = {m: 0 for m in skipped_global_modes}

            for si, (item, scene, stage1_eval_row) in enumerate(selected):
                t0_scene = time.perf_counter()
                print(
                    f"[stage3] scene {si + 1}/{len(selected)}: {scene.scene_id}",
                    flush=True,
                )
                scene_eff = select_scene_image_paths_fastvggt(scene, args.max_frames)
                if not scene_eff.image_paths:
                    raise SystemExit(
                        f"Stage3 skip-loss requires complete data, but scene {scene.scene_id} has no valid frames "
                        "after FastVGGT selection."
                    )
                if bool(args.skip_gt_eval):
                    raise SystemExit("GT drop loss requires GT eval. Please set SKIP_GT_EVAL=0 for Stage3.")
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
                replay_concat_layers = _load_scene_replay_cache(args.cache_dir, item, num_layers)
                pos_frame, pos_global = _build_aggregator_positions(model, images)
                if not bool(args.disable_stage1_sanity_check):
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
                            skipped_global_mode=skipped_global_modes[0],
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

                total_pairs = 0
                skip_max_span = max(1, int(args.skip_range_max_span))
                for ii in range(num_layers):
                    total_pairs += max(0, min(num_layers - 1, ii + skip_max_span - 1) - ii + 1)
                print(
                    f"[stage3] scene {si + 1}/{len(selected)} {scene.scene_id}: "
                    f"{len(skipped_global_modes)} modes, {total_pairs} valid pairs/mode",
                    flush=True,
                )

                scene_mode_rows: Dict[str, Dict[str, List[float]]] = {}
                for mi, skipped_global_mode in enumerate(skipped_global_modes):
                    print(
                        f"[stage3] scene {si + 1}/{len(selected)} {scene.scene_id} | "
                        f"mode {mi + 1}/{len(skipped_global_modes)}: {skipped_global_mode}",
                        flush=True,
                    )
                    done_pairs = 0
                    scene_loss_upper_by_metric = {k: [] for k in GT_KEYS_FOR_SKIP}
                    for i in range(num_layers):
                        for j in range(i, num_layers):
                            if j - i + 1 > skip_max_span:
                                for k in GT_KEYS_FOR_SKIP:
                                    scene_loss_upper_by_metric[k].append(float("nan"))
                                continue
                            t_pair0 = time.perf_counter()
                            t0 = time.perf_counter()
                            with torch.no_grad(), torch.autocast(
                                device_type=autocast_device_type, dtype=skip_loss_dtype, enabled=use_amp
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
                            if bool(args.export_visualization) and vis_root_stage2 is not None:
                                tv0 = time.perf_counter()
                                _export_stage2_skip_pair_visualization(
                                    stage2_vis_root=vis_root_stage2,
                                    scene_eff=scene_eff,
                                    pair_name=f"skip_l{i:02d}_l{j:02d}_{skipped_global_mode}" if multi_mode else f"skip_l{i:02d}_l{j:02d}",
                                    images=images,
                                    prediction_payload=payload,
                                    depth_conf_thresh=args.depth_conf_thresh,
                                )
                                t_vis = time.perf_counter() - tv0
                            t0 = time.perf_counter()
                            pruned_eval_row = build_scannet_eval_row_from_payload_with_context(
                                payload,
                                scene_eff,
                                eval_ctx,
                                depth_conf_thresh=args.depth_conf_thresh,
                                chamfer_max_dist=args.chamfer_max_dist,
                                inference_time_ms=0.0,
                            )
                            t_eval = time.perf_counter() - t0
                            t0 = time.perf_counter()
                            pruned_gt = metrics_float_dict(pruned_eval_row)
                            per_key = _aligned_change_per_key(stage1_gt, pruned_gt)
                            for k in GT_KEYS_FOR_SKIP:
                                scene_loss_upper_by_metric[k].append(float(per_key[k]) if k in per_key else float("nan"))
                            t_post = time.perf_counter() - t0
                            del pred, pred_cpu
                            done_pairs += 1
                            pair_total = time.perf_counter() - t_pair0
                            mode_timing_totals[skipped_global_mode]["replay_forward"] += t_forward
                            mode_timing_totals[skipped_global_mode]["to_cpu_payload"] += t_payload
                            mode_timing_totals[skipped_global_mode]["export_vis"] += t_vis
                            mode_timing_totals[skipped_global_mode]["gt_eval"] += t_eval
                            mode_timing_totals[skipped_global_mode]["postprocess"] += t_post
                            mode_timing_totals[skipped_global_mode]["pair_total"] += pair_total
                            mode_timing_pairs[skipped_global_mode] += 1
                            if args.timing:
                                span = j - i + 1
                                print(
                                    f"[timing] skip-loss scene {si + 1}/{len(selected)} {scene.scene_id} | "
                                    f"mode {mi + 1}/{len(skipped_global_modes)} {skipped_global_mode} | "
                                    f"pair {done_pairs}/{total_pairs} (i={i}, j={j}, span={span}): "
                                    f"total={pair_total:.2f}s",
                                    flush=True,
                                )
                    scene_mode_rows[skipped_global_mode] = scene_loss_upper_by_metric

                safe_id = scene.scene_id.replace("/", "_")
                scene_scores_raw = item.get("scene_scores", {})
                scene_scores = dict(scene_scores_raw) if isinstance(scene_scores_raw, dict) else {}
                scene_path = stream_per_scene_dir / f"{safe_id}{per_scene_suffix}"
                existing_scene_row = _read_json_if_exists(scene_path)
                scene_row = dict(existing_scene_row) if existing_scene_row else {"scene_id": scene.scene_id}
                scene_row["num_layers"] = num_layers
                scene_row["scene_scores"] = scene_scores
                merged_scene_modes = _extract_scene_skip_by_mode(scene_row)
                merged_scene_modes.update(scene_mode_rows)
                if len(merged_scene_modes) > 1:
                    scene_row["global_skip_range_loss_upper_by_metric_by_mode"] = merged_scene_modes
                    scene_row.pop("global_skip_range_loss_upper_by_metric", None)
                    scene_row.pop("skipped_global_mode", None)
                else:
                    only_mode = next(iter(merged_scene_modes))
                    scene_row["skipped_global_mode"] = only_mode
                    scene_row["global_skip_range_loss_upper_by_metric"] = merged_scene_modes[only_mode]
                    scene_row.pop("global_skip_range_loss_upper_by_metric_by_mode", None)
                scene_path.write_text(json.dumps(scene_row, indent=2), encoding="utf-8")
                print(
                    f"[stage3] wrote per-scene result: {scene_path}",
                    flush=True,
                )
                if save_per_scene:
                    per_scene_reports[scene.scene_id] = scene_row

                if args.save_heatmap_png:
                    for mode, mode_map in scene_mode_rows.items():
                        for k in GT_KEYS_FOR_SKIP:
                            _write_skip_loss_heatmap(
                                mode_map[k],
                                stream_per_scene_dir,
                                num_layers,
                                fill_value,
                                metric_key=k,
                                title_suffix=f" ({scene.scene_id}, {mode})" if multi_mode else f" ({scene.scene_id})",
                                fname_prefix=f"{safe_id}_",
                                mode_suffix=f"_{mode}" if multi_mode else "",
                            )

                if args.timing:
                    print(
                        f"[timing] skip-loss scene {si + 1}/{len(selected)} {scene.scene_id}: "
                        f"{time.perf_counter() - t0_scene:.2f}s",
                        flush=True,
                    )
            per_scene_written_during_skip_stream = True

            aggregated_modes = _aggregate_skip_modes_from_per_scene_files(
                stream_per_scene_dir,
                [str(item["scene_id"]) for item in scenes_meta],
                per_scene_report_suffix=per_scene_suffix,
            )
            print(
                f"[stage3] aggregated per-scene results from {len(selected)} scenes "
                f"under {stream_per_scene_dir}",
                flush=True,
            )
            for mode in skipped_global_modes:
                agg = aggregated_modes.get(mode)
                if agg is None:
                    continue
                timing_totals = mode_timing_totals.get(mode, {})
                timing_pairs = int(mode_timing_pairs.get(mode, 0))
                current_skip_loss_by_mode[mode] = {
                    "enabled": True,
                    "metric": "neg_direction_aligned_change_vs_stage1_per_metric",
                    "gt_keys_used": list(GT_KEYS_FOR_SKIP),
                    "skip_max_span": int(max(1, int(args.skip_range_max_span))),
                    "skipped_global_mode": mode,
                    "num_scenes_used": int(agg.get("num_scenes_used", 0)),
                    "stage1_sanity_check": not bool(args.disable_stage1_sanity_check),
                    "stage1_sanity_atol": float(args.stage1_sanity_atol),
                    "stage1_sanity_rtol": float(args.stage1_sanity_rtol),
                    "skip_range_loss_upper_by_metric": agg["metric_map"],
                    "timing_breakdown_total_seconds": timing_totals if timing_pairs > 0 else {},
                    "timing_breakdown_pairs": timing_pairs,
                }
            if (not save_per_scene) and stream_per_scene_dir.name == "_tmp_skip_stream":
                for p in stream_per_scene_dir.glob(f"*{per_scene_suffix}"):
                    p.unlink(missing_ok=True)
                for p in stream_per_scene_dir.glob("*.png"):
                    p.unlink(missing_ok=True)
                stream_per_scene_dir.rmdir()
        else:
            skip_loss_reports_by_mode: Dict[str, Dict[str, Any]] = {}
            skip_rows_by_mode: Dict[str, List[Dict[str, Any]]] = {}
            for skipped_global_mode in skipped_global_modes:
                t0_skip = time.perf_counter()
                mode_report, mode_rows = _collect_skip_range_loss(
                    scenes_meta=scenes_meta,
                    cache_dir=args.cache_dir,
                    data_root=args.data_root,
                    hf_model=args.hf_model,
                    max_frames=args.max_frames,
                    device=skip_loss_device,
                    amp_dtype=skip_loss_dtype,
                    num_layers=num_layers,
                    timing=args.timing,
                    output_dir=out_dir,
                    gt_ply_dir=args.gt_ply_dir,
                    depth_conf_thresh=args.depth_conf_thresh,
                    chamfer_max_dist=args.chamfer_max_dist,
                    skip_gt_eval=bool(args.skip_gt_eval),
                    skip_max_span=max(1, int(args.skip_range_max_span)),
                    export_visualization=bool(args.export_visualization),
                    stage2_vis_root=vis_root_stage2 if args.export_visualization else None,
                    skipped_global_mode=skipped_global_mode,
                    vis_pair_suffix=f"_{skipped_global_mode}" if multi_mode else "",
                    stage1_sanity_check=not bool(args.disable_stage1_sanity_check),
                    stage1_sanity_atol=float(args.stage1_sanity_atol),
                    stage1_sanity_rtol=float(args.stage1_sanity_rtol),
                )
                skip_loss_reports_by_mode[skipped_global_mode] = mode_report
                skip_rows_by_mode[skipped_global_mode] = mode_rows
                if args.timing:
                    print(
                        f"[timing] skip-range loss total ({skipped_global_mode}): {time.perf_counter() - t0_skip:.2f}s",
                        flush=True,
                    )
            current_skip_loss_by_mode = dict(skip_loss_reports_by_mode)
            if save_per_scene:
                current_scene_skip_by_mode: Dict[str, Dict[str, Any]] = {sid: {} for sid in scene_ids}
                for mode, rows in skip_rows_by_mode.items():
                    for row in rows:
                        sid = str(row["scene_id"])
                        current_scene_skip_by_mode.setdefault(sid, {})
                        current_scene_skip_by_mode[sid][mode] = row["skip_range_loss_upper_by_metric"]

                for sid in scene_ids:
                    if sid not in per_scene_reports:
                        continue
                    merged_scene_modes = dict(existing_per_scene_skip_by_mode.get(sid, {}))
                    merged_scene_modes.update(current_scene_skip_by_mode.get(sid, {}))
                    if len(skipped_global_modes) > 1:
                        if merged_scene_modes:
                            per_scene_reports[sid]["global_skip_range_loss_upper_by_metric_by_mode"] = merged_scene_modes
                        per_scene_reports[sid].pop("global_skip_range_loss_upper_by_metric", None)
                    else:
                        only_mode = next(iter(merged_scene_modes))
                        only_map = merged_scene_modes.get(only_mode)
                        if only_map is not None:
                            per_scene_reports[sid]["skipped_global_mode"] = only_mode
                            per_scene_reports[sid]["global_skip_range_loss_upper_by_metric"] = only_map
                        per_scene_reports[sid].pop("global_skip_range_loss_upper_by_metric_by_mode", None)

        merged_skip_loss_by_mode = dict(existing_skip_loss_by_mode)
        merged_skip_loss_by_mode.update(current_skip_loss_by_mode)
        merged_skip_modes_order = list(merged_skip_loss_by_mode.keys())
        report["skipped_global_modes"] = merged_skip_modes_order

        if len(merged_skip_loss_by_mode) > 1:
            report["global_skip_range_loss_by_mode"] = merged_skip_loss_by_mode
            skip_loss_report = None
        else:
            only_mode = next(iter(merged_skip_loss_by_mode))
            skip_loss_report = merged_skip_loss_by_mode[only_mode]
            report["global_skip_range_loss"] = skip_loss_report

        if args.timing and multi_mode:
            print(f"[timing] skip-range loss total (all modes): {time.perf_counter() - t0_skip_all:.2f}s", flush=True)

    n = num_layers
    t_post = time.perf_counter()
    if args.timing:
        print(f"[timing] all scenes loop total: {t_post - t_loop_start:.2f}s", flush=True)

    if args.save_heatmap_png and similarity_enabled:
        _write_redundancy_heatmaps(
            patch_avg_upper,
            special_avg_upper,
            out_dir,
            n,
            fill_value,
            args.enable_pwcca,
            similarity_mode=similarity_mode,
            title_suffix=" (mean)",
            fname_prefix="",
        )
    if args.save_heatmap_png and skip_loss_enabled:
        use_mode_suffix = len(merged_skip_loss_by_mode) > 1
        if not use_mode_suffix and skip_loss_report is not None:
            for k in skip_loss_report["gt_keys_used"]:
                _write_skip_loss_heatmap(
                    skip_loss_report["skip_range_loss_upper_by_metric"][k],
                    out_dir,
                    n,
                    fill_value,
                    metric_key=k,
                    title_suffix=" (mean)",
                    fname_prefix="",
                )
        else:
            for mode in skipped_global_modes:
                mode_report = current_skip_loss_by_mode[mode]
                for k in mode_report["gt_keys_used"]:
                    _write_skip_loss_heatmap(
                        mode_report["skip_range_loss_upper_by_metric"][k],
                        out_dir,
                        n,
                        fill_value,
                        metric_key=k,
                        title_suffix=f" (mean, {mode})",
                        fname_prefix="",
                        mode_suffix=f"_{mode}" if use_mode_suffix else "",
                    )

    def _mean_off(packed: List[float]) -> float:
        vals: List[float] = []
        idx = 0
        for i in range(n):
            for j in range(i, n):
                v = float(packed[idx])
                idx += 1
                if i == j or math.isnan(v):
                    continue
                vals.append(v)
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    summary_rows: List[Dict[str, Any]] = []
    for branch, avg in (
        ("global_patch", patch_avg_upper),
        ("global_special", special_avg_upper),
    ):
        if not isinstance(avg, dict):
            continue
        if not {"cka", "cosine", "neg_mse"}.issubset(set(avg.keys())):
            continue
        if similarity_mode == "intra_layer_frame_vs_global":
            summary_rows.append({"metric": f"{branch}_cka_mean_diag", "value": _mean_diag(avg["cka"], n)})
            summary_rows.append({"metric": f"{branch}_cosine_mean_diag", "value": _mean_diag(avg["cosine"], n)})
            if args.enable_pwcca:
                summary_rows.append({"metric": f"{branch}_pwcca_mean_diag", "value": _mean_diag(avg["pwcca"], n)})
            summary_rows.append({"metric": f"{branch}_neg_mse_mean_diag", "value": _mean_diag(avg["neg_mse"], n)})
        else:
            summary_rows.append({"metric": f"{branch}_cka_mean_offdiag_legacy", "value": _mean_off(avg["cka"])})
            summary_rows.append({"metric": f"{branch}_cosine_mean_offdiag_legacy", "value": _mean_off(avg["cosine"])})
            if args.enable_pwcca:
                summary_rows.append({"metric": f"{branch}_pwcca_mean_offdiag_legacy", "value": _mean_off(avg["pwcca"])})
            summary_rows.append({"metric": f"{branch}_neg_mse_mean_offdiag_legacy", "value": _mean_off(avg["neg_mse"])})
    if skip_loss_enabled:
        if len(merged_skip_loss_by_mode) <= 1 and skip_loss_report is not None:
            for k in skip_loss_report["gt_keys_used"]:
                summary_rows.append(
                    {
                        "metric": f"global_skip_range_loss_{k}_mean_finite",
                        "value": _nanmean_finite(skip_loss_report["skip_range_loss_upper_by_metric"][k]),
                    }
                )
        else:
            for mode in merged_skip_modes_order:
                mode_report = merged_skip_loss_by_mode[mode]
                for k in mode_report["gt_keys_used"]:
                    summary_rows.append(
                        {
                            "metric": f"global_skip_range_loss_{k}_{mode}_mean_finite",
                            "value": _nanmean_finite(mode_report["skip_range_loss_upper_by_metric"][k]),
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
            summary_rows.append({"metric": f"scene_avg_{k}", "value": sum(vals) / len(vals)})

    with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    with (out_dir / "per_scene_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["scene_id"] + score_keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_scene_rows_final)

    if save_per_scene and not per_scene_written_during_skip_stream:
        assert per_scene_dir is not None
        for sid in scene_ids:
            safe_id = sid.replace("/", "_")
            (per_scene_dir / f"{safe_id}{per_scene_suffix}").write_text(
                json.dumps(per_scene_reports[sid], indent=2), encoding="utf-8"
            )
            if args.save_heatmap_png:
                if "global_skip_range_loss_upper_by_metric" in per_scene_reports[sid]:
                    for k in GT_KEYS_FOR_SKIP:
                        _write_skip_loss_heatmap(
                            per_scene_reports[sid]["global_skip_range_loss_upper_by_metric"][k],
                            per_scene_dir,
                            n,
                            fill_value,
                            metric_key=k,
                            title_suffix=f" ({sid})",
                            fname_prefix=f"{safe_id}_",
                        )
                if "global_skip_range_loss_upper_by_metric_by_mode" in per_scene_reports[sid]:
                    for mode in skipped_global_modes:
                        mode_map = per_scene_reports[sid]["global_skip_range_loss_upper_by_metric_by_mode"].get(mode)
                        if not isinstance(mode_map, dict):
                            continue
                        for k in GT_KEYS_FOR_SKIP:
                            _write_skip_loss_heatmap(
                                mode_map[k],
                                per_scene_dir,
                                n,
                                fill_value,
                                metric_key=k,
                                title_suffix=f" ({sid}, {mode})",
                                fname_prefix=f"{safe_id}_",
                                mode_suffix=f"_{mode}",
                            )

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.timing:
        print(f"[timing] post-loop (heatmaps+csv+full_report): {time.perf_counter() - t_post:.2f}s", flush=True)
        print(
            f"[timing] stage2 total (incl. skip-range): {time.perf_counter() - t_main_start:.2f}s "
            f"(scene loop only: {t_post - t_loop_start:.2f}s)",
            flush=True,
        )
    print(
        f"Wrote reports to {report_path} (heatmap PNGs {'enabled' if args.save_heatmap_png else 'disabled'})"
    )


if __name__ == "__main__":
    main()


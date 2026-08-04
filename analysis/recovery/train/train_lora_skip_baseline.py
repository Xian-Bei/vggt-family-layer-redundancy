from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

# Keep OpenCV/Matplotlib in headless mode for eval_utils imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
# evo.tools.plot has its own settings.plot_backend that is applied at import time
# (`apply_settings(SETTINGS)`) and ignores MPLBACKEND. Without this override, on
# systems with a hostile/empty $DISPLAY the import would fail with:
#   "Cannot load backend 'Qt5Agg' ... as 'headless' is currently running".
try:
    from evo.tools.settings import SETTINGS as _EVO_SETTINGS  # type: ignore

    _EVO_SETTINGS.plot_backend = "Agg"
except Exception:
    pass

from analysis.eval.eval_utils import (
    get_vgg_input_imgs,
    load_images_rgb,
    load_poses,
)
from analysis.backbones.cache_replay import get_cache_replay_engine
from analysis.backbones.registry import get_backbone
from analysis.backbones.cache_replay.common import (
    align_activation_term,
    infer_num_layers,
    teacher_pred_to_map,
)
from analysis.prune_plan import (
    BUDGET_PLAN_BUNDLE_SCHEMA_VERSION,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    UnitPrunePlan,
    drop_intervals_from_plan,
    load_named_prune_plan,
    lora_layers_from_plan,
    plan_group_from_name,
    prune_actions_from_plan,
    replay_seed_spec,
    unit_index_to_kind_layer,
    unit_index_to_unit_id,
)
from analysis.prune_plan.recovery_plan import compose_recovery_unit_prune_plan
from analysis.schema.v3 import dataset_output_tag, parse_dataset_plugin_args
from analysis.backbones.cache_replay.vggt_family import mask_from_depth
from analysis.recovery.train.model import build_skipped_layer_indices
from analysis.recovery.train.checkpoint_v3 import (
    atomic_publish_latest,
    atomic_save_checkpoint,
)
from analysis.redundancy.cache_layout import (
    _replay_layer_files,
    load_scene_metadata,
    load_scene_pred,
    load_scene_replay_layer,
    load_scene_unit,
)
from analysis.eval.prediction_eval import _load_gt_depth_for_scene
from analysis.eval.scannet_eval import (
    SceneData,
    discover_scannet_scenes,
    select_scene_image_paths_fastvggt,
)


def _load_unit_prune_plan(
    path: Path,
    plan_name: str = "",
    recovery_plan_root: Optional[Path] = None,
    recovery_plan_name: str = "",
) -> UnitPrunePlan:
    if not path.exists():
        raise SystemExit(f"Missing --prune-plan-file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = None
    if isinstance(raw, dict) and str(raw.get("schema_version", "")) == BUDGET_PLAN_BUNDLE_SCHEMA_VERSION:
        selector = str(plan_name or "").strip()
        if not selector:
            raise SystemExit("--prune-plan-name is required when --prune-plan-file is a legacy budget bundle")
        return load_named_prune_plan(path.parent, selector)
    if isinstance(raw, dict) and str(raw.get("schema_version", "")) == PRUNE_PLAN_BUNDLE_SCHEMA_VERSION:
        selector = str(plan_name or "").strip()
        recovery_name = str(recovery_plan_name or "").strip()
        if not selector or not recovery_name:
            raise SystemExit(
                "--prune-plan-name and --recovery-plan-name are required when "
                "--prune-plan-file is prune_plan_all.json"
            )
        return compose_recovery_unit_prune_plan(
            prune_plan_root=path.parent,
            recovery_plan_root=recovery_plan_root or path.parent,
            prune_selector=selector,
            recovery_plan=recovery_name,
        )
    return UnitPrunePlan.load(path)


def _training_anchor_unit_indices(plan: UnitPrunePlan) -> List[int]:
    if plan.lora:
        return sorted(int(k) for k in plan.lora.keys())
    raw = plan.extra.get("train_unit_indices") if isinstance(plan.extra, dict) else None
    if isinstance(raw, list):
        return sorted({int(x) for x in raw})
    return []


def _replay_seed_spec_from_unit_indices(indices: Sequence[int]) -> Tuple[int, str, int]:
    if not indices:
        raise ValueError("training plan has no train/lora unit indices; cannot derive replay seed")
    min_u = min(int(k) for k in indices)
    replay_start_layer = int(min_u) // 2
    if min_u <= 0:
        return 0, "none", int(replay_start_layer)
    seed_u = int(min_u) - 1
    seed_kind, seed_layer = unit_index_to_kind_layer(seed_u)
    return int(seed_layer), str(seed_kind), int(replay_start_layer)


def _enable_original_train_units(model: nn.Module, backbone: str, unit_indices: Sequence[int]) -> List[nn.Parameter]:
    adapter = get_backbone(backbone)
    params: List[nn.Parameter] = []
    seen: set[int] = set()
    for unit_index in sorted({int(x) for x in unit_indices}):
        block = adapter.block_for_unit(model, unit_index_to_unit_id(unit_index))
        for param in block.parameters():
            param.requires_grad_(True)
            ident = id(param)
            if ident not in seen:
                seen.add(ident)
                params.append(param)
    return params


def _infer_cache_dataset_tag(cache_dir: Path) -> str:
    p = Path(cache_dir)
    if p.name == "cache" and p.parent.name:
        return p.parent.name
    return ""


def _load_manifest(cache_dir: Path) -> List[Dict[str, Any]]:
    from analysis.redundancy.cache_layout_v3 import is_v3_cache_dir, load_v3_manifest, manifest_scenes_for_training

    if is_v3_cache_dir(cache_dir):
        manifest = load_v3_manifest(cache_dir)
        scenes = manifest_scenes_for_training(manifest)
        if not scenes:
            raise SystemExit(f"v3 manifest has no scenes: {cache_dir / 'cache_manifest_v3.json'}")
        dataset = str(manifest.dataset or "").strip() or _infer_cache_dataset_tag(cache_dir)
        for item in scenes:
            item.setdefault("dataset", dataset)
        return scenes
    p = cache_dir / "manifest.json"
    if not p.exists():
        raise SystemExit(f"Missing cache manifest: {p} (or cache_manifest_v3.json for v3)")
    payload = json.loads(p.read_text(encoding="utf-8"))
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise SystemExit(f"Manifest has no scenes: {p}")
    dataset = str(payload.get("dataset", "")).strip() or _infer_cache_dataset_tag(cache_dir)
    for item in scenes:
        if isinstance(item, dict):
            item.setdefault("dataset", dataset)
    return scenes


def _load_scene_cache_row(cache_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(load_scene_metadata(cache_dir, item))
    row["pred"] = load_scene_pred(cache_dir, item)
    return row


def _resolve_cache_dirs(args: argparse.Namespace) -> List[Path]:
    raw = getattr(args, "cache_dir", None)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [Path(p) for p in raw]
    return [Path(raw)]


def _load_training_scenes(cache_dirs: Sequence[Path], max_scenes: int) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    for cache_dir in cache_dirs:
        dataset_scenes = _load_manifest(Path(cache_dir))
        if int(max_scenes) > 0:
            dataset_scenes = dataset_scenes[: int(max_scenes)]
        for item in dataset_scenes:
            if not isinstance(item, dict):
                continue
            scene_item = dict(item)
            scene_item["_cache_dir"] = str(cache_dir)
            scenes.append(scene_item)
    return scenes


def _load_scannet_intrinsics(scene_dir: Path) -> np.ndarray:
    intrinsic_dir = scene_dir / "intrinsic"
    candidates = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
        scene_dir / "intrinsic.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        vals: List[List[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals.append([float(x) for x in line.split()])
        arr = np.array(vals, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr[:3, :3].copy()
        if arr.shape == (3, 3):
            return arr.copy()
    raise FileNotFoundError(f"Cannot find ScanNet intrinsics under {scene_dir}")


def _resize_intrinsics_like_vgg_input(
    k: np.ndarray,
    *,
    rgb_width: int,
    rgb_height: int,
    new_width: int = 518,
) -> np.ndarray:
    k = k.astype(np.float32, copy=True)
    scale = float(new_width) / float(rgb_width)
    new_height = round(rgb_height * scale / 14) * 14
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] *= scale
    k[1, 2] *= scale
    if new_height > 518:
        start_y = (new_height - 518) // 2
        k[1, 2] -= float(start_y)
    return k


def _select_scannet_scene_with_frames(
    scene: SceneData,
    *,
    max_frames: int,
) -> SceneData:
    selected = select_scene_image_paths_fastvggt(scene, max_frames=max_frames)
    return selected


def _build_target_batch_from_dataset(
    *,
    scene: SceneData,
    expected_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    selected = _select_scannet_scene_with_frames(scene, max_frames=expected_seq_len)
    if len(selected.image_paths) != int(expected_seq_len):
        raise RuntimeError(
            f"Scene {scene.scene_id} selected {len(selected.image_paths)} frames, expected {expected_seq_len}"
        )

    images_np = load_images_rgb(selected.image_paths)
    vgg_input, _pw, _ph = get_vgg_input_imgs(np.array(images_np))
    image_hw = (int(vgg_input.shape[-2]), int(vgg_input.shape[-1]))
    gt_depth_np = _load_gt_depth_for_scene(selected, image_hw=image_hw)
    if gt_depth_np is None:
        raise RuntimeError(f"Failed to load GT depth for scene {scene.scene_id}")

    pose_dir = selected.scene_dir / "pose"
    if not pose_dir.is_dir():
        raise RuntimeError(f"Missing pose dir for scene {scene.scene_id}: {pose_dir}")
    c2ws_all, _first_gt_pose, available_frame_ids = load_poses(pose_dir)
    pose_frame_to_idx = {int(fid): idx for idx, fid in enumerate(available_frame_ids.tolist())}
    frame_ids = [int(p.stem) for p in selected.image_paths]
    indices = [pose_frame_to_idx[int(fid)] for fid in frame_ids]
    c2ws = c2ws_all[np.array(indices, dtype=np.int64)]
    w2cs = np.linalg.inv(c2ws)[:, :3, :4].astype(np.float32, copy=False)

    k_src = _load_scannet_intrinsics(selected.scene_dir)
    k_list: List[np.ndarray] = []
    for p in selected.image_paths:
        with Image.open(p) as im:
            rgb_w, rgb_h = im.size
        k_list.append(
            _resize_intrinsics_like_vgg_input(
                k_src,
                rgb_width=int(rgb_w),
                rgb_height=int(rgb_h),
                new_width=int(image_hw[1]),
            )
        )
    intrinsics = np.stack(k_list, axis=0).astype(np.float32, copy=False)

    depths = torch.from_numpy(gt_depth_np).float().unsqueeze(0).to(device)
    images = vgg_input.float().unsqueeze(0).to(device)
    point_masks = mask_from_depth(depths)
    return {
        "images": images,
        "extrinsics": torch.from_numpy(w2cs).float().unsqueeze(0).to(device),
        "intrinsics": torch.from_numpy(intrinsics).float().unsqueeze(0).to(device),
        "depths": depths,
        "point_masks": point_masks.bool(),
    }


def _compute_total_steps(num_scenes: int, epochs: int, max_steps: int) -> int:
    natural = max(0, int(num_scenes) * int(epochs))
    if max_steps > 0:
        return min(natural, int(max_steps))
    return natural


def _format_param_count(n: int) -> str:
    v = float(n)
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return str(int(v))


def _write_curves(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "train_curves.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    numeric_keys = sorted(
        {
            str(k)
            for r in rows
            for k, v in r.items()
            if k != "epoch" and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    )
    epoch_buckets: Dict[int, Dict[str, float]] = {}
    epoch_counts: Dict[int, int] = {}
    for r in rows:
        ep = int(r.get("epoch", 0))
        if ep not in epoch_buckets:
            epoch_buckets[ep] = {k: 0.0 for k in numeric_keys}
            epoch_counts[ep] = 0
        for k in numeric_keys:
            epoch_buckets[ep][k] += float(r.get(k, 0.0))
        epoch_counts[ep] += 1

    epoch_rows: List[Dict[str, Any]] = []
    for ep in sorted(epoch_buckets.keys()):
        c = max(1, epoch_counts[ep])
        row: Dict[str, Any] = {"epoch": ep, "steps_in_epoch": c}
        for k in numeric_keys:
            row[f"mean_{k}"] = epoch_buckets[ep][k] / c
        epoch_rows.append(row)
    with (out_dir / "train_curves_epoch_avg.csv").open("w", encoding="utf-8", newline="") as f:
        epoch_fields = sorted({k for r in epoch_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=epoch_fields)
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        epochs = [int(r.get("epoch", 0)) for r in epoch_rows]
        plt.figure(figsize=(9, 5))
        for col, label in (
            ("mean_loss_total", "mean_loss_total"),
            ("mean_loss_align", "mean_loss_align"),
            ("mean_loss_pred", "mean_loss_pred"),
        ):
            ys = [float(r.get(col, float("nan"))) for r in epoch_rows]
            if any(np.isfinite(y) for y in ys):
                plt.plot(epochs, ys, label=label)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train Curves (epoch average, scans_dev50)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "train_curves.png", dpi=150)
        plt.close()

        # Single figure: only terms with independent weights (values are contributions to total loss).
        # mean_loss_depth is w_pred times the official backbone depth loss contribution.
        weighted_cols = [
            ("mean_loss_align_mse", "align_mse"),
            ("mean_loss_align_cos", "align_cos"),
            ("mean_loss_align_cka", "align_cka"),
            ("mean_loss_camera", "pred_camera"),
            ("mean_loss_depth", "pred_depth"),
        ]
        plt.figure(figsize=(11, 6))
        any_plotted = False
        for col, label in weighted_cols:
            ys = [float(r.get(col, float("nan"))) for r in epoch_rows]
            if not any(np.isfinite(y) for y in ys):
                continue
            plt.plot(epochs, ys, label=label)
            any_plotted = True
        if any_plotted:
            plt.xlabel("epoch")
            plt.ylabel("weighted_loss_component")
            plt.title("Weighted loss components")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "train_loss_weighted_components.png", dpi=150)
        plt.close()
    except Exception:
        pass


def _float_metrics_from_aggregate(agg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in agg.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


_SCANNET_KEEP_KEYS = (
    "acc",
    "comp",
    "cd",
    "chamfer_distance",
    "ape",
    "are",
    "rpe_rot",
    "rpe_trans",
)

# Omitted from periodic-eval history / CSV / plots (still computed inside eval helpers).
_EVAL_HISTORY_SKIP_KEYS = frozenset({"inference_time_ms", "infer_peak_vram_gb"})


# Periodic eval covers the enabled dataset tags. Two metric families:
#   - scannet_dev50, scannet_test50 → ScanNet metrics (chamfer/ape/are/...)
#   - 7scenes, nrgbd, nuscenes, waymo → reconstruction + trajectory (acc/comp/cd, ape/are/rpe, timing)
# Plotting groups (`_write_eval_artifacts`) share these splits.
EVAL_SCANNET_DATASETS = ("scannet_dev50", "scannet_test50")
EVAL_RECON_DATASETS = ("7scenes", "nrgbd", "nuscenes", "waymo")

# Non-ScanNet metrics kept in periodic-eval history / plots (subset of average_metrics.json).
_RECON_KEEP_KEYS = (
    "acc",
    "comp",
    "cd",
    "acc_med",
    "comp_med",
    "cd_med",
    "ape",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "infer_peak_vram_gb",
)


def _resolve_enabled_eval_datasets(args: argparse.Namespace) -> Tuple[str, ...]:
    """Return periodic-eval dataset order after per-dataset boolean switches."""
    selected: List[str] = []
    if bool(args.eval_scannet_dev50):
        selected.append("scannet_dev50")
    if bool(args.eval_scannet_test50):
        selected.append("scannet_test50")
    if bool(args.eval_7scenes):
        selected.append(dataset_output_tag("7scenes", args.eval_7scenes_split))
    if bool(args.eval_nrgbd):
        selected.append("nrgbd")
    if bool(args.eval_nuscenes):
        selected.append(dataset_output_tag("nuscenes", args.eval_nuscenes_split))
    if bool(args.eval_waymo):
        selected.append(dataset_output_tag("waymo", args.eval_waymo_split))
    return tuple(selected)


def _dataset_for_bounds(eval_dataset_name: str) -> str:
    # Bounds are stored under analysis_outputs_v3/<backbone>/<dataset_tag>/eval/...
    # Keep ScanNet splits distinct (scannet_dev50 vs scannet_test50).
    if eval_dataset_name in ("scannet_dev50", "scannet_test50"):
        return eval_dataset_name
    if eval_dataset_name.startswith("7scenes"):
        return eval_dataset_name
    if eval_dataset_name == "nrgbd":
        return eval_dataset_name
    if eval_dataset_name.startswith("nuscenes"):
        return eval_dataset_name
    if eval_dataset_name.startswith("waymo"):
        return eval_dataset_name
    raise ValueError(f"Unsupported periodic eval dataset: {eval_dataset_name}")


def _root_for_eval_dataset(args: argparse.Namespace, dataset_name: str) -> Path:
    if dataset_name == "7scenes":
        return Path(args.root_7scenes)
    if dataset_name == "nrgbd":
        return Path(args.root_nrgbd)
    if dataset_name == "nuscenes":
        return Path(args.root_nuscenes)
    if dataset_name == "waymo":
        return Path(args.root_waymo)
    raise ValueError(f"Unsupported periodic eval dataset root: {dataset_name}")


def _load_eval_bounds_from_v3(
    *,
    eval_bounds_root: Path,
    backbone: str,
    eval_datasets: Sequence[str],
    plan_name: str = "",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    bounds: Dict[str, Dict[str, Dict[str, float]]] = {}
    backbone_key = str(backbone).strip().lower()
    for ds in eval_datasets:
        source_ds = _dataset_for_bounds(str(ds))
        ds_root = Path(eval_bounds_root) / backbone_key / source_ds / "eval"
        upper_path = ds_root / "upper" / "eval_record_v3.json"
        lower_path = ds_root / "lower" / "eval_record_v3.json"
        if not upper_path.exists():
            # Backward-compat for legacy naming.
            upper_legacy = ds_root / f"upper_{backbone_key}" / "eval_record_v3.json"
            if upper_legacy.exists():
                upper_path = upper_legacy
        if not lower_path.exists():
            plan_key = str(plan_name or "").strip()
            if plan_key:
                group_key = plan_group_from_name(plan_key)
                for candidate in (group_key, plan_key):
                    lower_plan = ds_root / "lower" / candidate / "eval_record_v3.json"
                    if lower_plan.exists():
                        lower_path = lower_plan
                        break
        if not lower_path.exists():
            lower_legacy = ds_root / f"lower_{backbone_key}" / "eval_record_v3.json"
            if lower_legacy.exists():
                lower_path = lower_legacy
        if not upper_path.exists() or not lower_path.exists():
            missing = []
            if not upper_path.exists():
                missing.append(str(upper_path))
            if not lower_path.exists():
                missing.append(str(lower_path))
            raise SystemExit(
                "Missing periodic-eval bound files:\n" + "\n".join(missing)
            )
        upper_payload = json.loads(upper_path.read_text(encoding="utf-8"))
        lower_payload = json.loads(lower_path.read_text(encoding="utf-8"))
        upper_avg = _float_metrics_from_aggregate(dict(upper_payload.get("average", {})))
        lower_avg = _float_metrics_from_aggregate(dict(lower_payload.get("average", {})))
        if not upper_avg or not lower_avg:
            raise SystemExit(
                f"Invalid bound payload for dataset={ds}: empty average in "
                f"{upper_path} or {lower_path}"
            )
        bounds[str(ds)] = {"upper": upper_avg, "lower": lower_avg}
    return bounds


def _append_eval_history(
    history_rows: List[Dict[str, Any]],
    *,
    epoch: int,
    dataset: str,
    tag: str,
    metrics: Dict[str, float],
) -> None:
    for k, v in metrics.items():
        if str(k) in _EVAL_HISTORY_SKIP_KEYS:
            continue
        history_rows.append(
            {
                "epoch": int(epoch),
                "dataset": str(dataset),
                "tag": str(tag),
                "metric": str(k),
                "value": float(v),
            }
        )


def _append_lower_bounds_as_epoch_zero(
    history_rows: List[Dict[str, Any]],
    bounds: Dict[str, Dict[str, Dict[str, float]]],
    eval_datasets: Sequence[str],
) -> int:
    """Seed periodic-eval curves with the precomputed lower result at epoch 0."""
    n_rows_before = len(history_rows)
    existing_epoch0: Set[str] = set()
    for r in history_rows:
        if str(r.get("tag", "")) != "train":
            continue
        try:
            row_epoch = int(r.get("epoch", -1))
        except (TypeError, ValueError):
            continue
        if row_epoch == 0:
            existing_epoch0.add(str(r.get("dataset", "")))
    for ds in eval_datasets:
        ds_key = str(ds)
        if ds_key in existing_epoch0:
            continue
        lower_metrics = bounds.get(ds_key, {}).get("lower", {})
        if not lower_metrics:
            continue
        _append_eval_history(
            history_rows,
            epoch=0,
            dataset=ds_key,
            tag="train",
            metrics=lower_metrics,
        )
    return len(history_rows) - n_rows_before


def _run_periodic_eval_for_epoch(
    *,
    eval_epoch: int,
    model: nn.Module,
    eval_datasets: Sequence[str],
    args: argparse.Namespace,
    backbone: str,
    device: torch.device,
    out_dir: Path,
    eval_history_rows: List[Dict[str, Any]],
    log_fn,
) -> None:
    """Run periodic eval (eval_v3) and append metrics for ``eval_epoch`` (0 = before training)."""
    from analysis.datasets.registry import get_dataset
    from analysis.eval.eval_v3 import evaluate_model_v3

    adapter = get_backbone(backbone)
    model_was_training = bool(model.training)
    model.eval()
    log_fn(f"[eval] epoch={eval_epoch}: evaluating current LoRA model on {', '.join(eval_datasets)}")
    for ds in eval_datasets:
        t0 = time.perf_counter()
        if ds == "scannet_dev50":
            ds_plugin = get_dataset("scannet")
            out_p = out_dir / "eval_epochs" / f"epoch_{eval_epoch}" / "scannet_dev50"
            evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=ds_plugin,
                experiment="train",
                dataset="scannet",
                data_root=Path(args.dev_data_root),
                backbone=backbone,
                model_name=f"Train_epoch_{eval_epoch}",
                device=device,
                output_dir=out_p,
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.max_eval_views),
            )
        elif ds == "scannet_test50":
            ds_plugin = get_dataset("scannet")
            out_p = out_dir / "eval_epochs" / f"epoch_{eval_epoch}" / "scannet_test50"
            evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=ds_plugin,
                experiment="train",
                dataset="scannet",
                data_root=Path(args.test_data_dir),
                backbone=backbone,
                model_name=f"Train_epoch_{eval_epoch}",
                device=device,
                output_dir=out_p,
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.max_eval_views),
            )
        else:
            plugin_name, split = parse_dataset_plugin_args(ds)
            root = _root_for_eval_dataset(args, plugin_name)
            ds_plugin = get_dataset(plugin_name)
            out_p = out_dir / "eval_epochs" / f"epoch_{eval_epoch}" / ds
            evaluate_model_v3(
                adapter=adapter,
                model=model,
                ds_plugin=ds_plugin,
                experiment="train",
                dataset=plugin_name,
                data_root=root,
                backbone=backbone,
                model_name=f"Train_epoch_{eval_epoch}",
                device=device,
                output_dir=out_p,
                max_scenes=int(args.eval_max_scenes),
                max_frames=int(args.max_eval_views),
                split=split,
                output_dataset=ds,
            )
        log_fn(f"[eval][epoch={eval_epoch}][{ds}] elapsed={time.perf_counter() - t0:.2f}s")
        avg_path = out_p / "eval_record_v3.json"
        if not avg_path.exists():
            raise SystemExit(f"Periodic eval missing output: {avg_path}")
        payload = json.loads(avg_path.read_text(encoding="utf-8"))
        avg_metrics = _float_metrics_from_aggregate(dict(payload.get("average", {})))
        _append_eval_history(
            eval_history_rows,
            epoch=int(eval_epoch),
            dataset=str(ds),
            tag="train",
            metrics=avg_metrics,
        )
    if model_was_training:
        model.train()


def _plot_metric_dataset(
    out_dir: Path,
    history_rows: List[Dict[str, Any]],
    bounds: Dict[str, Dict[str, Dict[str, float]]],
    *,
    dataset: str,
    color: str,
    metric_whitelist: Optional[Sequence[str]] = None,
) -> None:
    """Render one PNG per metric for one dataset.

    A metric is plotted when this dataset has upper/lower bounds and at least one
    history row. When ``metric_whitelist`` is provided, only those metrics are considered (drops
    plot-clutter rows like ``num_scenes`` that are stored in bounds for
    transparency but not meaningful in periodic-eval curves).
    """
    import matplotlib.pyplot as plt  # type: ignore

    ds_rows_all = [
        r
        for r in history_rows
        if str(r.get("tag", "")) == "train" and str(r.get("dataset", "")) == dataset
    ]
    if not ds_rows_all:
        return
    candidate_metrics = sorted({str(r["metric"]) for r in ds_rows_all})
    if metric_whitelist is not None:
        allowed = set(metric_whitelist)
        candidate_metrics = [m for m in candidate_metrics if m in allowed]

    for metric in candidate_metrics:
        ub = bounds.get(dataset, {}).get("upper", {}).get(metric)
        lb = bounds.get(dataset, {}).get("lower", {}).get(metric)
        if ub is None or lb is None:
            continue
        ds_rows = sorted(
            (r for r in ds_rows_all if str(r.get("metric", "")) == metric),
            key=lambda r: int(r["epoch"]),
        )
        if not ds_rows:
            continue

        plt.figure(figsize=(10, 6))
        z_band = 1
        z_curve = 3
        plt.axhline(
            float(ub), linestyle="-.", color=color, linewidth=1.6,
            label=f"{dataset}_upper", zorder=z_band, alpha=0.95,
        )
        plt.axhline(
            float(lb), linestyle=":", color=color, linewidth=1.6,
            label=f"{dataset}_lower", zorder=z_band, alpha=0.95,
        )
        xs = [int(r["epoch"]) for r in ds_rows]
        ys = [float(r["value"]) for r in ds_rows]
        plt.plot(
            xs, ys,
            marker="o",
            linestyle="-",
            color=color, linewidth=2.0,
            label=dataset, zorder=z_curve,
        )
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(f"{dataset} {metric} (upper/lower/curve)")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"eval_plot_{dataset}_{metric}.png", dpi=150)
        plt.close()


def _write_eval_artifacts(
    out_dir: Path,
    bounds: Dict[str, Dict[str, Dict[str, float]]],
    history_rows: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    with (out_dir / "eval_metrics_history.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["epoch", "dataset", "tag", "metric", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in history_rows:
            w.writerow(r)
    try:
        # Plot every dataset separately so split curves are not overlaid.
        scannet_active = [ds for ds in EVAL_SCANNET_DATASETS if ds in bounds]
        recon_active = [ds for ds in EVAL_RECON_DATASETS if ds in bounds]
        recon_active.extend(
            ds
            for ds in bounds
            if (ds.startswith("7scenes_") or ds.startswith("nuscenes_") or ds.startswith("waymo_"))
            and ds not in recon_active
        )
        scannet_colors = {"scannet_dev50": "#c62828", "scannet_test50": "#1565c0"}
        recon_colors = {"nrgbd": "#1565c0", "nuscenes": "#2e7d32", "waymo": "#6a1b9a"}
        for ds in recon_active:
            if ds.startswith("7scenes"):
                recon_colors[ds] = "#c62828"
            elif ds.startswith("nuscenes"):
                recon_colors[ds] = "#2e7d32"
            elif ds.startswith("waymo"):
                recon_colors[ds] = "#6a1b9a"
        for ds in scannet_active:
            _plot_metric_dataset(
                out_dir,
                history_rows,
                bounds,
                dataset=ds,
                color=scannet_colors.get(ds, "#444444"),
                metric_whitelist=_SCANNET_KEEP_KEYS,
            )
        for ds in recon_active:
            _plot_metric_dataset(
                out_dir,
                history_rows,
                bounds,
                dataset=ds,
                color=recon_colors.get(ds, "#444444"),
                metric_whitelist=_RECON_KEEP_KEYS,
            )
    except Exception:
        pass


def _fmt_metric_brief(metrics: Dict[str, float], keys: Sequence[str]) -> str:
    parts: List[str] = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.6f}")
    return " ".join(parts)


def _extract_bound_payload(
    bounds: Dict[str, Dict[str, Dict[str, float]]],
    tag: Literal["upper", "lower"],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for ds in bounds.keys():
        out[ds] = dict(bounds.get(ds, {}).get(tag, {}))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train-based cache-first unit-prune + LoRA recovery.")
    p.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name (default: BACKEND env var, else vggt).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        required=True,
        help="Training cache directory. Repeat to train on multiple cache datasets.",
    )
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--prune-plan-file",
        type=Path,
        required=True,
        help="Unit prune plan JSON, or prune_plan_all.json with --prune-plan-name + --recovery-plan-name selectors.",
    )
    p.add_argument(
        "--prune-plan-name",
        dest="prune_plan_name",
        type=str,
        default="",
        help="Prune selector for resolving split prune/training bundles (e.g. prune_span10).",
    )
    p.add_argument(
        "--recovery-plan-root",
        type=Path,
        default=None,
        help="Directory with recovery plan bundles. Defaults to --prune-plan-file parent.",
    )
    p.add_argument(
        "--recovery-plan-name",
        type=str,
        default="",
        help="Recovery plan name (e.g. recovery_lora_l1_r32).",
    )
    p.add_argument("--variant", type=str, default="unit_prune_plan", choices=("unit_prune_plan",))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes.")
    p.add_argument("--max-steps", type=int, default=0, help="0 means no early stop.")
    p.add_argument("--save-every-epoch", type=int, default=10)
    p.add_argument("--resume-from", type=Path, default=None, help="Path to checkpoint to resume from.")
    p.add_argument(
        "--bounds-cache-file",
        type=Path,
        default=None,
        help=(
            "Deprecated: bounds are always read directly from analysis_outputs/evaluation/<scans_set>_input_frame_<N>/"
            "<Baseline|Ours>_<backend>/average_metrics.json via EVAL_ROOT/BACKEND env vars."
        ),
    )
    p.add_argument("--eval-epoch", type=int, default=0, help="Run dev/test eval every N epochs. 0 disables periodic eval.")
    p.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run periodic eval once before training (epoch=0), independent of --eval-epoch.",
    )
    p.add_argument(
        "--eval-bounds-root",
        type=Path,
        default=Path("analysis_outputs_v3"),
        help="Root containing precomputed upper/lower eval_record_v3.json for periodic-eval plots.",
    )
    p.add_argument("--dev-data-root", type=Path, default=Path("data/scans_dev50"))
    p.add_argument("--test-data-dir", type=Path, default=Path("data/scans_test50"))
    p.add_argument("--test-gt-ply-dir", type=Path, default=Path("data/scans_test50"))
    p.add_argument(
        "--test-sanity-average-metrics-file",
        type=Path,
        default=None,
        help="Path to run_eval ScanNet baseline average_metrics.json used for upper(test) sanity check.",
    )
    p.add_argument("--eval-max-scenes", type=int, default=0, help="0 means full split.")
    p.add_argument("--eval-input-frame", type=int, default=30)
    p.add_argument(
        "--max-eval-views",
        type=int,
        default=50,
        help="Max frames per scene for periodic eval (passed to eval_v3 --max-frames).",
    )
    p.add_argument(
        "--eval-scannet-dev50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on scannet_dev50.",
    )
    p.add_argument(
        "--eval-scannet-test50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on scannet_test50.",
    )
    p.add_argument(
        "--eval-7scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on 7scenes.",
    )
    p.add_argument(
        "--eval-7scenes-split",
        type=str,
        default="test",
        choices=("train", "test"),
        help="7scenes split for periodic eval (output tag: 7scenes_<split>).",
    )
    p.add_argument(
        "--eval-nrgbd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable periodic eval on nrgbd.",
    )
    p.add_argument(
        "--eval-nuscenes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic eval on nuscenes.",
    )
    p.add_argument(
        "--eval-nuscenes-split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="nuScenes split for periodic eval (output tag: nuscenes_<split>).",
    )
    p.add_argument(
        "--eval-waymo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic eval on waymo.",
    )
    p.add_argument(
        "--eval-waymo-split",
        type=str,
        default="validation",
        choices=("training", "validation", "train", "val"),
        help="Waymo split for periodic eval (output tag: waymo_<split>).",
    )
    # Non-ScanNet periodic eval roots (mirrors scripts/run_eval.sh defaults).
    p.add_argument("--root-7scenes", type=Path, default=Path("data/7scenes"))
    p.add_argument("--root-nrgbd", type=Path, default=Path("data/neural_rgbd_data"))
    p.add_argument("--root-nuscenes", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--root-waymo", type=Path, default=Path("data/waymo"))
    p.add_argument(
        "--eval-7andn-kf",
        type=int,
        default=10,
        help="Keyframe step for 7scenes/NRGBD eval (matches scripts/run_eval.sh KF=10).",
    )
    p.add_argument(
        "--eval-7andn-max-scenes-per-dataset",
        type=int,
        default=0,
        help="Cap scenes per 7andn dataset for fast eval. 0 = use full split.",
    )
    p.add_argument(
        "--test-sanity-atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for upper(test) vs run_eval average sanity check.",
    )
    p.add_argument(
        "--test-sanity-rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for upper(test) vs run_eval average sanity check.",
    )
    p.add_argument("--target-source", type=str, default="teacher", choices=("teacher", "dataset"))
    p.add_argument(
        "--gt-data-root",
        type=Path,
        default=Path("data/scans_dev50"),
        help="ScanNet-style GT root for --target-source=dataset.",
    )
    p.add_argument(
        "--use-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for replayed frame/global blocks.",
    )
    p.add_argument(
        "--print-step-timing",
        action="store_true",
        help="Print per-step timing breakdown to help identify bottlenecks.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    backbone = str(args.backbone or os.environ.get("BACKEND", "vggt")).strip() or "vggt"
    engine = get_cache_replay_engine(backbone)
    train_cfg = engine.training_config()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[str(args.dtype)]
    use_amp = device.type == "cuda" and str(args.dtype) in ("float16", "bfloat16")
    test_sanity_avg_path = (
        Path(args.test_sanity_average_metrics_file)
        if args.test_sanity_average_metrics_file is not None
        else Path("analysis_outputs/evaluation/scans_test50/baseline") / f"input_frame_{int(args.eval_input_frame)}" / "average_metrics.json"
    )

    prune_plan = _load_unit_prune_plan(
        args.prune_plan_file,
        args.prune_plan_name,
        args.recovery_plan_root,
        args.recovery_plan_name,
    )
    training_method = str(
        prune_plan.extra.get("training_method", "lora_prune")
        if isinstance(prune_plan.extra, dict)
        else "lora_prune"
    ).strip().lower()
    frame_skip_intervals, global_skip_intervals, frame_layers, global_layers = drop_intervals_from_plan(prune_plan)
    intervals = sorted(set(frame_skip_intervals).union(set(global_skip_intervals)))
    lora_unit_indices = sorted(int(k) for k in prune_plan.lora.keys())
    train_unit_indices = _training_anchor_unit_indices(prune_plan)
    lora_anchor_layers = (
        lora_layers_from_plan(prune_plan)
        if prune_plan.lora
        else sorted({unit_index_to_kind_layer(int(idx))[1] for idx in train_unit_indices})
    )
    seed_layer, seed_kind, _replay_start = (
        replay_seed_spec(prune_plan)
        if prune_plan.lora
        else _replay_seed_spec_from_unit_indices(train_unit_indices)
    )
    drop_actions, lora_actions = prune_actions_from_plan(prune_plan)
    if not lora_anchor_layers:
        raise SystemExit("Training plan has no trainable anchors; nothing to train.")

    cache_dirs = _resolve_cache_dirs(args)
    if not cache_dirs:
        raise SystemExit("At least one --cache-dir is required.")
    scenes = _load_training_scenes(cache_dirs, int(args.max_scenes))
    if not scenes:
        raise SystemExit("No scenes selected from cache.")

    num_layers = infer_num_layers(cache_dirs[0], [s for s in scenes if str(s.get("_cache_dir", "")) == str(cache_dirs[0])])
    for cache_dir in cache_dirs[1:]:
        cache_scenes = [s for s in scenes if str(s.get("_cache_dir", "")) == str(cache_dir)]
        if not cache_scenes:
            continue
        cache_num_layers = infer_num_layers(cache_dir, cache_scenes)
        if int(cache_num_layers) != int(num_layers):
            raise SystemExit(
                f"All training caches must have the same num_layers; got {cache_dirs[0]}={num_layers} "
                f"and {cache_dir}={cache_num_layers}"
            )
    if prune_plan.num_units != int(num_layers * 2):
        raise SystemExit(
            f"prune plan num_units={prune_plan.num_units} != expected {int(num_layers * 2)} from cache"
        )
    if any(r >= num_layers for (_l, r) in frame_skip_intervals + global_skip_intervals):
        raise SystemExit(
            f"Plan-derived drop intervals out of range for num_layers={num_layers}: "
            f"frame={frame_skip_intervals}, global={global_skip_intervals}"
        )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "backbone": backbone,
                "cache_dir": str(cache_dirs[0]),
                "cache_dirs": [str(p) for p in cache_dirs],
                "hf_model": args.hf_model,
                "drop_indices": sorted(set(int(x) for x in prune_plan.drop)),
                "lora_indices": lora_unit_indices,
                "train_unit_indices": train_unit_indices,
                "training_method": training_method,
                "frame_drop_layers": frame_layers,
                "global_drop_layers": global_layers,
                "variant": args.variant,
                "epochs": int(args.epochs),
                "max_scenes": int(args.max_scenes),
                "max_steps": int(args.max_steps),
                "save_every_epoch": int(args.save_every_epoch),
                "eval_epoch": int(args.eval_epoch),
                "eval_at_start": bool(args.eval_at_start),
                "eval_bounds_root": str(args.eval_bounds_root),
                "eval_max_scenes": int(args.eval_max_scenes),
                "eval_input_frame": int(args.eval_input_frame),
                "dev_data_root": str(args.dev_data_root),
                "test_data_dir": str(args.test_data_dir),
                "test_gt_ply_dir": str(args.test_gt_ply_dir),
                "test_sanity_average_metrics_file": (
                    str(args.test_sanity_average_metrics_file) if args.test_sanity_average_metrics_file else None
                ),
                "test_sanity_atol": float(args.test_sanity_atol),
                "test_sanity_rtol": float(args.test_sanity_rtol),
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "prune_plan_file": str(args.prune_plan_file),
                "prune_plan_name": str(args.prune_plan_name),
                "recovery_plan_root": str(args.recovery_plan_root) if args.recovery_plan_root else None,
                "recovery_plan_name": str(args.recovery_plan_name),
                "device": str(device),
                "dtype": str(args.dtype),
                "training_config": {
                    "optimizer_cls": getattr(train_cfg.optimizer_cls, "__name__", str(train_cfg.optimizer_cls)),
                    "lr": float(train_cfg.lr),
                    "weight_decay": float(train_cfg.weight_decay),
                    "min_lr": float(train_cfg.min_lr),
                    "warmup_ratio": float(train_cfg.warmup_ratio),
                    "grad_clip_max_norm": float(train_cfg.grad_clip_max_norm),
                    "w_align": float(train_cfg.w_align),
                    "w_align_mse": float(train_cfg.w_align_mse),
                    "w_align_cos": float(train_cfg.w_align_cos),
                    "w_align_cka": float(train_cfg.w_align_cka),
                    "w_pred": float(train_cfg.w_pred),
                    "w_reg": float(train_cfg.w_reg),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_file = (out_dir / "train.log").open("a", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    skipped_global_idxs = build_skipped_layer_indices(global_skip_intervals)
    skipped_frame_idxs = build_skipped_layer_indices(frame_skip_intervals)
    skipped_idxs = sorted(set(skipped_global_idxs).union(skipped_frame_idxs))
    # Periodic eval uses the same implementation as stage-4 eval (`analysis.eval.eval_v3`),
    # and runs on the in-memory model (current LoRA weights).
    eval_datasets = _resolve_enabled_eval_datasets(args) if engine.supports_periodic_eval() else ()
    eval_bounds: Dict[str, Dict[str, Dict[str, float]]] = {}
    eval_history_rows: List[Dict[str, Any]] = []
    periodic_eval_enabled = int(args.eval_epoch) > 0
    eval_at_start_enabled = bool(args.eval_at_start)
    if eval_datasets and (periodic_eval_enabled or eval_at_start_enabled):
        eval_bounds = _load_eval_bounds_from_v3(
            eval_bounds_root=Path(args.eval_bounds_root),
            backbone=backbone,
            eval_datasets=eval_datasets,
            plan_name=str(args.prune_plan_name),
        )
        _log(
            "[eval] loaded periodic bounds from "
            f"{Path(args.eval_bounds_root)} for datasets={','.join(eval_datasets)}"
        )
        if periodic_eval_enabled and not eval_at_start_enabled:
            seeded_rows = _append_lower_bounds_as_epoch_zero(
                eval_history_rows,
                eval_bounds,
                eval_datasets,
            )
            if seeded_rows > 0:
                _log(
                    "[eval] seeded epoch=0 from precomputed lower bounds "
                    f"({seeded_rows} metric rows)"
                )

    model, trainable, pre_prune_total_params, removed_params = engine.build_train_model(
        hf_model=args.hf_model,
        device=device,
        drop_actions=drop_actions,
        frame_skip_intervals=frame_skip_intervals,
        global_skip_intervals=global_skip_intervals,
        skipped_frame_idxs=skipped_frame_idxs,
        skipped_global_idxs=skipped_global_idxs,
        lora_actions=lora_actions,
        prune=True,
        inject_lora=(training_method == "lora_prune"),
    )
    if training_method == "train_original":
        trainable = _enable_original_train_units(model, backbone, train_unit_indices)
    if not trainable:
        raise SystemExit(f"No trainable parameters were enabled for training_method={training_method!r}.")
    gt_scene_map: Dict[str, SceneData] = {}
    if str(args.target_source) == "dataset":
        if not engine.supports_dataset_targets():
            raise SystemExit(f"--target-source=dataset is not supported for backbone={backbone}")
        gt_scenes = discover_scannet_scenes(Path(args.gt_data_root))
        gt_scene_map = {str(s.scene_id): s for s in gt_scenes}
        if not gt_scene_map:
            raise SystemExit(f"No GT scenes discovered under --gt-data-root={args.gt_data_root}")

    accounting_params = list(engine.parameter_accounting_parameters(model))
    total_params = sum(p.numel() for p in accounting_params)
    trainable_params = sum(p.numel() for p in accounting_params if p.requires_grad)
    frozen_params = total_params - trainable_params
    post_vs_pre_ratio = (100.0 * total_params / pre_prune_total_params) if pre_prune_total_params > 0 else 0.0
    frozen_ratio = (100.0 * frozen_params / total_params) if total_params > 0 else 0.0
    train_ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    _log(
        f"[params] {backbone} transformer blocks (excludes embeddings and heads): "
        f"pre_prune={_format_param_count(pre_prune_total_params)} "
        f"post_prune={_format_param_count(total_params)} ({post_vs_pre_ratio:.2f}% of pre-prune) "
        f"frozen={_format_param_count(frozen_params)} ({frozen_ratio:.2f}% of post-prune) "
        f"trainable={_format_param_count(trainable_params)} ({train_ratio:.2f}% of post-prune) "
        f"skipped {len(skipped_idxs)} layers"
    )

    optim = engine.build_optimizer(trainable, args=args)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rows: List[Dict[str, Any]] = []
    global_step = 0
    start_epoch = 0

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise SystemExit(f"--resume-from not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "trainable_state_dict" in ckpt:
            current_trainable = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            saved = ckpt["trainable_state_dict"]
            if set(saved) != set(current_trainable):
                missing = sorted(set(current_trainable) - set(saved))
                unexpected = sorted(set(saved) - set(current_trainable))
                raise RuntimeError(
                    "Delta checkpoint trainable parameter set mismatch: "
                    f"missing={missing[:8]} unexpected={unexpected[:8]}"
                )
            with torch.no_grad():
                for name, parameter in current_trainable.items():
                    value = saved[name]
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise RuntimeError(
                            f"Delta checkpoint shape mismatch for {name}: "
                            f"{tuple(value.shape)} vs {tuple(parameter.shape)}"
                        )
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        else:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scaler_state = ckpt.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        _log(f"[resume] from={ckpt_path} start_epoch={start_epoch} global_step={global_step}")

    if (
        engine.supports_periodic_eval()
        and eval_at_start_enabled
        and start_epoch == 0
        and eval_datasets
    ):
        _log("[eval] pre-train: running epoch=0 eval (--eval-at-start)")
        _run_periodic_eval_for_epoch(
            eval_epoch=0,
            model=model,
            eval_datasets=eval_datasets,
            args=args,
            backbone=backbone,
            device=device,
            out_dir=out_dir,
            eval_history_rows=eval_history_rows,
            log_fn=_log,
        )
        _write_eval_artifacts(out_dir, eval_bounds, eval_history_rows)

    t0_train = time.perf_counter()
    remaining_epochs = max(0, int(args.epochs) - int(start_epoch))
    total_steps = _compute_total_steps(len(scenes), remaining_epochs, int(args.max_steps))
    train_desc = (
        "    train"
        if os.environ.get("COMPACT_LOG", "").lower() in {"1", "true", "yes"}
        else "train"
    )
    pbar = tqdm(total=total_steps if total_steps > 0 else None, desc=train_desc, dynamic_ncols=True)
    def _save_checkpoint(ckpt_path: Path, epoch_idx: int) -> None:
        payload = {
                "global_step": global_step,
                "epoch": int(epoch_idx),
                "epochs": int(args.epochs),
                "drop_indices": sorted(set(int(x) for x in prune_plan.drop)),
                "lora_indices": lora_unit_indices,
                "train_unit_indices": train_unit_indices,
                "training_method": training_method,
                "prune_plan": asdict(prune_plan),
                "variant": "trainable_recovery_delta_v1",
                "training_variant": args.variant,
                "checkpoint_format": "trainable_recovery_delta_v1",
                "save_every_epoch": int(args.save_every_epoch),
                "trainable_state_dict": {
                    name: parameter.detach()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                },
                "optimizer_state_dict": optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }
        atomic_save_checkpoint(payload, ckpt_path)

    def _publish_latest(serialized_path: Path) -> None:
        atomic_publish_latest(serialized_path, serialized_path.parent / "latest.pt")

    for epoch in range(start_epoch, int(args.epochs)):
        epoch_scenes = list(scenes)
        random.Random(int(args.seed) + epoch).shuffle(epoch_scenes)
        for si, item in enumerate(epoch_scenes):
            t_step0 = time.perf_counter()
            sid = str(item.get("scene_id", ""))
            item_cache_dir_raw = str(item.get("_cache_dir", "")).strip()
            if not item_cache_dir_raw:
                raise RuntimeError(f"Scene {sid!r} is missing _cache_dir")
            item_cache_dir = Path(item_cache_dir_raw)
            t_load0 = time.perf_counter()
            row = _load_scene_cache_row(item_cache_dir, item)
            engine.validate_cache_row(row, num_layers=num_layers)
            layer_cpu_cache: Dict[int, torch.Tensor] = {}
            unit_cpu_cache: Dict[str, torch.Tensor] = {}

            def _get_layer_cpu(idx: int) -> torch.Tensor:
                cached = layer_cpu_cache.get(idx)
                if cached is not None:
                    return cached
                t = load_scene_replay_layer(
                    item_cache_dir,
                    item,
                    int(idx),
                    expected_num_layers=num_layers,
                    map_location="cpu",
                )
                layer_cpu_cache[idx] = t
                return t

            def _get_unit_cpu(unit_id: str) -> torch.Tensor:
                key = str(unit_id)
                cached = unit_cpu_cache.get(key)
                if cached is None:
                    cached = load_scene_unit(
                        item_cache_dir,
                        item,
                        key,
                        mmap=True,
                        map_location="cpu",
                    )
                    unit_cpu_cache[key] = cached
                return cached

            # Replay engines use this direct-unit callback for the seed while
            # preserving their public get_layer_cpu API for head concat layers.
            setattr(_get_layer_cpu, "unit_loader", _get_unit_cpu)

            from analysis.redundancy.cache_layout_v3 import require_minimal_cache_geometry

            require_minimal_cache_geometry(row, cache_dir=item_cache_dir)

            teacher_pred = teacher_pred_to_map(row["pred"])
            adapter = get_backbone(backbone)
            h, w, seq_len, _patch_start = adapter.infer_train_cache_geometry(
                row, model, _get_layer_cpu, cache_dir=item_cache_dir
            )
            dummy_images = adapter.build_train_dummy_images(row, model, device, _get_layer_cpu)
            t_load = time.perf_counter() - t_load0

            t_replay = 0.0
            t_align = 0.0
            t_pred = 0.0
            t_backward = 0.0
            t_optim = 0.0
            alignment_capture_layers = sorted(
                {int(k) // 2 for k in prune_plan.lora.keys()}
                if prune_plan.lora
                else {unit_index_to_kind_layer(int(idx))[1] for idx in train_unit_indices}
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                t_replay0 = time.perf_counter()
                initial_tokens = None
                if str(seed_kind).strip().lower() == "none":
                    from analysis.redundancy.cache_layout import load_scene_initial_tokens

                    initial_tokens = load_scene_initial_tokens(item_cache_dir, item).to(
                        device=device, non_blocking=(device.type == "cuda")
                    )
                replay_kw: Dict[str, Any] = dict(
                    frame_skip_intervals=frame_skip_intervals,
                    global_skip_intervals=global_skip_intervals,
                    lora_anchor_layers=lora_anchor_layers,
                    alignment_capture_layers=alignment_capture_layers,
                    seed_layer=int(seed_layer),
                    seed_kind=str(seed_kind),
                    initial_tokens=initial_tokens,
                    use_checkpoint=bool(args.use_checkpoint),
                    device=device,
                    dummy_images=dummy_images,
                    **adapter.train_replay_extra_kwargs(h=int(h), w=int(w)),
                    **engine.train_replay_extra_kwargs(h=int(h), w=int(w)),
                )
                replay = engine.replay_student_cache_first(
                    model,
                    get_layer_cpu=_get_layer_cpu,
                    num_layers=int(num_layers),
                    **replay_kw,
                )
                t_replay = time.perf_counter() - t_replay0

                loss_align = torch.zeros((), device=device)
                align_mse = torch.zeros((), device=device)
                align_cos = torch.zeros((), device=device)
                align_cka = torch.zeros((), device=device)
                wam = float(train_cfg.w_align_mse)
                wac = float(train_cfg.w_align_cos)
                wak = float(train_cfg.w_align_cka)
                n_terms = 0
                t_align0 = time.perf_counter()
                for k, spec in prune_plan.lora.items():
                    s_idx = int(k)
                    if spec.alignment_teacher_unit_index is None:
                        raise RuntimeError(f"lora[{k}] missing alignment_teacher_unit_index")
                    t_idx = int(spec.alignment_teacher_unit_index)
                    if s_idx < 0 or s_idx >= int(num_layers * 2) or t_idx < 0 or t_idx >= int(num_layers * 2):
                        raise RuntimeError(
                            f"Alignment index out of range: student={s_idx}, teacher={t_idx}, "
                            f"num_units={int(num_layers * 2)}"
                        )
                    s_layer = s_idx // 2
                    t_layer = t_idx // 2
                    s_kind = "frame" if (s_idx % 2 == 0) else "global"
                    t_kind = "frame" if (t_idx % 2 == 0) else "global"
                    s_concat = replay.concat_by_layer[s_layer]
                    t_unit_id = f"{t_kind}_{t_layer}"
                    t_term = _get_unit_cpu(t_unit_id).to(
                        device=device, non_blocking=(device.type == "cuda")
                    )
                    c_s = int(s_concat.shape[-1] // 2)
                    s_term = s_concat[..., :c_s] if s_kind == "frame" else s_concat[..., c_s:]
                    s_term = engine.normalize_alignment_term(s_term)
                    t_term = engine.normalize_alignment_term(t_term)
                    mse_i, cos_i, cka_i, term_loss = align_activation_term(
                        s_term,
                        t_term,
                        w_mse=wam,
                        w_cos=wac,
                        w_cka=wak,
                    )
                    loss_align = loss_align + term_loss
                    align_mse = align_mse + mse_i
                    align_cos = align_cos + cos_i
                    align_cka = align_cka + cka_i
                    n_terms += 1
                if n_terms > 0:
                    loss_align = loss_align / n_terms
                    align_mse = align_mse / n_terms
                    align_cos = align_cos / n_terms
                    align_cka = align_cka / n_terms
                t_align = time.perf_counter() - t_align0
                adapter.after_train_replay_release_cpu(
                    layer_cpu_cache,
                    rules=alignment_capture_layers,
                    prev_layers=lora_anchor_layers,
                    num_layers=int(num_layers),
                    seed_layer=int(seed_layer),
                )
                unit_cpu_cache.clear()
                t_pred0 = time.perf_counter()
                if str(args.target_source) == "teacher":
                    target_dataset_name = str(item.get("dataset", "") or row.get("dataset", "")).strip()
                    target_batch = engine.build_target_batch_from_teacher(
                        teacher_pred,
                        images_batched=dummy_images,
                        device=device,
                        dataset_name=target_dataset_name,
                    )
                else:
                    if sid not in gt_scene_map:
                        raise RuntimeError(
                            f"Scene {sid} not found under --gt-data-root={args.gt_data_root} "
                            f"for --target-source=dataset"
                        )
                    target_batch = _build_target_batch_from_dataset(
                        scene=gt_scene_map[sid],
                        expected_seq_len=int(seq_len),
                        device=device,
                    )
                loss_pred, pred_logs = engine.prediction_loss(
                    replay.pred,
                    target_batch,
                    args=args,
                )
                t_pred = time.perf_counter() - t_pred0

                loss_reg = engine.regularization_loss(trainable, args=args, device=device)

                loss_total = (
                    float(train_cfg.w_align) * loss_align
                    + float(train_cfg.w_pred) * loss_pred
                    + float(train_cfg.w_reg) * loss_reg
                )

            optim.zero_grad(set_to_none=True)
            t_bw0 = time.perf_counter()
            scaler.scale(loss_total).backward()
            t_backward = time.perf_counter() - t_bw0
            t_opt0 = time.perf_counter()
            lr_now = engine.update_optimizer_schedule(
                optim,
                args=args,
                global_step=global_step,
                total_steps=total_steps,
            )
            grad_norm = engine.clip_gradients(
                trainable,
                optimizer=optim,
                scaler=scaler,
                args=args,
                device=device,
            )
            scaler.step(optim)
            scaler.update()
            t_optim = time.perf_counter() - t_opt0

            global_step += 1
            step_s = time.perf_counter() - t_step0
            wa = float(train_cfg.w_align)
            wam = float(train_cfg.w_align_mse)
            wac = float(train_cfg.w_align_cos)
            wak = float(train_cfg.w_align_cka)
            wp = float(train_cfg.w_pred)
            wr = float(train_cfg.w_reg)
            target_dataset_name_log = str(item.get("dataset", "") or row.get("dataset", "")).strip()
            log_row: Dict[str, Any] = {
                "epoch": epoch,
                "scene_idx": si,
                "scene_id": sid,
                "target_dataset": target_dataset_name_log,
                "global_step": global_step,
                "loss_total": float(loss_total.detach().item()),
                "step_seconds": step_s,
                "time_load_s": t_load,
                "time_replay_s": t_replay,
                "time_align_s": t_align,
                "time_pred_s": t_pred,
                "time_backward_s": t_backward,
                "time_optim_s": t_optim,
                "optim_lr": float(lr_now),
                "grad_norm": float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            }
            if wa != 0.0:
                log_row["loss_align"] = float((wa * loss_align).detach().item())
                if wam != 0.0:
                    log_row["loss_align_mse"] = float((wa * wam * align_mse).detach().item())
                if wac != 0.0:
                    log_row["loss_align_cos"] = float((wa * wac * align_cos).detach().item())
                if wak != 0.0:
                    log_row["loss_align_cka"] = float((wa * wak * align_cka).detach().item())
            if wp != 0.0:
                log_row["loss_pred"] = float((wp * loss_pred).detach().item())
                log_row.update(dict(pred_logs))
            if wr != 0.0:
                log_row["loss_reg"] = float((wr * loss_reg).detach().item())
            rows.append(log_row)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "epoch": f"{epoch+1}/{int(args.epochs)}",
                    "scene": sid,
                    "loss": f"{log_row['loss_total']:.2f}",
                    "align_loss": f"{log_row.get('loss_align', 0.0):.2f}",
                    "prd_loss": f"{log_row.get('loss_pred', 0.0):.2f}",
                    # "step_s": f"{step_s:.1f}",
                    "rep": f"{t_replay:.1f}s",
                    "align": f"{t_align:.1f}s",
                    "prd": f"{t_pred:.1f}s",
                    "bw": f"{t_backward:.1f}s",
                    "opt": f"{t_optim:.1f}s",
                },
                refresh=False,
            )
            if bool(args.print_step_timing):
                _log(
                    f"[timing] step={global_step} load={t_load:.2f}s replay={t_replay:.2f}s "
                    f"align={t_align:.2f}s pred={t_pred:.2f}s backward={t_backward:.2f}s "
                    f"optim={t_optim:.2f}s total={step_s:.2f}s"
                )

            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                break
        if int(args.save_every_epoch) > 0 and (((epoch + 1) % int(args.save_every_epoch)) == 0):
            epoch_path = out_dir / "checkpoints" / f"epoch_{epoch+1}.pt"
            _save_checkpoint(epoch_path, epoch)
            _publish_latest(epoch_path)
        else:
            _save_checkpoint(out_dir / "checkpoints" / "latest.pt", epoch)
        _write_curves(out_dir, rows)
        if (
            engine.supports_periodic_eval()
            and periodic_eval_enabled
            and (((epoch + 1) % int(args.eval_epoch)) == 0)
        ):
            if not eval_datasets:
                _log(f"[eval] epoch={epoch+1}: skipped (no eval dataset enabled)")
                continue
            _run_periodic_eval_for_epoch(
                eval_epoch=int(epoch + 1),
                model=model,
                eval_datasets=eval_datasets,
                args=args,
                backbone=backbone,
                device=device,
                out_dir=out_dir,
                eval_history_rows=eval_history_rows,
                log_fn=_log,
            )
            _write_eval_artifacts(out_dir, eval_bounds, eval_history_rows)
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break
    pbar.close()

    if int(args.epochs) == 0 or start_epoch >= int(args.epochs):
        _save_checkpoint(out_dir / "checkpoints" / "latest.pt", max(0, start_epoch - 1))
    best = min(rows, key=lambda x: float(x.get("loss_total", float("inf")))) if rows else {}
    summary = {
        "global_steps": global_step,
        "elapsed_seconds": time.perf_counter() - t0_train,
        "best_train_loss_total": best.get("loss_total"),
        "best_train_step": best.get("global_step"),
        "num_scenes": len(scenes),
        "cache_dirs": [str(p) for p in cache_dirs],
        "drop_indices": sorted(set(int(x) for x in prune_plan.drop)),
        "lora_indices": lora_unit_indices,
        "train_unit_indices": train_unit_indices,
        "training_method": training_method,
        "prune_plan_file": str(args.prune_plan_file),
        "prune_plan_name": str(args.prune_plan_name),
        "recovery_plan_name": str(args.recovery_plan_name),
        "variant": args.variant,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_curves(out_dir, rows)
    if eval_bounds and eval_history_rows:
        _write_eval_artifacts(out_dir, eval_bounds, eval_history_rows)
    _log(f"[done] out_dir={out_dir} steps={global_step}")
    log_file.close()


if __name__ == "__main__":
    main()

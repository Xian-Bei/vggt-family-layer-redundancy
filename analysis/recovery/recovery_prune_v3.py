"""Training-free boundary recovery for pruned LaRe models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.image_batch import prepare_cache_batch
from analysis.prune_plan import (
    PruneAction,
    UnitPrunePlan,
    load_named_prune_plan,
    prune_plan_bundle_path,
    unit_index_to_kind_layer,
    unit_index_to_unit_id,
)
from analysis.recovery.operators import (
    apply_recovery_weights,
    fuse_replaceme_transform,
    install_recovery_modules,
    recovery_module_name,
)
from analysis.redundancy.cache_layout import load_scene_replay_layer
from analysis.redundancy.cache_layout_v3 import (
    load_scene_metadata_v3,
    load_v3_manifest,
    manifest_scenes_for_training,
)
from analysis.redundancy.cache_redundancy_v3 import forward_cache_with_auxiliaries
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir
from analysis.prune_plan.recovery_plan import (
    TRAINING_FREE_RECOVERY_METHODS,
    compose_recovery_unit_prune_plan,
    load_recovery_plan_bundle,
)
from analysis.redundancy.metrics import linear_cka_value_tensor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fit training-free recovery operators for a pruned model.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--prune-plan-root", type=Path, required=True)
    p.add_argument("--prune-plan-name", type=str, required=True)
    p.add_argument("--recovery-plan-root", type=Path, default=None)
    p.add_argument("--recovery-plan-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--ghost-cos-epochs", type=int, default=10)
    p.add_argument("--ghost-cos-more-epochs", type=int, default=30)
    p.add_argument("--ghost-cos-batch-size", type=int, default=1024)
    p.add_argument("--ghost-cos-lr", type=float, default=1e-4)
    p.add_argument("--ghost-cos-seed", type=int, default=0)
    p.add_argument("--ghost-more-cos-weight", type=float, default=0.04)
    p.add_argument("--ghost-more-cka-weight", type=float, default=0.10)
    p.add_argument("--ghost-more-ema-beta", type=float, default=0.95)
    p.add_argument("--ghost-more-ema-cos-min-weight", type=float, default=0.005)
    p.add_argument("--ghost-more-ema-cos-max-weight", type=float, default=0.20)
    p.add_argument("--ghost-more-ema-cka-min-weight", type=float, default=0.01)
    p.add_argument("--ghost-more-ema-cka-max-weight", type=float, default=1.0)
    p.add_argument("--ghost-mse-cka-epochs", type=int, default=30)
    p.add_argument("--ghost-mse-cka-lr", type=float, default=1e-5)
    p.add_argument("--ghost-mse-cka-weight", type=float, default=1.0)
    p.add_argument("--ghost-mse-cka-budget", type=float, default=0.01)
    p.add_argument(
        "--ghost-mse-cka-max-delta-ratio",
        type=float,
        default=0.0,
        help="Optional Frobenius trust radius ||W-W_ls||/||W_ls|| (0 disables projection).",
    )
    p.add_argument(
        "--ghost-mse-cka-gradient-mode",
        choices=("scene", "domain_agreement"),
        default="scene",
        help=(
            "Use the planned per-scene Adam updates, or retain only gradient "
            "coordinates whose signs agree across all training datasets."
        ),
    )
    p.add_argument("--ghost-mse-cka-seed", type=int, default=0)
    p.add_argument("--ghost-cka-only-epochs", type=int, default=60)
    p.add_argument("--ghost-cka-only-lr", type=float, default=1e-5)
    p.add_argument(
        "--ghost-cka-only-snapshot-epochs",
        type=str,
        default="5,20,60",
    )
    p.add_argument("--ghost-cka-only-seed", type=int, default=0)
    p.add_argument(
        "--ghost-cka-ridge-grid",
        type=str,
        default="0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2",
    )
    p.add_argument("--replaceme-ls-alpha", type=float, default=0.0)
    p.add_argument("--replaceme-cos-epochs", type=int, default=10)
    p.add_argument("--replaceme-cos-more-epochs", type=int, default=30)
    p.add_argument("--replaceme-cos-batch-size", type=int, default=1024)
    p.add_argument("--replaceme-cos-lr", type=float, default=1e-4)
    p.add_argument("--replaceme-seed", type=int, default=0)
    p.add_argument("--prunecomp-eps", type=float, default=1e-8)
    return p


def _parse_name_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").replace(",", " ").split() if x.strip()]


def _load_training_scenes(cache_dirs: Sequence[Path], max_scenes: int) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    for cache_dir in cache_dirs:
        manifest = load_v3_manifest(cache_dir)
        for item in manifest_scenes_for_training(manifest):
            item = dict(item)
            # The legacy training view intentionally contains only tensor-file
            # references. Progressive recovery additionally needs the exact
            # frame ids/paths and geometry saved in per-scene metadata.
            item.update(load_scene_metadata_v3(cache_dir, item))
            item["_cache_dir"] = str(cache_dir)
            scenes.append(item)
            if int(max_scenes) > 0 and len(scenes) >= int(max_scenes):
                return scenes
    return scenes


def _load_unit_tensor(
    cache_dir: Path,
    item: Dict[str, Any],
    unit_id: str,
    *,
    num_layers: int,
    map_location: Optional[str] = None,
) -> torch.Tensor:
    key = str(unit_id)
    kind, idx_s = key.split("_", 1)
    idx = int(idx_s)
    concat = load_scene_replay_layer(
        cache_dir,
        item,
        idx,
        expected_num_layers=int(num_layers),
        map_location=map_location,
    )
    c = int(concat.shape[-1] // 2)
    if kind == "frame":
        return concat[..., :c].contiguous()
    if kind == "global":
        return concat[..., c:].contiguous()
    raise ValueError(f"Unsupported unit id {unit_id!r}")


def _flatten(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 4:
        raise ValueError(f"expected 4D unit tensor [B,S,P,C], got {tuple(t.shape)}")
    return t.reshape(-1, int(t.shape[-1])).to(torch.float32)


def _align_replaceme_activations(
    h: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    *,
    scene_id: str,
    source_unit_id: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align boundary activations with the tokens updated by the source MLP.

    Some backbones run an inter-frame block on only the leading camera/register
    tokens while carrying patch tokens through unchanged.  The cached source and
    target activations retain all tokens, but the captured MLP residual naturally
    contains only that leading subset.
    """
    if h.dim() != 4 or y.dim() != 4 or r.dim() != 4:
        raise RuntimeError(
            f"ReplaceMe expects [B,S,P,C] activations for {scene_id}: "
            f"H={tuple(h.shape)} Y={tuple(y.shape)} R={tuple(r.shape)}"
        )
    if h.shape != y.shape:
        raise RuntimeError(
            f"ReplaceMe boundary mismatch for {scene_id}: H={tuple(h.shape)} Y={tuple(y.shape)}"
        )
    if h.shape == r.shape:
        return h, y, r
    same_layout = (
        int(h.shape[0]) == int(r.shape[0])
        and int(h.shape[1]) == int(r.shape[1])
        and int(h.shape[3]) == int(r.shape[3])
    )
    is_prefix_residual = (
        str(source_unit_id).startswith("global_")
        and same_layout
        and 0 < int(r.shape[2]) < int(h.shape[2])
    )
    if is_prefix_residual:
        token_count = int(r.shape[2])
        return h[:, :, :token_count, :], y[:, :, :token_count, :], r
    raise RuntimeError(
        f"ReplaceMe activation mismatch for {scene_id}: "
        f"H={tuple(h.shape)} Y={tuple(y.shape)} R={tuple(r.shape)}"
    )


def _load_auxiliary_tensor(
    cache_dir: Path,
    item: Dict[str, Any],
    auxiliary_id: str,
    *,
    map_location: Optional[str] = None,
) -> torch.Tensor:
    files = item.get("auxiliary_files")
    rel = str(files.get(auxiliary_id, "")).strip() if isinstance(files, dict) else ""
    if not rel:
        raise FileNotFoundError(
            f"Scene {item.get('scene_id')} lacks {auxiliary_id!r}; rerun stage 8 cache "
            "with the ReplaceMe recovery plan selected"
        )
    path = cache_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing recovery auxiliary tensor {path}")
    kwargs: Dict[str, Any] = {"weights_only": False}
    if map_location is not None:
        kwargs["map_location"] = map_location
    tensor = torch.load(path, **kwargs)
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
        raise ValueError(f"Expected [B,S,P,C] tensor at {path}, got {type(tensor)}")
    return tensor.contiguous()


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard(size: int, *, device: torch.device) -> torch.Tensor:
    if not _is_pow2(int(size)):
        raise ValueError(f"LinearPatch requires power-of-two channels, got {size}")
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    while int(h.shape[0]) < int(size):
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    return h / float(size) ** 0.5


def _pair_records(
    bundle: Dict[str, Any], prune_plan: UnitPrunePlan, recovery_plan_name: str
) -> Tuple[str, List[Dict[str, Any]]]:
    method = str(bundle.get("method", "")).strip().lower()
    if method not in TRAINING_FREE_RECOVERY_METHODS:
        raise ValueError(f"{recovery_plan_name} is not training-free recovery: method={method!r}")
    extra = prune_plan.extra
    boundaries = extra.get("recovery_boundaries")
    if isinstance(boundaries, list) and boundaries:
        records = [dict(item) for item in boundaries if isinstance(item, dict)]
        if len(records) != len(boundaries):
            raise ValueError("Recovery boundaries must contain only objects")
        return method, records
    # Backward compatibility for checkpoints/plans written before multi-boundary
    # recovery was introduced.
    return method, [
        {
            "source_unit_index": int(extra["recovery_source_unit_index"]),
            "source_unit_id": str(extra["recovery_source_unit_id"]),
            "target_unit_index": int(extra["recovery_target_unit_index"]),
            "target_unit_id": str(extra["recovery_target_unit_id"]),
        }
    ]


def _fit_boundary_weight(
    *,
    method: str,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    ridge: float,
) -> torch.Tensor:
    xtx: Optional[torch.Tensor] = None
    xtd: Optional[torch.Tensor] = None
    sum_x_abs: Optional[torch.Tensor] = None
    sum_y_abs: Optional[torch.Tensor] = None
    count = 0
    hmat: Optional[torch.Tensor] = None
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        map_loc = str(device) if device.type == "cuda" else "cpu"
        x = _flatten(_load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location=map_loc)).to(device)
        y = _flatten(_load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location=map_loc)).to(device)
        if x.shape != y.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(x.shape)} {target_unit_id}={tuple(y.shape)}"
            )
        if method == "recovery_ghost":
            x64 = x.to(torch.float64)
            delta64 = (y - x).to(torch.float64)
            xtx = x64.T @ x64 if xtx is None else xtx + x64.T @ x64
            xtd = x64.T @ delta64 if xtd is None else xtd + x64.T @ delta64
        else:
            xx = x
            yy = y
            if method == "recovery_linearpatch":
                if hmat is None:
                    hmat = _hadamard(int(x.shape[-1]), device=device)
                xx = xx @ hmat
                yy = yy @ hmat
            sx = xx.abs().sum(dim=0)
            sy = yy.abs().sum(dim=0)
            sum_x_abs = sx if sum_x_abs is None else sum_x_abs + sx
            sum_y_abs = sy if sum_y_abs is None else sum_y_abs + sy
            count += int(xx.shape[0])
        del x, y

    if method == "recovery_ghost":
        if xtx is None or xtd is None:
            raise RuntimeError("No activation rows were available for ghost recovery")
        return _solve_ghost_from_statistics(
            xtx, xtd, ridge=float(ridge)
        ).cpu()

    if sum_x_abs is None or sum_y_abs is None or count <= 0:
        raise RuntimeError(f"No activation rows were available for {method}")
    scale = sum_y_abs / (sum_x_abs + 1e-8)
    assert hmat is not None
    return (hmat @ torch.diag(scale.to(torch.float32)) @ hmat.T).to(torch.float32).cpu()


def _load_boundary_scene_rows(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        x = _flatten(
            _load_unit_tensor(
                cache_dir,
                item,
                source_unit_id,
                num_layers=num_layers,
                map_location="cpu",
            )
        )
        y = _flatten(
            _load_unit_tensor(
                cache_dir,
                item,
                target_unit_id,
                num_layers=num_layers,
                map_location="cpu",
            )
        )
        if x.shape != y.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(x.shape)} {target_unit_id}={tuple(y.shape)}"
            )
        dataset_name = cache_dir.parent.name
        scene_key = f"{dataset_name}/{item.get('scene_id', '')}"
        pairs.append((scene_key, x.contiguous(), y.contiguous()))
    if not pairs:
        raise RuntimeError("No activation rows were available for Ghosted recovery")
    return pairs


def _ghost_sufficient_statistics(
    pairs: Sequence[Tuple[str, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xtx: Optional[torch.Tensor] = None
    xtd: Optional[torch.Tensor] = None
    for _scene_id, x_cpu, y_cpu in pairs:
        x = x_cpu.to(device=device, dtype=torch.float64)
        delta = y_cpu.to(device=device, dtype=torch.float64) - x
        xx = x.T @ x
        xd = x.T @ delta
        xtx = xx if xtx is None else xtx + xx
        xtd = xd if xtd is None else xtd + xd
        del x, delta, xx, xd
    assert xtx is not None and xtd is not None
    return xtx, xtd


def _solve_ghost_from_statistics(
    xtx: torch.Tensor,
    xtd: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    channels = int(xtx.shape[0])
    reg = float(ridge) * torch.eye(channels, device=xtx.device, dtype=torch.float64)
    delta = torch.linalg.solve(xtx + reg, xtd)
    return (
        torch.eye(channels, device=xtx.device, dtype=torch.float64) + delta
    ).to(torch.float32)


def _evaluate_boundary_weight(
    weight: torch.Tensor,
    pairs: Sequence[Tuple[str, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    with_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    squared_error = torch.zeros((), device=device, dtype=torch.float64)
    num_values = 0
    cka_sum = torch.zeros((), device=device, dtype=torch.float64)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        for _scene_id, x_cpu, y_cpu in pairs:
            x = x_cpu.to(device=device, dtype=torch.float32)
            y = y_cpu.to(device=device, dtype=torch.float32)
            prediction = x @ weight
            squared_error = squared_error + (prediction - y).square().sum(dtype=torch.float64)
            num_values += int(y.numel())
            cka_sum = cka_sum + linear_cka_value_tensor(prediction, y).to(torch.float64)
    if num_values <= 0:
        raise RuntimeError("Cannot evaluate an empty boundary activation collection")
    return squared_error / float(num_values), cka_sum / float(len(pairs))


def _fit_ghost_mse_cka(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    ridge: float,
    epochs: int,
    lr: float,
    cka_weight: float,
    mse_budget: float,
    max_delta_ratio: float,
    gradient_mode: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if (
        epochs <= 0
        or lr <= 0
        or cka_weight < 0
        or mse_budget < 0
        or max_delta_ratio < 0
    ):
        raise ValueError("Ghost MSE+CKA epochs/lr must be positive and weights/budget non-negative")
    if gradient_mode not in ("scene", "domain_agreement"):
        raise ValueError(f"Unsupported Ghost MSE+CKA gradient mode {gradient_mode!r}")
    pairs = _load_boundary_scene_rows(
        source_unit_id=source_unit_id,
        target_unit_id=target_unit_id,
        scenes=scenes,
        num_layers=num_layers,
    )
    xtx, xtd = _ghost_sufficient_statistics(pairs, device=device)
    baseline = _solve_ghost_from_statistics(xtx, xtd, ridge=float(ridge))
    baseline_mse_t, baseline_cka_t = _evaluate_boundary_weight(
        baseline, pairs, device=device
    )
    baseline_mse = float(baseline_mse_t.item())
    baseline_cka = float(baseline_cka_t.item())
    if baseline_mse <= 0.0:
        return baseline.detach().cpu(), {
            "fallback": True,
            "fallback_reason": "zero_baseline_mse",
            "baseline_mse": baseline_mse,
            "baseline_cka": baseline_cka,
            "history": [],
        }

    cka_loss_scale = max(1.0 - baseline_cka, 1e-6)
    transform = baseline.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([transform], lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_weight = baseline.detach().clone()
    best_cka = baseline_cka
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    baseline_norm = baseline.norm().clamp_min(torch.finfo(torch.float32).eps)
    domain_indices: Dict[str, List[int]] = {}
    for pair_index, (scene_key, _x, _y) in enumerate(pairs):
        domain_name = scene_key.split("/", 1)[0]
        domain_indices.setdefault(domain_name, []).append(pair_index)

    for epoch in range(1, int(epochs) + 1):
        loss_sum = 0.0
        agreement_fraction = 1.0
        if gradient_mode == "scene":
            order = torch.randperm(len(pairs), generator=generator).tolist()
            update_groups = [[int(pair_index)] for pair_index in order]
        else:
            domain_names = list(domain_indices)
            domain_order = torch.randperm(
                len(domain_names), generator=generator
            ).tolist()
            update_groups = [
                domain_indices[domain_names[int(domain_position)]]
                for domain_position in domain_order
            ]

        if gradient_mode == "scene":
            for indices in update_groups:
                pair_index = indices[0]
                _scene_id, x_cpu, y_cpu = pairs[pair_index]
                x = x_cpu.to(device=device, dtype=torch.float32)
                y = y_cpu.to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                prediction = x @ transform
                mse = (prediction - y).square().mean()
                cka_loss = 1.0 - linear_cka_value_tensor(prediction, y)
                loss = mse / float(baseline_mse) + float(
                    cka_weight
                ) * cka_loss / float(cka_loss_scale)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Ghost MSE+CKA produced a non-finite loss")
                loss.backward()
                optimizer.step()
                if float(max_delta_ratio) > 0.0:
                    with torch.no_grad():
                        delta = transform - baseline
                        delta_ratio = delta.norm() / baseline_norm
                        if float(delta_ratio.item()) > float(max_delta_ratio):
                            transform.copy_(
                                baseline
                                + delta
                                * (float(max_delta_ratio) / float(delta_ratio.item()))
                            )
                loss_sum += float(loss.detach().item())
        else:
            domain_gradients: List[torch.Tensor] = []
            for indices in update_groups:
                optimizer.zero_grad(set_to_none=True)
                for pair_index in indices:
                    _scene_id, x_cpu, y_cpu = pairs[pair_index]
                    x = x_cpu.to(device=device, dtype=torch.float32)
                    y = y_cpu.to(device=device, dtype=torch.float32)
                    prediction = x @ transform
                    mse = (prediction - y).square().mean()
                    cka_loss = 1.0 - linear_cka_value_tensor(prediction, y)
                    loss = mse / float(baseline_mse) + float(
                        cka_weight
                    ) * cka_loss / float(cka_loss_scale)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Ghost MSE+CKA produced a non-finite loss")
                    (loss / float(len(indices))).backward()
                    loss_sum += float(loss.detach().item())
                assert transform.grad is not None
                domain_gradients.append(transform.grad.detach().clone())

            optimizer.zero_grad(set_to_none=True)
            stacked_gradients = torch.stack(domain_gradients, dim=0)
            positive = (stacked_gradients > 0).all(dim=0)
            negative = (stacked_gradients < 0).all(dim=0)
            agreement = positive | negative
            agreement_fraction = float(agreement.to(torch.float32).mean().item())
            transform.grad = stacked_gradients.mean(dim=0) * agreement
            optimizer.step()
            if float(max_delta_ratio) > 0.0:
                with torch.no_grad():
                    delta = transform - baseline
                    delta_ratio = delta.norm() / baseline_norm
                    if float(delta_ratio.item()) > float(max_delta_ratio):
                        transform.copy_(
                            baseline
                            + delta
                            * (float(max_delta_ratio) / float(delta_ratio.item()))
                        )

        epoch_mse_t, epoch_cka_t = _evaluate_boundary_weight(
            transform, pairs, device=device
        )
        epoch_mse = float(epoch_mse_t.item())
        epoch_cka = float(epoch_cka_t.item())
        mse_ratio = epoch_mse / baseline_mse
        feasible = mse_ratio <= 1.0 + float(mse_budget) + 1e-12
        if feasible and epoch_cka > best_cka:
            best_weight = transform.detach().clone()
            best_cka = epoch_cka
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "mean_step_loss": loss_sum / float(len(pairs)),
                "mse": epoch_mse,
                "mse_ratio": mse_ratio,
                "cka": epoch_cka,
                "feasible": feasible,
                "gradient_agreement_fraction": agreement_fraction,
            }
        )

    final_mse_t, final_cka_t = _evaluate_boundary_weight(
        best_weight, pairs, device=device
    )
    delta_ratio = float(
        (
            (best_weight - baseline).norm()
            / baseline.norm().clamp_min(torch.finfo(torch.float32).eps)
        ).item()
    )
    fallback = best_epoch == 0
    return best_weight.detach().cpu(), {
        "epochs": int(epochs),
        "lr": float(lr),
        "cka_weight": float(cka_weight),
        "mse_budget": float(mse_budget),
        "max_delta_ratio": float(max_delta_ratio),
        "gradient_mode": gradient_mode,
        "domains": {name: len(indices) for name, indices in domain_indices.items()},
        "seed": int(seed),
        "num_scenes": len(pairs),
        "num_rows": sum(int(x.shape[0]) for _sid, x, _y in pairs),
        "baseline_mse": baseline_mse,
        "baseline_cka": baseline_cka,
        "selected_epoch": best_epoch,
        "selected_mse": float(final_mse_t.item()),
        "selected_mse_ratio": float(final_mse_t.item()) / baseline_mse,
        "selected_cka": float(final_cka_t.item()),
        "weight_delta_ratio": delta_ratio,
        "fallback": fallback,
        "fallback_reason": "no_feasible_cka_improvement" if fallback else "",
        "history": history,
    }


def _parse_epoch_grid(raw: str, *, epochs: int) -> List[int]:
    values = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if any(value <= 0 or value > int(epochs) for value in values):
        raise ValueError(
            f"Snapshot epochs must be in [1, {int(epochs)}], got {raw!r}"
        )
    out: List[int] = []
    for value in values:
        if value not in out:
            out.append(value)
    if int(epochs) not in out:
        out.append(int(epochs))
    return sorted(out)


def _fit_ghost_cka_only(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    ridge: float,
    epochs: int,
    lr: float,
    snapshot_epochs: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[int, torch.Tensor]]:
    """Optimize scene-mean boundary CKA from the Ghosted LS solution.

    MSE is recorded only as a diagnostic. It is neither part of the loss nor
    used for checkpoint selection.
    """
    if epochs <= 0 or lr <= 0:
        raise ValueError("Ghost CKA-only epochs and learning rate must be positive")
    requested_snapshots = _parse_epoch_grid(snapshot_epochs, epochs=int(epochs))
    pairs = _load_boundary_scene_rows(
        source_unit_id=source_unit_id,
        target_unit_id=target_unit_id,
        scenes=scenes,
        num_layers=num_layers,
    )
    xtx, xtd = _ghost_sufficient_statistics(pairs, device=device)
    baseline = _solve_ghost_from_statistics(xtx, xtd, ridge=float(ridge))
    baseline_mse_t, baseline_cka_t = _evaluate_boundary_weight(
        baseline, pairs, device=device
    )
    baseline_mse = float(baseline_mse_t.item())
    baseline_cka = float(baseline_cka_t.item())
    cka_loss_scale = max(1.0 - baseline_cka, 1e-6)
    transform = baseline.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([transform], lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    baseline_norm = baseline.norm().clamp_min(torch.finfo(torch.float32).eps)
    best_weight = baseline.detach().clone()
    best_cka = baseline_cka
    best_epoch = 0
    snapshots: Dict[int, torch.Tensor] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        loss_sum = 0.0
        order = torch.randperm(len(pairs), generator=generator).tolist()
        for pair_index in order:
            _scene_id, x_cpu, y_cpu = pairs[int(pair_index)]
            x = x_cpu.to(device=device, dtype=torch.float32)
            y = y_cpu.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = x @ transform
            cka_loss = 1.0 - linear_cka_value_tensor(prediction, y)
            loss = cka_loss / float(cka_loss_scale)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Ghost CKA-only produced a non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item())

        epoch_mse_t, epoch_cka_t = _evaluate_boundary_weight(
            transform, pairs, device=device
        )
        epoch_mse = float(epoch_mse_t.item())
        epoch_cka = float(epoch_cka_t.item())
        delta_ratio = float(((transform - baseline).norm() / baseline_norm).item())
        if epoch_cka > best_cka:
            best_weight = transform.detach().clone()
            best_cka = epoch_cka
            best_epoch = epoch
        if epoch in requested_snapshots:
            snapshots[epoch] = transform.detach().cpu().to(torch.float32)
        history.append(
            {
                "epoch": epoch,
                "mean_step_loss": loss_sum / float(len(pairs)),
                "mse": epoch_mse,
                "mse_ratio": epoch_mse / baseline_mse,
                "cka": epoch_cka,
                "weight_delta_ratio": delta_ratio,
            }
        )

    selected_mse_t, selected_cka_t = _evaluate_boundary_weight(
        best_weight, pairs, device=device
    )
    selected_delta_ratio = float(
        ((best_weight - baseline).norm() / baseline_norm).item()
    )
    return best_weight.detach().cpu(), {
        "epochs": int(epochs),
        "lr": float(lr),
        "objective": "scene_mean_linear_cka_only",
        "mse_used_for_optimization": False,
        "mse_used_for_selection": False,
        "snapshot_epochs": requested_snapshots,
        "seed": int(seed),
        "num_scenes": len(pairs),
        "num_rows": sum(int(x.shape[0]) for _sid, x, _y in pairs),
        "baseline_mse": baseline_mse,
        "baseline_cka": baseline_cka,
        "selected_epoch": best_epoch,
        "selected_mse": float(selected_mse_t.item()),
        "selected_mse_ratio": float(selected_mse_t.item()) / baseline_mse,
        "selected_cka": float(selected_cka_t.item()),
        "weight_delta_ratio": selected_delta_ratio,
        "fallback": best_epoch == 0,
        "fallback_reason": "no_cka_improvement" if best_epoch == 0 else "",
        "history": history,
    }, snapshots


def _parse_float_grid(raw: str) -> List[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"Expected a non-empty non-negative float grid, got {raw!r}")
    out: List[float] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _fit_ghost_cka_ridge(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    ridge: float,
    ridge_grid: str,
    mse_budget: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if ridge <= 0 or mse_budget < 0:
        raise ValueError("Ghost CKA ridge and MSE budget must be positive/non-negative")
    pairs = _load_boundary_scene_rows(
        source_unit_id=source_unit_id,
        target_unit_id=target_unit_id,
        scenes=scenes,
        num_layers=num_layers,
    )
    xtx, xtd = _ghost_sufficient_statistics(pairs, device=device)
    baseline = _solve_ghost_from_statistics(xtx, xtd, ridge=float(ridge))
    baseline_mse_t, baseline_cka_t = _evaluate_boundary_weight(
        baseline, pairs, device=device
    )
    baseline_mse = float(baseline_mse_t.item())
    baseline_cka = float(baseline_cka_t.item())
    trace_scale = float((torch.trace(xtx) / float(xtx.shape[0])).item())
    candidates: List[Dict[str, Any]] = []
    selected_weight = baseline
    selected_cka = baseline_cka
    selected_mse = baseline_mse
    selected_rho = 0.0
    selected_ridge = float(ridge)

    for rho in _parse_float_grid(ridge_grid):
        absolute_ridge = float(ridge) if rho == 0.0 else max(
            float(ridge), float(rho) * trace_scale
        )
        weight = _solve_ghost_from_statistics(xtx, xtd, ridge=absolute_ridge)
        mse_t, cka_t = _evaluate_boundary_weight(weight, pairs, device=device)
        mse = float(mse_t.item())
        cka = float(cka_t.item())
        mse_ratio = mse / baseline_mse if baseline_mse > 0 else 1.0
        feasible = mse_ratio <= 1.0 + float(mse_budget) + 1e-12
        candidates.append(
            {
                "rho": float(rho),
                "absolute_ridge": absolute_ridge,
                "mse": mse,
                "mse_ratio": mse_ratio,
                "cka": cka,
                "feasible": feasible,
            }
        )
        # CKA is accumulated in float32; changes below 1e-7 are numerical
        # ties, not evidence that a different ridge is preferable.
        better = cka > selected_cka + 1e-7
        tied = abs(cka - selected_cka) <= 1e-7
        if feasible and (
            better
            or (tied and mse < selected_mse * (1.0 - 1e-8))
            or (
                tied
                and abs(mse - selected_mse) <= max(abs(selected_mse) * 1e-8, 1e-15)
                and absolute_ridge < selected_ridge
            )
        ):
            selected_weight = weight
            selected_cka = cka
            selected_mse = mse
            selected_rho = float(rho)
            selected_ridge = absolute_ridge

    fallback = selected_rho == 0.0
    delta_ratio = float(
        (
            (selected_weight - baseline).norm()
            / baseline.norm().clamp_min(torch.finfo(torch.float32).eps)
        ).item()
    )
    return selected_weight.detach().cpu(), {
        "num_scenes": len(pairs),
        "num_rows": sum(int(x.shape[0]) for _sid, x, _y in pairs),
        "mse_budget": float(mse_budget),
        "trace_scale": trace_scale,
        "baseline_ridge": float(ridge),
        "baseline_mse": baseline_mse,
        "baseline_cka": baseline_cka,
        "selected_rho": selected_rho,
        "selected_ridge": selected_ridge,
        "selected_mse": selected_mse,
        "selected_mse_ratio": selected_mse / baseline_mse if baseline_mse > 0 else 1.0,
        "selected_cka": selected_cka,
        "weight_delta_ratio": delta_ratio,
        "fallback": fallback,
        "fallback_reason": "baseline_won_cka_mse_selection" if fallback else "",
        "candidates": candidates,
    }


def _fit_replaceme_ls(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    rtr: Optional[torch.Tensor] = None
    rtd: Optional[torch.Tensor] = None
    rows = 0
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        map_loc = str(device) if device.type == "cuda" else "cpu"
        h_raw = _load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location=map_loc)
        y_raw = _load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location=map_loc)
        r_raw = _load_auxiliary_tensor(cache_dir, item, auxiliary_id, map_location=map_loc)
        h_raw, y_raw, r_raw = _align_replaceme_activations(
            h_raw,
            y_raw,
            r_raw,
            scene_id=str(item.get("scene_id", "<unknown>")),
            source_unit_id=source_unit_id,
        )
        h = _flatten(h_raw).to(device)
        y = _flatten(y_raw).to(device)
        r = _flatten(r_raw).to(device)
        r64 = r.to(torch.float64)
        d64 = (y + r - h).to(torch.float64)
        rtr = r64.T @ r64 if rtr is None else rtr + r64.T @ r64
        rtd = r64.T @ d64 if rtd is None else rtd + r64.T @ d64
        rows += int(r.shape[0])
        del h, y, r, r64, d64
    if rtr is None or rtd is None or rows <= 0:
        raise RuntimeError("No activation rows were available for ReplaceMe LS")
    channels = int(rtr.shape[0])
    reg = float(alpha) * torch.eye(channels, device=device, dtype=torch.float64)
    transform = torch.linalg.solve(rtr + reg, rtd).to(torch.float32).cpu()
    return transform, {"num_rows": rows, "ls_alpha": float(alpha)}


def _optimize_cosine_transform(
    *,
    input_rows: Sequence[torch.Tensor],
    desired_rows: Sequence[torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    method_name: str,
    restore_global_scale: bool = False,
    loss_components: Sequence[str] = ("cosine",),
    loss_weights: Optional[Dict[str, float]] = None,
    loss_balance_targets: Optional[Dict[str, float]] = None,
    loss_balance_ema_beta: float = 0.95,
    loss_balance_weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError(f"{method_name} epochs, batch size, and learning rate must be positive")
    if not input_rows or len(input_rows) != len(desired_rows):
        raise RuntimeError(f"No paired activation rows were available for {method_name}")
    all_x = torch.cat(list(input_rows), dim=0)
    all_y = torch.cat(list(desired_rows), dim=0)
    if all_x.shape != all_y.shape:
        raise RuntimeError(f"{method_name} activation mismatch: {tuple(all_x.shape)} vs {tuple(all_y.shape)}")
    component_names = tuple(str(name).strip().lower() for name in loss_components)
    allowed_components = {"mse", "cosine", "cka"}
    if (
        not component_names
        or len(set(component_names)) != len(component_names)
        or any(name not in allowed_components for name in component_names)
    ):
        raise ValueError(
            f"{method_name} expected unique loss components from "
            f"{sorted(allowed_components)}, got {component_names}"
        )
    component_weights = {
        name: float((loss_weights or {}).get(name, 1.0))
        for name in component_names
    }
    unexpected_weight_names = sorted(set(loss_weights or {}) - set(component_names))
    if unexpected_weight_names or any(
        not torch.isfinite(torch.tensor(weight)) or weight <= 0.0
        for weight in component_weights.values()
    ):
        raise ValueError(
            f"{method_name} expected finite positive weights for exactly "
            f"{component_names}, got {loss_weights}; unexpected={unexpected_weight_names}"
        )
    balance_targets = {
        str(name).strip().lower(): float(target)
        for name, target in (loss_balance_targets or {}).items()
    }
    if balance_targets:
        if loss_weights is not None:
            raise ValueError(
                f"{method_name} cannot combine explicit loss weights with EMA balancing"
            )
        if "mse" not in component_names or "mse" in balance_targets:
            raise ValueError(
                f"{method_name} EMA balancing requires mse as the fixed reference "
                "and only auxiliary targets"
            )
        if set(balance_targets) != set(component_names) - {"mse"}:
            raise ValueError(
                f"{method_name} EMA balancing needs one target for every auxiliary "
                f"component; components={component_names}, targets={balance_targets}"
            )
        if any(
            not torch.isfinite(torch.tensor(target)) or target <= 0.0
            for target in balance_targets.values()
        ):
            raise ValueError(
                f"{method_name} expected finite positive EMA targets, got "
                f"{balance_targets}"
            )
        if not (0.0 <= float(loss_balance_ema_beta) < 1.0):
            raise ValueError(
                f"{method_name} EMA beta must be in [0, 1), got "
                f"{loss_balance_ema_beta}"
            )
    balance_weight_bounds = {
        str(name).strip().lower(): (float(bounds[0]), float(bounds[1]))
        for name, bounds in (loss_balance_weight_bounds or {}).items()
    }
    if balance_targets:
        if set(balance_weight_bounds) != set(balance_targets):
            raise ValueError(
                f"{method_name} EMA balancing needs bounds for every auxiliary "
                f"component; targets={balance_targets}, bounds={balance_weight_bounds}"
            )
        if any(
            not torch.isfinite(torch.tensor(lower))
            or not torch.isfinite(torch.tensor(upper))
            or lower <= 0.0
            or upper < lower
            for lower, upper in balance_weight_bounds.values()
        ):
            raise ValueError(
                f"{method_name} expected finite positive ordered EMA weight bounds, "
                f"got {balance_weight_bounds}"
            )

    def component_values(
        prediction: torch.Tensor, desired: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        values: Dict[str, torch.Tensor] = {}
        if "mse" in component_names:
            values["mse"] = (prediction - desired).square().mean()
        if "cosine" in component_names:
            prediction_norm = prediction / prediction.norm(dim=1, keepdim=True)
            desired_norm = desired / desired.norm(dim=1, keepdim=True)
            values["cosine"] = 1.0 - (
                prediction_norm * desired_norm
            ).sum(dim=1).mean()
        if "cka" in component_names:
            values["cka"] = 1.0 - linear_cka_value_tensor(prediction, desired)
        return values

    channels = int(all_x.shape[-1])
    transform = torch.eye(channels, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([transform], lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    num_rows = int(all_x.shape[0])
    rows_seen = 0
    optimizer_steps = 0
    final_loss = float("nan")
    final_component_losses: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    probe_rows = min(num_rows, int(batch_size))
    probe_x = all_x[:probe_rows].to(device=device, dtype=torch.float32)
    probe_y = all_y[:probe_rows].to(device=device, dtype=torch.float32)
    initial_component_values: Dict[str, float] = {}
    initial_gradient_norms: Dict[str, float] = {}
    initial_weighted_gradient_norms: Dict[str, float] = {}
    initial_effective_weights = dict(component_weights)
    for component_name in component_names:
        probe_transform = torch.eye(
            channels, device=device, dtype=torch.float32, requires_grad=True
        )
        probe_prediction = probe_x @ probe_transform
        component = component_values(probe_prediction, probe_y)[component_name]
        gradient = torch.autograd.grad(component, probe_transform)[0]
        initial_component_values[component_name] = float(component.detach().item())
        initial_gradient_norms[component_name] = float(gradient.norm().item())
    if balance_targets:
        initial_mse = initial_component_values["mse"]
        for component_name, target in balance_targets.items():
            lower, upper = balance_weight_bounds[component_name]
            initial_effective_weights[component_name] = min(
                upper,
                max(
                    lower,
                    target
                    * initial_mse
                    / max(initial_component_values[component_name], 1e-12),
                ),
            )
    for component_name in component_names:
        initial_weighted_gradient_norms[component_name] = (
            initial_gradient_norms[component_name]
            * initial_effective_weights[component_name]
        )

    component_emas: Dict[str, torch.Tensor] = {}
    final_effective_weights = dict(initial_effective_weights)
    clamp_counts = {
        name: {"min": 0, "max": 0} for name in balance_targets
    }
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(num_rows, generator=generator)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_component_sums = {name: 0.0 for name in component_names}
        epoch_weight_sums = {name: 0.0 for name in component_names}
        epoch_weighted_component_sums = {
            name: 0.0 for name in component_names
        }
        for start in range(0, num_rows, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            xb = all_x[indices].to(device=device, dtype=torch.float32)
            yb = all_y[indices].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = xb @ transform
            components = component_values(prediction, yb)
            effective_weights: Dict[str, torch.Tensor] = {
                name: torch.as_tensor(
                    component_weights[name],
                    device=device,
                    dtype=torch.float32,
                )
                for name in component_names
            }
            if balance_targets:
                beta = float(loss_balance_ema_beta)
                for name, value in components.items():
                    detached_value = value.detach()
                    if name not in component_emas:
                        component_emas[name] = detached_value
                    else:
                        component_emas[name] = (
                            beta * component_emas[name]
                            + (1.0 - beta) * detached_value
                        )
                mse_ema = component_emas["mse"]
                for name, target in balance_targets.items():
                    lower, upper = balance_weight_bounds[name]
                    raw_weight = (
                        float(target)
                        * mse_ema
                        / component_emas[name].clamp_min(1e-12)
                    )
                    if float(raw_weight.item()) < lower:
                        clamp_counts[name]["min"] += 1
                    elif float(raw_weight.item()) > upper:
                        clamp_counts[name]["max"] += 1
                    effective_weights[name] = raw_weight.clamp(
                        min=lower, max=upper
                    ).detach()
            loss = sum(
                effective_weights[name] * components[name]
                for name in component_names
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{method_name} produced a non-finite loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
            final_component_losses = {
                name: float(value.detach().item())
                for name, value in components.items()
            }
            epoch_loss_sum += final_loss
            epoch_steps += 1
            for name, value in final_component_losses.items():
                epoch_component_sums[name] += value
                effective_weight = float(effective_weights[name].item())
                final_effective_weights[name] = effective_weight
                epoch_weight_sums[name] += effective_weight
                epoch_weighted_component_sums[name] += (
                    effective_weight * value
                )
            rows_seen += int(xb.shape[0])
            optimizer_steps += 1
        history.append(
            {
                "epoch": epoch,
                "mean_loss": epoch_loss_sum / float(epoch_steps),
                "mean_components": {
                    name: value / float(epoch_steps)
                    for name, value in epoch_component_sums.items()
                },
                "mean_effective_weights": {
                    name: value / float(epoch_steps)
                    for name, value in epoch_weight_sums.items()
                },
                "mean_weighted_components": {
                    name: value / float(epoch_steps)
                    for name, value in epoch_weighted_component_sums.items()
                },
            }
        )

    global_scale = 1.0
    if restore_global_scale:
        numerator = torch.zeros((), device=device, dtype=torch.float64)
        denominator = torch.zeros((), device=device, dtype=torch.float64)
        with torch.no_grad():
            for start in range(0, num_rows, int(batch_size)):
                xb = all_x[start : start + int(batch_size)].to(
                    device=device, dtype=torch.float32
                )
                yb = all_y[start : start + int(batch_size)].to(
                    device=device, dtype=torch.float32
                )
                prediction = xb @ transform
                numerator += (prediction * yb).sum(dtype=torch.float64)
                denominator += prediction.square().sum(dtype=torch.float64)
        if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
            raise FloatingPointError(f"{method_name} produced non-finite scale statistics")
        if float(denominator.item()) <= 0.0:
            raise FloatingPointError(f"{method_name} produced a non-positive scale denominator")
        global_scale = float((numerator / denominator).item())
        if not torch.isfinite(torch.tensor(global_scale)) or global_scale <= 0.0:
            raise FloatingPointError(
                f"{method_name} produced invalid closed-form global scale {global_scale}"
            )
        with torch.no_grad():
            transform.mul_(global_scale)

    result = transform.detach().cpu()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{method_name} produced a non-finite transform")
    return result, {
        "rows_seen": rows_seen,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "loss_components": list(component_names),
        "loss_weights": component_weights,
        "loss_balance": (
            "ema_value_to_mse"
            if balance_targets
            else (
                "raw_equal_coefficients"
                if all(weight == 1.0 for weight in component_weights.values())
                else "explicit_coefficients"
            )
        ),
        "loss_balance_targets": balance_targets,
        "loss_balance_ema_beta": (
            float(loss_balance_ema_beta) if balance_targets else None
        ),
        "loss_balance_weight_bounds": balance_weight_bounds,
        "initial_effective_weights": initial_effective_weights,
        "final_effective_weights": final_effective_weights,
        "loss_balance_clamp_counts": clamp_counts,
        "initial_component_values": initial_component_values,
        "initial_gradient_norms": initial_gradient_norms,
        "initial_weighted_gradient_norms": initial_weighted_gradient_norms,
        "final_loss": final_loss,
        "final_component_losses": final_component_losses,
        "optimizer_steps": optimizer_steps,
        "restore_global_scale": bool(restore_global_scale),
        "global_scale": global_scale,
        "history": history,
    }


def _fit_ghost_cos(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    restore_global_scale: bool = False,
    loss_components: Sequence[str] = ("cosine",),
    loss_weights: Optional[Dict[str, float]] = None,
    loss_balance_targets: Optional[Dict[str, float]] = None,
    loss_balance_ema_beta: float = 0.95,
    loss_balance_weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    method_name: str = "Ghost cosine",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    source_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        x = _flatten(
            _load_unit_tensor(
                cache_dir,
                item,
                source_unit_id,
                num_layers=num_layers,
                map_location="cpu",
            )
        )
        y = _flatten(
            _load_unit_tensor(
                cache_dir,
                item,
                target_unit_id,
                num_layers=num_layers,
                map_location="cpu",
            )
        )
        if x.shape != y.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(x.shape)} {target_unit_id}={tuple(y.shape)}"
            )
        source_rows.append(x)
        target_rows.append(y)
    return _optimize_cosine_transform(
        input_rows=source_rows,
        desired_rows=target_rows,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        method_name=method_name,
        restore_global_scale=restore_global_scale,
        loss_components=loss_components,
        loss_weights=loss_weights,
        loss_balance_targets=loss_balance_targets,
        loss_balance_ema_beta=loss_balance_ema_beta,
        loss_balance_weight_bounds=loss_balance_weight_bounds,
    )


def _fit_replaceme_cos(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    restore_global_scale: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    residual_rows: List[torch.Tensor] = []
    desired_rows: List[torch.Tensor] = []
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        h_raw = _load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu")
        y_raw = _load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu")
        r_raw = _load_auxiliary_tensor(cache_dir, item, auxiliary_id, map_location="cpu")
        h_raw, y_raw, r_raw = _align_replaceme_activations(
            h_raw,
            y_raw,
            r_raw,
            scene_id=str(item.get("scene_id", "<unknown>")),
            source_unit_id=source_unit_id,
        )
        h = _flatten(h_raw)
        y = _flatten(y_raw)
        r = _flatten(r_raw)
        residual_rows.append(r)
        desired_rows.append(y + r - h)
    return _optimize_cosine_transform(
        input_rows=residual_rows,
        desired_rows=desired_rows,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        method_name="ReplaceMe cosine",
        restore_global_scale=restore_global_scale,
    )


def _forward_current_cache_scene(
    *,
    model: Any,
    adapter: Any,
    item: Dict[str, Any],
    device: torch.device,
    auxiliary_ids: Set[str],
) -> Tuple[Any, Dict[str, torch.Tensor], torch.Tensor]:
    """Prepare and forward one cached scene through the current model."""
    image_paths = [str(path) for path in item.get("image_paths", []) if str(path)]
    if not image_paths:
        raise FileNotFoundError(
            f"Progressive recovery requires image_paths in cache metadata for "
            f"scene {item.get('scene_id', '<unknown>')}"
        )
    prepared = prepare_cache_batch(
        None,
        max_frames=0,
        meta={
            "image_paths": image_paths,
            "frame_ids": list(item.get("frame_ids") or []),
        },
        backbone_adapter=adapter,
    )
    images = prepared.images.to(device=device, non_blocking=False)
    forward, auxiliary_map = forward_cache_with_auxiliaries(
        adapter=adapter,
        model=model,
        images=images,
        device=device,
        auxiliary_ids=auxiliary_ids,
    )
    return forward, auxiliary_map, images


def _collect_current_boundary_rows(
    *,
    model: Any,
    adapter: Any,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Collect a Ghost boundary from the current progressively pruned model."""
    source_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    for item in scenes:
        forward, _auxiliary_map, images = _forward_current_cache_scene(
            model=model,
            adapter=adapter,
            item=item,
            device=device,
            auxiliary_ids=set(),
        )
        if source_unit_id not in forward.unit_map or target_unit_id not in forward.unit_map:
            raise KeyError(
                f"Sequential Ghost forward missed {source_unit_id}/{target_unit_id}"
            )
        source = _flatten(forward.unit_map[source_unit_id]).detach().cpu()
        target = _flatten(forward.unit_map[target_unit_id]).detach().cpu()
        if source.shape != target.shape:
            raise RuntimeError(
                f"source/target activation shape mismatch for {item.get('scene_id')}: "
                f"{source_unit_id}={tuple(source.shape)} "
                f"{target_unit_id}={tuple(target.shape)}"
            )
        source_rows.append(source)
        target_rows.append(target)
        del forward, images, source, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return source_rows, target_rows


def _fit_ghost_rows(
    *,
    source_rows: Sequence[torch.Tensor],
    target_rows: Sequence[torch.Tensor],
    device: torch.device,
    ridge: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if len(source_rows) != len(target_rows) or not source_rows:
        raise RuntimeError("Ghost recovery requires matching non-empty source/target rows")
    pairs = [
        (str(index), source.contiguous(), target.contiguous())
        for index, (source, target) in enumerate(zip(source_rows, target_rows))
    ]
    xtx, xtd = _ghost_sufficient_statistics(pairs, device=device)
    weight = _solve_ghost_from_statistics(xtx, xtd, ridge=float(ridge))
    return weight.detach().cpu(), {
        "fit_mode": "sequential_current_model",
        "num_scenes": len(pairs),
        "num_rows": sum(int(source.shape[0]) for source in source_rows),
        "ridge": float(ridge),
    }


def _collect_current_replaceme_rows(
    *,
    model: Any,
    adapter: Any,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run the current progressively recovered model and collect one boundary."""
    auxiliary_id = f"replaceme_residual:{source_unit_id}"
    residual_rows: List[torch.Tensor] = []
    desired_rows: List[torch.Tensor] = []
    for item in scenes:
        forward, auxiliary_map, images = _forward_current_cache_scene(
            adapter=adapter,
            model=model,
            item=item,
            device=device,
            auxiliary_ids={auxiliary_id},
        )
        if source_unit_id not in forward.unit_map or target_unit_id not in forward.unit_map:
            raise KeyError(
                f"Sequential ReplaceMe cosine forward missed {source_unit_id}/{target_unit_id}"
            )
        h_raw, y_raw, r_raw = _align_replaceme_activations(
            forward.unit_map[source_unit_id],
            forward.unit_map[target_unit_id],
            auxiliary_map[auxiliary_id],
            scene_id=str(item.get("scene_id", "<unknown>")),
            source_unit_id=source_unit_id,
        )
        h = _flatten(h_raw).detach().cpu()
        y = _flatten(y_raw).detach().cpu()
        r = _flatten(r_raw).detach().cpu()
        residual_rows.append(r)
        desired_rows.append(y + r - h)
        del images, forward, auxiliary_map, h_raw, y_raw, r_raw, h, y, r
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return residual_rows, desired_rows


def _fit_ghost_interval_sequential(
    *,
    adapter: Any,
    hf_model: str,
    records: Sequence[Dict[str, Any]],
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
    ridge: float,
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Fit interval Ghost operators front-to-back on the evolving model."""
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(device),
        dtype=None,
        eval_mode=True,
    )
    results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    handles: List[Any] = []
    for boundary_index, raw in enumerate(records):
        record = dict(raw)
        source_unit_id = str(record["source_unit_id"])
        target_unit_id = str(record["target_unit_id"])
        source_rows, target_rows = _collect_current_boundary_rows(
            model=model,
            adapter=adapter,
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            scenes=scenes,
            device=device,
        )
        weight, diagnostics = _fit_ghost_rows(
            source_rows=source_rows,
            target_rows=target_rows,
            device=device,
            ridge=ridge,
        )
        diagnostics["boundary_index"] = boundary_index
        module_name = recovery_module_name(source_unit_id)
        spec = {
            "type": "recovery_ghost_interval_sequential",
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "module_name": module_name,
            "weight_shape": [int(x) for x in weight.shape],
        }
        handles.extend(
            install_recovery_modules(model, adapter, [spec], device=device)
        )
        apply_recovery_weights(
            model,
            adapter,
            [spec],
            [
                {
                    "source_unit_id": source_unit_id,
                    "module_name": module_name,
                    "weight": weight,
                }
            ],
            device=device,
        )
        actions = [
            PruneAction(action="drop_unit", unit_id=unit_index_to_unit_id(index))
            for index in range(
                int(record["drop_start_index"]), int(record["drop_end_index"]) + 1
            )
        ]
        model = adapter.apply_prune_plan(model, actions)
        results.append((weight, diagnostics))
    for handle in handles:
        handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _fit_replaceme_cos_sequential(
    *,
    adapter: Any,
    hf_model: str,
    records: Sequence[Dict[str, Any]],
    scenes: Sequence[Dict[str, Any]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    restore_global_scale: bool,
) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Fit Multi-LT cosine transforms front-to-back on the evolving model."""
    model = adapter.load_pretrained(
        hf_model=str(hf_model or adapter.default_hf_model()),
        device=str(device),
        dtype=None,
        eval_mode=True,
    )
    results: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    for boundary_index, raw in enumerate(records):
        record = dict(raw)
        source_unit_id = str(record["source_unit_id"])
        target_unit_id = str(record["target_unit_id"])
        residual_rows, desired_rows = _collect_current_replaceme_rows(
            model=model,
            adapter=adapter,
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            scenes=scenes,
            device=device,
        )
        weight, diagnostics = _optimize_cosine_transform(
            input_rows=residual_rows,
            desired_rows=desired_rows,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=int(seed),
            method_name="ReplaceMe cosine sequential",
            restore_global_scale=restore_global_scale,
        )
        diagnostics.update(
            {
                "fit_mode": "sequential_current_model",
                "boundary_index": boundary_index,
            }
        )
        fuse_replaceme_transform(model, adapter, source_unit_id, weight.to(device))
        actions = [
            PruneAction(action="drop_unit", unit_id=unit_index_to_unit_id(index))
            for index in range(
                int(record["drop_start_index"]), int(record["drop_end_index"]) + 1
            )
        ]
        model = adapter.apply_prune_plan(model, actions)
        results.append((weight.cpu(), diagnostics))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _fit_prunecomp(
    *,
    source_unit_id: str,
    target_unit_id: str,
    scenes: Sequence[Dict[str, Any]],
    num_layers: int,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if eps <= 0:
        raise ValueError(f"Prune&Comp epsilon must be positive, got {eps}")
    alpha_sum = 0.0
    num_scenes = 0
    rows = 0
    num_channels = 0
    for item in scenes:
        cache_dir = Path(str(item["_cache_dir"]))
        x = _flatten(_load_unit_tensor(cache_dir, item, source_unit_id, num_layers=num_layers, map_location="cpu"))
        y = _flatten(_load_unit_tensor(cache_dir, item, target_unit_id, num_layers=num_layers, map_location="cpu"))
        if x.shape != y.shape:
            raise RuntimeError(f"Prune&Comp activation mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        # Match the official MagComp estimator: compute the per-channel
        # output/input magnitude ratio for each calibration batch first, then
        # average ratios across channels and calibration batches.  A cached
        # scene is the natural calibration-batch unit in LaRe.  This avoids
        # weighting scenes by their source magnitude, as a ratio of global
        # sums would do.
        x_magnitude = x.abs().mean(dim=0).to(torch.float64)
        y_magnitude = y.abs().mean(dim=0).to(torch.float64)
        scene_alpha = (y_magnitude / (x_magnitude + float(eps))).mean()
        if not bool(torch.isfinite(scene_alpha)):
            raise FloatingPointError(
                f"Prune&Comp produced a non-finite scene scale for {item.get('scene_id')}"
            )
        alpha_sum += float(scene_alpha.item())
        num_scenes += 1
        rows += int(x.shape[0])
        num_channels = int(x.shape[-1])
    if num_scenes <= 0 or rows <= 0:
        raise RuntimeError("No activation rows were available for Prune&Comp")
    alpha = alpha_sum / float(num_scenes)
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
        raise FloatingPointError(f"Invalid Prune&Comp scale {alpha}")
    return torch.tensor([alpha], dtype=torch.float32), {
        "alpha": alpha,
        "eps": float(eps),
        "num_rows": rows,
        "num_scenes": num_scenes,
        "num_channels": num_channels,
        "estimator": "mean_scene_mean_channel_magnitude_ratio",
    }


def main() -> None:
    args = _build_parser().parse_args()
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    plan_root = Path(args.prune_plan_root)
    recovery_root = Path(args.recovery_plan_root or args.prune_plan_root)
    recovery_plan_name = str(args.recovery_plan_name).strip()
    prune_plan_name = str(args.prune_plan_name).strip()
    prune_plan = compose_recovery_unit_prune_plan(
        prune_plan_root=plan_root,
        recovery_plan_root=recovery_root,
        prune_selector=prune_plan_name,
        recovery_plan=recovery_plan_name,
    )
    bundle = load_recovery_plan_bundle(recovery_root, recovery_plan_name)
    method, pair_records = _pair_records(bundle, prune_plan, recovery_plan_name)
    dataset_names = _parse_name_list(args.datasets or args.dataset)
    cache_dirs = [stage_dir(args.experiment, name, "cache") for name in dataset_names]
    scenes = _load_training_scenes(cache_dirs, int(args.max_scenes))
    if not scenes:
        raise SystemExit("No scenes selected from cache")
    first_manifest = load_v3_manifest(cache_dirs[0])
    num_layers = max((int(u.index) for u in first_manifest.layer_units if u.kind in ("frame", "global")), default=-1) + 1
    if int(prune_plan.num_units) != int(num_layers * 2):
        raise SystemExit(f"prune plan num_units={prune_plan.num_units} != cache num_units={num_layers * 2}")

    weights: List[torch.Tensor] = []
    recovery_specs: List[Dict[str, Any]] = []
    fit_diagnostics: List[Dict[str, Any]] = []
    snapshot_weights: Dict[int, List[torch.Tensor]] = {}
    sequential_replaceme_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    sequential_ghost_results: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]] = None
    if method == "recovery_ghost_interval_sequential" and len(pair_records) > 1:
        sequential_ghost_results = _fit_ghost_interval_sequential(
            adapter=adapter,
            hf_model=str(args.hf_model),
            records=pair_records,
            scenes=scenes,
            device=device,
            ridge=float(args.ridge),
        )
    if method in (
        "recovery_replaceme_cos",
        "recovery_replaceme_cos_more",
        "recovery_replaceme_cos_scale",
    ) and len(pair_records) > 1:
        sequential_replaceme_results = _fit_replaceme_cos_sequential(
            adapter=adapter,
            hf_model=str(args.hf_model),
            records=pair_records,
            scenes=scenes,
            device=device,
            epochs=int(
                args.replaceme_cos_more_epochs
                if method == "recovery_replaceme_cos_more"
                else args.replaceme_cos_epochs
            ),
            batch_size=int(args.replaceme_cos_batch_size),
            lr=float(args.replaceme_cos_lr),
            seed=int(args.replaceme_seed),
            restore_global_scale=method == "recovery_replaceme_cos_scale",
        )
    for record_index, record in enumerate(pair_records):
        source_unit_id = str(record.get("source_unit_id", ""))
        target_unit_id = str(record.get("target_unit_id", ""))
        diagnostics: Dict[str, Any] = {}
        if sequential_ghost_results is not None:
            weight, diagnostics = sequential_ghost_results[record_index]
        elif sequential_replaceme_results is not None:
            weight, diagnostics = sequential_replaceme_results[record_index]
        elif method in (
            "recovery_ghost",
            "recovery_ghost_interval",
            "recovery_ghost_interval_sequential",
            "recovery_linearpatch",
        ):
            weight = _fit_boundary_weight(
                method=(
                    "recovery_linearpatch"
                    if method == "recovery_linearpatch"
                    else "recovery_ghost"
                ),
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                ridge=float(args.ridge),
            )
        elif method == "recovery_ghost_mse_cka":
            weight, diagnostics = _fit_ghost_mse_cka(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                ridge=float(args.ridge),
                epochs=int(args.ghost_mse_cka_epochs),
                lr=float(args.ghost_mse_cka_lr),
                cka_weight=float(args.ghost_mse_cka_weight),
                mse_budget=float(args.ghost_mse_cka_budget),
                max_delta_ratio=float(args.ghost_mse_cka_max_delta_ratio),
                gradient_mode=str(args.ghost_mse_cka_gradient_mode),
                seed=int(args.ghost_mse_cka_seed),
            )
        elif method == "recovery_ghost_cka_only":
            weight, diagnostics, record_snapshots = _fit_ghost_cka_only(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                ridge=float(args.ridge),
                epochs=int(args.ghost_cka_only_epochs),
                lr=float(args.ghost_cka_only_lr),
                snapshot_epochs=str(args.ghost_cka_only_snapshot_epochs),
                seed=int(args.ghost_cka_only_seed),
            )
            for snapshot_epoch, snapshot_weight in record_snapshots.items():
                snapshot_weights.setdefault(int(snapshot_epoch), []).append(
                    snapshot_weight
                )
        elif method == "recovery_ghost_cka_ridge":
            weight, diagnostics = _fit_ghost_cka_ridge(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                ridge=float(args.ridge),
                ridge_grid=str(args.ghost_cka_ridge_grid),
                mse_budget=float(args.ghost_mse_cka_budget),
            )
        elif method in (
            "recovery_ghost_cos",
            "recovery_ghost_cos_more",
            "recovery_ghost_cos_scale",
            "recovery_ghost_mse_more",
            "recovery_ghost_mse_cos_more",
            "recovery_ghost_mse_cka_more",
            "recovery_ghost_mse_cos_cka_more",
            "recovery_ghost_mse_cos_balanced_more",
            "recovery_ghost_mse_cka_balanced_more",
            "recovery_ghost_mse_cos_cka_balanced_more",
            "recovery_ghost_mse_cka_ema_more",
            "recovery_ghost_mse_cos_cka_ema_half_more",
            "recovery_ghost_mse_cos_cka_ema_one_more",
        ):
            ghost_more_loss_components = {
                "recovery_ghost_mse_more": ("mse",),
                "recovery_ghost_mse_cos_more": ("mse", "cosine"),
                "recovery_ghost_mse_cka_more": ("mse", "cka"),
                "recovery_ghost_mse_cos_cka_more": ("mse", "cosine", "cka"),
                "recovery_ghost_mse_cos_balanced_more": ("mse", "cosine"),
                "recovery_ghost_mse_cka_balanced_more": ("mse", "cka"),
                "recovery_ghost_mse_cos_cka_balanced_more": (
                    "mse",
                    "cosine",
                    "cka",
                ),
                "recovery_ghost_mse_cka_ema_more": ("mse", "cka"),
                "recovery_ghost_mse_cos_cka_ema_half_more": (
                    "mse",
                    "cosine",
                    "cka",
                ),
                "recovery_ghost_mse_cos_cka_ema_one_more": (
                    "mse",
                    "cosine",
                    "cka",
                ),
            }
            ghost_more_loss_weights = {
                "recovery_ghost_mse_cos_balanced_more": {
                    "mse": 1.0,
                    "cosine": float(args.ghost_more_cos_weight),
                },
                "recovery_ghost_mse_cka_balanced_more": {
                    "mse": 1.0,
                    "cka": float(args.ghost_more_cka_weight),
                },
                "recovery_ghost_mse_cos_cka_balanced_more": {
                    "mse": 1.0,
                    "cosine": float(args.ghost_more_cos_weight),
                    "cka": float(args.ghost_more_cka_weight),
                },
            }
            ghost_more_balance_targets = {
                "recovery_ghost_mse_cka_ema_more": {"cka": 1.0},
                "recovery_ghost_mse_cos_cka_ema_half_more": {
                    "cosine": 0.5,
                    "cka": 0.5,
                },
                "recovery_ghost_mse_cos_cka_ema_one_more": {
                    "cosine": 1.0,
                    "cka": 1.0,
                },
            }
            balance_targets = ghost_more_balance_targets.get(method)
            balance_bounds = None
            if balance_targets:
                balance_bounds = {
                    name: (
                        float(
                            args.ghost_more_ema_cos_min_weight
                            if name == "cosine"
                            else args.ghost_more_ema_cka_min_weight
                        ),
                        float(
                            args.ghost_more_ema_cos_max_weight
                            if name == "cosine"
                            else args.ghost_more_ema_cka_max_weight
                        ),
                    )
                    for name in balance_targets
                }
            weight, diagnostics = _fit_ghost_cos(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                epochs=int(
                    args.ghost_cos_more_epochs
                    if method
                    in (
                        "recovery_ghost_cos_more",
                        *ghost_more_loss_components,
                    )
                    else args.ghost_cos_epochs
                ),
                batch_size=int(args.ghost_cos_batch_size),
                lr=float(args.ghost_cos_lr),
                seed=int(args.ghost_cos_seed),
                restore_global_scale=method == "recovery_ghost_cos_scale",
                loss_components=ghost_more_loss_components.get(
                    method, ("cosine",)
                ),
                loss_weights=ghost_more_loss_weights.get(method),
                loss_balance_targets=balance_targets,
                loss_balance_ema_beta=float(args.ghost_more_ema_beta),
                loss_balance_weight_bounds=balance_bounds,
                method_name=method,
            )
        elif method == "recovery_replaceme_ls":
            weight, diagnostics = _fit_replaceme_ls(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                alpha=float(args.replaceme_ls_alpha),
            )
        elif method in (
            "recovery_replaceme_cos",
            "recovery_replaceme_cos_more",
            "recovery_replaceme_cos_scale",
        ):
            weight, diagnostics = _fit_replaceme_cos(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                device=device,
                epochs=int(
                    args.replaceme_cos_more_epochs
                    if method == "recovery_replaceme_cos_more"
                    else args.replaceme_cos_epochs
                ),
                batch_size=int(args.replaceme_cos_batch_size),
                lr=float(args.replaceme_cos_lr),
                seed=int(args.replaceme_seed),
                restore_global_scale=method == "recovery_replaceme_cos_scale",
            )
        elif method == "recovery_prunecomp":
            weight, diagnostics = _fit_prunecomp(
                source_unit_id=source_unit_id,
                target_unit_id=target_unit_id,
                scenes=scenes,
                num_layers=num_layers,
                eps=float(args.prunecomp_eps),
            )
        else:
            raise ValueError(f"Unsupported training-free recovery method {method!r}")
        weights.append(weight)
        fit_diagnostics.append(diagnostics)
        recovery_specs.append(
            {
                "type": method,
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "source_unit_index": int(record.get("source_unit_index")),
                "target_unit_index": int(record.get("target_unit_index")),
                "module_name": recovery_module_name(source_unit_id),
                "weight_shape": [int(x) for x in weight.shape],
            }
        )

    default_output_root = Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT_DEFAULT)))
    out_dir = args.output_dir or default_output_root / adapter.name / "recovery" / prune_plan_name / recovery_plan_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    summary = {
        "backbone": adapter.name,
        "method": method,
        "prune_plan_name": prune_plan_name,
        "recovery_plan_name": recovery_plan_name,
        "num_scenes": len(scenes),
        "cache_dirs": [str(p) for p in cache_dirs],
        "recovery_specs": recovery_specs,
        "fit_diagnostics": fit_diagnostics,
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hf_model = str(args.hf_model or adapter.default_hf_model())
    def checkpoint_payload(
        checkpoint_weights: Sequence[torch.Tensor],
        *,
        epoch: int,
        variant: str,
    ) -> Dict[str, Any]:
        return {
            "global_step": 0,
            "epoch": int(epoch),
            "epochs": int(epoch),
            "drop_indices": sorted(set(int(x) for x in prune_plan.drop)),
            "lora_indices": [],
            "train_unit_indices": [],
            "training_method": method,
            "recovery_method": method,
            "recovery_specs": recovery_specs,
            "recovery_weights": [
                {
                    "source_unit_id": str(spec["source_unit_id"]),
                    "module_name": str(spec["module_name"]),
                    "weight": weight.detach().cpu().to(torch.float32),
                }
                for spec, weight in zip(recovery_specs, checkpoint_weights)
            ],
            "prune_plan": asdict(prune_plan),
            "variant": variant,
            "base_hf_model": hf_model,
            "optimizer_state_dict": {},
            "args": vars(args),
        }

    torch.save(
        checkpoint_payload(
            weights,
            epoch=int(
                fit_diagnostics[0].get("selected_epoch", 0)
                if fit_diagnostics
                else 0
            ),
            variant="training_free_recovery_delta",
        ),
        out_dir / "checkpoints" / "latest.pt",
    )
    if snapshot_weights:
        for snapshot_epoch, epoch_weights in sorted(snapshot_weights.items()):
            if len(epoch_weights) != len(recovery_specs):
                raise RuntimeError(
                    f"Incomplete recovery snapshot at epoch {snapshot_epoch}: "
                    f"{len(epoch_weights)} weights for {len(recovery_specs)} specs"
                )
            torch.save(
                checkpoint_payload(
                    epoch_weights,
                    epoch=int(snapshot_epoch),
                    variant="training_free_recovery_cka_only_snapshot",
                ),
                out_dir
                / "checkpoints"
                / f"epoch_{int(snapshot_epoch):03d}.pt",
            )
    print(f"[recovery] wrote {out_dir / 'checkpoints' / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()

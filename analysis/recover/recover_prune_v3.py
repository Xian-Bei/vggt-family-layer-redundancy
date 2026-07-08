"""Training-free boundary recovery for pruned LaRe models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    UnitPrunePlan,
    load_named_prune_plan,
    prune_plan_bundle_path,
    unit_index_to_kind_layer,
)
from analysis.recover.operators import recovery_module_name
from analysis.redundancy.cache_layout import load_scene_initial_tokens, load_scene_replay_layer
from analysis.redundancy.cache_layout_v3 import load_v3_manifest, manifest_scenes_for_training
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir
from analysis.prune_plan.recovery_plan import (
    TRAINING_FREE_RECOVERY_METHODS,
    compose_training_unit_prune_plan,
    load_training_plan_bundle,
)


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
    p.add_argument("--recovery-plan-root", "--training-plan-root", dest="training_plan_root", type=Path, default=None)
    p.add_argument("--recovery-plan-name", "--training-plan-name", dest="training_plan_name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-6)
    return p


def _parse_name_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").replace(",", " ").split() if x.strip()]


def _load_training_scenes(cache_dirs: Sequence[Path], max_scenes: int) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    for cache_dir in cache_dirs:
        manifest = load_v3_manifest(cache_dir)
        for item in manifest_scenes_for_training(manifest):
            item = dict(item)
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
    if key == "layer_-1":
        return load_scene_initial_tokens(cache_dir, item)
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


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard(size: int, *, device: torch.device) -> torch.Tensor:
    if not _is_pow2(int(size)):
        raise ValueError(f"Hadamard rotate recovery requires power-of-two channels, got {size}")
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    while int(h.shape[0]) < int(size):
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    return h / float(size) ** 0.5


def _pair_records(bundle: Dict[str, Any], prune_plan_name: str, recovery_plan_name: str) -> Tuple[str, List[Dict[str, Any]]]:
    method = str(bundle.get("method", "")).strip().lower()
    if method not in TRAINING_FREE_RECOVERY_METHODS:
        raise ValueError(f"{recovery_plan_name} is not training-free recovery: method={method!r}")
    span = int(str(prune_plan_name).removeprefix("prune_span"))
    record = bundle.get("spans", {}).get(str(span))
    if not isinstance(record, dict):
        raise FileNotFoundError(f"{recovery_plan_name} has no record for {prune_plan_name}")
    return method, [record]


def _fit_weight(
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
            if method == "recovery_rotate":
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
        c = int(xtx.shape[0])
        reg = float(ridge) * torch.eye(c, device=device, dtype=torch.float64)
        m_star = torch.linalg.solve(xtx + reg, xtd)
        return (torch.eye(c, device=device, dtype=torch.float64) + m_star).to(torch.float32).cpu()

    if sum_x_abs is None or sum_y_abs is None or count <= 0:
        raise RuntimeError(f"No activation rows were available for {method}")
    scale = sum_y_abs / (sum_x_abs + 1e-8)
    if method == "recovery_diag":
        return torch.diag(scale.to(torch.float32)).cpu()
    assert hmat is not None
    return (hmat @ torch.diag(scale.to(torch.float32)) @ hmat.T).to(torch.float32).cpu()


def main() -> None:
    args = _build_parser().parse_args()
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    plan_root = Path(args.prune_plan_root)
    recovery_root = Path(args.training_plan_root or args.prune_plan_root)
    recovery_plan_name = str(args.training_plan_name).strip()
    prune_plan_name = str(args.prune_plan_name).strip()
    prune_plan = compose_training_unit_prune_plan(
        prune_plan_root=plan_root,
        training_plan_root=recovery_root,
        prune_selector=prune_plan_name,
        training_plan=recovery_plan_name,
    )
    bundle = load_training_plan_bundle(recovery_root, recovery_plan_name)
    method, pair_records = _pair_records(bundle, prune_plan_name, recovery_plan_name)
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
    for record in pair_records:
        source_unit_id = str(record.get("source_unit_id", ""))
        target_unit_id = str(record.get("target_unit_id", ""))
        weight = _fit_weight(
            method=method,
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            scenes=scenes,
            num_layers=num_layers,
            device=device,
            ridge=float(args.ridge),
        )
        weights.append(weight)
        recovery_specs.append(
            {
                "type": method,
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "source_unit_index": int(record.get("source_unit_index")),
                "target_unit_index": int(record.get("target_unit_index")),
                "module_name": recovery_module_name(source_unit_id),
                "weight_shape": [int(weight.shape[0]), int(weight.shape[1])],
            }
        )

    default_output_root = Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT_DEFAULT)))
    out_dir = args.output_dir or default_output_root / adapter.name / "recover" / prune_plan_name / recovery_plan_name
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
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recover_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hf_model = str(args.hf_model or adapter.default_hf_model())
    torch.save(
        {
            "global_step": 0,
            "epoch": 0,
            "epochs": 0,
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
                for spec, weight in zip(recovery_specs, weights)
            ],
            "prune_plan": asdict(prune_plan),
            "variant": "training_free_recovery_delta",
            "base_hf_model": hf_model,
            "optimizer_state_dict": {},
            "args": vars(args),
        },
        out_dir / "checkpoints" / "latest.pt",
    )
    print(f"[recover] wrote {out_dir / 'checkpoints' / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()

"""Convert full training-free recovery checkpoints into compact delta checkpoints."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Compact closed-form recovery checkpoints in place.")
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def _find_recovery_weight(state: Dict[str, Any], spec: Dict[str, Any]) -> torch.Tensor:
    module_name = str(spec.get("module_name", "")).strip()
    source_unit_id = str(spec.get("source_unit_id", "")).strip()
    if not module_name:
        raise ValueError(f"recovery spec missing module_name: {spec}")
    suffix = f"{module_name}.weight"
    matches = [(key, value) for key, value in state.items() if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one state_dict entry for {source_unit_id}/{module_name}, found {len(matches)}"
        )
    _key, value = matches[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Recovery state entry for {source_unit_id}/{module_name} is not a tensor")
    return value.detach().cpu().to(torch.float32)


def _compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("variant") == "training_free_recovery_delta" and "model_state_dict" not in payload:
        return payload
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict to compact")
    recovery_specs = payload.get("recovery_specs")
    if not isinstance(recovery_specs, list) or not recovery_specs:
        raise ValueError("checkpoint has no recovery_specs to compact")
    recovery_weights: List[Dict[str, Any]] = []
    for raw in recovery_specs:
        spec = dict(raw)
        recovery_weights.append(
            {
                "source_unit_id": str(spec.get("source_unit_id", "")),
                "module_name": str(spec.get("module_name", "")),
                "weight": _find_recovery_weight(state, spec),
            }
        )

    compact = {key: value for key, value in payload.items() if key != "model_state_dict"}
    compact["variant"] = "training_free_recovery_delta"
    compact["original_variant"] = payload.get("variant", "")
    compact["recovery_weights"] = recovery_weights
    compact["optimizer_state_dict"] = {}
    return compact


def compact_checkpoint(path: Path, *, dry_run: bool = False) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload at {path}: {type(payload)}")
    before = path.stat().st_size
    compact = _compact_payload(payload)
    after_estimate = sum(
        int(entry["weight"].numel()) * int(entry["weight"].element_size())
        for entry in compact.get("recovery_weights", [])
        if isinstance(entry, dict) and torch.is_tensor(entry.get("weight"))
    )
    if dry_run:
        return f"{path}\t{before}\t~{after_estimate}\tdry-run"
    tmp_path = path.with_name(f"{path.name}.compact_tmp")
    torch.save(compact, tmp_path)
    os.replace(tmp_path, path)
    after = path.stat().st_size
    del payload, compact
    gc.collect()
    return f"{path}\t{before}\t{after}\tconverted"


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    for ckpt in args.checkpoints:
        print(compact_checkpoint(ckpt, dry_run=bool(args.dry_run)), flush=True)


if __name__ == "__main__":
    main()

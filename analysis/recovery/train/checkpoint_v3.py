"""Restore pruned + LoRA structure from train-based recovery checkpoints."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch.nn as nn

from analysis.backbones.contracts import BackboneAdapter
from analysis.prune_plan import UnitPrunePlan, prune_actions_from_plan

EvalMode = Literal["upper", "lower", "model"]


def atomic_save_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    """Serialize once to a same-directory temporary file, then atomically replace."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_publish_latest(serialized_path: Path, latest_path: Path) -> None:
    """Publish an existing checkpoint by hard link, with atomic-copy fallback."""
    serialized_path, latest_path = Path(serialized_path), Path(latest_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest_path.with_name(f".{latest_path.name}.tmp.{os.getpid()}")
    try:
        try:
            os.link(serialized_path, temporary)
        except OSError:
            shutil.copy2(serialized_path, temporary)
        os.replace(temporary, latest_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_train_checkpoint_payload(path, *, map_location=None) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def apply_checkpoint_structure(
    model: nn.Module,
    adapter: BackboneAdapter,
    payload: Dict[str, Any],
    *,
    prune_plan: UnitPrunePlan | None = None,
) -> None:
    """Apply skip/LoRA topology then load weights via backbone adapter."""
    adapter.apply_checkpoint_structure(model, payload, prune_plan=prune_plan)


def apply_eval_model_structure(
    model: nn.Module,
    adapter: BackboneAdapter,
    *,
    eval_mode: EvalMode,
    prune_plan: Optional[UnitPrunePlan] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    map_location: Any = None,
) -> None:
    """
    Prepare model for stage-4 eval.

    - upper: pretrained only, or load checkpoint weights without prune/LoRA topology.
    - lower: apply prune plan (skip layers only), no LoRA inject, no train checkpoint.
    - model: prune + LoRA topology from plan, then load finetuned checkpoint.
    """
    ckpt_path = Path(checkpoint) if checkpoint is not None else None
    has_ckpt = ckpt_path is not None and ckpt_path.is_file()

    if eval_mode == "upper":
        if has_ckpt:
            payload = load_train_checkpoint_payload(ckpt_path, map_location=map_location)
            if payload.get("variant") in {
                "training_free_recovery_delta", "trainable_recovery_delta_v1"
            } and "model_state_dict" not in payload:
                raise ValueError("upper eval cannot load a compact recovery delta checkpoint")
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=False)
        return

    if eval_mode == "lower":
        if prune_plan is None:
            raise ValueError("lower eval requires a prune plan from --prune-plan-root")
        drop_actions, _ = prune_actions_from_plan(prune_plan)
        adapter.apply_prune_plan(model, drop_actions)
        return

    if eval_mode == "model":
        if prune_plan is None:
            raise ValueError("model eval requires a prune plan from --prune-plan-root")
        if not has_ckpt:
            raise ValueError("model eval requires an existing --checkpoint file")
        payload = load_train_checkpoint_payload(ckpt_path, map_location=map_location)
        apply_checkpoint_structure(model, adapter, payload, prune_plan=prune_plan)
        return

    raise ValueError(f"Unknown eval_mode={eval_mode!r}")


def ckpt_lora_hparams(payload: Dict[str, Any]) -> Tuple[int, float, float]:
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    return (
        int(args.get("lora_rank", 8)),
        float(args.get("lora_alpha", 16.0)),
        float(args.get("lora_dropout", 0.0)),
    )

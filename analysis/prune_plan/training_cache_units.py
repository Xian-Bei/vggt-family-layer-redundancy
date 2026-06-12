"""Derive minimal on-disk cache unit ids from a UnitPrunePlan (LoRA training)."""

from __future__ import annotations

from typing import AbstractSet, Set, Tuple

import torch.nn as nn

from analysis.prune_plan import UnitPrunePlan, lora_layers_from_plan, unit_index_to_kind_layer, unit_index_to_unit_id


def layers_to_unit_ids(layers: AbstractSet[int]) -> Set[str]:
    out: Set[str] = set()
    for layer in layers:
        L = int(layer)
        out.add(f"frame_{L}")
        out.add(f"global_{L}")
    return out


def replay_seed_spec(plan: UnitPrunePlan) -> Tuple[int, str, int]:
    """Return (seed_load_layer, seed_kind, replay_start_layer) for cache-first replay.

    Seed unit is the heterotypic predecessor of the minimum LoRA unit index (u-1).
    When min LoRA unit is 0, there is no predecessor; replay starts at layer 0 with
    seed_kind ``none`` (no cached half-unit seed).
    """
    if not plan.lora:
        raise ValueError("Prune plan has no lora entries; cannot derive replay seed.")
    min_u = min(int(k) for k in plan.lora.keys())
    replay_start_layer = int(min_u) // 2
    if min_u <= 0:
        return 0, "none", int(replay_start_layer)
    seed_u = int(min_u) - 1
    seed_kind, seed_layer = unit_index_to_kind_layer(seed_u)
    return int(seed_layer), str(seed_kind), int(replay_start_layer)


def replay_start_layer(plan: UnitPrunePlan) -> int:
    """Layer index where cache-first replay begins."""
    _seed_layer, _seed_kind, start = replay_seed_spec(plan)
    return int(start)


def required_cache_layers_for_training(plan: UnitPrunePlan, model: nn.Module) -> Set[int]:
    """Backward-compat helper: layer indices whose full concat may be needed."""
    num_layers = int(plan.num_units) // 2
    if num_layers <= 0:
        raise ValueError(f"plan.num_units must be even and > 0, got {plan.num_units}")

    lora_layers = lora_layers_from_plan(plan)
    if not lora_layers:
        raise ValueError("Prune plan has no lora entries; cannot derive training cache layers.")

    start_layer = replay_start_layer(plan)
    seed_layer, seed_kind, _ = replay_seed_spec(plan)
    layers: set[int] = set()
    if str(seed_kind) != "none":
        layers.add(int(seed_layer))

    for spec in plan.lora.values():
        if spec.alignment_teacher_unit_index is None:
            raise ValueError("lora entry missing alignment_teacher_unit_index")
        layers.add(int(spec.alignment_teacher_unit_index) // 2)

    from analysis.backbones.registry import get_backbone

    adapter = get_backbone(str(plan.backbone))
    for l in adapter.required_head_layers_for_cache(model, num_layers):
        if int(l) < int(start_layer):
            layers.add(int(l))

    del seed_kind
    return layers


def required_cache_unit_ids(plan: UnitPrunePlan, model: nn.Module) -> Set[str]:
    """Unit ids to write under scene/layers/ for cache-first training.

    Half-unit aware:
    - replay seed is u-1 (heterotypic predecessor of min LoRA unit)
    - alignment uses exactly the teacher target half-unit
    - VGGT-family head layers before replay start need full concat (frame+global)
    """
    num_layers = int(plan.num_units) // 2
    if num_layers <= 0:
        raise ValueError(f"plan.num_units must be even and > 0, got {plan.num_units}")

    if not plan.lora:
        raise ValueError("Prune plan has no lora entries; cannot derive training cache units.")

    required_units: Set[str] = set()
    seed_layer, seed_kind, _ = replay_seed_spec(plan)
    if str(seed_kind) == "none":
        required_units.add("layer_-1")
    else:
        required_units.add(unit_index_to_unit_id(2 * int(seed_layer) + (0 if str(seed_kind) == "frame" else 1)))

    start_layer = replay_start_layer(plan)

    for spec in plan.lora.values():
        if spec.alignment_teacher_unit_index is None:
            raise ValueError("lora entry missing alignment_teacher_unit_index")
        t_idx = int(spec.alignment_teacher_unit_index)
        if t_idx < 0 or t_idx >= int(plan.num_units):
            raise ValueError(f"alignment_teacher_unit_index out of range: {t_idx}")
        required_units.add(unit_index_to_unit_id(t_idx))

    from analysis.backbones.registry import get_backbone

    adapter = get_backbone(str(plan.backbone))
    for l in adapter.required_head_layers_for_cache(model, num_layers):
        if int(l) < int(start_layer):
            required_units.add(f"frame_{int(l)}")
            required_units.add(f"global_{int(l)}")

    return required_units

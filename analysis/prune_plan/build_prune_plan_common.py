"""Shared helpers for prune and training plan builders."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan
from analysis.redundancy.graph_io import (
    SPECIAL_PATCH_MEAN,
    load_similarity as load_unit_similarity,
    parse_dataset_list,
)
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT


PRUNING_TOKEN_REGIONS = ("all", "special", "patch", SPECIAL_PATCH_MEAN)


def normalize_pruning_token_region(token_region: str) -> str:
    normalized_region = str(token_region).strip().lower()
    if normalized_region not in PRUNING_TOKEN_REGIONS:
        raise ValueError(
            f"Unsupported pruning token region {token_region!r}; "
            f"expected one of {PRUNING_TOKEN_REGIONS}"
        )
    return normalized_region


def plan_name(t: int, l: int) -> str:
    return f"plan_t{int(t)}_l{int(l)}"


def default_plan_root(backbone: str) -> Path:
    return OUTPUT_ROOT_DEFAULT / str(backbone).strip().lower()


def unit_id_to_index(unit_id: str) -> int:
    if str(unit_id).strip() == "layer_-1":
        return -1
    kind, layer_s = str(unit_id).split("_", 1)
    layer = int(layer_s)
    base = 2 * layer
    return base if kind == "frame" else (base + 1)


def load_similarity(
    *,
    experiment: str,
    dataset: str,
    token_region: str,
    metric: str,
) -> Tuple[List[str], np.ndarray, Dict[str, object]]:
    """Compatibility wrapper around the redundancy-layer raw similarity loader."""
    normalized_region = normalize_pruning_token_region(token_region)
    return load_unit_similarity(
        experiment=experiment,
        dataset=dataset,
        token_region=normalized_region,
        metric=metric,
        output_root=OUTPUT_ROOT_DEFAULT,
    )


def build_recovery_plan_from_interval(
    *,
    picks_prefix: Sequence,
    lora_prefix: int,
    lora_rank: int,
    lora_alpha: float,
    backbone: str,
    dataset: str,
    num_units: int,
    unit_ids: Sequence[str] = (),
    source_cache: str,
) -> Tuple[UnitPrunePlan, List[int]]:
    """Build recovery metadata for zero-based closed-interval picks."""
    drop_indices: set[int] = set()
    lora_specs: dict[str, UnitLoraSpec] = {}
    lora_indices: set[int] = set()
    for pick in picks_prefix:
        start = int(pick.i)
        end = int(pick.j)
        anchor = start - 1
        if anchor < 0:
            raise ValueError(
                f"closed interval [{start}, {end}] has no physical left-boundary unit "
                "for LoRA recovery"
            )
        for idx in range(start, end + 1):
            drop_indices.add(int(idx))
        for off in range(max(1, int(lora_prefix))):
            k = anchor - off
            if k < 0:
                break
            lora_indices.add(int(k))
            if str(int(k)) not in lora_specs:
                lora_specs[str(int(k))] = UnitLoraSpec(
                    rank=int(lora_rank),
                    alpha=float(lora_alpha),
                    dropout=0.0,
                    target_submodules=("qkv", "proj"),
                    alignment_teacher_unit_index=end,
                )
    drop_sorted = sorted(drop_indices)
    extra = {
        "triangles": [
            {
                "i": int(p.i),
                "j": int(p.j),
                "unit_i": str(p.unit_i),
                "unit_j": str(p.unit_j),
            }
            for p in picks_prefix
        ],
    }
    plan = UnitPrunePlan(
        backbone=backbone,
        dataset=dataset,
        num_units=int(num_units),
        unit_ids=[str(x) for x in unit_ids],
        drop=drop_sorted,
        drop_length=len(drop_sorted),
        lora=lora_specs,
        source_cache=source_cache,
        extra=extra,
    )
    return plan, sorted(lora_indices)

"""Shared helpers for prune and training plan builders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan
from analysis.redundancy.similarity_v3 import (
    list_redundancy_graph_regions,
    resolve_redundancy_graph_path,
)
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir


REDUNDANCY_AGGREGATIONS = frozenset({"mean", "min"})


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


def reduce_similarity_matrices(matrices: Sequence[np.ndarray], reduction: str) -> np.ndarray:
    if not matrices:
        raise ValueError("Cannot aggregate an empty collection of similarity matrices")
    stack = np.stack([np.asarray(matrix, dtype=np.float64) for matrix in matrices], axis=0)
    if reduction == "mean":
        return np.mean(stack, axis=0)
    if reduction == "min":
        return np.min(stack, axis=0)
    raise ValueError(f"Unknown aggregation {reduction!r}; expected mean or min")


def _load_similarity_graph(
    path: Path,
    *,
    metric_key: str,
) -> Tuple[List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        if "unit_ids" not in npz:
            raise KeyError(f"Missing unit_ids in {path}")
        if metric_key not in npz:
            raise KeyError(f"Missing {metric_key} in {path}")
        unit_ids = [str(unit_id) for unit_id in list(npz["unit_ids"])]
        sim = np.asarray(npz[metric_key], dtype=np.float64)
    if sim.shape != (len(unit_ids), len(unit_ids)):
        raise ValueError(
            f"Invalid similarity shape {sim.shape} in {path}; "
            f"expected {(len(unit_ids), len(unit_ids))} from unit_ids"
        )
    return unit_ids, sim


def _validate_compatible_graph(
    *,
    expected_unit_ids: Sequence[str],
    expected_shape: Tuple[int, ...],
    unit_ids: Sequence[str],
    sim: np.ndarray,
    path: Path,
) -> None:
    if list(unit_ids) != list(expected_unit_ids):
        raise ValueError(f"unit_ids in {path} do not match the other selected similarity graphs")
    if sim.shape != expected_shape:
        raise ValueError(
            f"Similarity shape {sim.shape} in {path} does not match {expected_shape}"
        )


def _load_dataset_similarity(
    *,
    experiment: str,
    dataset: str,
    token_region: str,
    metric_key: str,
) -> Tuple[List[str], np.ndarray, List[Path], List[str]]:
    red_dir = stage_dir(experiment, dataset, "redundancy")
    if token_region in REDUNDANCY_AGGREGATIONS:
        regions = list_redundancy_graph_regions(red_dir)
        if not regions:
            raise FileNotFoundError(
                f"No redundancy graphs found for mean/min aggregation under {red_dir}"
            )
    else:
        regions = [token_region]

    unit_ids: Optional[List[str]] = None
    matrices: List[np.ndarray] = []
    paths: List[Path] = []
    for region in regions:
        path = resolve_redundancy_graph_path(red_dir, region)
        if path is None:
            raise FileNotFoundError(
                f"Missing redundancy graph for token region {region!r} under {red_dir}"
            )
        current_unit_ids, current_sim = _load_similarity_graph(path, metric_key=metric_key)
        if unit_ids is None:
            unit_ids = current_unit_ids
        else:
            _validate_compatible_graph(
                expected_unit_ids=unit_ids,
                expected_shape=matrices[0].shape,
                unit_ids=current_unit_ids,
                sim=current_sim,
                path=path,
            )
        matrices.append(current_sim)
        paths.append(path)

    assert unit_ids is not None
    sim = (
        matrices[0]
        if token_region not in REDUNDANCY_AGGREGATIONS
        else reduce_similarity_matrices(matrices, token_region)
    )
    return unit_ids, sim, paths, regions


def _available_datasets(
    experiment: str,
    token_region: str,
) -> List[str]:
    experiment_root = OUTPUT_ROOT_DEFAULT / experiment
    if not experiment_root.is_dir():
        return []
    datasets: List[str] = []
    for child in sorted(experiment_root.iterdir(), key=lambda path: path.name):
        red_dir = child / "redundancy"
        if not red_dir.is_dir():
            continue
        regions = list_redundancy_graph_regions(red_dir)
        if token_region in REDUNDANCY_AGGREGATIONS:
            usable = bool(regions)
        else:
            usable = token_region in regions
        if usable:
            datasets.append(child.name)
    return datasets


def load_similarity(
    *,
    experiment: str,
    dataset: str,
    token_region: str,
    metric: str,
) -> Tuple[List[str], np.ndarray, Dict[str, object]]:
    dataset_selector = str(dataset).strip().lower()
    token_selector = str(token_region).strip().lower()
    metric_key = f"{metric.strip().lower()}_mean"
    datasets = (
        _available_datasets(experiment, token_selector)
        if dataset_selector in REDUNDANCY_AGGREGATIONS
        else [dataset]
    )
    if not datasets:
        raise FileNotFoundError(
            f"No datasets with token-region {token_selector!r} redundancy graphs found under "
            f"{OUTPUT_ROOT_DEFAULT / experiment}"
        )

    unit_ids: Optional[List[str]] = None
    dataset_matrices: List[np.ndarray] = []
    source_paths: List[Path] = []
    regions_by_dataset: Dict[str, List[str]] = {}
    for dataset_name in datasets:
        current_unit_ids, current_sim, paths, regions = _load_dataset_similarity(
            experiment=experiment,
            dataset=dataset_name,
            token_region=token_selector,
            metric_key=metric_key,
        )
        if unit_ids is None:
            unit_ids = current_unit_ids
        else:
            _validate_compatible_graph(
                expected_unit_ids=unit_ids,
                expected_shape=dataset_matrices[0].shape,
                unit_ids=current_unit_ids,
                sim=current_sim,
                path=paths[0],
            )
        dataset_matrices.append(current_sim)
        source_paths.extend(paths)
        regions_by_dataset[str(dataset_name)] = regions

    assert unit_ids is not None
    sim = (
        reduce_similarity_matrices(dataset_matrices, dataset_selector)
        if dataset_selector in REDUNDANCY_AGGREGATIONS
        else dataset_matrices[0]
    )
    provenance: Dict[str, object] = {
        "metric_key": metric_key,
        "dataset_selector": dataset_selector,
        "datasets": [str(name) for name in datasets],
        "token_region_selector": token_selector,
        "aggregation_order": ["token_region", "dataset"],
        "token_regions_by_dataset": regions_by_dataset,
        "source_graphs": [str(path) for path in source_paths],
    }
    return unit_ids, sim, provenance


def build_plan_from_prefix(
    *,
    picks_prefix: Sequence,
    lora_prefix: int,
    lora_rank: int,
    lora_alpha: float,
    backbone: str,
    dataset: str,
    num_units: int,
    source_cache: str,
) -> Tuple[UnitPrunePlan, List[int]]:
    drop_indices: set[int] = set()
    lora_specs: dict[str, UnitLoraSpec] = {}
    lora_indices: set[int] = set()
    for pick in picks_prefix:
        i = int(pick.i)
        j = int(pick.j)
        for idx in range(i + 1, j + 1):
            drop_indices.add(int(idx))
        for off in range(max(1, int(lora_prefix))):
            k = i - off
            if k < 0:
                break
            lora_indices.add(int(k))
            if str(int(k)) not in lora_specs:
                lora_specs[str(int(k))] = UnitLoraSpec(
                    rank=int(lora_rank),
                    alpha=float(lora_alpha),
                    dropout=0.0,
                    target_submodules=("qkv", "proj"),
                    alignment_teacher_unit_index=j,
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
        drop=drop_sorted,
        drop_length=len(drop_sorted),
        lora=lora_specs,
        source_cache=source_cache,
        extra=extra,
    )
    return plan, sorted(lora_indices)

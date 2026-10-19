"""Read and aggregate unit-output pairwise redundancy graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.redundancy.similarity_v3 import (
    UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS,
    resolve_redundancy_graph_path,
)
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT


SPECIAL_PATCH_MEAN = "special_patch_mean"
DATASET_LIST_AGGREGATION = "mean"


def reduce_similarity_matrices(
    matrices: Sequence[np.ndarray],
) -> np.ndarray:
    if not matrices:
        raise ValueError("Cannot aggregate an empty collection of similarity matrices")
    stack = np.stack([np.asarray(matrix, dtype=np.float64) for matrix in matrices], axis=0)
    return np.mean(stack, axis=0)


def _unit_id_order_key(unit_id: str) -> Tuple[int, str]:
    value = str(unit_id)
    try:
        kind, layer_s = value.split("_", 1)
        layer = int(layer_s)
    except (TypeError, ValueError):
        return (10**9, value)
    kind_offset = 0 if kind == "frame" else (1 if kind == "global" else 2)
    return (2 * layer + kind_offset, value)


def _align_graph_to_full_unit_catalog(
    *,
    path: Path,
    unit_ids: Sequence[str],
    similarity: np.ndarray,
    dropped_unit_ids: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    remaining = [str(unit_id) for unit_id in unit_ids]
    dropped = [str(unit_id) for unit_id in dropped_unit_ids]
    if not dropped:
        raise ValueError(f"Missing dropped_unit_ids required to align pruned graph {path}")
    overlap = sorted(set(remaining) & set(dropped))
    if overlap:
        raise ValueError(f"Pruned graph {path} marks units as both remaining and dropped: {overlap}")
    full_unit_ids = sorted([*remaining, *dropped], key=_unit_id_order_key)
    if len(set(full_unit_ids)) != len(full_unit_ids):
        raise ValueError(f"Pruned graph {path} contains duplicate unit ids")
    full_index = {unit_id: index for index, unit_id in enumerate(full_unit_ids)}
    remaining_indices = [full_index[unit_id] for unit_id in remaining]
    aligned = np.full(
        (len(full_unit_ids), len(full_unit_ids)),
        np.nan,
        dtype=np.float64,
    )
    aligned[np.ix_(remaining_indices, remaining_indices)] = similarity
    return full_unit_ids, aligned


def _load_graph(
    path: Path,
    metric_key: str,
    *,
    align_dropped_units: bool,
) -> Tuple[List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        if "unit_ids" not in npz or metric_key not in npz:
            missing = [key for key in ("unit_ids", metric_key) if key not in npz]
            raise KeyError(f"Missing {missing} in {path}")
        semantics = str(
            np.asarray(
                npz.get("graph_semantics", np.array([""])),
                dtype=object,
            ).reshape(-1)[0]
        )
        if semantics != UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS:
            raise ValueError(
                f"Redundancy graph {path} uses graph_semantics={semantics!r}; "
                f"expected {UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS!r}. "
                "Regenerate the redundancy graph with scripts/run_cache_redundancy.sh."
            )
        unit_ids = [str(value) for value in list(npz["unit_ids"])]
        similarity = np.asarray(npz[metric_key], dtype=np.float64)
        dropped_unit_ids = [
            str(value)
            for value in list(npz.get("dropped_unit_ids", np.array([], dtype=object)))
        ]
        declared_remaining = [
            str(value)
            for value in list(npz.get("remaining_unit_ids", np.array([], dtype=object)))
        ]
    expected = (len(unit_ids), len(unit_ids))
    if similarity.shape != expected:
        raise ValueError(f"Invalid similarity shape {similarity.shape} in {path}; expected {expected}")
    if not np.array_equal(similarity, similarity.T, equal_nan=True):
        raise ValueError(f"Unit-output similarity matrix in {path} is not symmetric")
    expected_diagonal = (
        np.zeros(len(unit_ids), dtype=np.float64)
        if metric_key == "angular_mean"
        else np.ones(len(unit_ids), dtype=np.float64)
    )
    if not np.array_equal(np.diag(similarity), expected_diagonal):
        expected_value = 0 if metric_key == "angular_mean" else 1
        raise ValueError(
            f"Unit-output {metric_key} diagonal in {path} is not exactly {expected_value}"
        )
    if declared_remaining and declared_remaining != unit_ids:
        raise ValueError(f"remaining_unit_ids in {path} do not match unit_ids")
    if align_dropped_units:
        unit_ids, similarity = _align_graph_to_full_unit_catalog(
            path=path,
            unit_ids=unit_ids,
            similarity=similarity,
            dropped_unit_ids=dropped_unit_ids,
        )
    return unit_ids, similarity


def _validate_compatible(
    expected_unit_ids: Sequence[str],
    expected_shape: Tuple[int, ...],
    unit_ids: Sequence[str],
    similarity: np.ndarray,
    path: Path,
) -> None:
    if list(unit_ids) != list(expected_unit_ids):
        raise ValueError(f"unit_ids in {path} do not match the other selected graphs")
    if similarity.shape != expected_shape:
        raise ValueError(f"Similarity shape {similarity.shape} in {path} != {expected_shape}")


def parse_dataset_list(dataset: str) -> List[str]:
    raw = str(dataset or "").strip()
    datasets = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    if not datasets:
        raise ValueError("Dataset is required")
    invalid = [name for name in datasets if name.lower() in {"mean", "min"}]
    if invalid:
        raise ValueError(
            "Dataset aggregation selectors are no longer supported: "
            f"{invalid}. Specify datasets explicitly."
        )
    out: List[str] = []
    seen: set[str] = set()
    for name in datasets:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def load_similarity(
    *,
    experiment: str,
    dataset: str,
    token_region: str,
    metric: str,
    output_root: Path = OUTPUT_ROOT_DEFAULT,
    artifact_stage: str = "redundancy",
    align_dropped_units: bool = False,
) -> Tuple[List[str], np.ndarray, Dict[str, object]]:
    """Load raw unit-output similarity, aggregating regions/datasets if requested."""
    stage_name = str(artifact_stage or "").strip()
    if not stage_name or Path(stage_name).name != stage_name:
        raise ValueError(f"artifact_stage must be one directory name, got {artifact_stage!r}")
    token_selector = str(token_region).strip().lower()
    metric_key = f"{metric.strip().lower()}_mean"
    datasets = parse_dataset_list(dataset)
    unit_ids: Optional[List[str]] = None
    dataset_matrices: List[np.ndarray] = []
    source_paths: List[Path] = []
    regions_by_dataset: Dict[str, List[str]] = {}

    for dataset_name in datasets:
        red_dir = Path(output_root) / experiment / dataset_name / stage_name
        if token_selector == SPECIAL_PATCH_MEAN:
            regions = ["special", "patch"]
        else:
            regions = [token_selector]
        region_matrices: List[np.ndarray] = []
        for region in regions:
            path = resolve_redundancy_graph_path(red_dir, region)
            if path is None:
                raise FileNotFoundError(
                    f"Missing similarity graph for token region {region!r} under {red_dir}"
                )
            current_ids, current_similarity = _load_graph(
                path,
                metric_key,
                align_dropped_units=bool(align_dropped_units),
            )
            if unit_ids is None:
                unit_ids = current_ids
            else:
                _validate_compatible(
                    unit_ids,
                    dataset_matrices[0].shape if dataset_matrices else region_matrices[0].shape,
                    current_ids,
                    current_similarity,
                    path,
                )
            if region_matrices:
                _validate_compatible(
                    current_ids,
                    region_matrices[0].shape,
                    current_ids,
                    current_similarity,
                    path,
                )
            region_matrices.append(current_similarity)
            source_paths.append(path)
        dataset_matrix = (
            region_matrices[0]
            if len(region_matrices) == 1
            else reduce_similarity_matrices(region_matrices)
        )
        dataset_matrices.append(dataset_matrix)
        regions_by_dataset[str(dataset_name)] = regions

    assert unit_ids is not None
    similarity = (
        reduce_similarity_matrices(dataset_matrices)
        if len(dataset_matrices) > 1
        else dataset_matrices[0]
    )
    provenance: Dict[str, object] = {
        "metric_key": metric_key,
        "artifact_stage": stage_name,
        "aligned_dropped_units": bool(align_dropped_units),
        "graph_semantics": UNIT_OUTPUT_PAIRWISE_GRAPH_SEMANTICS,
        "dataset_selector": ",".join(datasets),
        "datasets": list(datasets),
        "dataset_aggregation": DATASET_LIST_AGGREGATION if len(datasets) > 1 else None,
        "token_region_selector": token_selector,
        "aggregation_order": ["token_region", "dataset"],
        "token_regions_by_dataset": regions_by_dataset,
        "source_graphs": [str(path) for path in source_paths],
    }
    return unit_ids, similarity, provenance

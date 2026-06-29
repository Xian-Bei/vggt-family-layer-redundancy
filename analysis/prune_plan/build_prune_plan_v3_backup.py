"""Build per-backbone multi-plan JSONs from redundancy graph (triangle greedy)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.backbones.registry import get_backbone
from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan
from analysis.prune_plan.training_cache_units import required_cache_unit_ids
from analysis.redundancy.similarity_v3 import (
    DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
    REDUNDANCY_GRAPH_REGIONS,
    list_redundancy_graph_regions,
    parse_exclude_token_regions,
    regions_for_aggregation,
    resolve_redundancy_graph_path,
)
from analysis.prune_plan.triangle_prune import build_triangle_prune_actions
from analysis.prune_plan.visualize_triangle_prune import render_triangle_prune_heatmap
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi prune plans (v3) into analysis_outputs_v3/<backbone>/.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, or mean/min to aggregate all datasets available for this experiment.",
    )
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--metric", type=str, default="cka", help="Only cka is used (reads {metric}_mean).")
    p.add_argument(
        "--token-region",
        type=str,
        choices=(*REDUNDANCY_GRAPH_REGIONS, "mean", "min"),
        default="all",
        help=(
            "Token similarity graph to use. mean/min combines all token-region graphs "
            "available on disk for each dataset."
        ),
    )
    p.add_argument(
        "--exclude-token-regions",
        type=str,
        default="",
        help=(
            "Comma-separated token regions to exclude from mean/min aggregation "
            "(default: none)."
        ),
    )
    p.add_argument("--similarity-threshold", type=float, default=0.9)
    p.add_argument("--max-triangles", type=int, default=2)
    p.add_argument(
        "--max-lora-anchors",
        type=int,
        default=1,
        help="LoRA prefix expansion upper bound L (emit l=1..L for each triangle prefix).",
    )
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=None,
        help="Output directory for multi plans (default: analysis_outputs_v3/<backbone>).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Legacy single-file output path; if set, only plan_tT_lL is written there.",
    )
    p.add_argument(
        "--viz-out-dir",
        type=Path,
        default=None,
        help="If set, write CKA heatmap + triangle corner markers after building plan.",
    )
    p.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="Optional HF model override used to derive explicit cache unit ids.",
    )
    p.add_argument(
        "--cache-device",
        type=str,
        default="cuda",
        help="Device used to load model when deriving explicit cache unit ids.",
    )
    return p


def _plan_name(t: int, l: int) -> str:
    return f"plan_t{int(t)}_l{int(l)}"


def _default_plan_root(backbone: str) -> Path:
    return OUTPUT_ROOT_DEFAULT / str(backbone).strip().lower()


def _unit_id_to_index(unit_id: str) -> int:
    if str(unit_id).strip() == "layer_-1":
        return -1
    kind, layer_s = str(unit_id).split("_", 1)
    layer = int(layer_s)
    base = 2 * layer
    return base if kind == "frame" else (base + 1)


REDUNDANCY_AGGREGATIONS = frozenset({"mean", "min"})


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
    exclude_token_regions: Sequence[str] = DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
) -> Tuple[List[str], np.ndarray, List[Path], List[str]]:
    red_dir = stage_dir(experiment, dataset, "redundancy")
    if token_region in REDUNDANCY_AGGREGATIONS:
        regions = regions_for_aggregation(
            list_redundancy_graph_regions(red_dir),
            exclude_token_regions=exclude_token_regions,
        )
        if not regions:
            raise FileNotFoundError(
                f"No redundancy graphs left for mean/min aggregation under {red_dir} "
                f"after excluding {list(exclude_token_regions)!r}"
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
    *,
    exclude_token_regions: Sequence[str] = DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
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
            usable = bool(
                regions_for_aggregation(regions, exclude_token_regions=exclude_token_regions)
            )
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
    exclude_token_regions: Sequence[str] = DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
) -> Tuple[List[str], np.ndarray, Dict[str, object]]:
    dataset_selector = str(dataset).strip().lower()
    token_selector = str(token_region).strip().lower()
    metric_key = f"{metric.strip().lower()}_mean"
    datasets = (
        _available_datasets(
            experiment,
            token_selector,
            exclude_token_regions=exclude_token_regions,
        )
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
            exclude_token_regions=exclude_token_regions,
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
        "excluded_token_regions": [str(region) for region in exclude_token_regions],
        "aggregation_order": ["token_region", "dataset"],
        "token_regions_by_dataset": regions_by_dataset,
        "source_graphs": [str(path) for path in source_paths],
    }
    return unit_ids, sim, provenance


def _build_plan_from_prefix(
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
        # global-prefix LoRA expansion around anchor i: i, i-1, ... i-(l-1)
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


def run_build_prune_plan(
    *,
    experiment: str,
    dataset: str,
    backbone: str = "vggt",
    metric: str = "cka",
    token_region: str = "all",
    exclude_token_regions: Sequence[str] = DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
    similarity_threshold: float = 0.9,
    max_triangles: int = 2,
    max_lora_anchors: int = 1,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    prune_plan_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    viz_out_dir: Optional[Path] = None,
    hf_model: str = "",
    cache_device: str = "cuda",
) -> List[Path]:
    """Build multi plans from redundancy graph; returns written paths."""
    unit_ids, sim, similarity_source = load_similarity(
        experiment=experiment,
        dataset=dataset,
        token_region=token_region,
        metric=metric,
        exclude_token_regions=exclude_token_regions,
    )
    print(
        f"Similarity source: datasets={similarity_source['datasets']} "
        f"token_regions={similarity_source['token_regions_by_dataset']} "
        f"excluded_token_regions={similarity_source['excluded_token_regions']} "
        f"metric={similarity_source['metric_key']}",
        flush=True,
    )

    _actions, _extra, picks = build_triangle_prune_actions(
        sim,
        unit_ids,
        threshold=float(similarity_threshold),
        max_triangles=int(max_triangles),
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
    )

    written: List[Path] = []
    source_cache = (
        ""
        if dataset.strip().lower() in REDUNDANCY_AGGREGATIONS
        else str(stage_dir(experiment, dataset, "cache"))
    )
    plan_root = prune_plan_root or _default_plan_root(backbone)
    plan_root.mkdir(parents=True, exist_ok=True)
    adapter = get_backbone(backbone)
    hf = str(hf_model or adapter.default_hf_model())
    model = adapter.load_pretrained(hf_model=hf, device=str(cache_device), dtype=None, eval_mode=True)

    for k, pick in enumerate(picks):
        print(
            f"triangle[{k}] (i,j)=({pick.i},{pick.j}) span={pick.span} area={pick.area} "
            f"anchor={pick.unit_i} drop={len(pick.dropped_unit_ids)} units",
            flush=True,
        )
    max_t = max(0, min(int(max_triangles), len(picks)))
    max_l = max(1, int(max_lora_anchors))
    if output_path is not None:
        # Legacy mode: write only the terminal plan to explicit output.
        t = max(1, max_t if max_t > 0 else 1)
        l = max_l
        plan, _lora_idxs = _build_plan_from_prefix(
            picks_prefix=picks[:t],
            lora_prefix=l,
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            backbone=backbone,
            dataset=dataset,
            num_units=int(sim.shape[0]),
            source_cache=source_cache,
        )
        plan.extra["similarity_source"] = similarity_source
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.save(output_path)
        written.append(output_path)
        print(
            f"Wrote {output_path} (drop={len(plan.drop)} units, lora={len(plan.lora)} units, triangles={t}, lora_prefix={l})",
            flush=True,
        )
    else:
        for t in range(1, max_t + 1):
            picks_prefix = picks[:t]
            for l in range(1, max_l + 1):
                name = _plan_name(t, l)
                out_path = plan_root / f"{name}.json"
                plan, lora_idxs = _build_plan_from_prefix(
                    picks_prefix=picks_prefix,
                    lora_prefix=l,
                    lora_rank=int(lora_rank),
                    lora_alpha=float(lora_alpha),
                    backbone=backbone,
                    dataset=dataset,
                    num_units=int(sim.shape[0]),
                    source_cache=source_cache,
                )
                plan.extra["similarity_source"] = similarity_source
                cache_unit_ids = sorted(required_cache_unit_ids(plan, model), key=_unit_id_to_index)
                cache_unit_indices = [_unit_id_to_index(uid) for uid in cache_unit_ids]
                plan.extra["cache_unit_ids"] = cache_unit_ids
                plan.extra["cache_unit_indices"] = cache_unit_indices
                plan.save(out_path)
                written.append(out_path)
                print(
                    f"Wrote {out_path} (drop={len(plan.drop)} units, lora={len(plan.lora)} units, triangles={t}, lora_prefix={l})",
                    flush=True,
                )
                if viz_out_dir is not None:
                    viz_path = render_triangle_prune_heatmap(
                        sim,
                        unit_ids,
                        picks_prefix,
                        out_dir=viz_out_dir,
                        threshold=float(similarity_threshold),
                        tag=f"{name}_{backbone}",
                        lora_unit_indices=lora_idxs,
                        cache_unit_indices=cache_unit_indices,
                    )
                    print(f"Wrote visualization {viz_path}", flush=True)

    return written


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_build_prune_plan(
        experiment=args.experiment,
        dataset=args.dataset,
        backbone=args.backbone,
        metric=args.metric,
        token_region=args.token_region,
        exclude_token_regions=parse_exclude_token_regions(args.exclude_token_regions),
        similarity_threshold=float(args.similarity_threshold),
        max_triangles=int(args.max_triangles),
        max_lora_anchors=int(args.max_lora_anchors),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        prune_plan_root=args.prune_plan_root,
        output_path=args.output,
        viz_out_dir=args.viz_out_dir,
        hf_model=str(args.hf_model),
        cache_device=str(args.cache_device),
    )


if __name__ == "__main__":
    main()

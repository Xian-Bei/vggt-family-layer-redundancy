"""Build recovery-plan bundles from a prune-only span bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    UnitLoraSpec,
    load_prune_plan_bundle,
    prune_plan_bundle_path,
    unit_index_to_unit_id,
)
from analysis.prune_plan.build_prune_plan_common import (
    build_plan_from_prefix as _build_plan_from_prefix,
    unit_id_to_index as _unit_id_to_index,
)
from analysis.prune_plan.training_cache_units import required_cache_unit_ids
from analysis.prune_plan.visualize_triangle_prune import render_span_unit_action_grid
from analysis.prune_plan.recovery_plan import (
    TRAINING_FREE_RECOVERY_METHODS,
    TRAINING_PLAN_BUNDLE_SCHEMA_VERSION,
    TRAINING_PLAN_METHODS,
    execution_method,
    training_plan_name,
    training_plan_path,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Build recovery-plan bundle from prune_plan_all.json.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--prune-plan-root", type=Path, required=True)
    p.add_argument(
        "--recovery-plan-root",
        "--training-plan-root",
        dest="recovery_plan_root",
        type=Path,
        default=None,
    )
    p.add_argument("--method", type=str, choices=TRAINING_PLAN_METHODS, default="recovery_lora")
    p.add_argument("--lora-prefix", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--source-cache", type=str, default="")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--cache-device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--viz-out-dir", type=Path, default=None)
    return p


def _pick_from_span_record(record: dict) -> SimpleNamespace:
    return SimpleNamespace(
        i=int(record.get("i", 0)),
        j=int(record.get("j", 0)),
        span=int(record.get("span", 0)),
        unit_i=str(record.get("unit_i", "")),
        unit_j=str(record.get("unit_j", "")),
    )


def _anchor_indices_from_pick(pick: SimpleNamespace, lora_prefix: int) -> List[int]:
    out: List[int] = []
    for off in range(max(1, int(lora_prefix))):
        idx = int(pick.i) - off
        if idx < 0:
            break
        out.append(idx)
    return sorted(set(out))


def _unit_lora_specs(indices: List[int], *, rank: int, alpha: float, teacher_index: int) -> dict[str, UnitLoraSpec]:
    return {
        str(int(idx)): UnitLoraSpec(
            rank=int(rank),
            alpha=float(alpha),
            dropout=0.0,
            target_submodules=("qkv", "proj"),
            alignment_teacher_unit_index=int(teacher_index),
        )
        for idx in sorted(set(int(x) for x in indices))
    }


def run_build_recovery_plan(
    *,
    experiment: str,
    backbone: str,
    prune_plan_root: Path,
    training_plan_root: Optional[Path] = None,
    method: str = "recovery_lora",
    lora_prefix: int = 1,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    source_cache: str = "",
    hf_model: str = "",
    cache_device: str = "cuda",
    output_path: Optional[Path] = None,
    viz_out_dir: Optional[Path] = None,
) -> Path:
    method_key = str(method).strip().lower()
    exec_method = execution_method(method_key)
    if method_key not in TRAINING_PLAN_METHODS:
        raise ValueError(f"unknown recovery method: {method}")
    prefix = int(lora_prefix)
    if prefix < 1:
        raise ValueError(f"lora_prefix must be >= 1, got {prefix}")
    rank = int(lora_rank)
    if exec_method == "lora_prune" and rank <= 0:
        raise ValueError(f"lora_rank must be > 0 for recovery_lora, got {rank}")

    prune_bundle = load_prune_plan_bundle(prune_plan_root)
    spans_raw = prune_bundle.get("spans", {})
    if not isinstance(spans_raw, dict) or not spans_raw:
        raise ValueError(f"No spans in prune bundle: {prune_plan_bundle_path(prune_plan_root)}")
    num_units = int(prune_bundle.get("num_units", 0))
    dataset = str(prune_bundle.get("dataset", ""))
    source_cache_s = str(source_cache or "")

    out_root = training_plan_root or prune_plan_root
    name = training_plan_name(method=method_key, lora_prefix=prefix, lora_rank=rank)
    out_path = output_path or training_plan_path(out_root, name)
    visualization_root = viz_out_dir or out_path.parent

    adapter = get_backbone(backbone)
    model = None
    if method_key not in TRAINING_FREE_RECOVERY_METHODS:
        hf = str(hf_model or adapter.default_hf_model())
        model = adapter.load_pretrained(hf_model=hf, device=str(cache_device), dtype=None, eval_mode=True)

    span_to_drop_indices: dict[int, list[int]] = {}
    span_to_lora_indices: dict[int, list[int]] = {}
    span_to_cache_indices: dict[int, list[int]] = {}
    span_records: dict[str, dict] = {}

    for span_key, span_record_raw in sorted(
        spans_raw.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if not isinstance(span_record_raw, dict):
            continue
        span = int(span_key)
        pick = _pick_from_span_record(span_record_raw)
        source_idx = int(pick.i)
        target_idx = int(pick.j)
        source_unit_id = unit_index_to_unit_id(source_idx)
        target_unit_id = unit_index_to_unit_id(target_idx)
        if method_key in TRAINING_FREE_RECOVERY_METHODS:
            temp_plan = None
            lora_indices: List[int] = []
            train_unit_indices: List[int] = []
            cache_unit_ids = sorted({source_unit_id, target_unit_id}, key=_unit_id_to_index)
            cache_unit_indices = [_unit_id_to_index(uid) for uid in cache_unit_ids]
            lora_specs = {}
        else:
            temp_plan, lora_indices = _build_plan_from_prefix(
                picks_prefix=[pick],
                lora_prefix=prefix,
                lora_rank=rank,
                lora_alpha=float(lora_alpha),
                backbone=backbone,
                dataset=dataset,
                num_units=num_units,
                source_cache=source_cache_s,
            )
            assert model is not None
            cache_unit_ids = sorted(required_cache_unit_ids(temp_plan, model), key=_unit_id_to_index)
            cache_unit_indices = [_unit_id_to_index(uid) for uid in cache_unit_ids]
            train_unit_indices = _anchor_indices_from_pick(pick, prefix)
            lora_specs = (
                temp_plan.lora
                if exec_method == "lora_prune"
                else _unit_lora_specs(
                    train_unit_indices,
                    rank=max(1, rank),
                    alpha=float(lora_alpha),
                    teacher_index=target_idx,
                )
            )
        span_to_drop_indices[span] = [int(x) for x in list(span_record_raw.get("drop", []))]
        span_to_lora_indices[span] = sorted(int(x) for x in lora_indices)
        span_to_cache_indices[span] = cache_unit_indices
        span_records[str(span)] = {
            "span": span,
            "prune_selector": f"prune_span{span}",
            "drop": span_to_drop_indices[span],
            "lora": (
                {str(int(k)): asdict(v) for k, v in temp_plan.lora.items()}
                if exec_method == "lora_prune" and temp_plan is not None
                else {}
            ),
            "lora_indices": sorted(int(x) for x in lora_indices),
            "train_unit_indices": train_unit_indices,
            "train_unit_ids": [unit_index_to_unit_id(int(idx)) for idx in train_unit_indices],
            "source_unit_index": source_idx,
            "source_unit_id": source_unit_id,
            "target_unit_index": target_idx,
            "target_unit_id": target_unit_id,
            "cache_unit_ids": [str(x) for x in cache_unit_ids],
            "cache_unit_indices": cache_unit_indices,
            "alignment_teacher_unit_index": target_idx,
            "anchor_unit_index": source_idx,
            "cache_rule": (
                "recovery_boundary_source_target_units"
                if method_key in TRAINING_FREE_RECOVERY_METHODS
                else "required_cache_unit_ids_from_lora_anchor_replay"
            ),
            "original_train_lora_equivalent": (
                {str(int(k)): asdict(v) for k, v in lora_specs.items()}
                if exec_method == "train_original"
                else {}
            ),
        }

    bundle = {
        "schema_version": TRAINING_PLAN_BUNDLE_SCHEMA_VERSION,
        "method": method_key,
        "name": name,
        "backbone": str(backbone),
        "dataset": dataset,
        "num_units": num_units,
        "source_prune_plan": str(prune_plan_bundle_path(prune_plan_root)),
        "source_cache": source_cache_s,
        "lora_prefix": prefix,
        "lora_rank": rank,
        "lora_alpha": float(lora_alpha),
        "spans": span_records,
        "extra": {
            "recovery_train_rule": "train original parameters of the same unit anchors that recovery_lora would adapt",
            "training_free_recovery_rule": "fit one boundary operator from cached source/target teacher activations",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    grid_path = render_span_unit_action_grid(
        num_units=num_units,
        span_to_drop_indices=span_to_drop_indices,
        span_to_lora_indices=span_to_lora_indices,
        span_to_cache_indices=span_to_cache_indices,
        out_dir=visualization_root,
        backbone=backbone,
        lora_prefix=prefix,
        tag=f"recovery_plan_{backbone}_{name}_span_unit_grid",
        lora_label=(
            "+LoRA unit"
            if exec_method == "lora_prune"
            else ("Recovery boundary" if method_key in TRAINING_FREE_RECOVERY_METHODS else "Train original unit")
        ),
        lora_marker="+" if exec_method == "lora_prune" else "o",
        title=(
            f"{backbone}: span-by-unit actions (LoRA prefix l{prefix})"
            if exec_method == "lora_prune"
            else (
                f"{backbone}: span-by-unit actions ({method_key})"
                if method_key in TRAINING_FREE_RECOVERY_METHODS
                else f"{backbone}: span-by-unit actions (train-original prefix l{prefix})"
            )
        ),
    )
    print(f"Wrote recovery plan {out_path}", flush=True)
    print(f"Wrote recovery action grid {grid_path}", flush=True)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_build_recovery_plan(
        experiment=args.experiment,
        backbone=args.backbone,
        prune_plan_root=args.prune_plan_root,
        training_plan_root=args.recovery_plan_root,
        method=args.method,
        lora_prefix=args.lora_prefix,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        source_cache=args.source_cache,
        hf_model=args.hf_model,
        cache_device=args.cache_device,
        output_path=args.output,
        viz_out_dir=args.viz_out_dir,
    )


if __name__ == "__main__":
    main()

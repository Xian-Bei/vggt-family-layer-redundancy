"""Build per-backbone multi-plan JSONs from redundancy graph (triangle greedy)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.backbones.registry import get_backbone
from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan
from analysis.prune_plan.training_cache_units import required_cache_unit_ids
from analysis.redundancy.triangle_prune import build_triangle_prune_actions
from analysis.redundancy.visualize_triangle_prune import render_triangle_prune_heatmap
from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT, stage_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build multi prune plans (v3) into analysis_outputs_v3/<backbone>/.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--metric", type=str, default="cka", help="Only cka is used (reads {metric}_mean).")
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
    red_dir = stage_dir(experiment, dataset, "redundancy")
    npz = np.load(red_dir / "redundancy_graph_v3.npz", allow_pickle=True)
    unit_ids = [str(u) for u in list(npz["unit_ids"])]
    key = f"{metric.strip().lower()}_mean"
    if key not in npz:
        raise KeyError(f"Missing {key} in redundancy npz")
    sim = np.asarray(npz[key], dtype=np.float64)

    _actions, _extra, picks = build_triangle_prune_actions(
        sim,
        unit_ids,
        threshold=float(similarity_threshold),
        max_triangles=int(max_triangles),
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
    )

    written: List[Path] = []
    source_cache = str(stage_dir(experiment, dataset, "cache"))
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

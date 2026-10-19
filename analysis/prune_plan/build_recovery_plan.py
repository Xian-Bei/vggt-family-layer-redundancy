"""Build recovery-plan bundles from a prune-only span bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from analysis.backbones.registry import get_backbone
from analysis.prune_plan import (
    bundle_plan_records,
    load_prune_plan_bundle,
    prune_plan_bundle_path,
)
from analysis.prune_plan.build_prune_plan_common import (
    build_recovery_plan_from_interval,
    unit_id_to_index as _unit_id_to_index,
)
from analysis.prune_plan.training_cache_units import required_cache_unit_ids
from analysis.prune_plan.recovery_plan import (
    CACHE_FIT_RECOVERY_METHODS,
    RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
    RECOVERY_PLAN_METHODS,
    TOKEN_GROUPED_RECOVERY_BASE_METHODS,
    execution_method,
    recovery_plan_name,
    recovery_plan_path,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Build recovery-plan bundle from prune_plan_all.json.")
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--backbone", type=str, default="vggt")
    p.add_argument("--prune-plan-root", type=Path, required=True)
    p.add_argument(
        "--recovery-plan-root",
        type=Path,
        default=None,
    )
    p.add_argument("--method", type=str, choices=RECOVERY_PLAN_METHODS, default="recovery_lora")
    p.add_argument("--lora-prefix", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--source-cache", type=str, default="")
    p.add_argument("--hf-model", type=str, default="")
    p.add_argument("--cache-device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=None)
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
    anchor = int(pick.i) - 1
    for off in range(max(1, int(lora_prefix))):
        idx = anchor - off
        if idx < 0:
            break
        out.append(idx)
    return sorted(set(out))


def run_build_recovery_plan(
    *,
    experiment: str,
    backbone: str,
    prune_plan_root: Path,
    recovery_plan_root: Optional[Path] = None,
    method: str = "recovery_lora",
    lora_prefix: int = 1,
    lora_rank: int = 32,
    lora_alpha: float = 64.0,
    source_cache: str = "",
    hf_model: str = "",
    cache_device: str = "cuda",
    output_path: Optional[Path] = None,
) -> Path:
    method_key = str(method).strip().lower()
    exec_method = execution_method(method_key)
    if method_key not in RECOVERY_PLAN_METHODS:
        raise ValueError(f"unknown recovery method: {method}")
    prefix = int(lora_prefix)
    if prefix < 1:
        raise ValueError(f"lora_prefix must be >= 1, got {prefix}")
    rank = int(lora_rank)
    if exec_method == "lora_prune" and rank <= 0:
        raise ValueError(f"lora_rank must be > 0 for recovery_lora, got {rank}")

    prune_bundle = load_prune_plan_bundle(prune_plan_root)
    spans_raw = prune_bundle.get("spans", {})
    if not isinstance(spans_raw, dict):
        raise ValueError(
            f"Invalid spans object in prune bundle: "
            f"{prune_plan_bundle_path(prune_plan_root)}"
        )
    if method_key in CACHE_FIT_RECOVERY_METHODS:
        if not bundle_plan_records(prune_bundle):
            raise ValueError(
                f"No pruning plans in bundle: "
                f"{prune_plan_bundle_path(prune_plan_root)}"
            )
    elif not spans_raw:
        raise ValueError(
            f"No legacy span records for trainable recovery in prune bundle: "
            f"{prune_plan_bundle_path(prune_plan_root)}"
        )
    num_units = int(prune_bundle.get("num_units", 0))
    unit_ids = [str(x) for x in prune_bundle.get("unit_ids", [])]
    dataset = str(prune_bundle.get("dataset", ""))
    source_cache_s = str(source_cache or "")

    out_root = recovery_plan_root or prune_plan_root
    name = recovery_plan_name(method=method_key, lora_prefix=prefix, lora_rank=rank)
    out_path = output_path or recovery_plan_path(out_root)
    adapter = get_backbone(backbone)
    model = None
    if method_key not in CACHE_FIT_RECOVERY_METHODS:
        hf = str(hf_model or adapter.default_hf_model())
        model = adapter.load_pretrained(hf_model=hf, device=str(cache_device), dtype=None, eval_mode=True)

    span_records: dict[str, dict] = {}
    if method_key not in CACHE_FIT_RECOVERY_METHODS:
        for span_key, span_record_raw in sorted(
            spans_raw.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            if not isinstance(span_record_raw, dict):
                continue
            span = int(span_key)
            pick = _pick_from_span_record(span_record_raw)
            if int(pick.i) <= 0:
                raise ValueError(
                    f"Prune plan span {span} deletes protected unit 0; rebuild pruning plans first"
                )
            target_idx = int(pick.j)
            temp_plan, lora_indices = build_recovery_plan_from_interval(
                picks_prefix=[pick],
                lora_prefix=prefix,
                lora_rank=rank,
                lora_alpha=float(lora_alpha),
                backbone=backbone,
                dataset=dataset,
                num_units=num_units,
                unit_ids=unit_ids,
                source_cache=source_cache_s,
            )
            assert model is not None
            cache_unit_ids = sorted(required_cache_unit_ids(temp_plan, model), key=_unit_id_to_index)
            train_unit_indices = _anchor_indices_from_pick(pick, prefix)
            span_records[str(span)] = {
                "cache_unit_ids": [str(x) for x in cache_unit_ids],
                "lora_indices": sorted(int(x) for x in lora_indices),
                "train_unit_indices": train_unit_indices,
            }

    method_entry: dict = {"method": method_key}
    if method_key not in CACHE_FIT_RECOVERY_METHODS:
        method_entry.update(
            {
                "lora_prefix": prefix,
                "lora_rank": rank,
                "lora_alpha": float(lora_alpha),
                "spans": span_records,
            }
        )
    if out_path.is_file():
        bundle = json.loads(out_path.read_text(encoding="utf-8"))
        if str(bundle.get("schema_version", "")) != RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Refusing to update incompatible recovery plan {out_path}")
        if str(bundle.get("backbone", "")) != str(backbone):
            raise ValueError(f"Recovery plan backbone mismatch at {out_path}")
    else:
        bundle = {
            "schema_version": RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
            "backbone": str(backbone),
            "dataset": dataset,
            "num_units": num_units,
            "unit_ids": unit_ids,
            "methods": {},
        }
    methods = bundle.setdefault("methods", {})
    if not isinstance(methods, dict):
        raise ValueError(f"Invalid methods object in {out_path}")
    methods[name] = method_entry
    dependency_name = ""
    if (
        str(backbone).strip().lower() == "pi3"
        and method_key in TOKEN_GROUPED_RECOVERY_BASE_METHODS
    ):
        dependency_method = TOKEN_GROUPED_RECOVERY_BASE_METHODS[method_key]
        dependency_name = recovery_plan_name(
            method=dependency_method,
            lora_prefix=prefix,
            lora_rank=rank,
        )
        methods.setdefault(dependency_name, {"method": dependency_method})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"Updated {name} in unified recovery plan {out_path}", flush=True)
    if dependency_name:
        print(
            f"Ensured Pi3 token-group dependency {dependency_name} in {out_path}",
            flush=True,
        )
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_build_recovery_plan(
        experiment=args.experiment,
        backbone=args.backbone,
        prune_plan_root=args.prune_plan_root,
        recovery_plan_root=args.recovery_plan_root,
        method=args.method,
        lora_prefix=args.lora_prefix,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        source_cache=args.source_cache,
        hf_model=args.hf_model,
        cache_device=args.cache_device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

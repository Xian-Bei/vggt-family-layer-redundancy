from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis.prune_plan import (
    UnitLoraSpec,
    UnitPrunePlan,
    load_named_prune_plan,
    unit_index_to_unit_id,
)

TRAINING_PLAN_BUNDLE_SCHEMA_VERSION = "training_plan_bundle_v1"
TRAINING_PLAN_METHODS = ("lora_prune", "train_original")

_SPAN_SELECTOR_RE = re.compile(r"^prune_span(?P<span>\d+)$")


def training_plan_name(*, method: str, lora_prefix: int, lora_rank: int = 0) -> str:
    method_key = str(method).strip().lower()
    prefix = int(lora_prefix)
    if method_key == "lora_prune":
        return f"training_lora_l{prefix}_r{int(lora_rank)}"
    if method_key == "train_original":
        return f"training_original_l{prefix}"
    raise ValueError(f"unknown training plan method: {method!r}")


def training_plan_filename(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        raise ValueError("training plan name must be non-empty")
    return key if key.endswith(".json") else f"{key}.json"


def training_plan_path(root: Path, name: str) -> Path:
    return Path(root) / training_plan_filename(name)


def load_training_plan_bundle(root: Path, name: str) -> Dict[str, Any]:
    path = training_plan_path(root, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing training plan bundle: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid training plan bundle: expected object at {path}")
    schema = str(raw.get("schema_version", ""))
    if schema != TRAINING_PLAN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid training plan bundle schema at {path}: {schema!r} "
            f"(expected {TRAINING_PLAN_BUNDLE_SCHEMA_VERSION!r})"
        )
    return raw


def list_training_plan_names(root: Path) -> List[str]:
    out: List[str] = []
    for path in sorted(Path(root).glob("training_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict) and str(raw.get("schema_version", "")) == TRAINING_PLAN_BUNDLE_SCHEMA_VERSION:
            out.append(path.stem)
    return out


def span_from_prune_selector(selector: str) -> int:
    m = _SPAN_SELECTOR_RE.match(str(selector or "").strip())
    if not m:
        raise ValueError(f"expected prune span selector like prune_span10, got {selector!r}")
    return int(m.group("span"))


def unit_index_to_cache_unit_id(index: int) -> str:
    return unit_index_to_unit_id(int(index))


def _lora_spec_from_raw(raw: Dict[str, Any]) -> UnitLoraSpec:
    ati = raw.get("alignment_teacher_unit_index")
    return UnitLoraSpec(
        rank=int(raw.get("rank", 32)),
        alpha=float(raw.get("alpha", 64.0)),
        dropout=float(raw.get("dropout", 0.0)),
        target_submodules=tuple(raw.get("target_submodules", ("qkv", "proj"))),
        alignment_teacher_unit_index=int(ati) if ati is not None else None,
    )


def compose_training_unit_prune_plan(
    *,
    prune_plan_root: Path,
    training_plan_root: Path,
    prune_selector: str,
    training_plan: str,
) -> UnitPrunePlan:
    """Merge a prune-only span selector with a training-plan bundle.

    The returned object keeps the legacy ``UnitPrunePlan`` shape used by the
    current eval/train code, but the source of truth is split:
    prune bundle decides ``drop``; training bundle decides LoRA/train/cache.
    """
    prune_plan = load_named_prune_plan(prune_plan_root, prune_selector)
    span = span_from_prune_selector(prune_selector)
    bundle = load_training_plan_bundle(training_plan_root, training_plan)
    spans = bundle.get("spans", {})
    if not isinstance(spans, dict):
        raise ValueError("Invalid training plan bundle: spans must be an object")
    record = spans.get(str(span))
    if not isinstance(record, dict):
        raise FileNotFoundError(
            f"Training plan {training_plan!r} has no record for {prune_selector!r}"
        )
    method = str(bundle.get("method", "")).strip().lower()
    if method not in TRAINING_PLAN_METHODS:
        raise ValueError(f"Invalid training plan method: {method!r}")
    lora_raw = record.get("lora", {})
    lora: Dict[str, UnitLoraSpec] = {}
    if method == "lora_prune":
        if not isinstance(lora_raw, dict):
            raise ValueError(f"Invalid lora record for {prune_selector}: expected object")
        lora = {
            str(int(k)): _lora_spec_from_raw(v)
            for k, v in lora_raw.items()
            if isinstance(v, dict)
        }

    extra = dict(prune_plan.extra)
    extra.update(
        {
            "training_plan_schema_version": TRAINING_PLAN_BUNDLE_SCHEMA_VERSION,
            "training_plan_name": str(training_plan).removesuffix(".json"),
            "training_method": method,
            "lora_prefix": int(bundle.get("lora_prefix", 0)),
            "lora_rank": int(bundle.get("lora_rank", 0)),
            "lora_indices": [int(x) for x in list(record.get("lora_indices", []))],
            "train_unit_indices": [int(x) for x in list(record.get("train_unit_indices", []))],
            "cache_unit_ids": [str(x) for x in list(record.get("cache_unit_ids", []))],
            "cache_unit_indices": [int(x) for x in list(record.get("cache_unit_indices", []))],
        }
    )
    out = UnitPrunePlan(
        backbone=prune_plan.backbone,
        dataset=prune_plan.dataset,
        num_units=prune_plan.num_units,
        drop=list(prune_plan.drop),
        drop_length=len(prune_plan.drop),
        lora=lora,
        source_cache=str(bundle.get("source_cache", prune_plan.source_cache)),
        extra=extra,
    )
    out.validate()
    return out


def unit_lora_specs_to_json(lora: Dict[str, UnitLoraSpec]) -> Dict[str, Dict[str, Any]]:
    return {str(int(k)): asdict(v) for k, v in lora.items()}

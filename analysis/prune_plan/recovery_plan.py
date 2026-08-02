"""Unified recovery-plan helpers layered on top of ``prune_plan_all.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis.prune_plan import UnitLoraSpec, UnitPrunePlan, load_named_prune_plan, unit_index_to_unit_id

RECOVERY_PLAN_BUNDLE_FILENAME = "recovery_plan_all.json"
RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION = "recovery_plan_bundle_v3"
RECOVERY_PLAN_METHODS = (
    "recovery_lora",
    "recovery_train",
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_cos",
    "recovery_ghost_cos_more",
    "recovery_ghost_cos_scale",
    "recovery_ghost_mse_more",
    "recovery_ghost_mse_cos_more",
    "recovery_ghost_mse_cka_more",
    "recovery_ghost_mse_cos_cka_more",
    "recovery_ghost_mse_cos_balanced_more",
    "recovery_ghost_mse_cka_balanced_more",
    "recovery_ghost_mse_cos_cka_balanced_more",
    "recovery_ghost_mse_cka_ema_more",
    "recovery_ghost_mse_cos_cka_ema_half_more",
    "recovery_ghost_mse_cos_cka_ema_one_more",
    "recovery_ghost_mse_cka",
    "recovery_ghost_cka_only",
    "recovery_ghost_cka_ridge",
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_replaceme_cos_more",
    "recovery_replaceme_cos_scale",
    "recovery_prunecomp",
)
TRAINABLE_RECOVERY_METHODS = ("recovery_lora", "recovery_train")
TRAINING_FREE_RECOVERY_METHODS = (
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_cos",
    "recovery_ghost_cos_more",
    "recovery_ghost_cos_scale",
    "recovery_ghost_mse_more",
    "recovery_ghost_mse_cos_more",
    "recovery_ghost_mse_cka_more",
    "recovery_ghost_mse_cos_cka_more",
    "recovery_ghost_mse_cos_balanced_more",
    "recovery_ghost_mse_cka_balanced_more",
    "recovery_ghost_mse_cos_cka_balanced_more",
    "recovery_ghost_mse_cka_ema_more",
    "recovery_ghost_mse_cos_cka_ema_half_more",
    "recovery_ghost_mse_cos_cka_ema_one_more",
    "recovery_ghost_mse_cka",
    "recovery_ghost_cka_only",
    "recovery_ghost_cka_ridge",
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_replaceme_cos_more",
    "recovery_replaceme_cos_scale",
    "recovery_prunecomp",
)
REPLACEME_RECOVERY_METHODS = (
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_replaceme_cos_more",
    "recovery_replaceme_cos_scale",
)
GHOST_INTERVAL_RECOVERY_METHODS = (
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
)

_SPAN_SELECTOR_RE = re.compile(r"^prune_span(?P<span>\d+)$")


def execution_method(method: str) -> str:
    key = str(method).strip().lower()
    if key == "recovery_lora":
        return "lora_prune"
    if key == "recovery_train":
        return "train_original"
    return key


def recovery_plan_name(*, method: str, lora_prefix: int, lora_rank: int = 0) -> str:
    key = str(method).strip().lower()
    if key == "recovery_lora":
        return f"recovery_lora_l{int(lora_prefix)}_r{int(lora_rank)}"
    if key == "recovery_train":
        return f"recovery_train_l{int(lora_prefix)}"
    if key in TRAINING_FREE_RECOVERY_METHODS:
        return key
    raise ValueError(f"unknown recovery method: {method!r}")


def normalize_recovery_plan_name(name: str) -> str:
    return str(name or "").strip().removesuffix(".json")


def recovery_plan_path(root: Path) -> Path:
    return Path(root) / RECOVERY_PLAN_BUNDLE_FILENAME


def _validate_method_entry(name: str, entry: Dict[str, Any], path: Path) -> None:
    method = str(entry.get("method", "")).strip().lower()
    if method not in RECOVERY_PLAN_METHODS:
        raise ValueError(f"Invalid recovery method {method!r} for {name!r} at {path}")
    expected_name = recovery_plan_name(
        method=method,
        lora_prefix=int(entry.get("lora_prefix", 1)),
        lora_rank=int(entry.get("lora_rank", 0)),
    )
    if name != expected_name:
        raise ValueError(f"Recovery entry key {name!r} does not match its configuration {expected_name!r}")
    if method in TRAINING_FREE_RECOVERY_METHODS:
        unexpected = sorted(set(entry) - {"method"})
        if unexpected:
            raise ValueError(f"Training-free recovery entry {name!r} contains redundant fields: {unexpected}")
        return
    spans = entry.get("spans")
    if not isinstance(spans, dict) or not spans:
        raise ValueError(f"Trainable recovery entry {name!r} must contain non-empty spans")


def load_recovery_plan_all(root: Path) -> Dict[str, Any]:
    path = recovery_plan_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing unified recovery plan: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid recovery plan: expected object at {path}")
    if str(raw.get("schema_version", "")) != RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid recovery plan schema at {path}: {raw.get('schema_version')!r} "
            f"(expected {RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION!r})"
        )
    methods = raw.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError(f"Invalid recovery plan at {path}: missing non-empty methods")
    for name, entry in methods.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid recovery entry {name!r} at {path}: expected object")
        _validate_method_entry(str(name), entry, path)
    return raw


def load_recovery_plan_bundle(root: Path, name: str) -> Dict[str, Any]:
    """Return one named method view from the unified bundle."""
    plan_name = normalize_recovery_plan_name(name)
    all_plans = load_recovery_plan_all(root)
    entry = all_plans["methods"].get(plan_name)
    if not isinstance(entry, dict):
        raise FileNotFoundError(
            f"Recovery plan {plan_name!r} is not present in {recovery_plan_path(root)}"
        )
    return {
        "schema_version": all_plans["schema_version"],
        "backbone": all_plans.get("backbone", ""),
        "dataset": all_plans.get("dataset", ""),
        "num_units": all_plans.get("num_units", 0),
        "name": plan_name,
        **dict(entry),
    }


def list_recovery_plan_names(root: Path) -> List[str]:
    try:
        raw = load_recovery_plan_all(root)
    except FileNotFoundError:
        return []
    return sorted(str(name) for name in raw["methods"])


def span_from_prune_selector(selector: str) -> int:
    match = _SPAN_SELECTOR_RE.match(str(selector or "").strip())
    if not match:
        raise ValueError(f"expected prune span selector like prune_span10, got {selector!r}")
    return int(match.group("span"))


def _contiguous_drop_intervals(drop: List[int]) -> List[Tuple[int, int]]:
    values = sorted({int(x) for x in drop})
    if not values:
        return []
    intervals: List[Tuple[int, int]] = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        intervals.append((start, end))
        start = end = value
    intervals.append((start, end))
    return intervals


def _one_boundary_record(start: int, end: int, method: str) -> Dict[str, Any]:
    source_index = int(start) - 1
    target_index = int(end)
    source_id = unit_index_to_unit_id(source_index)
    target_id = unit_index_to_unit_id(target_index)
    auxiliary = (
        [f"replaceme_residual:{source_id}"] if method in REPLACEME_RECOVERY_METHODS else []
    )
    return {
        "drop_start_index": int(start),
        "drop_end_index": int(end),
        "source_unit_index": source_index,
        "source_unit_id": source_id,
        "target_unit_index": target_index,
        "target_unit_id": target_id,
        "cache_unit_ids": [source_id, target_id],
        "auxiliary_cache_ids": auxiliary,
    }


def _boundary_records(prune_plan: UnitPrunePlan, method: str) -> List[Dict[str, Any]]:
    """Derive baseline-specific recovery boundaries from an arbitrary drop set.

    The original Ghost and LinearPatch modes replace every removed unit when a
    plan has multiple intervals. Interval Ghost variants and ReplaceMe Multi-LT
    instead use one transform per maximal contiguous interval.
    """
    drop = sorted({int(x) for x in prune_plan.drop})
    if not drop:
        raise ValueError("Recovery requires a non-empty prune set")
    if drop[0] <= 0:
        raise ValueError("Recovery requires protected unit 0 to remain in the model")
    intervals = _contiguous_drop_intervals(drop)
    if method in GHOST_INTERVAL_RECOVERY_METHODS:
        starts_and_ends = intervals
    elif method == "recovery_linearpatch" or method.startswith("recovery_ghost"):
        if len(intervals) == 1:
            starts_and_ends = intervals
        else:
            starts_and_ends = [(index, index) for index in drop]
    elif method in REPLACEME_RECOVERY_METHODS:
        starts_and_ends = intervals
    elif method == "recovery_prunecomp":
        if len(intervals) != 1:
            raise ValueError(
                "Prune&Comp multi-interval recovery is intentionally unsupported"
            )
        starts_and_ends = intervals
    else:
        starts_and_ends = intervals
    return [_one_boundary_record(start, end, method) for start, end in starts_and_ends]


def recovery_record_for_selector(
    *,
    prune_plan: UnitPrunePlan,
    recovery_bundle: Dict[str, Any],
    prune_selector: str,
) -> Dict[str, Any]:
    method = str(recovery_bundle.get("method", "")).strip().lower()
    records = _boundary_records(prune_plan, method)
    record = dict(records[0])
    record["boundaries"] = records
    record["cache_unit_ids"] = sorted(
        {str(unit_id) for item in records for unit_id in item.get("cache_unit_ids", [])},
        key=lambda unit_id: int(unit_id.split("_", 1)[1]) * 2
        + (1 if unit_id.startswith("global_") else 0),
    )
    record["auxiliary_cache_ids"] = sorted(
        {
            str(auxiliary_id)
            for item in records
            for auxiliary_id in item.get("auxiliary_cache_ids", [])
        }
    )
    if method in TRAINING_FREE_RECOVERY_METHODS:
        record.update({"lora_indices": [], "train_unit_indices": [], "lora": {}})
        return record
    span = span_from_prune_selector(prune_selector)
    raw = recovery_bundle.get("spans", {}).get(str(span))
    if not isinstance(raw, dict):
        raise FileNotFoundError(
            f"Recovery plan {recovery_bundle.get('name')!r} has no record for {prune_selector!r}"
        )
    record.update(dict(raw))
    return record


def _lora_specs(entry: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, UnitLoraSpec]:
    teacher = int(record["target_unit_index"])
    return {
        str(int(index)): UnitLoraSpec(
            rank=int(entry.get("lora_rank", 32)),
            alpha=float(entry.get("lora_alpha", 64.0)),
            dropout=0.0,
            target_submodules=("qkv", "proj"),
            alignment_teacher_unit_index=teacher,
        )
        for index in record.get("lora_indices", [])
    }


def compose_recovery_unit_prune_plan(
    *,
    prune_plan_root: Path,
    recovery_plan_root: Path,
    prune_selector: str,
    recovery_plan: str,
) -> UnitPrunePlan:
    prune_plan = load_named_prune_plan(prune_plan_root, prune_selector)
    bundle = load_recovery_plan_bundle(recovery_plan_root, recovery_plan)
    method = str(bundle["method"])
    record = recovery_record_for_selector(
        prune_plan=prune_plan,
        recovery_bundle=bundle,
        prune_selector=prune_selector,
    )
    lora = _lora_specs(bundle, record) if method == "recovery_lora" else {}
    cache_ids = [str(x) for x in record.get("cache_unit_ids", [])]
    extra = dict(prune_plan.extra)
    extra.update(
        {
            "recovery_plan_schema_version": RECOVERY_PLAN_BUNDLE_SCHEMA_VERSION,
            "recovery_plan_name": normalize_recovery_plan_name(recovery_plan),
            "training_method": execution_method(method),
            "recovery_method": method,
            "lora_prefix": int(bundle.get("lora_prefix", 0)),
            "lora_rank": int(bundle.get("lora_rank", 0)),
            "lora_indices": [int(x) for x in record.get("lora_indices", [])],
            "train_unit_indices": [int(x) for x in record.get("train_unit_indices", [])],
            "cache_unit_ids": cache_ids,
            "recovery_boundaries": [dict(x) for x in record.get("boundaries", [])],
            "recovery_source_unit_index": int(record["source_unit_index"]),
            "recovery_target_unit_index": int(record["target_unit_index"]),
            "recovery_source_unit_id": str(record["source_unit_id"]),
            "recovery_target_unit_id": str(record["target_unit_id"]),
            "recovery_auxiliary_ids": [str(x) for x in record.get("auxiliary_cache_ids", [])],
        }
    )
    out = UnitPrunePlan(
        backbone=prune_plan.backbone,
        dataset=prune_plan.dataset,
        num_units=prune_plan.num_units,
        drop=list(prune_plan.drop),
        drop_length=len(prune_plan.drop),
        lora=lora,
        source_cache=prune_plan.source_cache,
        extra=extra,
    )
    out.validate()
    return out


def recovery_plan_supports_selector(
    *, root: Path, name: str, prune_plan_root: Path, prune_selector: str
) -> bool:
    try:
        compose_recovery_unit_prune_plan(
            prune_plan_root=prune_plan_root,
            recovery_plan_root=root,
            prune_selector=prune_selector,
            recovery_plan=name,
        )
    except FileNotFoundError:
        return False
    return True

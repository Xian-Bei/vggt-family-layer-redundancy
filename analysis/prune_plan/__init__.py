from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.schema.v3 import PruneAction

_FULL_PLAN_NAME_RE = re.compile(r"^(plan_t\d+)_l\d+$")
_PLAN_GROUP_RE = re.compile(r"^plan_t\d+$")
_BUDGET_FULL_PLAN_NAME_RE = re.compile(r"^(plan_budget\d+)_l\d+$")
_BUDGET_PLAN_RE = re.compile(r"^plan_budget\d+$")
_BUDGET_FULL_SELECTOR_RE = re.compile(r"^plan_budget(?P<budget>\d+)_l(?P<lora>\d+)$")
_BUDGET_GROUP_SELECTOR_RE = re.compile(r"^plan_budget(?P<budget>\d+)$")
_SPAN_SELECTOR_RE = re.compile(r"^prune_span(?P<span>\d+)$")
_INTERVAL_SELECTOR_RE = re.compile(r"^i(?P<i>\d+)_j(?P<j>\d+)$")

BUDGET_PLAN_BUNDLE_FILENAME = "plan_budget_all.json"
BUDGET_PLAN_BUNDLE_SCHEMA_VERSION = "budget_plan_bundle_v1"
PRUNE_PLAN_BUNDLE_FILENAME = "prune_plan_all.json"
PRUNE_PLAN_BUNDLE_SCHEMA_VERSION = "closed_interval_prune_plan_bundle_v3"
PRUNE_PLAN_BUNDLE_SCHEMA_V1 = "span_prune_plan_bundle_v1"
PRUNE_PLAN_BUNDLE_SCHEMA_V2 = "span_prune_plan_bundle_v2"
PRUNE_PLAN_BUNDLE_SCHEMA_VERSIONS = {
    PRUNE_PLAN_BUNDLE_SCHEMA_V1,
    PRUNE_PLAN_BUNDLE_SCHEMA_V2,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_prune_plan_root() -> Path:
    return _repo_root() / "analysis" / "prune_plan"


def resolve_prune_plan_path(root: Path, backbone: str) -> Path:
    key = str(backbone or "").strip().lower()
    if not key:
        raise ValueError("backbone must be non-empty")
    return Path(root) / f"{key}.json"


def resolve_named_prune_plan_path(root: Path, plan_name: str) -> Path:
    key = str(plan_name or "").strip()
    if not key:
        raise ValueError("plan_name must be non-empty")
    if key.endswith(".json"):
        return Path(root) / key
    return Path(root) / f"{key}.json"


def budget_plan_bundle_path(root: Path) -> Path:
    return Path(root) / BUDGET_PLAN_BUNDLE_FILENAME


def prune_plan_bundle_path(root: Path) -> Path:
    return Path(root) / PRUNE_PLAN_BUNDLE_FILENAME


def load_prune_plan_bundle(root: Path) -> Dict[str, Any]:
    path = prune_plan_bundle_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prune plan bundle: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid prune plan bundle: expected object at {path}")
    schema = str(raw.get("schema_version", ""))
    if schema not in PRUNE_PLAN_BUNDLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"Invalid prune plan bundle schema at {path}: {schema!r} "
            f"(expected one of {sorted(PRUNE_PLAN_BUNDLE_SCHEMA_VERSIONS)!r})"
        )
    return raw


def interval_selector(i: int, j: int) -> str:
    start = int(i)
    end = int(j)
    if start < 0 or end < start:
        raise ValueError(f"invalid interval selector bounds: i={start}, j={end}")
    return f"i{start}_j{end}"


def interval_bounds_from_selector(selector: str) -> Tuple[int, int]:
    m = _INTERVAL_SELECTOR_RE.match(str(selector or "").strip())
    if not m:
        raise ValueError(f"invalid interval selector: {selector!r}")
    return int(m.group("i")), int(m.group("j"))


def is_interval_selector(selector: str) -> bool:
    return bool(_INTERVAL_SELECTOR_RE.match(str(selector or "").strip()))


def interval_bounds_from_record(record: Dict[str, Any]) -> Tuple[int, int]:
    """Return closed drop bounds for a bundle interval record.

    v3 records store ``start/end`` directly.  Older span bundles used ``i/j``
    as an open-left corner, but most converted records also carry an explicit
    ``drop`` list, which is the least ambiguous source of truth.
    """
    if "start" in record and "end" in record:
        return int(record.get("start")), int(record.get("end"))
    drop = [int(x) for x in list(record.get("drop", []))]
    if drop:
        return min(drop), max(drop)
    return int(record.get("i", 0)) + 1, int(record.get("j", 0))


def interval_selector_from_record(record: Dict[str, Any]) -> str:
    start, end = interval_bounds_from_record(record)
    return interval_selector(start, end)


def _record_drop_count(record: Dict[str, Any]) -> int:
    if "drop_count" in record:
        return int(record.get("drop_count", 0))
    if "span" in record:
        return int(record.get("span", 0))
    return len(list(record.get("drop", [])))


def bundle_interval_records(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return interval-keyed prune records, accepting v1 span bundles as input."""
    raw_intervals = bundle.get("intervals", {})
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_intervals, dict):
        for key, record in raw_intervals.items():
            if not isinstance(record, dict):
                continue
            interval_key = str(record.get("selector") or key).strip()
            if not is_interval_selector(interval_key):
                try:
                    interval_key = interval_selector_from_record(record)
                except Exception:
                    continue
            out[interval_key] = dict(record, selector=interval_key)
    if out:
        return out
    spans = bundle.get("spans", {})
    if isinstance(spans, dict):
        for record in spans.values():
            if not isinstance(record, dict):
                continue
            try:
                key = interval_selector_from_record(record)
            except Exception:
                continue
            out[key] = dict(record, selector=key)
    return out


def bundle_strategy_records(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    strategies = bundle.get("strategies", {})
    if isinstance(strategies, dict) and strategies:
        return {str(k): dict(v) for k, v in strategies.items() if isinstance(v, dict)}
    spans = bundle.get("spans", {})
    points: Dict[str, str] = {}
    if isinstance(spans, dict):
        for span_key, record in spans.items():
            if not isinstance(record, dict):
                continue
            try:
                points[str(int(span_key))] = interval_selector_from_record(record)
            except Exception:
                continue
    if not points:
        return {}
    return {
        "ours": {
            "display_name": "Ours",
            "points": points,
            "source": "legacy_spans",
        }
    }


def strategy_interval_selectors(
    root: Path,
    *,
    strategies: Optional[Sequence[str]] = None,
    min_drop: Optional[int] = None,
    max_drop: Optional[int] = None,
    drop_step: Optional[int] = None,
) -> List[str]:
    bundle = load_prune_plan_bundle(root)
    intervals = bundle_interval_records(bundle)
    strategy_records = bundle_strategy_records(bundle)
    requested = [str(s).strip() for s in list(strategies or []) if str(s).strip()]
    if not requested:
        requested = list(strategy_records.keys())
    selected: List[str] = []
    seen: set[str] = set()
    for strategy in requested:
        record = strategy_records.get(strategy)
        if not isinstance(record, dict):
            raise FileNotFoundError(f"Strategy {strategy!r} is not present in prune plan bundle")
        points = record.get("points", {})
        if not isinstance(points, dict):
            continue
        for drop_key, selector in sorted(
            points.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            if not str(drop_key).isdigit():
                continue
            drop_count = int(drop_key)
            if min_drop is not None and drop_count < int(min_drop):
                continue
            if max_drop is not None and drop_count > int(max_drop):
                continue
            if drop_step is not None and int(drop_step) > 0:
                base = int(min_drop or 0)
                if (drop_count - base) % int(drop_step) != 0:
                    continue
            interval_key = str(selector)
            if interval_key not in intervals:
                raise FileNotFoundError(
                    f"Strategy {strategy!r} references missing interval {interval_key!r}"
                )
            if interval_key not in seen:
                seen.add(interval_key)
                selected.append(interval_key)
    return selected


def load_plan_bundle(root: Path) -> Dict[str, Any]:
    path = budget_plan_bundle_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing budget plan bundle: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid budget plan bundle: expected object at {path}")
    schema = str(raw.get("schema_version", ""))
    if schema != BUDGET_PLAN_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid budget plan bundle schema at {path}: {schema!r} "
            f"(expected {BUDGET_PLAN_BUNDLE_SCHEMA_VERSION!r})"
        )
    return raw


def _budget_group_number(group: str) -> int:
    m = _BUDGET_GROUP_SELECTOR_RE.match(str(group or "").strip())
    if not m:
        raise ValueError(f"invalid budget plan group: {group!r}")
    return int(m.group("budget"))


def _budget_full_selector(name: str) -> Tuple[int, int]:
    m = _BUDGET_FULL_SELECTOR_RE.match(str(name or "").strip())
    if not m:
        raise ValueError(f"invalid budget plan selector: {name!r}")
    return int(m.group("budget")), int(m.group("lora"))


def _bundle_budget_record(bundle: Dict[str, Any], budget: int) -> Dict[str, Any]:
    budgets = bundle.get("budgets", {})
    if not isinstance(budgets, dict):
        raise ValueError("Invalid budget plan bundle: budgets must be an object")
    raw = budgets.get(str(int(budget)))
    if not isinstance(raw, dict):
        raise FileNotFoundError(f"Budget selector plan_budget{int(budget)} is not present in bundle")
    if not bool(raw.get("available", False)):
        reason = str(raw.get("skip_reason", "") or "budget is unavailable")
        raise FileNotFoundError(f"Budget selector plan_budget{int(budget)} is unavailable: {reason}")
    return raw


def _bundle_lora_prefixes(bundle: Dict[str, Any]) -> List[int]:
    prefixes = [int(x) for x in list(bundle.get("lora_prefixes", []))]
    if not prefixes:
        raise FileNotFoundError("Budget plan bundle contains no LoRA prefixes")
    return sorted(set(prefixes))


def is_full_plan_name(name: str) -> bool:
    key = str(name or "").strip()
    return bool(
        _FULL_PLAN_NAME_RE.match(key)
        or _BUDGET_FULL_PLAN_NAME_RE.match(key)
        or _SPAN_SELECTOR_RE.match(key)
        or _INTERVAL_SELECTOR_RE.match(key)
    )


def is_plan_group_name(name: str) -> bool:
    key = str(name or "").strip()
    return bool(
        _PLAN_GROUP_RE.match(key)
        or _BUDGET_PLAN_RE.match(key)
        or _SPAN_SELECTOR_RE.match(key)
        or _INTERVAL_SELECTOR_RE.match(key)
    )


def plan_group_from_name(name: str) -> str:
    """Map a plan filename stem to its evaluation group."""
    key = str(name or "").strip()
    if not key:
        raise ValueError("plan name must be non-empty")
    m = _FULL_PLAN_NAME_RE.match(key)
    if m:
        return m.group(1)
    m = _BUDGET_FULL_PLAN_NAME_RE.match(key)
    if m:
        return m.group(1)
    if _SPAN_SELECTOR_RE.match(key):
        return key
    if _INTERVAL_SELECTOR_RE.match(key):
        return key
    if _PLAN_GROUP_RE.match(key) or _BUDGET_PLAN_RE.match(key):
        return key
    return key


def _lora_index_from_plan_stem(stem: str) -> int:
    m = _FULL_PLAN_NAME_RE.match(str(stem)) or _BUDGET_FULL_PLAN_NAME_RE.match(str(stem))
    if not m:
        return 0
    return int(str(stem).rsplit("_l", 1)[-1])


def resolve_prune_plan_path_for_group(root: Path, plan_group: str) -> Path:
    """Resolve a budget group to the bundle path after validating availability.

    New budget selectors are virtual; callers that need an actual plan should use
    ``load_group_canonical_prune_plan`` or ``canonical_plan_name_for_group``.
    """
    group = plan_group_from_name(plan_group)
    if not is_plan_group_name(group):
        raise ValueError(f"invalid plan group: {plan_group!r}")
    root = Path(root)
    if _SPAN_SELECTOR_RE.match(group):
        bundle = load_prune_plan_bundle(root)
        _span_record_from_bundle(bundle, group)
        return prune_plan_bundle_path(root)
    if _INTERVAL_SELECTOR_RE.match(group):
        bundle = load_prune_plan_bundle(root)
        if group not in bundle_interval_records(bundle):
            raise FileNotFoundError(f"No prune interval {group} under {root}")
        return prune_plan_bundle_path(root)
    if _BUDGET_PLAN_RE.match(group):
        bundle = load_plan_bundle(root)
        _bundle_budget_record(bundle, _budget_group_number(group))
        return budget_plan_bundle_path(root)
    matches = sorted(
        root.glob(f"{group}_l*.json"),
        key=lambda p: (_lora_index_from_plan_stem(p.stem), p.name),
    )
    if not matches:
        raise FileNotFoundError(f"No plan files for group {group!r} under {root}")
    return matches[0]


def list_plan_groups(root: Path) -> List[str]:
    """List available physical interval selectors from ``prune_plan_all.json``."""
    root = Path(root)
    if prune_plan_bundle_path(root).is_file():
        bundle = load_prune_plan_bundle(root)
        intervals = bundle_interval_records(bundle)
        return [
            key
            for key, record in sorted(
                intervals.items(),
                key=lambda item: (
                    _record_drop_count(item[1]),
                    int(item[1].get("i", 0)),
                    int(item[1].get("j", 0)),
                    item[0],
                ),
            )
        ]
    if not budget_plan_bundle_path(root).is_file():
        return []
    bundle = load_plan_bundle(root)
    budgets = bundle.get("budgets", {})
    if not isinstance(budgets, dict):
        return []
    out: List[str] = []
    for key, record in sorted(
        budgets.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
    ):
        if isinstance(record, dict) and bool(record.get("available", False)):
            out.append(f"plan_budget{int(key)}")
    return out


def list_full_plan_names(root: Path) -> List[str]:
    """List all available virtual budget×LoRA plan selectors from the bundle."""
    root = Path(root)
    if not budget_plan_bundle_path(root).is_file():
        return []
    bundle = load_plan_bundle(root)
    prefixes = _bundle_lora_prefixes(bundle)
    out: List[str] = []
    for group in list_plan_groups(root):
        for lora_prefix in prefixes:
            out.append(f"{group}_l{int(lora_prefix)}")
    return out


def canonical_plan_name_for_group(root: Path, plan_group: str) -> str:
    """Return the canonical virtual selector for a prune/span group."""
    group = plan_group_from_name(plan_group)
    if _SPAN_SELECTOR_RE.match(group):
        return interval_selector_from_record(
            _span_record_from_bundle(load_prune_plan_bundle(root), group)
        )
    if _INTERVAL_SELECTOR_RE.match(group):
        resolve_prune_plan_path_for_group(root, group)
        return group
    if not _BUDGET_PLAN_RE.match(group):
        path = resolve_prune_plan_path_for_group(root, group)
        return path.stem
    bundle = load_plan_bundle(root)
    _bundle_budget_record(bundle, _budget_group_number(group))
    return f"{group}_l{_bundle_lora_prefixes(bundle)[0]}"


def unit_index_to_kind_layer(index: int) -> Tuple[str, int]:
    i = int(index)
    if i < 0:
        raise ValueError(f"unit index must be >= 0, got {i}")
    return ("frame", i // 2) if (i % 2 == 0) else ("global", i // 2)


def unit_index_to_unit_id(index: int) -> str:
    kind, layer = unit_index_to_kind_layer(index)
    return f"{kind}_{layer}"


@dataclass
class UnitLoraSpec:
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    target_submodules: Sequence[str] = field(default_factory=lambda: ("qkv", "proj"))
    alignment_teacher_unit_index: Optional[int] = None


@dataclass
class UnitPrunePlan:
    backbone: str
    dataset: str
    num_units: int
    drop: List[int] = field(default_factory=list)
    drop_length: int = 0
    lora: Dict[str, UnitLoraSpec] = field(default_factory=dict)
    source_cache: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = int(self.num_units)
        if n <= 0:
            raise ValueError(f"num_units must be > 0, got {n}")
        self.drop = sorted({int(x) for x in self.drop})
        actual_drop_len = len(self.drop)
        if int(self.drop_length) != actual_drop_len:
            raise ValueError(
                f"drop_length={int(self.drop_length)} does not match len(drop)={actual_drop_len}"
            )
        bad_drop = [int(x) for x in self.drop if int(x) < 0 or int(x) >= n]
        if bad_drop:
            raise ValueError(f"drop indices out of range [0,{n}): {bad_drop}")
        for key, spec in self.lora.items():
            idx = int(key)
            if idx < 0 or idx >= n:
                raise ValueError(f"lora key out of range [0,{n}): {idx}")
            if int(spec.rank) <= 0:
                raise ValueError(f"lora[{idx}] rank must be > 0")
            if spec.alignment_teacher_unit_index is not None:
                t_idx = int(spec.alignment_teacher_unit_index)
                if t_idx < 0 or t_idx >= n:
                    raise ValueError(
                        f"lora[{idx}] alignment_teacher_unit_index out of range [0,{n}): {t_idx}"
                    )

    def save(self, path: Path) -> None:
        drop_sorted = sorted({int(x) for x in self.drop})
        self.drop = drop_sorted
        self.drop_length = len(drop_sorted)
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = asdict(self)
        payload["drop"] = drop_sorted
        payload["drop_length"] = int(self.drop_length)
        payload["lora"] = {str(int(k)): asdict(v) for k, v in self.lora.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UnitPrunePlan":
        if not isinstance(raw, dict):
            raise ValueError("Invalid prune plan: expected object")
        if "schema_version" in raw:
            raise ValueError("Legacy prune plans with schema_version are not supported")
        if "intervals" in raw or "actions" in raw:
            raise ValueError("Legacy prune plan formats are not supported")
        required = ("backbone", "dataset", "num_units", "drop", "lora")
        miss = [k for k in required if k not in raw]
        if miss:
            raise ValueError(f"Missing required keys in prune plan: {miss}")
        lora_raw = raw.get("lora", {})
        if not isinstance(lora_raw, dict):
            raise ValueError("lora must be an object mapping index -> config")
        lora: Dict[str, UnitLoraSpec] = {}
        for k, v in lora_raw.items():
            if not isinstance(v, dict):
                raise ValueError(f"lora[{k}] must be object")
            if "alignment" in v:
                raise ValueError(
                    f"lora[{k}] uses legacy 'alignment' object; use alignment_teacher_unit_index"
                )
            ati = v.get("alignment_teacher_unit_index")
            lora[str(int(k))] = UnitLoraSpec(
                rank=int(v.get("rank", 32)),
                alpha=float(v.get("alpha", 64.0)),
                dropout=float(v.get("dropout", 0.0)),
                target_submodules=tuple(v.get("target_submodules", ("qkv", "proj"))),
                alignment_teacher_unit_index=int(ati) if ati is not None else None,
            )
        drop_list = sorted({int(x) for x in list(raw.get("drop", []))})
        if "drop_length" in raw:
            drop_length = int(raw.get("drop_length"))
        else:
            drop_length = len(drop_list)
        plan = cls(
            backbone=str(raw.get("backbone", "")),
            dataset=str(raw.get("dataset", "")),
            num_units=int(raw.get("num_units", 0)),
            drop=drop_list,
            drop_length=drop_length,
            lora=lora,
            source_cache=str(raw.get("source_cache", "")),
            extra=dict(raw.get("extra", {})),
        )
        plan.validate()
        return plan

    @classmethod
    def load(cls, path: Path) -> "UnitPrunePlan":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def _unit_lora_spec_from_raw(raw: Dict[str, Any]) -> UnitLoraSpec:
    ati = raw.get("alignment_teacher_unit_index")
    return UnitLoraSpec(
        rank=int(raw.get("rank", 32)),
        alpha=float(raw.get("alpha", 64.0)),
        dropout=float(raw.get("dropout", 0.0)),
        target_submodules=tuple(raw.get("target_submodules", ("qkv", "proj"))),
        alignment_teacher_unit_index=int(ati) if ati is not None else None,
    )


def _load_named_budget_prune_plan(root: Path, plan_name: str) -> UnitPrunePlan:
    budget, lora_prefix = _budget_full_selector(plan_name)
    bundle = load_plan_bundle(root)
    budget_record = _bundle_budget_record(bundle, budget)
    if int(lora_prefix) not in _bundle_lora_prefixes(bundle):
        raise FileNotFoundError(
            f"Selector {plan_name!r} is unavailable: bundle has no lora prefix {lora_prefix}"
        )
    drop_count = int(budget_record.get("drop_count", 0))

    base_extra = dict(bundle.get("extra", {})) if isinstance(bundle.get("extra", {}), dict) else {}
    source_cache = str(bundle.get("source_cache", ""))
    if drop_count == 0:
        plan = UnitPrunePlan(
            backbone=str(bundle.get("backbone", "")),
            dataset=str(bundle.get("dataset", "")),
            num_units=int(bundle.get("num_units", 0)),
            drop=[],
            drop_length=0,
            lora={},
            source_cache=source_cache,
            extra={
                **base_extra,
                "schema_version": BUDGET_PLAN_BUNDLE_SCHEMA_VERSION,
                "selector": str(plan_name),
                "plan_group": f"plan_budget{budget}",
                "pruning_budget_percent": int(budget),
                "requested_drop_count": 0,
                "lora_prefix": int(lora_prefix),
                "selected_corner_similarity": 1.0,
                "cache_unit_ids": [],
                "cache_unit_indices": [],
            },
        )
        plan.validate()
        return plan

    spans = bundle.get("spans", {})
    if not isinstance(spans, dict):
        raise ValueError("Invalid budget plan bundle: spans must be an object")
    span_record = spans.get(str(drop_count))
    if not isinstance(span_record, dict):
        raise FileNotFoundError(
            f"Budget selector plan_budget{budget} requires drop_count={drop_count}, "
            "but that span is missing from bundle"
        )
    lora_by_prefix = span_record.get("lora", {})
    if not isinstance(lora_by_prefix, dict):
        raise ValueError(f"Invalid span record for drop_count={drop_count}: lora must be an object")
    lora_record = lora_by_prefix.get(str(lora_prefix))
    if not isinstance(lora_record, dict):
        raise FileNotFoundError(
            f"Selector {plan_name!r} is unavailable: bundle has no lora prefix {lora_prefix}"
        )

    lora_raw = lora_record.get("lora", {})
    if not isinstance(lora_raw, dict):
        raise ValueError(f"Invalid lora record for selector {plan_name!r}: lora must be an object")
    lora = {
        str(int(k)): _unit_lora_spec_from_raw(v)
        for k, v in lora_raw.items()
        if isinstance(v, dict)
    }
    drop = [int(x) for x in list(span_record.get("drop", []))]
    cache_unit_ids = [str(x) for x in list(lora_record.get("cache_unit_ids", []))]
    cache_unit_indices = [int(x) for x in list(lora_record.get("cache_unit_indices", []))]
    triangles = [
        {
            "i": int(span_record.get("i", 0)),
            "j": int(span_record.get("j", 0)),
            "span": int(span_record.get("span", drop_count)),
            "unit_i": str(span_record.get("unit_i", "")),
            "unit_j": str(span_record.get("unit_j", "")),
            "corner_similarity": float(span_record.get("corner_similarity", 1.0)),
        }
    ]
    plan = UnitPrunePlan(
        backbone=str(bundle.get("backbone", "")),
        dataset=str(bundle.get("dataset", "")),
        num_units=int(bundle.get("num_units", 0)),
        drop=drop,
        drop_length=len(drop),
        lora=lora,
        source_cache=source_cache,
        extra={
            **base_extra,
            "schema_version": BUDGET_PLAN_BUNDLE_SCHEMA_VERSION,
            "selector": str(plan_name),
            "plan_group": f"plan_budget{budget}",
            "pruning_budget_percent": int(budget),
            "requested_drop_count": int(drop_count),
            "lora_prefix": int(lora_prefix),
            "selected_corner_similarity": float(span_record.get("corner_similarity", 1.0)),
            "cache_unit_ids": cache_unit_ids,
            "cache_unit_indices": cache_unit_indices,
            "triangles": triangles,
        },
    )
    plan.validate()
    return plan


def _span_record_from_bundle(bundle: Dict[str, Any], selector: str) -> Dict[str, Any]:
    m = _SPAN_SELECTOR_RE.match(str(selector or "").strip())
    if not m:
        raise ValueError(f"invalid prune span selector: {selector!r}")
    span = int(m.group("span"))
    strategies = bundle_strategy_records(bundle)
    ours = strategies.get("ours", {})
    points = ours.get("points", {}) if isinstance(ours, dict) else {}
    if isinstance(points, dict):
        interval_key = points.get(str(span))
        if isinstance(interval_key, str):
            intervals = bundle_interval_records(bundle)
            record = intervals.get(interval_key)
            if isinstance(record, dict):
                return record
    spans = bundle.get("spans", {})
    if isinstance(spans, dict):
        record = spans.get(str(span))
        if isinstance(record, dict):
            return record
    raise FileNotFoundError(f"Prune span selector {selector!r} is not present in bundle")


def _unit_prune_plan_from_interval_record(
    *,
    bundle: Dict[str, Any],
    selector: str,
    record: Dict[str, Any],
) -> UnitPrunePlan:
    base_extra = dict(bundle.get("extra", {})) if isinstance(bundle.get("extra", {}), dict) else {}
    drop = [int(x) for x in list(record.get("drop", []))]
    if not drop:
        start, end = interval_bounds_from_record(record)
        drop = list(range(start, end + 1))
    drop_count = len(drop)
    interval_key = interval_selector_from_record(record)
    start, end = interval_bounds_from_record(record)
    triangles = [
        {
            "i": int(record.get("i", 0)),
            "j": int(record.get("j", 0)),
            "start": int(start),
            "end": int(end),
            "span": int(record.get("span", record.get("drop_count", drop_count))),
            "unit_i": str(record.get("unit_i", "")),
            "unit_j": str(record.get("unit_j", "")),
            "corner_similarity": float(record.get("corner_similarity", 1.0)),
        }
    ] if drop_count > 0 else []
    plan = UnitPrunePlan(
        backbone=str(bundle.get("backbone", "")),
        dataset=str(bundle.get("dataset", "")),
        num_units=int(bundle.get("num_units", 0)),
        drop=drop,
        drop_length=len(drop),
        lora={},
        source_cache=str(bundle.get("source_cache", "")),
        extra={
            **base_extra,
            "schema_version": str(bundle.get("schema_version", "")),
            "selector": str(selector),
            "plan_group": interval_key,
            "interval_selector": interval_key,
            "closed_interval": True,
            "start": int(start),
            "end": int(end),
            "requested_drop_count": int(record.get("drop_count", drop_count)),
            "selected_corner_similarity": float(record.get("corner_similarity", 1.0)),
            "triangles": triangles,
        },
    )
    plan.validate()
    return plan


def _load_named_span_prune_plan(root: Path, plan_name: str) -> UnitPrunePlan:
    bundle = load_prune_plan_bundle(root)
    return _unit_prune_plan_from_interval_record(
        bundle=bundle,
        selector=plan_name,
        record=_span_record_from_bundle(bundle, plan_name),
    )


def _load_named_interval_prune_plan(root: Path, plan_name: str) -> UnitPrunePlan:
    key = str(plan_name or "").strip()
    if not _INTERVAL_SELECTOR_RE.match(key):
        raise ValueError(f"invalid prune interval selector: {plan_name!r}")
    bundle = load_prune_plan_bundle(root)
    intervals = bundle_interval_records(bundle)
    record = intervals.get(key)
    if not isinstance(record, dict):
        raise FileNotFoundError(f"Prune interval selector {plan_name!r} is not present in bundle")
    return _unit_prune_plan_from_interval_record(bundle=bundle, selector=key, record=record)


def load_named_prune_plan(root: Path, plan_name: str) -> UnitPrunePlan:
    """Load a virtual budget selector from ``plan_budget_all.json``.

    Budget selectors may be either full names (``plan_budget20_l1``) or groups
    (``plan_budget20``), in which case the smallest LoRA prefix is used.
    """
    key = str(plan_name or "").strip()
    if not key:
        raise ValueError("plan_name must be non-empty")
    if _INTERVAL_SELECTOR_RE.match(key):
        return _load_named_interval_prune_plan(root, key)
    if _SPAN_SELECTOR_RE.match(key):
        return _load_named_span_prune_plan(root, key)
    if _BUDGET_PLAN_RE.match(key):
        key = canonical_plan_name_for_group(root, key)
    if _BUDGET_FULL_SELECTOR_RE.match(key):
        return _load_named_budget_prune_plan(root, key)
    path = resolve_named_prune_plan_path(root, key)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prune plan: {path}")
    return UnitPrunePlan.load(path)


def load_group_canonical_prune_plan(root: Path, plan_group: str) -> UnitPrunePlan:
    return load_named_prune_plan(root, canonical_plan_name_for_group(root, plan_group))


def prune_actions_from_plan(plan: UnitPrunePlan) -> Tuple[List[PruneAction], List[PruneAction]]:
    plan.validate()
    drop_actions: List[PruneAction] = []
    lora_actions: List[PruneAction] = []
    for idx in sorted({int(x) for x in plan.drop}):
        drop_actions.append(PruneAction(action="drop_unit", unit_id=unit_index_to_unit_id(idx)))
    for k, spec in sorted(plan.lora.items(), key=lambda x: int(x[0])):
        idx = int(k)
        lora_actions.append(
            PruneAction(
                action="lora_unit",
                unit_id=unit_index_to_unit_id(idx),
                rank=int(spec.rank),
                alpha=float(spec.alpha),
                dropout=float(spec.dropout),
                target_submodules=tuple(spec.target_submodules),
            )
        )
    return drop_actions, lora_actions


def _merge_contiguous(idxs: Sequence[int]) -> List[Tuple[int, int]]:
    vals = sorted({int(x) for x in idxs})
    if not vals:
        return []
    out: List[Tuple[int, int]] = []
    s = p = vals[0]
    for k in vals[1:]:
        if k == p + 1:
            p = k
            continue
        out.append((s, p))
        s = p = k
    out.append((s, p))
    return out


def drop_intervals_from_plan(
    plan: UnitPrunePlan,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[int], List[int]]:
    plan.validate()
    frame_layers: set[int] = set()
    global_layers: set[int] = set()
    for idx in plan.drop:
        kind, layer = unit_index_to_kind_layer(int(idx))
        if kind == "frame":
            frame_layers.add(layer)
        else:
            global_layers.add(layer)
    frame_vals = sorted(frame_layers)
    global_vals = sorted(global_layers)
    return _merge_contiguous(frame_vals), _merge_contiguous(global_vals), frame_vals, global_vals


def triangles_from_plan_extra(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read triangle metadata from plan extra (new or legacy schema)."""
    if not isinstance(extra, dict):
        return []
    if isinstance(extra.get("triangles"), list):
        return [dict(x) for x in extra["triangles"] if isinstance(x, dict)]
    legacy = extra.get("triangle_prune")
    if isinstance(legacy, dict) and isinstance(legacy.get("triangles"), list):
        return [dict(x) for x in legacy["triangles"] if isinstance(x, dict)]
    return []


def lora_layers_from_plan(plan: UnitPrunePlan) -> List[int]:
    vals = []
    for k in plan.lora.keys():
        _kind, layer = unit_index_to_kind_layer(int(k))
        vals.append(layer)
    return sorted({int(x) for x in vals})


try:
    from analysis.prune_plan.training_cache_units import (  # noqa: E402
        layers_to_unit_ids,
        replay_seed_spec,
        replay_start_layer,
        required_cache_layers_for_training,
        required_cache_unit_ids,
    )
except ImportError as _training_cache_import_error:  # pragma: no cover - depends on optional torch env
    def _missing_training_cache_units(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            "analysis.prune_plan.training_cache_units requires the torch environment"
        ) from _training_cache_import_error

    layers_to_unit_ids = _missing_training_cache_units
    replay_seed_spec = _missing_training_cache_units
    replay_start_layer = _missing_training_cache_units
    required_cache_layers_for_training = _missing_training_cache_units
    required_cache_unit_ids = _missing_training_cache_units

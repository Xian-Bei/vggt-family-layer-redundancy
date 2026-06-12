from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.schema.v3 import PruneAction


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


from analysis.prune_plan.training_cache_units import (  # noqa: E402
    layers_to_unit_ids,
    replay_seed_spec,
    replay_start_layer,
    required_cache_layers_for_training,
    required_cache_unit_ids,
)

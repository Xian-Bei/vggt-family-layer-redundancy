from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.prune_plan import interval_bounds_from_selector, is_interval_selector
from analysis.schema.v3 import EvalRecordV3, SCHEMA_VERSION

INDEX_FILENAME = "skip_eval_interval_index_v3.json"
RECOVERY_INDEX_FILENAME = "recovery_eval_index_v3.json"


def normalize_interval_metrics(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        val = float(value)
        if not math.isfinite(val):
            continue
        out[str(key)] = val
    if "cd" in out and "chamfer_distance" not in out:
        out["chamfer_distance"] = out["cd"]
    if "chamfer_distance" in out and "cd" not in out:
        out["cd"] = out["chamfer_distance"]
    return out


def _selector_bounds(selector: str, drop: Sequence[int]) -> tuple[Optional[int], Optional[int]]:
    if is_interval_selector(selector):
        return interval_bounds_from_selector(selector)
    if drop:
        vals = sorted(int(x) for x in drop)
        return vals[0], vals[-1]
    return None, None


def _index_for_result_root(root: Path) -> dict[str, Any]:
    intervals: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        selector = str(raw.get("selector") or summary_path.parent.name).strip()
        if not selector:
            continue
        intervals[selector] = {
            "selector": selector,
            "start": raw.get("start"),
            "end": raw.get("end"),
            "drop_count": raw.get("drop_count"),
            "num_scenes": raw.get("num_scenes"),
            "summary_path": str(summary_path),
            "scenes_path": str(summary_path.parent / "scenes.jsonl"),
            "eval_record_path": str(summary_path.parent / "eval_record_v3.json"),
        }
    return {
        "schema_version": "skip_eval_interval_index_v3",
        "kind": "eval_interval_index",
        "skip_eval_root": str(root),
        "num_intervals": len(intervals),
        "intervals": intervals,
    }


def _recovery_index_for_selector_root(root: Path, selector: str) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(root.glob("recovery_*/summary.json")):
        raw = _load_summary(summary_path)
        if not raw:
            continue
        recovery_name = str(
            raw.get("recovery_plan_name") or summary_path.parent.name
        ).strip()
        if not recovery_name:
            continue
        methods[recovery_name] = {
            "recovery_plan_name": recovery_name,
            "selector": str(raw.get("selector") or selector),
            "drop_count": raw.get("drop_count"),
            "num_scenes": raw.get("num_scenes"),
            "summary_path": str(summary_path),
            "eval_record_path": str(summary_path.parent / "eval_record_v3.json"),
        }
    return {
        "schema_version": "recovery_eval_index_v1",
        "kind": "recovery_eval_index",
        "selector": str(selector),
        "result_root": str(root),
        "methods": methods,
    }


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def materialize_eval_interval_result(
    out_dir: Path,
    *,
    selector: str,
    drop: Sequence[int],
    source_eval_mode: str,
    recovery_plan_name: str = "",
) -> Path:
    out_dir = Path(out_dir)
    record_path = out_dir / "eval_record_v3.json"
    record = EvalRecordV3.load(record_path)
    selector = str(selector or out_dir.name).strip()
    drop_list = sorted(int(x) for x in drop)
    start, end = _selector_bounds(selector, drop_list)
    metrics_mean = normalize_interval_metrics(dict(record.average))
    scene_rows = []
    for row in record.scenes.values():
        if not isinstance(row, dict):
            continue
        sid = str(row.get("scene_id", "")).strip()
        scene_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "eval_interval_scene",
                "selector": selector,
                "scene_id": sid,
                "metrics": normalize_interval_metrics(dict(row)),
            }
        )
    summary: dict[str, Any] = {
        "schema_version": "eval_interval_v3",
        "kind": "eval_interval_summary",
        "selector": selector,
        "start": start,
        "end": end,
        "drop": drop_list,
        "drop_count": len(drop_list),
        "num_scenes": len(scene_rows),
        "metrics_mean": metrics_mean,
        "eval_record_path": str(record_path),
        "source_eval_mode": str(source_eval_mode),
    }
    if recovery_plan_name:
        summary["recovery_plan_name"] = str(recovery_plan_name)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    scene_text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in scene_rows)
    (out_dir / "scenes.jsonl").write_text(scene_text, encoding="utf-8")
    if recovery_plan_name:
        index = _recovery_index_for_selector_root(out_dir.parent, selector)
        index_path = out_dir.parent / RECOVERY_INDEX_FILENAME
    else:
        index = _index_for_result_root(out_dir.parent)
        index_path = out_dir.parent / INDEX_FILENAME
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out_dir / "summary.json"

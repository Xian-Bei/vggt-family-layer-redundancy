from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from analysis.prune_plan import interval_selector


REMAINING_INTERNAL_DIR = "_remaining_skip"
REMAINING_INDEX_FILENAME = "remaining_skip_eval_index_v3.json"
REMAINING_INDEX_SCHEMA = "remaining_skip_eval_index_v3"
REMAINING_LAYOUT_SCHEMA = "original_index_double_interval_v1"


def logical_selector(span: Tuple[int, int]) -> str:
    return interval_selector(int(span[0]), int(span[1]))


def composite_selector(base_selector: str, second_selector: str) -> str:
    return f"{str(base_selector)}_{str(second_selector)}"


def build_remaining_span_records(
    *,
    full_unit_ids: Sequence[str],
    remaining_unit_ids: Sequence[str],
    dropped_indices: Sequence[int],
    logical_spans: Sequence[Tuple[int, int]],
    physical_to_logical: Mapping[Tuple[int, int], Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    full_ids = [str(unit_id) for unit_id in full_unit_ids]
    remaining_ids = [str(unit_id) for unit_id in remaining_unit_ids]
    full_index = {unit_id: index for index, unit_id in enumerate(full_ids)}
    if len(full_index) != len(full_ids):
        raise ValueError("full_unit_ids contains duplicates")
    try:
        remaining_physical = [full_index[unit_id] for unit_id in remaining_ids]
    except KeyError as exc:
        raise ValueError(f"remaining unit is absent from full catalog: {exc.args[0]}") from exc

    base_drop = sorted({int(index) for index in dropped_indices})
    if not base_drop:
        raise ValueError("base pruning interval is empty")
    if base_drop != list(range(base_drop[0], base_drop[-1] + 1)):
        raise ValueError(f"base pruning indices are not contiguous: {base_drop}")
    base_selector = interval_selector(base_drop[0], base_drop[-1])
    logical_to_physical = {
        (int(logical[0]), int(logical[1])): (int(physical[0]), int(physical[1]))
        for physical, logical in physical_to_logical.items()
    }

    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for logical_raw in logical_spans:
        logical = (int(logical_raw[0]), int(logical_raw[1]))
        if logical not in logical_to_physical:
            raise ValueError(f"missing physical endpoints for logical span {logical}")
        logical_i, logical_j = logical
        if logical_i < 0 or logical_j < logical_i or logical_j >= len(remaining_physical):
            raise ValueError(f"invalid logical remaining span: {logical}")
        physical_i, physical_j = logical_to_physical[logical]
        additional_drop = remaining_physical[logical_i : logical_j + 1]
        combined_drop = sorted(set(base_drop).union(additional_drop))
        combined_contiguous = combined_drop == list(
            range(combined_drop[0], combined_drop[-1] + 1)
        )
        second_selector = interval_selector(physical_i, physical_j)
        if combined_contiguous:
            result_selector = interval_selector(combined_drop[0], combined_drop[-1])
            result_kind = "reused_single_interval"
        else:
            if additional_drop != list(range(physical_i, physical_j + 1)):
                raise ValueError(
                    "A non-reusable second interval must be contiguous in original "
                    f"indices, got logical={logical}, physical={additional_drop}"
                )
            result_selector = composite_selector(base_selector, second_selector)
            result_kind = "evaluated_double_interval"
        records[logical] = {
            "logical_selector": logical_selector(logical),
            "logical_start": logical_i,
            "logical_end": logical_j,
            "base_selector": base_selector,
            "second_selector": second_selector,
            "second_start": physical_i,
            "second_end": physical_j,
            "result_selector": result_selector,
            "result_kind": result_kind,
            "base_drop": list(base_drop),
            "additional_drop": list(additional_drop),
            "combined_drop": list(combined_drop),
            "base_drop_count": len(base_drop),
            "additional_drop_count": len(additional_drop),
            "total_drop_count": len(combined_drop),
        }
    return records


def evaluated_logical_spans(
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> list[Tuple[int, int]]:
    return [
        span
        for span, record in records.items()
        if str(record.get("result_kind")) == "evaluated_double_interval"
    ]


def upgraded_manifest_config(
    config: Mapping[str, Any],
    *,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    output = dict(config)
    evaluated = evaluated_logical_spans(records)
    logical_to_record = {
        logical_selector(span): dict(record) for span, record in records.items()
    }
    output["remaining_skip_layout"] = REMAINING_LAYOUT_SCHEMA
    output["base_selector"] = next(
        iter(logical_to_record.values()), {}
    ).get("base_selector", "")
    output["candidate_spans"] = [
        [int(span[0]), int(span[1])] for span in records
    ]
    output["spans"] = [[int(span[0]), int(span[1])] for span in evaluated]
    output["physical_spans"] = [
        [
            int(records[span]["second_start"]),
            int(records[span]["second_end"]),
        ]
        for span in evaluated
    ]
    output["span_records"] = logical_to_record
    return output


def manifest_payload(config: Mapping[str, Any], *, schema_version: str = "v3") -> Dict[str, Any]:
    clean = dict(config)
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema_version": str(schema_version),
        "kind": "skip_gt_eval_v3_manifest",
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "config": clean,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return raw


def _scene_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"invalid scene row at {path}:{line_no}")
            scene_id = str(row.get("scene_id", "")).strip()
            if not scene_id:
                raise ValueError(f"scene row has no scene_id at {path}:{line_no}")
            rows[scene_id] = row
    return rows


def load_reused_span_metrics(
    *,
    pruning_root: Path,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
    root = Path(pruning_root)
    output: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = defaultdict(dict)
    cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for logical, record in records.items():
        if str(record.get("result_kind")) != "reused_single_interval":
            continue
        selector = str(record["result_selector"])
        summary_path = root / selector / "summary.json"
        scenes_path = root / selector / "scenes.jsonl"
        if not summary_path.is_file() or not scenes_path.is_file():
            raise FileNotFoundError(
                f"Reusable single-interval result is incomplete: {root / selector}"
            )
        summary = _load_json(summary_path)
        expected_drop = [int(value) for value in record["combined_drop"]]
        if [int(value) for value in summary.get("drop", [])] != expected_drop:
            raise ValueError(
                f"{summary_path}: drop does not match reusable union "
                f"{expected_drop}"
            )
        if selector not in cache:
            cache[selector] = _scene_rows(scenes_path)
        for scene_id, row in cache[selector].items():
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError(f"{scenes_path}: invalid metrics for scene {scene_id}")
            output[scene_id][logical] = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
    return dict(output)


def _checkpoint_rows(staging_root: Path) -> Dict[Tuple[int, int], list[Dict[str, Any]]]:
    by_span: Dict[Tuple[int, int], list[Dict[str, Any]]] = defaultdict(list)
    for path in sorted((Path(staging_root) / "checkpoints").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    continue
                try:
                    logical = (int(row["span_i"]), int(row["span_j"]))
                except (KeyError, TypeError, ValueError):
                    continue
                by_span[logical].append(row)
    return by_span


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[str(key)].append(float(value))
    return {
        key: float(np.mean(np.asarray(metric_values, dtype=np.float64)))
        for key, metric_values in sorted(values.items())
        if metric_values
    }


def materialize_evaluated_intervals(
    *,
    staging_root: Path,
    result_root: Path,
    public_pruning_root: Path,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rows_by_span = _checkpoint_rows(staging_root)
    output: Dict[str, Dict[str, Any]] = {}
    for logical, raw_record in records.items():
        if str(raw_record.get("result_kind")) != "evaluated_double_interval":
            continue
        record = dict(raw_record)
        rows = sorted(
            rows_by_span.get(logical, []),
            key=lambda row: str(row.get("scene_id", "")),
        )
        if not rows:
            raise ValueError(f"no checkpoint rows for evaluated span {logical}")
        scene_ids = {
            str(row.get("scene_id", "")).strip()
            for row in rows
            if str(row.get("scene_id", "")).strip()
        }
        if len(scene_ids) != len(rows):
            raise ValueError(f"duplicate or missing scene IDs for evaluated span {logical}")
        selector = str(record["result_selector"])
        interval_dir = Path(result_root) / selector
        public_dir = Path(public_pruning_root) / selector
        interval_dir.mkdir(parents=True, exist_ok=True)
        scene_lines: list[str] = []
        for raw_row in rows:
            row = dict(raw_row)
            row["selector"] = selector
            row["logical_span_i"] = int(logical[0])
            row["logical_span_j"] = int(logical[1])
            row["span_i"] = int(record["second_start"])
            row["span_j"] = int(record["second_end"])
            row["base_selector"] = str(record["base_selector"])
            row["base_prune_plan"] = dict(provenance)
            scene_lines.append(json.dumps(row, separators=(",", ":")))
        (interval_dir / "scenes.jsonl").write_text(
            "\n".join(scene_lines) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "remaining_skip_eval_interval_v3",
            "kind": "remaining_skip_eval_double_interval_summary",
            "selector": selector,
            "base_selector": str(record["base_selector"]),
            "second_selector": str(record["second_selector"]),
            "logical_selector": str(record["logical_selector"]),
            "start": int(record["second_start"]),
            "end": int(record["second_end"]),
            "base_drop": list(record["base_drop"]),
            "additional_drop": list(record["additional_drop"]),
            "drop": list(record["combined_drop"]),
            "base_drop_count": int(record["base_drop_count"]),
            "additional_drop_count": int(record["additional_drop_count"]),
            "drop_count": int(record["total_drop_count"]),
            "num_scenes": len(scene_ids),
            "metrics_mean": _mean_metrics(rows),
            "scenes_path": str(public_dir / "scenes.jsonl"),
            "source_staging_root": str(Path(staging_root)),
            "base_prune_plan": dict(provenance),
        }
        (interval_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        output[logical_selector(logical)] = {
            "num_scenes": len(scene_ids),
            "metrics_mean": dict(summary["metrics_mean"]),
        }
    return output


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def update_remaining_index(
    *,
    pruning_root: Path,
    staging_root: Path,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
    provenance: Mapping[str, Any],
    interval_metadata: Mapping[str, Mapping[str, Any]],
) -> Path:
    root = Path(pruning_root)
    index_path = root / REMAINING_INDEX_FILENAME
    if index_path.is_file():
        payload = _load_json(index_path)
        if str(payload.get("schema_version")) != REMAINING_INDEX_SCHEMA:
            raise ValueError(f"unsupported remaining-skip index: {index_path}")
    else:
        payload = {
            "schema_version": REMAINING_INDEX_SCHEMA,
            "kind": "remaining_skip_eval_index",
            "plans": {},
        }
    plans = payload.setdefault("plans", {})
    if not isinstance(plans, dict):
        raise ValueError(f"{index_path}: plans must be an object")
    first = next(iter(records.values()), {})
    base_selector = str(first.get("base_selector", "")).strip()
    if not base_selector:
        raise ValueError("cannot index remaining-skip results without a base selector")
    candidates: Dict[str, Dict[str, Any]] = {}
    for logical, raw_record in records.items():
        record = dict(raw_record)
        result_selector = str(record["result_selector"])
        logical_key = logical_selector(logical)
        metadata = dict(interval_metadata.get(logical_key, {}))
        record.update(metadata)
        record["materialized"] = logical_key in interval_metadata
        record["summary_path"] = str(root / result_selector / "summary.json")
        record["scenes_path"] = str(root / result_selector / "scenes.jsonl")
        candidates[logical_key] = record
    plans[base_selector] = {
        "base_selector": base_selector,
        "staging_root": str(Path(staging_root)),
        "base_prune_plan": dict(provenance),
        "num_candidates": len(records),
        "num_reused_single_interval": sum(
            str(record.get("result_kind")) == "reused_single_interval"
            for record in records.values()
        ),
        "num_evaluated_double_interval": sum(
            str(record.get("result_kind")) == "evaluated_double_interval"
            for record in records.values()
        ),
        "candidates": candidates,
    }
    payload["active_base_selector"] = base_selector
    payload["pruning_root"] = str(root)
    _atomic_write_json(index_path, payload)
    return index_path

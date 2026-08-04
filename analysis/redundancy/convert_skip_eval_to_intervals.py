from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analysis.prune_plan import interval_selector
SKIP_EVAL_METRICS = (
    "ape",
    "are",
    "rpe_rot",
    "rpe_trans",
    "acc",
    "comp",
    "chamfer_distance",
)
_EVAL_TO_VIEW_KEY = {
    "ate": "ape",
    "cd": "chamfer_distance",
}

INDEX_FILENAME = "skip_eval_interval_index_v3.json"


def _normalize_metrics(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        val = float(value)
        if not math.isfinite(val):
            continue
        public_key = _EVAL_TO_VIEW_KEY.get(str(key), str(key))
        if public_key in SKIP_EVAL_METRICS:
            out[public_key] = val
    return out


def _checkpoint_rows(skip_eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(skip_eval_root) / "checkpoints").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _mean_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in _normalize_metrics(metrics).items():
            val = float(value)
            if math.isfinite(val):
                values[str(key)].append(val)
    return {
        key: float(np.mean(np.asarray(vals, dtype=np.float64)))
        for key, vals in sorted(values.items())
        if vals
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def materialize_skip_eval_intervals(
    skip_eval_root: Path,
    *,
    commit: bool = False,
) -> dict[str, Any]:
    root = Path(skip_eval_root)
    rows = _checkpoint_rows(root)
    by_interval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            start = int(row["span_i"])
            end = int(row["span_j"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        by_interval[interval_selector(start, end)].append(row)

    summary_legacy = _load_json(root / "skip_eval_summary_v3.json")
    manifest = _load_json(root / "skip_eval_manifest_v3.json")
    intervals: dict[str, dict[str, Any]] = {}
    writes: list[tuple[Path, str]] = []

    for selector, interval_rows in sorted(
        by_interval.items(),
        key=lambda item: (
            int(item[0].split("_j", 1)[1]) - int(item[0][1:].split("_j", 1)[0]) + 1,
            int(item[0][1:].split("_j", 1)[0]),
            item[0],
        ),
    ):
        start = int(selector[1:].split("_j", 1)[0])
        end = int(selector.split("_j", 1)[1])
        scenes = sorted(interval_rows, key=lambda r: str(r.get("scene_id", "")))
        metrics_mean = _mean_metrics(scenes)
        interval_dir = root / selector
        summary = {
            "schema_version": "skip_eval_interval_v3",
            "kind": "skip_eval_interval_summary",
            "selector": selector,
            "start": start,
            "end": end,
            "drop": list(range(start, end + 1)),
            "drop_count": end - start + 1,
            "num_scenes": len({str(r.get("scene_id", "")) for r in scenes if str(r.get("scene_id", "")).strip()}),
            "metrics_mean": metrics_mean,
            "legacy_summary_path": str(root / "skip_eval_summary_v3.json"),
        }
        scene_lines = "\n".join(json.dumps(row, separators=(",", ":")) for row in scenes)
        if scene_lines:
            scene_lines += "\n"
        writes.append((interval_dir / "summary.json", json.dumps(summary, indent=2)))
        writes.append((interval_dir / "scenes.jsonl", scene_lines))
        intervals[selector] = {
            "selector": selector,
            "start": start,
            "end": end,
            "drop_count": end - start + 1,
            "num_scenes": summary["num_scenes"],
            "summary_path": str(interval_dir / "summary.json"),
            "scenes_path": str(interval_dir / "scenes.jsonl"),
        }

    index = {
        "schema_version": "skip_eval_interval_index_v3",
        "kind": "skip_eval_interval_index",
        "skip_eval_root": str(root),
        "backbone": summary_legacy.get("backbone") or manifest.get("config", {}).get("backbone"),
        "dataset": summary_legacy.get("dataset") or manifest.get("config", {}).get("dataset"),
        "num_units": summary_legacy.get("num_units"),
        "num_intervals": len(intervals),
        "intervals": intervals,
        "legacy_files": {
            "summary": str(root / "skip_eval_summary_v3.json"),
            "report": str(root / "skip_eval_report_v3.json"),
            "manifest": str(root / "skip_eval_manifest_v3.json"),
        },
    }
    writes.append((root / INDEX_FILENAME, json.dumps(index, indent=2)))

    if commit:
        for path, text in writes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize skip_eval checkpoint rows into closed interval directories."
    )
    parser.add_argument("skip_eval_root", type=Path)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    index = materialize_skip_eval_intervals(args.skip_eval_root, commit=bool(args.commit))
    mode = "COMMIT" if args.commit else "DRY-RUN"
    print(f"[{mode}] {args.skip_eval_root}: intervals={index['num_intervals']}")


if __name__ == "__main__":
    main()

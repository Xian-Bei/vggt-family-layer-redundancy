from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

from analysis.prune_plan import (
    PRUNE_PLAN_BUNDLE_FILENAME,
    PRUNE_PLAN_BUNDLE_SCHEMA_VERSION,
    interval_selector,
)


def _parse_interval_selector(selector: str) -> tuple[int, int]:
    text = str(selector or "").strip()
    if not text.startswith("i") or "_j" not in text:
        raise ValueError(f"invalid interval selector: {selector!r}")
    left, right = text[1:].split("_j", 1)
    return int(left), int(right)


def _iter_backbone_roots(output_root: Path, backbone: str = "") -> Iterable[Path]:
    root = Path(output_root)
    if backbone:
        candidate = root / str(backbone)
        if candidate.is_dir():
            yield candidate
        return
    for candidate in sorted(root.iterdir() if root.is_dir() else []):
        if candidate.is_dir():
            yield candidate


def _closed_selector_from_open_left_selector(selector: str) -> str:
    old_i, old_j = _parse_interval_selector(selector)
    start = old_i + 1
    end = old_j
    if start > end:
        raise ValueError(f"open-left selector {selector!r} maps to empty closed interval")
    return interval_selector(start, end)


def _closed_record_from_open_left(record: dict[str, Any], fallback_selector: str) -> dict[str, Any]:
    record_selector = str(record.get("selector") or "")
    source_selector = (
        record_selector
        if record_selector.startswith("i") and "_j" in record_selector
        else str(fallback_selector)
    )
    start, end = _parse_interval_selector(_closed_selector_from_open_left_selector(source_selector))
    drop = list(range(start, end + 1))
    out = dict(record)
    out.update(
        {
            "selector": interval_selector(start, end),
            "i": int(start),
            "j": int(end),
            "start": int(start),
            "end": int(end),
            "span": len(drop),
            "drop_count": len(drop),
            "drop": drop,
            "closed_interval": True,
            "migrated_from_open_left_selector": source_selector,
        }
    )
    return out


def convert_bundle_open_left_to_closed(bundle: dict[str, Any]) -> dict[str, Any]:
    intervals_out: dict[str, dict[str, Any]] = {}
    selector_map: dict[str, str] = {}

    raw_intervals = bundle.get("intervals", {})
    if isinstance(raw_intervals, dict) and raw_intervals:
        for key, record in raw_intervals.items():
            if not isinstance(record, dict):
                continue
            old_key = str(record.get("selector") or key)
            new_record = _closed_record_from_open_left(record, old_key)
            new_key = str(new_record["selector"])
            if new_key in intervals_out and intervals_out[new_key].get("drop") != new_record.get("drop"):
                raise RuntimeError(f"bundle interval collision: {old_key!r} -> {new_key!r}")
            intervals_out[new_key] = new_record
            selector_map[old_key] = new_key
    else:
        spans = bundle.get("spans", {})
        if isinstance(spans, dict):
            for span_key, record in spans.items():
                if not isinstance(record, dict):
                    continue
                selector = str(record.get("selector") or "")
                old_key = (
                    selector
                    if selector.startswith("i") and "_j" in selector
                    else interval_selector(int(record.get("i")), int(record.get("j")))
                )
                new_record = _closed_record_from_open_left(record, old_key)
                new_key = str(new_record["selector"])
                intervals_out[new_key] = new_record
                selector_map[old_key] = new_key
                selector_map[f"prune_span{span_key}"] = new_key

    strategies_out: dict[str, dict[str, Any]] = {}
    raw_strategies = bundle.get("strategies", {})
    if isinstance(raw_strategies, dict) and raw_strategies:
        for strategy_key, strategy in raw_strategies.items():
            if not isinstance(strategy, dict):
                continue
            out = dict(strategy)
            points = strategy.get("points", {})
            points_out: dict[str, str] = {}
            if isinstance(points, dict):
                for drop_key, selector in points.items():
                    old_selector = str(selector)
                    new_selector = selector_map.get(old_selector)
                    if new_selector is None:
                        new_selector = _closed_selector_from_open_left_selector(old_selector)
                    points_out[str(drop_key)] = new_selector
            out["points"] = points_out
            strategies_out[str(strategy_key)] = out
    if not strategies_out and intervals_out:
        points_out: dict[str, str] = {}
        for selector, record in intervals_out.items():
            drop_count = int(record.get("drop_count", 0))
            if drop_count <= 0:
                continue
            existing = points_out.get(str(drop_count))
            if existing is None:
                points_out[str(drop_count)] = selector
            else:
                existing_record = intervals_out.get(existing, {})
                if int(record.get("start", 0)) < int(existing_record.get("start", 10**9)):
                    points_out[str(drop_count)] = selector
        strategies_out["ours"] = {
            "display_name": "Ours",
            "x_axis": "drop_count",
            "points": {
                str(k): points_out[str(k)]
                for k in sorted((int(x) for x in points_out.keys()))
            },
            "scores": {},
            "score_name": "corner_similarity",
            "source": {"kind": "legacy_spans"},
        }

    ours_points = strategies_out.get("ours", {}).get("points", {})
    spans_out: dict[str, dict[str, Any]] = {}
    if isinstance(ours_points, dict):
        for drop_key, selector in ours_points.items():
            rec = intervals_out.get(str(selector))
            if isinstance(rec, dict):
                spans_out[str(drop_key)] = dict(rec)

    out = dict(bundle)
    out["schema_version"] = PRUNE_PLAN_BUNDLE_SCHEMA_VERSION
    out["intervals"] = {
        key: intervals_out[key]
        for key in sorted(
            intervals_out,
            key=lambda k: (
                int(intervals_out[k].get("drop_count", 0)),
                int(intervals_out[k].get("start", 0)),
                int(intervals_out[k].get("end", 0)),
                k,
            ),
        )
    }
    out["strategies"] = strategies_out
    out["spans"] = {
        str(k): spans_out[str(k)]
        for k in sorted((int(x) for x in spans_out.keys()))
    }
    extra = dict(out.get("extra", {})) if isinstance(out.get("extra", {}), dict) else {}
    extra["interval_semantics"] = "closed interval iS_jE drops units S through E"
    extra["migrated_from_open_left_intervals"] = True
    out["extra"] = extra
    return out


def _migrate_bundle(path: Path, *, commit: bool) -> bool:
    if not path.is_file():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("schema_version", "")) == PRUNE_PLAN_BUNDLE_SCHEMA_VERSION:
        return False
    converted = convert_bundle_open_left_to_closed(raw)
    if commit:
        backup = path.with_suffix(path.suffix + ".open_left.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return True


def run(
    *,
    output_root: Path,
    backbone: str = "",
    commit: bool = False,
    migrate_bundle: bool = True,
) -> list[tuple[Path, Path]]:
    output_root = Path(output_root)
    moves: list[tuple[Path, Path]] = []
    for backbone_root in _iter_backbone_roots(output_root, backbone):
        if migrate_bundle:
            _migrate_bundle(backbone_root / PRUNE_PLAN_BUNDLE_FILENAME, commit=commit)
        for lower_root in sorted(backbone_root.glob("*/eval/lower")):
            for source in sorted(lower_root.glob("i*_j*")):
                if not source.is_dir():
                    continue
                target_name = _closed_selector_from_open_left_selector(source.name)
                target = source.parent / target_name
                if target == source:
                    continue
                if target.exists():
                    raise RuntimeError(f"migration conflict: {source} -> {target} already exists")
                moves.append((source, target))
    if commit:
        for source, target in moves:
            source.rename(target)
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate open-left iA_jB lower dirs and prune_plan_all.json to closed i{A+1}_jB."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--no-bundle", action="store_true")
    args = parser.parse_args()
    moves = run(
        output_root=args.output_root,
        backbone=args.backbone,
        commit=bool(args.commit),
        migrate_bundle=not bool(args.no_bundle),
    )
    mode = "COMMIT" if args.commit else "DRY-RUN"
    print(f"[{mode}] lower interval moves: {len(moves)}")
    for source, target in moves:
        print(f"{source} -> {target}")


if __name__ == "__main__":
    main()

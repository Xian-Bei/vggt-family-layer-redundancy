#!/usr/bin/env python3
"""
Temporary normalizer for ScanNet redundancy logs.

Goal
----
Normalize existing JSON logs to match the structure produced/consumed by
`analysis/redundancy/analyze_scannet_cache.py`.

Scope
-----
1) per-scene similarity logs:
   - keep/use: fusion_by_mode[mode].global_{patch,special}_similarity_upper_by_metric
   - enforce report_kind="similarity"
   - normalize fusion_modes to actual fusion_by_mode keys
2) per-scene skip logs:
   - convert legacy global_skip_range_loss_* fields into:
       fusion_by_mode[mode].gt_delta_upper_by_metric
   - remove legacy global_skip_range_loss_* fields
   - enforce report_kind="skip"
   - set fusion_modes from fusion_by_mode keys
3) full report logs (similarity + skip):
   - normalize report_kind
   - normalize fusion_modes from fusion_by_mode keys
   - normalize fusion_compute_similarity from fusion_similarity_metrics
   - for skip full report, convert legacy global_skip_range_loss_by_mode if needed

Notes
-----
- This script does NOT synthesize missing data (e.g. similarity "both" per-scene).
- It is intentionally structural: data values are preserved whenever present.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical_mode(mode: str) -> str:
    m = str(mode).strip()
    if m == "patch_special_bipartite":
        return "patch_special"
    return m


def _extract_skip_metric_map(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Preferred legacy container
    by_mode = row.get("global_skip_range_loss_upper_by_metric_by_mode")
    if isinstance(by_mode, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in by_mode.items():
            if isinstance(v, dict):
                out[_canonical_mode(str(k))] = v
        if out:
            return out

    # Older single-mode fallback
    single = row.get("global_skip_range_loss_upper_by_metric")
    single_mode = row.get("skipped_global_mode")
    if isinstance(single, dict) and isinstance(single_mode, str) and single_mode.strip():
        return {_canonical_mode(single_mode): single}

    # Already normalized form
    fb = row.get("fusion_by_mode")
    if isinstance(fb, dict):
        out = {}
        for mode, payload in fb.items():
            if not isinstance(payload, dict):
                continue
            gt = payload.get("gt_delta_upper_by_metric")
            if isinstance(gt, dict):
                out[_canonical_mode(str(mode))] = gt
        if out:
            return out

    return {}


def _coerce_packed_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def _finite_mean_packed_rows(rows: List[List[float]]) -> List[float]:
    if not rows:
        return []
    width = len(rows[0])
    sums = [0.0] * width
    counts = [0] * width
    for row in rows:
        if len(row) != width:
            continue
        for i, v in enumerate(row):
            if math.isnan(v):
                continue
            sums[i] += float(v)
            counts[i] += 1
    out: List[float] = []
    for i in range(width):
        if counts[i] <= 0:
            out.append(float("nan"))
        else:
            out.append(sums[i] / counts[i])
    return out


def normalize_similarity_per_scene(row: Dict[str, Any]) -> Dict[str, Any]:
    fb = row.get("fusion_by_mode")
    clean_fb: Dict[str, Dict[str, Any]] = {}
    if isinstance(fb, dict):
        for mode, payload in fb.items():
            if not isinstance(payload, dict):
                continue
            patch = payload.get("global_patch_similarity_upper_by_metric")
            special = payload.get("global_special_similarity_upper_by_metric")
            clean_payload: Dict[str, Any] = {}
            if isinstance(patch, dict):
                clean_payload["global_patch_similarity_upper_by_metric"] = patch
            if isinstance(special, dict):
                clean_payload["global_special_similarity_upper_by_metric"] = special
            if clean_payload:
                clean_fb[_canonical_mode(str(mode))] = clean_payload

    out = dict(row)
    out["report_kind"] = "similarity"
    out["fusion_by_mode"] = clean_fb
    out["fusion_modes"] = sorted(clean_fb.keys())
    out.setdefault("fusion_similarity_metrics", [])
    out.setdefault("scene_scores", {})
    return out


def normalize_skip_per_scene(row: Dict[str, Any]) -> Dict[str, Any]:
    by_mode_metric = _extract_skip_metric_map(row)
    clean_fb = {
        mode: {"gt_delta_upper_by_metric": metric_map}
        for mode, metric_map in by_mode_metric.items()
        if isinstance(metric_map, dict)
    }

    out = dict(row)
    out["report_kind"] = "skip"
    out["fusion_by_mode"] = clean_fb
    out["fusion_modes"] = sorted(clean_fb.keys())
    out.setdefault("scene_scores", {})

    # Remove legacy containers to keep the format aligned.
    out.pop("global_skip_range_loss_upper_by_metric_by_mode", None)
    out.pop("global_skip_range_loss_upper_by_metric", None)
    out.pop("global_skip_range_loss_by_mode", None)
    out.pop("global_skip_range_loss", None)
    out.pop("skipped_global_mode", None)
    return out


def normalize_similarity_full(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["report_kind"] = "similarity"
    fb = out.get("fusion_by_mode", {})
    if not isinstance(fb, dict):
        fb = {}
    out["fusion_by_mode"] = fb
    out["fusion_modes"] = sorted(str(k) for k in fb.keys())
    metrics = out.get("fusion_similarity_metrics", [])
    if not isinstance(metrics, list):
        metrics = []
    out["fusion_similarity_metrics"] = metrics
    out["fusion_compute_similarity"] = bool(metrics)
    if isinstance(out.get("scene_ids"), list):
        out["num_scenes"] = len(out["scene_ids"])
    return out


def normalize_skip_full(row: Dict[str, Any], *, skip_per_scene_dir: Path | None = None) -> Dict[str, Any]:
    out = dict(row)
    out["report_kind"] = "skip"

    clean_fb: Dict[str, Dict[str, Any]] = {}

    # Preferred behavior: rebuild skip fusion_by_mode from per-scene logs.
    if skip_per_scene_dir is not None and skip_per_scene_dir.exists():
        mode_metric_rows: Dict[str, Dict[str, List[List[float]]]] = {}
        mode_scene_ids: Dict[str, set[str]] = {}
        scene_ids_seen: set[str] = set()
        for p in sorted(skip_per_scene_dir.glob("*_skip.json")):
            try:
                scene_row = _load_json(p)
            except Exception:
                continue
            sid = str(scene_row.get("scene_id", "")).strip()
            if sid:
                scene_ids_seen.add(sid)
            by_mode = _extract_skip_metric_map(scene_row)
            for mode, metric_map in by_mode.items():
                if not isinstance(metric_map, dict):
                    continue
                if sid:
                    mode_scene_ids.setdefault(mode, set()).add(sid)
                mm_rows = mode_metric_rows.setdefault(mode, {})
                for mk, packed_any in metric_map.items():
                    packed = _coerce_packed_list(packed_any)
                    if not packed:
                        continue
                    mm_rows.setdefault(str(mk), []).append(packed)

        for mode, per_metric_rows in mode_metric_rows.items():
            gt_map: Dict[str, List[float]] = {}
            for mk, rows in per_metric_rows.items():
                avg = _finite_mean_packed_rows(rows)
                if avg:
                    gt_map[str(mk)] = avg
            clean_fb[mode] = {
                "num_scenes_used": len(mode_scene_ids.get(mode, set())),
                "gt_delta_upper_by_metric": gt_map,
            }

        # Keep full-report scene metadata aligned with data source.
        if scene_ids_seen:
            out["scene_ids"] = sorted(scene_ids_seen)
            out["num_scenes"] = len(scene_ids_seen)

    # Legacy fallback at full-report level
    if not clean_fb:
        fb = out.get("fusion_by_mode")
        if isinstance(fb, dict):
            for mode, payload in fb.items():
                if not isinstance(payload, dict):
                    continue
                gt = payload.get("gt_delta_upper_by_metric")
                if isinstance(gt, dict):
                    clean_fb[_canonical_mode(str(mode))] = {
                        "num_scenes_used": int(payload.get("num_scenes_used", out.get("num_scenes", 0) or 0)),
                        "gt_delta_upper_by_metric": gt,
                    }

    if not clean_fb:
        legacy = out.get("global_skip_range_loss_by_mode")
        if isinstance(legacy, dict):
            for mode, payload in legacy.items():
                if not isinstance(payload, dict):
                    continue
                g = payload.get("global_skip_range_loss_upper_by_metric")
                if isinstance(g, dict):
                    cmode = _canonical_mode(str(mode))
                    clean_fb[cmode] = {
                        "num_scenes_used": int(out.get("num_scenes", 0) or 0),
                        "gt_delta_upper_by_metric": g,
                    }

    out["fusion_by_mode"] = clean_fb
    out["fusion_modes"] = sorted(clean_fb.keys())
    metrics = out.get("fusion_similarity_metrics", [])
    if not isinstance(metrics, list):
        metrics = []
    out["fusion_similarity_metrics"] = metrics
    out["fusion_compute_similarity"] = bool(metrics)
    if isinstance(out.get("scene_ids"), list):
        out["num_scenes"] = len(out["scene_ids"])

    # Remove legacy fields if present
    out.pop("global_skip_range_loss_by_mode", None)
    out.pop("global_skip_range_loss", None)
    return out


def _normalize_json_file(
    path: Path,
    *,
    kind: str,
    level: str,
    dry_run: bool,
    backup: bool,
    skip_per_scene_dir: Path | None = None,
) -> tuple[bool, str]:
    try:
        row = _load_json(path)
    except Exception as exc:
        return False, f"read/parse failed: {exc}"

    if kind == "similarity" and level == "per_scene":
        norm = normalize_similarity_per_scene(row)
    elif kind == "skip" and level == "per_scene":
        norm = normalize_skip_per_scene(row)
    elif kind == "similarity" and level == "full":
        norm = normalize_similarity_full(row)
    elif kind == "skip" and level == "full":
        norm = normalize_skip_full(row, skip_per_scene_dir=skip_per_scene_dir)
    else:
        return False, f"unsupported kind/level: {kind}/{level}"

    if norm == row:
        return True, "already normalized"

    if dry_run:
        return True, "would normalize"

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    _write_json(path, norm)
    return True, "normalized"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ScanNet redundancy logs to analyze_scannet_cache format.")
    parser.add_argument(
        "--root",
        default="analysis_outputs",
        help="Root directory containing scannet_redundancy_similarity and scannet_redundancy_skip",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not write files")
    parser.add_argument("--backup", action="store_true", help="Create .bak backups before overwrite")
    args = parser.parse_args()

    root = Path(args.root)
    sim_dir = root / "scannet_redundancy_similarity"
    skip_dir = root / "scannet_redundancy_skip"
    sim_ps = sim_dir / "per_scene"
    skip_ps = skip_dir / "per_scene"
    sim_full = sim_dir / "full_report_similarity.json"
    skip_full = skip_dir / "full_report_skip.json"

    jobs = []
    for p in sorted(sim_ps.glob("*_similarity.json")):
        jobs.append((p, "similarity", "per_scene"))
    for p in sorted(skip_ps.glob("*_skip.json")):
        jobs.append((p, "skip", "per_scene"))
    if sim_full.exists():
        jobs.append((sim_full, "similarity", "full"))
    if skip_full.exists():
        jobs.append((skip_full, "skip", "full"))

    if not jobs:
        print("[ERROR] no target files found")
        return 2

    ok, fail = 0, 0
    for path, kind, level in jobs:
        success, msg = _normalize_json_file(
            path,
            kind=kind,
            level=level,
            dry_run=bool(args.dry_run),
            backup=bool(args.backup),
            skip_per_scene_dir=skip_ps,
        )
        tag = "OK" if success else "FAIL"
        print(f"[{tag}] {kind}/{level}: {path} -> {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone. success={ok}, failed={fail}, total={len(jobs)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

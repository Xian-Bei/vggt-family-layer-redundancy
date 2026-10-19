"""Validation and result collection helpers for calibration-sample ablations."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from analysis.prune_plan.recovery_plan import (
    list_recovery_plan_names,
    load_recovery_plan_bundle,
)
from analysis.schema.v3 import CacheManifestV3


CACHE_FIT_METHODS = {
    "recovery_linearpatch",
    "recovery_ghost",
    "recovery_ghost_interval",
    "recovery_ghost_interval_sequential",
    "recovery_ghost_interval_sequential_grouped",
    "recovery_ghost_interval_sequential_autogroup",
    "recovery_replaceme_ls",
    "recovery_replaceme_cos",
    "recovery_streamline_ffn",
}


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_prefix_dirs(root: Path, scene_ids: Sequence[str], count: int) -> None:
    if len(scene_ids) < count:
        raise ValueError(f"{root}: manifest has {len(scene_ids)} scenes, need {count}")
    missing = [scene_id for scene_id in scene_ids[:count] if not (root / scene_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"{root}: missing selected scene directories {missing[:5]}")


def validate_data(args: argparse.Namespace) -> None:
    extra_needed = int(args.extra_needed)
    if extra_needed < 0:
        raise ValueError("--extra-needed must be non-negative")

    scannet_base = set(_read_lines(args.scannet_base_list))
    scannet_test = set(_read_lines(args.scannet_test_list))
    if len(scannet_base) < int(args.base_count):
        raise ValueError(f"ScanNet base split has {len(scannet_base)} scenes")

    nuscenes_base = _load_json(args.nuscenes_base_manifest).get("splits", {})
    if not isinstance(nuscenes_base, dict) or not isinstance(nuscenes_base.get("train"), list):
        raise ValueError(f"Invalid base nuScenes manifest: {args.nuscenes_base_manifest}")
    if len(nuscenes_base["train"]) < int(args.base_count):
        raise ValueError(f"nuScenes base train split has {len(nuscenes_base['train'])} scenes")

    if extra_needed == 0:
        return

    scannet_extra_payload = _load_json(args.scannet_extra_root / "selected_scenes.json")
    scannet_extra = scannet_extra_payload.get("scene_ids")
    if not isinstance(scannet_extra, list) or not all(isinstance(x, str) for x in scannet_extra):
        raise ValueError("Invalid ScanNet extra selection manifest")
    overlap = set(scannet_extra) & (scannet_base | scannet_test)
    if overlap:
        raise ValueError(f"ScanNet extra pool overlaps main splits: {sorted(overlap)[:5]}")
    _require_prefix_dirs(args.scannet_extra_root, scannet_extra, extra_needed)

    nuscenes_extra_payload = _load_json(args.nuscenes_extra_root / "selected_scenes.json")
    nuscenes_splits = nuscenes_extra_payload.get("splits", {})
    nuscenes_extra = nuscenes_splits.get("train") if isinstance(nuscenes_splits, dict) else None
    if not isinstance(nuscenes_extra, list) or not all(isinstance(x, str) for x in nuscenes_extra):
        raise ValueError("Invalid nuScenes extra selection manifest")
    base_names = {
        str(name)
        for names in nuscenes_base.values()
        if isinstance(names, list)
        for name in names
    }
    overlap = set(nuscenes_extra) & base_names
    if overlap:
        raise ValueError(f"nuScenes extra pool overlaps main splits: {sorted(overlap)[:5]}")
    _require_prefix_dirs(args.nuscenes_extra_root, nuscenes_extra, extra_needed)


def recovery_names(plan_root: Path, methods: Iterable[str]) -> list[str]:
    requested = set(methods)
    unknown = requested - CACHE_FIT_METHODS
    if unknown:
        raise ValueError(
            "Sample ablation supports cache-fit recovery only; unsupported methods: "
            + ", ".join(sorted(unknown))
        )
    names = []
    found_methods = set()
    for name in list_recovery_plan_names(plan_root):
        bundle = load_recovery_plan_bundle(plan_root, name)
        method = str(bundle.get("method", "")).strip()
        if method in requested:
            names.append(name)
            found_methods.add(method)
    missing = requested - found_methods
    if missing:
        raise ValueError(
            f"{plan_root}: no recovery plan for methods {', '.join(sorted(missing))}"
        )
    return names


def validate_cache(cache_root: Path, backbone: str, expected: Sequence[str]) -> None:
    for item in expected:
        dataset, count_raw = item.split("=", 1)
        count = int(count_raw)
        path = cache_root / backbone / dataset / "cache" / "cache_manifest_v3.json"
        manifest = CacheManifestV3.load(path)
        if manifest.dataset != dataset:
            raise ValueError(f"{path}: dataset={manifest.dataset!r}, expected {dataset!r}")
        if len(manifest.scenes) < count:
            raise ValueError(f"{path}: scenes={len(manifest.scenes)}, expected at least {count}")


def init_manifest(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "lare_samples_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_output_root": str(args.source_output_root.resolve()),
        "cache_root": str((args.output_root / "cache").resolve()),
        "sample_counts_per_domain": [int(x) for x in _split(args.samples)],
        "base_scenes_per_domain": int(args.base_count),
        "max_extra_scenes_per_domain": int(args.max_extra),
        "backbones": _split(args.backbones),
        "recovery_methods": _split(args.methods),
        "evaluation_datasets": _split(args.eval_datasets),
        "sampling_semantics": "N scenes per domain: fixed base prefix plus extra-pool prefix",
    }
    (args.output_root / "samples_ablation_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def summarize(output_root: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("samples_*/*/*/eval/recovery/*/*/eval_record_v3.json")):
        rel = path.relative_to(output_root)
        sample_dir, backbone, dataset, _, _, selector, recovery_plan, _ = rel.parts
        sample_count = int(sample_dir.removeprefix("samples_"))
        record = _load_json(path)
        average = record.get("average", {})
        recovery_summary_path = (
            output_root
            / sample_dir
            / backbone
            / "recovery"
            / selector
            / recovery_plan
            / "recovery_summary.json"
        )
        recovery_summary = _load_json(recovery_summary_path)
        fit_num_scenes = int(recovery_summary.get("num_scenes", 0))
        scene_limits = recovery_summary.get("scene_limits_by_dataset", {})
        for metric, value in sorted(average.items() if isinstance(average, dict) else []):
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "sample_count_per_domain": sample_count,
                    "backbone": backbone,
                    "dataset": dataset,
                    "selector": selector,
                    "recovery_plan": recovery_plan,
                    "metric": metric,
                    "value": value,
                    "num_eval_scenes": len(record.get("scenes", {})),
                    "fit_num_scenes": fit_num_scenes,
                    "scene_limits_by_dataset": json.dumps(scene_limits, sort_keys=True),
                }
            )
    csv_path = output_root / "samples_ablation_results.csv"
    fieldnames = [
        "sample_count_per_domain",
        "backbone",
        "dataset",
        "selector",
        "recovery_plan",
        "metric",
        "value",
        "num_eval_scenes",
        "fit_num_scenes",
        "scene_limits_by_dataset",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("validate-data")
    data.add_argument("--base-count", type=int, default=50)
    data.add_argument("--extra-needed", type=int, required=True)
    data.add_argument("--scannet-base-list", type=Path, required=True)
    data.add_argument("--scannet-test-list", type=Path, required=True)
    data.add_argument("--scannet-extra-root", type=Path, required=True)
    data.add_argument("--nuscenes-base-manifest", type=Path, required=True)
    data.add_argument("--nuscenes-extra-root", type=Path, required=True)

    names = sub.add_parser("recovery-names")
    names.add_argument("--plan-root", type=Path, required=True)
    names.add_argument("--methods", required=True)

    cache = sub.add_parser("validate-cache")
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--backbone", required=True)
    cache.add_argument("--expect", action="append", default=[], required=True)

    manifest = sub.add_parser("init-manifest")
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--source-output-root", type=Path, required=True)
    manifest.add_argument("--samples", required=True)
    manifest.add_argument("--base-count", type=int, required=True)
    manifest.add_argument("--max-extra", type=int, required=True)
    manifest.add_argument("--backbones", required=True)
    manifest.add_argument("--methods", required=True)
    manifest.add_argument("--eval-datasets", required=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-data":
        validate_data(args)
    elif args.command == "recovery-names":
        print(",".join(recovery_names(args.plan_root, _split(args.methods))))
    elif args.command == "validate-cache":
        validate_cache(args.cache_root, args.backbone, args.expect)
    elif args.command == "init-manifest":
        init_manifest(args)
    elif args.command == "summarize":
        print(f"rows={summarize(args.output_root)}")


if __name__ == "__main__":
    main()

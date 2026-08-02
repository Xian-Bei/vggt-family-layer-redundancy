from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from analysis.redundancy.remaining_skip_layout import (
    REMAINING_INTERNAL_DIR,
    build_remaining_span_records,
    evaluated_logical_spans,
    logical_selector,
    manifest_payload,
    materialize_evaluated_intervals,
    update_remaining_index,
    upgraded_manifest_config,
)


LEGACY_DIRNAME = "remaining_pruning.legacy"


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _json_lines(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"invalid JSONL row at {path}:{line_no}")
            rows.append(row)
    return rows


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _records_from_config(
    config: Mapping[str, Any],
) -> tuple[
    Dict[Tuple[int, int], Dict[str, Any]],
    list[int],
    list[str],
    list[str],
]:
    full_ids = [str(unit_id) for unit_id in config.get("full_unit_ids", [])]
    remaining_ids = [str(unit_id) for unit_id in config.get("unit_ids", [])]
    provenance = config.get("base_prune_plan", {})
    if not full_ids or not remaining_ids or not isinstance(provenance, Mapping):
        raise ValueError("legacy manifest lacks full/remaining unit provenance")
    full_index = {unit_id: index for index, unit_id in enumerate(full_ids)}
    dropped_indices = [
        full_index[str(unit_id)]
        for unit_id in provenance.get("dropped_unit_ids", [])
    ]
    logical_spans = [
        (int(span[0]), int(span[1])) for span in config.get("spans", [])
    ]
    physical_spans = [
        (int(span[0]), int(span[1]))
        for span in config.get("physical_spans", [])
    ]
    if len(logical_spans) != len(physical_spans):
        raise ValueError("legacy manifest logical/physical span counts differ")
    physical_to_logical = dict(zip(physical_spans, logical_spans))
    records = build_remaining_span_records(
        full_unit_ids=full_ids,
        remaining_unit_ids=remaining_ids,
        dropped_indices=dropped_indices,
        logical_spans=logical_spans,
        physical_to_logical=physical_to_logical,
    )
    return records, dropped_indices, full_ids, remaining_ids


def _scene_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row in _json_lines(path):
        scene_id = str(row.get("scene_id", "")).strip()
        if scene_id:
            output[scene_id] = dict(row.get("metrics", {}))
    return output


def _validate_reusable_results(
    *,
    old_root: Path,
    pruning_root: Path,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[str, int]:
    reusable = {
        span: record
        for span, record in records.items()
        if str(record.get("result_kind")) == "reused_single_interval"
    }
    direct_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_exact = 0
    scene_exact = 0
    for logical, record in reusable.items():
        direct_selector = str(record["result_selector"])
        direct_dir = pruning_root / direct_selector
        direct_summary_path = direct_dir / "summary.json"
        direct_scenes_path = direct_dir / "scenes.jsonl"
        if not direct_summary_path.is_file() or not direct_scenes_path.is_file():
            raise FileNotFoundError(
                f"missing reusable single-interval result: {direct_dir}"
            )
        direct_summary = _load_json(direct_summary_path)
        if [int(value) for value in direct_summary.get("drop", [])] != [
            int(value) for value in record["combined_drop"]
        ]:
            raise ValueError(f"drop mismatch for reusable result {direct_dir}")
        direct_cache[direct_selector] = _scene_metrics(direct_scenes_path)

        old_dir = old_root / logical_selector(logical)
        old_summary_path = old_dir / "summary.json"
        if old_summary_path.is_file():
            old_summary = _load_json(old_summary_path)
            if not _metrics_equal(
                old_summary.get("metrics_mean", {}),
                direct_summary.get("metrics_mean", {}),
            ):
                raise ValueError(
                    f"summary metrics differ: {old_summary_path} vs {direct_summary_path}"
                )
            summary_exact += 1
            old_scenes = _scene_metrics(old_dir / "scenes.jsonl")
            if set(old_scenes) != set(direct_cache[direct_selector]):
                raise ValueError(
                    f"scene sets differ: {old_dir} vs {direct_dir}"
                )
            for scene_id, metrics in old_scenes.items():
                if not _metrics_equal(
                    metrics, direct_cache[direct_selector][scene_id]
                ):
                    raise ValueError(
                        f"scene metrics differ for {scene_id}: {old_dir} vs {direct_dir}"
                    )
                scene_exact += 1

    checkpoint_exact = 0
    for checkpoint_path in sorted((old_root / "checkpoints").glob("*.jsonl")):
        for row in _json_lines(checkpoint_path):
            logical = (int(row["span_i"]), int(row["span_j"]))
            record = reusable.get(logical)
            if record is None:
                continue
            selector = str(record["result_selector"])
            scene_id = str(row.get("scene_id", "")).strip()
            direct_metrics = direct_cache[selector].get(scene_id)
            if direct_metrics is None:
                raise ValueError(
                    f"direct result {selector} lacks checkpoint scene {scene_id}"
                )
            if not _metrics_equal(row.get("metrics", {}), direct_metrics):
                raise ValueError(
                    f"checkpoint metrics differ for {logical}, scene={scene_id}"
                )
            checkpoint_exact += 1
    return {
        "reusable_candidates": len(reusable),
        "materialized_summaries_exact": summary_exact,
        "materialized_scene_rows_exact": scene_exact,
        "checkpoint_rows_exact": checkpoint_exact,
    }


def inspect_migration(old_root: Path) -> Dict[str, Any]:
    old_root = Path(old_root)
    manifest_path = old_root / "skip_eval_manifest_v3.json"
    config = _load_json(manifest_path).get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{manifest_path}: config must be an object")
    records, _dropped, _full, _remaining = _records_from_config(config)
    evaluated = evaluated_logical_spans(records)
    first = next(iter(records.values()))
    base_selector = str(first["base_selector"])
    pruning_root = old_root.parent / "pruning"
    staging_root = pruning_root / REMAINING_INTERNAL_DIR / base_selector
    legacy_root = old_root.parent / LEGACY_DIRNAME
    collisions = sorted(
        str(pruning_root / str(records[span]["result_selector"]))
        for span in evaluated
        if (pruning_root / str(records[span]["result_selector"])).exists()
    )
    validation = _validate_reusable_results(
        old_root=old_root,
        pruning_root=pruning_root,
        records=records,
    )
    materialized_evaluated = sum(
        (old_root / logical_selector(span) / "summary.json").is_file()
        for span in evaluated
    )
    checkpoint_files = list((old_root / "checkpoints").glob("*.jsonl"))
    checkpoint_rows = sum(len(_json_lines(path)) for path in checkpoint_files)
    return {
        "source": str(old_root),
        "pruning_root": str(pruning_root),
        "base_selector": base_selector,
        "destination_staging": str(staging_root),
        "legacy_backup": str(legacy_root),
        "num_candidates": len(records),
        "num_reused_single_interval": len(records) - len(evaluated),
        "num_evaluated_double_interval": len(evaluated),
        "num_materialized_double_interval": materialized_evaluated,
        "checkpoint_files": len(checkpoint_files),
        "checkpoint_rows": checkpoint_rows,
        "destination_collisions": collisions,
        "destination_staging_exists": staging_root.exists(),
        "legacy_backup_exists": legacy_root.exists(),
        "validation": validation,
    }


def _rewrite_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(old_root), str(new_root))
    if isinstance(value, list):
        return [
            _rewrite_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _rewrite_paths(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    return value


def _copy_central_outputs(old_root: Path, temporary_root: Path, final_root: Path) -> None:
    for filename in (
        "skip_eval_report_v3.json",
        "skip_eval_scenes_v3.json",
        "skip_eval_summary_v3.json",
    ):
        source = old_root / filename
        if not source.is_file():
            continue
        payload = _rewrite_paths(
            _load_json(source),
            old_root=old_root,
            new_root=final_root,
        )
        (temporary_root / filename).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    npz_source = old_root / "skip_eval_v3.npz"
    if npz_source.is_file():
        shutil.copy2(npz_source, temporary_root / npz_source.name)


def _copy_filtered_checkpoints(
    *,
    old_root: Path,
    temporary_root: Path,
    evaluated: set[Tuple[int, int]],
) -> int:
    destination = temporary_root / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted((old_root / "checkpoints").glob("*.jsonl")):
        lines = []
        for row in _json_lines(source):
            logical = (int(row["span_i"]), int(row["span_j"]))
            if logical in evaluated:
                lines.append(json.dumps(row, separators=(",", ":")))
                copied += 1
        (destination / source.name).write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
    return copied


def _direct_interval_metadata(
    pruning_root: Path,
    records: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for logical, record in records.items():
        if str(record.get("result_kind")) != "reused_single_interval":
            continue
        summary = _load_json(
            pruning_root / str(record["result_selector"]) / "summary.json"
        )
        output[logical_selector(logical)] = {
            "num_scenes": int(summary.get("num_scenes", 0)),
            "metrics_mean": dict(summary.get("metrics_mean", {})),
        }
    return output


def commit_migration(old_root: Path) -> Dict[str, Any]:
    inspection = inspect_migration(old_root)
    if inspection["destination_collisions"]:
        raise FileExistsError(
            f"composite result destinations already exist: "
            f"{inspection['destination_collisions'][:8]}"
        )
    if inspection["destination_staging_exists"]:
        raise FileExistsError(inspection["destination_staging"])
    if inspection["legacy_backup_exists"]:
        raise FileExistsError(inspection["legacy_backup"])

    old_root = Path(old_root)
    pruning_root = old_root.parent / "pruning"
    manifest = _load_json(old_root / "skip_eval_manifest_v3.json")
    config = dict(manifest["config"])
    records, _dropped, _full, _remaining = _records_from_config(config)
    evaluated = set(evaluated_logical_spans(records))
    provenance = dict(config["base_prune_plan"])
    base_selector = str(next(iter(records.values()))["base_selector"])
    final_staging = pruning_root / REMAINING_INTERNAL_DIR / base_selector
    final_staging.parent.mkdir(parents=True, exist_ok=True)
    temporary_staging = Path(
        tempfile.mkdtemp(
            prefix=f".migrate-{base_selector}-",
            dir=final_staging.parent,
        )
    )
    temporary_results = Path(
        tempfile.mkdtemp(
            prefix=f".migrate-results-{base_selector}-",
            dir=pruning_root,
        )
    )
    committed_results: list[Path] = []
    try:
        upgraded = upgraded_manifest_config(config, records=records)
        payload = manifest_payload(
            upgraded,
            schema_version=str(manifest.get("schema_version", "v3")),
        )
        (temporary_staging / "skip_eval_manifest_v3.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        copied_checkpoint_rows = _copy_filtered_checkpoints(
            old_root=old_root,
            temporary_root=temporary_staging,
            evaluated=evaluated,
        )
        _copy_central_outputs(old_root, temporary_staging, final_staging)

        interval_metadata = _direct_interval_metadata(pruning_root, records)
        has_materialized_results = (
            old_root / "skip_eval_interval_index_v3.json"
        ).is_file()
        if has_materialized_results:
            evaluated_metadata = materialize_evaluated_intervals(
                staging_root=temporary_staging,
                result_root=temporary_results,
                public_pruning_root=pruning_root,
                records=records,
                provenance=provenance,
            )
            interval_metadata.update(evaluated_metadata)

        for source in sorted(temporary_results.iterdir()):
            destination = pruning_root / source.name
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(source, destination)
            committed_results.append(destination)
        os.replace(temporary_staging, final_staging)
        legacy_root = old_root.parent / LEGACY_DIRNAME
        os.replace(old_root, legacy_root)
        index_path = update_remaining_index(
            pruning_root=pruning_root,
            staging_root=final_staging,
            records=records,
            provenance=provenance,
            interval_metadata=interval_metadata,
        )
        inspection["committed"] = True
        inspection["copied_evaluated_checkpoint_rows"] = copied_checkpoint_rows
        inspection["index_path"] = str(index_path)
        return inspection
    except Exception:
        if final_staging.exists() and not temporary_staging.exists():
            os.replace(final_staging, temporary_staging)
        legacy_root = old_root.parent / LEGACY_DIRNAME
        if not old_root.exists() and legacy_root.exists():
            os.replace(legacy_root, old_root)
        for destination in reversed(committed_results):
            if destination.exists():
                os.replace(destination, temporary_results / destination.name)
        raise
    finally:
        if temporary_staging.exists():
            shutil.rmtree(temporary_staging)
        if temporary_results.exists():
            shutil.rmtree(temporary_results)


def discover_legacy_roots(output_root: Path) -> Iterable[Path]:
    yield from sorted(Path(output_root).glob("*/*/eval/remaining_pruning"))


def _records_from_upgraded_config(
    config: Mapping[str, Any],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    raw_records = config.get("span_records", {})
    if not isinstance(raw_records, Mapping):
        raise ValueError("migrated manifest has no span_records mapping")
    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw in raw_records.values():
        if not isinstance(raw, Mapping):
            continue
        logical = (int(raw["logical_start"]), int(raw["logical_end"]))
        records[logical] = dict(raw)
    return records


def verify_migrated_backup(
    legacy_root: Path,
    *,
    refresh_index: bool = False,
) -> Dict[str, Any]:
    legacy_root = Path(legacy_root)
    pruning_root = legacy_root.parent / "pruning"
    legacy_manifest = _load_json(legacy_root / "skip_eval_manifest_v3.json")
    legacy_config = dict(legacy_manifest["config"])
    legacy_records, _drop, _full, _remaining = _records_from_config(legacy_config)
    base_selector = str(next(iter(legacy_records.values()))["base_selector"])
    staging_root = pruning_root / REMAINING_INTERNAL_DIR / base_selector
    migrated_manifest = _load_json(staging_root / "skip_eval_manifest_v3.json")
    migrated_config = dict(migrated_manifest["config"])
    expected_config = upgraded_manifest_config(
        legacy_config,
        records=legacy_records,
    )
    normalized_migrated = dict(migrated_config)
    normalized_expected = dict(expected_config)
    migrated_base = normalized_migrated.get("base_prune_plan", {})
    expected_base = normalized_expected.get("base_prune_plan", {})
    same_base_drop = (
        isinstance(migrated_base, Mapping)
        and isinstance(expected_base, Mapping)
        and list(migrated_base.get("dropped_unit_ids", []))
        == list(expected_base.get("dropped_unit_ids", []))
    )
    if same_base_drop:
        normalized_migrated["base_prune_plan"] = normalized_expected.get(
            "base_prune_plan"
        )
    if normalized_migrated != normalized_expected:
        raise ValueError(f"migrated manifest config differs at {staging_root}")
    expected_manifest = manifest_payload(
        migrated_config,
        schema_version=str(migrated_manifest.get("schema_version", "v3")),
    )
    if migrated_manifest.get("fingerprint") != expected_manifest["fingerprint"]:
        raise ValueError(f"migrated manifest fingerprint differs at {staging_root}")
    migrated_records = _records_from_upgraded_config(migrated_config)
    if migrated_records != legacy_records:
        raise ValueError(f"migrated span records differ at {staging_root}")

    evaluated = set(evaluated_logical_spans(legacy_records))
    checkpoint_rows = 0
    checkpoint_scenes: set[str] = set()
    for path in sorted((staging_root / "checkpoints").glob("*.jsonl")):
        for row in _json_lines(path):
            logical = (int(row["span_i"]), int(row["span_j"]))
            if logical not in evaluated:
                raise ValueError(
                    f"reused span remained in migrated checkpoint: {path}, {logical}"
                )
            checkpoint_rows += 1
            checkpoint_scenes.add(str(row.get("scene_id", "")))

    interval_metadata = _direct_interval_metadata(pruning_root, legacy_records)
    materialized_double = 0
    compared_summaries = 0
    compared_scene_rows = 0
    for logical in evaluated:
        record = legacy_records[logical]
        destination = pruning_root / str(record["result_selector"])
        summary_path = destination / "summary.json"
        if not summary_path.is_file():
            continue
        materialized_double += 1
        new_summary = _load_json(summary_path)
        old_dir = legacy_root / logical_selector(logical)
        old_summary_path = old_dir / "summary.json"
        if old_summary_path.is_file():
            old_summary = _load_json(old_summary_path)
            if not _metrics_equal(
                old_summary.get("metrics_mean", {}),
                new_summary.get("metrics_mean", {}),
            ):
                raise ValueError(
                    f"migrated summary metrics differ: {old_summary_path}"
                )
            compared_summaries += 1
            old_scenes = _scene_metrics(old_dir / "scenes.jsonl")
            new_scenes = _scene_metrics(destination / "scenes.jsonl")
            if old_scenes != new_scenes:
                raise ValueError(f"migrated scene metrics differ: {old_dir}")
            compared_scene_rows += len(old_scenes)
        interval_metadata[logical_selector(logical)] = {
            "num_scenes": int(new_summary.get("num_scenes", 0)),
            "metrics_mean": dict(new_summary.get("metrics_mean", {})),
        }
    if refresh_index:
        update_remaining_index(
            pruning_root=pruning_root,
            staging_root=staging_root,
            records=legacy_records,
            provenance=dict(legacy_config["base_prune_plan"]),
            interval_metadata=interval_metadata,
        )
    index = _load_json(pruning_root / "remaining_skip_eval_index_v3.json")
    plan_index = index.get("plans", {}).get(base_selector, {})
    if int(plan_index.get("num_candidates", -1)) != len(legacy_records):
        raise ValueError(f"remaining-skip index candidate count differs: {pruning_root}")
    return {
        "legacy_backup": str(legacy_root),
        "staging_root": str(staging_root),
        "base_selector": base_selector,
        "num_candidates": len(legacy_records),
        "num_reused_single_interval": len(legacy_records) - len(evaluated),
        "num_evaluated_double_interval": len(evaluated),
        "num_materialized_double_interval": materialized_double,
        "migrated_checkpoint_rows": checkpoint_rows,
        "checkpoint_scenes": len(checkpoint_scenes),
        "compared_summaries_exact": compared_summaries,
        "compared_scene_rows_exact": compared_scene_rows,
        "index_refreshed": bool(refresh_index),
    }


def discover_migrated_backups(output_root: Path) -> Iterable[Path]:
    yield from sorted(Path(output_root).glob("*/*/eval/remaining_pruning.legacy"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate remaining-skip results into eval/pruning with original-index "
            "double-interval selectors. The default mode is read-only."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs_v3"),
    )
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--verify-migrated", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_migrated or args.refresh_index:
        roots = list(discover_migrated_backups(args.output_root))
        reports = [
            verify_migrated_backup(root, refresh_index=bool(args.refresh_index))
            for root in roots
        ]
        mode = "refresh-index" if args.refresh_index else "verify-migrated"
    else:
        roots = list(discover_legacy_roots(args.output_root))
        reports = [
            commit_migration(root) if args.commit else inspect_migration(root)
            for root in roots
        ]
        mode = "commit" if args.commit else "dry-run"
    print(
        json.dumps(
            {
                "mode": mode,
                "num_roots": len(roots),
                "reports": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

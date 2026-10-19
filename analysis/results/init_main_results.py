"""Refresh reusable analysis artifacts in an isolated main-results tree."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


INIT_MANIFEST = "main_results_init.json"


def _split_list(raw: str | Sequence[str]) -> list[str]:
    values = raw if not isinstance(raw, str) else raw.replace(",", " ").split()
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_upper_record(path: Path, *, backbone: str, dataset: str) -> None:
    record = _load_object(path)
    if str(record.get("backbone", "")).strip() != backbone:
        raise ValueError(f"upper record backbone mismatch: {path}")
    recorded_dataset = str(record.get("dataset", "")).strip()
    accepted_datasets = {dataset, dataset.partition("_")[0]}
    if recorded_dataset not in accepted_datasets:
        raise ValueError(f"upper record dataset mismatch: {path}")
    if str(record.get("model_name", "")).strip() != "upper":
        raise ValueError(f"upper record model_name mismatch: {path}")


def _require_complete_skip_eval(
    pruning_root: Path,
    *,
    backbone: str,
    dataset: str,
    required_summary_max_drop: int,
) -> tuple[list[str], dict[str, int]]:
    report_path = pruning_root / "skip_eval_report_v3.json"
    report = _load_object(report_path)
    if str(report.get("backbone", "")).strip() != backbone:
        raise ValueError(f"skip-eval report backbone mismatch: {report_path}")
    if str(report.get("dataset", "")).strip() != dataset:
        raise ValueError(f"skip-eval report dataset mismatch: {report_path}")
    unit_ids_raw = report.get("unit_ids", [])
    if not isinstance(unit_ids_raw, list) or not unit_ids_raw:
        raise ValueError(f"skip-eval report has no unit_ids: {report_path}")
    unit_ids = [str(value).strip() for value in unit_ids_raw]
    if any(not value for value in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError(f"skip-eval report has invalid unit_ids: {report_path}")
    try:
        num_units = int(report.get("num_layers"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"skip-eval report has invalid num_layers: {report_path}") from exc
    if num_units != len(unit_ids):
        raise ValueError(
            f"skip-eval num_layers={num_units} does not match "
            f"unit_ids={len(unit_ids)}: {report_path}"
        )

    try:
        report_max_span = int(report.get("max_span", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"skip-eval report has invalid max_span: {report_path}") from exc
    modes = report.get("fusion_modes", [])
    fusion = report.get("fusion_by_mode", {})
    if not isinstance(modes, list) or not modes or not isinstance(fusion, dict):
        raise ValueError(f"skip-eval report has no fusion mode: {report_path}")
    mode = str(modes[0])
    mode_payload = fusion.get(mode)
    by_metric = (
        mode_payload.get("gt_degrade_pct_upper_by_metric", {})
        if isinstance(mode_payload, dict)
        else {}
    )
    if not isinstance(by_metric, dict) or not by_metric:
        raise ValueError(f"skip-eval report has no degradation matrices: {report_path}")

    expected_packed = num_units * (num_units + 1) // 2
    required_report_indices: list[tuple[int, int, int]] = []
    packed_index = 0
    for start in range(num_units):
        for end in range(start, num_units):
            span = end - start + 1
            # Unit 0 is structurally protected by the pruning-plan contract.
            if start >= 1 and (report_max_span <= 0 or span <= report_max_span):
                required_report_indices.append((packed_index, start, end))
            packed_index += 1
    for metric, packed in by_metric.items():
        if not isinstance(packed, list) or len(packed) != expected_packed:
            raise ValueError(
                f"skip-eval report metric {metric!r} has invalid packed length: "
                f"{report_path}"
            )
        missing_report = []
        for index, start, end in required_report_indices:
            value = packed[index]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                missing_report.append(f"i{start}_j{end}")
        if missing_report:
            preview = ",".join(missing_report[:8])
            suffix = "..." if len(missing_report) > 8 else ""
            raise ValueError(
                f"incomplete skip-eval report for {backbone}/{dataset}, "
                f"metric={metric}: missing {len(missing_report)} intervals "
                f"({preview}{suffix})"
            )

    summary_budget = min(int(required_summary_max_drop), num_units - 1)
    missing_summaries = [
        f"i{start}_j{end}"
        for start in range(1, num_units)
        for end in range(start, min(num_units - 1, start + summary_budget - 1) + 1)
        if not (pruning_root / f"i{start}_j{end}" / "summary.json").is_file()
    ]
    if missing_summaries:
        preview = ",".join(missing_summaries[:8])
        suffix = "..." if len(missing_summaries) > 8 else ""
        raise FileNotFoundError(
            f"incomplete skip-eval summaries for {backbone}/{dataset} within "
            f"drop budget {summary_budget}: missing {len(missing_summaries)} "
            f"intervals ({preview}{suffix})"
        )
    required_summary_count = sum(
        min(summary_budget, num_units - start) for start in range(1, num_units)
    )
    return unit_ids, {
        "complete_report_intervals": len(required_report_indices),
        "required_summary_intervals": required_summary_count,
        "required_summary_max_drop": summary_budget,
    }


def _pruning_summaries(pruning_root: Path) -> Iterable[Path]:
    if not pruning_root.is_dir():
        return ()
    return sorted(
        path
        for path in pruning_root.glob("*/summary.json")
        if path.parent.name and not path.parent.name.startswith("_")
    )


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_summary(
    source: Path,
    target: Path,
    *,
    source_root: Path,
    final_output_root: Path,
) -> None:
    payload = _load_object(source)
    selector = str(payload.get("selector") or source.parent.name).strip()
    if selector != source.parent.name:
        raise ValueError(
            f"pruning summary selector {selector!r} does not match directory "
            f"{source.parent.name!r}: {source}"
        )
    source_prefix = str(source_root)
    output_prefix = str(final_output_root)
    for key in ("eval_record_path", "summary_path", "scenes_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith(source_prefix):
            payload[key] = output_prefix + value[len(source_prefix) :]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_source(
    source_root: Path,
    *,
    backbones: Sequence[str],
    calib_datasets: Sequence[str],
    eval_datasets: Sequence[str],
    max_drop_budget: int,
) -> dict[str, Any]:
    datasets = list(dict.fromkeys([*calib_datasets, *eval_datasets]))
    validation: dict[str, Any] = {"backbones": {}}
    for backbone in backbones:
        unit_ids: list[str] | None = None
        backbone_payload: dict[str, Any] = {"training_skip_eval": {}}
        for dataset in datasets:
            upper_path = (
                source_root
                / backbone
                / dataset
                / "eval"
                / "upper"
                / "eval_record_v3.json"
            )
            _require_upper_record(upper_path, backbone=backbone, dataset=dataset)
        for dataset in calib_datasets:
            pruning_root = source_root / backbone / dataset / "eval" / "pruning"
            current_ids, skip_validation = _require_complete_skip_eval(
                pruning_root,
                backbone=backbone,
                dataset=dataset,
                required_summary_max_drop=max_drop_budget,
            )
            if unit_ids is None:
                unit_ids = current_ids
            elif current_ids != unit_ids:
                raise ValueError(
                    f"skip-eval unit_ids differ across training datasets for {backbone}"
                )
            backbone_payload["training_skip_eval"][dataset] = {
                "num_units": len(current_ids),
                **skip_validation,
            }
        validation["backbones"][backbone] = backbone_payload
    return validation


def initialize_main_results(
    *,
    source_root: Path,
    output_root: Path,
    backbones: Sequence[str],
    calib_datasets: Sequence[str],
    eval_datasets: Sequence[str],
    max_drop_budget: int = 24,
) -> str:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError(
            f"reference and main result roots must be separate trees: "
            f"source={source_root} output={output_root}"
        )
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"main results path is not a directory: {output_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"reference results root does not exist: {source_root}")

    backbone_names = _split_list(backbones)
    training_names = _split_list(calib_datasets)
    eval_names = _split_list(eval_datasets)
    if not backbone_names or not training_names or not eval_names:
        raise ValueError("backbones, training datasets, and eval datasets must be non-empty")
    if isinstance(max_drop_budget, bool) or int(max_drop_budget) <= 0:
        raise ValueError("max_drop_budget must be a positive integer")
    validation = _validate_source(
        source_root,
        backbones=backbone_names,
        calib_datasets=training_names,
        eval_datasets=eval_names,
        max_drop_budget=int(max_drop_budget),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.init.",
            dir=output_root.parent,
        )
    )
    copied = {"upper_records": 0, "skip_eval_reports": 0, "pruning_summaries": 0, "pruning_eval_records": 0}
    try:
        datasets = list(dict.fromkeys([*training_names, *eval_names]))
        for backbone in backbone_names:
            for dataset in datasets:
                source_eval_root = source_root / backbone / dataset / "eval"
                target_eval_root = staging_root / backbone / dataset / "eval"
                _copy_file(
                    source_eval_root / "upper" / "eval_record_v3.json",
                    target_eval_root / "upper" / "eval_record_v3.json",
                )
                copied["upper_records"] += 1

                source_pruning = source_eval_root / "pruning"
                target_pruning = target_eval_root / "pruning"
                for summary_path in _pruning_summaries(source_pruning):
                    relative = summary_path.relative_to(source_pruning)
                    target_summary = target_pruning / relative
                    _copy_summary(
                        summary_path,
                        target_summary,
                        source_root=source_root,
                        final_output_root=output_root,
                    )
                    copied["pruning_summaries"] += 1
                    source_record = summary_path.parent / "eval_record_v3.json"
                    if source_record.is_file():
                        _copy_file(
                            source_record,
                            target_summary.parent / "eval_record_v3.json",
                        )
                        copied["pruning_eval_records"] += 1

            for dataset in training_names:
                _copy_file(
                    source_root
                    / backbone
                    / dataset
                    / "eval"
                    / "pruning"
                    / "skip_eval_report_v3.json",
                    staging_root
                    / backbone
                    / dataset
                    / "eval"
                    / "pruning"
                    / "skip_eval_report_v3.json",
                )
                copied["skip_eval_reports"] += 1

        manifest = {
            "schema_version": "main_results_init_v1",
            "kind": "main_results_initialization",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "backbones": backbone_names,
            "calib_datasets": training_names,
            "eval_datasets": eval_names,
            "max_drop_budget": int(max_drop_budget),
            "copied": copied,
            "validation": validation,
            "excluded": [
                "prune_plan_all.json",
                "recovery_plan_all.json",
                "recovery/",
                "cache/",
                "eval/recovery/",
                "eval/testtime/",
            ],
        }
        (staging_root / INIT_MANIFEST).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if output_root.exists():
            # Refresh imported artifacts while preserving other main-run results.
            shutil.copytree(staging_root, output_root, dirs_exist_ok=True)
            shutil.rmtree(staging_root)
        else:
            staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(
        f"[main-results:init] copied reusable results from {source_root} "
        f"to {output_root}: {copied}"
    )
    return "copied"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize main_results from reusable analysis eval artifacts."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbones", required=True)
    parser.add_argument("--calib-datasets", required=True)
    parser.add_argument("--eval-datasets", required=True)
    parser.add_argument("--max-drop-budget", type=int, default=24)
    args = parser.parse_args()
    initialize_main_results(
        source_root=args.source_root,
        output_root=args.output_root,
        backbones=_split_list(args.backbones),
        calib_datasets=_split_list(args.calib_datasets),
        eval_datasets=_split_list(args.eval_datasets),
        max_drop_budget=args.max_drop_budget,
    )


if __name__ == "__main__":
    main()

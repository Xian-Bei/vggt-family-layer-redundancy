"""Check scientific export parity and generated asset integrity."""

import hashlib
import json
from pathlib import Path

import pytest

from analysis.vis.final_results_tab import FINAL_TABLE_DATASET_AVERAGE_GROUPS
from analysis.vis.summary_tab import collect_summary_rows


def test_final_web_table_matches_existing_python_summary():
    path = Path("web/data/tables/final_vggt.json")
    if not path.is_file() or not Path("main_results/vggt").is_dir():
        pytest.skip("Local results and generated web data are required")
    _, rows = collect_summary_rows(
        Path("main_results"),
        prune_strategies=("retention", "replaceme"),
        dataset_average_groups=FINAL_TABLE_DATASET_AVERAGE_GROUPS,
        separate_pruning_structures=True,
        winner_only_recovery=True,
    )
    expected = {(row.dataset, row.mode): row.metrics for row in rows if row.backbone == "vggt"}
    actual = json.loads(path.read_text())["rows"]
    assert {(row["dataset"], row["mode"]): row["display_metrics"] for row in actual} == expected


def test_web_manifest_has_no_missing_or_modified_assets():
    root = Path("web/data")
    if not (root / "manifest.json").is_file():
        pytest.skip("Generated web data is required")
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["file"]
        assert path.is_file(), path
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert sum(item["bytes"] for item in manifest["files"]) == manifest["total_bytes"]

"""Merge per-backbone summary tables into one markdown table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from analysis.schema.v3 import OUTPUT_ROOT_DEFAULT

_DATASET_ORDER: Tuple[str, ...] = (
    "scannet_dev50",
    "scannet_test50",
    "7scenes_test",
    "7scenes_train",
    "nrgbd",
)


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (_DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(_DATASET_ORDER), name)


_PERCENT_RE = re.compile(r"\(([+-]?\d+(?:\.\d+)?%)\)")


def _bold_percentages(text: str) -> str:
    return _PERCENT_RE.sub(r"(**\1**)", text)


def _mode_sort_key(mode: str) -> int:
    if mode == "upper":
        return 0
    if mode.startswith("plan"):
        return 1
    return 2


def _extract_table_rows(summary_md: Path) -> Tuple[List[str], List[List[str]]]:
    lines = summary_md.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 2:
        return [], []

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: List[List[str]] = []
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(cells)
    return header, rows


def _collect_rows(root: Path, backbones: Sequence[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    metric_headers: List[str] = []
    merged: List[Dict[str, str]] = []
    for backbone in backbones:
        summary_md = root / backbone / "summary_comparison.md"
        if not summary_md.is_file():
            print(f"[warn] missing summary file: {summary_md}", flush=True)
            continue

        header, rows = _extract_table_rows(summary_md)
        if not header:
            print(f"[warn] empty/invalid table in: {summary_md}", flush=True)
            continue
        if len(header) < 3:
            print(f"[warn] table has too few columns in: {summary_md}", flush=True)
            continue

        current_metrics = header[2:]
        if not metric_headers:
            metric_headers = current_metrics
        elif current_metrics != metric_headers:
            print(
                f"[warn] metric columns mismatch in {summary_md}, skip this backbone",
                flush=True,
            )
            continue

        for row in rows:
            merged.append(
                {
                    "dataset": row[0],
                    "backbone": backbone,
                    "mode": row[1],
                    "metrics": row[2:],
                }
            )
    return metric_headers, merged


def _render_markdown(metric_headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> str:
    headers = ["Dataset", "Backbone", "Mode", *metric_headers]
    n_cols = len(headers)

    out = [
        "# Evaluation summary — all backbones\n\n",
        "Merged table from each backbone's `summary_comparison.md` under `analysis_outputs_v3`.\n\n",
        "| " + " | ".join(headers) + " |\n",
        "| " + " | ".join(["---"] * n_cols) + " |\n",
    ]

    def sort_key(item: Dict[str, str]) -> Tuple:
        return (
            _dataset_sort_key(str(item["dataset"])),
            str(item["backbone"]),
            _mode_sort_key(str(item["mode"])),
            str(item["mode"]),
        )

    last_dataset: str | None = None
    for row in sorted(rows, key=sort_key):
        dataset = str(row["dataset"])
        dataset_cell = dataset if dataset != last_dataset else ""
        last_dataset = dataset
        metrics = [_bold_percentages(m) for m in row["metrics"]]
        cells = [dataset_cell, str(row["backbone"]), str(row["mode"]), *metrics]
        out.append("| " + " | ".join(cells) + " |\n")

    if len(rows) == 0:
        out.append("\n_No summary rows found._\n")
    return "".join(out)


def _discover_backbones(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    names: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "summary_comparison.md").is_file():
            names.append(p.name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="Root directory containing per-backbone folders.",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="",
        help="Optional comma-separated backbone names. Empty means auto-discover.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Default: <root>/summary_comparison_all_backbones.md",
    )
    args = parser.parse_args()

    root = args.root
    if args.backbones.strip():
        backbones = [bb.strip() for bb in args.backbones.split(",") if bb.strip()]
    else:
        backbones = _discover_backbones(root)

    metric_headers, rows = _collect_rows(root, backbones)
    content = _render_markdown(metric_headers, rows)
    out = args.output or (root / "summary_comparison_all_backbones.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

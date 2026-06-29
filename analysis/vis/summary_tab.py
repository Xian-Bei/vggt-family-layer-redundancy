"""Tab: interactive table loaded from per-backbone summary tables."""

from __future__ import annotations

import html
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr

_BACKBONE_ORDER: Tuple[str, ...] = (
    "vggt",
    "pi3",
    "vggt_omega",
)

_DATASET_ORDER: Tuple[str, ...] = (
    "scannet_dev50",
    "scans_dev50",
    "scannet_test50",
    "scans_test50",
    "7scenes_test",
    "7scenes",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "waymo_validation",
    "nuscenes_train",
    "waymo_training",
    "mean",
)

_PCT_RE = re.compile(r"\s*\(\*\*(.+?)\*\*\)")
_NUMBER_RE = re.compile(r"^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_PLAN_MODE_RE = re.compile(r"^(?:prune_span(\d+)|plan_budget(\d+)|plan_t(\d+))(?:_.*)?$")


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (_DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(_DATASET_ORDER), name)


def _mode_sort_key(name: str) -> Tuple[int, int, str]:
    key = str(name or "").strip()
    if key == "upper":
        return (0, 0, key)
    m = _PLAN_MODE_RE.match(key)
    if m:
        return (1, int(next(g for g in m.groups() if g is not None)), key)
    if key.startswith("prune_span") or key.startswith("plan"):
        return (1, 10_000, key)
    return (2, 0, key)


def _metric_key_for_header(header: str) -> Optional[str]:
    key = str(header or "").strip().lower()
    key = key.replace("↓", "").replace("↑", "").replace("(°)", "").strip()
    key = re.sub(r"\s+", " ", key)
    if key.startswith("ape"):
        return "ape"
    if key.startswith("are"):
        return "are"
    if key.startswith("rpe trans"):
        return "rpe_trans"
    if key.startswith("rpe rot"):
        return "rpe_rot"
    if key.startswith("absrel"):
        return "depth_abs_rel"
    if key.startswith("δ1") or key.startswith("delta1"):
        return "depth_delta1"
    if key.startswith("acc"):
        return "acc"
    if key.startswith("comp"):
        return "comp"
    if key == "cd":
        return "cd"
    if key.startswith("nc"):
        return "nc"
    return None


def _metric_higher_is_better(header: str, metric_key: str) -> bool:
    text = str(header or "")
    if "↑" in text:
        return True
    if "↓" in text:
        return False
    return metric_key in {"depth_delta1", "nc"}


def _format_training_metric(value: float, epoch: int) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.2f} (e{int(epoch)})"


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    backbone: str
    mode: str
    metrics: Dict[str, str]


def _parse_markdown_table(path: Path, *, backbone: Optional[str] = None) -> Tuple[List[str], List[SummaryRow]]:
    if not path.is_file():
        return [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 2:
        return [], []

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    if len(header) < 2:
        return [], []

    has_backbone_col = len(header) >= 3 and header[1].strip().lower() == "backbone"
    mode_col = 2 if has_backbone_col else 1
    metric_start_col = mode_col + 1
    if len(header) <= metric_start_col:
        return [], []

    metric_headers = header[metric_start_col:]
    normalized_header = ["Dataset", "Backbone", "Mode", *metric_headers]
    rows: List[SummaryRow] = []
    last_dataset = ""
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        dataset = cells[0] or last_dataset
        if cells[0]:
            last_dataset = cells[0]
        rows.append(
            SummaryRow(
                dataset=str(dataset),
                backbone=str(cells[1] if has_backbone_col else backbone or ""),
                mode=str(cells[mode_col]),
                metrics={
                    metric_headers[i]: cells[metric_start_col + i]
                    for i in range(len(metric_headers))
                },
            )
        )
    return normalized_header, rows


def _summary_paths_for_backbone(backbone_dir: Path) -> List[Path]:
    top_level = backbone_dir / "summary_comparison.md"
    if top_level.is_file():
        return [top_level]
    return sorted(
        (p for p in backbone_dir.glob("*/summary_comparison.md") if p.is_file()),
        key=lambda p: (_dataset_sort_key(p.parent.name), str(p)),
    )


def _load_per_backbone_summaries(output_root: Path) -> Tuple[List[str], List[SummaryRow], List[Path]]:
    root = Path(output_root)
    if not root.is_dir():
        return [], [], []

    metric_headers: List[str] = []
    rows: List[SummaryRow] = []
    source_paths: List[Path] = []
    seen: set[Tuple[str, str, str, Tuple[Tuple[str, str], ...]]] = set()
    for backbone_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _backbone_sort_key(p.name)):
        for summary_path in _summary_paths_for_backbone(backbone_dir):
            header, parsed_rows = _parse_markdown_table(summary_path, backbone=backbone_dir.name)
            if not header or not parsed_rows:
                continue
            current_metrics = header[3:]
            for metric_header in current_metrics:
                if metric_header not in metric_headers:
                    metric_headers.append(metric_header)
            source_paths.append(summary_path)
            for row in parsed_rows:
                if not row.backbone:
                    row = SummaryRow(
                        dataset=row.dataset,
                        backbone=backbone_dir.name,
                        mode=row.mode,
                        metrics=row.metrics,
                    )
                key = (
                    row.dataset,
                    row.backbone,
                    row.mode,
                    tuple(sorted((str(k), str(v)) for k, v in row.metrics.items())),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    header = ["Dataset", "Backbone", "Mode", *metric_headers] if metric_headers else []
    return header, rows, source_paths


def _numeric_cell_value(raw: str) -> Optional[float]:
    """Return the leading numeric value, ignoring any existing percentage suffix."""
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _numeric_cell_parts(raw: str) -> Optional[Tuple[float, str]]:
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value, text[match.end() :].strip()


def _mean_cell_value(values: Sequence[str]) -> str:
    numeric_values: List[float] = []
    suffixes: List[str] = []
    for raw in values:
        parts = _numeric_cell_parts(raw)
        if parts is None:
            continue
        value, suffix = parts
        numeric_values.append(value)
        if suffix:
            suffixes.append(suffix)
    if not numeric_values:
        return "—"
    mean_value = sum(numeric_values) / len(numeric_values)
    suffix = suffixes[0] if suffixes and all(s == suffixes[0] for s in suffixes) else ""
    return f"{mean_value:.2f}{suffix}"


def _add_mean_rows(rows: Sequence[SummaryRow], metric_headers: Sequence[str]) -> List[SummaryRow]:
    base_rows = [row for row in rows if row.dataset != "mean"]
    grouped: Dict[Tuple[str, str], List[SummaryRow]] = {}
    for row in base_rows:
        grouped.setdefault((row.backbone, row.mode), []).append(row)

    mean_rows: List[SummaryRow] = []
    for (backbone, mode), group_rows in sorted(
        grouped.items(),
        key=lambda item: (_backbone_sort_key(item[0][0]), _mode_sort_key(item[0][1])),
    ):
        metrics = {
            metric_header: _mean_cell_value(
                [row.metrics.get(metric_header, "—") for row in group_rows]
            )
            for metric_header in metric_headers
        }
        mean_rows.append(
            SummaryRow(
                dataset="mean",
                backbone=backbone,
                mode=mode,
                metrics=metrics,
            )
        )
    return [*base_rows, *mean_rows]


def _format_base_cell_text(raw: str) -> str:
    text = _PCT_RE.sub("", str(raw or "")).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return html.escape(text)
    try:
        value = float(match.group(1))
    except ValueError:
        return html.escape(text)
    if not math.isfinite(value):
        return html.escape(text)
    suffix = text[match.end() :]
    return html.escape(f"{value:.2f}{suffix}")


def _format_change_badge(
    change: Optional[float],
    *,
    percentage: bool,
    higher_is_better: bool,
) -> str:
    if change is None or not math.isfinite(change):
        return ""
    if percentage:
        label = f"{change:+.2f}%" if change else "0.00%"
    else:
        label = f"{change:+.2f}" if change else "0.00"
    cls = "summary-pct"
    if change:
        improved = change > 0 if higher_is_better else change < 0
        cls += " summary-pct-down" if improved else " summary-pct-up"
    return f'<span class="{cls}">{html.escape(label)}</span>'


def _format_cell(
    raw: str,
    *,
    pct_change: Optional[float] = None,
    abs_change: Optional[float] = None,
    higher_is_better: bool = False,
) -> str:
    text = str(raw or "").strip()
    if not text or text == "—":
        return '<span class="summary-na">—</span>'
    base = _format_base_cell_text(text)
    if abs_change is not None and math.isfinite(abs_change):
        return (
            f"{base}"
            f"{_format_change_badge(abs_change, percentage=False, higher_is_better=higher_is_better)}"
        )
    if pct_change is not None and math.isfinite(pct_change):
        return (
            f"{base}"
            f"{_format_change_badge(pct_change, percentage=True, higher_is_better=higher_is_better)}"
        )
    return base


def _render_summary_html(
    headers: Sequence[str],
    rows: Sequence[SummaryRow],
    *,
    metric_headers: Sequence[str],
    show_percentages: bool = False,
    show_absolute_changes: bool = False,
    reference_rows: Optional[Sequence[SummaryRow]] = None,
) -> str:
    if not rows:
        return (
            '<div class="summary-empty">No rows match the current filters. '
            "Select at least one backbone and one dataset.</div>"
        )

    head_cells = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    dataset_rowspans: Dict[int, int] = {}
    backbone_rowspans: Dict[int, int] = {}
    idx = 0
    while idx < len(rows):
        dataset = rows[idx].dataset
        end = idx + 1
        while end < len(rows) and rows[end].dataset == dataset:
            end += 1
        dataset_rowspans[idx] = end - idx
        bb_idx = idx
        while bb_idx < end:
            backbone = rows[bb_idx].backbone
            bb_end = bb_idx + 1
            while (
                bb_end < end
                and rows[bb_end].dataset == dataset
                and rows[bb_end].backbone == backbone
            ):
                bb_end += 1
            backbone_rowspans[bb_idx] = bb_end - bb_idx
            bb_idx = bb_end
        idx = end

    upper_metrics = {
        (row.dataset, row.backbone): row.metrics
        for row in (reference_rows if reference_rows is not None else rows)
        if row.mode == "upper"
    }

    body_rows: List[str] = []
    for row_idx, row in enumerate(rows):
        dataset_cell = ""
        rowspan = dataset_rowspans.get(row_idx)
        if rowspan is not None:
            dataset_cell = (
                f'<td class="summary-col-dataset" rowspan="{rowspan}">'
                f"{html.escape(row.dataset)}</td>"
            )
        backbone_cell = ""
        bb_rowspan = backbone_rowspans.get(row_idx)
        if bb_rowspan is not None:
            backbone_cell = (
                f'<td class="summary-col-backbone" rowspan="{bb_rowspan}">'
                f"{html.escape(row.backbone)}</td>"
            )
        mode_cls = "summary-mode-upper" if row.mode == "upper" else "summary-mode-plan"
        baseline_metrics = upper_metrics.get((row.dataset, row.backbone), {})
        metric_cells_list: List[str] = []
        for metric_header in metric_headers:
            raw = row.metrics.get(metric_header, "—")
            pct_change: Optional[float] = None
            abs_change: Optional[float] = None
            metric_key = _metric_key_for_header(metric_header)
            higher_is_better = (
                _metric_higher_is_better(metric_header, metric_key)
                if metric_key is not None
                else False
            )
            value = _numeric_cell_value(raw)
            baseline = _numeric_cell_value(baseline_metrics.get(metric_header, "—"))
            if row.mode == "upper":
                pass
            elif show_absolute_changes:
                if value is not None and baseline is not None:
                    abs_change = value - baseline
            elif show_percentages:
                if value is not None and baseline is not None and baseline != 0:
                    pct_change = (value - baseline) / abs(baseline) * 100.0
            metric_cells_list.append(
                f"<td>{_format_cell(raw, pct_change=pct_change, abs_change=abs_change, higher_is_better=higher_is_better)}</td>"
            )
        metric_cells = "".join(metric_cells_list)
        body_rows.append(
            "<tr>"
            f"{dataset_cell}"
            f"{backbone_cell}"
            f'<td class="summary-col-mode {mode_cls}">{html.escape(row.mode)}</td>'
            f"{metric_cells}"
            "</tr>"
        )

    return (
        '<div class="summary-table-wrap">'
        '<table class="summary-table">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _best_training_metrics_for_dataset(
    epochs: Sequence[Dict[str, Any]],
    dataset: str,
    metric_headers: Sequence[str],
) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for header in metric_headers:
        metric_key = _metric_key_for_header(header)
        if not metric_key:
            metrics[str(header)] = "—"
            continue
        higher_is_better = _metric_higher_is_better(header, metric_key)
        best_value: Optional[float] = None
        best_epoch: Optional[int] = None
        for item in epochs:
            if not isinstance(item, dict):
                continue
            raw_epoch = item.get("epoch", 0)
            try:
                epoch = int(raw_epoch)
            except (TypeError, ValueError):
                epoch = 0
            ds_metrics = item.get(dataset)
            if not isinstance(ds_metrics, dict) or metric_key not in ds_metrics:
                continue
            try:
                value = float(ds_metrics[metric_key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if best_value is None:
                best_value = value
                best_epoch = epoch
            elif higher_is_better and value > best_value:
                best_value = value
                best_epoch = epoch
            elif not higher_is_better and value < best_value:
                best_value = value
                best_epoch = epoch
        metrics[str(header)] = (
            _format_training_metric(best_value, int(best_epoch or 0))
            if best_value is not None
            else "—"
        )
    return metrics


def _load_training_epochs_from_csv(path: Path) -> List[Dict[str, Any]]:
    epochs_by_id: Dict[int, Dict[str, Any]] = {}
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("tag", "")).strip() != "train":
                    continue
                dataset = str(row.get("dataset", "")).strip()
                metric = str(row.get("metric", "")).strip()
                if not dataset or not metric:
                    continue
                try:
                    epoch = int(row.get("epoch", 0))
                    value = float(row.get("value", ""))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                epoch_row = epochs_by_id.setdefault(epoch, {"epoch": epoch})
                ds_metrics = epoch_row.setdefault(dataset, {})
                if isinstance(ds_metrics, dict):
                    ds_metrics[metric] = value
    except OSError:
        return []
    return [epochs_by_id[k] for k in sorted(epochs_by_id)]


def _load_training_best_rows(
    *,
    output_root: Path,
    metric_headers: Sequence[str],
) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    root = Path(output_root)
    if not root.is_dir():
        return rows
    for backbone_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _backbone_sort_key(p.name)):
        train_root = backbone_dir / "scannet_dev50" / "train"
        if not train_root.is_dir():
            continue
        for run_dir in sorted((p for p in train_root.iterdir() if p.is_dir()), key=lambda p: _mode_sort_key(p.name)):
            eval_path = run_dir / "eval_metrics_history.csv"
            if not eval_path.is_file():
                continue
            epochs = _load_training_epochs_from_csv(eval_path)
            if not epochs:
                continue
            datasets = sorted(
                {
                    str(k)
                    for item in epochs
                    if isinstance(item, dict)
                    for k, v in item.items()
                    if k != "epoch" and isinstance(v, dict)
                },
                key=_dataset_sort_key,
            )
            for dataset in datasets:
                rows.append(
                    SummaryRow(
                        dataset=dataset,
                        backbone=backbone_dir.name,
                        mode=run_dir.name,
                        metrics=_best_training_metrics_for_dataset(
                            epochs,
                            dataset,
                            metric_headers,
                        ),
                    )
                )
    return rows


def build_summary_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str | None = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    header, all_rows, source_paths = _load_per_backbone_summaries(output_root)
    metric_headers = header[3:] if len(header) > 3 else []
    initial_rows = _add_mean_rows(all_rows, metric_headers)
    backbone_choices = sorted({r.backbone for r in initial_rows}, key=_backbone_sort_key)
    dataset_choices = sorted({r.dataset for r in initial_rows}, key=_dataset_sort_key)
    mode_choices = sorted({r.mode for r in initial_rows}, key=_mode_sort_key)
    training_rows_cache: Optional[List[SummaryRow]] = None

    def _rows_for_state(training_loaded: bool) -> List[SummaryRow]:
        source_rows: List[SummaryRow]
        if training_loaded and training_rows_cache is not None:
            source_rows = [*all_rows, *training_rows_cache]
        else:
            source_rows = list(all_rows)
        return _add_mean_rows(source_rows, metric_headers)

    def _choices_for_rows(rows: Sequence[SummaryRow]) -> Tuple[List[str], List[str], List[str]]:
        return (
            sorted({r.backbone for r in rows}, key=_backbone_sort_key),
            sorted({r.dataset for r in rows}, key=_dataset_sort_key),
            sorted({r.mode for r in rows}, key=_mode_sort_key),
        )

    def _filter_rows(
        rows: Sequence[SummaryRow],
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_modes: Sequence[str],
    ) -> List[SummaryRow]:
        bb_set = {str(x) for x in (selected_backbones or [])}
        ds_set = {str(x) for x in (selected_datasets or [])}
        mode_set = {str(x) for x in (selected_modes or [])}
        if not bb_set or not ds_set or not mode_set:
            return []
        filtered = [
            r
            for r in rows
            if r.backbone in bb_set and r.dataset in ds_set and r.mode in mode_set
        ]
        return sorted(
            filtered,
            key=lambda r: (
                _dataset_sort_key(r.dataset),
                _backbone_sort_key(r.backbone),
                _mode_sort_key(r.mode),
            ),
        )

    def _refresh(
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_modes: Sequence[str],
        training_loaded: bool,
        show_percentages: bool,
        show_absolute_changes: bool,
    ):
        source_rows = _rows_for_state(bool(training_loaded))
        cur_backbone_choices, cur_dataset_choices, cur_mode_choices = _choices_for_rows(source_rows)
        rows = _filter_rows(source_rows, selected_backbones, selected_datasets, selected_modes)
        table_html = _render_summary_html(
            header,
            rows,
            metric_headers=metric_headers,
            show_percentages=bool(show_percentages),
            show_absolute_changes=bool(show_absolute_changes),
            reference_rows=source_rows,
        )
        status = (
            f"{len(rows)} rows · "
            f"{len(selected_backbones or [])}/{len(cur_backbone_choices)} backbones · "
            f"{len(selected_datasets or [])}/{len(cur_dataset_choices)} datasets · "
            f"{len(selected_modes or [])}/{len(cur_mode_choices)} modes"
        )
        if training_loaded:
            status += f" · training best rows={len(training_rows_cache or [])}"
        if not source_paths:
            status = f"No per-backbone summary files found under: {output_root}"
        return table_html, status

    def _toggle_training_rows(
        training_loaded: bool,
        selected_backbones: Sequence[str],
        selected_datasets: Sequence[str],
        selected_modes: Sequence[str],
        show_percentages: bool,
        show_absolute_changes: bool,
    ):
        nonlocal training_rows_cache
        next_loaded = not bool(training_loaded)
        if next_loaded and training_rows_cache is None:
            training_rows_cache = _load_training_best_rows(
                output_root=output_root,
                metric_headers=metric_headers,
            )
        source_rows = _rows_for_state(next_loaded)
        next_backbone_choices, next_dataset_choices, next_mode_choices = _choices_for_rows(source_rows)

        bb_set = {str(x) for x in (selected_backbones or [])}
        ds_set = {str(x) for x in (selected_datasets or [])}
        mode_set = {str(x) for x in (selected_modes or [])}
        if next_loaded:
            bb_values = sorted(bb_set | set(next_backbone_choices), key=_backbone_sort_key)
            ds_values = sorted(ds_set | set(next_dataset_choices), key=_dataset_sort_key)
            mode_values = sorted(mode_set | set(next_mode_choices), key=_mode_sort_key)
        else:
            bb_values = [x for x in next_backbone_choices if x in bb_set]
            ds_values = [x for x in next_dataset_choices if x in ds_set]
            mode_values = [x for x in next_mode_choices if x in mode_set]
            if not bb_values:
                bb_values = list(next_backbone_choices)
            if not ds_values:
                ds_values = list(next_dataset_choices)
            if not mode_values:
                mode_values = list(next_mode_choices)

        rows = _filter_rows(source_rows, bb_values, ds_values, mode_values)
        table_html = _render_summary_html(
            header,
            rows,
            metric_headers=metric_headers,
            show_percentages=bool(show_percentages),
            show_absolute_changes=bool(show_absolute_changes),
            reference_rows=source_rows,
        )
        status = (
            f"{len(rows)} rows · "
            f"{len(bb_values)}/{len(next_backbone_choices)} backbones · "
            f"{len(ds_values)}/{len(next_dataset_choices)} datasets · "
            f"{len(mode_values)}/{len(next_mode_choices)} modes"
        )
        if next_loaded:
            status += f" · training best rows={len(training_rows_cache or [])}"
        return (
            next_loaded,
            gr.update(
                value="Hide training eval best"
                if next_loaded
                else "Load training eval best"
            ),
            gr.update(choices=next_backbone_choices, value=bb_values),
            gr.update(choices=next_dataset_choices, value=ds_values),
            gr.update(choices=next_mode_choices, value=mode_values),
            table_html,
            status,
        )

    with gr.Tab("Eval Summary", id=tab_id) as tab_summary:
        source_label = f"{len(source_paths)} per-backbone summary file(s) under `{output_root}`"
        gr.Markdown(
            "## All-backbones evaluation summary\n"
            f"Source: {source_label}"
        )
        training_loaded_state = gr.State(False)
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                summary_training_btn = gr.Button(
                    value="Load training eval best",
                    variant="secondary",
                )
                summary_percentage_cb = gr.Checkbox(
                    label="Show change vs upper (%)",
                    value=False,
                )
                summary_absolute_cb = gr.Checkbox(
                    label="Show change vs upper (abs)",
                    value=False,
                )
                summary_backbone_cb = gr.CheckboxGroup(
                    label="Backbone",
                    choices=backbone_choices,
                    value=backbone_choices,
                )
                summary_dataset_cb = gr.CheckboxGroup(
                    label="Dataset",
                    choices=dataset_choices,
                    value=dataset_choices,
                )
                summary_mode_cb = gr.CheckboxGroup(
                    label="Plan / mode",
                    choices=mode_choices,
                    value=mode_choices,
                )
                summary_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                summary_html = gr.HTML(
                    value=_render_summary_html(
                        header,
                        _filter_rows(initial_rows, backbone_choices, dataset_choices, mode_choices),
                        metric_headers=metric_headers,
                    ),
                    elem_classes=["summary-panel"],
                )

        summary_inputs = [
            summary_backbone_cb,
            summary_dataset_cb,
            summary_mode_cb,
            training_loaded_state,
            summary_percentage_cb,
            summary_absolute_cb,
        ]
        summary_outputs = [summary_html, summary_status]
        for ctrl in [
            summary_backbone_cb,
            summary_dataset_cb,
            summary_mode_cb,
            summary_percentage_cb,
            summary_absolute_cb,
        ]:
            ctrl.change(
                _refresh,
                inputs=summary_inputs,
                outputs=summary_outputs,
                show_progress="hidden",
            )
        summary_training_btn.click(
            _toggle_training_rows,
            inputs=[
                training_loaded_state,
                summary_backbone_cb,
                summary_dataset_cb,
                summary_mode_cb,
                summary_percentage_cb,
                summary_absolute_cb,
            ],
            outputs=[
                training_loaded_state,
                summary_training_btn,
                summary_backbone_cb,
                summary_dataset_cb,
                summary_mode_cb,
                summary_html,
                summary_status,
            ],
            show_progress="minimal",
        )
        tab_summary.select(
            _refresh,
            inputs=summary_inputs,
            outputs=summary_outputs,
            show_progress="hidden",
        )
        blocks.load(
            _refresh,
            inputs=summary_inputs,
            outputs=summary_outputs,
            show_progress="hidden",
        )

    return {"tab": tab_summary}

"""Aggregate eval_record_v3.json files into a markdown summary."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    is_interval_selector,
    is_plan_group_name,
    load_prune_plan_bundle,
    plan_group_from_name,
    prune_plan_bundle_path,
)
from analysis.prune_plan.eval_plots import (
    RECOVERY_METHOD_ORDER,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    write_recovery_eval_plots,
)
from analysis.schema.v3 import EvalRecordV3, OUTPUT_ROOT_DEFAULT, UNIFIED_METRIC_KEYS

AGG_PARAMS_KEY = "agg_params"

# Paper-style column labels; arrow shows preferred direction.
METRIC_HEADERS: Dict[str, str] = {
    "ape": "APE ↓",
    "are": "ARE (°) ↓",
    "rpe_trans": "RPE trans ↓",
    "rpe_rot": "RPE rot (°) ↓",
    "depth_abs_rel": "AbsRel ↓",
    "depth_delta1": "δ1 ↑",
    "acc": "Acc ↓",
    "comp": "Comp ↓",
    "cd": "CD ↓",
    "nc": "NC ↑",
    "inference_time_ms": "Time (ms) ↓",
    "infer_peak_vram_gb": "VRAM (GB) ↓",
    AGG_PARAMS_KEY: "Block params ↓",
}

_DATASET_ORDER = (
    "scannet_dev50",
    "scannet_test50",
    "7scenes_test",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "waymo_validation",
    "nuscenes_train",
    "waymo_training",
)


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (_DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(_DATASET_ORDER), name)


_PLAN_GROUP_SORT_RE = re.compile(r"^(?:prune_span(\d+)|plan_budget(\d+)|plan_t(\d+))$")
_PRUNE_SPAN_RE = re.compile(r"^prune_span(\d+)$")
_PRUNE_SPAN_PREFIX_RE = re.compile(r"^(prune_span\d+)(?:[_/].*)?$")
_INTERVAL_LABEL_RE = re.compile(r"^i(?P<i>\d+)_j(?P<j>\d+)$")
_GENERIC_DROP_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_k(?P<k>\d+)$")
_RECOVERY_LABEL_RE = re.compile(
    r"^(?P<prune>prune_span\d+|i\d+_j\d+|[A-Za-z][A-Za-z0-9_]*_k\d+)_(?P<recovery>recovery_.+)$"
)


def _lower_label_sort_key(label: str) -> Tuple[int, int, int, str]:
    if label == "lower":
        return (0, -1, 0, label)
    m = _PLAN_GROUP_SORT_RE.match(label)
    if m:
        return (0, int(next(g for g in m.groups() if g is not None)), 0, label)
    span = _span_from_label(label)
    if span is not None:
        subrank = 0 if _label_base(label) == str(label or "").split("+", 1)[0] else 1
        return (0, int(span), subrank, label)
    interval_span = _interval_span_from_label(label)
    if interval_span is not None:
        i, j, span_value = interval_span
        return (0, int(span_value), 0, f"{i:04d}_{j:04d}_{label}")
    generic_drop = _generic_drop_from_label(label)
    if generic_drop is not None:
        return (0, int(generic_drop), 1, label)
    return (2, 0, 0, label)


def _label_base(label: str) -> str:
    base = str(label or "").split("+", 1)[0]
    recovery_match = _RECOVERY_LABEL_RE.match(base)
    if recovery_match:
        return recovery_match.group("prune")
    if base.startswith("model/"):
        base = base.split("/", 1)[1]
    if base.startswith("recovery/"):
        parts = base.split("/")
        base = parts[2] if len(parts) >= 3 else base
    if "/" in base:
        base = base.split("/", 1)[0]
    m = _PRUNE_SPAN_PREFIX_RE.match(base)
    if m:
        base = m.group(1)
    return base


def _span_from_label(label: str) -> Optional[int]:
    m = _PRUNE_SPAN_RE.match(_label_base(label))
    return int(m.group(1)) if m else None


def _interval_span_from_label(label: str) -> Optional[Tuple[int, int, int]]:
    m = _INTERVAL_LABEL_RE.match(_label_base(label))
    if not m:
        return None
    i = int(m.group("i"))
    j = int(m.group("j"))
    return i, j, j - i + 1


def _generic_drop_from_label(label: str) -> Optional[int]:
    m = _GENERIC_DROP_LABEL_RE.match(_label_base(label))
    return int(m.group("k")) if m else None


def _plain_lower_span_label(label: str) -> bool:
    base = str(label or "").split("+", 1)[0]
    return bool(
        _PLAN_GROUP_SORT_RE.match(base)
        or _INTERVAL_LABEL_RE.match(base)
        or _GENERIC_DROP_LABEL_RE.match(base)
    )


def _drop_count_from_lower_label(label: str) -> Optional[int]:
    span = _span_from_label(label)
    if span is not None:
        return int(span)
    interval = _interval_span_from_label(label)
    if interval is not None:
        return int(interval[2])
    return _generic_drop_from_label(label)


def _recovery_label_parts(label: str) -> Optional[Tuple[str, str]]:
    base = str(label or "").split("+", 1)[0]
    m = _RECOVERY_LABEL_RE.match(base)
    if not m:
        return None
    return m.group("prune"), m.group("recovery")


def _recovery_label(prune_plan: str, recovery_plan: str) -> str:
    prune = str(prune_plan or "").strip()
    recovery = str(recovery_plan or "").strip()
    return f"{prune}_{recovery}" if prune and recovery else prune or recovery


def _format_param_count(n: int) -> str:
    v = float(n)
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.2f}K"
    return str(int(v))


def _fmt_value(key: str, value: float) -> str:
    if not math.isfinite(value):
        return "—"
    if key == AGG_PARAMS_KEY:
        return _format_param_count(int(value))
    if key in ("inference_time_ms",):
        return f"{value:.1f}"
    if key in ("infer_peak_vram_gb",):
        return f"{value:.2f}"
    if key in ("depth_delta1", "nc"):
        return f"{value:.4f}"
    return f"{value:.4f}"


def _read_aggregator_param_counts_from_bundle(
    prune_plan_root: Path,
    labels: List[str],
) -> Tuple[Optional[int], Dict[str, int]]:
    """Read span parameter counts stored by the prune-plan builder."""
    prune_plan_root = Path(prune_plan_root)
    if not prune_plan_bundle_path(prune_plan_root).is_file():
        print(
            f"[warn] Missing prune plan bundle for parameter accounting: "
            f"{prune_plan_bundle_path(prune_plan_root)}",
            flush=True,
        )
        return None, {}

    try:
        bundle = load_prune_plan_bundle(prune_plan_root)
    except Exception as exc:
        print(f"[warn] Could not load prune plan bundle for parameter accounting: {exc}", flush=True)
        return None, {}

    param_accounting = bundle.get("param_accounting", {})
    if not isinstance(param_accounting, dict) or not bool(param_accounting.get("available", False)):
        reason = ""
        if isinstance(param_accounting, dict):
            reason = str(param_accounting.get("skip_reason", "") or "")
        print(
            "[warn] Prune plan bundle has no available param_accounting; "
            f"Block params will be omitted{': ' + reason if reason else ''}.",
            flush=True,
        )
        return None, {}

    try:
        upper = int(param_accounting["total_params"])
    except Exception:
        print("[warn] Invalid prune plan param_accounting.total_params; Block params omitted.", flush=True)
        return None, {}

    plan_records = bundle_plan_records(bundle)
    spans = bundle.get("spans", {})
    if not isinstance(spans, dict):
        spans = {}

    lower_by_label: Dict[str, int] = {}
    for label in labels:
        raw: Any = None
        base = _label_base(label)
        if is_interval_selector(base) or _GENERIC_DROP_LABEL_RE.match(base):
            raw = plan_records.get(base)
        if raw is None:
            span = _span_from_label(label)
            if span is not None:
                raw = spans.get(str(int(span)))
        if not isinstance(raw, dict):
            print(f"[warn] Missing plan record in prune plan bundle for label {label!r}.", flush=True)
            continue
        value: Any = raw.get("aggregator_params")
        if value is None:
            print(
                f"[warn] Plan record has no aggregator_params in prune plan bundle "
                f"for label {label!r}.",
                flush=True,
            )
            continue
        try:
            lower_by_label[label] = int(value)
        except Exception:
            print(
                f"[warn] Invalid aggregator_params={value!r} "
                f"and label {label!r}.",
                flush=True,
            )
    return upper, lower_by_label


def _normalize_lower_label(label: str) -> str:
    key = str(label or "").strip()
    if not key or key == "lower":
        return key
    return plan_group_from_name(key)


def _mode_candidate_score(mode: str, rel_run_dir: Path, raw_lower_label: str = "") -> int:
    parts = rel_run_dir.parts
    if not parts:
        return 0
    score = 0
    # Prefer canonical layout: eval/lower/... or eval/upper/...
    if parts[0] == mode:
        score += 100
    # Prefer nested plan runs (e.g. lower/plan_t1) over flat legacy runs.
    if len(parts) > 1:
        score += 10
    # Legacy fallback: lower_vggt / upper_vggt.
    if parts[0].startswith(f"{mode}_"):
        score += 20
    if mode == "lower" and is_plan_group_name(raw_lower_label):
        score += 50
    return score


def _lower_label_from_rel_run_dir(rel_run_dir: Path) -> str:
    parts = rel_run_dir.parts
    if not parts:
        return "lower"
    if parts[0] == "lower":
        if len(parts) >= 2:
            return parts[1]
        return "lower"
    if parts[0].startswith("lower_"):
        return parts[0]
    return parts[0]


def _compression_tag(rec: EvalRecordV3) -> str:
    raw = rec.extra.get("inference_compression", {}) if isinstance(rec.extra, dict) else {}
    if not isinstance(raw, dict):
        return ""
    name = str(raw.get("name", "none")).strip().lower()
    if name in ("", "none"):
        return ""
    if name == "fastvggt":
        ratio = f"{float(raw.get('merge_ratio', 0.9)):g}".replace(".", "p")
        return f"fastvggt_m{int(raw.get('merge_start', 0))}_r{ratio}"
    if name == "quantvggt":
        return f"quantvggt_{str(raw.get('dtype', 'w4a4')).lower()}"
    return name


def _record_label(rec: EvalRecordV3, rel_run_dir: Path) -> tuple[str, str, str]:
    extra = rec.extra if isinstance(rec.extra, dict) else {}
    parts = rel_run_dir.parts
    compression = _compression_tag(rec)
    if parts and parts[0] == "recovery":
        recovery_plan = parts[1] if len(parts) >= 2 else ""
        raw_plan = parts[2] if len(parts) >= 3 else ""
        base = _recovery_label(raw_plan, recovery_plan)
        return "recovery", f"{base}+{compression}" if compression else base, raw_plan

    mode = str(extra.get("eval_mode", "")).strip().lower()
    if parts and parts[0] == "pruning" and mode == "lower":
        raw_plan = parts[1] if len(parts) >= 2 else "lower"
        base = _normalize_lower_label(raw_plan)
        return mode, f"{base}+{compression}" if compression else base, raw_plan
    if mode not in ("upper", "lower", "model"):
        mode = str(rec.model_name).strip().lower().split("_", 1)[0]
    if mode not in ("upper", "lower", "model"):
        mode = parts[0].split("_", 1)[0].lower() if parts else "upper"

    raw_plan = ""
    if mode in ("lower", "model"):
        raw_plan = parts[1] if len(parts) >= 2 else mode
        if mode == "lower":
            base = _normalize_lower_label(raw_plan)
        else:
            recovery_plan = parts[2] if len(parts) >= 3 else ""
            base = _recovery_label(raw_plan, recovery_plan) if recovery_plan else _recovery_label(raw_plan, "model")
    else:
        base = "upper"
    return mode, f"{base}+{compression}" if compression else base, raw_plan


def _collect_by_dataset(root: Path) -> Dict[str, Dict[str, EvalRecordV3]]:
    grouped: Dict[str, Dict[str, EvalRecordV3]] = {}
    picked_scores: Dict[str, Dict[str, int]] = {}
    if not root.is_dir():
        return grouped
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        eval_root = dataset_dir / "eval"
        if not eval_root.is_dir():
            continue
        ds_name = dataset_dir.name
        for rec_path in eval_root.rglob("eval_record_v3.json"):
            run_dir = rec_path.parent
            rel_run_dir = run_dir.relative_to(eval_root)
            if (
                rel_run_dir.parts
                and rel_run_dir.parts[0] == "recovery"
                and (len(rel_run_dir.parts) < 2 or rel_run_dir.parts[1] not in RECOVERY_METHOD_ORDER)
            ):
                continue
            rec = EvalRecordV3.load(rec_path)
            mode, label, raw_plan = _record_label(rec, rel_run_dir)
            score = _mode_candidate_score(mode, rel_run_dir, raw_plan)
            best = picked_scores.setdefault(ds_name, {}).get(label, -1)
            if score >= best:
                picked_scores[ds_name][label] = score
                grouped.setdefault(ds_name, {})[label] = rec
    return grouped


def _filter_to_current_prune_spans(
    grouped: Dict[str, Dict[str, EvalRecordV3]],
    prune_plan_root: Path,
) -> Dict[str, Dict[str, EvalRecordV3]]:
    """Drop stale legacy plan labels when a split prune-plan bundle exists."""
    try:
        if not prune_plan_bundle_path(prune_plan_root).is_file():
            return grouped
        bundle = load_prune_plan_bundle(prune_plan_root)
        valid_selectors = set(bundle_plan_records(bundle).keys())
    except Exception:
        return grouped
    if not valid_selectors:
        return grouped

    filtered: Dict[str, Dict[str, EvalRecordV3]] = {}
    for ds_name, modes in grouped.items():
        keep: Dict[str, EvalRecordV3] = {}
        for label, rec in modes.items():
            base = _label_base(label)
            if label == "upper" or label.startswith("upper+") or base in valid_selectors:
                keep[label] = rec
        if keep:
            filtered[ds_name] = keep
    return filtered


def _write_metric_curve_grid(
    *,
    root: Path,
    ds_name: str,
    upper_rec: EvalRecordV3,
    span_items: List[Tuple[int, str, EvalRecordV3]],
    out_dir: Path,
    title_suffix: str,
) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    metric_grid: List[List[Optional[str]]] = [
        ["ape", "are", "rpe_trans", "rpe_rot"],
        ["acc", "comp", "cd", "nc"],
        ["depth_abs_rel", "depth_delta1", "inference_time_ms", "infer_peak_vram_gb"],
    ]
    n_rows = 3
    n_cols = 4

    xs = [0] + [int(span) for span, _label, _rec in span_items]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.25 * n_rows), squeeze=False)
    for row_idx, row in enumerate(metric_grid):
        for col_idx, key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if key is None:
                ax.axis("off")
                continue
            ys = [float(upper_rec.average.get(key, float("nan")))]
            ys.extend(float(rec.average.get(key, float("nan"))) for _span, _label, rec in span_items)
            has_data = any(math.isfinite(v) for v in ys)
            if not has_data:
                ax.axis("off")
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4)
            ax.set_title(METRIC_HEADERS.get(key, key), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units / prune span")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            ax.set_xticks(xs)
    fig.suptitle(
        f"{root.name} / {ds_name}: metrics from upper(span0) to max prune span{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    safe_ds = str(ds_name).replace("/", "_").replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_ds}_metrics_by_span.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _strategy_items_for_dataset(
    *,
    prune_plan_root: Path,
    modes: Dict[str, EvalRecordV3],
) -> Dict[str, List[Tuple[int, str, str, EvalRecordV3]]]:
    try:
        bundle = load_prune_plan_bundle(prune_plan_root)
    except Exception as exc:
        print(f"[warn] Could not load prune strategy bundle for plots: {exc}", flush=True)
        return {}
    plan_records = bundle_plan_records(bundle)
    strategies = bundle_strategy_records(bundle)
    out: Dict[str, List[Tuple[int, str, str, EvalRecordV3]]] = {}
    for strategy_key, strategy_record in strategies.items():
        display = str(strategy_record.get("display_name", strategy_key))
        points = strategy_record.get("points", {})
        if not isinstance(points, dict):
            continue
        items: List[Tuple[int, str, str, EvalRecordV3]] = []
        for drop_key, interval_key in sorted(
            points.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            if not str(drop_key).isdigit():
                continue
            interval_label = str(interval_key)
            if interval_label not in plan_records:
                print(
                    f"[warn] Strategy {strategy_key!r} references missing plan selector "
                    f"{interval_label!r}; skipping point.",
                    flush=True,
                )
                continue
            rec = modes.get(interval_label)
            if rec is None:
                print(
                    f"[warn] Missing lower eval for strategy {display} drop={drop_key}: "
                    f"lower/{interval_label}",
                    flush=True,
                )
                continue
            items.append((int(drop_key), display, interval_label, rec))
        if items:
            out[display] = items
    return out


def _write_strategy_metric_curve_grid(
    *,
    root: Path,
    ds_name: str,
    upper_rec: EvalRecordV3,
    strategy_items: Dict[str, List[Tuple[int, str, str, EvalRecordV3]]],
    out_dir: Path,
) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    if not strategy_items:
        return None
    metric_grid: List[List[Optional[str]]] = [
        ["ape", "are", "rpe_trans", "rpe_rot"],
        ["acc", "comp", "cd", "nc"],
        ["depth_abs_rel", "depth_delta1", "inference_time_ms", "infer_peak_vram_gb"],
    ]
    all_drops = sorted(
        {
            int(drop)
            for items in strategy_items.values()
            for drop, _display, _interval, _rec in items
        }
    )
    if not all_drops:
        return None
    fig, axes = plt.subplots(3, 4, figsize=(20, 9.75), squeeze=False)
    for row_idx, row in enumerate(metric_grid):
        for col_idx, key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if key is None:
                ax.axis("off")
                continue
            upper_value = float(upper_rec.average.get(key, float("nan")))
            has_data = math.isfinite(upper_value)
            if math.isfinite(upper_value):
                ax.scatter([0], [upper_value], color="#333333", marker="*", s=70, zorder=4)
                ax.axhline(
                    upper_value,
                    linestyle="--",
                    color="#333333",
                    linewidth=1.2,
                    alpha=0.65,
                    label="upper",
                )
            for display, items in strategy_items.items():
                xs: List[int] = []
                ys: List[float] = []
                if math.isfinite(upper_value):
                    xs.append(0)
                    ys.append(upper_value)
                for drop, _display, _interval, rec in sorted(items, key=lambda item: item[0]):
                    value = float(rec.average.get(key, float("nan")))
                    if not math.isfinite(value):
                        continue
                    xs.append(int(drop))
                    ys.append(value)
                if len(xs) <= (1 if math.isfinite(upper_value) else 0):
                    continue
                has_data = True
                ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=display)
            if not has_data:
                ax.axis("off")
                continue
            ax.set_title(METRIC_HEADERS.get(key, key), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            ax.set_xticks([0] + all_drops)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{root.name} / {ds_name}: lower strategies vs upper",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    safe_ds = str(ds_name).replace("/", "_").replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_ds}_lower_strategies_by_drop.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _write_recovery_strategy_comparison_grid(
    *,
    root: Path,
    ds_name: str,
    upper_rec: EvalRecordV3,
    lower_span_items: List[Tuple[int, str, EvalRecordV3]],
    recovery_items_by_plan: Dict[str, List[Tuple[int, str, EvalRecordV3]]],
) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    if not recovery_items_by_plan and not lower_span_items:
        return None

    metric_grid: List[List[Optional[str]]] = [
        ["ape", "are", "rpe_trans", "rpe_rot"],
        ["acc", "comp", "cd", "nc"],
        ["depth_abs_rel", "depth_delta1", "inference_time_ms", "infer_peak_vram_gb"],
    ]
    n_rows = 3
    n_cols = 4
    all_spans = sorted(
        {
            int(span)
            for span, _label, _rec in lower_span_items
        }
        | {
            int(span)
            for recovery_items in recovery_items_by_plan.values()
            for span, _label, _rec in recovery_items
        }
    )
    if not all_spans:
        return None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.25 * n_rows), squeeze=False)
    for row_idx, row in enumerate(metric_grid):
        for col_idx, key in enumerate(row):
            ax = axes[row_idx][col_idx]
            if key is None:
                ax.axis("off")
                continue

            upper_value = float(upper_rec.average.get(key, float("nan")))
            has_data = math.isfinite(upper_value)
            if math.isfinite(upper_value):
                ax.axhline(
                    upper_value,
                    linestyle="--",
                    color="#333333",
                    linewidth=1.4,
                    alpha=0.75,
                    label="upper",
                )
                ax.scatter([0], [upper_value], color="#333333", marker="*", s=70, zorder=4)

            lower_xs: List[int] = []
            lower_ys: List[float] = []
            lower_has_values = False
            if math.isfinite(upper_value):
                lower_xs.append(0)
                lower_ys.append(upper_value)
            for span, _label, rec in sorted(lower_span_items, key=lambda item: item[0]):
                value = float(rec.average.get(key, float("nan")))
                if not math.isfinite(value):
                    continue
                lower_xs.append(int(span))
                lower_ys.append(value)
                lower_has_values = True
            if lower_has_values:
                has_data = True
                ax.plot(
                    lower_xs,
                    lower_ys,
                    marker="s",
                    linestyle=":",
                    color="#666666",
                    linewidth=1.9,
                    markersize=4,
                    label="lower",
                )

            for recovery_plan in sorted(recovery_items_by_plan.keys()):
                recovery_items = sorted(recovery_items_by_plan[recovery_plan], key=lambda item: item[0])
                xs: List[int] = []
                ys: List[float] = []
                plan_has_values = False
                if math.isfinite(upper_value):
                    xs.append(0)
                    ys.append(upper_value)
                for span, _label, rec in recovery_items:
                    value = float(rec.average.get(key, float("nan")))
                    if not math.isfinite(value):
                        continue
                    xs.append(int(span))
                    ys.append(value)
                    plan_has_values = True
                if not plan_has_values:
                    continue
                has_data = True
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    label=recovery_plan.removeprefix("recovery_"),
                )

            if not has_data:
                ax.axis("off")
                continue
            ax.set_title(METRIC_HEADERS.get(key, key), fontsize=11, fontweight="bold")
            ax.set_xlabel("Dropped units / prune span")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0)
            ax.set_xticks([0] + all_spans)
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{root.name} / {ds_name}: lower and recovery strategies vs upper",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    safe_ds = str(ds_name).replace("/", "_").replace(" ", "_")
    out_dir = root / ds_name / "eval" / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_ds}_recovery_strategies_by_span.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_metric_curves_by_span(
    root: Path,
    grouped: Dict[str, Dict[str, EvalRecordV3]],
    prune_plan_root: Path,
) -> List[Path]:
    """For each dataset, plot eval metrics from upper/span0 to max prune span."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401  # type: ignore
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for summary metric curves: {exc}", flush=True)
        return []

    written: List[Path] = []

    for ds_name in sorted(grouped.keys(), key=_dataset_sort_key):
        modes = grouped[ds_name]
        upper_rec = modes.get("upper")
        if upper_rec is None:
            continue
        strategy_items = _strategy_items_for_dataset(
            prune_plan_root=prune_plan_root,
            modes=modes,
        )
        strategy_plot = _write_strategy_metric_curve_grid(
            root=root,
            ds_name=ds_name,
            upper_rec=upper_rec,
            strategy_items=strategy_items,
            out_dir=root / ds_name / "eval" / "lower",
        )
        if strategy_plot is not None:
            written.append(strategy_plot)
        span_items = sorted(
            (
                (span, label, rec)
                for label, rec in modes.items()
                if _plain_lower_span_label(label)
                if (span := _drop_count_from_lower_label(label)) is not None
            ),
            key=lambda item: item[0],
        )
        if not strategy_items and span_items:
            out_path = _write_metric_curve_grid(
                root=root,
                ds_name=ds_name,
                upper_rec=upper_rec,
                span_items=span_items,
                out_dir=root / ds_name / "eval" / "lower",
                title_suffix="",
            )
            if out_path is not None:
                written.append(out_path)

        recovery_items_by_plan: Dict[str, List[Tuple[int, str, EvalRecordV3]]] = {}
        for label, rec in modes.items():
            parts = _recovery_label_parts(label)
            if parts is None:
                continue
            _prune_plan, recovery_plan = parts
            span = _span_from_label(label)
            if span is None:
                continue
            recovery_items_by_plan.setdefault(recovery_plan, []).append((int(span), label, rec))
        for recovery_plan in sorted(recovery_items_by_plan.keys()):
            recovery_items = sorted(recovery_items_by_plan[recovery_plan], key=lambda item: item[0])
            if not recovery_items:
                continue
            out_path = _write_metric_curve_grid(
                root=root,
                ds_name=ds_name,
                upper_rec=upper_rec,
                span_items=recovery_items,
                out_dir=root / ds_name / "eval" / "recovery" / recovery_plan,
                title_suffix=f" ({recovery_plan})",
            )
            if out_path is not None:
                written.append(out_path)
        out_path = _write_recovery_strategy_comparison_grid(
            root=root,
            ds_name=ds_name,
            upper_rec=upper_rec,
            lower_span_items=span_items,
            recovery_items_by_plan=recovery_items_by_plan,
        )
        if out_path is not None:
            written.append(out_path)
    return written


def _table_separator(n_cols: int) -> str:
    return "| " + " | ".join(["---"] * n_cols) + " |\n"


def build_summary_markdown(
    experiment: str,
    grouped: Dict[str, Dict[str, EvalRecordV3]],
    *,
    upper_agg_params: Optional[int] = None,
    lower_agg_params_by_label: Optional[Dict[str, int]] = None,
) -> str:
    metric_keys = list(UNIFIED_METRIC_KEYS) + [AGG_PARAMS_KEY]
    headers = ["Dataset", "Mode"] + [METRIC_HEADERS.get(k, k) for k in metric_keys]
    n_cols = len(headers)

    lines = [
        f"# Evaluation summary — `{experiment}`\n\n",
        "Comparison of **upper** (full model), pruned plans, trained models, and deployment-compressed variants on disk. "
        "Column arrows indicate the preferred direction (↓ lower is better, ↑ higher is better). "
        "**Block params** counts prunable transformer blocks (Pi3 decoder or VGGT frame/global blocks), "
        "excluding embeddings and heads, "
        "matching the `[params]` log in `train_lora_skip_baseline.py`.\n\n",
        "| " + " | ".join(headers) + " |\n",
        _table_separator(n_cols),
    ]

    for ds_name in sorted(grouped.keys(), key=_dataset_sort_key):
        modes = grouped[ds_name]
        upper_rec = modes.get("upper")
        lower_labels = sorted(
            [k for k in modes.keys() if k != "upper"],
            key=_lower_label_sort_key,
        )
        for mode_label in ["upper"] + lower_labels:
            rec = modes.get(mode_label)
            if rec is None:
                continue
            is_baseline = mode_label == "upper"
            uses_upper_topology = is_baseline or mode_label.startswith("upper+")
            row = [ds_name, mode_label]
            for key in UNIFIED_METRIC_KEYS:
                val = float(rec.average.get(key, float("nan")))
                row.append(_fmt_value(key, val))
            if upper_agg_params is not None:
                if uses_upper_topology:
                    row.append(_fmt_value(AGG_PARAMS_KEY, float(upper_agg_params)))
                else:
                    lower_params = None
                    if lower_agg_params_by_label is not None:
                        lower_params = lower_agg_params_by_label.get(mode_label)
                    if lower_params is None:
                        row.append("—")
                    else:
                        row.append(_fmt_value(AGG_PARAMS_KEY, float(lower_params)))
            else:
                row.append("—")
            lines.append("| " + " | ".join(row) + " |\n")

    if len(lines) <= 4:
        lines.append("\n_No eval records found._\n")
    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default="default")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--prune-plan-root",
        type=Path,
        default=Path("analysis/prune_plan"),
        help="Directory containing per-backbone prune plan JSON files.",
    )
    p.add_argument(
        "--skip-param-count",
        action="store_true",
        help="Skip reading stored aggregator block parameter counts from prune_plan_all.json.",
    )
    args = p.parse_args()

    root = OUTPUT_ROOT_DEFAULT / args.experiment
    grouped = _collect_by_dataset(root)
    grouped = _filter_to_current_prune_spans(grouped, Path(args.prune_plan_root))

    upper_agg_params: Optional[int] = None
    lower_agg_params_by_label: Dict[str, int] = {}
    if not args.skip_param_count:
        labels = sorted(
            {
                k for ds in grouped.values() for k in ds.keys()
                if k != "upper" and not k.startswith("upper+")
            },
            key=_lower_label_sort_key,
        )
        upper_agg_params, lower_agg_params_by_label = _read_aggregator_param_counts_from_bundle(
            args.prune_plan_root,
            labels,
        )
        if upper_agg_params is not None:
            print(
                f"[params] {args.experiment}: upper={_format_param_count(upper_agg_params)} "
                f"loaded lower counts for {len(lower_agg_params_by_label)} labels from prune_plan_all.json",
                flush=True,
            )

    content = build_summary_markdown(
        args.experiment,
        grouped,
        upper_agg_params=upper_agg_params,
        lower_agg_params_by_label=lower_agg_params_by_label,
    )

    out = args.output or (root / "summary_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out}")
    pruning_payload = None
    try:
        pruning_payload = collect_pruning_eval_payload(
            experiment_root=root,
            prune_plan_root=Path(args.prune_plan_root),
        )
    except Exception as exc:
        print(f"[warn] Could not collect pruning comparisons: {exc}", flush=True)
    try:
        recovery_payload = collect_recovery_eval_payload(experiment_root=root)
        if pruning_payload is not None:
            recovery_payload = add_recovery_comparison_strategies(
                recovery_payload,
                pruning_payload,
            )
        for plot_path in write_recovery_eval_plots(
            experiment_root=root,
            backbone=str(args.experiment),
            payload=recovery_payload,
        ):
            print(f"Wrote {plot_path}", flush=True)
    except Exception as exc:
        print(f"[warn] Could not write recovery plots: {exc}", flush=True)


if __name__ == "__main__":
    main()

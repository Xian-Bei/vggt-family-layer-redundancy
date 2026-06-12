"""Aggregate eval_record_v3.json files into a markdown summary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

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
    AGG_PARAMS_KEY: "Agg params ↓",
}

_DATASET_ORDER = (
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


def _pct_change(upper: float, lower: float) -> Optional[float]:
    if not (math.isfinite(upper) and math.isfinite(lower)):
        return None
    if upper == 0.0:
        return None
    return (lower - upper) / upper * 100.0


def _fmt_pct(pct: float) -> str:
    sign = "+" if pct > 0 else ""
    return f"({sign}{pct:.1f}%)"


def _fmt_lower_cell(key: str, upper: float, lower: float) -> str:
    base = _fmt_value(key, lower)
    pct = _pct_change(upper, lower)
    if pct is None:
        return base
    return f"{base} {_fmt_pct(pct)}"


def _compute_aggregator_param_counts(
    backbone: str,
    prune_plan_root: Path,
    plan_name: str = "",
) -> Tuple[int, int]:
    """Mirror train_lora_skip_baseline: count aggregator/decoder stack params (excludes heads)."""
    from analysis.backbones.bootstrap import apply_pythonpath
    from analysis.backbones.cache_replay import get_cache_replay_engine
    from analysis.backbones.registry import get_backbone
    from analysis.prune_plan import (
        UnitPrunePlan,
        drop_intervals_from_plan,
        prune_actions_from_plan,
        resolve_named_prune_plan_path,
        resolve_prune_plan_path,
    )
    from analysis.train.model import build_skipped_layer_indices

    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    engine = get_cache_replay_engine(backbone)
    device = torch.device("cpu")
    hf_model = adapter.default_hf_model()

    plan_name = str(plan_name or "").strip()
    if plan_name:
        plan_path = resolve_named_prune_plan_path(prune_plan_root, plan_name)
    else:
        plan_path = resolve_prune_plan_path(prune_plan_root, backbone)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing prune plan for {backbone}: {plan_path}")

    plan = UnitPrunePlan.load(plan_path)
    drop_actions, lora_actions = prune_actions_from_plan(plan)
    frame_int, global_int, _, _ = drop_intervals_from_plan(plan)
    skipped_frame = build_skipped_layer_indices(frame_int)
    skipped_global = build_skipped_layer_indices(global_int)

    _, _, upper_count, _ = engine.build_train_model(
        hf_model=hf_model,
        device=device,
        drop_actions=[],
        lora_actions=[],
        frame_skip_intervals=[],
        global_skip_intervals=[],
        skipped_frame_idxs=[],
        skipped_global_idxs=[],
        prune=False,
        inject_lora=False,
    )

    model_lower, _, _, _removed = engine.build_train_model(
        hf_model=hf_model,
        device=device,
        drop_actions=drop_actions,
        lora_actions=lora_actions,
        frame_skip_intervals=frame_int,
        global_skip_intervals=global_int,
        skipped_frame_idxs=skipped_frame,
        skipped_global_idxs=skipped_global,
        prune=True,
        inject_lora=False,
    )
    lower_count = sum(p.numel() for p in engine.trainable_param_module(model_lower).parameters())
    return int(upper_count), int(lower_count)


def _mode_candidate_score(mode: str, rel_run_dir: Path) -> int:
    parts = rel_run_dir.parts
    if not parts:
        return 0
    score = 0
    # Prefer canonical layout: eval/lower/... or eval/upper/...
    if parts[0] == mode:
        score += 100
    # Prefer nested plan runs (e.g. lower/plan_t1_l1) over flat legacy runs.
    if len(parts) > 1:
        score += 10
    # Legacy fallback: lower_vggt / upper_vggt.
    if parts[0].startswith(f"{mode}_"):
        score += 20
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
            rec = EvalRecordV3.load(rec_path)
            mode = str(rec.model_name).strip().lower()
            rel_run_dir = run_dir.relative_to(eval_root)
            if mode not in ("upper", "lower"):
                mode = run_dir.name.split("_", 1)[0].lower()
            label = mode
            if mode == "lower":
                label = _lower_label_from_rel_run_dir(rel_run_dir)
            score = _mode_candidate_score(mode, rel_run_dir)
            best = picked_scores.setdefault(ds_name, {}).get(label, -1)
            if score >= best:
                picked_scores[ds_name][label] = score
                grouped.setdefault(ds_name, {})[label] = rec
    return grouped


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
        "Comparison of **upper** (full model) vs **pruned plan runs**. "
        "Plan-row values show percent change relative to upper in parentheses. "
        "Column arrows indicate the preferred direction (↓ lower is better, ↑ higher is better). "
        "**Agg params** counts the aggregator stack (decoder for Pi3), excluding heads, "
        "matching the `[params]` log in `train_lora_skip_baseline.py`.\n\n",
        "| " + " | ".join(headers) + " |\n",
        _table_separator(n_cols),
    ]

    for ds_name in sorted(grouped.keys(), key=_dataset_sort_key):
        modes = grouped[ds_name]
        upper_rec = modes.get("upper")
        lower_labels = sorted([k for k in modes.keys() if k != "upper"])
        for mode_label in ["upper"] + lower_labels:
            rec = modes.get(mode_label)
            if rec is None:
                continue
            is_lower = mode_label != "upper"
            row = [ds_name, mode_label]
            for key in UNIFIED_METRIC_KEYS:
                val = float(rec.average.get(key, float("nan")))
                if is_lower and upper_rec is not None:
                    upper_val = float(upper_rec.average.get(key, float("nan")))
                    row.append(_fmt_lower_cell(key, upper_val, val))
                else:
                    row.append(_fmt_value(key, val))
            if upper_agg_params is not None:
                if not is_lower:
                    row.append(_fmt_value(AGG_PARAMS_KEY, float(upper_agg_params)))
                else:
                    lower_params = None
                    if lower_agg_params_by_label is not None:
                        lower_params = lower_agg_params_by_label.get(mode_label)
                    if lower_params is None:
                        row.append("—")
                    else:
                        row.append(_fmt_lower_cell(AGG_PARAMS_KEY, float(upper_agg_params), float(lower_params)))
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
        help="Skip loading models to count aggregator parameters.",
    )
    p.add_argument(
        "--plan-name",
        type=str,
        default="",
        help="Optional prune plan name (e.g. plan_t1_l1) for aggregator param counting.",
    )
    args = p.parse_args()

    root = OUTPUT_ROOT_DEFAULT / args.experiment
    grouped = _collect_by_dataset(root)

    upper_agg_params: Optional[int] = None
    lower_agg_params_by_label: Dict[str, int] = {}
    if not args.skip_param_count:
        labels = sorted({k for ds in grouped.values() for k in ds.keys() if k != "upper"})
        if args.plan_name:
            labels = [str(args.plan_name).strip()]
        for label in labels:
            plan_name_for_count = "" if label == "lower" else label
            try:
                upper_i, lower_i = _compute_aggregator_param_counts(
                    args.experiment,
                    args.prune_plan_root,
                    plan_name_for_count,
                )
                if upper_agg_params is None:
                    upper_agg_params = upper_i
                lower_agg_params_by_label[label] = lower_i
                print(
                    f"[params] {args.experiment}/{label}: upper={_format_param_count(upper_i)} "
                    f"lower={_format_param_count(lower_i)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[warn] Could not compute aggregator params for {args.experiment}/{label}: {exc}",
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


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCANNET_METRIC_KEYS = (
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "depth_abs_rel",
    "depth_delta1",
)

SEVEN_METRIC_KEYS = ("acc", "comp", "nc", "acc_med", "comp_med", "nc_med")


def _fmt_cell(v: Any) -> str:
    try:
        fv = float(v)
    except Exception:
        return "nan"
    if fv != fv:
        return "nan"
    return f"{fv:.6f}"


def _md_metric_table(title: str, mode_col: str, metric_keys: Iterable[str], rows: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    if not rows:
        return []
    metric_keys = list(metric_keys)
    lines = [f"### {title}", ""]
    header = "| " + " | ".join([mode_col] + metric_keys) + " |"
    sep = "| " + " | ".join([":---"] + ["---:"] * len(metric_keys)) + " |"
    lines.extend([header, sep])
    for mode, d in rows:
        vals = [_fmt_cell(d.get(k, float("nan"))) for k in metric_keys]
        lines.append("| " + " | ".join([mode] + vals) + " |")
    lines.append("")
    return lines


def _scan_scannet_blocks(root: Path) -> List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]]:
    """
    Discover ScanNet eval outputs under:
      <root>/<tag>_input_frame_<N>/<model_backend[_ckpt]>/average_metrics.json
    Return list of blocks: (title, rows).
    """
    blocks: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]] = []
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        m = re.match(r"^(?P<tag>.+)_input_frame_(?P<n>[0-9]+)$", p.name)
        if not m:
            continue
        tag = m.group("tag")
        n = int(m.group("n"))
        rows: List[Tuple[str, Dict[str, Any]]] = []
        for model_dir in sorted([d for d in p.iterdir() if d.is_dir()]):
            avg = model_dir / "average_metrics.json"
            if not avg.exists():
                continue
            try:
                payload = json.loads(avg.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            rows.append((model_dir.name, payload))
        if rows:
            blocks.append((f"ScanNet ({tag}, input_frame={n})", rows))
    return blocks


_IDX_7N_LINE = re.compile(
    r"Idx:\s*[^,]+,\s*(?:NumImgs:\s*[^,]+,\s*)?"
    r"Acc:\s*(?P<acc>[^,]+),\s*Comp:\s*(?P<comp>[^,]+),\s*"
    r"NC1:\s*(?P<nc1>[^,]+),\s*NC2:\s*(?P<nc2>[^,]+)\s*-\s*"
    r"Acc_med:\s*(?P<acc_med>[^,]+),\s*Compc_med:\s*(?P<comp_med>[^,]+),\s*"
    r"NC1c_med:\s*(?P<nc1_med>[^,]+),\s*NC2c_med:\s*(?P<nc2_med>[^\s]+)"
)


def _aggregate_7n_from_idx_lines(text: str) -> Dict[str, float]:
    acc: List[float] = []
    comp: List[float] = []
    nc: List[float] = []
    acc_med: List[float] = []
    comp_med: List[float] = []
    nc_med: List[float] = []
    for line in text.splitlines():
        m = _IDX_7N_LINE.search(line)
        if not m:
            continue
        d = m.groupdict()
        acc.append(float(d["acc"]))
        comp.append(float(d["comp"]))
        nc.append((float(d["nc1"]) + float(d["nc2"])) / 2.0)
        acc_med.append(float(d["acc_med"]))
        comp_med.append(float(d["comp_med"]))
        nc_med.append((float(d["nc1_med"]) + float(d["nc2_med"])) / 2.0)
    if not acc:
        return {}
    n = float(len(acc))
    return {
        "acc": sum(acc) / n,
        "comp": sum(comp) / n,
        "nc": sum(nc) / n,
        "acc_med": sum(acc_med) / n,
        "comp_med": sum(comp_med) / n,
        "nc_med": sum(nc_med) / n,
    }


def _parse_7n_logs(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    mean_line = None
    for line in text.splitlines():
        if line.strip().startswith("mean"):
            mean_line = line
            break
    out: Dict[str, float] = {}
    if mean_line is not None:
        pairs = re.findall(r"([a-zA-Z0-9_]+):\s*([0-9.+-eE]+)", mean_line)
        for k, v in pairs:
            try:
                out[k] = float(v)
            except Exception:
                pass
    need = set(SEVEN_METRIC_KEYS)
    if not need.issubset(out.keys()):
        merged = _aggregate_7n_from_idx_lines(text)
        for k in SEVEN_METRIC_KEYS:
            if k in merged:
                out[k] = merged[k]
    return out


def _load_7andn_dataset_metrics(dataset_dir: Path) -> Dict[str, Any]:
    """Prefer the canonical ``average_metrics.json`` (written by eval_7andN.py /
    backfill); fall back to parsing ``logs_all.txt`` for legacy outputs without
    a JSON sibling.
    """
    avg = dataset_dir / "average_metrics.json"
    if avg.exists():
        try:
            return json.loads(avg.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _parse_7n_logs(dataset_dir / "logs_all.txt")


def _scan_7andn_blocks(root: Path) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]]]:
    """
    Discover 7andn outputs under:
      <root>/7andn/<model_backend[_ckpt]>/<kf>/<dataset>/{average_metrics.json,logs_all.txt}
    Returns (7scenes_rows, nrgbd_rows).
    """
    base = root / "7andn"
    if not base.is_dir():
        return [], []
    rows_7: List[Tuple[str, Dict[str, Any]]] = []
    rows_n: List[Tuple[str, Dict[str, Any]]] = []
    for model_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        kf_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        if not kf_dirs:
            continue
        latest_kf = kf_dirs[-1]
        d7 = _load_7andn_dataset_metrics(latest_kf / "7scenes")
        dn = _load_7andn_dataset_metrics(latest_kf / "NRGBD")
        rows_7.append((model_dir.name, d7))
        rows_n.append((model_dir.name, dn))
    return rows_7, rows_n


def write_summary(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    blocks = _scan_scannet_blocks(output_dir)
    rows_7, rows_n = _scan_7andn_blocks(output_dir)

    lines: List[str] = ["# Evaluation summary", ""]
    for title, rows in blocks:
        lines.extend(_md_metric_table(title, "mode", SCANNET_METRIC_KEYS, rows))
    lines.extend(_md_metric_table("7Scenes", "mode", SEVEN_METRIC_KEYS, rows_7))
    lines.extend(_md_metric_table("NRGBD", "mode", SEVEN_METRIC_KEYS, rows_n))
    while lines and lines[-1] == "":
        lines.pop()
    text = "\n".join(lines) + "\n"

    out = output_dir / "summary_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser("Evaluation summary generator")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = write_summary(Path(args.output_dir))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()


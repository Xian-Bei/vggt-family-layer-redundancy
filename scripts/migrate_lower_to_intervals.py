"""Rename lower eval outputs from prune_spanN to physical i_j interval dirs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SPAN_DIR_RE = re.compile(r"^prune_span(?P<span>\d+)$")
OUTPUT_ROOT_DEFAULT = Path("analysis_outputs_v3")


def _interval_selector(i: int, j: int) -> str:
    start = int(i)
    end = int(j)
    if start < 0 or end <= start:
        raise ValueError(f"invalid interval selector bounds: i={start}, j={end}")
    return f"i{start}_j{end}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="Output root to migrate. Defaults to analysis_outputs_v3.",
    )
    p.add_argument("--backbone", type=str, required=True)
    p.add_argument(
        "--commit",
        action="store_true",
        help="Actually rename directories. Default is dry-run only.",
    )
    p.add_argument(
        "--allow-non-primary-output-root",
        action="store_true",
        help=(
            "Allow migrating an output root other than analysis_outputs_v3. "
            "Use only for explicitly requested roots."
        ),
    )
    return p


def _load_span_mapping(plan_path: Path) -> Dict[int, str]:
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid prune plan bundle: {plan_path}")
    spans = raw.get("spans", {})
    if not isinstance(spans, dict):
        raise ValueError(f"Bundle has no spans mapping for migration: {plan_path}")
    out: Dict[int, str] = {}
    for key, record in spans.items():
        if not str(key).isdigit() or not isinstance(record, dict):
            continue
        out[int(key)] = _interval_selector(int(record["i"]), int(record["j"]))
    if not out:
        raise ValueError(f"No span -> interval mappings found in {plan_path}")
    return out


def _planned_moves(backbone_root: Path, mapping: Dict[int, str]) -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []
    for lower_root in sorted(backbone_root.glob("*/eval/lower")):
        if not lower_root.is_dir():
            continue
        for child in sorted(lower_root.iterdir()):
            if not child.is_dir():
                continue
            m = _SPAN_DIR_RE.match(child.name)
            if not m:
                continue
            span = int(m.group("span"))
            target_name = mapping.get(span)
            if not target_name:
                raise FileNotFoundError(f"No interval mapping for {child}")
            moves.append((child, lower_root / target_name))
    return moves


def _validate_moves(moves: List[Tuple[Path, Path]]) -> None:
    for source, target in moves:
        if source == target:
            continue
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing target: {target} (source: {source})"
            )


def run(
    *,
    output_root: Path,
    backbone: str,
    commit: bool,
    allow_non_primary_output_root: bool = False,
) -> List[Tuple[Path, Path]]:
    output_root = Path(output_root)
    if output_root.name != "analysis_outputs_v3" and not bool(allow_non_primary_output_root):
        raise ValueError(
            f"Refusing to migrate non-primary output root {output_root}. "
            "Use analysis_outputs_v3 only, or pass --allow-non-primary-output-root."
        )
    backbone_root = output_root / str(backbone)
    plan_path = backbone_root / "prune_plan_all.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing prune plan bundle: {plan_path}")
    mapping = _load_span_mapping(plan_path)
    moves = _planned_moves(backbone_root, mapping)
    _validate_moves(moves)
    for source, target in moves:
        action = "rename" if commit else "dry-run"
        print(f"[{action}] {source} -> {target}", flush=True)
        if commit:
            source.rename(target)
    print(
        f"{'Renamed' if commit else 'Would rename'} {len(moves)} lower eval directories "
        f"under {backbone_root}",
        flush=True,
    )
    return moves


def main() -> None:
    args = _build_parser().parse_args()
    run(
        output_root=args.output_root,
        backbone=args.backbone,
        commit=bool(args.commit),
        allow_non_primary_output_root=bool(args.allow_non_primary_output_root),
    )


if __name__ == "__main__":
    main()

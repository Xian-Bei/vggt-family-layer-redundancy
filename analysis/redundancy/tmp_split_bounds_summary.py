from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expect JSON object: {path}")
    return payload


def _to_metric_dict(payload: Any, *, split: str, tag: str, src: Path) -> Dict[str, float]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid payload for {split}.{tag}: {src}")
    out: Dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Split legacy bounds_summary.json into upper/lower files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis_outputs/lora_skip_baseline/bounds_summary.json"),
        help="Legacy combined bounds_summary.json path.",
    )
    parser.add_argument(
        "--upper-output",
        type=Path,
        default=None,
        help="Output path for upper-only payload. Default: <input_stem>_upper.json",
    )
    parser.add_argument(
        "--lower-output",
        type=Path,
        default=None,
        help="Output path for lower-only payload. Default: <input_stem>_lower.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    suffix = input_path.suffix if input_path.suffix else ".json"
    upper_output = Path(args.upper_output) if args.upper_output is not None else input_path.with_name(f"{input_path.stem}_upper{suffix}")
    lower_output = Path(args.lower_output) if args.lower_output is not None else input_path.with_name(f"{input_path.stem}_lower{suffix}")

    payload = _read_json(input_path)
    upper: Dict[str, Dict[str, float]] = {}
    lower: Dict[str, Dict[str, float]] = {}
    for split in ("dev", "test"):
        split_payload = payload.get(split, {})
        if split_payload is None:
            split_payload = {}
        if not isinstance(split_payload, dict):
            raise SystemExit(f"Invalid split payload for '{split}': {input_path}")
        upper[split] = _to_metric_dict(split_payload.get("upper"), split=split, tag="upper", src=input_path)
        lower[split] = _to_metric_dict(split_payload.get("lower"), split=split, tag="lower", src=input_path)

    upper_output.parent.mkdir(parents=True, exist_ok=True)
    lower_output.parent.mkdir(parents=True, exist_ok=True)
    upper_output.write_text(json.dumps(upper, indent=2), encoding="utf-8")
    lower_output.write_text(json.dumps(lower, indent=2), encoding="utf-8")
    print(f"Wrote upper bounds: {upper_output}")
    print(f"Wrote lower bounds: {lower_output}")


if __name__ == "__main__":
    main()

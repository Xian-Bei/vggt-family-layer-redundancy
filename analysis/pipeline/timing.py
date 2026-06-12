"""Stage timing collection and summary for v3 pipeline runs."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class StepTiming:
    name: str
    seconds: float
    status: str = "ok"
    detail: str = ""


@dataclass
class StageTiming:
    stage: str
    steps: List[StepTiming] = field(default_factory=list)
    total_seconds: float = 0.0
    status: str = "ok"

    def add(self, name: str, seconds: float, *, status: str = "ok", detail: str = "") -> None:
        self.steps.append(StepTiming(name=name, seconds=float(seconds), status=status, detail=detail))
        self.total_seconds += float(seconds)


@dataclass
class RunTimingReport:
    experiment: str
    backbone: str
    dataset: str = "scannet"
    stages: List[StageTiming] = field(default_factory=list)
    grand_total_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: StageTiming) -> None:
        self.stages.append(stage)
        self.grand_total_seconds += float(stage.total_seconds)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["stages"] = [
            {**asdict(s), "steps": [asdict(x) for x in s.steps]} for s in self.stages
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def timed_step(stage: StageTiming, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    status, detail = "ok", ""
    try:
        yield
    except Exception as e:
        status = "failed"
        detail = str(e)
        raise
    finally:
        stage.add(name, time.perf_counter() - t0, status=status, detail=detail)


def append_global_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pipeline timing summary", ""]
    for r in rows:
        lines.append(f"## {r.get('label', 'run')}")
        lines.append(f"- status: **{r.get('status', '?')}**")
        lines.append(f"- total: **{r.get('grand_total_seconds', 0):.1f}s**")
        for st in r.get("stages", []):
            lines.append(f"  - {st['stage']}: {st['total_seconds']:.1f}s ({st.get('status', 'ok')})")
            for step in st.get("steps", []):
                lines.append(f"    - {step['name']}: {step['seconds']:.1f}s")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

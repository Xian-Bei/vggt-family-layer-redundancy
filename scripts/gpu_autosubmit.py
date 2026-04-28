#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GpuMem:
    index: int
    used_mb: int
    total_mb: int

    @property
    def frac_used(self) -> float:
        if self.total_mb <= 0:
            return 1.0
        return float(self.used_mb) / float(self.total_mb)


def _run_nvidia_smi() -> str:
    # nounits makes parsing robust across locales
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc.output.strip()}") from exc
    return out


def _parse_nvidia_smi_csv(text: str) -> List[GpuMem]:
    gpus: List[GpuMem] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            used = int(float(parts[1]))
            total = int(float(parts[2]))
        except Exception:
            continue
        gpus.append(GpuMem(index=idx, used_mb=used, total_mb=total))
    return sorted(gpus, key=lambda x: x.index)


def _read_commands(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        out.append(s)
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _pop_first_command(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        idx = i
        break
    if idx is None:
        return None
    cmd = lines[idx].strip()
    new_lines = lines[:idx] + lines[idx + 1 :]
    _atomic_write(path, "\n".join(new_lines).rstrip() + ("\n" if new_lines else ""))
    return cmd


def _launch_on_gpu(cmd: str, gpu_index: int, *, log_dir: Path) -> subprocess.Popen[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in cmd)[:80]
    out_path = log_dir / f"gpu{gpu_index}_{ts}_{safe}.out"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Run through bash so the user can put shell pipelines / exports in the command file.
    # We still log the exact original command line.
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"[gpu_autosubmit] cuda_visible_devices={gpu_index}\n")
        f.write(f"[gpu_autosubmit] command={cmd}\n\n")
        f.flush()
        return subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )


def _fmt_gpu(g: GpuMem) -> str:
    pct = 100.0 * g.frac_used
    return f"GPU{g.index}: {g.used_mb}/{g.total_mb} MiB ({pct:.1f}%)"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Monitor GPU memory usage; when a GPU is < threshold, "
            "pop the first command from a list and launch it on that GPU."
        )
    )
    p.add_argument(
        "--commands",
        type=Path,
        required=True,
        help="Text file of commands (one per line). Blank lines and lines starting with # are ignored.",
    )
    p.add_argument("--threshold", type=float, default=0.10, help="Memory-used fraction threshold (default: 0.10)")
    p.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("gpu_autosubmit_logs"),
        help="Directory to write per-job stdout/stderr logs",
    )
    p.add_argument("--once", action="store_true", help="Do a single poll/dispatch attempt then exit")
    p.add_argument("--dry-run", action="store_true", help="Print what would be launched, but do not launch/pop")
    args = p.parse_args()

    if args.threshold <= 0.0 or args.threshold >= 1.0:
        print("[ERROR] --threshold must be in (0, 1)", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("[ERROR] --interval must be > 0", file=sys.stderr)
        return 2

    running: Dict[int, subprocess.Popen[str]] = {}
    last_status_t = 0.0

    while True:
        # Reap finished children.
        for gpu, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                print(f"[DONE] GPU{gpu} exit_code={rc}")
                running.pop(gpu, None)

        try:
            gpus = _parse_nvidia_smi_csv(_run_nvidia_smi())
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

        now = time.time()
        if now - last_status_t >= 30.0:
            print("[STATUS] " + " | ".join(_fmt_gpu(g) for g in gpus))
            last_status_t = now

        # Try dispatch: first available GPU by index.
        dispatched = False
        for g in gpus:
            if g.index in running:
                continue
            if g.frac_used >= float(args.threshold):
                continue

            if args.dry_run:
                cmds = _read_commands(args.commands)
                if not cmds:
                    break
                print(f"[DRY-RUN] would launch on GPU{g.index}: {cmds[0]}")
                dispatched = True
                break

            cmd = _pop_first_command(args.commands)
            if not cmd:
                break
            proc = _launch_on_gpu(cmd, g.index, log_dir=args.log_dir)
            running[g.index] = proc
            print(f"[LAUNCHED] GPU{g.index} pid={proc.pid} cmd={shlex.quote(cmd)}")
            dispatched = True
            break

        if args.once:
            return 0

        # Exit when no work left and nothing running.
        if not running and not _read_commands(args.commands):
            print("[EXIT] no commands left and no running jobs")
            return 0

        if not dispatched:
            time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())


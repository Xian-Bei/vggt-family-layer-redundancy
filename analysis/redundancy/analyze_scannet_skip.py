from __future__ import annotations

import sys
from typing import List

from . import analyze_scannet_cache as _core


def _has_flag(argv: List[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _strip_flag(argv: List[str], flag: str) -> List[str]:
    out: List[str] = []
    for arg in argv:
        if arg == flag:
            continue
        if arg.startswith(f"{flag}="):
            continue
        out.append(arg)
    return out


def main() -> None:
    argv = _strip_flag(list(sys.argv[1:]), "--disable-similarity-analysis")
    injected: List[str] = []
    injected.append("--disable-similarity-analysis")
    if not _has_flag(argv, "--full-report-name"):
        injected.extend(["--full-report-name", "full_report_skip.json"])
    if not _has_flag(argv, "--per-scene-report-suffix"):
        injected.extend(["--per-scene-report-suffix", "_skip.json"])

    sys.argv = [sys.argv[0], *argv, *injected]
    _core.main()


if __name__ == "__main__":
    main()


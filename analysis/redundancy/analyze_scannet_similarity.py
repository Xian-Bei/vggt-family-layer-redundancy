from __future__ import annotations

import sys
from typing import List

from . import analyze_scannet_cache as _core


def _has_flag(argv: List[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main() -> None:
    argv = list(sys.argv[1:])
    injected: List[str] = []
    if not _has_flag(argv, "--disable-skip-range-loss"):
        injected.append("--disable-skip-range-loss")
    if not _has_flag(argv, "--full-report-name"):
        injected.extend(["--full-report-name", "full_report_similarity.json"])
    if not _has_flag(argv, "--per-scene-report-suffix"):
        injected.extend(["--per-scene-report-suffix", "_similarity.json"])

    sys.argv = [sys.argv[0], *argv, *injected]
    _core.main()


if __name__ == "__main__":
    main()


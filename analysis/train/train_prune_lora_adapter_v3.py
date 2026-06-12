"""Deprecated entry: delegates to unified train_prune_lora_v3."""

from __future__ import annotations

import sys

from analysis.train import train_prune_lora_v3


def main() -> None:
    print(
        "[warn] train_prune_lora_adapter_v3 is deprecated; use train_prune_lora_v3",
        file=sys.stderr,
        flush=True,
    )
    train_prune_lora_v3.main()


if __name__ == "__main__":
    main()

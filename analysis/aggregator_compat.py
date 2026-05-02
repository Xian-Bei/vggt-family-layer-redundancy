"""VGGT vs FastVGGT :class:`Aggregator` API differences (single source of truth)."""

from __future__ import annotations

import inspect
from typing import Any


def aggregator_supports_return_layers(aggregator: Any) -> bool:
    """Whether ``aggregator.forward`` accepts ``return_layers`` / ``layer_callback``.

    Main-repo VGGT does; FastVGGT vendor :class:`Aggregator` does not.
    """
    try:
        sig = inspect.signature(aggregator.forward)
    except (TypeError, ValueError):
        return False
    return "return_layers" in sig.parameters

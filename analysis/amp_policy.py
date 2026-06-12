"""
Single-source AMP dtype for analysis code (eval, cache, redundancy, LoRA training).

Rule (no user overrides):
- CUDA + ``torch.cuda.is_bf16_supported()`` → ``torch.bfloat16`` (FastVGGT-style on typical Ampere+).
- CUDA otherwise → ``torch.float16``.
- CPU (or non-CUDA device) → ``torch.float32`` (stable fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_dtype_tag(dtype: torch.dtype) -> str:
    """Manifest / metadata string for cache format versioning."""
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).split(".")[-1]


def dtype_from_replay_amp_tag(tag: str) -> torch.dtype:
    """Map Stage1 manifest ``replay_amp_dtype`` string to ``torch.dtype``."""
    s = str(tag).strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        raise ValueError("Legacy float32 replay caches are not supported; re-run Stage1 (run_cache.sh).")
    raise ValueError(f"Unknown replay_amp_dtype tag: {tag!r}")


def manifest_replay_dtype(cache_dir: Path) -> torch.dtype:
    """Read ``replay_amp_dtype`` from Stage1 ``<cache_dir>/manifest.json`` (required)."""
    import json

    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {path}")
    man = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(man, dict):
        raise ValueError(f"Invalid manifest JSON (expected dict): {path}")
    tag = man.get("replay_amp_dtype")
    if tag is None and "replay_layer_storage_dtype" in man:
        raise ValueError(
            "Cache manifest uses deprecated key replay_layer_storage_dtype; "
            "re-run Stage1 caching to write replay_amp_dtype (bf16/fp16 only)."
        )
    if tag is None:
        raise ValueError(f"Manifest missing replay_amp_dtype: {path}")
    return dtype_from_replay_amp_tag(str(tag))


def autocast_device_dtype(device: torch.device) -> Dict[str, Any]:
    """Kwargs compatible with ``torch.autocast`` / ``torch.cuda.amp.autocast``."""
    amp_dtype = amp_dtype_for_device(device)
    if device.type == "cuda":
        return {"device_type": "cuda", "dtype": amp_dtype, "enabled": True}
    # CPU: autocast with float32 is a no-op for precision; keep disabled to avoid surprises.
    return {"device_type": "cpu", "dtype": torch.float32, "enabled": False}

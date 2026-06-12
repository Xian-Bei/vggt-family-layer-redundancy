"""Vendored Pi3 official training loss helpers.

Source: backbones/Pi3/Pi3-training (official Pi3 training code).
We vendor these here to keep analysis pipeline backbone-specific logic under
analysis/backbones/ and to avoid sys.path/package-name conflicts with the Pi3
inference package.
"""

from .loss import Pi3Loss

__all__ = ["Pi3Loss"]


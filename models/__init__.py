"""Use calibrated LaRe models without running the research pipeline."""

from pathlib import Path

# Pi3's optional CUDA extension also lives under the models namespace.
__path__.append(str(
    Path(__file__).resolve().parents[1]
    / "analysis/backbones/vendor_extensions/pi3_curope/models"
))

from .model import LaRe, load_calibrated

__all__ = ["LaRe", "load_calibrated"]

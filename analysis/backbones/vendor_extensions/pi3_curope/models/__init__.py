"""LaRe-owned compatibility package for Pi3's absolute ``models.curope`` import."""

from pkgutil import extend_path

# Keep the public model API available when Pi3's paths precede the repo root.
__path__ = extend_path(__path__, __name__)

from .model import LaRe, load_calibrated

__all__ = ["LaRe", "load_calibrated"]

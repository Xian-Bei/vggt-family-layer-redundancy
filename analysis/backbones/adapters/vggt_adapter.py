from __future__ import annotations

from pathlib import Path

from analysis.backbones.adapters._vggt_base import VGGTFamilyAdapterMixin
from analysis.backbones.registry import register


class VGGTAdapter(VGGTFamilyAdapterMixin):
    name = "vggt"

    def default_hf_model(self) -> str:
        return "facebook/VGGT-1B"

    def _expected_vendor_root(self) -> Path:
        return Path(__file__).resolve().parents[3] / "backbones" / "vggt"

    def pythonpath_entries(self, repo_root: Path) -> list[Path]:
        return [
            repo_root / "backbones" / "vggt",
            repo_root / "backbones" / "vggt" / "training",
            repo_root,
        ]


register(VGGTAdapter())

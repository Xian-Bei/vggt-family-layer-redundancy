from __future__ import annotations

from pathlib import Path

from analysis.backbones.adapters._vggt_base import VGGTFamilyAdapterMixin
from analysis.backbones.registry import register


class FastVGGTAdapter(VGGTFamilyAdapterMixin):
    name = "fastvggt"

    def default_hf_model(self) -> str:
        return "facebook/VGGT-1B"

    def _expected_vendor_root(self) -> Path:
        return Path(__file__).resolve().parents[3] / "backbones" / "FastVGGT"

    def pythonpath_entries(self, repo_root: Path) -> list[Path]:
        vendor = repo_root / "backbones" / "FastVGGT"
        if not vendor.is_dir():
            raise FileNotFoundError(
                f"FastVGGT backend not found at {vendor}. "
                "Install or clone FastVGGT under backbones/FastVGGT."
            )
        return [
            vendor,
            repo_root / "backbones" / "vggt" / "training",
            repo_root,
        ]


register(FastVGGTAdapter())

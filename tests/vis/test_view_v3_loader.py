from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.vis.view_v3_loader import (
    detect_v3_backbone_root,
    discover_v3_backbones,
)


class V3BackboneDiscoveryTest(unittest.TestCase):
    def test_every_top_level_directory_is_a_visible_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mapanything" / "scannet_dev50" / "eval").mkdir(parents=True)
            (root / "mapanything_v1").mkdir()
            (root / ".staging").mkdir()
            (root / "README.md").write_text("not a backbone", encoding="utf-8")

            self.assertEqual(
                discover_v3_backbones(root),
                ["mapanything", "mapanything_v1"],
            )

    def test_detects_incomplete_backbone_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "mapanything" / "scannet_dev50"
            dataset.mkdir(parents=True)

            self.assertEqual(
                detect_v3_backbone_root(dataset, output_root=root),
                root / "mapanything",
            )


if __name__ == "__main__":
    unittest.main()

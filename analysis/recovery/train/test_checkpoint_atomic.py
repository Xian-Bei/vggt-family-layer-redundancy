from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from analysis.recovery.train.checkpoint_v3 import (
    atomic_publish_latest,
    atomic_save_checkpoint,
)


class AtomicCheckpointTests(unittest.TestCase):
    def test_epoch_and_latest_share_inode_when_hardlinks_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epoch = Path(tmp) / "epoch_1.pt"
            latest = Path(tmp) / "latest.pt"
            atomic_save_checkpoint({"value": torch.tensor([1])}, epoch)
            atomic_publish_latest(epoch, latest)
            self.assertEqual(epoch.stat().st_ino, latest.stat().st_ino)
            self.assertEqual(torch.load(latest, weights_only=False)["value"].item(), 1)

    def test_failed_serialization_preserves_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.pt"
            atomic_save_checkpoint({"value": torch.tensor([1])}, latest)
            with mock.patch("torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_save_checkpoint({"value": torch.tensor([2])}, latest)
            self.assertEqual(torch.load(latest, weights_only=False)["value"].item(), 1)
            self.assertFalse(any(path.name.startswith(".latest.pt.tmp") for path in Path(tmp).iterdir()))

    def test_copy_fallback_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epoch = Path(tmp) / "epoch_1.pt"
            latest = Path(tmp) / "latest.pt"
            atomic_save_checkpoint({"value": 3}, epoch)
            with mock.patch("os.link", side_effect=OSError("unsupported")):
                atomic_publish_latest(epoch, latest)
            self.assertNotEqual(epoch.stat().st_ino, latest.stat().st_ino)
            self.assertEqual(torch.load(latest, weights_only=False)["value"], 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from analysis.results.samples_ablation import summarize, validate_cache, validate_data
from analysis.schema.v3 import CacheManifestV3


class SamplesAblationTest(unittest.TestCase):
    def test_script_cleanup_scope_and_failed_exit(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        for before, after in [('1', '0'), ('0', '1'), ('1', '1'), ('0', '0')]:
            with self.subTest(before=before, after=after), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'ablation'
                selected = root / 'cache/pi3/keep.txt'
                other = root / 'cache/vggt/keep.txt'
                result_file = root / 'samples_025/pi3/result.txt'
                for path in (selected, other, result_file):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('fixture')
                result = subprocess.run(
                    ['bash', 'experiments/ablation_samples.sh'], cwd=repo,
                    env={**os.environ, 'SAMPLE_COUNTS': '25', 'BACKBONES': 'pi3',
                         'ABLATION_OUTPUT_ROOT': str(root), 'DRY_RUN': '0',
                         'CLEAN_CACHE_BEFORE': before, 'CLEAN_CACHE_AFTER': after,
                         'SCANNET_BASE_LIST': str(Path(tmp) / 'missing.txt')},
                    text=True, capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(selected.exists(), before == after == '0')
                self.assertTrue(other.exists())
                self.assertTrue(result_file.exists())

    def test_dry_run_never_cleans_even_with_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ablation'
            marker = root / 'cache/pi3/keep.txt'
            marker.parent.mkdir(parents=True)
            marker.write_text('fixture')
            result = subprocess.run(
                ['bash', 'experiments/ablation_samples.sh'], cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, 'SAMPLE_COUNTS': '25', 'BACKBONES': 'pi3',
                     'ABLATION_OUTPUT_ROOT': str(root), 'DRY_RUN': '1',
                     'REBUILD_CACHE': '1', 'CLEAN_CACHE_BEFORE': '1', 'CLEAN_CACHE_AFTER': '1'},
                text=True, capture_output=True, check=True,
            )
            self.assertTrue(marker.exists())
            self.assertIn('cleanup before', result.stdout)
            self.assertIn('cleanup after', result.stdout)

    def test_script_supports_small_and_mixed_prefixes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for counts in ('25', '25,75,125'):
            result = subprocess.run(
                ['bash', 'experiments/ablation_samples.sh'], cwd=root,
                env={**os.environ, 'SAMPLE_COUNTS': counts, 'DRY_RUN': '1',
                     'BACKBONES': 'pi3', 'CACHE_ONLY': '0', 'REBUILD_CACHE': '0',
                     'BASE_SCENES_PER_DOMAIN': '50'},
                text=True, capture_output=True, check=True,
            )
            self.assertIn('scannet_dev50=25', result.stdout)
            self.assertIn('nuscenes_train=25', result.stdout)
            self.assertIn('samples_025', result.stdout)
            if counts == '25':
                self.assertNotIn('scannet_dev_extra100', result.stdout)
                self.assertIn('cached extra per domain: 0', result.stdout)
            else:
                self.assertIn('scannet_dev_extra100=25', result.stdout)
                self.assertIn('scannet_dev_extra100=75', result.stdout)

    def test_validate_data_checks_ordered_prefixes_and_disjointness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan_base = root / "scan_base.txt"
            scan_test = root / "scan_test.txt"
            scan_base.write_text("scene-base\n", encoding="utf-8")
            scan_test.write_text("scene-test\n", encoding="utf-8")
            scan_extra = root / "scan_extra"
            scan_extra.mkdir()
            (scan_extra / "scene-extra").mkdir()
            (scan_extra / "selected_scenes.json").write_text(
                json.dumps({"scene_ids": ["scene-extra"]}), encoding="utf-8"
            )
            nusc_base = root / "nusc_base.json"
            nusc_base.write_text(
                json.dumps({"splits": {"train": ["scene-base"], "val": ["scene-val"]}}),
                encoding="utf-8",
            )
            nusc_extra = root / "nusc_extra"
            nusc_extra.mkdir()
            (nusc_extra / "scene-extra").mkdir()
            (nusc_extra / "selected_scenes.json").write_text(
                json.dumps({"splits": {"train": ["scene-extra"], "val": []}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                extra_needed=1,
                base_count=1,
                scannet_base_list=scan_base,
                scannet_test_list=scan_test,
                scannet_extra_root=scan_extra,
                nuscenes_base_manifest=nusc_base,
                nuscenes_extra_root=nusc_extra,
            )

            validate_data(args)
            (scan_extra / "selected_scenes.json").write_text(
                json.dumps({"scene_ids": ["scene-base"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "overlaps main splits"):
                validate_data(args)

    def test_validate_cache_accepts_larger_prefix_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache" / "vggt" / "scannet_dev50" / "cache"
            CacheManifestV3(
                backbone="vggt",
                dataset="scannet_dev50",
                scenes=[{"scene_id": "a"}, {"scene_id": "b"}],
            ).save(cache / "cache_manifest_v3.json")

            validate_cache(root / "cache", "vggt", ["scannet_dev50=2"])
            validate_cache(root / "cache", "vggt", ["scannet_dev50=1"])
            with self.assertRaisesRegex(ValueError, "expected at least 3"):
                validate_cache(root / "cache", "vggt", ["scannet_dev50=3"])

    def test_summarize_writes_long_form_metric_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = (
                root
                / "samples_100"
                / "vggt"
                / "scannet_test50"
                / "eval"
                / "recovery"
                / "i1_j2"
                / "recovery_ghost"
            )
            eval_dir.mkdir(parents=True)
            (eval_dir / "eval_record_v3.json").write_text(
                json.dumps({"average": {"cd": 0.5}, "scenes": {"0": {"cd": 0.5}}}),
                encoding="utf-8",
            )
            recovery_dir = (
                root
                / "samples_100"
                / "vggt"
                / "recovery"
                / "i1_j2"
                / "recovery_ghost"
            )
            recovery_dir.mkdir(parents=True)
            (recovery_dir / "recovery_summary.json").write_text(
                json.dumps(
                    {
                        "num_scenes": 200,
                        "scene_limits_by_dataset": {"scannet_dev50": 50},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(summarize(root), 1)
            with (root / "samples_ablation_results.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["sample_count_per_domain"], "100")
            self.assertEqual(rows[0]["fit_num_scenes"], "200")
            self.assertEqual(rows[0]["metric"], "cd")


if __name__ == "__main__":
    unittest.main()

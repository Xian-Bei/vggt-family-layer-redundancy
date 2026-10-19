import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from data.download_waymo_subset import _exclude_scenes, _sample, _scene_id


class ExtraWaymoSelectionTest(unittest.TestCase):
    def test_excludes_all_splits_and_deduplicates_scene_ids(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp) / split for split in ('training', 'validation')]
            for root, scene in zip(roots, ('1', '2')):
                (root / scene).mkdir(parents=True)
            paths = [f'gs://bucket/training/segment-{sid}_{part}.tfrecord'
                     for sid, part in [('1', 'a'), ('2', 'a'), ('3', 'a'), ('3', 'b'), ('4', 'a')]]
            selected = _exclude_scenes(paths, roots)
            self.assertEqual([_scene_id(p) for p in selected], ['3', '4'])
            self.assertEqual(_sample(selected, seed=5, split='training', count=2), selected)

    def test_missing_exclusion_fails_closed(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _exclude_scenes([], [Path(tmp) / 'missing'])

    def test_insufficient_candidates_fails(self):
        with self.assertRaises(RuntimeError):
            _sample([], seed=5, split='training', count=100)


if __name__ == '__main__':
    unittest.main()

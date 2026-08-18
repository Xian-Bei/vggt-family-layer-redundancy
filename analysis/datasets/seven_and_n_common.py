"""Shared 7Scenes / NRGBD scene loading for dataset adaptors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, List, Type

from analysis.datasets.contracts import EvalSampleBatch
from analysis.datasets.frame_selection import cap_7andn_views
from analysis.datasets.scene_id import scene_id_from_eval_views
from analysis.schema.v3 import DatasetEvalProfile

_EVAL_DATASETS: dict[str, str] = {
    "7scenes": "SevenScenes",
    "nrgbd": "NRGBD",
}


def _ensure_eval_data_import_path() -> Path:
    """Prefer ``analysis/eval/data.py`` over vendored ``backbones/*/training/data``."""
    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    eval_s = str(eval_dir)
    sys.path = [
        p
        for p in sys.path
        if p != eval_s and not (p.endswith("/training/data") or p.endswith("\\training\\data"))
    ]
    sys.path.insert(0, eval_s)
    eval_tag = "/analysis/eval/"
    for mod_name in ("data", "base", "dataset_utils"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        mod_file = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        if eval_tag in mod_file or mod_file.endswith("/analysis/eval/data.py"):
            continue
        del sys.modules[mod_name]
    return eval_dir


def _load_7andn_dataset_cls(dataset: str) -> Type[Any]:
    if dataset not in _EVAL_DATASETS:
        raise ValueError(f"Unknown 7andn dataset: {dataset}")
    _ensure_eval_data_import_path()
    mod = __import__("data", fromlist=[_EVAL_DATASETS[dataset]])
    return getattr(mod, _EVAL_DATASETS[dataset])


def count_7andn_scenes(
    *,
    dataset: str,
    data_root: Path,
    prof: DatasetEvalProfile,
    max_scenes: int,
    split: str | None = None,
) -> int:
    dataset_cls = _load_7andn_dataset_cls(dataset)
    res = (518, 392) if prof.size == 518 else prof.size
    ds_kwargs: dict[str, Any] = {
        "ROOT": str(data_root),
        "resolution": res,
        "kf_every": int(prof.kf_every),
    }
    if dataset == "7scenes":
        if not split:
            raise ValueError("7scenes requires split='train' or 'test'")
        ds_kwargs["split"] = split
    ds = dataset_cls(**ds_kwargs)
    n = len(ds)
    return n if max_scenes <= 0 else min(n, max_scenes)


def iter_7andn_scenes(
    *,
    dataset: str,
    data_root: Path,
    prof: DatasetEvalProfile,
    max_scenes: int,
    max_frames: int,
    split: str | None = None,
) -> Iterator[EvalSampleBatch]:
    dataset_cls = _load_7andn_dataset_cls(dataset)
    res = (518, 392) if prof.size == 518 else prof.size
    ds_kwargs: dict[str, Any] = {
        "ROOT": str(data_root),
        "resolution": res,
        "kf_every": int(prof.kf_every),
    }
    if dataset == "7scenes":
        if not split:
            raise ValueError("7scenes requires split='train' or 'test'")
        ds_kwargs["split"] = split
        # Select after kf_every/shuffle but before image/depth decoding.  The
        # post-load cap below remains as a defensive invariant.
        ds_kwargs["eval_max_frames"] = int(max_frames)
    ds = dataset_cls(**ds_kwargs)
    n = len(ds) if max_scenes <= 0 else min(len(ds), max_scenes)
    for i in range(n):
        views = cap_7andn_views(list(ds[i]), max_frames)
        sid = scene_id_from_eval_views(views)
        yield EvalSampleBatch(scene_id=sid, images=views, meta={"index": i, "max_frames": int(max_frames)})


def load_7andn_scene(
    *,
    dataset: str,
    data_root: Path,
    scene_id: str,
    prof: DatasetEvalProfile,
    max_frames: int,
    split: str | None = None,
) -> EvalSampleBatch:
    for batch in iter_7andn_scenes(
        dataset=dataset,
        data_root=data_root,
        prof=prof,
        max_scenes=0,
        max_frames=max_frames,
        split=split,
    ):
        if batch.scene_id == scene_id:
            return batch
    raise FileNotFoundError(scene_id)

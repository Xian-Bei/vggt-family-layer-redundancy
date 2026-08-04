"""Unified schema v3 for LaRe four-stage pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

SCHEMA_VERSION = "v3"
OUTPUT_ROOT_DEFAULT = Path(os.environ.get("OUTPUT_ROOT", "analysis_outputs_v3"))

LayerUnitKind = Literal["frame", "global"]


@dataclass
class LayerUnitSpec:
    unit_id: str
    kind: LayerUnitKind
    index: int
    module_path: str
    droppable: bool = True
    lora_target: bool = True


@dataclass
class GeometryEvalProfile:
    """Shared geometry params for Acc/Comp/CD on one dataset."""

    dist_clip: float = 0.1
    max_points: int = 999_999
    voxel_size: float = 0.05
    icp_max_correspondence: float = 0.1
    use_icp: bool = True
    use_umeyama: bool = False
    conf_thresh: float = 1.0


@dataclass
class DatasetEvalProfile:
    dataset: str
    geometry: GeometryEvalProfile = field(default_factory=GeometryEvalProfile)
    kf_every: int = 10
    size: int = 518


@dataclass
class CacheManifestV3:
    schema_version: str = SCHEMA_VERSION
    backbone: str = ""
    hf_model: str = ""
    dataset: str = ""
    layer_units: List[LayerUnitSpec] = field(default_factory=list)
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    num_heads: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["layer_units"] = [asdict(u) for u in self.layer_units]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CacheManifestV3":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache manifest at {path}: expected schema_version={SCHEMA_VERSION!r}, "
                f"got {raw.get('schema_version')!r}. Old formats are not supported."
            )
        units = [LayerUnitSpec(**u) for u in raw.get("layer_units", [])]
        return cls(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            backbone=str(raw.get("backbone", "")),
            hf_model=str(raw.get("hf_model", "")),
            dataset=str(raw.get("dataset", "")),
            layer_units=units,
            scenes=list(raw.get("scenes", [])),
            num_heads=int(raw.get("num_heads", 0)),
            extra=dict(raw.get("extra", {})),
        )


@dataclass
class PruneAction:
    action: Literal["drop_unit", "lora_unit"]
    unit_id: str
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    target_submodules: Sequence[str] = field(default_factory=lambda: ("qkv", "proj"))


@dataclass
class TrainRecordV3:
    schema_version: str = SCHEMA_VERSION
    backbone: str = ""
    dataset: str = ""
    prune_plan_path: str = ""
    checkpoint_dir: str = ""
    best_checkpoint: str = ""
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainRecordV3":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported train record at {path}")
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


# Unified metric keys (no median variants)
UNIFIED_METRIC_KEYS = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "acc",
    "comp",
    "cd",
    "inference_time_ms",
    "infer_peak_vram_gb",
)


@dataclass
class EvalRecordV3:
    schema_version: str = SCHEMA_VERSION
    backbone: str = ""
    dataset: str = ""
    model_name: str = ""
    dataset_eval_profile: Dict[str, Any] = field(default_factory=dict)
    scenes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    average: Dict[str, float] = field(default_factory=dict)
    vis_refs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvalRecordV3":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported eval record at {path}: expected {SCHEMA_VERSION!r}. "
                "Open results produced by the v3 pipeline only."
            )
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


SEVENSCENES_SPLITS = ("train", "test")
NUSCENES_SPLITS = ("train", "val")
WAYMO_SPLITS = ("training", "validation")
WAYMO_SPLIT_ALIASES = {"train": "training", "val": "validation"}


def dataset_output_tag(dataset: str, split: str | None = None) -> str:
    """Map logical dataset + optional split to on-disk output directory name."""
    if dataset == "7scenes":
        if not split:
            raise ValueError("7scenes requires split='train' or 'test'")
        split_norm = str(split).strip().lower()
        if split_norm not in SEVENSCENES_SPLITS:
            raise ValueError(f"Invalid 7scenes split: {split!r} (expected train or test)")
        return f"7scenes_{split_norm}"
    if dataset == "nuscenes":
        if not split:
            raise ValueError("nuscenes requires split='train' or 'val'")
        split_norm = str(split).strip().lower()
        if split_norm not in NUSCENES_SPLITS:
            raise ValueError(f"Invalid nuscenes split: {split!r} (expected train or val)")
        return f"nuscenes_{split_norm}"
    if dataset == "waymo":
        if not split:
            raise ValueError("waymo requires split='training' or 'validation'")
        split_norm = WAYMO_SPLIT_ALIASES.get(str(split).strip().lower(), str(split).strip().lower())
        if split_norm not in WAYMO_SPLITS:
            raise ValueError(f"Invalid waymo split: {split!r} (expected training or validation)")
        return f"waymo_{split_norm}"
    return dataset


def parse_dataset_plugin_args(dataset_tag: str) -> tuple[str, str | None]:
    """Return ``(dataset_name, split)`` for tags like ``7scenes_test`` or ``nuscenes_val``."""
    if dataset_tag.startswith("7scenes_"):
        split = dataset_tag[len("7scenes_") :]
        if split not in SEVENSCENES_SPLITS:
            raise ValueError(f"Invalid 7scenes split in tag: {dataset_tag!r}")
        return "7scenes", split
    if dataset_tag.startswith("nuscenes_"):
        split = dataset_tag[len("nuscenes_") :]
        if split not in NUSCENES_SPLITS:
            raise ValueError(f"Invalid nuscenes split in tag: {dataset_tag!r}")
        return "nuscenes", split
    if dataset_tag.startswith("waymo_"):
        split = dataset_tag[len("waymo_") :]
        split = WAYMO_SPLIT_ALIASES.get(split, split)
        if split not in WAYMO_SPLITS:
            raise ValueError(f"Invalid waymo split in tag: {dataset_tag!r}")
        return "waymo", split
    return dataset_tag, None


def parse_7scenes_plugin_args(dataset_tag: str) -> tuple[str, str | None]:
    """Backward-compatible alias for older training code."""
    return parse_dataset_plugin_args(dataset_tag)


def stage_dir(experiment: str, dataset: str, stage: str) -> Path:
    return OUTPUT_ROOT_DEFAULT / experiment / dataset / stage


def require_v3_manifest(path: Path) -> CacheManifestV3:
    manifest_path = path / "cache_manifest_v3.json" if path.is_dir() else path
    return CacheManifestV3.load(manifest_path)

# Data Setup Guide (One-Click, FastVGGT-Aligned)

This folder includes reproducible scripts to prepare all datasets used by the evaluation workflows:

- ScanNet (`scans_test50` + `scans_dev50`)
- 7Scenes
- Neural RGB-D (NRGBD)
- nuScenes front-camera subset (requires a local raw nuScenes copy)
- Waymo front-camera subset (optional; requires Waymo terms + Google Cloud auth)

## One-command setup

From repo root:

```bash
bash data/setup_all_eval_data.sh
```

This wrapper runs:

1. `data/setup_scannet.sh`
2. `data/setup_7scenes.sh`
3. `data/setup_nrgbd.sh`
4. `python data/extract_nuscenes.py`
5. `data/setup_waymo.sh` only when `RUN_WAYMO=1`

Waymo is skipped by default because it requires a Google-authenticated bucket
download and roughly 100 GB for the 50+50 raw subset:

```bash
RUN_WAYMO=1 bash data/setup_all_eval_data.sh
```

Waymo-related wrapper variables:

```bash
RUN_WAYMO=1 \
WAYMO_SEED=20260616 \
WAYMO_SCENES_PER_SPLIT=50 \
WAYMO_RAW_ROOT=data/waymo_raw \
WAYMO_TARGET_ROOT=data/waymo \
WAYMO_EXTRACT_CONDA_ENV=waymo_new \
bash data/setup_all_eval_data.sh
```

Use `RUN_WAYMO_DOWNLOAD=0` to reuse existing raw files,
`RUN_WAYMO_EXTRACT=0` to download only, and `WAYMO_FORCE_EXTRACT=1` to
overwrite/re-extract already prepared split directories.

## Dataset-Specific Setup

### ScanNet Dev/Test

ScanNet setup prepares two FastVGGT-aligned splits:

- `data/scans_test50`: fixed test split aligned with FastVGGT-style ScanNet-50 usage
- `data/scans_dev50`: reproducible random dev split, disjoint from `scans_test50`

Run only the ScanNet setup:

```bash
bash data/setup_scannet.sh
```

The script performs four steps:

1. Prepare split lists
   - read or create `data/scannet_test50.txt`
   - generate `data/scannet_dev50.txt` with a fixed seed
   - verify `dev50` and `test50` do not overlap
2. Download raw files for every scene in both splits
   - `<scene>.sens`
   - `<scene>.txt`
   - `<scene>_vh_clean_2.ply`
3. Extract `.sens` files into FastVGGT-compatible folders
   - `color/*.jpg`
   - `depth/*.png`
   - `pose/*.txt`
   - `intrinsic/*.txt`
4. Validate file completeness and split disjointness

Optional environment variables:

```bash
DEV_SEED=20260420 \
DEV_COUNT=50 \
MAX_WORKERS=6 \
RETRY_PER_FILE=5 \
bash data/setup_scannet.sh
```

- `DEV_SEED`: random seed for dev split sampling
- `DEV_COUNT`: number of dev scenes
- `MAX_WORKERS`: parallel download workers
- `RETRY_PER_FILE`: retry count per file

Expected ScanNet outputs:

```text
data/
├── scannet_test50.txt
├── scannet_dev50.txt
├── scans_test50/
│   └── <scene_id>/
│       ├── <scene_id>.sens
│       ├── <scene_id>.txt
│       ├── <scene_id>_vh_clean_2.ply
│       ├── color/0.jpg
│       ├── depth/0.png
│       ├── pose/0.txt
│       └── intrinsic/intrinsic_color.txt
└── scans_dev50/
    └── <scene_id>/...
```

If downloads fail, the script writes:

```text
data/scannet_setup_failures.txt
```

ScanNet validation requires:

- both split lists have the expected counts, typically 50 each;
- `scannet_dev50.txt` and `scannet_test50.txt` have no scene overlap;
- every scene has non-empty raw files and extracted `color`, `depth`, `pose`, and `intrinsic` outputs.

Notes:

- The script is idempotent: existing valid files are skipped.
- Extraction uses `data/export_scannet_sens.py`.
- Downloads use official ScanNet URLs from `kaldir.vc.in.tum.de`; `.sens` files follow the v1 ScanNet path convention.

### 7Scenes

Run only the 7Scenes setup:

```bash
bash data/setup_7scenes.sh
```

This downloads and extracts 7Scenes, then preprocesses depth maps into
`frame-*.depth.proj.png` files compatible with FastVGGT / SimpleRecon-style
evaluation.

By default, projection preprocessing runs for every sequence listed in both
`TrainSplit.txt` and `TestSplit.txt`. For a faster test-only setup:

```bash
ONLY_TEST_PROJ=1 bash data/setup_7scenes.sh
```

### Neural RGB-D

Run only the NRGBD setup:

```bash
bash data/setup_nrgbd.sh
```

This downloads/extracts Neural RGB-D, normalizes the folder layout, and validates
the files required by the evaluation pipeline.

### nuScenes Front-Camera Subset

nuScenes is not downloaded by the setup script. The wrapper expects an existing
raw nuScenes `v1.0-trainval` tree and then extracts a reproducible front-camera
subset.

Run only the nuScenes extraction:

```bash
python data/extract_nuscenes.py \
  --data_root /media/bb/nuscenes \
  --target_dir data/nuscenes \
  --version v1.0-trainval \
  --seed 0 \
  --num_scenes_per_split 50
```

The extractor:

- uses the official nuScenes `train` and `val` scene splits;
- randomly samples 50 train scenes and 50 val scenes with a fixed seed;
- writes the sampled scene list to `data/nuscenes/selected_scenes.json`;
- saves only the front camera `CAM_FRONT`;
- saves world-coordinate LiDAR point clouds and per-frame `cam2world`;
- calls the official nuScenes devkit `map_pointcloud_to_image` projection for
  every keyframe and saves compact sparse camera-Z depth samples;
- rejects frames with too few valid official projections.

Expected nuScenes output:

```text
data/nuscenes/
├── selected_scenes.json
└── <scene_name>/
    ├── image/CAM_FRONT/000.jpg
    ├── cam2world/CAM_FRONT/000.txt
    ├── intrinsic/CAM_FRONT.txt
    ├── lidar/000.bin
    └── depth_points/CAM_FRONT/000.npz
```

Wrapper environment variables:

```bash
NUSCENES_DATA_ROOT=/media/bb/nuscenes \
NUSCENES_TARGET_DIR=data/nuscenes \
NUSCENES_VERSION=v1.0-trainval \
NUSCENES_SEED=0 \
NUSCENES_SCENES_PER_SPLIT=50 \
NUSCENES_EXTRACT_CONDA_ENV=vggt \
DRIVING_DEPTH_MIN=0.1 \
DRIVING_DEPTH_MAX=80.0 \
DRIVING_DEPTH_MIN_SAMPLES=100 \
bash data/setup_all_eval_data.sh
```

By default the top-level wrapper runs the nuScenes extractor in the `vggt`
conda environment. Set `NUSCENES_EXTRACT_PYTHON=/path/to/python` to use an
explicit interpreter with `nuscenes-devkit` installed.

Set `RUN_NUSCENES=0` to skip this step. If `NUSCENES_DATA_ROOT` does not exist,
the wrapper prints a skip message and continues with the other datasets.

### Waymo Front-Camera Subset

For an independent pool of **100 additional training scenes**, run from the
`vggt` environment:

```bash
bash data/setup_waymo_train_extra100.sh
```

This saves raw TFRecords to `/media/bb/waymo_raw_extra100` and processed
front-camera data to `data/waymo_train_extra100/training`. It excludes all scene
IDs in the existing `data/waymo/training` and `data/waymo/validation` directories,
plus `/media/cc/TALO/Data/waymo` when present. Selection uses seed `20260916`,
deduplicates the scene IDs used by the extractor, and refuses a changed selection
on resume. The selection is recorded under the raw root's `manifests` directory
before downloading, and copied to `selected_scenes.json` in the output root only
after extraction and sparse-depth validation succeed. Extraction uses the
existing `waymo_new` environment. This pool is not automatically added to any
calibration experiment.

Waymo raw data is downloaded from the official Google Cloud bucket after
accepting the Waymo Open Dataset terms and running `gcloud auth login`.

Run the complete Waymo setup only:

```bash
bash data/setup_waymo.sh
```

Important environment details:

- `data/download_waymo_subset.py` is pure Python and can run with the default
  project Python.
- `data/extract_waymo.py` needs the official Waymo devkit. On this machine,
  `conda run -n waymo_new python ...` works; the `waymo_extract` env has a
  broken namespace install and cannot import `dataset_pb2`.
- Python 3.12 cannot directly install the current Waymo devkit dependency set
  because `jaxlib==0.4.13` has no Python 3.12 wheel. Use a Python 3.10 env.

One reproducible environment recipe is:

```bash
conda create -n waymo_new python=3.10 -y
conda run -n waymo_new python -m pip install \
  tensorflow==2.11.0 \
  waymo-open-dataset-tf-2-11-0==1.6.1

conda run -n waymo_new python - <<'PY'
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
import tensorflow as tf
print("waymo devkit ok", tf.__version__)
PY
```

If you prefer not to use conda in the setup script, point directly to a Python
binary that passes the import test:

```bash
WAYMO_EXTRACT_PYTHON=/path/to/python bash data/setup_waymo.sh
```

Useful knobs:

```bash
WAYMO_SEED=20260616 \
WAYMO_SCENES_PER_SPLIT=50 \
WAYMO_RAW_ROOT=data/waymo_raw \
WAYMO_TARGET_ROOT=data/waymo \
WAYMO_EXTRACT_CONDA_ENV=waymo_new \
bash data/setup_waymo.sh
```

To download only, set `RUN_WAYMO_EXTRACT=0`. To extract existing raw files
only, set `RUN_WAYMO_DOWNLOAD=0`.

The downloader writes deterministic manifests under
`data/waymo_raw/manifests/`, including `<split>_selected.txt`.
The setup script checks for an active `gcloud` account before downloading:

```bash
gcloud auth login
gcloud auth list
```

Expected Waymo output:

```text
data/waymo/
├── training/<scene_id>/
│   ├── image/FRONT/000.jpg
│   ├── cam2world/FRONT/000.txt
│   ├── intrinsic/FRONT.txt
│   ├── lidar/000.bin
│   └── depth_points/FRONT/000.npz
└── validation/<scene_id>/...
```

Smoke-test extraction for one scene can be done by placing or symlinking one
`.tfrecord` under a temporary raw directory, then running:

```bash
conda run -n waymo_new python data/extract_waymo.py \
  --data_root data/waymo_smoke_raw \
  --target_dir data/waymo_smoke_extract
```

Waymo sparse depth is built through the official
`convert_range_image_to_point_cloud` helper and its paired camera-projection
records. The associated vehicle-frame points are transformed into the OpenCV
`FRONT` camera frame, and their camera Z coordinate is stored. The raw Waymo
range-image value is a LiDAR ray distance and is intentionally **not** used as
depth, because VGGT predicts projective camera-Z depth.

## Driving sparse-depth format and validation

nuScenes and Waymo use the same compressed per-frame NPZ schema:

```text
u, v                 float32 projected coordinates in the original RGB image
depth                float32 OpenCV camera Z in metres
image_width/height   original camera image dimensions
depth_type           "camera_z"
source               official devkit projection path
format_version       sparse-depth schema version
```

This is deliberately sparse: no full-resolution mostly-zero depth image is
written. During evaluation, `(u, v)` is transformed by the exact VGGT
resize/crop geometry and rasterized with a nearest-point z-buffer.

Both the nuScenes step in `setup_all_eval_data.sh` and each Waymo split in
`setup_waymo.sh` run validation automatically. Manual validation with visual
overlays is available as:

```bash
python data/validate_sparse_depth.py data/nuscenes \
  --min-samples 100 \
  --preview-dir /tmp/nuscenes_depth_preview

python data/validate_sparse_depth.py data/waymo/validation \
  --min-samples 100 \
  --preview-dir /tmp/waymo_depth_preview
```

The validator checks the schema, finite values, image bounds, RGB pairing,
image dimensions and minimum samples. It writes
`sparse_depth_validation.json`; preview points are colored by depth and should
follow road, vehicle and building surfaces.

Existing Waymo scene folders without `depth_points/FRONT/*.npz` are considered
incomplete by `setup_waymo.sh` and are re-extracted automatically. Existing
nuScenes data must likewise be re-run through `extract_nuscenes.py` before
driving-depth evaluation.

## Keep only final usable data

To automatically remove download caches after setup:

```bash
CLEAN_DOWNLOADS=1 bash data/setup_all_eval_data.sh
```

Manual cleanup command:

```bash
rm -rf data/downloads_7scenes data/downloads_nrgbd
```

## Final dataset layout (after successful setup)

```text
data/
├── scans_test50/
├── scans_dev50/
├── 7scenes/
│   ├── chess/
│   ├── fire/
│   ├── heads/
│   ├── office/
│   ├── pumpkin/
│   ├── redkitchen/
│   └── stairs/
├── neural_rgbd_data/
│   ├── breakfast_room/
│   ├── complete_kitchen/
│   ├── green_room/
│   ├── grey_white_room/
│   ├── kitchen/
│   ├── morning_apartment/
│   ├── staircase/
│   ├── thin_geometry/
│   └── whiteroom/
├── nuscenes/
    ├── selected_scenes.json
    └── <scene_name>/
        ├── image/CAM_FRONT/
        ├── cam2world/CAM_FRONT/
        ├── intrinsic/CAM_FRONT.txt
        ├── lidar/
        └── depth_points/CAM_FRONT/
├── waymo/
│   ├── training/<scene_id>/
│   │   ├── image/FRONT/
│   │   ├── cam2world/FRONT/
│   │   ├── intrinsic/FRONT.txt
│   │   ├── lidar/
│   │   └── depth_points/FRONT/
│   └── validation/<scene_id>/
```

## Notes on reproducibility

- `setup_scannet.sh` generates `scannet_dev50.txt` with a fixed seed and enforces no overlap with `scannet_test50.txt`.
- `setup_7scenes.sh` includes depth projection preprocessing to produce `frame-*.depth.proj.png` (FastVGGT / SimpleRecon–compatible). **By default** it runs projection for **every sequence in both `TrainSplit.txt` and `TestSplit.txt`** under each scene. For a faster setup limited to test sequences only, run with `ONLY_TEST_PROJ=1`.
- `setup_nrgbd.sh` normalizes the extracted folder layout and verifies required files.
- `extract_nuscenes.py` samples train/val scenes with `NUSCENES_SEED`, records the selected names in `selected_scenes.json`, and records the official sparse-depth format/source metadata.
- `setup_waymo.sh` downloads/extracts a fixed Waymo `training`/`validation` subset when `RUN_WAYMO=1`.
- `download_waymo_subset.py` samples Waymo `training`/`validation` tfrecords with a fixed seed and records manifests. With seed `20260616` and count `50`, the official bucket currently exposes 798 training and 202 validation scenes before sampling.

## Additional calibration pools for the sample-count ablation

The sample-count ablation keeps all main-experiment datasets unchanged and writes
two independent, ordered pools:

```text
data/scans_dev_extra100/
└── selected_scenes.json
data/nuscenes_train_extra100/
└── selected_scenes.json
```

Prepare them from the `vggt` environment:

```bash
conda activate vggt
bash data/setup_scannet_dev_extra100.sh

NUSCENES_RAW_ROOT=/media/bb/nuscenes \
  bash data/setup_nuscenes_train_extra100.sh
```

The ScanNet wrapper excludes the exact scene IDs in both `scannet_dev50.txt` and
`scannet_test50.txt`. The nuScenes wrapper excludes every train/val scene in the
main `data/nuscenes/selected_scenes.json`. Both fail if the requested 100 unique
extra scenes are unavailable, preserve a fixed seeded-random order, and skip
already complete extraction output when rerun.

Run the calibration ablation with:

```bash
# Inspect all cache and fit/eval commands without touching data or outputs.
DRY_RUN=1 BACKBONES=vggt bash ablation_samples.sh

# N is scenes per domain, not a cross-dataset total.
SAMPLE_COUNTS=50,100,150 bash ablation_samples.sh
```

For `N=150`, fitting uses 50 main + 100 extra ScanNet scenes and 50 main + 100
extra nuScenes scenes. The script caches the largest requested prefix once under
`analysis_outputs_v3_samples_ablation/cache`, reuses it for every N, and writes
each result under `samples_050`, `samples_100`, or `samples_150`. It reads plans
and evaluation bounds from `analysis_outputs_v3` without modifying that tree.

## Script index

- `setup_all_eval_data.sh`: top-level one-click entry
- `setup_scannet.sh`: ScanNet test/dev setup + extraction + validation
- `setup_scannet_dev_extra100.sh`: disjoint ordered ScanNet calibration pool
- `setup_7scenes.sh`: 7Scenes download + extraction + depth projection + validation
- `setup_nrgbd.sh`: NRGBD download + extraction + layout normalization + validation
- `setup_waymo.sh`: Waymo raw subset download + extraction wrapper
- `preprocess_7scenes_depth_proj.py`: 7Scenes `depth.png -> depth.proj.png`
- `extract_nuscenes.py`: nuScenes front-camera subset extraction
- `setup_nuscenes_train_extra100.sh`: disjoint ordered nuScenes train calibration pool
- `download_waymo_subset.py`: Waymo raw subset download from GCS
- `extract_waymo.py`: Waymo front-camera/lidar extraction
- `sparse_depth.py`: shared compressed sparse camera-Z schema and rasterizer
- `validate_sparse_depth.py`: schema/RGB/coverage validation and projection overlays

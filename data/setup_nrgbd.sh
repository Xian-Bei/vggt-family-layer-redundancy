#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/neural_rgbd_data}"
TMP_DIR="${TMP_DIR:-${DATA_DIR}/downloads_nrgbd}"
ZIP_URL="${ZIP_URL:-http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip}"
export OUT_DIR

mkdir -p "${OUT_DIR}" "${TMP_DIR}"
ZIP_PATH="${TMP_DIR}/neural_rgbd_data.zip"

echo "== NRGBD setup =="
echo "OUT_DIR=${OUT_DIR}"
echo "ZIP_URL=${ZIP_URL}"
echo

if [[ ! -s "${ZIP_PATH}" ]]; then
  echo "[download] neural_rgbd_data.zip"
  wget -c -q --show-progress "${ZIP_URL}" -O "${ZIP_PATH}"
else
  echo "[skip] ${ZIP_PATH} already exists"
fi

if ! unzip -tqq "${ZIP_PATH}" >/dev/null 2>&1; then
  echo "[warn] existing zip is incomplete/corrupted, redownloading..."
  rm -f "${ZIP_PATH}"
  wget -c -q --show-progress "${ZIP_URL}" -O "${ZIP_PATH}"
fi

if [[ ! -f "${OUT_DIR}/.extract_done" ]]; then
  echo "[unzip] neural_rgbd_data.zip -> ${OUT_DIR}"
  unzip -q "${ZIP_PATH}" -d "${OUT_DIR}"
  touch "${OUT_DIR}/.extract_done"
else
  echo "[skip] extraction already marked done"
fi

echo
echo "== normalize layout for FastVGGT eval_7andN =="
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
if not root.exists():
    raise SystemExit(f"missing root: {root}")

# Some archives extract as OUT_DIR/neural_rgbd_data/<scene>/*
# Flatten one level if needed.
inner = root / "neural_rgbd_data"
if inner.is_dir():
    for item in inner.iterdir():
        target = root / item.name
        if not target.exists():
            item.rename(target)
    try:
        inner.rmdir()
    except OSError:
        pass

# FastVGGT expects depth/depth{i}.png
# If only depth_filtered exists, create a symlink/copy fallback.
scenes = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
for sd in scenes:
    depth = sd / "depth"
    depth_filtered = sd / "depth_filtered"
    if (not depth.exists()) and depth_filtered.exists():
        try:
            depth.symlink_to(depth_filtered.name)
        except Exception:
            depth.mkdir(parents=True, exist_ok=True)
            for f in depth_filtered.glob("depth*.png"):
                target = depth / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())

print(f"scene_count={len(scenes)}")
PY

echo
echo "== quick validation =="
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
scenes = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
missing = []
for sd in scenes:
    if not (sd / "images").is_dir():
        missing.append(str(sd / "images"))
    if not (sd / "depth").is_dir():
        missing.append(str(sd / "depth"))
    if not (sd / "poses.txt").is_file():
        missing.append(str(sd / "poses.txt"))
    if (sd / "images").is_dir() and not any((sd / "images").glob("img*.png")):
        missing.append(f"{sd}/images/img*.png")
    if (sd / "depth").is_dir() and not any((sd / "depth").glob("depth*.png")):
        missing.append(f"{sd}/depth/depth*.png")

print(f"root={root}")
print(f"scene_count={len(scenes)}")
print(f"missing_count={len(missing)}")
if missing:
    for x in missing[:20]:
        print(" -", x)
    raise SystemExit(2)
print("validation_ok=1")
PY

echo "Done NRGBD setup."

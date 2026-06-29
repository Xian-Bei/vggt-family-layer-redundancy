#!/usr/bin/env bash
# 7Scenes: download / unzip + depth registration for FastVGGT / SimpleRecon-style eval.
#
# By default we run depth projection for every sequence listed in BOTH TrainSplit.txt
# and TestSplit.txt under each scene (full 7Scenes coverage for train or test dataloaders).
# For a faster one-off setup when you only need TestSplit sequences, set ONLY_TEST_PROJ=1.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/7scenes}"
TMP_DIR="${TMP_DIR:-${DATA_DIR}/downloads_7scenes}"
ONLY_TEST_PROJ="${ONLY_TEST_PROJ:-0}"
export OUT_DIR

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

BASE_URL="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
SCENES=(chess fire heads office pumpkin redkitchen stairs)

echo "== 7Scenes setup =="
echo "OUT_DIR=${OUT_DIR}"
echo "TMP_DIR=${TMP_DIR}"
echo

for scene in "${SCENES[@]}"; do
  zip_path="${TMP_DIR}/${scene}.zip"
  if [[ ! -s "${zip_path}" ]]; then
    echo "[download] ${scene}.zip"
    wget -c -q --show-progress "${BASE_URL}/${scene}.zip" -O "${zip_path}"
  else
    echo "[skip] ${zip_path} already exists"
  fi

  if ! unzip -tqq "${zip_path}" >/dev/null 2>&1; then
    echo "[warn] ${scene}.zip is incomplete/corrupted, redownloading..."
    rm -f "${zip_path}"
    wget -c -q --show-progress "${BASE_URL}/${scene}.zip" -O "${zip_path}"
  fi

  if [[ ! -d "${OUT_DIR}/${scene}" ]]; then
    echo "[unzip] ${scene}.zip -> ${OUT_DIR}"
    unzip -oq "${zip_path}" -d "${OUT_DIR}"
  else
    echo "[skip] scene dir exists: ${OUT_DIR}/${scene}"
  fi

  # Unpack nested sequence zip files (if still present)
  shopt -s nullglob
  nested=( "${OUT_DIR}/${scene}"/*.zip )
  if (( ${#nested[@]} > 0 )); then
    echo "[nested unzip] ${scene} (${#nested[@]} files)"
    for nz in "${nested[@]}"; do
      # Some partially restored nested zips may emit warnings/non-zero exit but
      # the sequence may already be fully extracted from a previous run.
      unzip -oq "${nz}" -d "${OUT_DIR}/${scene}" || true
      rm -f "${nz}"
    done
  fi
  shopt -u nullglob
done

echo
echo "== preprocess depth.proj for FastVGGT / SimpleRecon layout =="
echo "ONLY_TEST_PROJ=${ONLY_TEST_PROJ} (0=all Train+Test sequences, 1=TestSplit only)"
if [[ "${ONLY_TEST_PROJ}" == "1" ]]; then
  python "${DATA_DIR}/preprocess_7scenes_depth_proj.py" --root "${OUT_DIR}" --only-test --skip-existing
else
  python "${DATA_DIR}/preprocess_7scenes_depth_proj.py" --root "${OUT_DIR}" --skip-existing
fi

echo
echo "== quick validation =="
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
missing = []
for s in scenes:
    sd = root / s
    if not sd.exists():
        missing.append(str(sd))
        continue
    for split in ["TrainSplit.txt", "TestSplit.txt"]:
        if not (sd / split).is_file():
            missing.append(str(sd / split))
    # Every extracted seq-* must have matching depth.png and depth.proj.png counts.
    seqs = sorted([p for p in sd.glob("seq-*") if p.is_dir()])
    if not seqs:
        missing.append(f"{sd}/seq-*")
        continue
    for seq in seqs:
        n_color = len(list(seq.glob("frame-*.color.png")))
        n_depth = len([p for p in seq.glob("frame-*.depth.png") if "proj" not in p.name])
        n_proj = len(list(seq.glob("frame-*.depth.proj.png")))
        n_pose = len(list(seq.glob("frame-*.pose.txt")))
        if n_color == 0:
            missing.append(f"{seq}: no frame-*.color.png")
        if n_depth == 0:
            missing.append(f"{seq}: no frame-*.depth.png")
        if n_proj != n_depth:
            missing.append(f"{seq}: depth.proj count {n_proj} != depth.png count {n_depth}")
        if n_pose == 0:
            missing.append(f"{seq}: no frame-*.pose.txt")

print(f"root={root}")
print(f"missing_count={len(missing)}")
if missing:
    for x in missing[:20]:
        print(" -", x)
    raise SystemExit(2)
print("validation_ok=1")
PY

echo "Done 7Scenes setup."

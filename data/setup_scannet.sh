#!/usr/bin/env bash
set -euo pipefail

# One-click ScanNet setup for:
# - scans_test50 (fixed benchmark split)
# - scans_dev50  (random split, non-overlap with test50)
#
# It performs:
# 1) split preparation
# 2) raw file download (.sens/.txt/_vh_clean_2.ply)
# 3) extraction to color/depth/pose/intrinsic
# 4) final validation
#
# Usage:
#   bash data/setup_scannet_dev_test.sh
#
# Optional env vars:
#   REPO_ROOT=/path/to/vggtlayer
#   DEV_SEED=20260420
#   DEV_COUNT=50
#   MAX_WORKERS=6
#   RETRY_PER_FILE=5

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"

TEST_LIST="${DATA_DIR}/scannet_test50.txt"
DEV_LIST="${DATA_DIR}/scannet_dev50.txt"

SCANS_TEST_DIR="${DATA_DIR}/scans_test50"
SCANS_DEV_DIR="${DATA_DIR}/scans_dev50"

DEV_SEED="${DEV_SEED:-20260420}"
DEV_COUNT="${DEV_COUNT:-50}"
MAX_WORKERS="${MAX_WORKERS:-6}"
RETRY_PER_FILE="${RETRY_PER_FILE:-5}"

BASE_URL="http://kaldir.vc.in.tum.de/scannet"
RELEASE_SCANS_TXT="${BASE_URL}/v2/scans.txt"

export DATA_DIR TEST_LIST DEV_LIST SCANS_TEST_DIR SCANS_DEV_DIR
export DEV_SEED DEV_COUNT MAX_WORKERS RETRY_PER_FILE
export BASE_URL RELEASE_SCANS_TXT

echo "== Config =="
echo "REPO_ROOT=${REPO_ROOT}"
echo "TEST_LIST=${TEST_LIST}"
echo "DEV_LIST=${DEV_LIST}"
echo "SCANS_TEST_DIR=${SCANS_TEST_DIR}"
echo "SCANS_DEV_DIR=${SCANS_DEV_DIR}"
echo "DEV_SEED=${DEV_SEED} DEV_COUNT=${DEV_COUNT}"
echo "MAX_WORKERS=${MAX_WORKERS} RETRY_PER_FILE=${RETRY_PER_FILE}"
echo

mkdir -p "${SCANS_TEST_DIR}" "${SCANS_DEV_DIR}"

echo "== Step 1/4: prepare split lists =="
python - <<'PY'
import os
import random
import urllib.request
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
test_list = Path(os.environ["TEST_LIST"])
dev_list = Path(os.environ["DEV_LIST"])
scans_test_dir = Path(os.environ["SCANS_TEST_DIR"])
release_scans_txt = os.environ["RELEASE_SCANS_TXT"]
dev_seed = int(os.environ["DEV_SEED"])
dev_count = int(os.environ["DEV_COUNT"])

if not test_list.exists():
    if scans_test_dir.exists():
        test_scenes = sorted([d.name for d in scans_test_dir.iterdir() if d.is_dir()])
        if test_scenes:
            test_list.write_text("\n".join(test_scenes) + "\n", encoding="utf-8")
            print(f"[split] generated {test_list} from existing scans_test50 dirs ({len(test_scenes)} scenes)")
        else:
            raise SystemExit(
                f"[split] missing {test_list} and no scene dirs under {scans_test_dir}. "
                "Please provide test50 scene list first."
            )
    else:
        raise SystemExit(
            f"[split] missing {test_list} and {scans_test_dir} does not exist. "
            "Please provide test50 scene list first."
        )

test_scenes = {
    ln.strip() for ln in test_list.read_text(encoding="utf-8").splitlines() if ln.strip()
}
if len(test_scenes) == 0:
    raise SystemExit("[split] test50 list is empty.")

all_scans = [
    ln.decode("utf8").strip()
    for ln in urllib.request.urlopen(release_scans_txt)
    if ln.decode("utf8").strip()
]

candidates = [s for s in all_scans if s not in test_scenes]
if len(candidates) < dev_count:
    raise SystemExit(
        f"[split] not enough candidates for dev split: {len(candidates)} < {dev_count}"
    )

rng = random.Random(dev_seed)
dev_scenes = sorted(rng.sample(candidates, dev_count))
dev_list.write_text("\n".join(dev_scenes) + "\n", encoding="utf-8")

overlap = set(dev_scenes) & test_scenes
if overlap:
    raise SystemExit(f"[split] overlap detected unexpectedly: {sorted(overlap)[:5]}")

print(f"[split] test50 scenes: {len(test_scenes)}")
print(f"[split] dev50 scenes:  {len(dev_scenes)}")
print(f"[split] wrote {dev_list} (seed={dev_seed})")
PY

echo
echo "== Step 2/4: download raw files for test50 + dev50 =="
python - <<'PY'
import os
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

socket.setdefaulttimeout(60)

base_url = os.environ["BASE_URL"]
test_list = Path(os.environ["TEST_LIST"])
dev_list = Path(os.environ["DEV_LIST"])
scans_test_dir = Path(os.environ["SCANS_TEST_DIR"])
scans_dev_dir = Path(os.environ["SCANS_DEV_DIR"])
max_workers = int(os.environ["MAX_WORKERS"])
retry_per_file = int(os.environ["RETRY_PER_FILE"])

required_files = [
    (".sens", "v1/scans"),  # v2 uses v1 sens files
    (".txt", "v2/scans"),
    ("_vh_clean_2.ply", "v2/scans"),
]

def read_list(path: Path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def build_tasks(scenes, out_root: Path):
    tasks = []
    out_root.mkdir(parents=True, exist_ok=True)
    for sid in scenes:
        sdir = out_root / sid
        sdir.mkdir(parents=True, exist_ok=True)
        for suffix, rel in required_files:
            out = sdir / f"{sid}{suffix}"
            if out.exists() and out.stat().st_size > 0:
                continue
            url = f"{base_url}/{rel}/{sid}/{sid}{suffix}"
            tasks.append((sid, suffix, url, out))
    return tasks

test_scenes = read_list(test_list)
dev_scenes = read_list(dev_list)
all_tasks = build_tasks(test_scenes, scans_test_dir) + build_tasks(dev_scenes, scans_dev_dir)
print(f"[download] pending files: {len(all_tasks)}")

def download_one(task):
    sid, suffix, url, out = task
    last_err = ""
    for _ in range(retry_per_file):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(out, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            return True, sid, suffix, ""
        except Exception as e:
            last_err = str(e)
    return False, sid, suffix, last_err

failures = []
if all_tasks:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_one, t) for t in all_tasks]
        for idx, fu in enumerate(as_completed(futures), start=1):
            ok, sid, suffix, err = fu.result()
            if ok:
                if idx % 50 == 0 or idx == len(all_tasks):
                    print(f"[download] {idx}/{len(all_tasks)} done")
            else:
                failures.append((sid, suffix, err))

if failures:
    fail_path = Path(os.environ["DATA_DIR"]) / "scannet_setup_failures.txt"
    fail_path.write_text(
        "\n".join([f"{sid}\t{suffix}\t{err}" for sid, suffix, err in failures]) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(
        f"[download] failures={len(failures)}; see {fail_path}"
    )

print("[download] all required files are present.")
PY

echo
echo "== Step 3/4: extract sens -> color/depth/pose/intrinsic =="
python "${DATA_DIR}/export_scannet_sens.py" \
  --scans-root "${SCANS_TEST_DIR}" \
  --output-root "${SCANS_TEST_DIR}" \
  --scene-list "${TEST_LIST}" \
  --depth --intrinsics --skip-existing

python "${DATA_DIR}/export_scannet_sens.py" \
  --scans-root "${SCANS_DEV_DIR}" \
  --output-root "${SCANS_DEV_DIR}" \
  --scene-list "${DEV_LIST}" \
  --depth --intrinsics --skip-existing

echo
echo "== Step 4/4: validate completeness and split disjointness =="
python - <<'PY'
import os
from pathlib import Path

test_list = Path(os.environ["TEST_LIST"])
dev_list = Path(os.environ["DEV_LIST"])
scans_test_dir = Path(os.environ["SCANS_TEST_DIR"])
scans_dev_dir = Path(os.environ["SCANS_DEV_DIR"])

def read_list(path: Path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def validate_split(name: str, scenes, root: Path):
    missing = []
    for sid in scenes:
        d = root / sid
        required = [
            d / f"{sid}.sens",
            d / f"{sid}.txt",
            d / f"{sid}_vh_clean_2.ply",
            d / "color" / "0.jpg",
            d / "pose" / "0.txt",
            d / "depth" / "0.png",
            d / "intrinsic" / "intrinsic_color.txt",
        ]
        for p in required:
            if (not p.is_file()) or p.stat().st_size == 0:
                missing.append(str(p))
    print(f"[validate] {name}: scenes={len(scenes)} missing={len(missing)}")
    if missing:
        print("[validate] first missing examples:")
        for x in missing[:10]:
            print("  -", x)
        raise SystemExit(f"[validate] {name} failed.")

test_scenes = set(read_list(test_list))
dev_scenes = set(read_list(dev_list))
overlap = sorted(test_scenes & dev_scenes)
print(f"[validate] overlap(test,dev)={len(overlap)}")
if overlap:
    raise SystemExit(f"[validate] split overlap found: {overlap[:10]}")

validate_split("test50", sorted(test_scenes), scans_test_dir)
validate_split("dev50", sorted(dev_scenes), scans_dev_dir)
print("[validate] all checks passed.")
PY

echo
echo "Done."
echo "  test50 list: ${TEST_LIST}"
echo "  dev50 list:  ${DEV_LIST}"
echo "  test50 dir:  ${SCANS_TEST_DIR}"
echo "  dev50 dir:   ${SCANS_DEV_DIR}"

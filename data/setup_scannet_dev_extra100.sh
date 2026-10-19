#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${REPO_ROOT}/data"

# Build a deterministic *additional* calibration pool. Exact ScanNet scene IDs
# already used by either main split are excluded; the existing split data is
# only read and is never rewritten by this wrapper.
TEST_LIST="${TEST_LIST:-${DATA_DIR}/scannet_test50.txt}" \
DEV_LIST="${DEV_LIST:-${DATA_DIR}/scannet_dev_extra100.txt}" \
SCANS_TEST_DIR="${SCANS_TEST_DIR:-${DATA_DIR}/scans_test50}" \
SCANS_DEV_DIR="${SCANS_DEV_DIR:-${DATA_DIR}/scans_dev_extra100}" \
DEV_COUNT="${DEV_COUNT:-100}" \
DEV_SEED="${DEV_SEED:-20260915}" \
DEV_EXCLUDE_LISTS="${DEV_EXCLUDE_LISTS:-${DATA_DIR}/scannet_dev50.txt}" \
DEV_SORT_SELECTION="${DEV_SORT_SELECTION:-0}" \
DEV_SELECTION_MANIFEST="${DEV_SELECTION_MANIFEST:-${DATA_DIR}/scans_dev_extra100/selected_scenes.json}" \
PROCESS_TEST_SPLIT=0 \
SCANNET_FAILURES_FILE="${SCANNET_FAILURES_FILE:-${DATA_DIR}/scannet_dev_extra100_failures.txt}" \
REPO_ROOT="${REPO_ROOT}" \
  bash "${DATA_DIR}/setup_scannet.sh"

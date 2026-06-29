#!/usr/bin/env bash
set -euo pipefail

# Download the official QuantVGGT W4A4 calibration parameters from Hugging Face.
# These are deployment parameters, not evaluation dataset payloads or VGGT weights.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST_DIR="${QUANTVGGT_PARAMS_DIR:-${REPO_ROOT}/data/quantvggt/w4a4_official}"
FORCE_DOWNLOAD="${QUANTVGGT_FORCE_DOWNLOAD:-0}"
HF_REVISION="831f53a9351aa2b857069d56f37523febd0e0218"
BASE_URL="https://huggingface.co/wlfeng/QuantVGGT/resolve/${HF_REVISION}/a44_quant_model_tracker_fixed_e20.pt_sym"

FRAME_FILE="qs_frame_parameters_total.pth"
FRAME_SIZE=1915247930
FRAME_SHA256="44b74f7f0826e6fa53d4f4a67f73a04c01b9ae4b5320792e6c5a23908e8d96f7"
GLOBAL_FILE="qs_global_parameters_total.pth"
GLOBAL_SIZE=1915248510
GLOBAL_SHA256="7da7f224230fd28711c0c8b844157b60485ff8bb3caead1642085bae3807659a"

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "error: sha256sum is required" >&2; exit 1; }
mkdir -p "${DEST_DIR}"

download_one() {
  local filename="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local path="${DEST_DIR}/${filename}"
  local current_size=0

  if [[ -f "${path}" ]]; then
    current_size="$(stat -c '%s' "${path}")"
    if [[ "${current_size}" -eq "${expected_size}" ]]; then
      if echo "${expected_sha256}  ${path}" | sha256sum --check --status; then
        echo "[quantvggt] verified, skipping: ${path}"
        return
      fi
      if [[ "${FORCE_DOWNLOAD}" != "1" ]]; then
        echo "error: ${path} has the expected size but failed SHA-256 verification" >&2
        echo "       Re-run with QUANTVGGT_FORCE_DOWNLOAD=1 to preserve it as .invalid and download again." >&2
        exit 1
      fi
      mv "${path}" "${path}.invalid.$(date +%s)"
      current_size=0
    elif [[ "${current_size}" -gt "${expected_size}" ]]; then
      if [[ "${FORCE_DOWNLOAD}" != "1" ]]; then
        echo "error: ${path} is larger than the official file (${current_size} > ${expected_size})" >&2
        exit 1
      fi
      mv "${path}" "${path}.invalid.$(date +%s)"
      current_size=0
    else
      echo "[quantvggt] resuming ${filename} at ${current_size}/${expected_size} bytes"
    fi
  fi

  curl -L --fail --retry 5 --retry-delay 3 -C - \
    -o "${path}" "${BASE_URL}/${filename}?download=true"

  local downloaded_size
  downloaded_size="$(stat -c '%s' "${path}")"
  if [[ "${downloaded_size}" -ne "${expected_size}" ]]; then
    echo "error: size mismatch for ${path}: ${downloaded_size} != ${expected_size}" >&2
    exit 1
  fi
  echo "${expected_sha256}  ${path}" | sha256sum --check
}

echo "== QuantVGGT official W4A4 parameters =="
echo "Destination: ${DEST_DIR}"
echo "Download size: 3.83 GB (two files)"
download_one "${FRAME_FILE}" "${FRAME_SIZE}" "${FRAME_SHA256}"
download_one "${GLOBAL_FILE}" "${GLOBAL_SIZE}" "${GLOBAL_SHA256}"

echo
echo "QuantVGGT parameters are ready. Use:"
echo "  QUANTVGGT_DTYPE=w4a4"
echo "  QUANTVGGT_PARAMS_DIR=${DEST_DIR}"

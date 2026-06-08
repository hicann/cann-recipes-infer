#!/usr/bin/env bash
# Fetch a sample test image for Vision (scheme A) end-to-end inference.
#
# The image binary itself is NOT committed to the repo (licensing); this helper
# downloads a public sample. You can also use ANY local image directly:
#   bash ../infer_vision.sh <yaml> "<prompt>" /path/to/your_image.jpg
#
# Usage:
#   bash dataset/fetch_test_image.sh                 # -> dataset/test_image.jpg
#   TEST_IMAGE_URL=<url> bash dataset/fetch_test_image.sh [out_path]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${DIR}/test_image.jpg}"
# Default: COCO val2017 sample (two cats on a couch) — the canonical HF VLM demo image.
URL="${TEST_IMAGE_URL:-http://images.cocodataset.org/val2017/000000039769.jpg}"

echo "[fetch_test_image] downloading sample -> ${OUT}"
echo "[fetch_test_image] source: ${URL}"
if ! curl -fsSL -m 60 -o "${OUT}" "${URL}"; then
  echo "[fetch_test_image] download failed (network/proxy). Either retry, set TEST_IMAGE_URL," >&2
  echo "                   or just pass your own image to infer_vision.sh as the 3rd arg." >&2
  exit 1
fi
bytes="$(wc -c < "${OUT}")"
echo "[fetch_test_image] saved ${bytes} bytes."
echo "[fetch_test_image] now run:  bash infer_vision.sh   (default yaml + this image)"

#!/bin/bash
# SPDX-License-Identifier: MIT
# Convert DeepSeek-OCR-2 to Ascend NPU version
# Usage: ./convert_to_npu.sh [source_dir] [target_dir]

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR=${1:-"DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"}
TARGET_DIR=${2:-"deepseek_ocr2_npu"}

echo "[INFO] Converting DeepSeek-OCR-2 to NPU version"
echo "[INFO] Source: $SOURCE_DIR"
echo "[INFO] Target: $TARGET_DIR"

# Install Python dependencies
echo "[INFO] Installing Python dependencies..."
pip install einops addict easydict triton-ascend PyMuPDF img2pdf \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple --quiet || {
    echo "[WARNING] Some dependencies may have failed to install"
}

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[INFO] Cloning DeepSeek-OCR-2..."
    git clone --depth=1 https://github.com/deepseek-ai/DeepSeek-OCR-2.git
    SOURCE_DIR="DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"
fi

echo "[INFO] Copying source files..."
rm -rf "$TARGET_DIR"
cp -r "$SOURCE_DIR" "$TARGET_DIR"

echo "[INFO] Adding NPU patch files..."
cp "$SCRIPT_DIR/npu_patch/deepseek_ocr2_npu.py" "$TARGET_DIR/"
cp "$SCRIPT_DIR/npu_patch/set_env.sh" "$TARGET_DIR/"

echo "[INFO] Patching sam_vary_sdpa.py..."
sed -i 's/^from flash_attn/# from flash_attn/' "$TARGET_DIR/deepencoderv2/sam_vary_sdpa.py"

echo "[INFO] Patching config.py..."
cat >> "$TARGET_DIR/config.py" << 'CFGEOF'

# ==================== NPU Configuration ====================
DEVICE = 'npu'
ENFORCE_EAGER = True
MAX_MODEL_LEN = 8192
SWAP_SPACE = 0
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
DISABLE_MM_PREPROCESSOR_CACHE = True
CFGEOF

# Patch official scripts to add NPU support and fix engine parameters
for script in run_dpsk_ocr2_image.py run_dpsk_ocr2_pdf.py run_dpsk_ocr2_eval_batch.py; do
    if [ -f "$TARGET_DIR/$script" ]; then
        echo "[INFO] Patching $script..."
        sed -i '/^import torch$/a import os\nos.environ["VLLM_USE_V1"] = "0"\nimport deepseek_ocr2_npu' "$TARGET_DIR/$script"
        sed -i 's/enforce_eager=False/enforce_eager=True/g' "$TARGET_DIR/$script"
        sed -i 's/gpu_memory_utilization=0\.75/gpu_memory_utilization=0.85/g' "$TARGET_DIR/$script"
        sed -i 's/gpu_memory_utilization=0\.9/gpu_memory_utilization=0.85/g' "$TARGET_DIR/$script"
        sed -i 's/gpu_memory_utilization=0\.7/gpu_memory_utilization=0.85/g' "$TARGET_DIR/$script"
    fi
done

# For run_dpsk_ocr2_image.py, add swap_space=0 and disable_mm_preprocessor_cache
if [ -f "$TARGET_DIR/run_dpsk_ocr2_image.py" ]; then
    echo "[INFO] Adding extra NPU params to run_dpsk_ocr2_image.py..."
    if ! grep -q 'swap_space=0' "$TARGET_DIR/run_dpsk_ocr2_image.py"; then
        sed -i '/gpu_memory_utilization=0\.85,$/a\        swap_space=0,' "$TARGET_DIR/run_dpsk_ocr2_image.py"
    fi
    if ! grep -q 'disable_mm_preprocessor_cache' "$TARGET_DIR/run_dpsk_ocr2_image.py"; then
        sed -i '/swap_space=0,$/a\        disable_mm_preprocessor_cache=True,' "$TARGET_DIR/run_dpsk_ocr2_image.py"
    fi
fi

echo "[INFO] Conversion complete: $TARGET_DIR"
echo "[INFO] Scripts available:"
echo "  - run_dpsk_ocr2_image.py  (Image streaming)"
echo "  - run_dpsk_ocr2_pdf.py    (PDF processing)"
echo "  - run_dpsk_ocr2_eval_batch.py (Batch evaluation)"

echo "[INFO] Copying benchmark.py..."
cp "$SCRIPT_DIR/npu_patch/benchmark.py" "$TARGET_DIR/"

echo "  - benchmark.py            (Performance testing)"

#!/bin/bash

set -e

echo "=== 加载 Stage 1 Patches ==="
git am ../cann-recipes-infer/models/qwen3-next/patches/stage1/*.patch

echo "=== 加载 Stage 2 Patches ==="
git am ../cann-recipes-infer/models/qwen3-next/patches/stage2/*.patch

echo "=== 加载 Stage 3 Patches ==="
git am ../cann-recipes-infer/models/qwen3-next/patches/stage3/*.patch

echo "=== 加载 Stage 4 Patches ==="
git am ../cann-recipes-infer/models/qwen3-next/patches/stage4/*.patch

echo "所有 Patch 加载完成！"
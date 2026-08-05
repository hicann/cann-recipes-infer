#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# ==================== 参数解析 ====================
REVERSE=false
MONKEY_MODE=false
for arg in "${@}"; do
    case "$arg" in
        --reverse|-r) REVERSE=true ;;
        --monkey|-m) MONKEY_MODE=true ;;
        -h|--help)
            echo "Usage: $0 [--reverse|-r] [--monkey|-m]"
            echo ""
            echo "  --reverse, -r    Revert previously applied patches (git-patch mode only)"
            echo "  --monkey, -m     Apply patches via in-place source modification instead of"
            echo "                   git apply. Equivalent end result to the default git-patch"
            echo "                   mode, but does not require a git repo for the Python"
            echo "                   sources. vllm-ascend C++ patches still use git apply."
            exit 0
            ;;
    esac
done

# ==================== 基础信息 ====================
echo "🚀 Bailing v3 0day Adapter for vLLM-Ascend 0.20.2rc1"
echo "Mode: $(if [ "$MONKEY_MODE" = true ]; then echo "Monkey-patch"; else echo "Git patch"; fi)"
echo "=================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$MONKEY_MODE" = true ] && [ "$REVERSE" = true ]; then
    echo "❌ --reverse is not applicable in monkey-patch mode"
    exit 1
fi

# ==================== Monkey-Patch 模式 ====================
# Applies the same changes as the Git-patch mode, but without requiring a git
# repository for the Python sources:
#   * vllm Python changes        -> in-place source diff (bailing_v3_patches/vllm_diffs)
#   * vllm-ascend Python changes -> in-place source diff (bailing_v3_patches/vllm_ascend_diffs)
#   * new model/op files         -> copied into the installed packages
#   * vllm-ascend C++ changes    -> git apply (requires a vllm-ascend source checkout)
# All Python changes are written persistently to the installed source files,
# which is equivalent to `git apply` of bailing_v3_vllm.patch and the Python
# sections of bailing_v3_vllm_ascend.patch.
if [ "$MONKEY_MODE" = true ]; then
    # ---- Locate vllm / vllm-ascend (|| true so set -e does not exit silently) ----
    VLLM_PATH=$(python -c "import vllm, os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null || true)
    ASCEND_PATH=$(python -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__path__[0]))" 2>/dev/null || true)

    if [ -z "$VLLM_PATH" ] || [ -z "$ASCEND_PATH" ]; then
        echo "❌ Error: vllm or vllm-ascend not found in current Python environment"
        echo "   vllm path:        ${VLLM_PATH:-<not found>}"
        echo "   vllm-ascend path: ${ASCEND_PATH:-<not found>}"
        exit 1
    fi

    echo "📍 vllm path:        $VLLM_PATH"
    echo "📍 vllm-ascend path: $ASCEND_PATH"

    # ---- Apply Python patches (new files + vllm diffs + vllm-ascend diffs) ----
    echo ""
    echo "🔧 Applying vllm + vllm-ascend Python patches (source modification)..."
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c \
        "from bailing_v3_patches import apply_all; apply_all()" \
        || { echo "❌ Python patch application failed"; exit 1; }

    # ---- Apply vllm-ascend C++ patches via git apply (needs source checkout) ----
    ASCEND_CPP_PATCH="$SCRIPT_DIR/bailing_v3_vllm_ascend_cpp.patch"
    if [ -f "$ASCEND_CPP_PATCH" ]; then
        git config --global --add safe.directory "$ASCEND_PATH" 2>/dev/null || true
        echo ""
        echo "🔧 Applying vllm-ascend C++ patches (git apply)..."
        if ( cd "$ASCEND_PATH" && git apply -p1 "$ASCEND_CPP_PATCH" ); then
            echo "✅ vllm-ascend C++ patches applied"
        else
            echo "⚠️  vllm-ascend C++ patch did not apply (may already be applied, or csrc/ absent)"
        fi
    else
        echo "⚠️  C++ patch not found: $ASCEND_CPP_PATCH"
    fi

    echo ""
    echo "🎉 Monkey-patch mode done. You can now run Bailing v3 inference."
    echo "   vllm Python changes:        source file modification (in place)"
    echo "   vllm-ascend Python changes: source file modification (in place)"
    echo "   vllm-ascend C++ changes:    git apply"
    echo "   New model/op files:         copied into installed packages"
    exit 0
fi

# ==================== Git Patch 模式 ====================
VLLM_PATH=$(python -c "import vllm, os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null)
ASCEND_PATH=$(python -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__path__[0]))" 2>/dev/null)

if [ -z "$VLLM_PATH" ] || [ -z "$ASCEND_PATH" ]; then
    echo "❌ Error: vllm or vllm-ascend not found in current Python environment"
    exit 1
fi

echo "📍 vllm path:        $VLLM_PATH"
echo "📍 vllm-ascend path: $ASCEND_PATH"

# ==================== Git 安全目录配置 ====================
echo ""
git config --global --add safe.directory "$VLLM_PATH"
git config --global --add safe.directory "$ASCEND_PATH"

# ==================== 根据模式设置变量 ====================
if [ "$REVERSE" = true ]; then
    ACTION_ICON="🔙"
    ACTION_VERB="Reversing"
    ACTION_PAST="reversed"
    GIT_APPLY_FLAG="-R"
else
    ACTION_ICON="🔧"
    ACTION_VERB="Applying"
    ACTION_PAST="applied"
    GIT_APPLY_FLAG=""
fi

# ==================== 应用/回退 vllm 补丁 ====================
VLLM_PATCH="$SCRIPT_DIR/bailing_v3_vllm.patch"
if [ ! -f "$VLLM_PATCH" ]; then
    echo "❌ Error: Patch file not found: $VLLM_PATCH"
    exit 1
fi

echo "$ACTION_ICON $ACTION_VERB vllm Bailing v3 patches..."
(
    cd "$VLLM_PATH"
    git apply $GIT_APPLY_FLAG -p1 "$VLLM_PATCH"
)
echo "✅ vllm patches $ACTION_PAST successfully"

# ==================== 应用/回退 vllm-ascend 补丁 ====================
ASCEND_PATCH="$SCRIPT_DIR/bailing_v3_vllm_ascend.patch"
if [ ! -f "$ASCEND_PATCH" ]; then
    echo "❌ Error: Patch file not found: $ASCEND_PATCH"
    exit 1
fi

echo "$ACTION_ICON $ACTION_VERB vllm-ascend Bailing v3 patches..."
(
    cd "$ASCEND_PATH"
    git apply $GIT_APPLY_FLAG -p1 "$ASCEND_PATCH"
)
echo "✅ vllm-ascend patches $ACTION_PAST successfully"

# ==================== 完成提示 ====================
echo ""
if [ "$REVERSE" = true ]; then
    echo "🎉 All done! Bailing v3 patches have been cleanly reverted."
else
    echo "🎉 All done! You can now run Bailing v3 inference."
fi
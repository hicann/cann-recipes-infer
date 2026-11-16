#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Check implementer self-verification in the agent's associated progress.md."""

import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

SELF_VERIFY_KEYWORDS = ["代码加载", "编译", "推理", "输出"]

SELF_VERIFY_TEMPLATE = """自验证记录不完整：progress.md 缺少以下自验证项：{missing}。
请按本轮任务补齐自验证结果，未执行或不适用项注明原因。
格式参考：
### 自验证结果
- 使用的 skill（如有）:
- 代码加载: 确认推理加载的是修改后的模型模块和正确的模型配置
- 编译: 检查方式及结果（不适用时注明）
- 推理: 通过/失败
- 输出: 合理/异常"""


def find_progress_md(data):
    """Resolve the progress.md associated with this subagent."""
    agent_id = data.get("agent_id")
    if agent_id:
        try:
            with open(f"/tmp/hook_read_progress_{agent_id}.marker") as f:
                path = f.read().strip()
            if os.path.isfile(path):
                return path
        except OSError:
            pass

    return None


def check_self_verification(data, content):
    """Reject an implementer that has not recorded every required self-check item."""
    if data.get("agent_type") != "model-infer-implementer":
        return None

    if "### 自验证结果" not in content:
        missing = "、".join(SELF_VERIFY_KEYWORDS)
        return SELF_VERIFY_TEMPLATE.format(missing=missing)

    verify_start = content.rindex("### 自验证结果")
    verify_section = content[verify_start:]
    next_section = verify_section.find("\n### ", 1)
    if next_section > 0:
        verify_section = verify_section[:next_section]

    missing = [keyword for keyword in SELF_VERIFY_KEYWORDS if keyword not in verify_section]
    if missing:
        return SELF_VERIFY_TEMPLATE.format(missing="、".join(missing))

    return None


def main():
    data = json.load(sys.stdin)

    # Set by the platform once a previous stop was blocked; letting it through
    # keeps a still-failing check from blocking the subagent indefinitely.
    if data.get("stop_hook_active"):
        sys.exit(0)

    progress_path = find_progress_md(data)
    if not progress_path:
        sys.exit(0)

    try:
        with open(progress_path) as f:
            content = f.read()
    except OSError:
        sys.exit(0)

    reason = check_self_verification(data, content)
    if reason:
        logger.error(reason)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()

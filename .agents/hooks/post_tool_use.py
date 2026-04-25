#!/usr/bin/env python3
"""PostToolUse hook: 标记 progress.md 已读（配合 pre_tool_use.py 检查 2）"""

import json
import os
import sys


def main():
    data = json.load(sys.stdin)

    agent_id = data.get("agent_id", "")
    if not agent_id:
        sys.exit(0)

    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path or not file_path.endswith("progress.md"):
        sys.exit(0)

    marker = f"/tmp/hook_read_progress_{agent_id}.marker"
    with open(marker, "w") as f:
        f.write(file_path)

    sys.exit(0)


if __name__ == "__main__":
    main()

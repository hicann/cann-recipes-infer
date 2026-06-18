#!/usr/bin/env bash
# init-agent.sh - Install Agent Skills into Codex, Claude Code, and/or OpenCode.
#
# Source of truth: .agents/
#     .agents/skills/       - skill directories
#     .agents/agents/       - subagent markdown files
#     .agents/hooks/        - Claude Code hook scripts
#     .agents/settings.json - Claude Code settings (hook registration)
#
# Generated (gitignored):
#     .codex/agents/*.toml                       - Codex custom agent view
#     .claude/{skills,agents,hooks,settings.json} - Claude Code view
#     .opencode/{skills,agents}                    - OpenCode view
#     CLAUDE.md -> AGENTS.md                       - Claude Code entry
#
# Usage:
#     bash scripts/init-agent.sh              # all platforms (default)
#     bash scripts/init-agent.sh --codex      # Codex only
#     bash scripts/init-agent.sh --claude     # Claude Code only
#     bash scripts/init-agent.sh --opencode   # OpenCode only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/.agents"

arg="${1:-}"
case "$arg" in
    ""|--all)        target="all" ;;
    --both)          target="legacy" ;;
    --codex)         target="codex" ;;
    --claude)        target="claude" ;;
    --opencode)      target="opencode" ;;
    -h|--help)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
    *)
        echo "Unknown argument: $arg" >&2
        echo "Usage: $0 [--all|--codex|--claude|--opencode]" >&2
        exit 1 ;;
esac

require_dir() {
    local path="$1" label="$2"
    if [ ! -d "$path" ]; then
        echo "Error: missing $label directory: $path" >&2
        exit 1
    fi
}

require_file() {
    local path="$1" label="$2"
    if [ ! -f "$path" ]; then
        echo "Error: missing $label file: $path" >&2
        exit 1
    fi
}

validate_source() {
    require_dir "$SRC" ".agents source"
    require_dir "$SRC/skills" ".agents/skills"
    require_dir "$SRC/agents" ".agents/agents"
    require_file "$REPO_ROOT/AGENTS.md" "AGENTS.md"
}

link_children() {
    # Create child symlinks in $2 for every entry in $1 (absolute paths).
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue
        local target="$dst/$(basename "$entry")"
        # Replace any pre-existing real file/dir at the target so ln does not nest into it
        if [ -e "$target" ] && [ ! -L "$target" ]; then
            rm -rf "$target"
        fi
        ln -sfn "$(realpath "$entry")" "$target"
    done
}

generate_codex_agents() {
    local src="$SRC/agents"
    local dst="$REPO_ROOT/.codex/agents"

    mkdir -p "$dst"
    rm -f "$dst"/*.toml

    python3 - "$src" "$dst" <<'PY'
from pathlib import Path
import json
import re
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

def parse_agent(path: Path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ValueError(f"{path}: missing YAML frontmatter")

    frontmatter, body = match.group(1), match.group(2).strip()
    data = {}
    current_key = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        key_match = re.match(r"^([A-Za-z0-9_-]+):(?:\s*(.*))?$", line)
        if key_match:
            key, value = key_match.group(1), (key_match.group(2) or "").strip()
            if value:
                data[key] = value.strip('"').strip("'")
                current_key = None
            else:
                data[key] = []
                current_key = key
            continue

        if current_key and stripped.startswith("- "):
            data[current_key].append(stripped[2:].strip().strip('"').strip("'"))

    return data, body

def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)

for path in sorted(src.glob("*.md")):
    frontmatter, body = parse_agent(path)
    name = frontmatter.get("name") or path.stem
    description = frontmatter.get("description", "")
    skills = frontmatter.get("skills") or []

    instructions = body
    if skills:
        instructions += "\n\nAvailable skills for this role:\n"
        instructions += "\n".join(f"- {skill}" for skill in skills)
        instructions += "\n"

    output = "\n".join([
        f"name = {toml_string(name)}",
        f"description = {toml_string(description)}",
        "",
        f"developer_instructions = {toml_string(instructions)}",
        "",
    ])

    target = dst / f"{name}.toml"
    target.write_text(output, encoding="utf-8")
    print(f"  {path.name} -> {target}")
PY
}

install_codex() {
    validate_source
    echo "Installing Codex -> $REPO_ROOT/.codex/agents/"
    generate_codex_agents
    echo "  + Codex reads $SRC/skills directly"
    echo "  + generated $REPO_ROOT/.codex/agents/*.toml"
    echo
    echo "Recommended Codex prompt:"
    echo "  请使用 Codex subagent workflow，调用 model-infer-optimize，对 models/{model_name} 做端到端 NPU 推理优化。"
    echo "  按 model-infer-analyzer / model-infer-implementer / model-infer-reviewer 三个 agent 执行，并通过 progress.md 传递状态。"
}

install_claude() {
    validate_source
    local base="$REPO_ROOT/.claude"
    echo "Installing Claude Code -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    link_children "$SRC/hooks"  "$base/hooks"
    if [ -f "$SRC/settings.json" ]; then
        cp "$SRC/settings.json" "$base/settings.json"
    fi
    ln -sfn AGENTS.md "$REPO_ROOT/CLAUDE.md"
    echo "  + $base/{skills,agents,hooks,settings.json}"
    echo "  + $REPO_ROOT/CLAUDE.md -> AGENTS.md"
}

install_opencode() {
    validate_source
    local base="$REPO_ROOT/.opencode"
    echo "Installing OpenCode -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    echo "  $base/{skills,agents} linked"
}

case "$target" in
    all)      install_codex; install_claude; install_opencode ;;
    legacy)   install_claude; install_opencode ;;
    codex)    install_codex ;;
    claude)   install_claude ;;
    opencode) install_opencode ;;
esac

echo "Done."

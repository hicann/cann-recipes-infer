#!/usr/bin/env bash
# init-agent.sh - Install Agent Skills into Claude Code and/or OpenCode.
#
# Source of truth: .agents/
#     .agents/skills/       - skill directories
#     .agents/agents/       - subagent markdown files
#     .agents/hooks/        - Claude Code hook scripts
#     .agents/settings.json - Claude Code settings (hook registration)
#
# Generated (gitignored):
#     .claude/{skills,agents,hooks,settings.json} - Claude Code view
#     .opencode/{skills,agents}                    - OpenCode view
#     CLAUDE.md -> AGENTS.md                       - Claude Code entry
#
# Usage:
#     bash scripts/init-agent.sh              # both platforms (default)
#     bash scripts/init-agent.sh --claude     # Claude Code only
#     bash scripts/init-agent.sh --opencode   # OpenCode only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/.agents"

if [ ! -d "$SRC" ]; then
    echo "Error: source dir $SRC not found." >&2
    exit 1
fi

arg="${1:-}"
case "$arg" in
    ""|--both|--all) target="both" ;;
    --claude)        target="claude" ;;
    --opencode)      target="opencode" ;;
    -h|--help)
        sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
    *)
        echo "Unknown argument: $arg" >&2
        echo "Usage: $0 [--claude|--opencode]" >&2
        exit 1 ;;
esac

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

install_claude() {
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
    local base="$REPO_ROOT/.opencode"
    echo "Installing OpenCode -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    echo "  $base/{skills,agents} linked"
}

case "$target" in
    both)     install_claude; install_opencode ;;
    claude)   install_claude ;;
    opencode) install_opencode ;;
esac

echo "Done."

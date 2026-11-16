#!/usr/bin/env bash
# init-agent.sh - Install model agents and cannbot-knowledge for the selected clients.
#
# Source of truth: .agents/
#     .agents/skills/       - skill directories
#     .agents/agents/       - subagent markdown files
#     .agents/hooks/        - shared hook scripts
#     .agents/settings.json - shared hook registration
#
# Generated (gitignored):
#     .codex/agents/*.toml                       - Codex custom agent view
#     .codex/hooks.json                          - Codex hook registration
#     .claude/{skills,agents,hooks,settings.json} - Claude Code view
#     .opencode/{skills,agents}                    - OpenCode view
#     CLAUDE.md with @AGENTS.md                   - shared project instructions
#     .cannbot/knowledge.env                      - knowledge repository configuration
#
# Usage:
#     bash scripts/init-agent.sh              # all platforms (default)
#     bash scripts/init-agent.sh --codex      # Codex only
#     bash scripts/init-agent.sh --claude     # Claude Code only
#     bash scripts/init-agent.sh --opencode   # OpenCode only
#
# Knowledge options:
#     --knowledge-profile consumer|contributor  # default: installed profile or consumer
#     --knowledge-root PATH                    # reuse or install a knowledge checkout
#     --skip-resources                         # explicitly offline, knowledge-only install

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/.agents"

target="all"
knowledge_profile=""
knowledge_root=""
skip_resources=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --all)      target="all" ;;
        --both)     target="legacy" ;;
        --codex)    target="codex" ;;
        --claude)   target="claude" ;;
        --opencode) target="opencode" ;;
        --knowledge-profile|--knowledge-root)
            if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "Error: $1 requires a value" >&2
                exit 1
            fi
            case "$1" in
                --knowledge-profile) knowledge_profile="$2" ;;
                --knowledge-root) knowledge_root="$2" ;;
            esac
            shift ;;
        --skip-resources) skip_resources=true ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
    shift
done
case "$knowledge_profile" in
    ""|consumer|contributor) ;;
    *) echo "Unknown knowledge profile: $knowledge_profile" >&2; exit 1 ;;
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

prepare_claude_entry() {
    python3 - "$REPO_ROOT" <<'PYTHON'
from pathlib import Path
import sys

root = Path(sys.argv[1])
entry = root / "CLAUDE.md"
if entry.is_symlink():
    if entry.resolve() != (root / "AGENTS.md").resolve():
        raise SystemExit(f"Error: unexpected CLAUDE.md symlink: {entry}")
    entry.unlink()
content = entry.read_bytes() if entry.exists() else b""
if b"@AGENTS.md" not in content.splitlines():
    entry.write_bytes(b"@AGENTS.md\n" + (b"\n" + content if content else b""))
PYTHON
}

install_knowledge() (
    local requested_root="${knowledge_root:-${CANNBOT_KNOWLEDGE_INSTALL_ROOT:-${CANNBOT_KNOWLEDGE_ROOT:-}}}"
    local requested_profile="${knowledge_profile:-${CANNBOT_KNOWLEDGE_PROFILE:-}}"
    if [ -f "$REPO_ROOT/.cannbot/knowledge.env" ]; then
        source "$REPO_ROOT/.cannbot/knowledge.env"
    fi
    local root="${requested_root:-${CANNBOT_KNOWLEDGE_ROOT:-$HOME/.cannbot/cannbot-knowledge}}"
    local profile="${requested_profile:-${CANNBOT_KNOWLEDGE_PROFILE:-consumer}}"
    local installer="$root/install.sh"
    local options=(--knowledge-root "$root")
    if [ -f "$installer" ]; then
        options+=(--skip-repository-update)
    else
        installer="$(mktemp)"
        trap 'rm -f "$installer"' EXIT
        curl -fsSL https://raw.gitcode.com/cann/cannbot-knowledge/raw/master/install.sh -o "$installer"
    fi

    if [ "$skip_resources" = true ]; then
        options+=(--skip-resources)
    fi

    # The codex layout installs the shared .agents/skills entries without client adapters.
    bash "$installer" codex "$REPO_ROOT" "$profile" "${options[@]}"
    # Keep the repository's concise knowledge guidance instead of a second managed block.
    python3 - "$REPO_ROOT" "$installer" <<'PYTHON'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
entries = [("codex", root / "AGENTS.md")]
claude_entry = root / "CLAUDE.md"
if claude_entry.is_file() and b"@AGENTS.md" in claude_entry.read_bytes().splitlines():
    entries.append(("claude", claude_entry))
for client, path in entries:
    original = path.read_bytes()
    subprocess.run(["bash", sys.argv[2], client, str(root), "--prompt-action", "uninstall"], check=True)
    content = path.read_bytes()
    if content != original:
        newline = b"\r\n" if b"\r\n" in original else b"\n"
        path.write_bytes(content.rstrip(b"\r\n") + newline)
PYTHON
)

link_children() {
    # Create child symlinks in $2 for every entry in $1 (absolute paths).
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue
        local target="$dst/$(basename "$entry")"
        if [ -e "$target" ] && [ ! -L "$target" ]; then
            echo "Error: client view conflicts with an existing file or directory: $target" >&2
            return 1
        fi
        ln -sfn "$(realpath "$entry")" "$target"
    done
}

generate_codex_agents() {
    local src="$SRC/agents"
    local dst="$REPO_ROOT/.codex/agents"

    mkdir -p "$dst"

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

register_hooks() {
    python3 - "$REPO_ROOT" "$1" <<'PY'
from pathlib import Path
import json
import re
import shlex
import sys

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
source = json.loads((root / ".agents/settings.json").read_text())
config = json.loads(destination.read_text()) if destination.exists() else {}
hooks = config.setdefault("hooks", {})
script_names = [path.name for path in (root / ".agents/hooks").glob("*.py")]
owned_command = re.compile(r"\.(?:agents|claude)/hooks/(?:" + "|".join(map(re.escape, script_names)) + r")\b")

for event, groups in source["hooks"].items():
    retained = []
    for group in hooks.get(event, []):
        handlers = [hook for hook in group["hooks"] if not owned_command.search(hook.get("command", ""))]
        if handlers:
            retained.append({**group, "hooks": handlers})
    for group in groups:
        for hook in group["hooks"]:
            parts = shlex.split(hook["command"])
            hook["command"] = shlex.join([
                str(root / part) if part.startswith(".agents/hooks/") else part for part in parts
            ])
    hooks[event] = retained + groups

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
}

install_codex() {
    validate_source
    echo "Installing Codex -> $REPO_ROOT/.codex/agents/"
    generate_codex_agents
    register_hooks "$REPO_ROOT/.codex/hooks.json"
    echo "  + Codex reads $SRC/skills directly"
    echo "  + generated $REPO_ROOT/.codex/agents/*.toml"
    echo "  + generated $REPO_ROOT/.codex/hooks.json; review and trust the hooks in Codex /hooks"
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
    register_hooks "$base/settings.json"
    echo "  + $base/{skills,agents,hooks,settings.json}"
    echo "  + $REPO_ROOT/CLAUDE.md imports AGENTS.md"
}

install_opencode() {
    validate_source
    local base="$REPO_ROOT/.opencode"
    echo "Installing OpenCode -> $base/"
    link_children "$SRC/skills" "$base/skills"
    link_children "$SRC/agents" "$base/agents"
    echo "  $base/{skills,agents} linked"
}

validate_source
if [ "$target" = all ] || [ "$target" = legacy ] || [ "$target" = claude ] || [ -L "$REPO_ROOT/CLAUDE.md" ]; then
    prepare_claude_entry
fi
install_knowledge

case "$target" in
    all)      install_codex; install_claude; install_opencode ;;
    legacy)   install_claude; install_opencode ;;
    codex)    install_codex ;;
    claude)   install_claude ;;
    opencode) install_opencode ;;
esac

echo "Done."

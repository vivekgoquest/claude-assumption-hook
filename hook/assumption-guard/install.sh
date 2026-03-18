#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="assumption-guard"
CLAUDE_DIR="${CLAUDE_DIR:-$HOME/.claude}"
HOOKS_DIR="$CLAUDE_DIR/hooks"
TARGET_DIR="$HOOKS_DIR/$PACKAGE_NAME"
STATE_DIR="$TARGET_DIR/state"
SETTINGS_PATH="$CLAUDE_DIR/settings.json"
COMMAND="python3 ~/.claude/hooks/assumption-guard/assumption-guard.py"

mkdir -p "$HOOKS_DIR"

if [ "$SCRIPT_DIR" != "$TARGET_DIR" ]; then
  TMP_DIR="$HOOKS_DIR/.${PACKAGE_NAME}.tmp.$$"
  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"
  cp -R "$SCRIPT_DIR"/. "$TMP_DIR"/
  rm -rf "$TARGET_DIR"
  mv "$TMP_DIR" "$TARGET_DIR"
else
  echo "Assumption Guard is already installed at $TARGET_DIR"
fi

mkdir -p "$STATE_DIR"

python3 - "$SETTINGS_PATH" "$COMMAND" <<'PY'
import json
import sys
from pathlib import Path

settings_path = Path(sys.argv[1])
command = sys.argv[2]

if settings_path.exists():
    settings = json.loads(settings_path.read_text())
else:
    settings = {}

hooks = settings.setdefault("hooks", {})
stop_entries = hooks.setdefault("Stop", [])

command_exists = False
for entry in stop_entries:
    for hook in entry.get("hooks", []):
        if hook.get("type") == "command" and hook.get("command") == command:
            command_exists = True
            hook.setdefault("timeout", 10)
            break
    if command_exists:
        break

if not command_exists:
    stop_entries.append(
        {
            "hooks": [
                {
                    "type": "command",
                    "command": command,
                    "timeout": 10,
                }
            ]
        }
    )

settings_path.parent.mkdir(parents=True, exist_ok=True)
settings_path.write_text(json.dumps(settings, indent=2) + "\n")
PY

cat <<EOF
Installed Assumption Guard to:
  $TARGET_DIR

Updated Claude settings:
  $SETTINGS_PATH

Stop hook command:
  $COMMAND

Mutable runtime state lives in:
  $STATE_DIR
EOF

# Assumption Guard Hook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Claude Code Stop hook that detects hedging/assumption language in Opus responses and forces re-verification.

**Architecture:** Single Python 3.8+ script (stdlib only) triggered by a `Stop` hook. Reads the JSONL transcript, extracts all assistant text from the current turn, runs ~30 regex patterns against it, and returns `{"decision": "block", "reason": "..."}` if hedging is found.

**Tech Stack:** Python 3.8+ (json, re, sys, os), Claude Code hooks (settings.json)

**Spec:** `docs/superpowers/specs/2026-03-17-assumption-guard-hook-design.md`

---

### File Structure

- Create: `.claude/hooks/assumption-guard.py` — the single self-contained script
- Create: `.claude/settings.json` — Stop hook configuration

---

### Task 1: Write the assumption-guard.py script

**Files:**
- Create: `.claude/hooks/assumption-guard.py`

- [ ] **Step 1: Write the complete script**

```python
#!/usr/bin/env python3
"""
Assumption Guard Hook for Claude Code.

A Stop hook that detects hedging/assumption language in Claude's responses
and blocks until Claude verifies each claim using its tools.

Strict mode: flags ANY hedging language regardless of context.
"""

import json
import os
import re
import sys

# ============================================================================
# HEDGING PATTERNS — All 8 categories, \b-bounded, hardcoded
# ============================================================================

PATTERNS = [
    # Category 1: Epistemic modals
    r"\b(likely|unlikely|probably|possibly|perhaps|maybe|conceivably)\b",
    r"\b(might|may|could)\s+(be|have|cause|lead|result|work|mean|indicate)",

    # Category 2: Shields
    r"\b(I think|I believe|I suppose|I imagine|I suspect|I expect)\b",
    r"\b(I would say|I'd guess|I feel like|my understanding is)\b",
    r"\b(it seems|it appears|it looks like|it sounds like)\b",
    r"\b(seemingly|apparently|ostensibly)\b",

    # Category 3: Assumption markers
    r"\b(I assume|I presume|I'm guessing|presumably)\b",
    r"\b(my assumption is|assuming that|I would assume)\b",

    # Category 4: Disclaimers
    r"\b(I'm not sure|I'm not certain|I could be wrong)\b",
    r"\b(don't quote me|I haven't verified|off the top of my head)\b",
    r"\b(I'm fairly confident|I'm reasonably sure|I'm pretty sure)\b",
    r"\b(I'm not entirely sure|I'm somewhat uncertain)\b",

    # Category 5: Attribution hedges
    r"\b(based on my understanding|from what I can tell)\b",
    r"\b(from what I've seen|from what I recall)\b",
    r"\b(as far as I can tell|as far as I'm aware)\b",
    r"\b(if memory serves|if I recall correctly)\b",

    # Category 6: Conditional hedges
    r"\b(if I'm not mistaken|unless I'm wrong)\b",
    r"\b(correct me if I'm wrong)\b",

    # Category 7: Approximators
    r"(approximately|roughly|around|about)\s+\d",
    r"\b(somewhat|fairly|rather|relatively|sort of|kind of)\b",

    # Category 8: AI-specific patterns
    r"\b(if I recall correctly|from what I remember)\b",
    r"\b(as of my last update|based on my training)\b",
    r"\b(to the best of my knowledge|to my knowledge)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def read_jsonl(path):
    """Read a JSONL file and return list of parsed JSON objects."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def find_last_user_text_index(entries):
    """Find the index of the last user text message (not tool_result).

    The transcript has top-level entries with 'type' field.
    User text messages have type='user' and message.content is a string
    (the typed prompt), or message.content is a list containing text blocks.
    We skip tool_result entries which also have type='user'.
    """
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("type") != "user":
            continue
        msg = entry.get("message", {})
        content = msg.get("content", "")
        # If content is a string, it's a user-typed prompt
        if isinstance(content, str):
            return i
        # If content is a list, check for text blocks (not just tool_result)
        if isinstance(content, list):
            has_text = any(
                c.get("type") == "text" for c in content if isinstance(c, dict)
            )
            has_only_tool_results = all(
                c.get("type") == "tool_result" for c in content if isinstance(c, dict)
            )
            if has_text and not has_only_tool_results:
                return i
    return 0


def extract_assistant_texts(entries, from_index):
    """Extract all assistant text content from entries after from_index."""
    texts = []
    for entry in entries[from_index:]:
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
    return texts


def get_line_containing(text, pos):
    """Get the full line containing the character at position pos."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def scan_for_hedging(text):
    """Scan text for hedging patterns. Returns dict of line -> set of matched words."""
    flagged_lines = {}
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            line = get_line_containing(text, match.start())
            if line:
                flagged_lines.setdefault(line, set()).add(match.group())
    return flagged_lines


def format_reason(flagged_lines):
    """Format the block reason with flagged lines and instructions."""
    parts = ["Your response contains unverified assumptions:\n"]
    for i, (line, words) in enumerate(flagged_lines.items(), 1):
        truncated = line[:200] + "..." if len(line) > 200 else line
        flagged_words = ", ".join(f'"{w}"' for w in sorted(words))
        parts.append(f'  {i}. "{truncated}"')
        parts.append(f"     Flagged: {flagged_words}\n")
    parts.append(
        "Verify EACH flagged claim using your tools (Read, Grep, Bash).\n"
        "Replace assumption language with verified facts, or state\n"
        "explicitly what you checked and why it cannot be confirmed.\n"
        "Do NOT simply rephrase — actually verify."
    )
    return "\n".join(parts)


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    transcript_path = input_data.get("transcript_path", "")

    if not transcript_path or not os.path.exists(transcript_path):
        sys.exit(0)

    entries = read_jsonl(transcript_path)
    if not entries:
        sys.exit(0)

    turn_start = find_last_user_text_index(entries)
    assistant_texts = extract_assistant_texts(entries, turn_start)

    if not assistant_texts:
        sys.exit(0)

    full_text = "\n".join(assistant_texts)
    flagged_lines = scan_for_hedging(full_text)

    if not flagged_lines:
        sys.exit(0)

    reason = format_reason(flagged_lines)
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x .claude/hooks/assumption-guard.py`

- [ ] **Step 3: Smoke test the script with mock input**

Run:
```bash
echo '{"transcript_path": "/dev/null"}' | python3 .claude/hooks/assumption-guard.py
echo "Exit code: $?"
```
Expected: No output, exit code 0 (empty transcript = no flags = allow stop)

- [ ] **Step 4: Test with a mock transcript containing hedging**

Run:
```bash
# Create a temp JSONL with hedging language
TMPFILE=$(mktemp)
echo '{"type":"user","message":{"content":"explain the bug"}}' > "$TMPFILE"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"I think the function probably returns null. It seems like the timeout is likely 30 seconds."}]}}' >> "$TMPFILE"
echo "{\"transcript_path\": \"$TMPFILE\"}" | python3 .claude/hooks/assumption-guard.py
echo "Exit code: $?"
rm "$TMPFILE"
```
Expected: JSON output with `"decision": "block"` and reason listing "I think", "probably", "It seems", "likely"

- [ ] **Step 5: Test with a clean transcript (no hedging)**

Run:
```bash
TMPFILE=$(mktemp)
echo '{"type":"user","message":{"content":"explain the bug"}}' > "$TMPFILE"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"The function returns null when the input array is empty. The timeout is 30 seconds, configured in config.yaml line 42."}]}}' >> "$TMPFILE"
echo "{\"transcript_path\": \"$TMPFILE\"}" | python3 .claude/hooks/assumption-guard.py
echo "Exit code: $?"
rm "$TMPFILE"
```
Expected: No output, exit code 0

- [ ] **Step 6: Commit**

```bash
git add .claude/hooks/assumption-guard.py
git commit -m "feat: add assumption-guard hook script"
```

---

### Task 2: Configure the Stop hook in settings.json

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Write the settings.json**

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/assumption-guard.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Validate the JSON is well-formed**

Run: `python3 -c "import json; json.load(open('.claude/settings.json')); print('Valid JSON')"`
Expected: `Valid JSON`

- [ ] **Step 3: Commit**

```bash
git add .claude/settings.json
git commit -m "feat: configure Stop hook for assumption guard"
```

---

### Task 3: End-to-end validation

- [ ] **Step 1: Verify file structure**

Run: `find .claude -type f | sort`
Expected:
```
.claude/hooks/assumption-guard.py
.claude/settings.json
```

- [ ] **Step 2: Verify script runs without errors on real-world-like input**

Run:
```bash
TMPFILE=$(mktemp)
# Simulate a multi-message turn with tool use in between
echo '{"type":"user","message":{"content":"what does this function do"}}' > "$TMPFILE"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"Let me check the code."}]}}' >> "$TMPFILE"
echo '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","id":"tu1","input":{"file_path":"src/app.py"}}]}}' >> "$TMPFILE"
echo '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu1","content":"def process(): return None"}]}}' >> "$TMPFILE"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"The function processes data and returns None. Based on my understanding, it might be handling edge cases."}]}}' >> "$TMPFILE"
echo "{\"transcript_path\": \"$TMPFILE\"}" | python3 .claude/hooks/assumption-guard.py
echo "Exit code: $?"
rm "$TMPFILE"
```
Expected: JSON with `"decision": "block"` — flags "Based on my understanding" and "might be"

- [ ] **Step 3: Final commit with both files**

Only if any fixes were needed during validation.

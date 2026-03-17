# Assumption Guard Hook — Design Spec

## Problem

Claude Code (Opus) frequently makes unverified assumptions and communicates them using hedging language ("likely", "I think", "probably", "if I recall correctly"). This erodes trust and leads to incorrect guidance. There is no mechanism today to catch this behavior before the response reaches the user.

## Solution

A Claude Code `Stop` hook that:

1. Fires every time Claude finishes responding
2. Scans all assistant messages in the latest turn for hedging/assumption language
3. If found, blocks Claude from stopping and feeds back the specific flagged sentences
4. Claude then uses its tools (Read, Grep, Bash) to actually verify the flagged claims
5. Loop continues until the response is clean of hedging language

## Architecture

### Files

```
.claude/hooks/assumption-guard.py    # Single self-contained script
.claude/settings.json                # Stop hook entry (or add to existing)
```

Two files total. No config files, no dependencies, no external packages.

### Detection Mode

**Strict mode** — flag ANY hedging language regardless of context. No false-positive filtering. No contextual judgment. If the word matches, it gets flagged. Code blocks (fenced with triple backticks) are not excluded from scanning. This is intentional for v1; code-block exclusion may be added later if real usage shows excessive false positives from code content.

Rationale: start restrictive, relax based on real usage data.

### Hook Type

**`command` hook** running a Python 3 script. No LLM-as-judge (Haiku), no agent hook. The regex script detects hedging, and Claude Code (Opus) itself does the verification work — it has full tool access and can actually check whether the assumption is correct, unlike Haiku which can only judge surface-level text.

### No Infinite Loop Handling

No max-retry cap. No escape phrases. The loop runs until clean. Guardrails will be added later based on real usage patterns.

Note: The Stop hook input includes a `stop_hook_active` boolean (true when Claude is already continuing from a previous block). This field is intentionally ignored in v1. Future loop-protection work should use it.

## Flow

```
Claude finishes responding
       |
       v
Stop hook fires -> assumption-guard.py receives JSON on stdin
       |
       v
Script reads transcript_path -> parses JSONL
       |
       v
Extracts ALL assistant text messages from the latest turn
(everything after the last user message)
       |
       v
Runs regex matching against hardcoded hedging dictionary
       |
       v
No matches? -> exit 0 -> Claude stops normally
       |
       v
Matches found? -> stdout JSON with decision: "block" and reason
listing each flagged sentence + the specific flagged word
       |
       v
Claude receives the reason, uses its tools to verify each claim,
responds again with verified facts
       |
       v
Stop hook fires again -> re-checks -> loop until clean
```

## Hedging Dictionary

All 8 categories included. ~30 regex patterns covering ~150+ hedging terms. All hardcoded in the script.

### Category 1: Epistemic Modals

```
likely, unlikely, probably, possibly, perhaps, maybe, conceivably
might/may/could + verb (be, have, cause, lead, result, work, mean, indicate)
```

### Category 2: Shields

```
I think, I believe, I suppose, I imagine, I suspect, I expect
I would say, I'd guess, I feel like, my understanding is
it seems, it appears, it looks like, it sounds like
seemingly, apparently, ostensibly
```

### Category 3: Assumption Markers

```
I assume, I presume, I'm guessing, presumably
my assumption is, assuming that, I would assume
```

### Category 4: Disclaimers

```
I'm not sure, I'm not certain, I could be wrong
don't quote me, I haven't verified, off the top of my head
I'm fairly confident, I'm reasonably sure, I'm pretty sure
I'm not entirely sure, I'm somewhat uncertain
```

### Category 5: Attribution Hedges

```
based on my understanding, from what I can tell
from what I've seen, from what I recall
as far as I can tell, as far as I'm aware
if memory serves, if I recall correctly
```

### Category 6: Conditional Hedges

```
if I'm not mistaken, unless I'm wrong
correct me if I'm wrong
```

### Category 7: Approximators

```
approximately/roughly/around/about + number
somewhat, fairly, rather, relatively, sort of, kind of
```

### Category 8: AI-Specific Patterns

```
if I recall correctly, from what I remember
as of my last update, based on my training
to the best of my knowledge, to my knowledge
```

## Turn Boundary Detection

The transcript JSONL contains interleaved entries with `role` and `type` fields:

```jsonl
{"role": "user", "type": "text", "content": "..."}
{"role": "assistant", "type": "text", "content": "Let me check..."}
{"role": "assistant", "type": "tool_use", "name": "Read", ...}
{"role": "user", "type": "tool_result", ...}
{"role": "assistant", "type": "text", "content": "Based on this, I think..."}
```

Note: `tool_result` entries have `role: "user"`, so the turn boundary logic must find the last entry where `role == "user"` AND `type == "text"` — not just any `role: "user"` entry. Every `assistant` text entry after that boundary belongs to the current turn and gets scanned. This ensures hedging that occurs before tool calls (not just in the final message) is caught.

Note: The Stop hook also provides a `last_assistant_message` convenience field containing only the final response text. We deliberately parse the full transcript instead, because hedging can appear in earlier assistant messages within the same turn (e.g., before tool calls).

**Re-scan behavior:** Because the script scans all assistant messages in the turn (not just the latest), corrected messages from previous block iterations will also be re-scanned. Claude must avoid hedging language even when describing its correction process (e.g., "I verified that X" is fine, but "I verified what I previously assumed" would re-flag "assumed").

## Hook Reason Format

When hedging is detected, the hook returns a structured reason that:

1. Lists each flagged sentence verbatim
2. Identifies the specific flagged word/phrase
3. Instructs Claude to verify using tools or explicitly state why verification is impossible

Example reason output:

```
Your response contains unverified assumptions:

  1. "This is likely caused by a race condition"
     Flagged: "likely"

  2. "I think the timeout is 30 seconds"
     Flagged: "I think"

  3. "The function probably returns null on failure"
     Flagged: "probably"

Verify EACH flagged claim using your tools (Read, Grep, Bash).
Replace assumption language with verified facts, or state
explicitly what you checked and why it cannot be confirmed.
Do NOT simply rephrase — actually verify.
```

## Settings.json Configuration

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

The `$CLAUDE_PROJECT_DIR` variable ensures the script is found regardless of the shell's working directory at hook execution time. Timeout of 10s should suffice for most conversations; may need increasing for very long sessions with large transcripts.

## Script Structure (pseudocode)

All regex patterns use `\b` word boundaries for single-word terms (e.g., `\blikely\b`) to prevent matching inside other words (e.g., "unlikely" is its own match, "dismay" does not match "may"). Multi-word phrases are naturally bounded by spaces.

Sentence extraction: split the text on newlines first (treating each line as a unit), then flag the entire line containing the match. This avoids the complexity of sentence-boundary detection (abbreviations, code, bullet points) while giving Claude enough context to understand what to verify.

Deduplication: if the same line contains multiple hedging words, it appears once in the output with all flagged words listed.

Minimum Python version: 3.8+ (for walrus operator convenience, though 3.6+ would work with minor syntax changes).

```
PATTERNS = [...]  # All 8 categories, \b-bounded, hardcoded

def main():
    input_data = json.load(sys.stdin)
    transcript_path = input_data["transcript_path"]

    # Gracefully handle missing/empty transcript
    if not os.path.exists(transcript_path):
        sys.exit(0)

    messages = read_jsonl(transcript_path)

    # Find last user TEXT message = turn boundary
    # (tool_result entries also have role="user", so filter by type="text")
    turn_start = find_last_index(messages, role="user", type="text")

    # Collect all assistant text after turn boundary
    assistant_texts = []
    for msg in messages[turn_start:]:
        if msg["role"] == "assistant" and msg.get("type") == "text":
            assistant_texts.append(msg["content"])

    full_text = "\n".join(assistant_texts)

    # Scan for hedging — deduplicate by line
    flagged_lines = {}  # line_text -> set of matched words
    for pattern in PATTERNS:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            line = get_line_containing(full_text, match.start())
            flagged_lines.setdefault(line, set()).add(match.group())

    if not flagged_lines:
        sys.exit(0)

    # Build reason and block
    reason = format_reason(flagged_lines)
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)
```

## Dependencies

- Python 3.8+ (stdlib only: `json`, `re`, `sys`, `os`)
- No pip packages
- No external config files

## What This Does NOT Do

- No contextual/semantic analysis — strict keyword matching only
- No false-positive filtering — every match is flagged
- No infinite loop protection — runs until clean
- No logging or metrics — ships bare
- No Haiku/LLM-as-judge — Claude (Opus) does its own verification
- No detection of **confident assertions without verification** (Failure Mode 2 — see below)

---

## Known Limitation: Failure Mode 2 — Confident Confabulation

### The Problem This Hook Does NOT Solve

This hook catches hedging language — words like "likely", "I think", "probably." But analysis of a real Opus conversation (Fish Audio S2 speed-control research, 2026-03-17) revealed a **more dangerous failure mode**: Opus makes specific technical claims with full confidence, backed by zero tool verification.

The hedging hook catches the fish that signal they're uncertain. Failure Mode 2 is the fish that swim through with full confidence while being wrong.

### Evidence From the Transcript

Session: `~/.claude/projects/.../a015ffa7-b92b-4e73-b7f2-a8097e26f5cd.jsonl`

**Turn 13** — Opus read 6 code files, then wrote a 7000-char analysis. It mixed verified code facts with ungrounded claims, all stated with equal confidence:

| Claim (stated as fact) | Tool-verified? | Later proven... |
|------------------------|---------------|-----------------|
| "Fish S2 already supports `[slow]`, `[in a hurry]` tags — speed control is already solved" | No — never searched for `[slow]` in code or tested it | Wrong. No `[slow]` tag exists in the trained model. |
| "93.3% tag-activation rate" applies to speed control | No — number is from tech report about emotion/style tags | Misleading. Does not apply to duration. |
| "There's no special speaker embedding layer" | Partial — quick code scan | Correct, but was not rigorously verified. |

This single unverified claim ("NL tags already solve speed control") led Opus to dismiss the user's entire recommendation as "largely misguided." The verdict was built on an assumption, not evidence.

**Turn 14** — User said "my use case is dubbing." Opus produced a full pipeline design with specific numbers (21.5 Hz frame rate, 46.5ms per token, WSOLA recommendations, pipeline diagrams). **Zero tool calls in this turn.** Every number was derived from the mental model built in Turn 13, which was already partially wrong.

**Turn 17** — After the user corrected Opus on IndexTTS, Opus reversed its position and produced a new design with LoRA hyperparameters ("rank 16-32, learning rate ~1e-4") and the critical training data claim ("every existing sample already has duration — no new data needed"). **Zero tool calls.** Opus later admitted: "recommended LoRA hyperparameters by analogy rather than grounding."

**Turn 23** — Opus fetched config.json (good), then repeated the training data claim alongside verified config values. The verified and unverified facts were visually indistinguishable.

### The Pattern

Opus reads code in early turns, builds a mental model, then **free-associates from that model across multiple subsequent turns without going back to verify**. Each turn drifts further from grounded facts. Verified code details and ungrounded inferences are presented with identical confidence and formatting.

The user had to explicitly say "check every assumption" to trigger self-correction (Turn 30). Without that prompt, the unverified claims would have stood.

### Why Regex Can't Catch This

Failure Mode 2 has no linguistic signal. The claims use no hedging words:
- "Fish S2 already supports inline tags like `[slow]`" — no hedge
- "Every training sample already has duration information" — no hedge
- "LoRA: rank 16-32 on q_proj, k_proj, v_proj" — no hedge

These are stated as facts. The signal is structural, not lexical: **the ratio of specific technical claims to tool calls in the same turn is too high.**

### v2 Direction: Claims-to-Tools Ratio Detection

A future hook could detect this by:

1. **Counting technical claims** in the assistant's response — look for:
   - Specific numbers (frame rates, dimensions, percentages)
   - Function/class/file name references not seen in tool results this turn
   - Architecture assertions ("X uses Y", "X supports Z", "X does NOT have Z")
   - Recommendations with specific parameters (hyperparameters, bin counts, layer names)

2. **Counting tool calls** in the same turn — Read, Grep, Bash, Agent, WebFetch

3. **Flagging when the ratio is suspicious** — e.g., 10+ specific technical claims but 0 tool calls in this turn. The block reason would say: "You made N specific technical claims in this response but used 0 tools to verify them. Go back and verify: [list of claims]."

This would require either:
- A `prompt` hook (Haiku extracts claims from the response, compares to tool usage)
- An `agent` hook (subagent with tool access parses the transcript and counts)
- A more sophisticated `command` hook with heuristic claim detection (regex for numbers, function names, file paths not in tool results)

### The Fundamental Insight

> **Opus's most dangerous mode is not when it hedges — it's when it doesn't.**
>
> Hedging at least signals uncertainty to the user. Confident confabulation gives the user no signal at all. The v1 hook catches the honest failures. A v2 hook would need to catch the dishonest ones — claims that sound verified but aren't.

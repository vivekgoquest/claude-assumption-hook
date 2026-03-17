# Assumption Guard — Complete System Documentation

## What It Is

A Claude Code **Stop hook** that detects when Claude makes unverified claims using hedging language ("I think", "probably", "likely") and blocks the response until Claude goes back and verifies each claim with its tools (Read, Grep, Bash).

Two-stage architecture:
1. **Regex pre-filter** — 30 compiled patterns scan all assistant text in the current turn. Zero-cost when clean (most responses). Catches 150+ hedging terms across 8 categories.
2. **ML classifier** — A trained scikit-learn model (TF-IDF + intent features) filters out false positives. Only runs when regex finds something. Classifies each flagged line as genuine hedging vs. quotation/idiom/type analysis/conditional/code/acknowledged limitation.

Every invocation is logged to JSONL with both stages visible, enabling continuous model refinement.

---

## Deployed Files

All files live in `~/.claude/` (global, applies to all Claude Code sessions):

```
~/.claude/
├── hooks/
│   └── assumption-guard.py          # The hook script (14KB)
├── assumption-guard-model.pkl        # Trained ML model (156KB)
├── assumption-guard-training-labeled.jsonl  # Training data (215 examples)
├── assumption-guard.log.jsonl        # Runtime log (growing)
└── settings.json                     # Hook registration (Stop hook entry)
```

### settings.json hook entry

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.claude/hooks/assumption-guard.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

---

## How It Works — Execution Flow

```
Claude finishes responding
        │
        v
Stop hook fires → assumption-guard.py receives JSON on stdin:
  { "session_id", "transcript_path", "cwd",
    "stop_hook_active", "last_assistant_message" }
        │
        v
Parse JSONL transcript → find last user text message (turn boundary)
  - Skips tool_result entries (also type="user" but not user-typed text)
  - Handles string content and list-of-blocks content
  - Hook feedback from previous blocks creates a new turn boundary
        │
        v
Extract ALL assistant text blocks from turn boundary onward
  - Handles both list-of-blocks and string-typed content
  - Includes text before and after tool calls within the same turn
        │
        v
Stage 1: Regex scan (30 compiled patterns, <1ms)
  - No matches → log "pass" at "stage": "regex" → exit 0
  - Matches found → proceed to Stage 2
        │
        v
Stage 2: ML classifier filters each flagged line
  - Model predicts BLOCK (1) or PASS (0) per line
  - Genuine hedging survives; false positives are filtered out
  - All flags removed → log "pass" at "stage": "model" → exit 0
  - If model fails to load/predict → falls back to regex-only
        │
        v
BLOCK: Output {"decision": "block", "reason": "..."} on stdout
  - Lists each flagged line with matched words
  - Instructs Claude to verify with tools or state why verification is impossible
        │
        v
Claude receives the block reason → uses tools to verify → responds again
  - Stop hook fires again on the new response
  - Turn boundary advances to the hook feedback (injected as type="user")
  - Old flagged text from before the block is NOT re-scanned
```

---

## The ML Model

### Architecture

```
Input: single line of text (the regex-flagged line)
        │
        v
┌─────────────────────────────────────────────┐
│           CombinedFeatures                  │
│                                             │
│  ┌──────────────┐  ┌────────────────────┐   │
│  │   TF-IDF     │  │  IntentFeatures    │   │
│  │  (1,3)-grams │  │  10 structural     │   │
│  │  5000 feats  │  │  signals           │   │
│  └──────┬───────┘  └────────┬───────────┘   │
│         └──────┬────────────┘               │
│                v                            │
│         hstack (sparse concat)              │
└────────────────┬────────────────────────────┘
                 v
     LogisticRegression (balanced weights)
                 │
                 v
         0 = PASS, 1 = BLOCK
```

### Intent Features (10 signals)

These structural features capture the **intent** behind the hedging words — the reason regex alone fails.

| # | Feature | Signal | Example |
|---|---------|--------|---------|
| 1 | `is_table_row` | Line starts with `\|` and contains another `\|` | `\| I think... \| BLOCK \|` → PASS |
| 2 | `is_code_comment` | Line starts with `#`, `//`, `/*`, or `` ``` `` | `# TODO: might need refactoring` → PASS |
| 3 | `trigger_only_in_backticks` | After removing backtick-enclosed spans, no hedging words remain | `` The model classifies `I think` as... `` → PASS |
| 4 | `has_meta_words` | Contains: flagged, detected, pattern, regex, hook, blocked, caught, triggered, matched, classified, filter, example | "The hook flagged 'probably'" → PASS |
| 5 | `has_type_keywords` | Contains: null, None, undefined, string, number, array, list, optional, type, return, throw, etc. | "The value could be None" → PASS |
| 6 | `starts_with_conditional` | Starts with: If, Unless, Without, When, Removing, With | "If the flag is set, would skip" → PASS |
| 7 | `trigger_in_quotes` | Hedging word appears inside unbalanced quote marks | "The word 'probably' was flagged" → PASS |
| 8 | `hedge_density` | Count of hedging words / 3.0, capped at 1.0 | Multiple hedges in one line → more likely genuine |
| 9 | `line_length` | len(text) / 200.0, capped at 1.0 | Longer prose lines → more likely genuine |
| 10 | `first_person_start` | Line starts with: I think, I believe, I suspect, I assume, I presume, I expect, I feel, I imagine, I suppose, My understanding | "I think the timeout is 30s" → strong BLOCK signal |

### Training Data

**215 intent-contrastive examples** at `~/.claude/assumption-guard-training-labeled.jsonl`.

The core principle: **same hedging words, different intents**. The model learns to distinguish:

| BLOCK intent | PASS intent | Same trigger word |
|---|---|---|
| "I think the timeout is 30 seconds" | "The hook flagged 'I think the timeout is 30s'" | "I think" |
| "This is probably a race condition" | `\| Probably a race condition \| BLOCK \|` | "probably" |
| "The value is likely null" | "The return value could be None or a string" | "could be" / "likely" |
| "The implementation is probably slow" | "The implementation is fairly straightforward" | "fairly" / "probably" |

**Intent distribution:**

| Intent | Count | Decision | Description |
|--------|-------|----------|-------------|
| `assert_unverified` | 70 | BLOCK | Speaker states a factual claim without verification |
| `reference_language` | 43 | PASS | Speaker discusses/quotes/mentions hedging words |
| `compare` | 43 | PASS | Idiomatic comparisons ("rather than") and degree words ("fairly simple") |
| `describe_type` | 20 | PASS | Code type/state analysis ("could be None or string") |
| `reason_conditionally` | 15 | PASS | Logical if-then reasoning ("If X, then Y could happen") |
| `acknowledge_limit` | 12 | PASS | Honest uncertainty ("Cannot verify without the logs") |
| `code_content` | 12 | PASS | Hedging words inside code comments or blocks |

### Performance

| Metric | Value |
|--------|-------|
| 5-fold CV F1 (macro) | 0.864 |
| Critical test cases | 16/16 (100%) |
| Model file size | 156KB |
| Inference time | <1ms per line |
| Training time | <100ms |
| Dependencies | scikit-learn, numpy, scipy |

### How to Retrain

When the log shows misclassifications (genuine hedging in the `filtered` field, or false blocks in the `flags` field):

1. **Add examples** to `~/.claude/assumption-guard-training-labeled.jsonl`:
   ```json
   {"sentence": "the new example text", "intent": "assert_unverified", "block": true}
   ```

2. **Run the training script** (recreate from the pattern below):
   ```python
   import json, os, pickle
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   # Import CombinedFeatures from the hook script or redefine here

   data = [json.loads(l) for l in open(TRAINING_PATH)]
   sentences = [d["sentence"] for d in data]
   labels = [1 if d["block"] else 0 for d in data]

   pipe = Pipeline([
       ("features", CombinedFeatures()),
       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
   ])
   pipe.fit(sentences, labels)

   with open(MODEL_PATH, "wb") as f:
       pickle.dump(pipe, f)
   ```

3. The hook picks up the new model on next invocation (no restart needed).

---

## Regex Patterns (Stage 1)

8 categories, 22 compiled patterns, ~150+ hedging terms:

| Category | Patterns | Examples |
|----------|----------|---------|
| 1. Epistemic modals | `likely`, `probably`, `possibly`, `perhaps`, `maybe`, `conceivably`; `might/may/could` + verb | "The bug is probably here" |
| 2. Shields | `I think`, `I believe`, `I suppose`, `I imagine`, `I suspect`, `I expect`; `it seems`, `it appears`, `it looks like`, `it sounds like`; `seemingly`, `apparently` | "I think the config is wrong" |
| 3. Assumption markers | `I assume`, `I presume`, `presumably`, `assuming that` | "I assume the API returns JSON" |
| 4. Disclaimers | `I'm not sure`, `I could be wrong`, `off the top of my head`, `I'm fairly confident`, `I'm pretty sure` | "I'm not sure about the encoding" |
| 5. Attribution hedges | `based on my understanding`, `from what I recall`, `as far as I can tell`, `if memory serves` | "From what I recall, the limit is 1000" |
| 6. Conditional hedges | `if I'm not mistaken`, `unless I'm wrong`, `correct me if I'm wrong` | "If I'm not mistaken, it uses OAuth" |
| 7. Approximators | `approximately/roughly/around/about` + digit; `somewhat`, `fairly`, `rather`, `relatively`, `sort of`, `kind of` | "The file has about 500 lines" |
| 8. AI-specific | `if I recall correctly`, `as of my last update`, `based on my training`, `to my knowledge` | "To my knowledge, the feature is supported" |

All patterns use `\b` word boundaries and `re.IGNORECASE`.

---

## Log Format

Every invocation writes one JSONL line to `~/.claude/assumption-guard.log.jsonl`:

### Clean pass (regex found nothing):
```json
{
  "session": "abc123",
  "project": "my-project",
  "retry": false,
  "text_blocks": 3,
  "text_chars": 1450,
  "result": "pass",
  "stage": "regex",
  "flag_count": 0,
  "flags": [],
  "ts": "2026-03-17T10:30:00.000Z"
}
```

### Model-filtered pass (regex found flags, model filtered all out):
```json
{
  "result": "pass",
  "stage": "model",
  "regex_flags": 2,
  "model_flags": 0,
  "filtered_out": 2,
  "flags": [],
  "filtered": [
    {"line": "Use hashmap rather than list", "words": ["rather"]},
    {"line": "The return value could be None", "words": ["could be"]}
  ]
}
```

### Block (genuine hedging confirmed):
```json
{
  "result": "block",
  "stage": "model",
  "regex_flags": 1,
  "model_flags": 1,
  "filtered_out": 0,
  "flags": [
    {"line": "I think the timeout is 30 seconds", "words": ["I think"]}
  ],
  "filtered": []
}
```

### Error (hook failed, exited 0 fail-open):
```json
{
  "result": "error",
  "error": "ValueError: unexpected content format"
}
```

### Key fields for refinement:

- **`filtered`** — Lines the model let pass. Review these for genuine hedging that slipped through. Add any misses to training data.
- **`flags`** — Lines that triggered a block. Review these for false blocks. Add any false positives to training data as PASS examples.
- **`regex_flags` vs `model_flags`** — The difference is the model's value-add. A high `filtered_out` count means the model is saving Claude from many false blocks.
- **`retry`** — `true` means this was a re-check after a previous block in the same turn.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| stdin parse fails | Exit 0 (allow) |
| transcript_path missing/empty | Exit 0 (allow) |
| transcript file doesn't exist | Exit 0 (allow) |
| JSONL line parse fails | Skip that line, continue |
| Model pickle fails to load | Fall back to regex-only |
| `model.predict()` raises exception | Fall back to regex-only for entire batch |
| Any exception in main pipeline | Log error entry, exit 0 (allow) |
| Log file write fails | Silently continue (OSError caught) |

The hook is **fail-open** — any error results in exit 0 (allow the response through). Hedging protection is never worth crashing Claude Code.

---

## Turn Boundary Behavior

The transcript is a JSONL file where each line has a top-level `type` field:
- `"user"` — user-typed prompts and tool results
- `"assistant"` — Claude's text, thinking, and tool calls
- `"progress"`, `"system"`, `"file-history-snapshot"` — metadata (ignored)

**Turn boundary detection:**
1. Walk entries backwards
2. Find the last entry where `type == "user"` AND `message.content` is a string (user-typed) or a list containing text blocks (not only tool_result blocks)
3. All assistant text entries from that point onward belong to the current turn

**After a block:**
- Hook feedback is injected as `type: "user"` entries with string content
- `find_last_user_text_index` finds these as the new turn boundary
- Old flagged text from before the block is excluded from re-scanning
- No infinite loop — the turn boundary advances past the blocked text

---

## Dependencies

**Runtime (required on PATH):**
- Python 3.8+
- `scikit-learn` (tested with 1.2.2)
- `numpy`
- `scipy`

**No external API calls. No network access. Fully offline.**

---

## Known Limitations

1. **The model has 215 training examples.** Edge cases not represented in the training data will fall back to regex behavior. The log captures these for ongoing refinement.

2. **Failure Mode 2 (confident confabulation) is not addressed.** The hook catches hedging language — when Claude signals uncertainty. It does not catch when Claude states false claims with full confidence and zero tool verification. This would require a claims-to-tools ratio detector (described in the original design spec).

3. **Large transcripts read fully into memory.** A 200MB transcript parses in ~1.3s (verified). As sessions grow beyond this, parsing time will approach the 10s timeout. A future optimization could read the file backwards to find the turn boundary first.

4. **The model cannot be retrained in-place.** Retraining requires a Python script with the CombinedFeatures/IntentFeatures classes defined (either imported from the hook script or redefined). The training data JSONL and the pickle must use matching class definitions.

---

## Origin

Designed and built on 2026-03-17 in a single Claude Code session. The project evolved through several iterations:

1. **v1 (regex-only)** — Strict mode, flagged all hedging regardless of context. Worked mechanically but had a 60% false positive rate on behavioral tests. Key false positives: "rather than" (idiom), "could be None" (type analysis), "fairly straightforward" (degree word).

2. **v1.5 (regex + logging)** — Added JSONL logging to capture every invocation for refinement data.

3. **v2 (regex + ML classifier)** — Added a trained scikit-learn model as a second stage. Initial model used TF-IDF only (417KB, 0.807 CV F1). Solved false positives but had a critical bug: custom sklearn classes were not defined in the hook script, so pickle deserialization failed silently and the model never actually ran.

4. **v3 (intent-trained model)** — Rebuilt training data around intent-contrastive pairs. Added 10 structural IntentFeatures. Reduced model to 156KB, improved CV F1 to 0.864, achieved 16/16 on critical cases. Fixed the pickle bug by defining classes in the hook script. Added exception handling around model.predict, top-level try/except, string-content handling, and word-boundary fix on approximator pattern.

# Assumption Guard

**Stop Claude from guessing. Make it verify.**

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) hook that catches when Claude makes unverified claims using hedging language — "I think", "probably", "likely" — and blocks the response until Claude goes back, uses its tools, and actually checks.

## The Problem

Claude is great at using tools (reading files, searching code, running commands) to verify facts. But sometimes it takes a shortcut and guesses instead:

> "I think the timeout is 30 seconds"
>
> "The database probably uses PostgreSQL"
>
> "It seems like the cache is stale"

Every one of these could be verified with a quick `Read` or `Grep`. But Claude chose to guess, and that guess might be wrong.

## What This Does

When Claude finishes responding, the hook scans the response for hedging language. If it finds genuine unverified claims, it blocks the response and tells Claude:

```
Your response contains unverified assumptions:

  1. "I think the timeout is 30 seconds"
     Flagged: "I think"

Verify EACH flagged claim using your tools (Read, Grep, Bash).
Replace assumption language with verified facts, or state
explicitly what you checked and why it cannot be confirmed.
Do NOT simply rephrase — actually verify.
```

Claude then uses its tools to check the facts, responds with verified information, and the hook lets it through.

## Real Example

Before the hook:
> "No text to verify — could be Arafta but can't confirm"

After the hook blocked and Claude re-verified:
> "Empty caption videos (234 total): 49 confirmed Arafta by author handle, 15 confirmed by character name, 170 unverifiable from metadata alone"

One block turned a vague guess into a verified breakdown with specific numbers backed by actual data queries.

## Smart False-Positive Filtering

The hook doesn't just match words — it understands **intent**. A trained ML classifier filters out false positives so Claude doesn't get blocked for:

| What Claude wrote | Regex says | Model says | Why |
|---|---|---|---|
| "Use map() rather than forEach()" | BLOCK | **PASS** | English idiom, not hedging |
| "The return value could be None" | BLOCK | **PASS** | Type analysis, not a guess |
| "The hook caught 'probably'" | BLOCK | **PASS** | Quoting the word, not using it |
| "If the flag is set, this skips" | BLOCK | **PASS** | Conditional reasoning |
| "# TODO: might need refactoring" | BLOCK | **PASS** | Code comment |
| "I think the timeout is 30 seconds" | BLOCK | **BLOCK** | Genuine unverified claim |

## Installation

### Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- Python 3.8+
- scikit-learn, numpy, scipy (`pip install scikit-learn numpy scipy`)

### Setup

1. **Copy the hook script:**
   ```bash
   mkdir -p ~/.claude/hooks
   cp hook/assumption-guard.py ~/.claude/hooks/
   ```

2. **Copy the trained model:**
   ```bash
   cp model/assumption-guard-model.pkl ~/.claude/
   ```

3. **Add the hook to your Claude Code settings** (`~/.claude/settings.json`):
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
   If you already have a `settings.json`, merge the `Stop` hook into your existing `hooks` section.

4. **Start a new Claude Code session.** The hook activates automatically.

### Verify It Works

Ask Claude something it should look up but might guess about:

> "What port does this app run on?"

If Claude responds with "I think it's port 3000" instead of checking the config file, the hook will block and force verification.

## How It Works

Two-stage detection, both running locally:

```
Claude responds
    │
    v
Stage 1: Regex scan (<1ms)
    30 patterns catch hedging words
    No matches → response goes through instantly
    │
    v
Stage 2: ML classifier (<1ms)
    Filters false positives using intent analysis
    Quotations, idioms, type analysis → PASS
    Genuine unverified claims → BLOCK
```

**No API calls. No cloud. Fully offline.** The ML model is a 156KB scikit-learn classifier that runs in under 1 millisecond.

## Logging

Every invocation is logged to `~/.claude/assumption-guard.log.jsonl`:

```json
{
  "result": "pass",
  "stage": "model",
  "regex_flags": 2,
  "model_flags": 0,
  "filtered_out": 2
}
```

Check the log after a few days to see how the hook is performing. The `filtered` field shows what the model saved you from (false positives that would have blocked with regex alone).

## Retraining the Model

The model can be improved with your own data:

1. Review the log for misclassifications
2. Add corrected examples to `training/assumption-guard-training-labeled.jsonl`:
   ```json
   {"sentence": "your example", "intent": "assert_unverified", "block": true}
   ```
3. Retrain (see [Model Training Documentation](docs/model-training.md) for the full script)
4. Copy the new model to `~/.claude/assumption-guard-model.pkl`

The hook picks up the new model immediately — no restart needed.

## Project Structure

```
hook/
├── assumption-guard.py          # The hook script (install to ~/.claude/hooks/)
└── settings-snippet.json        # Settings to merge into ~/.claude/settings.json
model/
└── assumption-guard-model.pkl   # Trained classifier (install to ~/.claude/)
training/
└── assumption-guard-training-labeled.jsonl  # 215 labeled training examples
docs/
├── assumption-guard-system.md   # Complete system documentation
├── model-training.md            # How the model was built
└── archive/
    ├── v1-design-spec.md        # Original regex-only design
    └── v1-implementation-plan.md
```

## FAQ

**Does this slow down Claude?**
No. Most responses have no hedging words — the regex passes them in microseconds. When hedging is found, the ML model adds <1ms. The 10-second timeout is never approached.

**What if the model is wrong?**
The hook is fail-open. If the model file is missing, corrupted, or crashes, the hook falls back to regex-only mode. If the entire hook crashes, Claude continues normally.

**Can I disable it temporarily?**
Remove or comment out the Stop hook entry in `~/.claude/settings.json`. Or rename the script so the command can't find it.

**Does it work with all Claude models?**
It works with any Claude Code session. The hook reads the transcript format, which is the same regardless of which Claude model is running.

**Will it cause infinite loops?**
No. When the hook blocks a response, the feedback is injected as a new user message in the transcript. The hook only scans text *after* the most recent user message, so old flagged text is never re-scanned.

---

## Technical Details

### Regex Patterns

8 categories, 22 compiled patterns covering 150+ hedging terms:

1. **Epistemic modals** — likely, probably, possibly, perhaps, maybe, conceivably; might/may/could + verb
2. **Shields** — I think, I believe, I suppose, I imagine, I suspect, I expect; it seems, it appears, it looks like
3. **Assumption markers** — I assume, I presume, presumably, assuming that
4. **Disclaimers** — I'm not sure, I could be wrong, off the top of my head, I'm fairly confident
5. **Attribution hedges** — based on my understanding, from what I recall, as far as I can tell
6. **Conditional hedges** — if I'm not mistaken, unless I'm wrong, correct me if I'm wrong
7. **Approximators** — approximately/roughly/around/about + number; somewhat, fairly, rather, relatively, sort of, kind of
8. **AI-specific** — if I recall correctly, as of my last update, based on my training, to my knowledge

### ML Classifier

**Architecture:** TF-IDF (5000 features, 1-3 grams) + 10 intent-signal features → LogisticRegression

**Intent features:**
1. Is this a markdown table row?
2. Is this a code comment?
3. Is the hedging word only inside backticks?
4. Contains meta-discussion words (flagged, detected, hook, pattern)?
5. Contains type keywords (null, None, string, array)?
6. Starts with conditional (If, Unless, Without)?
7. Is the hedging word inside quote marks?
8. Hedging word density (multiple hedges = more likely genuine)
9. Line length (longer = more likely genuine prose)
10. Starts with first-person assertion (I think, I believe)?

**Training data:** 215 intent-contrastive examples — same hedging words paired with different intents (genuine hedging vs. quotation/idiom/type analysis).

**Performance:** 5-fold CV F1: 0.864 | 16/16 critical cases | 156KB model | <1ms inference

### Transcript Parsing

The hook reads the Claude Code session transcript (JSONL) and extracts assistant text from the current turn only. The turn boundary is the last `type: "user"` entry with string content (not tool results). This ensures hedging in previous turns or before hook feedback is never re-scanned.

### Error Handling

Every failure mode exits with code 0 (fail-open): stdin parse error, missing transcript, model load failure, model prediction error, any uncaught exception. Errors are logged to the JSONL log file. The hook never prevents Claude from responding due to its own bugs.

---

## License

MIT

## Credits

Built with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) in a single session on 2026-03-17.

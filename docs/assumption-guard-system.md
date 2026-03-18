# Assumption Guard — System Documentation

## Purpose

Assumption Guard is a Claude Code Stop hook that blocks responses containing unverified uncertainty. In v1.1, the policy is deliberately strict:

- unverified factual guesses block
- speculative recommendations block
- explicit “I have not checked / cannot confirm yet” limitations block

The only uncertain statements that should pass are ones that already describe a concrete verification attempt and explain why the available evidence is still insufficient.

## Execution Flow

1. Claude finishes a turn.
2. Claude Code invokes `assumption-guard.py` as a Stop hook.
3. The hook reads the transcript JSONL and extracts assistant text from the current turn only.
4. Stage 1 regex scans for uncertain or unchecked language.
5. If regex finds matches, Stage 2 optionally loads the sklearn model and filters out false positives.
6. If confirmed flags remain, the hook prints:

```json
{"decision":"block","reason":"..."}
```

7. Claude gets the block reason, re-checks with tools, and responds again.

The turn boundary is the most recent user-authored text message, so old flagged text is not re-scanned after the hook injects feedback.

## Runtime Modes

### `regex`

- No regex matches.
- Fast pass.
- No ML load attempt needed.

### `model`

- Regex found matches.
- ML dependencies and the trained pickle loaded successfully.
- Model filtered the flagged lines.

### `regex_only`

- Regex found matches.
- ML could not be used.
- Hook falls back to regex-only blocking and logs why.

Fallback reasons:

- `ml_import_error`
- `model_missing`
- `pickle_load_error`
- `predict_error`

### `hook_error`

- Unexpected runtime failure inside the hook.
- Still exits `0` to avoid breaking Claude Code.

## Detection Policy

### BLOCK

- `I think the timeout is 30 seconds`
- `Maybe rename this helper to be clearer`
- `This could be simplified by extracting a helper`
- `We should probably add a unit test here`
- `I'm not sure because I have not checked the config yet`

### PASS

- `The hook flagged "probably" in the response`
- `The return value could be None or a string`
- `When the cache is cold, this may take longer`
- `Use map rather than forEach here`
- `I checked package.json and requirements.txt and cannot confirm the config value from this repo.`

## Regex Layer

The hook currently uses **27 compiled regex patterns** across these categories:

1. Epistemic modals
2. Shields
3. Assumption markers
4. Disclaimers
5. Attribution hedges
6. Conditional hedges
7. Approximators
8. AI-specific patterns
9. Explicitly unchecked / not-yet-verified language

Regex is stdlib-only and always available.

## ML Layer

The optional ML layer lives in `hook/assumption_guard_model.py`.

Architecture:

- TF-IDF vectorizer: 1-3 grams, 5000 features
- Intent features: 14 structural signals
- Classifier: `LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)`

The current intent features include:

1. Markdown table row
2. Code comment
3. Trigger only inside backticks
4. Meta words present
5. Type keywords present
6. Starts with a conditional
7. Trigger appears inside quotes
8. Hedge density
9. Line length
10. First-person epistemic start
11. Improvement / review words
12. Verification evidence present
13. Unchecked limitation language present
14. Starts with uncertainty admission

## Logging

Every invocation writes a JSON object to `~/.claude/assumption-guard.log.jsonl`.

Important fields:

- `result`
- `stage`
- `ml_available`
- `fallback_reason`
- `error_stage`
- `error_detail`
- `regex_flags`
- `model_flags`
- `filtered_out`
- `flags`
- `filtered`

Example regex-only fallback:

```json
{
  "result": "block",
  "stage": "regex_only",
  "ml_available": false,
  "fallback_reason": "pickle_load_error",
  "error_stage": "model_load"
}
```

## Files

Runtime files:

- `hook/assumption-guard.py`
- `hook/assumption_guard_model.py`
- `hook/settings-snippet.json`

Training and verification files:

- `training/assumption-guard-training-labeled.jsonl`
- `training/assumption-guard-regression-cases.json`
- `training/train_assumption_guard.py`
- `tests/test_assumption_guard.py`

Generated artifacts:

- `model/assumption-guard-model.pkl`
- `model/assumption-guard-metrics.json`

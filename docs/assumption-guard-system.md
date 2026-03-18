# Assumption Guard — System Documentation

## Purpose

Assumption Guard is a Claude Code Stop hook that blocks uncertain assistant clauses until Claude verifies them with tools or rewrites them as evidence-backed conclusions.

The v2 policy treats these as BLOCK:

- `assert_unverified`
- `recommend_unverified`
- `capability_promise_unverified`
- `unchecked_limitation`

It treats these as PASS:

- `verification_narration`
- `reference_language`
- `verified_limitation`
- `dependency_gap_grounded`
- `describe_type`
- `reason_conditionally`
- `code_content`
- `idiomatic_compare`
- `recommend_supported`

## Execution Flow

1. Claude finishes a turn.
2. Claude Code invokes `hook/assumption-guard.py`.
3. The hook reads the transcript JSONL and extracts only the current assistant turn.
4. The assistant text is split into clauses.
5. Each clause goes through the v2 runtime pipeline:
   - hard-pass suppressors
   - hard-block rules
   - stdlib regex prefilter
   - optional ONNX multiclass classification
6. If any clause remains blocked, the hook prints:

```json
{"decision":"block","reason":"..."}
```

7. Claude gets the block reason and must verify, justify, or rewrite before the turn can finish.

The hook always exits `0`, even on fallback or internal errors, so it does not crash Claude Code.

## Runtime Pipeline

### 1. Clause Splitting

The runtime unit is a clause, not a whole message.

`hook/assumption-guard.py` splits on:

- sentence boundaries
- `;`
- em/en dashes
- `, so ...`

This keeps a recommendation or uncertainty phrase from hiding inside a longer paragraph.

### 2. Hard-Pass Suppressors

These skip ML entirely:

- fenced code
- inline-code-only/code-comment clauses
- quoted examples
- meta/report language such as `the hook flagged`, `regression case`, `classifier`, `label`, `metrics report`
- type/shape clauses such as `The return value could be None or a string`
- conditional reasoning such as `If the token expires, the middleware would return 401`
- idiomatic comparisons such as `Use map rather than forEach here`
- explicit verification narration such as `Let me verify whether ...`
- evidence-backed limitations and recommendations
- grounded dependency gaps such as `I confirmed X. But without Y, I can't Z.`

### 3. Hard-Block Rules

These also skip ML and block immediately:

- `I need to verify ...`
- `I haven't checked ...`
- `I can't confirm ...`
- `off the top of my head`
- `to know for sure`
- `requires running / checking / benchmarking`

### 4. Regex Prefilter

Regex is stdlib-only and always available. It catches the candidate clauses that should be examined further. The current pattern groups cover:

1. epistemic modals and approximators
2. uncertainty intros
3. speculative or unsupported recommendation language
4. unsupported capability promises such as `I can check ... and remove ...`
5. explicit unchecked/needs-verification language

### 5. ONNX Classifier

The ONNX model runs only on ambiguous clauses that survive the earlier filters.

Runtime artifacts:

- `model/assumption-guard-v2.onnx`
- `model/assumption-guard-v2-tokenizer.json`
- `model/assumption-guard-v2-meta.json`

The classifier produces 13 intent probabilities. Runtime sums the four BLOCK-class probabilities into `p_block` and blocks when `p_block >= threshold`.

The current metadata threshold is `0.20`.

## Runtime Modes

### `onnx`

- v2 assets loaded successfully
- ambiguous clauses were classified by the ONNX model

### `regex`

- no ONNX model was used
- only regex/hard-rule detection was active

### `hook_error`

- unexpected runtime failure inside the hook
- hook still exits `0`

## Fail-Open Behavior

The hook stays operational even when the ML runtime is unavailable.

Fallback reasons currently emitted in logs:

- `ml_import_error`
- `meta_missing`
- `tokenizer_missing`
- `model_missing`
- `asset_load_error`
- `model_load_error`
- `predict_error`

In those cases, Assumption Guard falls back to regex-only decisions and records the exact reason.

## Logging

Each invocation appends one JSON line to `~/.claude/assumption-guard.log.jsonl`.

Important fields:

- `result`
- `stage`
- `backend`
- `ml_available`
- `fallback_reason`
- `error_stage`
- `error_detail`
- `predicted_intent`
- `p_block`
- `threshold`
- `flags`
- `filtered`

Example fallback record:

```json
{
  "result": "block",
  "stage": "regex_only",
  "backend": "regex",
  "ml_available": false,
  "fallback_reason": "ml_import_error",
  "error_stage": "ml_import"
}
```

## Learning Queue

Sprint 1 adds an always-on local learning loop around the live hook.

Default local state:

- `~/.claude/assumption-guard-state/learning-queue.jsonl`
- `~/.claude/assumption-guard-state/queue/YYYY-MM-DD.jsonl`
- `~/.claude/assumption-guard-state/training-overlay.jsonl`
- `~/.claude/assumption-guard-state/regression-overlay.jsonl`
- `~/.claude/assumption-guard-state/replay-overlay.jsonl`
- `~/.claude/assumption-guard-state/reviewed-claude.jsonl`
- `~/.claude/assumption-guard-state/current-model-report.json`

The hook now queues:

- blocked ONNX clauses
- low-margin ONNX passes
- judgement-heavy heuristic rescues
- mixed evidence / capability-gap clauses
- a deterministic sample of safe passes

Queue rows are sanitized before they are written.

## Files

Runtime:

- `hook/assumption-guard.py`
- `hook/settings-snippet.json`

Training and evaluation:

- `training/assumption-guard-training-labeled.jsonl`
- `training/assumption-guard-regression-cases.json`
- `training/assumption-guard-replay-cases.json`
- `training/build_review_batch.py`
- `training/mine_transcripts_v2.py`
- `training/promote_reviewed_examples.py`
- `training/replay_eval_v2.py`
- `training/review_with_claude.py`
- `training/run_learning_cycle.py`
- `training/train_v2.py`

Generated v2 artifacts:

- `model/assumption-guard-v2.onnx`
- `model/assumption-guard-v2-tokenizer.json`
- `model/assumption-guard-v2-meta.json`
- `model/assumption-guard-v2-report.json`

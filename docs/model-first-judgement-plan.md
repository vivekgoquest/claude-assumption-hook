# Plan: Model-First Judgement Upgrade

**Generated**: 2026-03-18
**Estimated Complexity**: High

## Overview

Shift Assumption Guard from a heuristic-heavy runtime toward a model-first decision boundary. The runtime should keep only mechanical, low-ambiguity rules such as code/quote suppression, explicit fail-open fallback, and hook I/O. Nuanced distinctions like verification narration versus unsupported capability promises should move into the training data and model outputs.

The immediate target is not a brand new runtime architecture. It is a cleaner version of the current ONNX clause classifier with stronger transcript-derived supervision, a richer intent taxonomy, and a smaller hard-rule surface.

## Goals

- Keep the Claude Stop-hook contract unchanged.
- Reduce false positives caused by over-broad runtime heuristics.
- Encode more of the judgement boundary in the model instead of in regex rules.
- Preserve fast local CPU inference and fail-open behavior.

## Non-Goals

- Replacing local inference with a hosted model.
- Removing every deterministic rule. Mechanical suppressors and safety fallbacks should remain in code.
- Changing the user-facing block payload or Claude hook registration flow.

## Principles

1. Keep code for mechanics, not judgement.
   - Code should decide things like "this is fenced code", "ML assets are missing", or "this clause is quoted meta-language".
   - The model should decide things like "is this a capability over-promise or legitimate verification narration?"

2. Retrain from real transcripts, not intuition.
   - Mine from `/Users/vivek/.claude/projects`.
   - Commit only sanitized, derived clauses.

3. Judge at clause level.
   - Keep the current clause-level runtime unit and improve the training corpus around actual clause boundaries.

4. Treat low-margin cases as labeling opportunities.
   - Every live miss should become either a new regression case, a replay case, or a relabeled training example.

## Prerequisites

- Access to local Claude transcript history at `/Users/vivek/.claude/projects`
- Existing v2 training pipeline:
  - `training/mine_transcripts_v2.py`
  - `training/train_v2.py`
  - `training/replay_eval_v2.py`
- Current runtime and assets:
  - `hook/assumption-guard.py`
  - `model/assumption-guard-v2.onnx`
  - `model/assumption-guard-v2-tokenizer.json`
  - `model/assumption-guard-v2-meta.json`

## Sprint 1: Redraw The Label Boundary

**Goal**: Make the policy boundary explicit in the dataset so the model can learn the distinctions we currently patch in code.

**Demo/Validation**:
- Updated taxonomy documented in `docs/model-training.md`
- Expanded labeled corpus with new transcript-derived examples
- Updated regression fixture covering narration, capability promises, and grounded limitations

### Task 1.1: Add new intent labels
- **Location**: `training/assumption-guard-training-labeled.jsonl`, `docs/model-training.md`
- **Description**: Add or formalize these intents:
  - `verification_narration` -> PASS
  - `capability_promise_unverified` -> BLOCK
  - `dependency_gap_grounded` -> PASS
- **Dependencies**: none
- **Acceptance Criteria**:
  - New labels are defined with examples and decision rules
  - Existing overlapping examples are relabeled where needed
- **Validation**:
  - Schema sweep confirms all rows still have valid labels

### Task 1.2: Mine transcript clauses for the weak zones
- **Location**: `training/mine_transcripts_v2.py`
- **Description**: Add targeted mining for:
  - `let me verify`, `i'll check`, `i'm going to confirm`
  - `i can ...` / `i can check ... and remove ...`
  - `without X, I can't Y`
  - `i confirmed X. but without Y, I can't Z`
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Mining script emits candidate rows for each target family
  - Output remains sanitized and reproducible
- **Validation**:
  - Run mining script and inspect a sample of mined rows

### Task 1.3: Rebuild the regression and replay fixtures
- **Location**: `training/assumption-guard-regression-cases.json`, `training/assumption-guard-replay-cases.json`
- **Description**: Expand fixtures around the new boundary so every live miss has an explicit expected outcome.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Add at least 20 new cases, with emphasis on narration, supported limitations, and unsupported capability claims
- **Validation**:
  - `python3.11 training/replay_eval_v2.py ...` matches all cases

## Sprint 2: Retrain The Model With More Judgement

**Goal**: Make the ONNX classifier carry the nuanced boundary so the runtime can stop making conversational distinctions itself.

**Demo/Validation**:
- Fresh ONNX model and tokenizer artifacts
- Updated meta/report files
- Candidate comparison with model-first acceptance notes

### Task 2.1: Retrain all candidates from scratch
- **Location**: `training/train_v2.py`, `model/`
- **Description**: Retrain the baseline SVM, MiniLM candidate, and DeBERTa-small candidate from the revised corpus.
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Fresh model artifacts are generated from the updated dataset
  - Candidate comparison is recorded in `model/assumption-guard-v2-report.json`
- **Validation**:
  - `python3.11 training/train_v2.py`

### Task 2.2: Optimize for judgement-heavy mistakes
- **Location**: `training/train_v2.py`
- **Description**: Adjust class weighting, threshold search, and report views so the model is explicitly optimized against:
  - verification narration false positives
  - grounded limitation false positives
  - capability-promise false negatives
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Report includes targeted counts for these three error families
  - Threshold is selected with these families visible, not hidden inside aggregate metrics
- **Validation**:
  - Inspect report sections and threshold search output

### Task 2.3: Prefer model output over heuristic overrides
- **Location**: `hook/assumption-guard.py`
- **Description**: After retraining, remove or narrow any runtime rule that exists only because the previous model could not carry the boundary.
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Mechanical runtime rules remain
  - Judgement-heavy runtime rules are reduced or deleted if the retrained model now handles them reliably
- **Validation**:
  - Diff review shows fewer judgement rules in runtime
  - Replay and live probes still pass

## Sprint 3: Tighten Evaluation Around Real-World Judgement

**Goal**: Prevent the next round of conversational false positives from surprising us in live use.

**Demo/Validation**:
- Expanded report artifact
- Low-margin review workflow
- Clear go/no-go criteria for shrinking code heuristics

### Task 3.1: Add disagreement-focused reporting
- **Location**: `training/train_v2.py`, `model/assumption-guard-v2-report.json`
- **Description**: Add report sections for:
  - top false positives by intent
  - top false negatives by intent
  - low-margin examples near threshold
  - live-miss families promoted from logs
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Report highlights conversational boundary cases explicitly
- **Validation**:
  - Generated report includes the new sections

### Task 3.2: Build a log-to-dataset review loop
- **Location**: `training/mine_transcripts_v2.py`, `README.md`
- **Description**: Define a repeatable process for turning new lines from `~/.claude/assumption-guard.log.jsonl` into:
  - regression fixtures
  - replay fixtures
  - labeled dataset rows
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Documented workflow exists and is reproducible
- **Validation**:
  - Follow the workflow once on a fresh sample

## Sprint 4: Reduce Runtime Heuristics To The Minimum Set

**Goal**: Leave code responsible only for deterministic mechanics and make the model the primary judge.

**Demo/Validation**:
- Smaller hard-pass/hard-block section in runtime
- Same or better replay accuracy
- No new live false-positive cluster

### Task 4.1: Lock the allowed code-side rules
- **Location**: `hook/assumption-guard.py`, `docs/assumption-guard-system.md`
- **Description**: Keep only these code-side categories:
  - code and quote suppression
  - meta/report suppression
  - clause splitting
  - fail-open/runtime asset handling
  - optionally one or two truly invariant block phrases if they remain universally correct
- **Dependencies**: Sprint 3 complete
- **Acceptance Criteria**:
  - Runtime no longer contains several narrow conversational judgement regexes
- **Validation**:
  - Rule inventory review before/after

### Task 4.2: Re-run live soak on the model-first version
- **Location**: `~/.claude/hooks/assumption-guard.py`, `~/.claude/assumption-guard.log.jsonl`
- **Description**: Install the reduced-heuristic runtime and review fresh live blocks for a few days.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - No obvious regression cluster appears in live usage
  - New misses are explainable and promotable into data
- **Validation**:
  - Manual review of recent live log entries

## Testing Strategy

- Keep `python3 -m unittest discover -s tests -v` as the runtime gate.
- Keep `python3.11 training/replay_eval_v2.py ...` as the fast decision-boundary gate.
- Use `python3.11 training/train_v2.py` to regenerate:
  - ONNX model
  - tokenizer
  - meta
  - report
- Add a required pre-ship check that compares:
  - current runtime heuristic inventory
  - current model/report regression results

## Acceptance Criteria

The model-first upgrade is ready when all of these are true:

- Regression fixture matches 100%
- Replay recall stays at or above the current bar
- Verification narration is learned by the model, not just rescued by a rule
- Grounded limitations are learned by the model, not just rescued by a rule
- Unsupported capability promises still block
- The runtime rule surface is smaller than it is today

## Risks And Gotchas

- Transcript mining can overfit on one writing style if the sample is too local.
  - Mitigation: keep synthetic counterexamples for gaps and dedupe aggressively.
- If we remove heuristics too early, live false positives may drop but false negatives may rise.
  - Mitigation: remove only after retrained replay passes and live soak looks clean.
- New labels can drift if they are not defined tightly.
  - Mitigation: keep positive/negative examples beside each label in `docs/model-training.md`.

## Rollback Plan

- Keep the current committed ONNX assets available until the retrained model is accepted.
- If the retrained model underperforms, revert only the new model artifacts and keep the current runtime.
- If live false positives spike after reducing heuristics, restore the previous runtime rules and continue training from the expanded dataset.

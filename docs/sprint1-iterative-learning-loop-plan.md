# Plan: Sprint 1 Iterative Learning Loop

**Generated**: 2026-03-18
**Estimated Complexity**: High

## Overview

Execute Sprint 1 of the model-first judgement roadmap in a way that is iterative and baked into the hook's everyday behavior.

The key design choice is:

- the hook should automatically collect high-value learning examples every time it runs
- the hook should not retrain or hot-swap the model on every invocation
- promotion of new labels and new ONNX assets should stay gated behind replay, regression, and report checks

This gives us an always-on learning loop without turning the live hook into a self-mutating system.

## Problem Statement

Right now the hook already logs decisions to `~/.claude/assumption-guard.log.jsonl`, but that log is operational, not a true learning pipeline. It lacks an explicit review queue, bounded sampling strategy, candidate-label workflow, and promotion path from live miss -> reviewed example -> retrained model.

The next step is to make the hook itself feed that loop automatically.

## Desired Outcome

After Sprint 1:

- every hook invocation can emit curated learning candidates, not just operational logs
- difficult clauses are automatically queued for review with enough sanitized context to label them well
- transcript mining and live queue capture produce one consistent review schema
- reviewed cases can be promoted into:
  - `training/assumption-guard-training-labeled.jsonl`
  - `training/assumption-guard-regression-cases.json`
  - `training/assumption-guard-replay-cases.json`
- retraining remains explicit and gated, but the data collection loop is automatic

## Scope

### In Scope

- Hook-side capture of learning candidates
- New iterative review queue artifacts
- Expanded dataset taxonomy for the new judgement families
- A reproducible review -> relabel -> retrain flow
- Documentation and scripts for repeated iteration

### Out Of Scope

- Automatic retraining during a live hook run
- Automatic promotion of a new ONNX model without human review
- Changing the Claude Stop-hook payload or stdout contract

## Principles

1. Bake collection into runtime, not retraining.
   - The hook should always be collecting evidence.
   - The hook should never silently retrain itself during a normal Claude response.

2. Prefer model learning over code heuristics.
   - New runtime heuristics should be temporary bridges only.
   - The long-term destination is to move nuanced judgement into the model.

3. Review queue quality matters more than raw volume.
   - We want high-signal candidates, not every clause ever seen.

4. Every live miss should have a promotion path.
   - If a line was annoying, surprising, or clearly wrong, it should become data.

## Sprint 1A: Hook-Side Learning Capture

**Goal**: Make the live hook emit structured, high-value review candidates on every invocation.

**Demo/Validation**:
- Hook still blocks/passes exactly as before
- New review queue file is populated automatically
- Queue rows are deduped, sanitized, and bounded

### Task 1A.1: Add a dedicated learning queue artifact
- **Location**: `hook/assumption-guard.py`, `README.md`, `docs/assumption-guard-system.md`
- **Description**: Add a second JSONL artifact alongside the operational log:
  - `~/.claude/assumption-guard-learning-queue.jsonl`
- **Dependencies**: none
- **Acceptance Criteria**:
  - Hook appends learning-candidate rows without changing the current operational log
  - Queue path is configurable via env var
- **Validation**:
  - Run the hook locally and verify both files are written

### Task 1A.2: Define the learning queue schema
- **Location**: `hook/assumption-guard.py`, `docs/model-training.md`
- **Description**: Emit one row per candidate with fields such as:
  - `candidate_id`
  - `ts`
  - `text`
  - `sanitized_text`
  - `decision`
  - `intent`
  - `source`
  - `p_block`
  - `threshold`
  - `stage`
  - `matched_terms`
  - `previous_clause`
  - `next_clause`
  - `evidence_present`
  - `quoted_or_code`
  - `candidate_reason`
  - `review_status`
  - `session_hash`
  - `message_hash`
- **Dependencies**: Task 1A.1
- **Acceptance Criteria**:
  - Schema is documented and versioned
  - Rows contain enough context for labeling without exposing raw transcript secrets
- **Validation**:
  - Sample queue rows are readable and labelable by hand

### Task 1A.3: Capture only high-value candidates
- **Location**: `hook/assumption-guard.py`
- **Description**: Queue clauses only when they are likely to improve the model:
  - all blocked ONNX clauses
  - all near-threshold ONNX passes within a configurable margin
  - all clauses rescued by a judgement-heavy runtime rule
  - all mixed evidence/no-capability clauses
  - optionally a small random sample of clean passes for calibration
- **Dependencies**: Task 1A.2
- **Acceptance Criteria**:
  - Queue growth is bounded and signal-heavy
  - Trivial safe content is not flooding the queue
- **Validation**:
  - One day of normal use yields a manageable queue size and useful mix

### Task 1A.4: Add dedupe, rotation, and capture controls
- **Location**: `hook/assumption-guard.py`
- **Description**: Prevent queue bloat by adding:
  - text-hash dedupe
  - per-day file rotation or size cap
  - environment variables for capture mode, queue path, and low-margin window
- **Dependencies**: Task 1A.3
- **Acceptance Criteria**:
  - Repeated identical clauses do not endlessly duplicate
  - Queue remains safe to leave on for long periods
- **Validation**:
  - Repeated probe runs produce stable dedupe behavior

## Sprint 1B: Review And Promotion Workflow

**Goal**: Turn captured live candidates into reviewed training data with a reproducible path to regression and replay fixtures.

**Demo/Validation**:
- One command builds a review batch from the live queue
- Reviewed rows can be promoted into the committed corpus and fixtures

### Task 1B.1: Build a review-batch script
- **Location**: `training/build_review_batch.py`
- **Description**: Create a script that merges:
  - `~/.claude/assumption-guard-learning-queue.jsonl`
  - transcript-mined rows from `training/mine_transcripts_v2.py`
  - optional recent operational log disagreements
  into one sanitized review batch.
- **Dependencies**: Sprint 1A complete
- **Acceptance Criteria**:
  - Script outputs a deduped review file in the v2 row schema
  - Rows carry `review_status=needs_review`
- **Validation**:
  - Run script and inspect output size and content quality

### Task 1B.2: Add explicit review states
- **Location**: `training/assumption-guard-training-labeled.jsonl`, `docs/model-training.md`
- **Description**: Standardize statuses:
  - `needs_review`
  - `approved`
  - `rejected`
  - `promoted_to_regression`
  - `promoted_to_replay`
- **Dependencies**: Task 1B.1
- **Acceptance Criteria**:
  - Review state is consistent across mined rows and committed rows
- **Validation**:
  - Schema sweep across JSONL rows

### Task 1B.3: Add promotion helpers
- **Location**: `training/promote_reviewed_examples.py`
- **Description**: Create a script that takes reviewed rows and appends them to the right targets:
  - labeled corpus
  - regression fixture
  - replay fixture
- **Dependencies**: Task 1B.2
- **Acceptance Criteria**:
  - Promotion is reproducible and idempotent
  - Duplicate promoted rows are avoided
- **Validation**:
  - Promote a small batch twice and verify dedupe works

## Sprint 1C: Taxonomy Expansion For The New Judgement Boundary

**Goal**: Give the model explicit labels for the cases we currently distinguish with runtime judgement.

**Demo/Validation**:
- Dataset schema and docs reflect the new intent families
- Transcript-derived examples exist for each new family

### Task 1C.1: Add new intent labels
- **Location**: `training/assumption-guard-training-labeled.jsonl`, `docs/model-training.md`, `training/train_v2.py`
- **Description**: Add and document:
  - `verification_narration` -> PASS
  - `capability_promise_unverified` -> BLOCK
  - `dependency_gap_grounded` -> PASS
- **Dependencies**: Sprint 1B complete
- **Acceptance Criteria**:
  - Labels are supported end-to-end in data loading and training
- **Validation**:
  - Training script schema load succeeds with the new labels

### Task 1C.2: Relabel overlapping existing rows
- **Location**: `training/assumption-guard-training-labeled.jsonl`
- **Description**: Revisit older rows currently forced into:
  - `reference_language`
  - `verified_limitation`
  - `assert_unverified`
  where the new labels are a better semantic fit.
- **Dependencies**: Task 1C.1
- **Acceptance Criteria**:
  - Existing ambiguous rows are normalized to the new taxonomy
- **Validation**:
  - Dataset audit by intent counts and sample review

### Task 1C.3: Expand transcript mining for the new labels
- **Location**: `training/mine_transcripts_v2.py`
- **Description**: Add targeted mining patterns for:
  - `let me verify`, `i'll check`, `i am going to confirm`
  - `i can check ... and remove ...`
  - `without X, I can't Y`
  - `i confirmed X. but without Y, I can't Z`
- **Dependencies**: Task 1C.1
- **Acceptance Criteria**:
  - Mining script surfaces candidate rows for all new families
- **Validation**:
  - Run mining script and inspect the new class distribution

## Sprint 1D: Retrain And Compare

**Goal**: Retrain from the revised corpus and check whether the model can carry the new judgement boundary well enough to reduce runtime rules.

**Demo/Validation**:
- Fresh ONNX assets
- Updated report with targeted error-family metrics
- Clear recommendation on which runtime heuristics can be removed

### Task 1D.1: Add targeted evaluation buckets
- **Location**: `training/train_v2.py`, `model/assumption-guard-v2-report.json`
- **Description**: Report dedicated metrics for:
  - verification narration false positives
  - grounded limitation false positives
  - capability promise false negatives
- **Dependencies**: Sprint 1C complete
- **Acceptance Criteria**:
  - Report explicitly surfaces these families
- **Validation**:
  - Generated report contains dedicated sections or counts

### Task 1D.2: Retrain all candidates from scratch
- **Location**: `training/train_v2.py`, `model/`
- **Description**: Retrain baseline SVM, MiniLM, and DeBERTa from the revised labeled corpus.
- **Dependencies**: Task 1D.1
- **Acceptance Criteria**:
  - New ONNX artifacts are produced
  - Candidate comparison reflects the new taxonomy
- **Validation**:
  - `python3.11 training/train_v2.py`

### Task 1D.3: Decide which judgement rules can be removed
- **Location**: `hook/assumption-guard.py`
- **Description**: Compare the new report against current runtime heuristics and decide whether to reduce or delete:
  - verification narration rescue
  - grounded dependency-gap rescue
- **Dependencies**: Task 1D.2
- **Acceptance Criteria**:
  - Any retained heuristic is justified as a temporary bridge or invariant mechanical rule
- **Validation**:
  - Diff review plus replay/regression re-run

## Iterative Loop Design

This should become the normal operating cycle:

1. Hook runs during everyday Claude usage.
2. Hook writes:
   - operational log
   - learning queue candidates
3. Review-batch script consolidates recent candidates.
4. Reviewed rows are promoted into:
   - training data
   - regression fixture
   - replay fixture
5. Retrain from scratch.
6. Compare report + replay + regression.
7. Promote new ONNX assets only if gates pass.
8. Trim runtime heuristics if the new model proved the distinction.

## Operational Cadence

Recommended cadence:

- always-on queue capture during live use
- review batch generation daily or every few sessions
- retraining after enough reviewed rows accumulate, for example:
  - at least 20 new approved rows, or
  - any new false-positive cluster, or
  - any new false-negative cluster

Do not retrain on every hook run.

## Testing Strategy

- Runtime gate:
  - `python3 -m unittest discover -s tests -v`
- Replay gate:
  - `python3.11 training/replay_eval_v2.py ...`
- Full retrain gate:
  - `python3.11 training/train_v2.py`
- Queue workflow gate:
  - build review batch
  - promote sample rows
  - retrain on updated dataset

## Acceptance Criteria

Sprint 1 is successful when:

- the hook automatically captures learning candidates
- the capture queue is deduped, bounded, and label-friendly
- the new judgement labels exist in the corpus and training pipeline
- the retrained model improves on the new conversational boundary
- at least one current runtime judgement rule is a candidate for removal

## Risks And Gotchas

- If the hook captures too much, the queue becomes noise.
  - Mitigation: capture only blocked, low-margin, and heuristic-rescued clauses.
- If the hook captures too little, we miss rare but important patterns.
  - Mitigation: add a small random pass sample and transcript mining backfill.
- If auto-capture writes raw secrets, the queue becomes unsafe to commit.
  - Mitigation: reuse and strengthen current sanitization before queue write.
- If the model gains new labels but runtime still assumes old ones, training/export breaks.
  - Mitigation: update label enums, loaders, and reports together in one slice.

## Rollback Plan

- Keep current live ONNX assets while building the queue and review flow.
- If queue capture causes noise or instability, disable it via env flag and keep the current operational log only.
- If retrained assets underperform, revert only the model artifacts and keep the new data-collection pipeline.

# Multi-Reviewer Learning Loop Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-reviewer relearn step with a staged, council-based learning loop that is safer to run autonomously.

**Architecture:** Keep the current single-file orchestrator in `hook/assumption-guard/assumption-guard.py`, but split reviewed data into `staged-*` and `accepted-*` overlays, run three independent reviewer subprocesses, resolve agreement deterministically first, use a consolidator only for disputed cases, and only merge staged data into accepted overlays after a successful candidate promotion.

**Tech Stack:** Python 3.8+, current ONNX runtime path, `claude -p` subprocess reviewers, JSONL state files, existing unittest suite.

---

## Overview

The current system is operationally safe but epistemically fragile because a single clean `claude -p` review can flow into training overlays. This plan hardens the loop in four ways:

1. no reviewed data goes directly into accepted overlays
2. three independent reviewers must agree before rows are trusted automatically
3. deterministic vote rules handle most cases; an optional consolidator only handles disputed rows
4. accepted overlays advance only after a winning model promotion

## Files To Modify

- Modify: `hook/assumption-guard/assumption-guard.py`
- Modify: `tests/test_assumption_guard.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/training.md`
- Create: `docs/multi-reviewer-learning-loop-plan.md`

## New State Files

All under `hook/assumption-guard/state/` at install time:

- `reviewed-a.jsonl`
- `reviewed-b.jsonl`
- `reviewed-c.jsonl`
- `review-consensus.jsonl`
- `review-consolidated.jsonl`
- `staged-training-overlay.jsonl`
- `staged-regression-overlay.jsonl`
- `staged-replay-overlay.jsonl`
- `accepted-training-overlay.jsonl`
- `accepted-regression-overlay.jsonl`
- `accepted-replay-overlay.jsonl`
- `promotion-manifest.jsonl`

## Sprint 1: Split Staged Data From Accepted Data

**Goal:** Ensure new reviewed rows cannot permanently affect future retrains unless a model trained with them is actually promoted.

**Demo/Validation:**
- Learning cycle runs and writes staged overlays only
- Rejected candidate leaves accepted overlays unchanged
- Successful promotion advances staged rows into accepted rows

### Task 1.1: Add Staged And Accepted Overlay Paths

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Add state-path constants for:
  - `staged-training-overlay.jsonl`
  - `staged-regression-overlay.jsonl`
  - `staged-replay-overlay.jsonl`
  - `accepted-training-overlay.jsonl`
  - `accepted-regression-overlay.jsonl`
  - `accepted-replay-overlay.jsonl`
- [ ] Keep old overlay names as migration aliases only if needed for backward compatibility
- [ ] Write a failing test that asserts the learning cycle uses staged overlays first
- [ ] Run the targeted test to verify it fails for the right reason
- [ ] Implement the path wiring
- [ ] Re-run the targeted test to verify it passes

### Task 1.2: Stage Promotions Instead Of Writing Directly To Accepted

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Split `promote_reviewed_rows()` into:
  - `promote_reviewed_rows_to_staged()`
  - `advance_staged_rows_to_accepted()`
- [ ] Ensure learning-cycle writes only staged rows before retraining
- [ ] Ensure accepted overlays remain unchanged when a candidate is rejected
- [ ] Write a failing test for “candidate rejected -> staged changed, accepted unchanged”
- [ ] Implement the minimal code to pass it

### Task 1.3: Train From Baseline + Accepted + Optional Staged

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Update `command_train()` to accept both accepted and staged overlay inputs
- [ ] Default normal retraining to `baseline + accepted + current-cycle staged`
- [ ] Keep replay/regression evaluation explicit and deterministic
- [ ] Add a test that the current-cycle staged overlay is included in candidate training but not yet accepted

## Sprint 2: Add Three Independent Reviewers

**Goal:** Replace the single-reviewer labeling step with three independent `claude -p` reviews whose outputs can be compared.

**Demo/Validation:**
- One learning cycle produces three reviewer output files
- Reviewer row order is preserved after per-reviewer randomization is reversed
- Review batch can continue even if one reviewer batch is malformed and quarantined

### Task 2.1: Add Reviewer Identity And Output Files

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Extend reviewer output schema with `reviewer_id`
- [ ] Add output files:
  - `reviewed-a.jsonl`
  - `reviewed-b.jsonl`
  - `reviewed-c.jsonl`
- [ ] Add a helper to run one reviewer with:
  - same prompt contract
  - same label set
  - randomized row order
  - preserved `candidate_id`
- [ ] Write a failing test that expects three reviewer files from the learning cycle
- [ ] Implement the code to satisfy it

### Task 2.2: Build Review Council Mode

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Add a `review-council` internal mode or helper
- [ ] Call reviewer A/B/C sequentially in the learning-cycle path
- [ ] Keep `ASSUMPTION_GUARD_DISABLE=1` for every reviewer subprocess
- [ ] Fail fast to quarantine if a reviewer returns malformed structure after retry
- [ ] Add a test that malformed reviewer output is quarantined and does not enter staged overlays

## Sprint 3: Deterministic Consensus And Consolidation

**Goal:** Auto-accept only high-agreement rows, quarantine clearly unsafe rows, and use a consolidator only for disputed-but-valuable cases.

**Demo/Validation:**
- 3/3 agreement -> row reaches staged training
- 2/3 soft agreement -> row reaches staged fixture path only when allowed
- block/pass split disagreement -> quarantine

### Task 3.1: Add Deterministic Consensus Reducer

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Add `review-consensus.jsonl`
- [ ] Implement reduction logic per candidate:
  - exact label agreement
  - block/pass agreement
  - median confidence
  - confidence spread
  - pattern family agreement
- [ ] Encode outcomes:
  - `staged_training`
  - `staged_regression`
  - `staged_replay`
  - `needs_consolidation`
  - `quarantine`
- [ ] Write failing tests for:
  - unanimous 3/3 accept
  - 2/3 same label, unanimous block/pass
  - split on block/pass

### Task 3.2: Add Consolidator For Disputed Cases

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Add a consolidator prompt that only sees:
  - original candidate row
  - three reviewer outputs
- [ ] Restrict consolidator outputs to:
  - `staged_training`
  - `staged_regression`
  - `staged_replay`
  - `quarantine`
- [ ] Never allow consolidator to write directly to accepted overlays
- [ ] Write failing tests for:
  - disputed high-value row resolved to staged-only
  - unresolved row quarantined

## Sprint 4: Promotion Safety And Acceptance Gates

**Goal:** Ensure only demonstrably better candidates can promote, and only then can staged rows become accepted rows.

**Demo/Validation:**
- Candidate fails gates -> no live swap, no accepted-overlay advancement
- Candidate wins -> atomic asset swap, manifest written, staged rows advanced

### Task 4.1: Tighten Candidate Promotion Flow

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Update `command_learning_cycle()` so promotion order is:
  1. train candidate from baseline + accepted + staged
  2. evaluate candidate
  3. if candidate wins, atomically swap live assets
  4. only then advance staged rows to accepted overlays
- [ ] Keep current regression/replay/targeted-family gates
- [ ] Add a failing test for “candidate rejected -> accepted overlays unchanged”

### Task 4.2: Add Promotion Manifest And Rollback Metadata

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Write `promotion-manifest.jsonl` with:
  - candidate dir
  - selected candidate name
  - dataset hash
  - accepted overlay hash
  - staged overlay hash
  - review batch ids
  - promotion timestamp
- [ ] Add a test that a successful promotion writes a manifest row

## Sprint 5: Data Health Gates

**Goal:** Prevent noisy or unbalanced review batches from retraining automatically.

**Demo/Validation:**
- unhealthy staged batch -> retrain skipped
- healthy staged batch -> retrain allowed

### Task 5.1: Add Batch Health Metrics

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Test: `tests/test_assumption_guard.py`

- [ ] Compute:
  - label distribution drift
  - family concentration
  - duplicate rate
  - disagreement rate
  - consolidator usage rate
- [ ] Add gating rules for unhealthy batches
- [ ] Write failing tests for:
  - one-family flood
  - high disagreement batch
  - duplicate-heavy batch

## Sprint 6: Optional Shadow Gate

**Goal:** Require live-like evidence before a candidate becomes the active model.

**Demo/Validation:**
- candidate can pass offline gates but still be held in shadow
- promotion only occurs after shadow acceptance when enabled

### Task 6.1: Add Shadow Evaluation Mode

**Files:**
- Modify: `hook/assumption-guard/assumption-guard.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Test: `tests/test_assumption_guard.py`

- [ ] Add a shadow mode that compares current live vs candidate on the next N queued/live clauses
- [ ] Add thresholds for disagreement and safe-lane regressions
- [ ] Keep this feature off by default initially
- [ ] Add tests for:
  - shadow gate blocks promotion
  - shadow gate allows promotion

## Testing Strategy

- Run targeted tests after each task, then the full suite:
  - `python3 -m unittest discover -s tests -v`
- Add focused tests for:
  - staged vs accepted overlay behavior
  - 3-reviewer council outputs
  - deterministic consensus rules
  - consolidator quarantine behavior
  - promotion manifest creation
  - no accepted-overlay advancement on failed candidate
- Keep the current end-to-end packaged training smoke in place

## Potential Risks And Gotchas

- Three reviewers can still fail in correlated ways if prompts are too similar. Use randomized order and separate reviewer ids, but do not assume perfect independence.
- Consolidator can accidentally become the new single point of failure if it is allowed to override too much. Keep it staged-only.
- Staged overlay growth can become noisy. Add health gates before retraining.
- Shadow mode can stall learning if thresholds are too strict. Start with reporting-only mode if needed.

## Rollback Plan

- If Sprint 2 introduces instability, keep the current single-reviewer path behind a feature flag and leave it disabled by default.
- If staged/accepted split causes migration issues, keep accepted overlays empty and continue using packaged baseline while the staging code is stabilized.
- If a promoted candidate causes regressions, use the previous `promotion-manifest.jsonl` row to restore the earlier asset set.

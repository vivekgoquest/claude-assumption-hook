# Assumption Guard Training

## Overview

Assumption Guard v2 is a clause-level multiclass classifier with a deterministic runtime wrapper. The model does not replace rules; it handles only the ambiguous middle after hard-pass, hard-block, and regex filtering.

The committed runtime label set is:

- `assert_unverified`
- `recommend_unverified`
- `capability_promise_unverified`
- `unchecked_limitation`
- `verification_narration`
- `reference_language`
- `verified_limitation`
- `dependency_gap_grounded`
- `describe_type`
- `reason_conditionally`
- `code_content`
- `idiomatic_compare`
- `recommend_supported`

Runtime maps the first three classes to BLOCK and the rest to PASS.

## Dataset

The committed corpus lives inside the packaged baseline bundle at `hook/assumption-guard/baseline/assumption-guard-training-labeled.jsonl`.

Current counts from the committed v2 report:

- rows: `485`
- train/dev/test split: `344 / 72 / 69`

Intent counts:

- `assert_unverified`: `120`
- `recommend_unverified`: `66`
- `capability_promise_unverified`: `6`
- `unchecked_limitation`: `45`
- `verification_narration`: `6`
- `reference_language`: `61`
- `verified_limitation`: `27`
- `dependency_gap_grounded`: `6`
- `describe_type`: `44`
- `reason_conditionally`: `37`
- `code_content`: `21`
- `idiomatic_compare`: `28`
- `recommend_supported`: `18`

Each row uses the v2 schema:

- `id`
- `text`
- `block`
- `intent`
- `source_type`
- `source_hash`
- `evidence_present`
- `quoted_or_code`
- `review_status`
- `notes`

`source_hash` is used for split stability so clauses from the same source do not get randomly scattered across train/dev/test.

## Transcript Mining

Transcript mining now runs through the single-file orchestrator:

```bash
python3.11 hook/assumption-guard/assumption-guard.py mine --output /tmp/assumption-guard-mined-v2.jsonl
```

It currently:

- reads local Claude transcript JSONL files from `/Users/vivek/.claude/projects`
- extracts assistant text only
- splits messages into clauses
- weak-matches risky and safe-lane language
- sanitizes paths, URLs, emails, IDs, and obvious secrets
- emits reviewable JSONL rows in the v2 schema

Only sanitized, derived examples should be committed to the repo.

## Training Stack

The current checked-in v2 training path uses:

- Python 3.11
- `torch`
- `transformers`
- `onnx`
- `onnxruntime`
- `scikit-learn`
- `rapidfuzz`

Candidate families trained by `hook/assumption-guard/assumption-guard.py train`:

1. baseline: hashed TF-IDF + adaptive calibrated linear SVM fallback
2. candidate A: `sentence-transformers/all-MiniLM-L6-v2`
3. candidate B: `microsoft/deberta-v3-small`

## Train Command

Run from the repo root:

```bash
python3.11 hook/assumption-guard/assumption-guard.py train
```

The script will:

1. load and normalize the labeled dataset
2. split by stable `source_hash`
3. train the baseline and transformer candidates
4. search thresholds from `0.20` to `0.80` in `0.02` steps
5. evaluate regression and replay fixtures
6. export the selected transformer to ONNX
7. write the tokenizer, metadata, and final report

Generated artifacts:

- `hook/assumption-guard/assumption-guard-v2.onnx`
- `hook/assumption-guard/assumption-guard-v2-tokenizer.json`
- `hook/assumption-guard/assumption-guard-v2-meta.json`
- `hook/assumption-guard/baseline/assumption-guard-v2-report.json`

Local overlays can be merged into the training run without changing the committed baseline corpus:

```bash
python3.11 hook/assumption-guard/assumption-guard.py train \
  --overlay-training-data ~/.claude/hooks/assumption-guard/state/training-overlay.jsonl \
  --overlay-regression-cases ~/.claude/hooks/assumption-guard/state/regression-overlay.jsonl \
  --overlay-replay-cases ~/.claude/hooks/assumption-guard/state/replay-overlay.jsonl
```

The objective trigger for when to run that training is handled separately by:

```bash
python3.11 hook/assumption-guard/assumption-guard.py maybe-trigger
```

That gatekeeper checks only pending queue rows, not raw log-file size, and launches `hook/assumption-guard/assumption-guard.py learning-cycle` only when the configured pending-row, cluster, age, and cooldown conditions are met.

## Selection Logic

Candidate selection currently enforces:

- full regression fixture match
- dev-set recall floor
- replay coverage floors
- preference for the smallest transformer that clears the gates

The chosen runtime threshold is written into `hook/assumption-guard/assumption-guard-v2-meta.json` and is not tuned at runtime.

Current selected model:

- `sentence-transformers/all-MiniLM-L6-v2`
- threshold: `0.20`

## Current Report Snapshot

From `hook/assumption-guard/baseline/assumption-guard-v2-report.json`:

- regression fixture: `62 / 62`
- replay corpus: `17 / 17`
- held-out test confusion: `27 TP / 42 TN / 0 FP / 0 FN`
- targeted family accuracy:
  - `verification_narration`: `1.0`
  - `dependency_gap_grounded`: `1.0`
  - `capability_promise_unverified`: `1.0`
- ONNX parity max abs delta: within the configured `1e-3` bound
- all acceptance gates: `true`

Candidate summary:

- baseline linear SVM: replay `0.9 / 1.0`, regression `55 / 62`
- MiniLM-L6: replay `1.0 / 1.0`, regression `62 / 62`
- DeBERTa-v3-small: replay `1.0 / 1.0`, regression `62 / 62`

MiniLM is selected because it clears the gates and is smaller than DeBERTa.

## Acceptance Gates

The committed report evaluates:

- `regression_full_match`
- `replay_block_recall >= 0.95`
- `replay_pass_recall >= 0.95`
- `verification_narration` targeted family gate
- `dependency_gap_grounded` targeted family gate
- `capability_promise_unverified` targeted family gate
- `reference_language` accuracy gate
- `verified_limitation` accuracy gate
- `reason_conditionally` accuracy gate
- `describe_type` accuracy gate
- `idiomatic_compare` accuracy gate
- ONNX parity gate
- latency gate

On non-native Apple Silicon training environments, the report records `latency_environment_matches_target: false` and does not fail the run on that hardware mismatch.

## Replay Evaluation

Use the replay script to re-check committed assets without retraining:

```bash
python3.11 hook/assumption-guard/assumption-guard.py replay \
  --regression-cases hook/assumption-guard/baseline/assumption-guard-regression-cases.json \
  --model hook/assumption-guard/assumption-guard-v2.onnx \
  --tokenizer hook/assumption-guard/assumption-guard-v2-tokenizer.json \
  --meta hook/assumption-guard/assumption-guard-v2-meta.json \
  --output /tmp/assumption-guard-v2-replay.json
```

## Verification

After retraining, run:

```bash
python3 -m unittest discover -s tests -v
```

That suite verifies:

- clause splitting
- fallback behavior when ONNX deps or assets are missing
- regression fixture behavior through the hook
- replay evaluation against the committed assets
- presence of the committed v2 artifacts and report gates

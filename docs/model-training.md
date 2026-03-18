# Assumption Guard — Model Training

## Overview

The Assumption Guard classifier is a binary line-level model:

- `1` = BLOCK
- `0` = PASS

It is trained only on regex-flagged lines. Its job is not to replace regex, but to decide whether the flagged line is genuinely unverified uncertainty or a known false positive.

## Current Dataset

The committed dataset lives at `training/assumption-guard-training-labeled.jsonl`.

Current counts:

- rows: 253
- block: 102
- pass: 151

Intent buckets:

- `assert_unverified`
- `recommend_unverified`
- `unchecked_limitation`
- `reference_language`
- `describe_type`
- `compare`
- `reason_conditionally`
- `verified_limitation`
- `code_content`

The key v1.1 shift is that unchecked limitations and uncertain recommendations are now BLOCK examples. Verified limitations remain PASS only when the sentence already states what was checked.

## Feature Set

The training pipeline reuses the production definitions in `hook/assumption_guard_model.py`.

### Text features

- TF-IDF
- n-grams: 1-3
- max features: 5000
- sublinear TF scaling

### Intent features

There are 14 structural signals:

1. table row
2. code comment
3. trigger only in backticks
4. meta words
5. type keywords
6. conditional start
7. trigger in quotes
8. hedge density
9. line length
10. first-person epistemic start
11. improvement/review words
12. verification evidence
13. unchecked limitation language
14. uncertainty admission start

## Training Command

Run from the repo root:

```bash
python3 training/train_assumption_guard.py
```

This command:

- loads the labeled JSONL dataset
- runs 5-fold stratified CV with `random_state=42`
- trains the final pipeline on the full dataset
- writes `model/assumption-guard-model.pkl`
- writes `model/assumption-guard-metrics.json`
- evaluates the regression fixture at `training/assumption-guard-regression-cases.json`

## Current Metrics

From the committed metrics artifact:

- F1 macro mean: 0.887265
- F1 macro std: 0.029485
- regression fixture: 12/12 matched
- model size: 191,925 bytes

## Regression Fixture

The regression fixture is checked in at `training/assumption-guard-regression-cases.json`.

It covers:

- strict-policy blocks for uncertain recommendations
- strict-policy blocks for unchecked limitations
- factual guesses
- known PASS cases for meta discussion, type analysis, conditionals, idioms, and verified limitations

This fixture is part of both the training output and the unittest suite.

## Verification

After retraining, run:

```bash
python3 -m unittest discover -s tests -v
```

That suite verifies:

- hook behavior against the regression fixture
- regex-only fallback when ML imports fail
- regex-only fallback when the pickle is corrupt
- reproducible training metrics and pickle loadability

# Assumption Guard

**Stop Claude from guessing or hand-waving. Make it verify.**

Assumption Guard is a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) Stop hook that blocks responses containing unverified uncertainty. It catches:

- factual guesses: `I think the timeout is 30 seconds`
- speculative recommendations: `Maybe rename this helper`
- unchecked limitations: `I'm not sure because I haven't checked the config yet`

The hook then pushes Claude back to its tools so it can verify, justify, or explicitly describe what it checked and why the evidence is still insufficient.

## What It Blocks

The current policy is intentionally strict. A flagged line is blocked when Claude is uncertain and has not yet grounded the statement in evidence. That includes:

- `Maybe rename this helper to be clearer`
- `This could be simplified by extracting a helper`
- `We should probably add a unit test here`
- `I'm not sure because I have not checked the config yet`
- `The app probably listens on port 3000`

## What It Lets Through

The ML filter still removes obvious false positives that regex alone cannot distinguish:

- meta discussion: `The hook flagged "probably" in the response`
- type analysis: `The return value could be None or a string`
- conditionals: `When the cache is cold, this may take longer`
- idioms: `Use map rather than forEach here`
- verified limitations: `I checked package.json and requirements.txt and cannot confirm the config value from this repo.`

## Runtime Design

Two-stage detection, fully local:

1. Regex pre-filter, implemented with stdlib only.
2. Optional sklearn classifier, loaded lazily when dependencies and model are available.

If ML dependencies are missing, the pickle is missing/corrupt, or prediction fails, the hook stays fail-open and falls back to regex-only mode. It logs the exact reason instead of crashing.

## Installation

### Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- Python 3.8+
- Optional for ML filtering and retraining: `scikit-learn`, `numpy`, `scipy`

Install the optional ML dependencies with:

```bash
pip install scikit-learn numpy scipy
```

Without those packages, the hook still runs in regex-only mode.

### Setup

1. Copy the hook files:

```bash
mkdir -p ~/.claude/hooks
cp hook/assumption-guard.py ~/.claude/hooks/
cp hook/assumption_guard_model.py ~/.claude/hooks/
```

2. Copy the trained model:

```bash
cp model/assumption-guard-model.pkl ~/.claude/
```

3. Merge the Stop hook into `~/.claude/settings.json`:

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

4. Start a new Claude Code session.

## Logging

Every invocation appends JSONL to `~/.claude/assumption-guard.log.jsonl`.

Important fields:

- `stage`: `regex`, `model`, `regex_only`, or `hook_error`
- `ml_available`: `true`, `false`, or `null`
- `fallback_reason`: `ml_import_error`, `model_missing`, `pickle_load_error`, `predict_error`, or `null`
- `error_stage`: `ml_import`, `model_load`, `model_predict`, `hook_runtime`, or `null`
- `flags` / `filtered`: what blocked and what the ML model filtered out

## Retraining

The repo now includes a checked-in training entrypoint and a regression fixture.

Run:

```bash
python3 training/train_assumption_guard.py
```

This command:

- loads `training/assumption-guard-training-labeled.jsonl`
- reuses the production feature definitions
- runs 5-fold CV
- writes `model/assumption-guard-model.pkl`
- writes `model/assumption-guard-metrics.json`
- checks the shipped regression fixture in `training/assumption-guard-regression-cases.json`

## Current Model Snapshot

From `model/assumption-guard-metrics.json`:

- training rows: 253
- labels: 102 block / 151 pass
- cross-validation F1 macro: 0.887265
- regression fixture: 12/12 matched
- model size: 191,925 bytes (~187.4 KB)

Architecture:

- TF-IDF: 1-3 grams, 5000 max features
- intent features: 14 structural signals
- classifier: `LogisticRegression(class_weight="balanced")`

## Project Structure

```text
hook/
├── assumption-guard.py
├── assumption_guard_model.py
└── settings-snippet.json
model/
├── assumption-guard-model.pkl
└── assumption-guard-metrics.json
training/
├── assumption-guard-training-labeled.jsonl
├── assumption-guard-regression-cases.json
└── train_assumption_guard.py
tests/
└── test_assumption_guard.py
docs/
├── assumption-guard-system.md
├── model-training.md
└── archive/
```

## Verification

Run the full suite with:

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT

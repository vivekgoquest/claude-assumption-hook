# Assumption Guard

**Stop Claude from guessing, hand-waving, or recommending things it has not verified.**

Assumption Guard is a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) Stop hook that blocks uncertain assistant clauses until they are backed by evidence. The default runtime is v2:

- clause-level detection instead of whole-message classification
- stdlib regex and hard-rule prefiltering
- optional ONNX multiclass classifier for ambiguous cases
- fail-open regex-only mode when model assets or ML deps are unavailable

The policy is intentionally strict. It blocks:

- unverified factual claims: `I think the timeout is 30 seconds`
- unsupported recommendations: `Maybe rename this helper to be clearer`
- unchecked limitations: `I'm not sure because I haven't checked the config yet`

It allows clearly bounded safe lanes such as meta discussion, quoted examples, type/shape analysis, conditional reasoning, idiomatic comparisons, and verified limitations that explicitly say what was checked.

## Runtime Design

The hook contract stays simple:

- input: the Claude Stop-hook JSON payload on stdin
- pass: no stdout, exit `0`
- block: `{"decision":"block","reason":"..."}`

The v2 runtime works in layers:

1. Extract only the current assistant turn from the transcript.
2. Split the assistant text into clauses.
3. Apply hard-pass suppressors for code, quotes, and meta/report language.
4. Apply hard-block rules for explicit unchecked language such as `I need to verify ...`.
5. Run the ONNX classifier only on ambiguous clauses that survive the earlier filters.
6. Sum the BLOCK-class probabilities and block when `p_block >= threshold`.

If `onnxruntime`, `tokenizers`, the tokenizer asset, metadata JSON, or the ONNX model are missing or broken, the hook does not crash. It logs the exact fallback reason and continues in regex-only mode.

The legacy sklearn path is still available with:

```bash
ASSUMPTION_GUARD_BACKEND=v1_legacy
```

## Installation

### Runtime Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- Python 3.8+
- optional ML runtime packages for v2: `onnxruntime`, `tokenizers`

Install the optional runtime packages with:

```bash
pip install onnxruntime tokenizers
```

Without those packages, the hook still runs in regex-only mode.

### Setup

1. Copy the hook files:

```bash
mkdir -p ~/.claude/hooks
cp hook/assumption-guard.py ~/.claude/hooks/
cp hook/assumption_guard_v2.py ~/.claude/hooks/
cp hook/assumption_guard_model.py ~/.claude/hooks/
```

2. Copy the default v2 artifacts:

```bash
cp model/assumption-guard-v2.onnx ~/.claude/
cp model/assumption-guard-v2-tokenizer.json ~/.claude/
cp model/assumption-guard-v2-meta.json ~/.claude/
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

- `stage`: `regex`, `regex_only`, `model`, `hook_error`
- `backend`: `onnx`, `regex`, `legacy`
- `ml_available`
- `fallback_reason`
- `error_stage`
- `predicted_intent`
- `p_block`
- `threshold`
- `flags` / `filtered`

Common fallback reasons:

- `ml_import_error`
- `meta_missing`
- `tokenizer_missing`
- `model_missing`
- `asset_load_error`
- `model_load_error`
- `predict_error`

## Transcript Mining And Training

v2 is trained from a transcript-derived, sanitized clause corpus. The default mining source is:

```text
/Users/vivek/.claude/projects
```

Only sanitized, derived clause examples are committed back into the repo.

### Training Requirements

The checked-in training path currently uses:

- Python 3.11
- `torch`
- `transformers`
- `onnx`
- `onnxruntime`
- `scikit-learn`
- `rapidfuzz`

### Commands

Mine candidate clauses from local transcripts:

```bash
python3.11 training/mine_transcripts_v2.py --output /tmp/assumption-guard-mined-v2.jsonl
```

Train, compare candidates, export ONNX, and write the final report:

```bash
python3.11 training/train_v2.py
```

Replay the committed regression fixture against the exported v2 assets:

```bash
python3.11 training/replay_eval_v2.py \
  --regression-cases training/assumption-guard-regression-cases.json \
  --model model/assumption-guard-v2.onnx \
  --tokenizer model/assumption-guard-v2-tokenizer.json \
  --meta model/assumption-guard-v2-meta.json \
  --output /tmp/assumption-guard-v2-replay.json
```

## Current v2 Snapshot

From `model/assumption-guard-v2-report.json`:

- dataset rows: `467`
- split: `334 train / 67 dev / 66 test`
- intents: `10`
- selected candidate: `sentence-transformers/all-MiniLM-L6-v2`
- threshold: `0.20`
- regression fixture: `56 / 56`
- replay corpus: `15 / 15`
- held-out test split: `27 TP / 39 TN / 0 FP / 0 FN`
- ONNX size: about `22 MB`
- tokenizer size: about `695 KB`

Candidate comparison:

- baseline: hashed TF-IDF + calibrated linear SVM
- candidate A: MiniLM-L6 sequence classifier
- candidate B: DeBERTa-v3-small sequence classifier

The final report currently shows all three candidates, with MiniLM selected as the smallest model that clears the replay and regression gates.

## Project Structure

```text
hook/
├── assumption-guard.py
├── assumption_guard_model.py
├── assumption_guard_v2.py
└── settings-snippet.json
model/
├── assumption-guard-model.pkl
├── assumption-guard-metrics.json
├── assumption-guard-eval.json
├── assumption-guard-v2.onnx
├── assumption-guard-v2-tokenizer.json
├── assumption-guard-v2-meta.json
└── assumption-guard-v2-report.json
training/
├── assumption-guard-training-labeled.jsonl
├── assumption-guard-regression-cases.json
├── assumption-guard-replay-cases.json
├── mine_transcripts_v2.py
├── replay_eval_v2.py
├── train_assumption_guard.py
└── train_v2.py
tests/
└── test_assumption_guard.py
docs/
├── assumption-guard-system.md
├── model-training.md
└── archive/
```

## Verification

Run the full test suite with:

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT

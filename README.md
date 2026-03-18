# Assumption Guard

Assumption Guard is a Claude Code Stop hook that blocks unsupported claims before they are sent.

The point of this project is simple: Claude Code is useful, but it still tends to guess unless something forces it to verify. Assumption Guard sits at the end of a response, catches risky language, and makes Claude go back to the repo, logs, tests, or commands before the answer is allowed through.

## What It Blocks

Assumption Guard is intentionally strict. It blocks clauses like:

- unverified factual claims
- unsupported recommendations
- unsupported capability promises
- unchecked “I have not verified this yet” language

Examples:

- `I think the timeout is 30 seconds`
- `This should probably be refactored`
- `I can identify the new records and remove them`
- `I haven't checked the config yet`

## What It Allows

It keeps a few safe lanes open so Claude can still work naturally:

- verification narration: `Let me verify whether...`
- quoted or meta discussion about the detector
- type and shape analysis
- conditional reasoning
- grounded limitations that say what was checked and what is still missing

Examples:

- `Let me verify whether that endpoint exists`
- `The return value could be None or a string`
- `If the flag is enabled, the middleware would return 401`
- `I confirmed DELETE /videos/bulk exists, but without saved ids I can't remove only the new rows`

## How It Works

The runtime is layered:

1. read only the current assistant turn from the Claude transcript
2. split the response into clauses
3. apply hard-pass suppressors for code, quotes, type language, and meta language
4. apply hard-block rules for explicit unchecked language
5. run the ONNX classifier only on ambiguous clauses
6. block if any clause still crosses the configured threshold

The hook contract stays simple:

- input: Claude Stop-hook JSON on stdin
- pass: no stdout, exit `0`
- block: `{"decision":"block","reason":"..."}`

If ONNX runtime dependencies or model assets are missing, the hook fails open into regex-only mode and logs the fallback reason.

## Quick Start

Requirements:

- Claude Code
- Python 3.8+
- optional for ONNX mode: `onnxruntime`, `tokenizers`

Install the optional runtime dependencies:

```bash
pip install onnxruntime tokenizers
```

Run the installer:

```bash
bash hook/assumption-guard/install.sh
```

The installer:

- copies `hook/assumption-guard/` into `~/.claude/hooks/assumption-guard/`
- creates `~/.claude/hooks/assumption-guard/state/`
- adds the Stop hook command to `~/.claude/settings.json` without removing unrelated settings

The installed Stop hook command is:

```bash
python3 ~/.claude/hooks/assumption-guard/assumption-guard.py
```

Then start a new Claude Code session.

If you prefer to install manually, the equivalent settings snippet is in `hook/assumption-guard/settings-snippet.json`.

## Repo Layout

This repo is intentionally small:

```text
hook/
└── assumption-guard/
    ├── assumption-guard.py
    ├── install.sh
    ├── settings-snippet.json
    ├── assumption-guard-v2.onnx
    ├── assumption-guard-v2-tokenizer.json
    ├── assumption-guard-v2-meta.json
    └── baseline/
        ├── assumption-guard-training-labeled.jsonl
        ├── assumption-guard-regression-cases.json
        ├── assumption-guard-replay-cases.json
        └── assumption-guard-v2-report.json

docs/
├── architecture.md
└── training.md

tests/
└── test_assumption_guard.py
```

At runtime the installed hook creates a local `state/` folder inside that same package:

```text
~/.claude/hooks/assumption-guard/state/
├── assumption-guard.log.jsonl
├── learning-queue.jsonl
├── queue/
├── reviewed-claude.jsonl
├── current-model-report.json
└── candidates/
```

The important split is:

- `baseline/` is packaged seed data and the committed report
- `state/` is mutable local runtime state and is not committed

## Runtime State

By default the hook writes:

- log: `~/.claude/hooks/assumption-guard/state/assumption-guard.log.jsonl`
- learning queue: `~/.claude/hooks/assumption-guard/state/learning-queue.jsonl`
- current model report: `~/.claude/hooks/assumption-guard/state/current-model-report.json`

Useful env vars:

- `ASSUMPTION_GUARD_STATE_DIR`
- `ASSUMPTION_GUARD_LOG_PATH`
- `ASSUMPTION_GUARD_QUEUE_PATH`
- `ASSUMPTION_GUARD_CAPTURE_MODE`
- `ASSUMPTION_GUARD_LOW_MARGIN`
- `ASSUMPTION_GUARD_TRIGGER_MODE`
- `ASSUMPTION_GUARD_DISABLE`

`ASSUMPTION_GUARD_TRIGGER_MODE` defaults to `post_append`, so the hook checks after queue writes whether a learning cycle should be launched. Set it to `off` if you want enforcement without background trigger checks.

## Learning Loop

The learning loop is local and iterative:

1. the live hook writes high-value clauses to the learning queue
2. `maybe-trigger` checks whether there is enough pending signal
3. `learning-cycle` builds a review batch, calls `claude -p`, promotes reviewed rows into overlays, retrains from scratch, and only swaps the live assets if the candidate beats the current model

Useful commands:

```bash
python3.11 hook/assumption-guard/assumption-guard.py maybe-trigger
python3.11 hook/assumption-guard/assumption-guard.py learning-cycle
python3.11 hook/assumption-guard/assumption-guard.py mine --output /tmp/assumption-guard-mined.jsonl
python3.11 hook/assumption-guard/assumption-guard.py train
python3.11 hook/assumption-guard/assumption-guard.py replay \
  --regression-cases hook/assumption-guard/baseline/assumption-guard-regression-cases.json \
  --model hook/assumption-guard/assumption-guard-v2.onnx \
  --tokenizer hook/assumption-guard/assumption-guard-v2-tokenizer.json \
  --meta hook/assumption-guard/assumption-guard-v2-meta.json \
  --output /tmp/assumption-guard-replay.json
```

## Current Model

The packaged model is a clause-level ONNX classifier with 13 intent labels.

Current packaged snapshot:

- selected model: `sentence-transformers/all-MiniLM-L6-v2`
- dataset rows: `485`
- threshold: `0.20`
- regression: `62 / 62`
- replay: `17 / 17`

More detail:

- [Architecture](docs/architecture.md)
- [Training](docs/training.md)

## Verify

Run:

```bash
python3 -m unittest discover -s tests -v
```

That covers runtime behavior, fallback behavior, packaged layout, replay, learning-loop plumbing, and training smoke checks.

#!/usr/bin/env python3
"""Run replay evaluation against committed regression or replay corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = REPO_ROOT / "hook"
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

import assumption_guard_v2 as v2  # pylint: disable=wrong-import-position


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = json.loads(Path(args.regression_cases).read_text())
    classifier = v2.OnnxIntentClassifier(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        meta_path=args.meta,
    ).load()

    rows = []
    matched = 0
    for case in cases:
        decisions = v2.evaluate_text(case["text"], classifier)
        blocked = bool(decisions)
        if blocked == case["expected_block"]:
            matched += 1
        rows.append(
            {
                "text": case["text"],
                "expected_block": case["expected_block"],
                "blocked": blocked,
                "matched": blocked == case["expected_block"],
                "intents": [decision.intent for decision in decisions],
            }
        )

    output = {
        "summary": {
            "matched": matched,
            "total": len(rows),
            "accuracy": matched / len(rows) if rows else 0.0,
        },
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"]))


if __name__ == "__main__":
    main()

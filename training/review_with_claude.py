#!/usr/bin/env python3
"""Review sanitized learning candidates with a headless Claude CLI batch."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CLAUDE_CMD = os.environ.get("ASSUMPTION_GUARD_REVIEW_CLAUDE_CMD", "claude")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Review batch JSONL path.")
    parser.add_argument("--output", required=True, help="Reviewed output JSONL path.")
    parser.add_argument("--quarantine-dir", required=True, help="Directory for malformed output batches.")
    parser.add_argument("--claude-cmd", default=DEFAULT_CLAUDE_CMD, help="Command used to invoke Claude.")
    parser.add_argument("--max-candidates", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(rows):
    return (
        "You are labeling sanitized assumption-guard learning candidates.\n"
        "Return ONLY valid JSON: an array of objects with keys "
        "`candidate_id`, `final_intent`, `final_block`, `confidence`, `rationale`, `pattern_family`.\n"
        "Use the intent labels exactly as given in the input when they fit, otherwise choose the closest label.\n"
        "Confidence must be a float between 0 and 1.\n"
        "Cases:\n"
        + json.dumps(rows, ensure_ascii=True, indent=2)
    )


def parse_review_output(raw_output: str):
    parsed = json.loads(raw_output)
    if not isinstance(parsed, list):
        raise ValueError("review output must be a JSON array")
    required = {"candidate_id", "final_intent", "final_block", "confidence", "rationale", "pattern_family"}
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("review row must be an object")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"missing review keys: {sorted(missing)}")
    return parsed


def run_review(claude_cmd: str, prompt: str):
    command = shlex.split(claude_cmd) + ["-p", prompt]
    env = os.environ.copy()
    env["ASSUMPTION_GUARD_DISABLE"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Claude exited with {result.returncode}")
    return result.stdout.strip()


def main():
    args = parse_args()
    batch_path = Path(args.batch)
    output_path = Path(args.output)
    quarantine_dir = Path(args.quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    input_rows = read_jsonl(batch_path)[: args.max_candidates]
    prompt = build_prompt(input_rows)

    last_error = None
    for attempt in range(2):
        try:
            raw_output = run_review(args.claude_cmd, prompt)
            reviewed = parse_review_output(raw_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for row in reviewed:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(json.dumps({"reviewed": len(reviewed), "output": str(output_path)}))
            return
        except Exception as exc:  # pragma: no cover - exercised in manual runs
            last_error = exc

    quarantine_path = quarantine_dir / f"{batch_path.stem}.failed.jsonl"
    quarantine_path.write_text(batch_path.read_text())
    raise SystemExit(json.dumps({"error": str(last_error), "quarantined": str(quarantine_path)}))


if __name__ == "__main__":
    main()

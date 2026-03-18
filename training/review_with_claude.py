#!/usr/bin/env python3
"""Review sanitized learning candidates with a headless Claude CLI batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hook" / "assumption-guard.py"
DEFAULT_CLAUDE_CMD = os.environ.get("ASSUMPTION_GUARD_REVIEW_CLAUDE_CMD", "claude")
REQUIRED_KEYS = {"candidate_id", "final_intent", "final_block", "confidence", "rationale", "pattern_family"}


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("assumption_guard_runtime", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v2 = load_runtime_module()


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
        "You are a labeling worker for Assumption Guard.\n"
        "You are not solving the user's problem. You are not giving advice. "
        "You are not rewriting the response. Your only job is to classify each candidate clause.\n"
        "Return ONLY valid JSON.\n"
        "Return one output object per input object.\n"
        "Preserve candidate_id exactly and keep the same order as input.\n"
        "Do not invent new labels.\n"
        "Do not add keys.\n"
        "Do not return markdown or prose.\n"
        "If a case is ambiguous, choose the closest label and lower confidence instead of discussing alternatives.\n"
        "Rationale must be one short sentence grounded in the clause text.\n"
        "Allowed intent labels:\n"
        + "\n".join(f"- {label}" for label in v2.LABELS)
        + "\n"
        "Decision rules:\n"
        "- verification_narration: the speaker says they are about to check or verify something now\n"
        "- capability_promise_unverified: the speaker claims they can perform an action or determine something without evidence\n"
        "- dependency_gap_grounded: the speaker names something already confirmed, then states a concrete missing dependency that blocks the next action\n"
        "- recommend_supported: the recommendation already cites concrete evidence\n"
        "- recommend_unverified: the recommendation is unsupported\n"
        "- unchecked_limitation: the speaker says they do not know or have not checked yet\n"
        "- verified_limitation: the speaker says what they checked and then gives a bounded non-conclusion\n"
        "- reference_language: the clause talks about the hook, labels, examples, or policy rather than making the claim itself\n"
        "Failure conditions:\n"
        "- any text outside valid JSON\n"
        "- any missing candidate_id\n"
        "- any unknown label\n"
        "- any changed order\n"
        "Examples:\n"
        '- "Let me verify whether I can actually identify the new videos and remove them." -> verification_narration -> final_block=false\n'
        '- "But if you want to revert, I can check which ones are new and remove them." -> capability_promise_unverified -> final_block=true\n'
        '- "I confirmed DELETE /videos/bulk exists and requires ids. But without a saved list of the newly created ids, I can\'t selectively remove them." -> dependency_gap_grounded -> final_block=false\n'
        "Output schema:\n"
        '[{"candidate_id":"...","final_intent":"...","final_block":true,"confidence":0.93,"rationale":"...","pattern_family":"..."}]\n'
        "Cases:\n"
        + json.dumps(rows, ensure_ascii=True, indent=2)
    )


def parse_review_output(raw_output: str, input_rows):
    parsed = json.loads(raw_output)
    if not isinstance(parsed, list):
        raise ValueError("review output must be a JSON array")
    if len(parsed) != len(input_rows):
        raise ValueError("review output length must match input length")
    expected_ids = [row.get("id") or row.get("candidate_id") for row in input_rows]
    actual_ids = []
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("review row must be an object")
        missing = REQUIRED_KEYS.difference(row)
        if missing:
            raise ValueError(f"missing review keys: {sorted(missing)}")
        extras = set(row).difference(REQUIRED_KEYS)
        if extras:
            raise ValueError(f"unexpected review keys: {sorted(extras)}")
        if row["final_intent"] not in v2.LABELS:
            raise ValueError(f"unknown final_intent: {row['final_intent']}")
        if not isinstance(row["final_block"], bool):
            raise ValueError("final_block must be a boolean")
        if row["final_block"] != (row["final_intent"] in v2.BLOCK_LABELS):
            raise ValueError("final_block must match the policy implied by final_intent")
        if not isinstance(row["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0.0 <= float(row["confidence"]) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("rationale must be a non-empty string")
        if len(row["rationale"].split()) > 25:
            raise ValueError("rationale must stay short")
        actual_ids.append(row["candidate_id"])
    if actual_ids != expected_ids:
        raise ValueError("candidate_id order must match the input order exactly")
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
            reviewed = parse_review_output(raw_output, input_rows)
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

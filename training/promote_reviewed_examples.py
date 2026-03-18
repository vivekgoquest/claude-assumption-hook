#!/usr/bin/env python3
"""Promote Claude-reviewed rows into local overlay corpora."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hook" / "assumption-guard.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("assumption_guard_runtime", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v2 = load_runtime_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True, help="Reviewed JSONL path.")
    parser.add_argument("--training-overlay", required=True)
    parser.add_argument("--regression-overlay", required=True)
    parser.add_argument("--replay-overlay", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_unique_jsonl(path: Path, rows: Iterable[dict], key_field: str):
    existing = set()
    if path.exists():
        existing = {row.get(key_field) for row in read_jsonl(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = row.get(key_field)
            if key in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            existing.add(key)
            count += 1
    return count


def training_row_from_review(row: dict) -> dict:
    text = row["text"]
    final_intent = row["final_intent"]
    candidate_id = row.get("candidate_id") or hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"reviewed-{candidate_id}",
        "text": text,
        "block": bool(row["final_block"]),
        "intent": final_intent,
        "source_type": "claude_review",
        "source_hash": hashlib.sha1(f"review:{candidate_id}:{text}".encode("utf-8")).hexdigest()[:16],
        "evidence_present": bool(v2.EVIDENCE_PATTERN.search(text)),
        "quoted_or_code": bool(v2.quoted_or_code_clause(text)),
        "review_status": "approved",
        "notes": f"confidence={row.get('confidence', 0)} pattern={row.get('pattern_family', 'unknown')}",
    }


def regression_row_from_review(row: dict) -> dict:
    return {
        "group": row["final_intent"],
        "text": row["text"],
        "expected_block": bool(row["final_block"]),
        "reason": f"promoted from claude review ({row.get('candidate_reason', 'unknown')})",
    }


def replay_row_from_review(row: dict) -> dict:
    return {
        "group": row["final_intent"],
        "text": row["text"],
        "expected_block": bool(row["final_block"]),
        "reason": f"pattern family {row.get('pattern_family', row['final_intent'])}",
    }


def main():
    args = parse_args()
    reviewed_rows = read_jsonl(Path(args.reviewed))
    family_counts = Counter(row.get("pattern_family", row.get("final_intent", "unknown")) for row in reviewed_rows)

    training_rows = [training_row_from_review(row) for row in reviewed_rows]
    regression_rows = []
    replay_rows = []

    existing_replay_families = {
        row.get("group")
        for row in read_jsonl(Path(args.replay_overlay))
    }

    for row in reviewed_rows:
        candidate_reason = row.get("candidate_reason", "")
        confidence = float(row.get("confidence", 0.0))
        pattern_family = row.get("pattern_family", row.get("final_intent", "unknown"))
        if row.get("disagrees_with_runtime") or confidence < 0.80 or candidate_reason in {"heuristic_rescue", "mixed_evidence_gap"}:
            regression_rows.append(regression_row_from_review(row))
        if pattern_family not in existing_replay_families or family_counts[pattern_family] >= 3:
            replay_rows.append(replay_row_from_review(row))

    training_count = append_unique_jsonl(Path(args.training_overlay), training_rows, "id")
    regression_count = append_unique_jsonl(
        Path(args.regression_overlay),
        [{**row, "case_hash": hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:16]} for row in regression_rows],
        "case_hash",
    )
    replay_count = append_unique_jsonl(
        Path(args.replay_overlay),
        [{**row, "case_hash": hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:16]} for row in replay_rows],
        "case_hash",
    )

    print(
        json.dumps(
            {
                "training_promoted": training_count,
                "regression_promoted": regression_count,
                "replay_promoted": replay_count,
            }
        )
    )


if __name__ == "__main__":
    main()

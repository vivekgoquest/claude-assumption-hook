#!/usr/bin/env python3
"""Build a deduped review batch from live queue, mined rows, and recent logs."""

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
    parser.add_argument("--queue", action="append", default=[], help="Learning queue JSONL path.")
    parser.add_argument("--mined", action="append", default=[], help="Transcript-mined JSONL path.")
    parser.add_argument("--log", action="append", default=[], help="Operational log JSONL path.")
    parser.add_argument("--output", required=True, help="Path to write the merged review batch JSONL.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to emit.")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum repeated log clause count to keep.")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def source_hash_for(text: str, namespace: str) -> str:
    return hashlib.sha1(f"{namespace}:{text}".encode("utf-8")).hexdigest()[:16]


def review_row_from_queue(row: dict) -> dict:
    text = row.get("sanitized_text") or row.get("text") or ""
    intent = row.get("intent") or "reference_language"
    return {
        "id": row.get("candidate_id") or f"queue-{source_hash_for(text, 'queue')}",
        "text": text,
        "block": row.get("decision") == "block" or intent in v2.BLOCK_LABELS,
        "intent": intent,
        "source_type": "queue",
        "source_hash": row.get("candidate_hash") or source_hash_for(text, "queue"),
        "evidence_present": bool(row.get("evidence_present")),
        "quoted_or_code": bool(row.get("quoted_or_code")),
        "review_status": "needs_review",
        "notes": f"queue:{row.get('candidate_reason', 'unknown')}",
        "candidate_reason": row.get("candidate_reason", "unknown"),
        "candidate_ids": [row.get("candidate_id")] if row.get("candidate_id") else [],
        "pattern_family": row.get("intent") or "unknown",
    }


def review_row_from_mined(row: dict) -> dict:
    text = row["text"]
    return {
        "id": row.get("id", f"mined-{source_hash_for(text, 'mined')}"),
        "text": text,
        "block": bool(row.get("block", row["intent"] in v2.BLOCK_LABELS)),
        "intent": row["intent"],
        "source_type": row.get("source_type", "transcript"),
        "source_hash": row.get("source_hash", source_hash_for(text, "mined")),
        "evidence_present": bool(row.get("evidence_present")),
        "quoted_or_code": bool(row.get("quoted_or_code")),
        "review_status": "needs_review",
        "notes": row.get("notes", "mined"),
        "candidate_reason": row.get("candidate_reason", "mined_transcript"),
        "candidate_ids": row.get("candidate_ids", []),
        "pattern_family": row.get("pattern_family", row["intent"]),
    }


def review_rows_from_log(rows: Iterable[dict], min_cluster_size: int) -> List[dict]:
    clauses = []
    for row in rows:
        for flag in row.get("flags", []):
            clause = flag.get("clause")
            if clause:
                clauses.append(v2.sanitize_learning_text(clause))
    counts = Counter(clauses)
    review_rows = []
    for clause, count in counts.items():
        if count < min_cluster_size:
            continue
        review_rows.append(
            {
                "id": f"log-{source_hash_for(clause, 'log')}",
                "text": clause,
                "block": True,
                "intent": "assert_unverified",
                "source_type": "log_cluster",
                "source_hash": source_hash_for(clause, "log"),
                "evidence_present": bool(v2.EVIDENCE_PATTERN.search(clause)),
                "quoted_or_code": bool(v2.quoted_or_code_clause(clause)),
                "review_status": "needs_review",
                "notes": f"log-cluster:{count}",
                "candidate_reason": "log_cluster",
                "candidate_ids": [],
                "pattern_family": "log_cluster",
            }
        )
    return review_rows


def dedupe_rows(rows: List[dict], limit: int) -> List[dict]:
    deduped = {}
    for row in rows:
        key = row["text"].strip().lower()
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        merged = dict(existing)
        merged["candidate_ids"] = sorted(
            {*(existing.get("candidate_ids", []) or []), *(row.get("candidate_ids", []) or [])}
        )
        note_parts = []
        for note in (existing.get("notes", ""), row.get("notes", "")):
            if note and note not in note_parts:
                note_parts.append(note)
        merged["notes"] = "; ".join(note_parts)
        deduped[key] = merged
    return list(deduped.values())[:limit]


def main():
    args = parse_args()
    rows: List[dict] = []
    for queue_path in args.queue:
        rows.extend(review_row_from_queue(row) for row in read_jsonl(Path(queue_path)))
    for mined_path in args.mined:
        rows.extend(review_row_from_mined(row) for row in read_jsonl(Path(mined_path)))
    for log_path in args.log:
        rows.extend(review_rows_from_log(read_jsonl(Path(log_path)), args.min_cluster_size))

    output_rows = dedupe_rows(rows, args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(output_rows), "output": str(output_path)}))


if __name__ == "__main__":
    main()

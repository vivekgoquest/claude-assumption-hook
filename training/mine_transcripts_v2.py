#!/usr/bin/env python3
"""Mine sanitized assistant clauses from Claude transcript JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = REPO_ROOT / "hook"
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

import assumption_guard_v2 as v2  # pylint: disable=wrong-import-position


UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+")
PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(?:/[\w .:+@-]+)+")
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
CHANNEL_ID_RE = re.compile(r"\bUC[a-zA-Z0-9_-]{10,}\b")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/Users/vivek/.claude/projects",
        help="Root directory containing Claude transcript JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write mined JSONL rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="Maximum number of rows to emit after dedupe.",
    )
    parser.add_argument(
        "--include-safe",
        action="store_true",
        help="Keep both risky and safe-lane clauses. Default keeps both; this flag is retained for readability.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_assistant_text(entry):
    if entry.get("type") != "assistant":
        return []
    message = entry.get("message", {})
    content = message.get("content", [])
    if isinstance(content, str):
        return [content] if content.strip() else []
    texts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.strip():
                    texts.append(text)
    return texts


def sanitize_text(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = CHANNEL_ID_RE.sub("<CHANNEL_ID>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = LONG_NUMBER_RE.sub("<NUM>", text)
    return re.sub(r"\s+", " ", text).strip()


def weak_label_for_clause(clause: str):
    hard_pass = v2.hard_pass_intent(clause)
    if hard_pass:
        if hard_pass == "code_content":
            if not (
                v2.INLINE_CODE.search(clause)
                and (v2.should_consider_clause(clause) or v2.META_REPORT_PATTERN.search(clause.lower()) or v2.TYPE_WORDS.search(clause))
            ):
                return None, False
        return hard_pass, hard_pass not in v2.BLOCK_LABELS
    hard_block = v2.hard_block_decision(clause)
    if hard_block:
        return hard_block.intent, True
    if v2.should_consider_clause(clause):
        return v2.regex_only_intent(clause), True
    return None, False


def source_hash_for(path: Path, text: str) -> str:
    return hashlib.sha1(f"{path}:{text}".encode("utf-8")).hexdigest()[:16]


def mine_rows(root: Path, limit: int):
    seen = set()
    rows = []
    for transcript in root.rglob("*.jsonl"):
        for entry in read_jsonl(transcript):
            for block in extract_assistant_text(entry):
                for clause in v2.split_text_to_clauses(block):
                    sanitized = sanitize_text(clause)
                    if len(sanitized) < 12:
                        continue
                    intent, keep = weak_label_for_clause(sanitized)
                    if not keep or not intent:
                        continue
                    dedupe_key = sanitized.lower()
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    rows.append(
                        {
                            "id": f"transcript-{len(rows)+1:04d}",
                            "text": sanitized,
                            "block": intent in v2.BLOCK_LABELS,
                            "intent": intent,
                            "source_type": "transcript",
                            "source_hash": source_hash_for(transcript, sanitized),
                            "evidence_present": bool(v2.EVIDENCE_PATTERN.search(sanitized)),
                            "quoted_or_code": bool(v2.INLINE_CODE.search(sanitized) or sanitized.startswith("```")),
                            "review_status": "needs_review",
                            "notes": f"mined from {transcript.name}",
                        }
                    )
                    if len(rows) >= limit:
                        return rows
    return rows


def main():
    args = parse_args()
    root = Path(args.root)
    rows = mine_rows(root, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

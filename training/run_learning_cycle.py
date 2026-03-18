#!/usr/bin/env python3
"""Run one iterative Assumption Guard learning cycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = Path.home() / ".claude"
STATE_DIR_DEFAULT = CLAUDE_DIR / "assumption-guard-state"
QUEUE_PATH_DEFAULT = STATE_DIR_DEFAULT / "learning-queue.jsonl"
LOG_PATH_DEFAULT = CLAUDE_DIR / "assumption-guard.log.jsonl"
CURRENT_REPORT_FALLBACK = REPO_ROOT / "model" / "assumption-guard-v2-report.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=str(STATE_DIR_DEFAULT))
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--log-path", default=str(LOG_PATH_DEFAULT))
    parser.add_argument("--transcript-root", default="/Users/vivek/.claude/projects")
    parser.add_argument("--min-review-batch", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--training-python", default=os.environ.get("ASSUMPTION_GUARD_TRAIN_PYTHON") or shutil.which("python3.11") or sys.executable)
    parser.add_argument("--claude-cmd", default=os.environ.get("ASSUMPTION_GUARD_REVIEW_CLAUDE_CMD", "claude"))
    return parser.parse_args()


def run_json_command(command, cwd=REPO_ROOT):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {command}")
    return json.loads(result.stdout.strip() or "{}")


def count_jsonl(path: Path):
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_unique_jsonl(path: Path, rows, key_field: str):
    existing = set()
    if path.exists():
        existing = {row.get(key_field) for row in read_jsonl(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = row.get(key_field)
            if key in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            existing.add(key)
            appended += 1
    return appended


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def targeted_family_metrics(report: dict):
    results = {}
    rows = report.get("replay_results", {}).get("rows", [])
    for family in ["verification_narration", "dependency_gap_grounded", "capability_promise_unverified"]:
        family_rows = [row for row in rows if row.get("group") == family]
        if not family_rows:
            results[family] = 0.0
            continue
        matched = sum(1 for row in family_rows if row.get("matched"))
        results[family] = matched / len(family_rows)
    return results


def should_retrain(promote_summary: dict, reviewed_rows: list[dict]):
    if promote_summary.get("regression_promoted", 0) > 0:
        return True
    if promote_summary.get("training_promoted", 0) >= 20:
        return True
    family_counts = Counter(row.get("pattern_family", "unknown") for row in reviewed_rows)
    return any(count >= 5 for count in family_counts.values())


def better_than_current(candidate_report: dict, current_report: dict):
    candidate_families = targeted_family_metrics(candidate_report)
    current_families = targeted_family_metrics(current_report)
    candidate_replay = candidate_report["replay_results"]
    current_replay = current_report["replay_results"]
    if candidate_replay["block_recall"] < current_replay["block_recall"]:
        return False
    if candidate_replay["pass_recall"] < current_replay["pass_recall"]:
        return False
    if all(candidate_families[name] >= 0.95 for name in candidate_families):
        if (
            candidate_replay["block_recall"] == current_replay["block_recall"]
            and candidate_replay["pass_recall"] == current_replay["pass_recall"]
            and candidate_families == current_families
        ):
            return False
        return True
    return False


def main():
    args = parse_args()
    state_dir = Path(args.state_dir)
    queue_path = Path(args.queue_path)
    log_path = Path(args.log_path)
    lock_path = state_dir / "learning-cycle.lock"
    mined_path = state_dir / "mined-review-input.jsonl"
    review_batch_path = state_dir / "review-batch.jsonl"
    reviewed_output_path = state_dir / "reviewed-claude.raw.jsonl"
    reviewed_merged_path = state_dir / "reviewed-claude.jsonl"
    quarantine_dir = state_dir / "quarantine"
    training_overlay = state_dir / "training-overlay.jsonl"
    regression_overlay = state_dir / "regression-overlay.jsonl"
    replay_overlay = state_dir / "replay-overlay.jsonl"
    current_report_path = state_dir / "current-model-report.json"
    candidates_root = state_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    try:
        with file_lock(lock_path):
            run_json_command(
                [
                    args.training_python,
                    str(REPO_ROOT / "training" / "mine_transcripts_v2.py"),
                    "--root",
                    args.transcript_root,
                    "--output",
                    str(mined_path),
                ]
            )
            run_json_command(
                [
                    args.training_python,
                    str(REPO_ROOT / "training" / "build_review_batch.py"),
                    "--queue",
                    str(queue_path),
                    "--mined",
                    str(mined_path),
                    "--log",
                    str(log_path),
                    "--output",
                    str(review_batch_path),
                    "--limit",
                    str(args.max_candidates),
                ]
            )
            review_rows = read_jsonl(review_batch_path)
            reason_counts = Counter(row.get("candidate_reason", "unknown") for row in review_rows)
            if len(review_rows) < args.min_review_batch and max(reason_counts.values(), default=0) < 3:
                print(json.dumps({"status": "skipped", "reason": "insufficient_review_batch", "rows": len(review_rows)}))
                return

            run_json_command(
                [
                    args.training_python,
                    str(REPO_ROOT / "training" / "review_with_claude.py"),
                    "--batch",
                    str(review_batch_path),
                    "--output",
                    str(reviewed_output_path),
                    "--quarantine-dir",
                    str(quarantine_dir),
                    "--claude-cmd",
                    args.claude_cmd,
                    "--max-candidates",
                    str(args.max_candidates),
                ]
            )

            reviewed_rows = read_jsonl(reviewed_output_path)
            batch_by_id = {row.get("id") or row.get("candidate_id"): row for row in review_rows}
            merged_rows = []
            for reviewed in reviewed_rows:
                source = batch_by_id.get(reviewed["candidate_id"], {})
                merged_rows.append({**source, **reviewed})
            append_unique_jsonl(reviewed_merged_path, merged_rows, "candidate_id")

            promote_summary = run_json_command(
                [
                    args.training_python,
                    str(REPO_ROOT / "training" / "promote_reviewed_examples.py"),
                    "--reviewed",
                    str(reviewed_merged_path),
                    "--training-overlay",
                    str(training_overlay),
                    "--regression-overlay",
                    str(regression_overlay),
                    "--replay-overlay",
                    str(replay_overlay),
                ]
            )
            if not should_retrain(promote_summary, merged_rows):
                print(json.dumps({"status": "captured", "promoted": promote_summary, "retrain": False}))
                return

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            candidate_dir = candidates_root / timestamp
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model_path = candidate_dir / "assumption-guard-v2.onnx"
            tokenizer_path = candidate_dir / "assumption-guard-v2-tokenizer.json"
            meta_path = candidate_dir / "assumption-guard-v2-meta.json"
            report_path = candidate_dir / "assumption-guard-v2-report.json"

            run_json_command(
                [
                    args.training_python,
                    str(REPO_ROOT / "training" / "train_v2.py"),
                    "--overlay-training-data",
                    str(training_overlay),
                    "--overlay-regression-cases",
                    str(regression_overlay),
                    "--overlay-replay-cases",
                    str(replay_overlay),
                    "--output-model",
                    str(model_path),
                    "--output-tokenizer",
                    str(tokenizer_path),
                    "--output-meta",
                    str(meta_path),
                    "--output-report",
                    str(report_path),
                ]
            )

            candidate_report = json.loads(report_path.read_text())
            current_report = json.loads((current_report_path if current_report_path.exists() else CURRENT_REPORT_FALLBACK).read_text())
            family_metrics = targeted_family_metrics(candidate_report)
            accepted = (
                candidate_report["regression_results"]["matched"] == candidate_report["regression_results"]["total"]
                and candidate_report["replay_results"]["block_recall"] >= 0.95
                and candidate_report["replay_results"]["pass_recall"] >= 0.95
                and all(value >= 0.95 for value in family_metrics.values())
                and better_than_current(candidate_report, current_report)
            )
            if not accepted:
                print(json.dumps({"status": "trained", "promoted": False, "candidate_dir": str(candidate_dir)}))
                return

            live_model = CLAUDE_DIR / "assumption-guard-v2.onnx"
            live_tokenizer = CLAUDE_DIR / "assumption-guard-v2-tokenizer.json"
            live_meta = CLAUDE_DIR / "assumption-guard-v2-meta.json"
            os.replace(model_path, live_model)
            os.replace(tokenizer_path, live_tokenizer)
            os.replace(meta_path, live_meta)
            current_report_path.write_text(json.dumps(candidate_report, indent=2))
            print(json.dumps({"status": "promoted", "candidate_dir": str(candidate_dir), "family_metrics": family_metrics}))
    except FileExistsError:
        print(json.dumps({"status": "skipped", "reason": "lock_exists", "lock_path": str(lock_path)}))


if __name__ == "__main__":
    main()

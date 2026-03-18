#!/usr/bin/env python3
"""Launch the learning cycle only when objective pending-queue thresholds are met."""

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
REVIEWED_PATH_DEFAULT = STATE_DIR_DEFAULT / "reviewed-claude.jsonl"
REVIEW_STATE_PATH_DEFAULT = STATE_DIR_DEFAULT / "review-state.json"
RUN_SCRIPT_DEFAULT = REPO_ROOT / "training" / "run_learning_cycle.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default=str(STATE_DIR_DEFAULT))
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--log-path", default=str(LOG_PATH_DEFAULT))
    parser.add_argument("--reviewed-path", default=str(REVIEWED_PATH_DEFAULT))
    parser.add_argument("--review-state-path", default=str(REVIEW_STATE_PATH_DEFAULT))
    parser.add_argument("--run-script", default=str(RUN_SCRIPT_DEFAULT))
    parser.add_argument("--training-python", default=os.environ.get("ASSUMPTION_GUARD_TRAIN_PYTHON") or shutil.which("python3.11") or sys.executable)
    parser.add_argument("--claude-cmd", default=os.environ.get("ASSUMPTION_GUARD_REVIEW_CLAUDE_CMD", "claude"))
    parser.add_argument("--transcript-root", default="/Users/vivek/.claude/projects")
    parser.add_argument("--pending-threshold", type=int, default=25)
    parser.add_argument("--same-reason-threshold", type=int, default=5)
    parser.add_argument("--same-family-threshold", type=int, default=4)
    parser.add_argument("--oldest-age-seconds", type=int, default=12 * 60 * 60)
    parser.add_argument("--cooldown-seconds", type=int, default=2 * 60 * 60)
    parser.add_argument("--foreground", action="store_true", help="Run the learning cycle synchronously for testing.")
    return parser.parse_args()


def read_jsonl(path: Path):
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


def read_state(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def parse_ts(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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


def processed_hashes(reviewed_rows):
    hashes = set()
    for row in reviewed_rows:
        if row.get("candidate_hash"):
            hashes.add(row["candidate_hash"])
        elif row.get("candidate_id"):
            hashes.add(row["candidate_id"])
    return hashes


def pending_rows(queue_rows, reviewed_rows):
    seen = processed_hashes(reviewed_rows)
    pending = []
    for row in queue_rows:
        key = row.get("candidate_hash") or row.get("candidate_id")
        if key in seen:
            continue
        pending.append(row)
    return pending


def oldest_pending_age_seconds(rows):
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    ages = []
    for row in rows:
        ts = parse_ts(row.get("ts"))
        if ts is None:
            continue
        ages.append(max(0.0, (now - ts).total_seconds()))
    return max(ages) if ages else 0


def trigger_reason(rows, args):
    if not rows:
        return None, {}
    reason_counts = Counter(row.get("candidate_reason", "unknown") for row in rows)
    family_counts = Counter(row.get("pattern_family") or row.get("intent") or "unknown" for row in rows)
    oldest_age = oldest_pending_age_seconds(rows)

    if len(rows) >= args.pending_threshold:
        return "pending_rows", {
            "pending_rows": len(rows),
            "largest_reason_cluster": max(reason_counts.values(), default=0),
            "largest_family_cluster": max(family_counts.values(), default=0),
            "oldest_pending_age_seconds": oldest_age,
        }
    if reason_counts:
        top_reason, top_reason_count = reason_counts.most_common(1)[0]
        if top_reason_count >= args.same_reason_threshold:
            return f"candidate_reason:{top_reason}", {
                "pending_rows": len(rows),
                "largest_reason_cluster": top_reason_count,
                "largest_family_cluster": max(family_counts.values(), default=0),
                "oldest_pending_age_seconds": oldest_age,
            }
    if family_counts:
        top_family, top_family_count = family_counts.most_common(1)[0]
        if top_family_count >= args.same_family_threshold:
            return f"pattern_family:{top_family}", {
                "pending_rows": len(rows),
                "largest_reason_cluster": max(reason_counts.values(), default=0),
                "largest_family_cluster": top_family_count,
                "oldest_pending_age_seconds": oldest_age,
            }
    if oldest_age >= args.oldest_age_seconds:
        return "oldest_pending_age", {
            "pending_rows": len(rows),
            "largest_reason_cluster": max(reason_counts.values(), default=0),
            "largest_family_cluster": max(family_counts.values(), default=0),
            "oldest_pending_age_seconds": oldest_age,
        }
    return None, {
        "pending_rows": len(rows),
        "largest_reason_cluster": max(reason_counts.values(), default=0),
        "largest_family_cluster": max(family_counts.values(), default=0),
        "oldest_pending_age_seconds": oldest_age,
    }


def cooldown_remaining_seconds(state: dict, cooldown_seconds: int):
    last_cycle = parse_ts(state.get("last_cycle_ts"))
    if last_cycle is None:
        return 0
    elapsed = (datetime.now(timezone.utc) - last_cycle).total_seconds()
    return max(0, int(cooldown_seconds - elapsed))


def launch_learning_cycle(args, state_dir: Path, log_path: Path):
    command = [
        args.training_python,
        str(Path(args.run_script)),
        "--state-dir",
        str(state_dir),
        "--queue-path",
        str(Path(args.queue_path)),
        "--log-path",
        str(log_path),
        "--transcript-root",
        args.transcript_root,
        "--claude-cmd",
        args.claude_cmd,
        "--training-python",
        args.training_python,
    ]
    if args.foreground:
        result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "learning cycle failed")
        return {"mode": "foreground", "stdout": result.stdout.strip()}

    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_log = logs_dir / f"learning-cycle-{launched_at}.log"
    with output_log.open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    return {"mode": "detached", "log_path": str(output_log)}


def main():
    args = parse_args()
    state_dir = Path(args.state_dir)
    queue_path = Path(args.queue_path)
    log_path = Path(args.log_path)
    reviewed_path = Path(args.reviewed_path)
    review_state_path = Path(args.review_state_path)
    learning_lock_path = state_dir / "learning-cycle.lock"
    trigger_lock_path = state_dir / "trigger-check.lock"
    now = datetime.now(timezone.utc).isoformat()

    try:
        with file_lock(trigger_lock_path):
            queue_rows = read_jsonl(queue_path)
            reviewed_rows = read_jsonl(reviewed_path)
            pending = pending_rows(queue_rows, reviewed_rows)
            state = read_state(review_state_path)

            if learning_lock_path.exists():
                state.update(
                    {
                        "last_check_ts": now,
                        "pending_rows": len(pending),
                        "last_decision": "skipped",
                        "last_skip_reason": "learning_cycle_running",
                    }
                )
                write_state(review_state_path, state)
                print(json.dumps({"status": "skipped", "reason": "learning_cycle_running", "pending_rows": len(pending)}))
                return

            reason, metrics = trigger_reason(pending, args)
            remaining = cooldown_remaining_seconds(state, args.cooldown_seconds)
            if reason and remaining > 0:
                state.update(
                    {
                        "last_check_ts": now,
                        "pending_rows": len(pending),
                        "last_decision": "skipped",
                        "last_skip_reason": "cooldown_active",
                        "cooldown_remaining_seconds": remaining,
                        **metrics,
                    }
                )
                write_state(review_state_path, state)
                print(json.dumps({"status": "skipped", "reason": "cooldown_active", "pending_rows": len(pending), "cooldown_remaining_seconds": remaining}))
                return

            if reason is None:
                state.update(
                    {
                        "last_check_ts": now,
                        "pending_rows": len(pending),
                        "last_decision": "skipped",
                        "last_skip_reason": "threshold_not_met",
                        **metrics,
                    }
                )
                write_state(review_state_path, state)
                print(json.dumps({"status": "skipped", "reason": "threshold_not_met", "pending_rows": len(pending), **metrics}))
                return

            launch_info = launch_learning_cycle(args, state_dir, log_path)
            state.update(
                {
                    "last_check_ts": now,
                    "last_cycle_ts": now,
                    "pending_rows": len(pending),
                    "last_decision": "launched",
                    "last_trigger_reason": reason,
                    **metrics,
                    **launch_info,
                }
            )
            write_state(review_state_path, state)
            print(json.dumps({"status": "launched", "trigger_reason": reason, "pending_rows": len(pending), **metrics, **launch_info}))
    except FileExistsError:
        print(json.dumps({"status": "skipped", "reason": "trigger_check_running", "lock_path": str(trigger_lock_path)}))


if __name__ == "__main__":
    main()

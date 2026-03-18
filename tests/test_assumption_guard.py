import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hook" / "assumption-guard.py"
MODEL_PATH = REPO_ROOT / "model" / "assumption-guard-v2.onnx"
TOKENIZER_PATH = REPO_ROOT / "model" / "assumption-guard-v2-tokenizer.json"
META_PATH = REPO_ROOT / "model" / "assumption-guard-v2-meta.json"
REPORT_PATH = REPO_ROOT / "model" / "assumption-guard-v2-report.json"
REGRESSION_CASES_PATH = REPO_ROOT / "training" / "assumption-guard-regression-cases.json"
LABELED_DATA_PATH = REPO_ROOT / "training" / "assumption-guard-training-labeled.jsonl"
LEGACY_RUNTIME_PATH = REPO_ROOT / "hook" / "assumption_guard_v2.py"
LEGACY_MODEL_HELPER_PATH = REPO_ROOT / "hook" / "assumption_guard_model.py"
LEGACY_TRAINING_SCRIPT = REPO_ROOT / "training" / "train_assumption_guard.py"
LEGACY_MODEL_PATH = REPO_ROOT / "model" / "assumption-guard-model.pkl"
LEGACY_METRICS_PATH = REPO_ROOT / "model" / "assumption-guard-metrics.json"
LEGACY_EVAL_PATH = REPO_ROOT / "model" / "assumption-guard-eval.json"
LEGACY_HELPER_SCRIPTS = [
    REPO_ROOT / "training" / "run_learning_cycle.py",
    REPO_ROOT / "training" / "maybe_trigger_learning_cycle.py",
    REPO_ROOT / "training" / "build_review_batch.py",
    REPO_ROOT / "training" / "review_with_claude.py",
    REPO_ROOT / "training" / "promote_reviewed_examples.py",
    REPO_ROOT / "training" / "train_v2.py",
    REPO_ROOT / "training" / "replay_eval_v2.py",
    REPO_ROOT / "training" / "mine_transcripts_v2.py",
]
TRAINING_PYTHON = os.environ.get("ASSUMPTION_GUARD_TRAIN_PYTHON") or shutil.which("python3.11") or sys.executable


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("assumption_guard_runtime", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def regression_cases():
    return json.loads(REGRESSION_CASES_PATH.read_text())


def run_mode(mode, *args, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [TRAINING_PYTHON, str(HOOK_PATH), mode, *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=merged_env,
        check=False,
    )


def run_hook(
    assistant_text,
    *,
    backend="v2",
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    meta_path=META_PATH,
    import_blocker=False,
    extra_env=None,
    keep_tmpdir=False,
):
    temp_context = tempfile.TemporaryDirectory() if not keep_tmpdir else None
    try:
        tmp = Path(temp_context.name if temp_context is not None else tempfile.mkdtemp())
        transcript_path = tmp / "transcript.jsonl"
        log_path = tmp / "assumption-guard.log.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "user", "message": {"content": "review this change"}}),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": assistant_text}]
                            },
                        }
                    ),
                ]
            )
        )
        payload = {
            "session_id": "test-session",
            "transcript_path": str(transcript_path),
            "cwd": str(REPO_ROOT),
            "stop_hook_active": False,
        }
        env = os.environ.copy()
        env["ASSUMPTION_GUARD_BACKEND"] = backend
        env["ASSUMPTION_GUARD_MODEL_PATH"] = str(model_path)
        env["ASSUMPTION_GUARD_TOKENIZER_PATH"] = str(tokenizer_path)
        env["ASSUMPTION_GUARD_META_PATH"] = str(meta_path)
        env["ASSUMPTION_GUARD_LOG_PATH"] = str(log_path)
        env["ASSUMPTION_GUARD_TRIGGER_MODE"] = "off"
        if extra_env:
            env.update(extra_env)
        if import_blocker:
            blocker = f"""
import builtins
import runpy

real_import = builtins.__import__
blocked = {{"onnxruntime", "tokenizers"}}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in blocked:
        raise ModuleNotFoundError(f"No module named '{{name.split('.')[0]}}'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
runpy.run_path({str(HOOK_PATH)!r}, run_name="__main__")
"""
            command = [sys.executable, "-c", blocker]
        else:
            command = [sys.executable, str(HOOK_PATH)]

        result = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        log_entries = []
        if log_path.exists():
            log_entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        return result, log_entries, tmp
    finally:
        if temp_context is not None:
            temp_context.cleanup()


class AssumptionGuardRuntimeTests(unittest.TestCase):
    def test_clause_splitter_breaks_conservative_subclauses(self):
        runtime = load_runtime_module()
        clauses = runtime.split_text_to_clauses(
            "Maybe rename this helper to be clearer; I checked the repo, so renaming it should be safe. "
            "The return value could be None or a string — if the callback throws, the promise may reject."
        )
        self.assertEqual(
            clauses,
            [
                "Maybe rename this helper to be clearer",
                "I checked the repo",
                "renaming it should be safe.",
                "The return value could be None or a string",
                "if the callback throws, the promise may reject.",
            ],
        )

    def test_regression_cases_follow_v2_policy(self):
        cases = regression_cases()
        self.assertGreaterEqual(len(cases), 40)
        for case in cases:
            with self.subTest(text=case["text"]):
                result, _, _ = run_hook(case["text"])
                self.assertEqual(result.returncode, 0, result.stderr)
                if case["expected_block"]:
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload["decision"], "block")
                else:
                    self.assertEqual(result.stdout.strip(), "", result.stdout)

    def test_block_reason_includes_clause_and_predicted_intent(self):
        result, _, _ = run_hook("Maybe rename this helper to be clearer")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "block")
        self.assertIn("Maybe rename this helper to be clearer", payload["reason"])
        self.assertIn("recommend_unverified", payload["reason"])

    def test_verification_narration_and_grounded_limitations_follow_policy(self):
        verification_narration = (
            "Let me verify whether I can actually identify the new videos and remove them."
        )
        grounded_limitation = (
            "I confirmed DELETE /videos/bulk exists and requires ids. "
            "But without a saved list of the newly created ids, I can't selectively remove them."
        )
        unsupported_capability = (
            "But if you want to revert, I can check which ones are new and remove them."
        )

        result, _, _ = run_hook(verification_narration)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", result.stdout)

        result, _, _ = run_hook(grounded_limitation)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", result.stdout)

        result, _, _ = run_hook(unsupported_capability)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "block")

    def test_learning_queue_captures_blocked_and_heuristic_rescue_cases(self):
        state_dir = Path(tempfile.mkdtemp())
        queue_path = state_dir / "learning-queue.jsonl"
        common_env = {
            "ASSUMPTION_GUARD_STATE_DIR": str(state_dir),
            "ASSUMPTION_GUARD_QUEUE_PATH": str(queue_path),
        }

        blocked = "But if you want to revert, I can check which ones are new and remove them."
        rescued = "Let me verify whether I can actually identify the new videos and remove them."

        result, _, _ = run_hook(blocked, extra_env=common_env)
        self.assertEqual(result.returncode, 0, result.stderr)
        result, _, _ = run_hook(rescued, extra_env=common_env)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertTrue(queue_path.exists(), "expected learning queue to be written")
        rows = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
        reasons = {row["candidate_reason"] for row in rows}
        self.assertIn("blocked_clause", reasons)
        self.assertIn("heuristic_rescue", reasons)
        for row in rows:
            self.assertIn("candidate_hash", row)
            self.assertIn("sanitized_text", row)

    def test_disable_env_bypasses_hook_and_queue_capture(self):
        state_dir = Path(tempfile.mkdtemp())
        queue_path = state_dir / "learning-queue.jsonl"
        result, log_entries, tmpdir = run_hook(
            "Maybe rename this helper to be clearer",
            extra_env={
                "ASSUMPTION_GUARD_DISABLE": "1",
                "ASSUMPTION_GUARD_STATE_DIR": str(state_dir),
                "ASSUMPTION_GUARD_QUEUE_PATH": str(queue_path),
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(log_entries, [])
        self.assertFalse(queue_path.exists())
        self.assertFalse((tmpdir / "assumption-guard.log.jsonl").exists())

    def test_missing_onnx_dependencies_fall_back_to_regex_only(self):
        result, log_entries, _ = run_hook(
            "I think the timeout is 30 seconds",
            import_blocker=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")
        self.assertTrue(log_entries, "expected a log entry")
        latest = log_entries[-1]
        self.assertEqual(latest["stage"], "regex_only")
        self.assertEqual(latest["backend"], "regex")
        self.assertFalse(latest["ml_available"])
        self.assertEqual(latest["fallback_reason"], "ml_import_error")
        self.assertEqual(latest["error_stage"], "ml_import")

    def test_corrupt_v2_assets_fall_back_to_regex_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corrupt_model = tmp / "broken-model.onnx"
            corrupt_tokenizer = tmp / "tokenizer.json"
            corrupt_meta = tmp / "meta.json"
            corrupt_model.write_text("not an onnx model")
            corrupt_tokenizer.write_text("{\"version\":\"1.0\"}")
            corrupt_meta.write_text("{\"threshold\":0.5,\"labels\":[\"assert_unverified\"]}")
            result, log_entries, _ = run_hook(
                "I think the timeout is 30 seconds",
                model_path=corrupt_model,
                tokenizer_path=corrupt_tokenizer,
                meta_path=corrupt_meta,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")
        self.assertTrue(log_entries, "expected a log entry")
        latest = log_entries[-1]
        self.assertEqual(latest["stage"], "regex_only")
        self.assertEqual(latest["backend"], "regex")
        self.assertFalse(latest["ml_available"])
        self.assertIn(latest["fallback_reason"], {"asset_load_error", "model_load_error"})
        self.assertIn(latest["error_stage"], {"model_load", "tokenizer_load", "meta_load"})


class AssumptionGuardTrainingAndArtifactsTests(unittest.TestCase):
    def test_single_hook_entrypoint_exists(self):
        self.assertTrue(HOOK_PATH.exists())

    def test_repo_only_keeps_single_runtime_and_v2_artifacts(self):
        self.assertFalse(LEGACY_RUNTIME_PATH.exists())
        self.assertFalse(LEGACY_MODEL_HELPER_PATH.exists())
        self.assertFalse(LEGACY_TRAINING_SCRIPT.exists())
        self.assertFalse(LEGACY_MODEL_PATH.exists())
        self.assertFalse(LEGACY_METRICS_PATH.exists())
        self.assertFalse(LEGACY_EVAL_PATH.exists())
        for path in LEGACY_HELPER_SCRIPTS:
            self.assertFalse(path.exists(), path)

    def test_replay_script_runs_against_committed_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "replay-report.json"
            result = subprocess.run(
                [TRAINING_PYTHON, str(HOOK_PATH), "replay", "--regression-cases", str(REGRESSION_CASES_PATH), "--model", str(MODEL_PATH), "--tokenizer", str(TOKENIZER_PATH), "--meta", str(META_PATH), "--output", str(output_path)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            replay = json.loads(output_path.read_text())
            self.assertEqual(replay["summary"]["matched"], replay["summary"]["total"])

    def test_build_review_batch_merges_and_dedupes_queue_and_mined_rows(self):
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            queue_path = tmp / "queue.jsonl"
            mined_path = tmp / "mined.jsonl"
            queue_rows = [
                {
                    "candidate_id": "q-1",
                    "sanitized_text": "Let me verify whether I can actually identify the new videos and remove them.",
                    "candidate_reason": "heuristic_rescue",
                    "intent": "verification_narration",
                    "decision": "pass",
                    "review_status": "needs_review",
                }
            ]
            mined_rows = [
                {
                    "id": "m-1",
                    "text": "Let me verify whether I can actually identify the new videos and remove them.",
                    "block": False,
                    "intent": "verification_narration",
                    "source_type": "transcript",
                    "source_hash": "abc123",
                    "evidence_present": False,
                    "quoted_or_code": False,
                    "review_status": "needs_review",
                    "notes": "mined",
                },
                {
                    "id": "m-2",
                    "text": "But if you want to revert, I can check which ones are new and remove them.",
                    "block": True,
                    "intent": "capability_promise_unverified",
                    "source_type": "transcript",
                    "source_hash": "def456",
                    "evidence_present": False,
                    "quoted_or_code": False,
                    "review_status": "needs_review",
                    "notes": "mined",
                },
            ]
            queue_path.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n")
            mined_path.write_text("\n".join(json.dumps(row) for row in mined_rows) + "\n")
            rows = runtime.build_review_batch_rows(
                [queue_path],
                [mined_path],
                [],
                limit=200,
            )
            self.assertEqual(len(rows), 2)
            texts = {row["text"] for row in rows}
            self.assertIn("Let me verify whether I can actually identify the new videos and remove them.", texts)
            self.assertIn("But if you want to revert, I can check which ones are new and remove them.", texts)

    def test_review_prompt_is_strict_and_examples_are_present(self):
        review = load_runtime_module()
        rows = [
            {
                "id": "candidate-a",
                "text": "Let me verify whether I can actually identify the new videos and remove them.",
                "intent": "verification_narration",
            }
        ]
        prompt = review.build_review_prompt(rows)
        self.assertIn("You are a labeling worker for Assumption Guard.", prompt)
        self.assertIn("You are not solving the user's problem.", prompt)
        self.assertIn("Preserve candidate_id exactly and keep the same order as input.", prompt)
        self.assertIn("Do not invent new labels.", prompt)
        self.assertIn("Failure conditions:", prompt)
        self.assertIn("verification_narration", prompt)
        self.assertIn("capability_promise_unverified", prompt)
        self.assertIn("dependency_gap_grounded", prompt)

    def test_review_output_validation_rejects_drift(self):
        review = load_runtime_module()
        input_rows = [
            {"id": "candidate-a"},
            {"id": "candidate-b"},
        ]
        valid = json.dumps(
            [
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "verification_narration",
                    "final_block": False,
                    "confidence": 0.91,
                    "rationale": "The clause says it will verify now.",
                    "pattern_family": "verification_narration",
                },
                {
                    "candidate_id": "candidate-b",
                    "final_intent": "capability_promise_unverified",
                    "final_block": True,
                    "confidence": 0.94,
                    "rationale": "The clause promises action without evidence.",
                    "pattern_family": "capability_promise_unverified",
                },
            ]
        )
        parsed = review.parse_review_output(valid, input_rows)
        self.assertEqual(len(parsed), 2)

        wrong_order = json.dumps(
            [
                {
                    "candidate_id": "candidate-b",
                    "final_intent": "capability_promise_unverified",
                    "final_block": True,
                    "confidence": 0.94,
                    "rationale": "The clause promises action without evidence.",
                    "pattern_family": "capability_promise_unverified",
                },
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "verification_narration",
                    "final_block": False,
                    "confidence": 0.91,
                    "rationale": "The clause says it will verify now.",
                    "pattern_family": "verification_narration",
                },
            ]
        )
        with self.assertRaises(ValueError):
            review.parse_review_output(wrong_order, input_rows)

        extra_key = json.dumps(
            [
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "verification_narration",
                    "final_block": False,
                    "confidence": 0.91,
                    "rationale": "The clause says it will verify now.",
                    "pattern_family": "verification_narration",
                    "extra": "nope",
                },
                {
                    "candidate_id": "candidate-b",
                    "final_intent": "capability_promise_unverified",
                    "final_block": True,
                    "confidence": 0.94,
                    "rationale": "The clause promises action without evidence.",
                    "pattern_family": "capability_promise_unverified",
                },
            ]
        )
        with self.assertRaises(ValueError):
            review.parse_review_output(extra_key, input_rows)

    def test_promote_reviewed_examples_routes_rows_to_overlays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reviewed_path = tmp / "reviewed.jsonl"
            training_overlay = tmp / "training-overlay.jsonl"
            regression_overlay = tmp / "regression-overlay.jsonl"
            replay_overlay = tmp / "replay-overlay.jsonl"
            reviewed_rows = [
                {
                    "candidate_id": "r-1",
                    "text": "Let me verify whether I can actually identify the new videos and remove them.",
                    "final_intent": "verification_narration",
                    "final_block": False,
                    "confidence": 0.92,
                    "pattern_family": "verification_narration",
                    "candidate_reason": "heuristic_rescue",
                },
                {
                    "candidate_id": "r-2",
                    "text": "But if you want to revert, I can check which ones are new and remove them.",
                    "final_intent": "capability_promise_unverified",
                    "final_block": True,
                    "confidence": 0.95,
                    "pattern_family": "capability_promise_unverified",
                    "candidate_reason": "blocked_clause",
                },
                {
                    "candidate_id": "r-3",
                    "text": "I confirmed DELETE /videos/bulk exists and requires ids. But without a saved list of the newly created ids, I can't selectively remove them.",
                    "final_intent": "dependency_gap_grounded",
                    "final_block": False,
                    "confidence": 0.72,
                    "pattern_family": "dependency_gap_grounded",
                    "candidate_reason": "mixed_evidence_gap",
                },
            ]
            reviewed_path.write_text("\n".join(json.dumps(row) for row in reviewed_rows) + "\n")
            result = run_mode(
                "promote",
                "--reviewed",
                reviewed_path,
                "--training-overlay",
                training_overlay,
                "--regression-overlay",
                regression_overlay,
                "--replay-overlay",
                replay_overlay,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            training_rows = [json.loads(line) for line in training_overlay.read_text().splitlines() if line.strip()]
            regression_rows = [json.loads(line) for line in regression_overlay.read_text().splitlines() if line.strip()]
            replay_rows = [json.loads(line) for line in replay_overlay.read_text().splitlines() if line.strip()]
            self.assertEqual(len(training_rows), 3)
            self.assertGreaterEqual(len(regression_rows), 2)
            self.assertGreaterEqual(len(replay_rows), 1)

    def test_learning_cycle_reviews_and_promotes_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            reviewer_script = tmp / "fake_claude.py"
            reviewer_script.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        "payload = [",
                        "  {",
                        '    "candidate_id": "candidate-a",',
                        '    "final_intent": "verification_narration",',
                        '    "final_block": False,',
                        '    "confidence": 0.93,',
                        '    "rationale": "explicit verification narration",',
                        '    "pattern_family": "verification_narration"',
                        "  },",
                        "  {",
                        '    "candidate_id": "candidate-b",',
                        '    "final_intent": "capability_promise_unverified",',
                        '    "final_block": True,',
                        '    "confidence": 0.95,',
                        '    "rationale": "unsupported capability promise",',
                        '    "pattern_family": "capability_promise_unverified"',
                        "  }",
                        "]",
                        "print(json.dumps(payload))",
                    ]
                )
            )
            queue_rows = [
                {
                    "candidate_id": "candidate-a",
                    "candidate_hash": "hash-a",
                    "sanitized_text": "Let me verify whether I can actually identify the new videos and remove them.",
                    "text": "Let me verify whether I can actually identify the new videos and remove them.",
                    "decision": "pass",
                    "intent": "verification_narration",
                    "candidate_reason": "blocked_clause",
                    "evidence_present": False,
                    "quoted_or_code": False,
                },
                {
                    "candidate_id": "candidate-b",
                    "candidate_hash": "hash-b",
                    "sanitized_text": "But if you want to revert, I can check which ones are new and remove them.",
                    "text": "But if you want to revert, I can check which ones are new and remove them.",
                    "decision": "block",
                    "intent": "capability_promise_unverified",
                    "candidate_reason": "blocked_clause",
                    "evidence_present": False,
                    "quoted_or_code": False,
                },
            ]
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n")
            result = run_mode(
                "learning-cycle",
                "--state-dir",
                state_dir,
                "--queue-path",
                queue_path,
                "--log-path",
                log_path,
                "--transcript-root",
                transcript_root,
                "--min-review-batch",
                "1",
                "--claude-cmd",
                f"{sys.executable} {reviewer_script}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "captured")
            self.assertFalse(payload["retrain"])
            training_overlay = [json.loads(line) for line in (state_dir / "training-overlay.jsonl").read_text().splitlines() if line.strip()]
            self.assertEqual(len(training_overlay), 2)
            reviewed_rows = [json.loads(line) for line in (state_dir / "reviewed-claude.jsonl").read_text().splitlines() if line.strip()]
            self.assertEqual(len(reviewed_rows), 2)

    def test_maybe_trigger_uses_pending_rows_not_total_queue_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            reviewed_path = state_dir / "reviewed-claude.jsonl"
            review_state_path = state_dir / "review-state.json"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_rows = [
                {
                    "candidate_id": "candidate-a",
                    "candidate_hash": "hash-a",
                    "candidate_reason": "blocked_clause",
                    "intent": "capability_promise_unverified",
                    "ts": "2026-03-18T00:00:00+00:00",
                },
                {
                    "candidate_id": "candidate-b",
                    "candidate_hash": "hash-b",
                    "candidate_reason": "blocked_clause",
                    "intent": "capability_promise_unverified",
                    "ts": "2026-03-18T00:00:00+00:00",
                },
                {
                    "candidate_id": "candidate-c",
                    "candidate_hash": "hash-c",
                    "candidate_reason": "heuristic_rescue",
                    "intent": "verification_narration",
                    "ts": "2026-03-18T00:00:00+00:00",
                },
            ]
            reviewed_rows = [
                {"candidate_id": "candidate-a", "candidate_hash": "hash-a"},
                {"candidate_id": "candidate-b", "candidate_hash": "hash-b"},
            ]
            queue_path.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n")
            reviewed_path.write_text("\n".join(json.dumps(row) for row in reviewed_rows) + "\n")
            result = run_mode(
                "maybe-trigger",
                "--state-dir",
                state_dir,
                "--queue-path",
                queue_path,
                "--reviewed-path",
                reviewed_path,
                "--review-state-path",
                review_state_path,
                "--pending-threshold",
                "2",
                "--same-reason-threshold",
                "3",
                "--same-family-threshold",
                "3",
                "--oldest-age-seconds",
                "999999",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "skipped")
            self.assertEqual(payload["reason"], "threshold_not_met")
            self.assertEqual(payload["pending_rows"], 1)

    def test_maybe_trigger_launches_learning_cycle_when_threshold_met(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            reviewed_path = state_dir / "reviewed-claude.jsonl"
            review_state_path = state_dir / "review-state.json"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_rows = [
                {
                    "candidate_id": "candidate-a",
                    "candidate_hash": "hash-a",
                    "candidate_reason": "blocked_clause",
                    "intent": "capability_promise_unverified",
                    "ts": "2026-03-18T00:00:00+00:00",
                },
                {
                    "candidate_id": "candidate-b",
                    "candidate_hash": "hash-b",
                    "candidate_reason": "blocked_clause",
                    "intent": "capability_promise_unverified",
                    "ts": "2026-03-18T00:00:00+00:00",
                },
            ]
            queue_path.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n")
            result = run_mode(
                "maybe-trigger",
                "--state-dir",
                state_dir,
                "--queue-path",
                queue_path,
                "--reviewed-path",
                reviewed_path,
                "--review-state-path",
                review_state_path,
                "--transcript-root",
                transcript_root,
                "--pending-threshold",
                "2",
                "--cooldown-seconds",
                "0",
                "--foreground",
                env={
                    "ASSUMPTION_GUARD_DISABLE": "1",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "launched")
            self.assertEqual(payload["trigger_reason"], "pending_rows")
            review_state = json.loads(review_state_path.read_text())
            self.assertEqual(review_state["last_decision"], "launched")

    def test_committed_report_meets_acceptance_gates(self):
        report = json.loads(REPORT_PATH.read_text())
        self.assertIn("candidates", report)
        self.assertIn("selected_candidate", report)
        self.assertIn("replay_results", report)
        self.assertIn("regression_results", report)
        self.assertIn("threshold_search", report)
        self.assertIn("latency", report)
        self.assertTrue(report["acceptance"]["meets_acceptance_bar"], report["acceptance"])
        self.assertGreaterEqual(report["replay_results"]["block_recall"], 0.95)
        self.assertGreaterEqual(report["replay_results"]["pass_recall"], 0.95)
        self.assertEqual(report["regression_results"]["matched"], report["regression_results"]["total"])

    def test_v2_assets_exist(self):
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(TOKENIZER_PATH.exists())
        self.assertTrue(META_PATH.exists())
        self.assertTrue(REPORT_PATH.exists())


if __name__ == "__main__":
    unittest.main()

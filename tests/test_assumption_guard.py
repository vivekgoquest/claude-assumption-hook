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
HOOK_PACKAGE_DIR = REPO_ROOT / "hook" / "assumption-guard"
HOOK_PATH = HOOK_PACKAGE_DIR / "assumption-guard.py"
INSTALLER_PATH = HOOK_PACKAGE_DIR / "install.sh"
MODEL_PATH = HOOK_PACKAGE_DIR / "assumption-guard-v2.onnx"
TOKENIZER_PATH = HOOK_PACKAGE_DIR / "assumption-guard-v2-tokenizer.json"
META_PATH = HOOK_PACKAGE_DIR / "assumption-guard-v2-meta.json"
BASELINE_DIR = HOOK_PACKAGE_DIR / "baseline"
REPORT_PATH = BASELINE_DIR / "assumption-guard-v2-report.json"
REGRESSION_CASES_PATH = BASELINE_DIR / "assumption-guard-regression-cases.json"
LABELED_DATA_PATH = BASELINE_DIR / "assumption-guard-training-labeled.jsonl"
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

    def test_packaged_assets_and_baseline_exist(self):
        self.assertTrue(HOOK_PACKAGE_DIR.exists())
        self.assertTrue(INSTALLER_PATH.exists())
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(TOKENIZER_PATH.exists())
        self.assertTrue(META_PATH.exists())
        self.assertTrue(BASELINE_DIR.exists())
        self.assertTrue(LABELED_DATA_PATH.exists())
        self.assertTrue(REGRESSION_CASES_PATH.exists())
        self.assertTrue(REPORT_PATH.exists())

    def test_packaged_defaults_keep_mutable_state_under_package_dir(self):
        runtime = load_runtime_module()
        self.assertEqual(runtime.STATE_PATH, HOOK_PACKAGE_DIR / "state")
        self.assertEqual(Path(runtime.LOG_PATH), HOOK_PACKAGE_DIR / "state" / "assumption-guard.log.jsonl")
        self.assertEqual(Path(runtime.QUEUE_PATH), HOOK_PACKAGE_DIR / "state" / "learning-queue.jsonl")

    def test_settings_snippet_points_to_packaged_script(self):
        settings = json.loads((HOOK_PACKAGE_DIR / "settings-snippet.json").read_text())
        command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        self.assertEqual(command, "python3 ~/.claude/hooks/assumption-guard/assumption-guard.py")

    def test_installer_copies_package_and_merges_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home_dir = Path(tmpdir) / "home"
            claude_dir = home_dir / ".claude"
            hooks_dir = claude_dir / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            settings_path = claude_dir / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "theme": "dark",
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "hooks": [
                                        {
                                            "type": "command",
                                            "command": "echo pretool",
                                            "timeout": 5,
                                        }
                                    ]
                                }
                            ]
                        },
                    },
                    indent=2,
                )
            )
            result = subprocess.run(
                ["bash", str(INSTALLER_PATH)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(home_dir)},
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            installed_dir = hooks_dir / "assumption-guard"
            installed_script = installed_dir / "assumption-guard.py"
            installed_settings_snippet = installed_dir / "settings-snippet.json"
            self.assertTrue(installed_dir.exists())
            self.assertTrue(installed_script.exists())
            self.assertTrue(installed_settings_snippet.exists())
            self.assertTrue((installed_dir / "baseline" / "assumption-guard-training-labeled.jsonl").exists())
            self.assertTrue((installed_dir / "state").exists())

            merged_settings = json.loads(settings_path.read_text())
            self.assertEqual(merged_settings["theme"], "dark")
            self.assertIn("PreToolUse", merged_settings["hooks"])
            stop_entries = merged_settings["hooks"]["Stop"]
            commands = [
                hook["command"]
                for entry in stop_entries
                for hook in entry.get("hooks", [])
                if hook.get("type") == "command"
            ]
            self.assertIn("python3 ~/.claude/hooks/assumption-guard/assumption-guard.py", commands)

            rerun = subprocess.run(
                ["bash", str(INSTALLER_PATH)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env={**os.environ, "HOME": str(home_dir)},
                check=False,
            )
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            rerun_settings = json.loads(settings_path.read_text())
            stop_commands = [
                hook["command"]
                for entry in rerun_settings["hooks"]["Stop"]
                for hook in entry.get("hooks", [])
                if hook.get("type") == "command"
            ]
            self.assertEqual(
                stop_commands.count("python3 ~/.claude/hooks/assumption-guard/assumption-guard.py"),
                1,
            )

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

    def test_packaged_train_works_with_empty_state_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            output_model = Path(tmpdir) / "out.onnx"
            output_tokenizer = Path(tmpdir) / "out-tokenizer.json"
            output_meta = Path(tmpdir) / "out-meta.json"
            output_report = Path(tmpdir) / "out-report.json"
            result = run_mode(
                "train",
                "--epochs",
                "1",
                "--skip-deberta",
                "--output-model",
                output_model,
                "--output-tokenizer",
                output_tokenizer,
                "--output-meta",
                output_meta,
                "--output-report",
                output_report,
                env={"ASSUMPTION_GUARD_STATE_DIR": str(state_dir)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_model.exists())
            self.assertTrue(output_tokenizer.exists())
            self.assertTrue(output_meta.exists())
            self.assertTrue(output_report.exists())

    def test_baseline_pipeline_uses_logistic_regression_when_rarest_class_is_two(self):
        runtime = load_runtime_module()
        runtime.ensure_training_dependencies()
        pipe = runtime.baseline_pipeline(2)
        self.assertEqual(pipe.named_steps["clf"].__class__.__name__, "LogisticRegression")

    def test_command_train_returns_structured_failure_when_no_transformer_candidate_is_available(self):
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_model = tmp / "model.onnx"
            output_tokenizer = tmp / "tokenizer.json"
            output_meta = tmp / "meta.json"
            output_report = tmp / "report.json"

            baseline_candidate = {
                "name": "baseline_linear_svm",
                "kind": "baseline",
                "model_object": object(),
                "threshold": 0.5,
                "threshold_search": [],
                "dev": {"confusion": {"block_recall": 1.0, "pass_recall": 1.0}},
                "test": {"confusion": {"block_recall": 1.0, "pass_recall": 1.0}, "per_intent": {}},
                "regression": {"matched": 1, "total": 1, "rows": []},
                "replay": {
                    "matched": 1,
                    "total": 1,
                    "block_recall": 1.0,
                    "pass_recall": 1.0,
                    "targeted_family_accuracy": {
                        "verification_narration": 1.0,
                        "dependency_gap_grounded": 1.0,
                        "capability_promise_unverified": 1.0,
                    },
                    "rows": [],
                },
                "false_negatives": [],
                "false_positives": [],
                "low_margin_examples": [],
            }

            original_baseline = runtime.run_baseline_candidate
            original_transformer = runtime.train_transformer_candidate
            original_dependencies = runtime.ensure_training_dependencies
            original_choose = runtime.choose_candidate
            original_set_seed = runtime.set_seed
            try:
                runtime.ensure_training_dependencies = lambda: None
                runtime.set_seed = lambda seed: None
                runtime.run_baseline_candidate = lambda *args, **kwargs: baseline_candidate
                runtime.train_transformer_candidate = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("transformers unavailable"))
                runtime.choose_candidate = lambda candidates: candidates[0]
                payload = runtime.command_train(
                    [
                        "--output-model",
                        str(output_model),
                        "--output-tokenizer",
                        str(output_tokenizer),
                        "--output-meta",
                        str(output_meta),
                        "--output-report",
                        str(output_report),
                    ],
                    emit_output=False,
                )
            finally:
                runtime.run_baseline_candidate = original_baseline
                runtime.train_transformer_candidate = original_transformer
                runtime.ensure_training_dependencies = original_dependencies
                runtime.choose_candidate = original_choose
                runtime.set_seed = original_set_seed

            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["reason"], "no_transformer_candidate")
            self.assertTrue(output_report.exists())
            report = json.loads(output_report.read_text())
            self.assertEqual(report["selected_candidate"], "baseline_linear_svm")
            self.assertFalse(output_model.exists())

    def test_learning_cycle_returns_structured_train_failure(self):
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"candidate_id": "candidate-a"}) + "\n")

            original_review = runtime.review_rows_with_council
            original_train = runtime.run_self_json_command
            try:
                pattern_families = [
                    "verification_narration",
                    "capability_promise_unverified",
                    "dependency_gap_grounded",
                    "reference_language",
                    "describe_type",
                ]
                runtime.review_rows_with_council = lambda *args, **kwargs: {
                    "consensus_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "route": "staged_training",
                            "final_intent": pattern_families[index % len(pattern_families)],
                            "final_block": pattern_families[index % len(pattern_families)] in {"capability_promise_unverified"},
                            "confidence": 0.96,
                            "pattern_family": pattern_families[index % len(pattern_families)],
                            "reviewer_ids": ["a", "b", "c"],
                        }
                        for index in range(20)
                    ],
                    "merged_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "text": f"candidate text {index}",
                            "final_intent": pattern_families[index % len(pattern_families)],
                            "final_block": pattern_families[index % len(pattern_families)] in {"capability_promise_unverified"},
                            "confidence": 0.96,
                            "pattern_family": pattern_families[index % len(pattern_families)],
                            "candidate_reason": "blocked_clause",
                        }
                        for index in range(20)
                    ],
                    "consolidated_rows": [],
                }
                runtime.run_self_json_command = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("train exploded"))
                payload = runtime.command_learning_cycle(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--queue-path",
                        str(queue_path),
                        "--log-path",
                        str(log_path),
                        "--transcript-root",
                        str(transcript_root),
                        "--min-review-batch",
                        "1",
                    ],
                    emit_output=False,
                )
            finally:
                runtime.review_rows_with_council = original_review
                runtime.run_self_json_command = original_train

            self.assertEqual(payload["status"], "captured")
            self.assertEqual(payload["reason"], "train_failed")
            self.assertFalse(payload["retrain"])

    def test_learning_cycle_passes_training_python_to_train_subprocess(self):
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"candidate_id": "candidate-a"}) + "\n")

            original_review = runtime.review_rows_with_council
            original_train = runtime.run_self_json_command
            seen = {}
            try:
                pattern_families = [
                    "verification_narration",
                    "capability_promise_unverified",
                    "dependency_gap_grounded",
                    "reference_language",
                    "describe_type",
                ]
                runtime.review_rows_with_council = lambda *args, **kwargs: {
                    "consensus_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "route": "staged_training",
                            "final_intent": pattern_families[index % len(pattern_families)],
                            "final_block": pattern_families[index % len(pattern_families)] in {"capability_promise_unverified"},
                            "confidence": 0.96,
                            "pattern_family": pattern_families[index % len(pattern_families)],
                            "reviewer_ids": ["a", "b", "c"],
                        }
                        for index in range(20)
                    ],
                    "merged_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "text": f"candidate text {index}",
                            "final_intent": pattern_families[index % len(pattern_families)],
                            "final_block": pattern_families[index % len(pattern_families)] in {"capability_promise_unverified"},
                            "confidence": 0.96,
                            "pattern_family": pattern_families[index % len(pattern_families)],
                            "candidate_reason": "blocked_clause",
                        }
                        for index in range(20)
                    ],
                    "consolidated_rows": [],
                }

                def fake_train(*args, **kwargs):
                    seen["python_executable"] = kwargs.get("python_executable")
                    raise RuntimeError("train exploded")

                runtime.run_self_json_command = fake_train
                payload = runtime.command_learning_cycle(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--queue-path",
                        str(queue_path),
                        "--log-path",
                        str(log_path),
                        "--transcript-root",
                        str(transcript_root),
                        "--min-review-batch",
                        "1",
                        "--training-python",
                        "/tmp/fake-python3.11",
                    ],
                    emit_output=False,
                )
            finally:
                runtime.review_rows_with_council = original_review
                runtime.run_self_json_command = original_train

            self.assertEqual(seen["python_executable"], "/tmp/fake-python3.11")
            self.assertEqual(payload["status"], "captured")
            self.assertEqual(payload["reason"], "train_failed")

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

    def test_promote_reviewed_rows_stage_rows_without_touching_accepted(self):
        review = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            staged_training = tmp / "staged-training-overlay.jsonl"
            staged_regression = tmp / "staged-regression-overlay.jsonl"
            staged_replay = tmp / "staged-replay-overlay.jsonl"
            accepted_training = tmp / "accepted-training-overlay.jsonl"
            accepted_regression = tmp / "accepted-regression-overlay.jsonl"
            accepted_replay = tmp / "accepted-replay-overlay.jsonl"
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
            ]
            result = review.promote_reviewed_rows_to_staged(
                reviewed_rows,
                staged_training_overlay=staged_training,
                staged_regression_overlay=staged_regression,
                staged_replay_overlay=staged_replay,
            )
            self.assertEqual(result["training_promoted"], 2)
            self.assertTrue(staged_training.exists())
            self.assertFalse(accepted_training.exists())
            self.assertFalse(accepted_regression.exists())
            self.assertFalse(accepted_replay.exists())

    def test_advance_staged_rows_moves_only_unique_rows_into_accepted(self):
        review = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            staged_training = tmp / "staged-training-overlay.jsonl"
            staged_regression = tmp / "staged-regression-overlay.jsonl"
            staged_replay = tmp / "staged-replay-overlay.jsonl"
            accepted_training = tmp / "accepted-training-overlay.jsonl"
            accepted_regression = tmp / "accepted-regression-overlay.jsonl"
            accepted_replay = tmp / "accepted-replay-overlay.jsonl"

            staged_training.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "row-a", "text": "A"}),
                        json.dumps({"id": "row-b", "text": "B"}),
                    ]
                )
                + "\n"
            )
            staged_regression.write_text(
                json.dumps({"case_hash": "case-a", "text": "A"}) + "\n"
            )
            staged_replay.write_text(
                json.dumps({"case_hash": "case-b", "text": "B"}) + "\n"
            )
            accepted_training.write_text(json.dumps({"id": "row-a", "text": "A"}) + "\n")

            result = review.advance_staged_rows_to_accepted(
                staged_training_overlay=staged_training,
                staged_regression_overlay=staged_regression,
                staged_replay_overlay=staged_replay,
                accepted_training_overlay=accepted_training,
                accepted_regression_overlay=accepted_regression,
                accepted_replay_overlay=accepted_replay,
            )
            self.assertEqual(result["training_advanced"], 1)
            self.assertEqual(result["regression_advanced"], 1)
            self.assertEqual(result["replay_advanced"], 1)
            accepted_training_rows = [
                json.loads(line)
                for line in accepted_training.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual({row["id"] for row in accepted_training_rows}, {"row-a", "row-b"})

    def test_reduce_review_council_routes_unanimous_and_split_rows(self):
        review = load_runtime_module()
        unanimous_reviews = [
            {
                "reviewer_id": "a",
                "candidate_id": "candidate-a",
                "final_intent": "verification_narration",
                "final_block": False,
                "confidence": 0.93,
                "rationale": "explicit verification narration",
                "pattern_family": "verification_narration",
            },
            {
                "reviewer_id": "b",
                "candidate_id": "candidate-a",
                "final_intent": "verification_narration",
                "final_block": False,
                "confidence": 0.88,
                "rationale": "explicit verification narration",
                "pattern_family": "verification_narration",
            },
            {
                "reviewer_id": "c",
                "candidate_id": "candidate-a",
                "final_intent": "verification_narration",
                "final_block": False,
                "confidence": 0.9,
                "rationale": "explicit verification narration",
                "pattern_family": "verification_narration",
            },
        ]
        unanimous = review.reduce_review_council(unanimous_reviews)
        self.assertEqual(unanimous["route"], "staged_training")

        split_reviews = [
            {
                "reviewer_id": "a",
                "candidate_id": "candidate-b",
                "final_intent": "verification_narration",
                "final_block": False,
                "confidence": 0.91,
                "rationale": "explicit verification narration",
                "pattern_family": "verification_narration",
            },
            {
                "reviewer_id": "b",
                "candidate_id": "candidate-b",
                "final_intent": "capability_promise_unverified",
                "final_block": True,
                "confidence": 0.95,
                "rationale": "unsupported promise",
                "pattern_family": "capability_promise_unverified",
            },
            {
                "reviewer_id": "c",
                "candidate_id": "candidate-b",
                "final_intent": "capability_promise_unverified",
                "final_block": True,
                "confidence": 0.89,
                "rationale": "unsupported promise",
                "pattern_family": "capability_promise_unverified",
            },
        ]
        split = review.reduce_review_council(split_reviews)
        self.assertEqual(split["route"], "quarantine")

    def test_review_batch_health_rejects_one_family_flood(self):
        review = load_runtime_module()
        consensus_rows = [
            {
                "candidate_id": f"candidate-{index}",
                "route": "staged_training",
                "final_intent": "capability_promise_unverified",
                "final_block": True,
                "confidence": 0.96,
                "pattern_family": "capability_promise_unverified",
                "reviewer_ids": ["a", "b", "c"],
            }
            for index in range(6)
        ]
        metrics = review.review_batch_health_metrics(consensus_rows)
        self.assertGreater(metrics["largest_family_share"], 0.8)
        self.assertFalse(review.review_batch_is_healthy(metrics))

    def test_review_batch_health_rejects_high_disagreement(self):
        review = load_runtime_module()
        consensus_rows = [
            {
                "candidate_id": "candidate-a",
                "route": "quarantine",
                "final_intent": "verification_narration",
                "final_block": None,
                "confidence": 0.72,
                "pattern_family": "verification_narration",
                "reviewer_ids": ["a", "b", "c"],
            },
            {
                "candidate_id": "candidate-b",
                "route": "needs_consolidation",
                "final_intent": "reference_language",
                "final_block": False,
                "confidence": 0.7,
                "pattern_family": "reference_language",
                "reviewer_ids": ["a", "b", "c"],
            },
            {
                "candidate_id": "candidate-c",
                "route": "staged_training",
                "final_intent": "verification_narration",
                "final_block": False,
                "confidence": 0.92,
                "pattern_family": "verification_narration",
                "reviewer_ids": ["a", "b", "c"],
            },
        ]
        metrics = review.review_batch_health_metrics(consensus_rows)
        self.assertGreater(metrics["disagreement_rate"], 0.5)
        self.assertFalse(review.review_batch_is_healthy(metrics))

    def test_review_rows_with_council_uses_consolidator_for_disputed_rows(self):
        review = load_runtime_module()
        input_rows = [
            {
                "id": "candidate-a",
                "text": "I checked package.json and cannot confirm the config value from this repo.",
                "candidate_reason": "mixed_evidence_gap",
            }
        ]
        reviewer_outputs = {
            "a": [
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "verified_limitation",
                    "final_block": False,
                    "confidence": 0.72,
                    "rationale": "checked evidence then bounded non-conclusion",
                    "pattern_family": "verified_limitation",
                    "reviewer_id": "a",
                }
            ],
            "b": [
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "dependency_gap_grounded",
                    "final_block": False,
                    "confidence": 0.71,
                    "rationale": "missing dependency blocks next step",
                    "pattern_family": "dependency_gap_grounded",
                    "reviewer_id": "b",
                }
            ],
            "c": [
                {
                    "candidate_id": "candidate-a",
                    "final_intent": "reference_language",
                    "final_block": False,
                    "confidence": 0.69,
                    "rationale": "policy-oriented reference phrasing",
                    "pattern_family": "reference_language",
                    "reviewer_id": "c",
                }
            ],
        }

        original_review_rows_with_reviewer = review.review_rows_with_reviewer
        original_consolidate_disputed_rows = review.consolidate_disputed_rows
        try:
            review.review_rows_with_reviewer = (
                lambda input_rows, claude_cmd, quarantine_dir, reviewer_id, output_path: reviewer_outputs[reviewer_id]
            )
            review.consolidate_disputed_rows = lambda disputed_rows, **kwargs: [
                {
                    "candidate_id": "candidate-a",
                    "route": "staged_training",
                    "final_intent": "dependency_gap_grounded",
                    "final_block": False,
                    "confidence": 0.83,
                    "pattern_family": "dependency_gap_grounded",
                    "resolution_reason": "resolved from reviewer disagreement",
                }
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                result = review.review_rows_with_council(
                    input_rows,
                    claude_cmd="claude",
                    quarantine_dir=tmp / "quarantine",
                    output_dir=tmp,
                )
                self.assertEqual(len(result["merged_rows"]), 1)
                self.assertEqual(result["merged_rows"][0]["final_intent"], "dependency_gap_grounded")
                consolidated_rows = [
                    json.loads(line)
                    for line in (tmp / "review-consolidated.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(consolidated_rows), 1)
                self.assertEqual(consolidated_rows[0]["route"], "staged_training")
        finally:
            review.review_rows_with_reviewer = original_review_rows_with_reviewer
            review.consolidate_disputed_rows = original_consolidate_disputed_rows

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
            staged_training_overlay = [
                json.loads(line)
                for line in (state_dir / "staged-training-overlay.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(staged_training_overlay), 2)
            self.assertFalse((state_dir / "accepted-training-overlay.jsonl").exists())
            for reviewer_id in ("a", "b", "c"):
                self.assertTrue((state_dir / f"reviewed-{reviewer_id}.jsonl").exists())
            consensus_rows = [
                json.loads(line)
                for line in (state_dir / "review-consensus.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(consensus_rows), 2)
            self.assertTrue(all(row["route"] == "staged_training" for row in consensus_rows))

    def test_learning_cycle_rejected_candidate_keeps_accepted_overlays_unchanged(self):
        review = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"candidate_id": "candidate-a"}) + "\n")

            accepted_training = state_dir / "accepted-training-overlay.jsonl"
            accepted_training.write_text(json.dumps({"id": "accepted-row", "text": "keep me"}) + "\n")
            current_report_path = state_dir / "current-model-report.json"
            current_report = json.loads(REPORT_PATH.read_text())
            current_report_path.write_text(json.dumps(current_report))

            original_review = review.review_rows_with_council
            original_train = review.run_self_json_command
            try:
                review.review_rows_with_council = lambda *args, **kwargs: {
                    "consensus_rows": [
                        {
                            "candidate_id": "candidate-a",
                            "route": "staged_training",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "reviewer_ids": ["a", "b", "c"],
                        }
                    ],
                    "merged_rows": [
                        {
                            "candidate_id": "candidate-a",
                            "text": "But if you want to revert, I can check which ones are new and remove them.",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "candidate_reason": "heuristic_rescue",
                        }
                    ],
                }

                def fake_train(mode, *extra_args, **kwargs):
                    self.assertEqual(mode, "train")
                    args = list(extra_args)
                    model_path = Path(args[args.index("--output-model") + 1])
                    tokenizer_path = Path(args[args.index("--output-tokenizer") + 1])
                    meta_path = Path(args[args.index("--output-meta") + 1])
                    report_path = Path(args[args.index("--output-report") + 1])
                    model_path.write_text("model")
                    tokenizer_path.write_text("{}")
                    meta_path.write_text("{}")
                    report_path.write_text(json.dumps(current_report))
                    return {}

                review.run_self_json_command = fake_train
                payload = review.command_learning_cycle(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--queue-path",
                        str(queue_path),
                        "--log-path",
                        str(log_path),
                        "--transcript-root",
                        str(transcript_root),
                        "--min-review-batch",
                        "1",
                    ],
                    emit_output=False,
                )
            finally:
                review.review_rows_with_council = original_review
                review.run_self_json_command = original_train

            self.assertEqual(payload["status"], "trained")
            self.assertFalse(payload["promoted"])
            accepted_rows = [
                json.loads(line)
                for line in accepted_training.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(accepted_rows), 1)
            self.assertEqual(accepted_rows[0]["id"], "accepted-row")
            staged_rows = [
                json.loads(line)
                for line in (state_dir / "staged-training-overlay.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(staged_rows), 1)

    def test_learning_cycle_promoted_candidate_advances_staged_and_writes_manifest(self):
        review = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"candidate_id": "candidate-a"}) + "\n")

            current_report = json.loads(REPORT_PATH.read_text())
            weaker_report = json.loads(REPORT_PATH.read_text())
            weaker_report["replay_results"]["block_recall"] = 0.95
            weaker_report["replay_results"]["pass_recall"] = 0.95
            weaker_report["regression_results"]["matched"] = weaker_report["regression_results"]["total"]
            (state_dir / "current-model-report.json").write_text(json.dumps(weaker_report))

            original_model = (HOOK_PACKAGE_DIR / "assumption-guard-v2.onnx").read_bytes()
            original_tokenizer = TOKENIZER_PATH.read_text()
            original_meta = META_PATH.read_text()

            original_review = review.review_rows_with_council
            original_train = review.run_self_json_command
            try:
                review.review_rows_with_council = lambda *args, **kwargs: {
                    "consensus_rows": [
                        {
                            "candidate_id": "candidate-a",
                            "route": "staged_training",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "reviewer_ids": ["a", "b", "c"],
                        }
                    ],
                    "merged_rows": [
                        {
                            "candidate_id": "candidate-a",
                            "text": "But if you want to revert, I can check which ones are new and remove them.",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "candidate_reason": "heuristic_rescue",
                        }
                    ],
                }

                def fake_train(mode, *extra_args, **kwargs):
                    self.assertEqual(mode, "train")
                    args = list(extra_args)
                    model_path = Path(args[args.index("--output-model") + 1])
                    tokenizer_path = Path(args[args.index("--output-tokenizer") + 1])
                    meta_path = Path(args[args.index("--output-meta") + 1])
                    report_path = Path(args[args.index("--output-report") + 1])
                    model_path.write_text("promoted-model")
                    tokenizer_path.write_text('{"ok":true}')
                    meta_path.write_text('{"threshold":0.2}')
                    report_path.write_text(json.dumps(current_report))
                    return {}

                review.run_self_json_command = fake_train
                payload = review.command_learning_cycle(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--queue-path",
                        str(queue_path),
                        "--log-path",
                        str(log_path),
                        "--transcript-root",
                        str(transcript_root),
                        "--min-review-batch",
                        "1",
                    ],
                    emit_output=False,
                )
            finally:
                review.review_rows_with_council = original_review
                review.run_self_json_command = original_train
                (HOOK_PACKAGE_DIR / "assumption-guard-v2.onnx").write_bytes(original_model)
                TOKENIZER_PATH.write_text(original_tokenizer)
                META_PATH.write_text(original_meta)

            self.assertEqual(payload["status"], "promoted")
            self.assertFalse((state_dir / "staged-training-overlay.jsonl").exists())
            accepted_rows = [
                json.loads(line)
                for line in (state_dir / "accepted-training-overlay.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(accepted_rows), 1)
            manifest_rows = [
                json.loads(line)
                for line in (state_dir / "promotion-manifest.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["training_advanced"], 1)

    def test_learning_cycle_skips_retrain_for_unhealthy_review_batch(self):
        review = load_runtime_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_dir = tmp / "state"
            queue_path = state_dir / "learning-queue.jsonl"
            log_path = tmp / "assumption-guard.log.jsonl"
            transcript_root = tmp / "projects"
            transcript_root.mkdir(parents=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps({"candidate_id": "candidate-a"}) + "\n")

            original_review = review.review_rows_with_council
            original_train = review.run_self_json_command
            try:
                review.review_rows_with_council = lambda *args, **kwargs: {
                    "consensus_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "route": "staged_training",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "reviewer_ids": ["a", "b", "c"],
                        }
                        for index in range(6)
                    ],
                    "merged_rows": [
                        {
                            "candidate_id": f"candidate-{index}",
                            "text": f"candidate text {index}",
                            "final_intent": "capability_promise_unverified",
                            "final_block": True,
                            "confidence": 0.96,
                            "pattern_family": "capability_promise_unverified",
                            "candidate_reason": "blocked_clause",
                        }
                        for index in range(6)
                    ],
                    "consolidated_rows": [],
                }

                def fail_if_train_called(*args, **kwargs):
                    raise AssertionError("train should not run for an unhealthy batch")

                review.run_self_json_command = fail_if_train_called
                payload = review.command_learning_cycle(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--queue-path",
                        str(queue_path),
                        "--log-path",
                        str(log_path),
                        "--transcript-root",
                        str(transcript_root),
                        "--min-review-batch",
                        "1",
                    ],
                    emit_output=False,
                )
            finally:
                review.review_rows_with_council = original_review
                review.run_self_json_command = original_train

            self.assertEqual(payload["status"], "captured")
            self.assertFalse(payload["retrain"])
            self.assertEqual(payload["reason"], "unhealthy_review_batch")

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

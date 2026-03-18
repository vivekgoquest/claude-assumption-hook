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
TRAINING_SCRIPT = REPO_ROOT / "training" / "train_v2.py"
REPLAY_SCRIPT = REPO_ROOT / "training" / "replay_eval_v2.py"
MINING_SCRIPT = REPO_ROOT / "training" / "mine_transcripts_v2.py"
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
TRAINING_PYTHON = os.environ.get("ASSUMPTION_GUARD_TRAIN_PYTHON") or shutil.which("python3.11") or sys.executable


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("assumption_guard_runtime", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def regression_cases():
    return json.loads(REGRESSION_CASES_PATH.read_text())


def run_hook(
    assistant_text,
    *,
    backend="v2",
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    meta_path=META_PATH,
    import_blocker=False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
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
        return result, log_entries


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
                result, _ = run_hook(case["text"])
                self.assertEqual(result.returncode, 0, result.stderr)
                if case["expected_block"]:
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload["decision"], "block")
                else:
                    self.assertEqual(result.stdout.strip(), "", result.stdout)

    def test_block_reason_includes_clause_and_predicted_intent(self):
        result, _ = run_hook("Maybe rename this helper to be clearer")
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

        result, _ = run_hook(verification_narration)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", result.stdout)

        result, _ = run_hook(grounded_limitation)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", result.stdout)

        result, _ = run_hook(unsupported_capability)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "block")

    def test_missing_onnx_dependencies_fall_back_to_regex_only(self):
        result, log_entries = run_hook(
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
            result, log_entries = run_hook(
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
    def test_mining_and_training_entrypoints_exist(self):
        self.assertTrue(MINING_SCRIPT.exists())
        self.assertTrue(TRAINING_SCRIPT.exists())
        self.assertTrue(REPLAY_SCRIPT.exists())

    def test_repo_only_keeps_single_runtime_and_v2_artifacts(self):
        self.assertFalse(LEGACY_RUNTIME_PATH.exists())
        self.assertFalse(LEGACY_MODEL_HELPER_PATH.exists())
        self.assertFalse(LEGACY_TRAINING_SCRIPT.exists())
        self.assertFalse(LEGACY_MODEL_PATH.exists())
        self.assertFalse(LEGACY_METRICS_PATH.exists())
        self.assertFalse(LEGACY_EVAL_PATH.exists())

    def test_replay_script_runs_against_committed_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "replay-report.json"
            result = subprocess.run(
                [
                    TRAINING_PYTHON,
                    str(REPLAY_SCRIPT),
                    "--regression-cases",
                    str(REGRESSION_CASES_PATH),
                    "--model",
                    str(MODEL_PATH),
                    "--tokenizer",
                    str(TOKENIZER_PATH),
                    "--meta",
                    str(META_PATH),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            replay = json.loads(output_path.read_text())
            self.assertEqual(replay["summary"]["matched"], replay["summary"]["total"])

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

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hook" / "assumption-guard.py"
TRAINING_SCRIPT = REPO_ROOT / "training" / "train_assumption_guard.py"
MODEL_PATH = REPO_ROOT / "model" / "assumption-guard-model.pkl"
METRICS_PATH = REPO_ROOT / "model" / "assumption-guard-metrics.json"
REGRESSION_CASES_PATH = REPO_ROOT / "training" / "assumption-guard-regression-cases.json"
LABELED_DATA_PATH = REPO_ROOT / "training" / "assumption-guard-training-labeled.jsonl"


def regression_cases():
    return json.loads(REGRESSION_CASES_PATH.read_text())


def run_hook(assistant_text, *, model_path=MODEL_PATH, import_blocker=False):
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
        env["ASSUMPTION_GUARD_MODEL_PATH"] = str(model_path)
        env["ASSUMPTION_GUARD_LOG_PATH"] = str(log_path)
        if import_blocker:
            blocker = f"""
import builtins
import runpy
import sys

real_import = builtins.__import__
blocked = {{"numpy", "scipy", "sklearn"}}

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


class AssumptionGuardHookTests(unittest.TestCase):
    def test_regression_cases_follow_strict_policy(self):
        for case in regression_cases():
            with self.subTest(text=case["text"]):
                result, _ = run_hook(case["text"])
                self.assertEqual(result.returncode, 0, result.stderr)
                if case["expected_block"]:
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload["decision"], "block")
                else:
                    self.assertEqual(result.stdout.strip(), "", result.stdout)

    def test_missing_ml_dependencies_fall_back_to_regex_only(self):
        result, log_entries = run_hook(
            "I think the timeout is 30 seconds",
            import_blocker=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")
        self.assertTrue(log_entries, "expected a log entry")
        latest = log_entries[-1]
        self.assertEqual(latest["stage"], "regex_only")
        self.assertFalse(latest["ml_available"])
        self.assertEqual(latest["fallback_reason"], "ml_import_error")
        self.assertEqual(latest["error_stage"], "ml_import")

    def test_corrupt_model_falls_back_to_regex_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_model = Path(tmpdir) / "broken-model.pkl"
            corrupt_model.write_text("not a pickle")
            result, log_entries = run_hook(
                "I think the timeout is 30 seconds",
                model_path=corrupt_model,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision"], "block")
        self.assertTrue(log_entries, "expected a log entry")
        latest = log_entries[-1]
        self.assertEqual(latest["stage"], "regex_only")
        self.assertFalse(latest["ml_available"])
        self.assertEqual(latest["fallback_reason"], "pickle_load_error")
        self.assertEqual(latest["error_stage"], "model_load")


class AssumptionGuardTrainingTests(unittest.TestCase):
    def test_training_script_reproduces_metrics_and_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_model = tmp / "assumption-guard-model.pkl"
            output_metrics = tmp / "assumption-guard-metrics.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(TRAINING_SCRIPT),
                    "--training-data",
                    str(LABELED_DATA_PATH),
                    "--regression-cases",
                    str(REGRESSION_CASES_PATH),
                    "--output-model",
                    str(output_model),
                    "--output-metrics",
                    str(output_metrics),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(output_metrics.read_text())
            committed_metrics = json.loads(METRICS_PATH.read_text())
            self.assertEqual(metrics["dataset"]["rows"], committed_metrics["dataset"]["rows"])
            self.assertEqual(metrics["dataset"]["label_counts"], committed_metrics["dataset"]["label_counts"])
            self.assertAlmostEqual(
                metrics["cross_validation"]["f1_macro_mean"],
                committed_metrics["cross_validation"]["f1_macro_mean"],
                places=6,
            )
            self.assertTrue(
                all(case["matched"] for case in metrics["regression_cases"]),
                metrics["regression_cases"],
            )

            smoke_result, _ = run_hook(
                "I think the timeout is 30 seconds",
                model_path=output_model,
            )
            self.assertEqual(smoke_result.returncode, 0, smoke_result.stderr)
            self.assertEqual(json.loads(smoke_result.stdout)["decision"], "block")


if __name__ == "__main__":
    unittest.main()

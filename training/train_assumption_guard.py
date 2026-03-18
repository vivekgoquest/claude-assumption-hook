#!/usr/bin/env python3
"""Train and evaluate the Assumption Guard classifier."""

import argparse
import importlib.util
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "hook" / "assumption-guard.py"
DEFAULT_TRAINING_DATA = ROOT / "training" / "assumption-guard-training-labeled.jsonl"
DEFAULT_REGRESSION_CASES = ROOT / "training" / "assumption-guard-regression-cases.json"
DEFAULT_OUTPUT_MODEL = ROOT / "model" / "assumption-guard-model.pkl"
DEFAULT_OUTPUT_METRICS = ROOT / "model" / "assumption-guard-metrics.json"


def load_hook_module():
    spec = importlib.util.spec_from_file_location("assumption_guard", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_regression_cases(path):
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def regression_results(pipe, cases):
    results = []
    for case in cases:
        block_probability = float(pipe.predict_proba([case["text"]])[0][1])
        predicted_block = bool(pipe.predict([case["text"]])[0])
        results.append(
            {
                "text": case["text"],
                "reason": case.get("reason"),
                "expected_block": case["expected_block"],
                "predicted_block": predicted_block,
                "matched": predicted_block == case["expected_block"],
                "block_probability": round(block_probability, 6),
            }
        )
    return results


def build_metrics(rows, scores, model_path, cases, case_results):
    label_counts = Counter("block" if row["block"] else "pass" for row in rows)
    intent_counts = Counter(row["intent"] for row in rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": len(rows),
            "label_counts": dict(label_counts),
            "intent_counts": dict(intent_counts),
        },
        "cross_validation": {
            "folds": len(scores),
            "f1_macro_mean": round(float(scores.mean()), 6),
            "f1_macro_std": round(float(scores.std()), 6),
        },
        "model": {
            "pipeline": "TF-IDF (1-3 grams, 5000 features) + 14 intent features -> LogisticRegression",
            "size_bytes": model_path.stat().st_size,
            "size_kb": round(model_path.stat().st_size / 1024.0, 3),
        },
        "regression_cases": case_results,
        "regression_summary": {
            "cases": len(cases),
            "matched": sum(1 for result in case_results if result["matched"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--regression-cases", default=str(DEFAULT_REGRESSION_CASES))
    parser.add_argument("--output-model", default=str(DEFAULT_OUTPUT_MODEL))
    parser.add_argument("--output-metrics", default=str(DEFAULT_OUTPUT_METRICS))
    args = parser.parse_args()

    hook = load_hook_module()
    hook.initialize_ml(raise_on_error=True)

    from sklearn.model_selection import StratifiedKFold, cross_val_score  # pylint: disable=import-outside-toplevel

    rows = load_jsonl(args.training_data)
    sentences = [row["sentence"] for row in rows]
    labels = [1 if row["block"] else 0 for row in rows]

    pipeline = hook.build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, sentences, labels, cv=cv, scoring="f1_macro")

    pipeline.fit(sentences, labels)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, "wb") as file_obj:
        pickle.dump(pipeline, file_obj)

    cases = load_regression_cases(args.regression_cases)
    case_results = regression_results(pipeline, cases)

    metrics = build_metrics(rows, scores, output_model, cases, case_results)
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_model": str(output_model),
                "output_metrics": str(output_metrics),
                "rows": len(rows),
                "f1_macro_mean": metrics["cross_validation"]["f1_macro_mean"],
                "regression_matched": metrics["regression_summary"]["matched"],
                "regression_cases": metrics["regression_summary"]["cases"],
            }
        )
    )


if __name__ == "__main__":
    main()

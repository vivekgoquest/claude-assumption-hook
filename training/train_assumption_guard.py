#!/usr/bin/env python3
"""Train and evaluate the Assumption Guard classifier."""

import argparse
import importlib.util
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "hook" / "assumption-guard.py"
DEFAULT_TRAINING_DATA = ROOT / "training" / "assumption-guard-training-labeled.jsonl"
DEFAULT_REGRESSION_CASES = ROOT / "training" / "assumption-guard-regression-cases.json"
DEFAULT_OUTPUT_MODEL = ROOT / "model" / "assumption-guard-model.pkl"
DEFAULT_OUTPUT_METRICS = ROOT / "model" / "assumption-guard-metrics.json"
DEFAULT_OUTPUT_EVAL = ROOT / "model" / "assumption-guard-eval.json"

BLOCK_RECALL_TARGET = 0.95
PASS_RECALL_TARGET = 0.85
REVIEW_INTENTS = ("reference_language", "verified_limitation", "reason_conditionally")
IMPROVEMENT_INTENTS = {
    "describe_type": 0.8,
    "idiomatic_compare": 0.8,
}
THRESHOLD_CANDIDATES = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
LOW_MARGIN_LIMIT = 20
MISCLASSIFICATION_LIMIT = 8


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


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def predict_block_probability(pipe, texts):
    probabilities = pipe.predict_proba(texts)
    return [float(row[1]) for row in probabilities]


def confusion_from_probabilities(labels, probabilities, threshold):
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for expected, predicted in zip(labels, predictions) if expected == 1 and predicted == 1)
    tn = sum(1 for expected, predicted in zip(labels, predictions) if expected == 0 and predicted == 0)
    fp = sum(1 for expected, predicted in zip(labels, predictions) if expected == 0 and predicted == 1)
    fn = sum(1 for expected, predicted in zip(labels, predictions) if expected == 1 and predicted == 0)
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "block_recall": round(safe_divide(tp, positives), 6),
        "pass_recall": round(safe_divide(tn, negatives), 6),
        "accuracy": round(safe_divide(tp + tn, len(labels)), 6),
        "predictions": predictions,
    }


def threshold_review(labels, probabilities):
    reviews = []
    best = None
    for threshold in THRESHOLD_CANDIDATES:
        confusion = confusion_from_probabilities(labels, probabilities, threshold)
        score = (
            confusion["block_recall"] * 4.0
            + confusion["pass_recall"] * 1.5
            + confusion["accuracy"]
        )
        candidate = {
            "threshold": threshold,
            "block_recall": confusion["block_recall"],
            "pass_recall": confusion["pass_recall"],
            "accuracy": confusion["accuracy"],
            "score": round(score, 6),
        }
        reviews.append(candidate)
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    default = next(candidate for candidate in reviews if candidate["threshold"] == 0.5)
    return {
        "value": 0.5,
        "decision": "kept_default_threshold_for_runtime_compatibility",
        "best_candidate": best,
        "reviewed_candidates": reviews,
        "default_threshold_metrics": default,
    }


def per_intent_accuracy(rows, predictions):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row, predicted in zip(rows, predictions):
        bucket = stats[row["intent"]]
        bucket["total"] += 1
        bucket["correct"] += int(predicted == (1 if row["block"] else 0))
    return {
        intent: {
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": round(safe_divide(values["correct"], values["total"]), 6),
        }
        for intent, values in sorted(stats.items())
    }


def low_margin_examples(rows, probabilities, predictions, threshold):
    examples = []
    for row, probability, predicted in zip(rows, probabilities, predictions):
        examples.append(
            {
                "sentence": row["sentence"],
                "intent": row["intent"],
                "expected_block": row["block"],
                "predicted_block": bool(predicted),
                "block_probability": round(probability, 6),
                "margin": round(abs(probability - threshold), 6),
                "matched": bool(predicted) == row["block"],
            }
        )
    examples.sort(key=lambda item: (item["margin"], not item["matched"]))
    return examples[:LOW_MARGIN_LIMIT]


def misclassifications_by_intent(rows, probabilities, predictions):
    buckets = defaultdict(list)
    for row, probability, predicted in zip(rows, probabilities, predictions):
        expected = 1 if row["block"] else 0
        if predicted == expected:
            continue
        buckets[row["intent"]].append(
            {
                "sentence": row["sentence"],
                "expected_block": row["block"],
                "predicted_block": bool(predicted),
                "block_probability": round(probability, 6),
                "margin": round(abs(probability - 0.5), 6),
            }
        )
    return {
        intent: sorted(items, key=lambda item: item["margin"])[:MISCLASSIFICATION_LIMIT]
        for intent, items in sorted(buckets.items())
    }


def regression_results(pipe, cases, threshold):
    results = []
    for case in cases:
        block_probability = predict_block_probability(pipe, [case["text"]])[0]
        predicted_block = block_probability >= threshold
        results.append(
            {
                "text": case["text"],
                "group": case.get("group"),
                "reason": case.get("reason"),
                "expected_block": case["expected_block"],
                "predicted_block": predicted_block,
                "matched": predicted_block == case["expected_block"],
                "block_probability": round(block_probability, 6),
            }
        )
    return results


def acceptance_summary(per_intent, confusion, regression, threshold):
    failures = []
    if not all(case["matched"] for case in regression):
        failures.append("regression_cases")
    if confusion["block_recall"] < BLOCK_RECALL_TARGET:
        failures.append("block_recall")
    if confusion["pass_recall"] < PASS_RECALL_TARGET:
        failures.append("pass_recall")
    for intent in REVIEW_INTENTS:
        if per_intent.get(intent, {}).get("accuracy", 0.0) < 0.95:
            failures.append(f"{intent}_stability")
    for intent, minimum in IMPROVEMENT_INTENTS.items():
        if per_intent.get(intent, {}).get("accuracy", 0.0) < minimum:
            failures.append(f"{intent}_improvement")
    if threshold["value"] != 0.5:
        failures.append("runtime_threshold_changed")
    return {
        "criteria": {
            "regression_cases_match": all(case["matched"] for case in regression),
            "block_recall_target": BLOCK_RECALL_TARGET,
            "pass_recall_target": PASS_RECALL_TARGET,
            "review_intents": {intent: per_intent.get(intent, {}) for intent in REVIEW_INTENTS},
            "improvement_intents": {
                intent: {
                    "minimum_accuracy": minimum,
                    "actual": per_intent.get(intent, {}).get("accuracy", 0.0),
                }
                for intent, minimum in IMPROVEMENT_INTENTS.items()
            },
        },
        "failed_checks": failures,
        "meets_acceptance_bar": not failures,
    }


def evaluate_candidate(name, pipeline, description, rows, cases):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score  # pylint: disable=import-outside-toplevel

    sentences = [row["sentence"] for row in rows]
    labels = [1 if row["block"] else 0 for row in rows]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, sentences, labels, cv=cv, scoring="f1_macro")
    probabilities = cross_val_predict(
        pipeline,
        sentences,
        labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    threshold = threshold_review(labels, probabilities)
    confusion = confusion_from_probabilities(labels, probabilities, threshold["value"])
    per_intent = per_intent_accuracy(rows, confusion["predictions"])
    misclassified = misclassifications_by_intent(rows, probabilities, confusion["predictions"])
    low_margin = low_margin_examples(rows, probabilities, confusion["predictions"], threshold["value"])

    pipeline.fit(sentences, labels)
    regression = regression_results(pipeline, cases, threshold["value"])
    acceptance = acceptance_summary(per_intent, confusion, regression, threshold)

    return {
        "summary": {
            "name": name,
            "description": description,
            "cross_validation": {
                "folds": len(scores),
                "f1_macro_mean": round(float(scores.mean()), 6),
                "f1_macro_std": round(float(scores.std()), 6),
            },
            "threshold": threshold,
            "evaluation": {
                "confusion": {key: value for key, value in confusion.items() if key != "predictions"},
                "per_intent_accuracy": per_intent,
                "low_margin_examples": low_margin,
                "misclassifications_by_intent": misclassified,
            },
            "regression_cases": regression,
            "regression_summary": {
                "cases": len(cases),
                "matched": sum(1 for result in regression if result["matched"]),
            },
            "acceptance": acceptance,
        },
        "fitted_pipeline": pipeline,
    }


def choose_recommended_candidate(candidate_summaries):
    def sort_key(item):
        acceptance = item["summary"]["acceptance"]["meets_acceptance_bar"]
        confusion = item["summary"]["evaluation"]["confusion"]
        cv = item["summary"]["cross_validation"]
        regression = item["summary"]["regression_summary"]["matched"]
        return (
            int(acceptance),
            confusion["block_recall"],
            confusion["pass_recall"],
            regression,
            cv["f1_macro_mean"],
        )

    return max(candidate_summaries, key=sort_key)


def build_metrics(rows, selected_summary, model_path):
    label_counts = Counter("block" if row["block"] else "pass" for row in rows)
    intent_counts = Counter(row["intent"] for row in rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": len(rows),
            "label_counts": dict(label_counts),
            "intent_counts": dict(intent_counts),
        },
        "cross_validation": selected_summary["cross_validation"],
        "model": {
            "name": selected_summary["name"],
            "pipeline": selected_summary["description"],
            "size_bytes": model_path.stat().st_size,
            "size_kb": round(model_path.stat().st_size / 1024.0, 3),
        },
        "threshold": selected_summary["threshold"],
        "evaluation": selected_summary["evaluation"],
        "regression_cases": selected_summary["regression_cases"],
        "regression_summary": selected_summary["regression_summary"],
        "acceptance": selected_summary["acceptance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--regression-cases", default=str(DEFAULT_REGRESSION_CASES))
    parser.add_argument("--output-model", default=str(DEFAULT_OUTPUT_MODEL))
    parser.add_argument("--output-metrics", default=str(DEFAULT_OUTPUT_METRICS))
    parser.add_argument("--output-eval", default=str(DEFAULT_OUTPUT_EVAL))
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Evaluate the fallback LinearSVC candidate alongside the default logistic model.",
    )
    args = parser.parse_args()

    hook = load_hook_module()
    hook.initialize_ml(raise_on_error=True)

    rows = load_jsonl(args.training_data)
    cases = load_regression_cases(args.regression_cases)
    descriptions = hook.model_descriptions()
    model_names = ["logreg", "linear_svm"] if args.compare_models else ["logreg"]
    candidates = hook.build_candidate_pipelines(model_names=model_names)

    candidate_results = []
    for name, pipeline in candidates.items():
        candidate_results.append(
            evaluate_candidate(
                name=name,
                pipeline=pipeline,
                description=descriptions[name],
                rows=rows,
                cases=cases,
            )
        )

    selected = choose_recommended_candidate(candidate_results)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, "wb") as file_obj:
        pickle.dump(selected["fitted_pipeline"], file_obj)

    metrics = build_metrics(rows, selected["summary"], output_model)
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    output_eval = Path(args.output_eval)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    output_eval.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "candidate_models": [result["summary"] for result in candidate_results],
                "recommended_model": {
                    "name": selected["summary"]["name"],
                    "reason": (
                        "highest-ranked candidate by acceptance bar, block recall, "
                        "pass recall, regression coverage, and macro F1"
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_model": str(output_model),
                "output_metrics": str(output_metrics),
                "output_eval": str(output_eval),
                "rows": len(rows),
                "selected_model": selected["summary"]["name"],
                "f1_macro_mean": metrics["cross_validation"]["f1_macro_mean"],
                "regression_matched": metrics["regression_summary"]["matched"],
                "regression_cases": metrics["regression_summary"]["cases"],
                "meets_acceptance_bar": metrics["acceptance"]["meets_acceptance_bar"],
            }
        )
    )


if __name__ == "__main__":
    main()

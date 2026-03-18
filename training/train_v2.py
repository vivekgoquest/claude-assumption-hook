#!/usr/bin/env python3
"""Train, evaluate, and export the Assumption Guard v2 classifier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / "hook" / "assumption-guard.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("assumption_guard_runtime", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v2 = load_runtime_module()


DEFAULT_DATASET = REPO_ROOT / "training" / "assumption-guard-training-labeled.jsonl"
DEFAULT_REGRESSION = REPO_ROOT / "training" / "assumption-guard-regression-cases.json"
DEFAULT_REPLAY = REPO_ROOT / "training" / "assumption-guard-replay-cases.json"
DEFAULT_MODEL = REPO_ROOT / "model" / "assumption-guard-v2.onnx"
DEFAULT_TOKENIZER = REPO_ROOT / "model" / "assumption-guard-v2-tokenizer.json"
DEFAULT_META = REPO_ROOT / "model" / "assumption-guard-v2-meta.json"
DEFAULT_REPORT = REPO_ROOT / "model" / "assumption-guard-v2-report.json"

LABEL_TO_ID = {label: index for index, label in enumerate(v2.LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass
class SplitData:
    train: List[dict]
    dev: List[dict]
    test: List[dict]


class TextDataset(Dataset):
    def __init__(self, rows: Sequence[dict], tokenizer, max_length: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_TO_ID[row["intent"]], dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=str(DEFAULT_DATASET))
    parser.add_argument("--regression-cases", default=str(DEFAULT_REGRESSION))
    parser.add_argument("--replay-cases", default=str(DEFAULT_REPLAY))
    parser.add_argument("--output-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--output-meta", default=str(DEFAULT_META))
    parser.add_argument("--output-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-deberta", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def upgrade_row(row: dict, index: int) -> dict:
    text = row.get("text") or row.get("sentence") or ""
    intent = row["intent"]
    source_hash = row.get("source_hash") or hashlib.sha1(
        f"{intent}:{text}:{index}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "id": row.get("id", f"row-{index:04d}"),
        "text": text,
        "block": bool(row.get("block", intent in v2.BLOCK_LABELS)),
        "intent": intent,
        "source_type": row.get("source_type", "synthetic"),
        "source_hash": source_hash,
        "evidence_present": bool(
            row.get("evidence_present", v2.EVIDENCE_PATTERN.search(text))
        ),
        "quoted_or_code": bool(
            row.get("quoted_or_code", v2.INLINE_CODE.search(text) or text.startswith("```"))
        ),
        "review_status": row.get("review_status", "reviewed"),
        "notes": row.get("notes", ""),
    }


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append(upgrade_row(json.loads(line), index))
    return rows


def load_cases(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def split_rows(rows: Sequence[dict]) -> SplitData:
    train, dev, test = [], [], []
    for row in rows:
        bucket = stable_hash(row["source_hash"]) % 100
        if bucket < 70:
            train.append(row)
        elif bucket < 85:
            dev.append(row)
        else:
            test.append(row)
    return SplitData(train=train, dev=dev, test=test)


def binary_label(intent: str) -> bool:
    return intent in v2.BLOCK_LABELS


def compute_binary_metrics(expected: Sequence[bool], predicted: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for exp, pred in zip(expected, predicted) if exp and pred)
    tn = sum(1 for exp, pred in zip(expected, predicted) if (not exp) and (not pred))
    fp = sum(1 for exp, pred in zip(expected, predicted) if (not exp) and pred)
    fn = sum(1 for exp, pred in zip(expected, predicted) if exp and (not pred))
    block_recall = tp / (tp + fn) if (tp + fn) else 0.0
    pass_recall = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "block_recall": block_recall,
        "pass_recall": pass_recall,
        "precision": precision,
        "accuracy": accuracy,
    }


def compute_per_intent_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, dict]:
    metrics = {}
    for intent in v2.LABELS:
        actual_mask = [value == intent for value in expected]
        predicted_mask = [value == intent for value in predicted]
        tp = sum(1 for actual, pred in zip(actual_mask, predicted_mask) if actual and pred)
        fp = sum(1 for actual, pred in zip(actual_mask, predicted_mask) if (not actual) and pred)
        fn = sum(1 for actual, pred in zip(actual_mask, predicted_mask) if actual and (not pred))
        total_actual = sum(actual_mask)
        accuracy = (
            sum(1 for actual, pred in zip(expected, predicted) if actual == intent and pred == intent) / total_actual
            if total_actual
            else 1.0
        )
        metrics[intent] = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "accuracy": accuracy,
            "support": total_actual,
        }
    return metrics


def threshold_candidates():
    value = 0.20
    while value <= 0.8001:
        yield round(value, 2)
        value += 0.02


def select_threshold(dev_rows, predict_fn, regression_cases):
    search_results = []
    winning = None
    for threshold in threshold_candidates():
        dev_results = [runtime_decision_for_text(row["text"], predict_fn, threshold) for row in dev_rows]
        expected = [binary_label(row["intent"]) for row in dev_rows]
        predicted = [result["blocked"] for result in dev_results]
        metrics = compute_binary_metrics(expected, predicted)
        regression_matched = 0
        for case in regression_cases:
            result = runtime_decision_for_text(case["text"], predict_fn, threshold)
            if result["blocked"] == case["expected_block"]:
                regression_matched += 1
        row = {
            "threshold": threshold,
            "block_recall": metrics["block_recall"],
            "pass_recall": metrics["pass_recall"],
            "regression_matched": regression_matched,
            "regression_total": len(regression_cases),
        }
        search_results.append(row)
        if metrics["pass_recall"] >= 0.95 and regression_matched == len(regression_cases):
            if winning is None:
                winning = row
            elif row["block_recall"] > winning["block_recall"] + 1e-9:
                winning = row
            elif abs(row["block_recall"] - winning["block_recall"]) < 1e-9 and row["threshold"] < winning["threshold"]:
                winning = row

    if winning is None:
        winning = max(
            search_results,
            key=lambda row: (
                row["block_recall"],
                row["pass_recall"],
                row["regression_matched"],
                -row["threshold"],
            ),
        )
    return winning["threshold"], search_results


def predict_cases_with_probs(texts: Sequence[str], predict_fn, threshold_map: Sequence[float]):
    results = []
    for text in texts:
        probabilities, labels = predict_fn([text])
        p_block = probabilities[0]
        intent = labels[0]
        matched_by_threshold = {
            f"{threshold:.2f}": (p_block >= threshold) for threshold in threshold_map
        }
        results.append({"text": text, "p_block": p_block, "intent": intent, "matched_by_threshold": matched_by_threshold})
    return results


def infer_safe_intent(clause: str) -> str:
    inferred = v2.hard_pass_intent(clause)
    if inferred:
        return inferred
    if v2.TYPE_WORDS.search(clause):
        return "describe_type"
    if v2.CONDITIONAL_START.match(clause):
        return "reason_conditionally"
    if v2.INLINE_CODE.search(clause) or clause.startswith("```"):
        return "code_content"
    if v2.IDIOMATIC_COMPARE.search(clause):
        return "idiomatic_compare"
    return "reference_language"


def runtime_decision_for_text(text: str, predict_fn, threshold: float) -> dict:
    blocked_decisions = []
    observed_intent = None
    previous_clause = None
    previous_had_evidence = False

    for clause in v2.split_text_to_clauses(text):
        if previous_had_evidence and v2.RECOMMENDATION_HINT.search(clause):
            observed_intent = "recommend_supported"
            previous_clause = clause
            previous_had_evidence = False
            continue

        hard_pass = v2.hard_pass_intent(clause)
        if hard_pass:
            observed_intent = observed_intent or hard_pass
            previous_had_evidence = bool(v2.EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue

        hard_block = v2.hard_block_decision(clause)
        if hard_block:
            observed_intent = hard_block.intent
            blocked_decisions.append(
                {
                    "clause": clause,
                    "intent": hard_block.intent,
                    "source": hard_block.source,
                    "p_block": 1.0,
                }
            )
            previous_had_evidence = bool(v2.EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue

        if not v2.should_consider_clause(clause):
            observed_intent = observed_intent or infer_safe_intent(clause)
            previous_had_evidence = bool(v2.EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue

        probs, intents = predict_fn([clause])
        p_block = probs[0]
        intent = intents[0]
        observed_intent = intent
        if p_block >= threshold:
            blocked_decisions.append(
                {
                    "clause": clause,
                    "intent": intent,
                    "source": "model",
                    "p_block": p_block,
                }
            )
        previous_had_evidence = bool(v2.EVIDENCE_PATTERN.search(clause))
        previous_clause = clause

    return {
        "blocked": bool(blocked_decisions),
        "intent": observed_intent or "reference_language",
        "p_block": max((item["p_block"] for item in blocked_decisions), default=0.0),
        "decisions": blocked_decisions,
    }


def baseline_pipeline():
    return Pipeline(
        [
            ("hash", HashingVectorizer(ngram_range=(1, 3), n_features=2**18, alternate_sign=False)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )


def p_block_from_probabilities(probabilities: Sequence[float]) -> float:
    return sum(probabilities[LABEL_TO_ID[label]] for label in v2.BLOCK_LABELS)


def run_baseline_candidate(rows: SplitData, regression_cases: Sequence[dict], replay_cases: Sequence[dict]):
    pipe = baseline_pipeline()
    train_texts = [row["text"] for row in rows.train]
    train_labels = [row["intent"] for row in rows.train]
    pipe.fit(train_texts, train_labels)

    def predict_fn(texts):
        proba = pipe.predict_proba(texts)
        labels = pipe.predict(texts)
        p_blocks = [p_block_from_probabilities(row) for row in proba]
        return p_blocks, list(labels)

    threshold, threshold_search = select_threshold(rows.dev, predict_fn, regression_cases)
    return evaluate_candidate(
        name="baseline_linear_svm",
        kind="baseline",
        rows=rows,
        predict_fn=predict_fn,
        threshold=threshold,
        threshold_search=threshold_search,
        regression_cases=regression_cases,
        replay_cases=replay_cases,
        model_object=pipe,
    )


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_transformer(model, tokenizer, texts: Sequence[str], max_length: int, device) -> tuple[list[float], list[str], list[list[float]]]:
    model.eval()
    p_blocks: List[float] = []
    labels: List[str] = []
    logits_rows: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch_texts = list(texts[start : start + 64])
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            for logit_row, probability_row in zip(logits.tolist(), probabilities.tolist()):
                logits_rows.append(logit_row)
                labels.append(v2.LABELS[max(range(len(probability_row)), key=lambda idx: probability_row[idx])])
                p_blocks.append(p_block_from_probabilities(probability_row))
    return p_blocks, labels, logits_rows


def train_transformer_candidate(
    candidate_name: str,
    model_name: str,
    rows: SplitData,
    args,
    regression_cases: Sequence[dict],
    replay_cases: Sequence[dict],
):
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(v2.LABELS))
    model.to(device)

    train_dataset = TextDataset(rows.train, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )

    best_state = None
    best_snapshot = None
    patience_remaining = 2
    regression_total = len(regression_cases)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        dev_probs, dev_labels, _ = predict_transformer(
            model,
            tokenizer,
            [row["text"] for row in rows.dev],
            args.max_length,
            device,
        )
        predict_fn = lambda texts: predict_transformer(model, tokenizer, texts, args.max_length, device)[:2]
        threshold, threshold_search = select_threshold(rows.dev, predict_fn, regression_cases)
        dev_runtime_results = [runtime_decision_for_text(row["text"], predict_fn, threshold) for row in rows.dev]
        expected = [binary_label(row["intent"]) for row in rows.dev]
        predicted = [result["blocked"] for result in dev_runtime_results]
        metrics = compute_binary_metrics(expected, predicted)
        regression_matched = sum(
            1
            for case in regression_cases
            if runtime_decision_for_text(case["text"], predict_fn, threshold)["blocked"] == case["expected_block"]
        )
        snapshot = {
            "threshold": threshold,
            "threshold_search": threshold_search,
            "dev_metrics": metrics,
            "regression_matched": regression_matched,
        }
        is_better = False
        if best_snapshot is None:
            is_better = True
        else:
            current_tuple = (
                regression_matched == regression_total,
                metrics["pass_recall"] >= 0.95,
                metrics["block_recall"],
                metrics["pass_recall"],
            )
            best_tuple = (
                best_snapshot["regression_matched"] == regression_total,
                best_snapshot["dev_metrics"]["pass_recall"] >= 0.95,
                best_snapshot["dev_metrics"]["block_recall"],
                best_snapshot["dev_metrics"]["pass_recall"],
            )
            is_better = current_tuple > best_tuple

        if is_better:
            best_state = copy.deepcopy(model.state_dict())
            best_snapshot = snapshot
            patience_remaining = 2
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is None:
        raise RuntimeError(f"{candidate_name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return evaluate_candidate(
        name=candidate_name,
        kind="transformer",
        rows=rows,
        predict_fn=lambda texts: predict_transformer(model, tokenizer, texts, args.max_length, device)[:2],
        threshold=best_snapshot["threshold"],
        threshold_search=best_snapshot["threshold_search"],
        regression_cases=regression_cases,
        replay_cases=replay_cases,
        model_object=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        max_length=args.max_length,
    )


def evaluate_candidate(
    name: str,
    kind: str,
    rows: SplitData,
    predict_fn,
    threshold: float,
    threshold_search: Sequence[dict],
    regression_cases: Sequence[dict],
    replay_cases: Sequence[dict],
    model_object,
    tokenizer=None,
    device=None,
    model_name=None,
    max_length=128,
):
    def evaluate_rows(dataset_rows: Sequence[dict]):
        runtime_results = [runtime_decision_for_text(row["text"], predict_fn, threshold) for row in dataset_rows]
        expected_binary = [binary_label(row["intent"]) for row in dataset_rows]
        predicted_binary = [result["blocked"] for result in runtime_results]
        confusion = compute_binary_metrics(expected_binary, predicted_binary)
        per_intent = compute_per_intent_metrics(
            [row["intent"] for row in dataset_rows],
            [result["intent"] for result in runtime_results],
        )
        return runtime_results, confusion, per_intent

    dev_results, dev_confusion, dev_per_intent = evaluate_rows(rows.dev)
    test_results, test_confusion, test_per_intent = evaluate_rows(rows.test)

    regression_rows = []
    for case in regression_cases:
        result = runtime_decision_for_text(case["text"], predict_fn, threshold)
        blocked = result["blocked"]
        regression_rows.append(
            {
                "group": case.get("group", ""),
                "text": case["text"],
                "expected_block": case["expected_block"],
                "blocked": blocked,
                "matched": blocked == case["expected_block"],
                "p_block": result["p_block"],
                "intent": result["intent"],
            }
        )

    replay_rows = []
    for case in replay_cases:
        result = runtime_decision_for_text(case["text"], predict_fn, threshold)
        blocked = result["blocked"]
        replay_rows.append(
            {
                "group": case.get("group", ""),
                "text": case["text"],
                "expected_block": case["expected_block"],
                "blocked": blocked,
                "matched": blocked == case["expected_block"],
                "p_block": result["p_block"],
                "intent": result["intent"],
            }
        )

    replay_confusion = compute_binary_metrics(
        [case["expected_block"] for case in replay_cases],
        [row["blocked"] for row in replay_rows],
    )

    low_margin = []
    for row, result in zip(rows.test, test_results):
        low_margin.append(
            {
                "text": row["text"],
                "intent": row["intent"],
                "predicted_intent": result["intent"],
                "p_block": result["p_block"],
                "distance_to_threshold": abs(result["p_block"] - threshold),
            }
        )
    low_margin.sort(key=lambda item: item["distance_to_threshold"])

    false_negatives = []
    false_positives = []
    for row, result in zip(rows.test, test_results):
        blocked = result["blocked"]
        if binary_label(row["intent"]) and not blocked:
            false_negatives.append({"text": row["text"], "intent": row["intent"], "predicted_intent": result["intent"], "p_block": result["p_block"]})
        elif (not binary_label(row["intent"])) and blocked:
            false_positives.append({"text": row["text"], "intent": row["intent"], "predicted_intent": result["intent"], "p_block": result["p_block"]})

    candidate = {
        "name": name,
        "kind": kind,
        "model_name": model_name,
        "threshold": threshold,
        "threshold_search": list(threshold_search),
        "dev": {"confusion": dev_confusion, "per_intent": dev_per_intent},
        "test": {"confusion": test_confusion, "per_intent": test_per_intent},
        "regression": {
            "matched": sum(1 for row in regression_rows if row["matched"]),
            "total": len(regression_rows),
            "rows": regression_rows,
        },
        "replay": {
            **replay_confusion,
            "matched": sum(1 for row in replay_rows if row["matched"]),
            "total": len(replay_rows),
            "rows": replay_rows,
        },
        "false_negatives": false_negatives[:15],
        "false_positives": false_positives[:15],
        "low_margin_examples": low_margin[:20],
    }
    if kind == "transformer":
        candidate["tokenizer"] = tokenizer
        candidate["device"] = str(device)
        candidate["model_object"] = model_object
        candidate["max_length"] = max_length
    else:
        candidate["model_object"] = model_object
    return candidate


def candidate_passes_selection_rules(candidate: dict) -> bool:
    return (
        candidate["regression"]["matched"] == candidate["regression"]["total"]
        and candidate["dev"]["confusion"]["block_recall"] >= 0.95
        and candidate["dev"]["confusion"]["pass_recall"] >= 0.90
    )


def choose_candidate(candidates: Sequence[dict]) -> dict:
    survivors = [candidate for candidate in candidates if candidate_passes_selection_rules(candidate)]
    replay_survivors = [
        candidate
        for candidate in survivors
        if candidate["replay"]["block_recall"] >= 0.95 and candidate["replay"]["pass_recall"] >= 0.95
    ]
    ranking = {"baseline": 0, "transformer": 1}
    pool = replay_survivors or survivors or list(candidates)
    return max(
        pool,
        key=lambda candidate: (
            candidate["replay"]["block_recall"],
            candidate["replay"]["pass_recall"],
            -ranking.get(candidate["kind"], 9),
            -candidate["threshold"],
        ),
    )


def export_transformer_to_onnx(candidate: dict, output_model: Path):
    model = candidate["model_object"].cpu()
    tokenizer = candidate["tokenizer"]
    max_length = candidate["max_length"]
    dummy = tokenizer(
        ["placeholder clause"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    raw_path = output_model.with_suffix(".raw.onnx")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        raw_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    quantize_dynamic(raw_path, output_model, weight_type=QuantType.QInt8)
    return raw_path


def save_tokenizer(candidate: dict, output_tokenizer: Path):
    token_dir = output_tokenizer.parent / ".tokenizer-export"
    token_dir.mkdir(parents=True, exist_ok=True)
    candidate["tokenizer"].save_pretrained(token_dir)
    tokenizer_json = token_dir / "tokenizer.json"
    output_tokenizer.write_text(tokenizer_json.read_text())
    for child in token_dir.iterdir():
        child.unlink()
    token_dir.rmdir()


def onnx_predict(meta: dict, tokenizer_path: Path, model_path: Path, texts: Sequence[str]):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    p_blocks = []
    intents = []
    logits_rows = []
    for text in texts:
        encoding = tokenizer.encode(text)
        input_ids = encoding.ids[: meta["max_length"]]
        attention_mask = encoding.attention_mask[: meta["max_length"]]
        while len(input_ids) < meta["max_length"]:
            input_ids.append(0)
            attention_mask.append(0)
        logits = session.run(None, {"input_ids": [input_ids], "attention_mask": [attention_mask]})[0][0].tolist()
        logits_rows.append(logits)
        probabilities = v2.softmax(logits)
        intents.append(meta["labels"][max(range(len(probabilities)), key=lambda idx: probabilities[idx])])
        p_blocks.append(sum(probabilities[meta["labels"].index(label)] for label in meta["block_labels"]))
    return p_blocks, intents, logits_rows


def compute_onnx_parity(candidate: dict, meta: dict, tokenizer_path: Path, model_path: Path):
    sample_texts = [row["text"] for row in candidate["replay"]["rows"][:8]]
    if not sample_texts:
        return 0.0
    model_device = next(candidate["model_object"].parameters()).device
    _, _, torch_logits = predict_transformer(
        candidate["model_object"],
        candidate["tokenizer"],
        sample_texts,
        candidate["max_length"],
        model_device,
    )
    _, _, onnx_logits = onnx_predict(meta, tokenizer_path, model_path, sample_texts)
    max_delta = 0.0
    for torch_row, onnx_row in zip(torch_logits, onnx_logits):
        for torch_value, onnx_value in zip(torch_row, onnx_row):
            max_delta = max(max_delta, abs(torch_value - onnx_value))
    return max_delta


def benchmark_latency(meta: dict, tokenizer_path: Path, model_path: Path, sample_texts: Sequence[str]):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def run_once(text):
        encoding = tokenizer.encode(text)
        input_ids = encoding.ids[: meta["max_length"]]
        attention_mask = encoding.attention_mask[: meta["max_length"]]
        while len(input_ids) < meta["max_length"]:
            input_ids.append(0)
            attention_mask.append(0)
        session.run(None, {"input_ids": [input_ids], "attention_mask": [attention_mask]})

    single_runs = []
    for text in sample_texts[: min(20, len(sample_texts))]:
        start = time.perf_counter()
        run_once(text)
        single_runs.append((time.perf_counter() - start) * 1000)

    multi_start = time.perf_counter()
    for text in sample_texts[:20]:
        run_once(text)
    multi_duration = (time.perf_counter() - multi_start) * 1000

    single_runs.sort()
    p50 = single_runs[len(single_runs) // 2] if single_runs else 0.0
    p95 = single_runs[min(len(single_runs) - 1, math.floor(len(single_runs) * 0.95))] if single_runs else 0.0
    return {"p50_ms": p50, "p95_ms": p95, "batch20_ms": multi_duration}


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.training_data)
    regression_path = Path(args.regression_cases)
    replay_path = Path(args.replay_cases)
    output_model = Path(args.output_model)
    output_tokenizer = Path(args.output_tokenizer)
    output_meta = Path(args.output_meta)
    output_report = Path(args.output_report)

    output_model.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    splits = split_rows(rows)
    regression_cases = load_cases(regression_path)
    replay_cases = load_cases(replay_path)

    candidates = []
    candidates.append(run_baseline_candidate(splits, regression_cases, replay_cases))
    candidates.append(
        train_transformer_candidate(
            candidate_name="minilm_l6",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            rows=splits,
            args=args,
            regression_cases=regression_cases,
            replay_cases=replay_cases,
        )
    )
    if not args.skip_deberta:
        try:
            candidates.append(
                train_transformer_candidate(
                    candidate_name="deberta_v3_small",
                    model_name="microsoft/deberta-v3-small",
                    rows=splits,
                    args=args,
                    regression_cases=regression_cases,
                    replay_cases=replay_cases,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised during manual runs
            candidates.append(
                {
                    "name": "deberta_v3_small",
                    "kind": "transformer",
                    "error": f"{type(exc).__name__}: {exc}",
                    "threshold": None,
                    "dev": {"confusion": {"block_recall": 0.0, "pass_recall": 0.0}},
                    "replay": {"block_recall": 0.0, "pass_recall": 0.0, "matched": 0, "total": len(replay_cases), "rows": []},
                    "regression": {"matched": 0, "total": len(regression_cases), "rows": []},
                    "threshold_search": [],
                    "false_negatives": [],
                    "false_positives": [],
                    "low_margin_examples": [],
                }
            )

    selected = choose_candidate([candidate for candidate in candidates if "error" not in candidate])
    if selected["kind"] != "transformer":
        transformer_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("kind") == "transformer" and "error" not in candidate
        ]
        if not transformer_candidates:
            raise RuntimeError("No transformer candidate satisfied export requirements")
        selected = max(
            transformer_candidates,
            key=lambda candidate: (
                candidate["replay"]["block_recall"],
                candidate["replay"]["pass_recall"],
                candidate["dev"]["confusion"]["block_recall"],
                candidate["dev"]["confusion"]["pass_recall"],
            ),
        )

    save_tokenizer(selected, output_tokenizer)
    raw_onnx_path = export_transformer_to_onnx(selected, output_model)
    meta = {
        "version": "2.0",
        "labels": v2.LABELS,
        "block_labels": sorted(v2.BLOCK_LABELS),
        "threshold": selected["threshold"],
        "model_name": selected["model_name"],
        "max_length": args.max_length,
        "backend": "onnx",
    }
    output_meta.write_text(json.dumps(meta, indent=2))

    onnx_parity = compute_onnx_parity(selected, meta, output_tokenizer, raw_onnx_path)
    latency = benchmark_latency(meta, output_tokenizer, output_model, [row["text"] for row in replay_cases])
    raw_onnx_path.unlink(missing_ok=True)

    report = {
        "dataset": {
            "rows": len(rows),
            "train": len(splits.train),
            "dev": len(splits.dev),
            "test": len(splits.test),
            "label_counts": {
                label: sum(1 for row in rows if row["intent"] == label) for label in v2.LABELS
            },
        },
        "candidates": [],
        "selected_candidate": selected["name"],
        "threshold_search": selected["threshold_search"],
        "regression_results": {
            "matched": selected["regression"]["matched"],
            "total": selected["regression"]["total"],
            "rows": selected["regression"]["rows"],
        },
        "replay_results": {
            "matched": selected["replay"]["matched"],
            "total": selected["replay"]["total"],
            "block_recall": selected["replay"]["block_recall"],
            "pass_recall": selected["replay"]["pass_recall"],
            "rows": selected["replay"]["rows"],
        },
        "evaluation": {
            "test_confusion": selected["test"]["confusion"],
            "per_intent": selected["test"]["per_intent"],
            "false_negatives": selected["false_negatives"],
            "false_positives": selected["false_positives"],
            "low_margin_examples": selected["low_margin_examples"],
        },
        "onnx_parity_max_abs_delta": onnx_parity,
        "latency": latency,
    }
    for candidate in candidates:
        if "error" in candidate:
            report["candidates"].append(candidate)
            continue
        report["candidates"].append(
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "model_name": candidate.get("model_name"),
                "threshold": candidate["threshold"],
                "dev_confusion": candidate["dev"]["confusion"],
                "test_confusion": candidate["test"]["confusion"],
                "regression_matched": candidate["regression"]["matched"],
                "regression_total": candidate["regression"]["total"],
                "replay_block_recall": candidate["replay"]["block_recall"],
                "replay_pass_recall": candidate["replay"]["pass_recall"],
            }
        )

    native_arch = os.uname().machine
    latency_target_env = native_arch == "arm64"
    acceptance = {
        "regression_full_match": report["regression_results"]["matched"] == report["regression_results"]["total"],
        "replay_block_recall": report["replay_results"]["block_recall"] >= 0.95,
        "replay_pass_recall": report["replay_results"]["pass_recall"] >= 0.95,
        "reference_language_ok": report["evaluation"]["per_intent"]["reference_language"]["accuracy"] >= 0.97,
        "verified_limitation_ok": report["evaluation"]["per_intent"]["verified_limitation"]["accuracy"] >= 0.97,
        "reason_conditionally_ok": report["evaluation"]["per_intent"]["reason_conditionally"]["accuracy"] >= 0.97,
        "describe_type_ok": report["evaluation"]["per_intent"]["describe_type"]["accuracy"] >= 0.95,
        "idiomatic_compare_ok": report["evaluation"]["per_intent"]["idiomatic_compare"]["accuracy"] >= 0.95,
        "onnx_parity_ok": onnx_parity <= 1e-3,
        "latency_ok": True if not latency_target_env else (latency["p50_ms"] <= 15 and latency["p95_ms"] <= 40 and latency["batch20_ms"] <= 400),
    }
    acceptance["latency_environment_matches_target"] = latency_target_env
    acceptance["meets_acceptance_bar"] = all(
        value for key, value in acceptance.items() if key != "latency_environment_matches_target"
    )
    report["acceptance"] = acceptance

    output_report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "selected_candidate": selected["name"],
                "threshold": selected["threshold"],
                "acceptance": acceptance,
                "report": str(output_report),
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assumption Guard v2 hook and runtime."""

from __future__ import annotations

import argparse
import copy
import json
import hashlib
import io
import math
import os
import pickle
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


LABELS: List[str] = [
    "assert_unverified",
    "recommend_unverified",
    "capability_promise_unverified",
    "unchecked_limitation",
    "verification_narration",
    "reference_language",
    "verified_limitation",
    "dependency_gap_grounded",
    "describe_type",
    "reason_conditionally",
    "code_content",
    "idiomatic_compare",
    "recommend_supported",
]
BLOCK_LABELS = {
    "assert_unverified",
    "recommend_unverified",
    "capability_promise_unverified",
    "unchecked_limitation",
}

FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]+`")
DOUBLE_QUOTED = re.compile(r'"[^"\n]+"')
HARD_BLOCK_PATTERNS = [
    (
        "unchecked_limitation",
        re.compile(
            r"\b(I need to verify|need to verify|needs to verify|"
            r"I have not checked|I haven't checked|"
            r"I did not check|I didn't check|"
            r"I have not looked at|I haven't looked at|"
            r"I did not look at|I didn't look at|"
            r"I can't confirm|I cannot confirm|"
            r"I can't determine|I cannot determine|"
            r"I'm not sure|I am not sure|"
            r"I'm not certain|I am not certain|"
            r"off the top of my head|to know for sure|"
            r"requires benchmarking|requires checking|requires running|"
            r"requires verifying|needs running|needs checking)\b",
            re.IGNORECASE,
        ),
    ),
]
PREFILTER_PATTERNS = [
    re.compile(
        r"\b(maybe|perhaps|probably|likely|possibly|apparently|seemingly|"
        r"conceivably|roughly|around|about|approximately|relatively|fairly|somewhat|"
        r"kind of|sort of|rather)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(I think|I believe|I suspect|I assume|I presume|I imagine|"
        r"I guess|I'd guess|it seems|it appears|it looks like|"
        r"seems like|my understanding is|"
        r"from what I can tell|as far as I can tell|as far as I'm aware|"
        r"to my knowledge|if I recall correctly|if memory serves)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(might|may|could)\b", re.IGNORECASE),
    re.compile(
        r"\b(rename|refactor|extract|simplify|clearer|more readable|"
        r"unit test|integration test|split|cleanup|clean up|remove|"
        r"prefer|use|move this|normalize|validate|document|comment)\b",
        re.IGNORECASE,
    ),
]
EVIDENCE_PATTERN = re.compile(
    r"\b(I (checked|searched|read|reviewed|inspected|looked at|looked through|"
    r"tested|verified|grepped|grep'd|ran|traced|confirmed|found)\b|"
    r"after (checking|searching|reviewing|running|testing|reading)\b|"
    r"based on (the|this) (repo|codebase|tests|trace|logs|call sites|config|docs))\b",
    re.IGNORECASE,
)
META_REPORT_PATTERN = re.compile(
    r"\b(hook flagged|hook blocked|this sentence should block|regression case|"
    r"metrics report|report artifact|taxonomy|classifier|label|intent|"
    r"pattern matched|dataset row|training example|sample response|"
    r"phrase .* trigger the hook|detector matched|flag \d+|classifies)\b",
    re.IGNORECASE,
)
TYPE_WORDS = re.compile(
    r"\b(None|null|undefined|string|number|int|float|bool|boolean|array|list|"
    r"dict|map|optional|return value|callback|payload|handler|argument|"
    r"field|property|enum|variant|promise|response|token|variable|generic|"
    r"subclass|base|callable|client|instance|object|uninitialized)\b",
    re.IGNORECASE,
)
CONDITIONAL_START = re.compile(r"^\s*(if|when|unless|without)\b", re.IGNORECASE)
IDIOMATIC_COMPARE = re.compile(r"\b(rather than|instead of)\b", re.IGNORECASE)
COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*)")
RECOMMENDATION_HINT = re.compile(
    r"\b(rename|refactor|extract|simplify|clearer|more readable|"
    r"unit test|integration test|prefer|use|split|cleanup|clean up|"
    r"reuse|reusing|return early)\b",
    re.IGNORECASE,
)
LIMITATION_PATTERN = re.compile(
    r"\b(cannot confirm|can't confirm|cannot determine|can't determine|"
    r"need to verify|requires benchmarking|requires checking|"
    r"requires running|to know for sure|cannot find definitive evidence|"
    r"cannot find evidence|can't find evidence|haven't looked at|"
    r"have not looked at|didn't look at|did not look at)\b",
    re.IGNORECASE,
)
VERIFICATION_NARRATION = re.compile(
    r"^\s*(let me|i(?:'ll| will)|i(?:'m| am) going to)\s+"
    r"(verify|check|confirm|inspect|test|look(?:\s+into)?|see)\s+(if|whether)\b",
    re.IGNORECASE,
)
DEPENDENCY_GAP_LIMITATION = re.compile(
    r"^\s*(but\s+)?without\s+"
    r"(?!checking\b|running\b|verifying\b|looking\b|testing\b|searching\b|reviewing\b|confirming\b)"
    r"[^.]{1,160}\bI\s+can(?:not|'t)\b",
    re.IGNORECASE,
)
ARCHITECTURE_NOUNS = re.compile(
    r"\b(TypeScript|JavaScript|redis|staging|production|"
    r"state machine|queue worker|service|shared module)\b",
    re.IGNORECASE,
)
DIRECTIVE_START = re.compile(r"^\s*(use|prefer|return|reuse|deploy|consider|store)\b", re.IGNORECASE)
CAPABILITY_PROMISE_PATTERN = re.compile(
    r"\b(?:if you want,?\s+)?i can\s+"
    r"(check|identify|remove|revert|delete|fix|confirm|determine|find|undo|verify|figure out|work out)\b",
    re.IGNORECASE,
)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+")
PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(?:/[\w .:+@-]+)+")
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
CHANNEL_ID_RE = re.compile(r"\bUC[a-zA-Z0-9_-]{10,}\b")

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
WORKSPACE_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else PACKAGE_DIR
CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
STATE_DIR = os.environ.get(
    "ASSUMPTION_GUARD_STATE_DIR",
    str(PACKAGE_DIR / "state"),
)
STATE_PATH = Path(STATE_DIR)
BASELINE_DIR = PACKAGE_DIR / "baseline"
MODEL_PATH = os.environ.get(
    "ASSUMPTION_GUARD_MODEL_PATH",
    str(PACKAGE_DIR / "assumption-guard-v2.onnx"),
)
TOKENIZER_PATH = os.environ.get(
    "ASSUMPTION_GUARD_TOKENIZER_PATH",
    str(PACKAGE_DIR / "assumption-guard-v2-tokenizer.json"),
)
META_PATH = os.environ.get(
    "ASSUMPTION_GUARD_META_PATH",
    str(PACKAGE_DIR / "assumption-guard-v2-meta.json"),
)
LOG_PATH = os.environ.get(
    "ASSUMPTION_GUARD_LOG_PATH",
    os.path.join(STATE_DIR, "assumption-guard.log.jsonl"),
)
QUEUE_PATH = os.environ.get(
    "ASSUMPTION_GUARD_QUEUE_PATH",
    os.path.join(STATE_DIR, "learning-queue.jsonl"),
)
CAPTURE_MODE = os.environ.get("ASSUMPTION_GUARD_CAPTURE_MODE", "learning")
LOW_MARGIN = float(os.environ.get("ASSUMPTION_GUARD_LOW_MARGIN", "0.08"))
TRIGGER_MODE = os.environ.get("ASSUMPTION_GUARD_TRIGGER_MODE", "post_append")
TRIGGER_PYTHON = os.environ.get("ASSUMPTION_GUARD_TRIGGER_PYTHON", sys.executable)
TRANSCRIPT_ROOT = os.environ.get("ASSUMPTION_GUARD_TRANSCRIPT_ROOT", os.path.join(CLAUDE_DIR, "projects"))
REVIEW_CLAUDE_CMD = os.environ.get("ASSUMPTION_GUARD_REVIEW_CLAUDE_CMD", "claude")
DISABLE_HOOK = os.environ.get("ASSUMPTION_GUARD_DISABLE") == "1"
TRAINING_PYTHON = os.environ.get("ASSUMPTION_GUARD_TRAIN_PYTHON") or shutil.which("python3.11") or sys.executable
JUDGEMENT_PASS_LABELS = {
    "verification_narration",
    "verified_limitation",
    "dependency_gap_grounded",
    "recommend_supported",
}

_CLASSIFIER = None
_CLASSIFIER_STATUS = None
_CLASSIFIER_LOAD_ATTEMPTED = False


class _SelfProxy:
    def __getattr__(self, name):
        return globals()[name]


v2 = _SelfProxy()


def baseline_file_path(filename: str, repo_subdir: str) -> Path:
    return BASELINE_DIR / filename


DEFAULT_DATASET = baseline_file_path("assumption-guard-training-labeled.jsonl", "training")
DEFAULT_REGRESSION = baseline_file_path("assumption-guard-regression-cases.json", "training")
DEFAULT_REPLAY = baseline_file_path("assumption-guard-replay-cases.json", "training")
DEFAULT_REPORT = baseline_file_path("assumption-guard-v2-report.json", "model")


class ClauseDecision:
    def __init__(
        self,
        clause: str,
        blocked: bool,
        intent: str,
        source: str,
        p_block: Optional[float] = None,
        matched_terms: Optional[List[str]] = None,
    ):
        self.clause = clause
        self.blocked = blocked
        self.intent = intent
        self.source = source
        self.p_block = p_block
        self.matched_terms = matched_terms


class ClauseObservation:
    def __init__(
        self,
        clause: str,
        previous_clause: Optional[str],
        next_clause: Optional[str],
        blocked: bool,
        intent: str,
        source: str,
        p_block: Optional[float] = None,
        matched_terms: Optional[List[str]] = None,
        evidence_present: bool = False,
        quoted_or_code: bool = False,
        heuristic_rescue: bool = False,
        mixed_evidence_gap: bool = False,
        considered: bool = True,
    ):
        self.clause = clause
        self.previous_clause = previous_clause
        self.next_clause = next_clause
        self.blocked = blocked
        self.intent = intent
        self.source = source
        self.p_block = p_block
        self.matched_terms = matched_terms or []
        self.evidence_present = evidence_present
        self.quoted_or_code = quoted_or_code
        self.heuristic_rescue = heuristic_rescue
        self.mixed_evidence_gap = mixed_evidence_gap
        self.considered = considered


class OnnxIntentClassifier:
    """Optional ONNX-backed multiclass classifier."""

    def __init__(self, model_path: str, tokenizer_path: str, meta_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.meta_path = meta_path
        self._session = None
        self._tokenizer = None
        self._meta = None

    @property
    def meta(self) -> Dict[str, object]:
        if self._meta is None:
            raise RuntimeError("classifier metadata is not loaded")
        return self._meta

    def load(self):
        try:
            from tokenizers import Tokenizer  # pylint: disable=import-outside-toplevel
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - exercised in subprocess tests
            raise RuntimeError(f"ml_import_error:{type(exc).__name__}: {exc}") from exc

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"meta_missing:{self.meta_path}")
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"tokenizer_missing:{self.tokenizer_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model_missing:{self.model_path}")

        with open(self.meta_path, "r", encoding="utf-8") as handle:
            self._meta = json.load(handle)
        self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        return self

    def predict(self, clause: str) -> ClauseDecision:
        if self._session is None or self._tokenizer is None or self._meta is None:
            raise RuntimeError("classifier is not loaded")

        max_length = int(self._meta.get("max_length", 128))
        encoding = self._tokenizer.encode(clause)
        input_ids = encoding.ids[:max_length]
        attention_mask = encoding.attention_mask[:max_length]
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)

        logits = self._session.run(
            None,
            {"input_ids": [input_ids], "attention_mask": [attention_mask]},
        )[0][0].tolist()
        labels = self._meta["labels"]
        probabilities = softmax(logits)
        best_index = max(range(len(labels)), key=lambda index: probabilities[index])
        p_block = sum(
            probabilities[index]
            for index, label in enumerate(labels)
            if label in set(self._meta["block_labels"])
        )
        return ClauseDecision(
            clause=clause,
            blocked=p_block >= float(self._meta["threshold"]),
            intent=labels[best_index],
            source="onnx",
            p_block=p_block,
        )


def runtime_status(ml_available, fallback_reason=None, error_stage=None, error_detail=None):
    return {
        "ml_available": ml_available,
        "fallback_reason": fallback_reason,
        "error_stage": error_stage,
        "error_detail": error_detail,
    }


def stable_sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sanitize_learning_text(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = CHANNEL_ID_RE.sub("<CHANNEL_ID>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = LONG_NUMBER_RE.sub("<NUM>", text)
    return re.sub(r"\s+", " ", text).strip()


def daily_queue_path(queue_path: str) -> str:
    queue_root = os.path.join(os.path.dirname(queue_path), "queue")
    return os.path.join(queue_root, f"{datetime.now(timezone.utc).date().isoformat()}.jsonl")


def candidate_hash_for(sanitized_text: str, previous_clause: str, next_clause: str) -> str:
    return stable_sha1(f"{sanitized_text}\n{previous_clause}\n{next_clause}")


def deterministic_sample(candidate_hash: str, ratio: float = 0.02) -> bool:
    sample_value = int(candidate_hash[:8], 16) / 0xFFFFFFFF
    return sample_value < ratio


def quoted_or_code_clause(clause: str) -> bool:
    stripped = clause.strip()
    return bool(stripped.startswith("```") or COMMENT_LINE.match(stripped) or INLINE_CODE.search(stripped))


def serialize_observation(observation: ClauseObservation, threshold: Optional[float], session_id: str, cwd: str):
    now = datetime.now(timezone.utc).isoformat()
    sanitized_text = sanitize_learning_text(observation.clause)
    sanitized_previous = sanitize_learning_text(observation.previous_clause or "")
    sanitized_next = sanitize_learning_text(observation.next_clause or "")
    candidate_hash = candidate_hash_for(sanitized_text, sanitized_previous, sanitized_next)
    project = os.path.basename(cwd) if cwd else ""
    return {
        "candidate_id": f"candidate-{candidate_hash[:16]}",
        "candidate_hash": candidate_hash,
        "ts": now,
        "text": sanitized_text,
        "sanitized_text": sanitized_text,
        "previous_clause": sanitized_previous,
        "next_clause": sanitized_next,
        "decision": "block" if observation.blocked else "pass",
        "intent": observation.intent,
        "source": observation.source,
        "p_block": observation.p_block,
        "threshold": threshold,
        "stage": "capture",
        "matched_terms": observation.matched_terms,
        "evidence_present": observation.evidence_present,
        "quoted_or_code": observation.quoted_or_code,
        "review_status": "needs_review",
        "session_hash": stable_sha1(session_id or "unknown-session")[:16],
        "message_hash": stable_sha1(f"{project}:{sanitized_text}")[:16],
        "project": project,
    }


def append_learning_candidates(rows: Sequence[dict]):
    if CAPTURE_MODE == "off" or not rows:
        return False
    try:
        queue_path = QUEUE_PATH
        seen_path = os.path.join(os.path.dirname(queue_path), "seen-candidate-hashes.txt")
        day_path = daily_queue_path(queue_path)
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        os.makedirs(os.path.dirname(day_path), exist_ok=True)
        seen_hashes = set()
        if os.path.exists(seen_path):
            with open(seen_path, "r", encoding="utf-8") as handle:
                seen_hashes = {line.strip() for line in handle if line.strip()}

        new_rows = [row for row in rows if row["candidate_hash"] not in seen_hashes]
        if not new_rows:
            return False

        with open(queue_path, "a", encoding="utf-8") as merged, open(day_path, "a", encoding="utf-8") as daily, open(
            seen_path, "a", encoding="utf-8"
        ) as seen_handle:
            for row in new_rows:
                payload = json.dumps(row, default=str)
                merged.write(payload + "\n")
                daily.write(payload + "\n")
                seen_handle.write(row["candidate_hash"] + "\n")
        return True
    except OSError:
        return False


def self_command(mode: str, *extra_args: str) -> List[str]:
    return [TRIGGER_PYTHON, str(SCRIPT_PATH), mode, *extra_args]


def maybe_trigger_learning_cycle():
    if TRIGGER_MODE != "post_append":
        return
    try:
        subprocess.Popen(
            self_command(
                "maybe-trigger",
                "--state-dir",
                STATE_DIR,
                "--queue-path",
                QUEUE_PATH,
                "--log-path",
                LOG_PATH,
                "--transcript-root",
                TRANSCRIPT_ROOT,
                "--claude-cmd",
                REVIEW_CLAUDE_CMD,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return


def get_classifier():
    global _CLASSIFIER, _CLASSIFIER_STATUS, _CLASSIFIER_LOAD_ATTEMPTED

    if _CLASSIFIER_LOAD_ATTEMPTED:
        return _CLASSIFIER, dict(_CLASSIFIER_STATUS)

    _CLASSIFIER_LOAD_ATTEMPTED = True

    try:
        classifier = OnnxIntentClassifier(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            meta_path=META_PATH,
        ).load()
    except FileNotFoundError as exc:
        detail = str(exc)
        if detail.startswith("meta_missing:"):
            _CLASSIFIER_STATUS = runtime_status(False, "meta_missing", "meta_load", detail)
        elif detail.startswith("tokenizer_missing:"):
            _CLASSIFIER_STATUS = runtime_status(False, "tokenizer_missing", "tokenizer_load", detail)
        else:
            _CLASSIFIER_STATUS = runtime_status(False, "model_missing", "model_load", detail)
        return None, dict(_CLASSIFIER_STATUS)
    except RuntimeError as exc:
        detail = str(exc)
        if detail.startswith("ml_import_error:"):
            _CLASSIFIER_STATUS = runtime_status(False, "ml_import_error", "ml_import", detail)
        else:
            _CLASSIFIER_STATUS = runtime_status(False, "asset_load_error", "model_load", detail)
        return None, dict(_CLASSIFIER_STATUS)
    except json.JSONDecodeError as exc:
        _CLASSIFIER_STATUS = runtime_status(
            False,
            "asset_load_error",
            "meta_load",
            f"{type(exc).__name__}: {exc}",
        )
        return None, dict(_CLASSIFIER_STATUS)
    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        detail = f"{type(exc).__name__}: {exc}"
        error_stage = "tokenizer_load" if "tokenizer" in detail.lower() else "model_load"
        _CLASSIFIER_STATUS = runtime_status(False, "model_load_error", error_stage, detail)
        return None, dict(_CLASSIFIER_STATUS)

    _CLASSIFIER = classifier
    _CLASSIFIER_STATUS = runtime_status(True)
    return _CLASSIFIER, dict(_CLASSIFIER_STATUS)


def split_text_to_clauses(text: str) -> List[str]:
    if not text:
        return []

    blocks = re.split(r"(```.*?```)", text.replace("\r\n", "\n"), flags=re.DOTALL)
    clauses: List[str] = []
    for block in blocks:
        if not block or not block.strip():
            continue
        if block.startswith("```"):
            clauses.append(block.strip())
            continue
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            clauses.extend(_split_text_fragment(stripped))
    return [re.sub(r"\s+", " ", clause).strip() for clause in clauses if clause.strip()]


def _split_text_fragment(text: str) -> List[str]:
    fragments = [text]
    for pattern in (r"\s*;\s*", r"\s+[—–]\s+", r",\s+so\s+"):
        next_fragments: List[str] = []
        for fragment in fragments:
            next_fragments.extend(re.split(pattern, fragment, flags=re.IGNORECASE))
        fragments = next_fragments

    clauses: List[str] = []
    for fragment in fragments:
        clauses.extend(re.split(r"(?<=[.!?])\s+", fragment))
    return [clause for clause in clauses if clause.strip()]


def strip_inline_literals(text: str) -> str:
    text = INLINE_CODE.sub(" ", text)
    text = DOUBLE_QUOTED.sub(" ", text)
    return text


def hard_pass_intent(clause: str, previous_had_evidence: bool = False) -> Optional[str]:
    stripped = clause.strip()
    lowered = stripped.lower()
    if not stripped:
        return "reference_language"
    if stripped.startswith("```") or COMMENT_LINE.match(stripped):
        return "code_content"
    if VERIFICATION_NARRATION.match(stripped):
        return "verification_narration"
    if META_REPORT_PATTERN.search(lowered):
        return "reference_language"
    if lowered.startswith("the hook flagged ") or lowered.startswith("the hook blocked "):
        return "reference_language"
    if (
        CONDITIONAL_START.match(stripped)
        and (
            not RECOMMENDATION_HINT.search(stripped)
            or re.search(r"\b(would|could|may|might)\b", stripped, re.IGNORECASE)
        )
        and not LIMITATION_PATTERN.search(stripped)
        and "can't confirm" not in lowered
        and "cannot confirm" not in lowered
        and "need to verify" not in lowered
    ):
        return "reason_conditionally"
    if re.match(r"^\s*with\b", stripped, re.IGNORECASE) and re.search(
        r"\b(may|might|could)\b", stripped, re.IGNORECASE
    ):
        return "reason_conditionally"
    if TYPE_WORDS.search(stripped) and re.search(r"\b(or|may|might|could)\b", stripped, re.IGNORECASE):
        return "describe_type"
    if re.search(r"\bstore state\b", stripped, re.IGNORECASE) and re.search(
        r"\b(database|memory)\b", stripped, re.IGNORECASE
    ):
        return None
    if (
        IDIOMATIC_COMPARE.search(stripped)
        and not ARCHITECTURE_NOUNS.search(stripped)
        and not re.search(
            r"\b(maybe|perhaps|probably|likely|I think|I believe|I suspect|"
            r"not sure|can't confirm|cannot confirm|need to verify)\b",
            stripped,
            re.IGNORECASE,
        )
    ):
        return "idiomatic_compare"
    if DIRECTIVE_START.match(stripped) and INLINE_CODE.search(stripped) and not ARCHITECTURE_NOUNS.search(
        stripped
    ):
        return "code_content"
    if DIRECTIVE_START.match(stripped) and re.search(
        r"\b(typed object|dict|dictionary|hashmap|map|callable|client instance)\b",
        stripped,
        re.IGNORECASE,
    ):
        return "idiomatic_compare"
    if EVIDENCE_PATTERN.search(stripped) and (
        LIMITATION_PATTERN.search(stripped) or DEPENDENCY_GAP_LIMITATION.search(stripped)
    ):
        return "dependency_gap_grounded" if DEPENDENCY_GAP_LIMITATION.search(stripped) else "verified_limitation"
    if previous_had_evidence and DEPENDENCY_GAP_LIMITATION.search(stripped):
        return "dependency_gap_grounded"
    if EVIDENCE_PATTERN.search(stripped) and RECOMMENDATION_HINT.search(stripped):
        return "recommend_supported"
    outside_literals = strip_inline_literals(stripped)
    if INLINE_CODE.search(stripped) and not _contains_risky_marker(outside_literals):
        return "code_content"
    return None


def hard_block_decision(clause: str) -> Optional[ClauseDecision]:
    for intent, pattern in HARD_BLOCK_PATTERNS:
        match = pattern.search(clause)
        if match:
            return ClauseDecision(
                clause=clause,
                blocked=True,
                intent=intent,
                source="hard_rule",
                matched_terms=[match.group(0)],
            )
    return None


def should_consider_clause(clause: str) -> bool:
    cleaned = strip_inline_literals(clause)
    return any(pattern.search(cleaned) for pattern in PREFILTER_PATTERNS) or bool(
        CAPABILITY_PROMISE_PATTERN.search(cleaned)
    )


def _contains_risky_marker(text: str) -> bool:
    return should_consider_clause(text) or bool(LIMITATION_PATTERN.search(text))


def regex_only_intent(clause: str, previous_clause: Optional[str] = None) -> str:
    if CAPABILITY_PROMISE_PATTERN.search(clause):
        return "capability_promise_unverified"
    if RECOMMENDATION_HINT.search(clause):
        if previous_clause and EVIDENCE_PATTERN.search(previous_clause):
            return "recommend_supported"
        return "recommend_unverified"
    if LIMITATION_PATTERN.search(clause):
        if previous_clause and EVIDENCE_PATTERN.search(previous_clause):
            return "verified_limitation"
        return "unchecked_limitation"
    return "assert_unverified"


def softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def evaluate_text_with_trace(
    text: str,
    classifier: Optional[OnnxIntentClassifier] = None,
) -> tuple[List[ClauseDecision], List[ClauseObservation]]:
    decisions: List[ClauseDecision] = []
    observations: List[ClauseObservation] = []
    clauses = split_text_to_clauses(text)
    previous_clause = None
    previous_had_evidence = False
    for index, clause in enumerate(clauses):
        next_clause = clauses[index + 1] if index + 1 < len(clauses) else None
        evidence_present = bool(EVIDENCE_PATTERN.search(clause))
        mixed_evidence_gap = bool(previous_had_evidence and DEPENDENCY_GAP_LIMITATION.search(clause))
        common_kwargs = {
            "clause": clause,
            "previous_clause": previous_clause,
            "next_clause": next_clause,
            "evidence_present": evidence_present,
            "quoted_or_code": quoted_or_code_clause(clause),
            "mixed_evidence_gap": mixed_evidence_gap,
        }
        if previous_had_evidence and RECOMMENDATION_HINT.search(clause):
            observations.append(
                ClauseObservation(
                    blocked=False,
                    intent="recommend_supported",
                    source="carry_forward",
                    heuristic_rescue=True,
                    considered=True,
                    **common_kwargs,
                )
            )
            previous_clause = clause
            previous_had_evidence = False
            continue
        hard_pass = hard_pass_intent(clause, previous_had_evidence=previous_had_evidence)
        if hard_pass:
            observations.append(
                ClauseObservation(
                    blocked=False,
                    intent=hard_pass,
                    source="hard_pass",
                    heuristic_rescue=hard_pass in JUDGEMENT_PASS_LABELS,
                    considered=True,
                    **common_kwargs,
                )
            )
            previous_had_evidence = evidence_present
            previous_clause = clause
            continue
        hard_block = hard_block_decision(clause)
        if hard_block:
            decisions.append(hard_block)
            observations.append(
                ClauseObservation(
                    blocked=True,
                    intent=hard_block.intent,
                    source=hard_block.source,
                    matched_terms=hard_block.matched_terms,
                    p_block=hard_block.p_block,
                    considered=True,
                    **common_kwargs,
                )
            )
            previous_had_evidence = evidence_present
            previous_clause = clause
            continue
        if not should_consider_clause(clause):
            observations.append(
                ClauseObservation(
                    blocked=False,
                    intent=hard_pass_intent(clause, previous_had_evidence=previous_had_evidence)
                    or "reference_language",
                    source="ignored",
                    considered=False,
                    **common_kwargs,
                )
            )
            previous_had_evidence = evidence_present
            previous_clause = clause
            continue
        if classifier is None:
            intent = regex_only_intent(
                clause,
                previous_clause=previous_clause if previous_had_evidence else None,
            )
            if intent in {"recommend_supported", "verified_limitation"}:
                observations.append(
                    ClauseObservation(
                        blocked=False,
                        intent=intent,
                        source="regex_only",
                        heuristic_rescue=intent in JUDGEMENT_PASS_LABELS,
                        considered=True,
                        **common_kwargs,
                    )
                )
                previous_had_evidence = evidence_present
                previous_clause = clause
                continue
            decision = ClauseDecision(
                clause=clause,
                blocked=True,
                intent=intent,
                source="regex_only",
            )
            decisions.append(decision)
            observations.append(
                ClauseObservation(
                    blocked=True,
                    intent=intent,
                    source="regex_only",
                    considered=True,
                    **common_kwargs,
                )
            )
            previous_had_evidence = evidence_present
            previous_clause = clause
            continue
        prediction = classifier.predict(clause)
        if prediction.blocked:
            heuristic_intent = regex_only_intent(
                clause,
                previous_clause=previous_clause if previous_had_evidence else None,
            )
            if heuristic_intent in BLOCK_LABELS:
                prediction.intent = heuristic_intent
            decisions.append(prediction)
        observations.append(
            ClauseObservation(
                blocked=prediction.blocked,
                intent=prediction.intent,
                source=prediction.source,
                p_block=prediction.p_block,
                considered=True,
                **common_kwargs,
            )
        )
        previous_had_evidence = evidence_present
        previous_clause = clause
    return decisions, observations


def evaluate_text(text: str, classifier: Optional[OnnxIntentClassifier] = None) -> List[ClauseDecision]:
    decisions, _ = evaluate_text_with_trace(text, classifier)
    return decisions


def read_jsonl(path):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def find_last_user_text_index(entries):
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if entry.get("type") != "user":
            continue
        message = entry.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return index
        if isinstance(content, list):
            has_text = any(
                block.get("type") == "text"
                for block in content
                if isinstance(block, dict)
            )
            has_only_tool_results = all(
                block.get("type") == "tool_result"
                for block in content
                if isinstance(block, dict)
            )
            if has_text and not has_only_tool_results:
                return index
    return 0


def extract_assistant_texts(entries, from_index):
    texts = []
    for entry in entries[from_index:]:
        if entry.get("type") != "assistant":
            continue
        message = entry.get("message", {})
        content = message.get("content", [])
        if isinstance(content, str):
            if content:
                texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
    return texts


def format_reason(decisions):
    parts = ["Your response contains unchecked or unsupported claims:\n"]
    for index, decision in enumerate(decisions, start=1):
        truncated = decision.clause[:200] + "..." if len(decision.clause) > 200 else decision.clause
        parts.append(f'  {index}. "{truncated}"')
        parts.append(f"     Intent: {decision.intent}")
        parts.append(f"     Source: {decision.source}")
        if decision.p_block is not None:
            parts.append(f"     p_block: {decision.p_block:.3f}")
        if decision.matched_terms:
            parts.append(f"     Matched: {', '.join(decision.matched_terms)}")
        parts.append("")
    parts.append(
        "Verify each flagged clause with concrete evidence from the repo, logs,\n"
        "tests, or commands. Keep only conclusions or recommendations that you\n"
        "can support. If the evidence is insufficient, say exactly what you\n"
        "checked and what remains unknown."
    )
    return "\n".join(parts)


def log_event(event):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(event, default=str) + "\n")
    except OSError:
        pass


def candidate_reason_for(observation: ClauseObservation, threshold: Optional[float]) -> Optional[str]:
    if observation.blocked:
        return "blocked_clause"
    if observation.heuristic_rescue:
        return "heuristic_rescue"
    if observation.mixed_evidence_gap:
        return "mixed_evidence_gap"
    if threshold is not None and observation.p_block is not None and abs(observation.p_block - threshold) <= LOW_MARGIN:
        return "low_margin_pass"

    sanitized_text = sanitize_learning_text(observation.clause)
    sanitized_previous = sanitize_learning_text(observation.previous_clause or "")
    sanitized_next = sanitize_learning_text(observation.next_clause or "")
    if deterministic_sample(candidate_hash_for(sanitized_text, sanitized_previous, sanitized_next), 0.02):
        return "sampled_pass"
    return None


def capture_learning_candidates(
    observations: Sequence[ClauseObservation],
    threshold: Optional[float],
    session_id: str,
    cwd: str,
):
    if CAPTURE_MODE == "off":
        return
    candidate_rows = []
    for observation in observations:
        reason = candidate_reason_for(observation, threshold)
        if reason is None:
            continue
        row = serialize_observation(observation, threshold, session_id, cwd)
        row["candidate_reason"] = reason
        candidate_rows.append(row)
    if append_learning_candidates(candidate_rows):
        maybe_trigger_learning_cycle()


def hook_main():
    if DISABLE_HOOK:
        sys.exit(0)
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "")
    stop_hook_active = input_data.get("stop_hook_active", False)
    cwd = input_data.get("cwd", "")

    if not transcript_path or not os.path.exists(transcript_path):
        sys.exit(0)

    try:
        entries = read_jsonl(transcript_path)
        if not entries:
            sys.exit(0)

        turn_start = find_last_user_text_index(entries)
        assistant_texts = extract_assistant_texts(entries, turn_start)
        if not assistant_texts:
            sys.exit(0)

        full_text = "\n".join(assistant_texts)
        classifier, status = get_classifier()
        decisions, observations = evaluate_text_with_trace(full_text, classifier if status["ml_available"] else None)
        stage = "onnx" if status["ml_available"] else "regex_only"
        backend_name = "onnx" if status["ml_available"] else "regex"
        threshold = classifier.meta.get("threshold") if classifier is not None else None

        log_event(
            {
                "session": session_id,
                "project": os.path.basename(cwd) if cwd else "",
                "retry": stop_hook_active,
                "text_blocks": len(assistant_texts),
                "text_chars": len(full_text),
                "result": "block" if decisions else "pass",
                "stage": stage,
                "backend": backend_name,
                "ml_available": status["ml_available"],
                "fallback_reason": status["fallback_reason"],
                "error_stage": status["error_stage"],
                "error_detail": status["error_detail"],
                "flag_count": len(decisions),
                "flags": [
                    {
                        "clause": decision.clause[:200],
                        "intent": decision.intent,
                        "source": decision.source,
                        "p_block": decision.p_block,
                        "matched_terms": decision.matched_terms or [],
                    }
                    for decision in decisions
                ],
                "predicted_intent": [decision.intent for decision in decisions],
                "p_block": [decision.p_block for decision in decisions if decision.p_block is not None],
                "threshold": threshold,
            }
        )
        capture_learning_candidates(observations, threshold, session_id, cwd)

        if not decisions:
            sys.exit(0)

        print(json.dumps({"decision": "block", "reason": format_reason(decisions)}))
        sys.exit(0)

    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        log_event(
            {
                "session": session_id,
                "project": os.path.basename(cwd) if cwd else "",
                "result": "error",
                "stage": "hook_error",
                "backend": "regex",
                "ml_available": None,
                "fallback_reason": None,
                "error_stage": "hook_runtime",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
        )
        sys.exit(0)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_unique_jsonl(path: Path, rows: Iterable[dict], key_field: str) -> int:
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


def read_state_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_state_json(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def parse_iso_ts(value: Optional[str]):
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


def processed_hashes(reviewed_rows: Sequence[dict]):
    hashes = set()
    for row in reviewed_rows:
        if row.get("candidate_hash"):
            hashes.add(row["candidate_hash"])
        elif row.get("candidate_id"):
            hashes.add(row["candidate_id"])
    return hashes


def pending_learning_rows(queue_rows: Sequence[dict], reviewed_rows: Sequence[dict]):
    seen = processed_hashes(reviewed_rows)
    return [
        row
        for row in queue_rows
        if (row.get("candidate_hash") or row.get("candidate_id")) not in seen
    ]


def oldest_pending_age_seconds(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    now = datetime.now(timezone.utc)
    ages = []
    for row in rows:
        ts = parse_iso_ts(row.get("ts"))
        if ts is None:
            continue
        ages.append(max(0.0, (now - ts).total_seconds()))
    return max(ages) if ages else 0.0


def trigger_reason_for_rows(
    rows: Sequence[dict],
    pending_threshold: int,
    same_reason_threshold: int,
    same_family_threshold: int,
    oldest_age_seconds: int,
):
    if not rows:
        return None, {
            "pending_rows": 0,
            "largest_reason_cluster": 0,
            "largest_family_cluster": 0,
            "oldest_pending_age_seconds": 0,
        }
    reason_counts = Counter(row.get("candidate_reason", "unknown") for row in rows)
    family_counts = Counter(row.get("pattern_family") or row.get("intent") or "unknown" for row in rows)
    oldest_age = oldest_pending_age_seconds(rows)

    metrics = {
        "pending_rows": len(rows),
        "largest_reason_cluster": max(reason_counts.values(), default=0),
        "largest_family_cluster": max(family_counts.values(), default=0),
        "oldest_pending_age_seconds": oldest_age,
    }
    if len(rows) >= pending_threshold:
        return "pending_rows", metrics
    if reason_counts:
        top_reason, top_count = reason_counts.most_common(1)[0]
        if top_count >= same_reason_threshold:
            return f"candidate_reason:{top_reason}", metrics
    if family_counts:
        top_family, top_count = family_counts.most_common(1)[0]
        if top_count >= same_family_threshold:
            return f"pattern_family:{top_family}", metrics
    if oldest_age >= oldest_age_seconds:
        return "oldest_pending_age", metrics
    return None, metrics


def cooldown_remaining_seconds(state: dict, cooldown_seconds: int):
    last_cycle = parse_iso_ts(state.get("last_cycle_ts"))
    if last_cycle is None:
        return 0
    elapsed = (datetime.now(timezone.utc) - last_cycle).total_seconds()
    return max(0, int(cooldown_seconds - elapsed))


def run_self_json_command(mode: str, *extra_args: str, env_overrides: Optional[dict] = None):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        self_command(mode, *extra_args),
        capture_output=True,
        text=True,
        cwd=str(WORKSPACE_ROOT),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"{mode} failed")
    return json.loads(result.stdout.strip() or "{}")


def review_row_from_queue(row: dict) -> dict:
    text = row.get("sanitized_text") or row.get("text") or ""
    intent = row.get("intent") or "reference_language"
    return {
        "id": row.get("candidate_id") or f"queue-{stable_sha1(text)[:16]}",
        "text": text,
        "block": row.get("decision") == "block" or intent in BLOCK_LABELS,
        "intent": intent,
        "source_type": "queue",
        "source_hash": row.get("candidate_hash") or stable_sha1(f"queue:{text}")[:16],
        "evidence_present": bool(row.get("evidence_present")),
        "quoted_or_code": bool(row.get("quoted_or_code")),
        "review_status": "needs_review",
        "notes": f"queue:{row.get('candidate_reason', 'unknown')}",
        "candidate_reason": row.get("candidate_reason", "unknown"),
        "candidate_ids": [row.get("candidate_id")] if row.get("candidate_id") else [],
        "pattern_family": row.get("intent") or "unknown",
    }


def review_row_from_mined(row: dict) -> dict:
    text = row["text"]
    return {
        "id": row.get("id", f"mined-{stable_sha1(f'mined:{text}')[:16]}"),
        "text": text,
        "block": bool(row.get("block", row["intent"] in BLOCK_LABELS)),
        "intent": row["intent"],
        "source_type": row.get("source_type", "transcript"),
        "source_hash": row.get("source_hash", stable_sha1(f"mined:{text}")[:16]),
        "evidence_present": bool(row.get("evidence_present")),
        "quoted_or_code": bool(row.get("quoted_or_code")),
        "review_status": "needs_review",
        "notes": row.get("notes", "mined"),
        "candidate_reason": row.get("candidate_reason", "mined_transcript"),
        "candidate_ids": row.get("candidate_ids", []),
        "pattern_family": row.get("pattern_family", row["intent"]),
    }


def review_rows_from_log(rows: Iterable[dict], min_cluster_size: int) -> List[dict]:
    clauses = []
    for row in rows:
        for flag in row.get("flags", []):
            clause = flag.get("clause")
            if clause:
                clauses.append(sanitize_learning_text(clause))
    counts = Counter(clauses)
    review_rows = []
    for clause, count in counts.items():
        if count < min_cluster_size:
            continue
        review_rows.append(
            {
                "id": f"log-{stable_sha1(f'log:{clause}')[:16]}",
                "text": clause,
                "block": True,
                "intent": "assert_unverified",
                "source_type": "log_cluster",
                "source_hash": stable_sha1(f"log:{clause}")[:16],
                "evidence_present": bool(EVIDENCE_PATTERN.search(clause)),
                "quoted_or_code": bool(quoted_or_code_clause(clause)),
                "review_status": "needs_review",
                "notes": f"log-cluster:{count}",
                "candidate_reason": "log_cluster",
                "candidate_ids": [],
                "pattern_family": "log_cluster",
            }
        )
    return review_rows


def dedupe_review_rows(rows: List[dict], limit: int) -> List[dict]:
    deduped = {}
    for row in rows:
        key = row["text"].strip().lower()
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        merged = dict(existing)
        merged["candidate_ids"] = sorted(
            {*(existing.get("candidate_ids", []) or []), *(row.get("candidate_ids", []) or [])}
        )
        note_parts = []
        for note in (existing.get("notes", ""), row.get("notes", "")):
            if note and note not in note_parts:
                note_parts.append(note)
        merged["notes"] = "; ".join(note_parts)
        deduped[key] = merged
    return list(deduped.values())[:limit]


def build_review_batch_rows(
    queue_paths: Sequence[Path],
    mined_paths: Sequence[Path],
    log_paths: Sequence[Path],
    *,
    limit: int = 200,
    min_cluster_size: int = 3,
):
    rows: List[dict] = []
    for queue_path in queue_paths:
        rows.extend(review_row_from_queue(row) for row in read_jsonl(queue_path))
    for mined_path in mined_paths:
        rows.extend(review_row_from_mined(row) for row in read_jsonl(mined_path))
    for log_path in log_paths:
        rows.extend(review_rows_from_log(read_jsonl(log_path), min_cluster_size))
    return dedupe_review_rows(rows, limit)


REQUIRED_REVIEW_KEYS = {
    "candidate_id",
    "final_intent",
    "final_block",
    "confidence",
    "rationale",
    "pattern_family",
}
REQUIRED_CONSOLIDATION_KEYS = {
    "candidate_id",
    "route",
    "final_intent",
    "final_block",
    "confidence",
    "pattern_family",
    "resolution_reason",
}
CONSOLIDATION_ROUTES = {"staged_training", "staged_regression", "staged_replay", "quarantine"}


def build_review_prompt(rows: Sequence[dict]) -> str:
    return (
        "You are a labeling worker for Assumption Guard.\n"
        "You are not solving the user's problem. You are not giving advice. "
        "You are not rewriting the response. Your only job is to classify each candidate clause.\n"
        "Return ONLY valid JSON.\n"
        "Return one output object per input object.\n"
        "Preserve candidate_id exactly and keep the same order as input.\n"
        "Do not invent new labels.\n"
        "Do not add keys.\n"
        "Do not return markdown or prose.\n"
        "If a case is ambiguous, choose the closest label and lower confidence instead of discussing alternatives.\n"
        "Rationale must be one short sentence grounded in the clause text.\n"
        "Allowed intent labels:\n"
        + "\n".join(f"- {label}" for label in LABELS)
        + "\n"
        "Decision rules:\n"
        "- verification_narration: the speaker says they are about to check or verify something now\n"
        "- capability_promise_unverified: the speaker claims they can perform an action or determine something without evidence\n"
        "- dependency_gap_grounded: the speaker names something already confirmed, then states a concrete missing dependency that blocks the next action\n"
        "- recommend_supported: the recommendation already cites concrete evidence\n"
        "- recommend_unverified: the recommendation is unsupported\n"
        "- unchecked_limitation: the speaker says they do not know or have not checked yet\n"
        "- verified_limitation: the speaker says what they checked and then gives a bounded non-conclusion\n"
        "- reference_language: the clause talks about the hook, labels, examples, or policy rather than making the claim itself\n"
        "Failure conditions:\n"
        "- any text outside valid JSON\n"
        "- any missing candidate_id\n"
        "- any unknown label\n"
        "- any changed order\n"
        "Examples:\n"
        '- "Let me verify whether I can actually identify the new videos and remove them." -> verification_narration -> final_block=false\n'
        '- "But if you want to revert, I can check which ones are new and remove them." -> capability_promise_unverified -> final_block=true\n'
        '- "I confirmed DELETE /videos/bulk exists and requires ids. But without a saved list of the newly created ids, I can\'t selectively remove them." -> dependency_gap_grounded -> final_block=false\n'
        "Output schema:\n"
        '[{"candidate_id":"...","final_intent":"...","final_block":true,"confidence":0.93,"rationale":"...","pattern_family":"..."}]\n'
        "Cases:\n"
        + json.dumps(list(rows), ensure_ascii=True, indent=2)
    )


def parse_review_output(raw_output: str, input_rows: Sequence[dict]):
    parsed = json.loads(raw_output)
    if not isinstance(parsed, list):
        raise ValueError("review output must be a JSON array")
    if len(parsed) != len(input_rows):
        raise ValueError("review output length must match input length")
    expected_ids = [row.get("id") or row.get("candidate_id") for row in input_rows]
    actual_ids = []
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("review row must be an object")
        missing = REQUIRED_REVIEW_KEYS.difference(row)
        if missing:
            raise ValueError(f"missing review keys: {sorted(missing)}")
        extras = set(row).difference(REQUIRED_REVIEW_KEYS)
        if extras:
            raise ValueError(f"unexpected review keys: {sorted(extras)}")
        if row["final_intent"] not in LABELS:
            raise ValueError(f"unknown final_intent: {row['final_intent']}")
        if not isinstance(row["final_block"], bool):
            raise ValueError("final_block must be a boolean")
        if row["final_block"] != (row["final_intent"] in BLOCK_LABELS):
            raise ValueError("final_block must match the policy implied by final_intent")
        if not isinstance(row["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0.0 <= float(row["confidence"]) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("rationale must be a non-empty string")
        if len(row["rationale"].split()) > 25:
            raise ValueError("rationale must stay short")
        actual_ids.append(row["candidate_id"])
    if actual_ids != expected_ids:
        raise ValueError("candidate_id order must match the input order exactly")
    return parsed


def build_consolidator_prompt(disputed_rows: Sequence[dict]) -> str:
    return (
        "You are a consolidation worker for Assumption Guard.\n"
        "You are resolving disputed reviewer outputs.\n"
        "Return ONLY valid JSON and keep the same order as input.\n"
        "Allowed routes: staged_training, staged_regression, staged_replay, quarantine.\n"
        "Do not invent labels. Do not add keys.\n"
        "Only choose staged_training when the row is safe enough to enter staged training data.\n"
        "Use staged_regression or staged_replay for strategically valuable rows that should become fixtures first.\n"
        "Output schema:\n"
        '[{"candidate_id":"...","route":"staged_training","final_intent":"...","final_block":false,"confidence":0.83,"pattern_family":"...","resolution_reason":"..."}]\n'
        "Cases:\n"
        + json.dumps(list(disputed_rows), ensure_ascii=True, indent=2)
    )


def parse_consolidator_output(raw_output: str, disputed_rows: Sequence[dict]):
    parsed = json.loads(raw_output)
    if not isinstance(parsed, list):
        raise ValueError("consolidator output must be a JSON array")
    if len(parsed) != len(disputed_rows):
        raise ValueError("consolidator output length must match input length")
    expected_ids = [row["candidate_id"] for row in disputed_rows]
    actual_ids = []
    for row in parsed:
        if not isinstance(row, dict):
            raise ValueError("consolidator row must be an object")
        missing = REQUIRED_CONSOLIDATION_KEYS.difference(row)
        if missing:
            raise ValueError(f"missing consolidation keys: {sorted(missing)}")
        extras = set(row).difference(REQUIRED_CONSOLIDATION_KEYS)
        if extras:
            raise ValueError(f"unexpected consolidation keys: {sorted(extras)}")
        if row["route"] not in CONSOLIDATION_ROUTES:
            raise ValueError(f"unknown consolidation route: {row['route']}")
        if row["final_intent"] not in LABELS:
            raise ValueError(f"unknown final_intent: {row['final_intent']}")
        if not isinstance(row["final_block"], bool):
            raise ValueError("final_block must be a boolean")
        if row["final_block"] != (row["final_intent"] in BLOCK_LABELS):
            raise ValueError("final_block must match the policy implied by final_intent")
        if not isinstance(row["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0.0 <= float(row["confidence"]) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(row["resolution_reason"], str) or not row["resolution_reason"].strip():
            raise ValueError("resolution_reason must be a non-empty string")
        actual_ids.append(row["candidate_id"])
    if actual_ids != expected_ids:
        raise ValueError("candidate_id order must match the input order exactly")
    return parsed


def consolidate_disputed_rows(
    disputed_rows: Sequence[dict],
    *,
    claude_cmd: str,
    quarantine_dir: Path,
    output_path: Optional[Path] = None,
):
    if not disputed_rows:
        if output_path is not None:
            write_jsonl(output_path, [])
        return []
    prompt = build_consolidator_prompt(disputed_rows)
    last_error = None
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            raw_output = run_review_with_claude(claude_cmd, prompt)
            resolved = parse_consolidator_output(raw_output, disputed_rows)
            if output_path is not None:
                write_jsonl(output_path, resolved)
            return resolved
        except Exception as exc:  # pragma: no cover - exercised in manual runs
            last_error = exc
    quarantine_path = quarantine_dir / f"consolidation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.failed.jsonl"
    write_jsonl(quarantine_path, disputed_rows)
    raise RuntimeError(json.dumps({"error": str(last_error), "quarantined": str(quarantine_path)}))


def run_review_with_claude(claude_cmd: str, prompt: str):
    command = shlex.split(claude_cmd) + ["-p", prompt]
    env = os.environ.copy()
    env["ASSUMPTION_GUARD_DISABLE"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Claude exited with {result.returncode}")
    return result.stdout.strip()


def review_rows_with_claude(
    input_rows: Sequence[dict],
    *,
    claude_cmd: str,
    quarantine_dir: Path,
    output_path: Optional[Path] = None,
):
    prompt = build_review_prompt(input_rows)
    last_error = None
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            raw_output = run_review_with_claude(claude_cmd, prompt)
            reviewed = parse_review_output(raw_output, input_rows)
            if output_path is not None:
                write_jsonl(output_path, reviewed)
            return reviewed
        except Exception as exc:  # pragma: no cover - exercised in manual runs
            last_error = exc
    quarantine_path = quarantine_dir / f"review-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.failed.jsonl"
    write_jsonl(quarantine_path, input_rows)
    raise RuntimeError(json.dumps({"error": str(last_error), "quarantined": str(quarantine_path)}))


def review_rows_with_reviewer(
    input_rows: Sequence[dict],
    *,
    claude_cmd: str,
    quarantine_dir: Path,
    reviewer_id: str,
    output_path: Path,
):
    reviewed = review_rows_with_claude(
        input_rows,
        claude_cmd=claude_cmd,
        quarantine_dir=quarantine_dir,
    )
    stamped = [{**row, "reviewer_id": reviewer_id} for row in reviewed]
    write_jsonl(output_path, stamped)
    return stamped


def training_row_from_review(row: dict) -> dict:
    text = row["text"]
    final_intent = row["final_intent"]
    candidate_id = row.get("candidate_id") or stable_sha1(text)[:16]
    return {
        "id": f"reviewed-{candidate_id}",
        "text": text,
        "block": bool(row["final_block"]),
        "intent": final_intent,
        "source_type": "claude_review",
        "source_hash": stable_sha1(f"review:{candidate_id}:{text}")[:16],
        "evidence_present": bool(EVIDENCE_PATTERN.search(text)),
        "quoted_or_code": bool(quoted_or_code_clause(text)),
        "review_status": "approved",
        "notes": f"confidence={row.get('confidence', 0)} pattern={row.get('pattern_family', 'unknown')}",
    }


def regression_row_from_review(row: dict) -> dict:
    return {
        "group": row["final_intent"],
        "text": row["text"],
        "expected_block": bool(row["final_block"]),
        "reason": f"promoted from claude review ({row.get('candidate_reason', 'unknown')})",
    }


def replay_row_from_review(row: dict) -> dict:
    return {
        "group": row["final_intent"],
        "text": row["text"],
        "expected_block": bool(row["final_block"]),
        "reason": f"pattern family {row.get('pattern_family', row['final_intent'])}",
    }


def with_case_hash(rows: Iterable[dict]) -> List[dict]:
    return [{**row, "case_hash": stable_sha1(row["text"])[:16]} for row in rows]


def promote_reviewed_rows_to_staged(
    reviewed_rows: Sequence[dict],
    *,
    staged_training_overlay: Path,
    staged_regression_overlay: Path,
    staged_replay_overlay: Path,
    accepted_replay_overlay: Optional[Path] = None,
):
    family_counts = Counter(row.get("pattern_family", row.get("final_intent", "unknown")) for row in reviewed_rows)
    training_rows = []
    regression_rows = []
    replay_rows = []
    replay_sources = [staged_replay_overlay]
    if accepted_replay_overlay is not None:
        replay_sources.append(accepted_replay_overlay)
    existing_replay_families = {
        row.get("group")
        for path in replay_sources
        for row in read_jsonl(path)
    }

    for row in reviewed_rows:
        candidate_reason = row.get("candidate_reason", "")
        confidence = float(row.get("confidence", 0.0))
        pattern_family = row.get("pattern_family", row.get("final_intent", "unknown"))
        promotion_route = row.get("promotion_route", "staged_training")
        if promotion_route == "staged_training":
            training_rows.append(training_row_from_review(row))
        if row.get("disagrees_with_runtime") or confidence < 0.80 or candidate_reason in {"heuristic_rescue", "mixed_evidence_gap"}:
            regression_rows.append(regression_row_from_review(row))
        if promotion_route == "staged_regression":
            regression_rows.append(regression_row_from_review(row))
        if pattern_family not in existing_replay_families or family_counts[pattern_family] >= 3:
            replay_rows.append(replay_row_from_review(row))
        if promotion_route == "staged_replay":
            replay_rows.append(replay_row_from_review(row))

    training_count = append_unique_jsonl(staged_training_overlay, training_rows, "id")
    regression_count = append_unique_jsonl(
        staged_regression_overlay,
        with_case_hash(regression_rows),
        "case_hash",
    )
    replay_count = append_unique_jsonl(
        staged_replay_overlay,
        with_case_hash(replay_rows),
        "case_hash",
    )
    return {
        "training_promoted": training_count,
        "regression_promoted": regression_count,
        "replay_promoted": replay_count,
    }


def advance_staged_rows_to_accepted(
    *,
    staged_training_overlay: Path,
    staged_regression_overlay: Path,
    staged_replay_overlay: Path,
    accepted_training_overlay: Path,
    accepted_regression_overlay: Path,
    accepted_replay_overlay: Path,
):
    training_advanced = append_unique_jsonl(
        accepted_training_overlay,
        read_jsonl(staged_training_overlay),
        "id",
    )
    regression_advanced = append_unique_jsonl(
        accepted_regression_overlay,
        read_jsonl(staged_regression_overlay),
        "case_hash",
    )
    replay_advanced = append_unique_jsonl(
        accepted_replay_overlay,
        read_jsonl(staged_replay_overlay),
        "case_hash",
    )

    for path in (
        staged_training_overlay,
        staged_regression_overlay,
        staged_replay_overlay,
    ):
        if path.exists():
            path.unlink()

    return {
        "training_advanced": training_advanced,
        "regression_advanced": regression_advanced,
        "replay_advanced": replay_advanced,
    }


def reduce_review_council(review_rows: Sequence[dict]) -> dict:
    if len(review_rows) < 3:
        raise ValueError("review council requires three reviewer rows")
    candidate_id = review_rows[0].get("candidate_id")
    if any(row.get("candidate_id") != candidate_id for row in review_rows):
        raise ValueError("review council rows must belong to the same candidate")

    block_values = {bool(row["final_block"]) for row in review_rows}
    label_counts = Counter(row["final_intent"] for row in review_rows)
    family_counts = Counter(row.get("pattern_family", row["final_intent"]) for row in review_rows)
    confidences = [float(row["confidence"]) for row in review_rows]
    top_label, top_label_count = label_counts.most_common(1)[0]
    top_family, _ = family_counts.most_common(1)[0]
    unanimous_label = len(label_counts) == 1
    unanimous_block = len(block_values) == 1
    median_confidence = statistics.median(confidences)

    if not unanimous_block:
        route = "quarantine"
    elif unanimous_label and median_confidence >= 0.85:
        route = "staged_training"
    elif top_label_count >= 2 and median_confidence >= 0.75:
        route = "staged_training"
    else:
        route = "needs_consolidation"

    return {
        "candidate_id": candidate_id,
        "route": route,
        "final_intent": top_label,
        "final_block": bool(review_rows[0]["final_block"]) if unanimous_block else None,
        "confidence": median_confidence,
        "pattern_family": top_family,
        "reviewer_ids": [row.get("reviewer_id") for row in review_rows],
    }


def review_batch_health_metrics(consensus_rows: Sequence[dict]) -> dict:
    total = len(consensus_rows)
    if total == 0:
        return {
            "total_rows": 0,
            "largest_family_share": 0.0,
            "disagreement_rate": 0.0,
            "duplicate_rate": 0.0,
            "consolidator_usage_rate": 0.0,
        }
    family_counts = Counter(row.get("pattern_family", "unknown") for row in consensus_rows)
    candidate_ids = [row.get("candidate_id") for row in consensus_rows]
    duplicate_rate = 1.0 - (len(set(candidate_ids)) / total if total else 1.0)
    disagreement_rate = sum(
        1 for row in consensus_rows if row.get("route") in {"quarantine", "needs_consolidation"}
    ) / total
    consolidator_usage_rate = sum(
        1 for row in consensus_rows if row.get("route") == "needs_consolidation"
    ) / total
    return {
        "total_rows": total,
        "largest_family_share": max(family_counts.values(), default=0) / total,
        "disagreement_rate": disagreement_rate,
        "duplicate_rate": duplicate_rate,
        "consolidator_usage_rate": consolidator_usage_rate,
    }


def review_batch_is_healthy(metrics: dict) -> bool:
    total_rows = int(metrics.get("total_rows", 0))
    if total_rows >= 5 and float(metrics.get("largest_family_share", 0.0)) > 0.80:
        return False
    if float(metrics.get("disagreement_rate", 0.0)) > 0.50:
        return False
    if float(metrics.get("duplicate_rate", 0.0)) > 0.10:
        return False
    if float(metrics.get("consolidator_usage_rate", 0.0)) > 0.50:
        return False
    return True


def review_rows_with_council(
    input_rows: Sequence[dict],
    *,
    claude_cmd: str,
    quarantine_dir: Path,
    output_dir: Path,
):
    reviewer_outputs = {}
    for reviewer_id in ("a", "b", "c"):
        reviewer_outputs[reviewer_id] = review_rows_with_reviewer(
            input_rows,
            claude_cmd=claude_cmd,
            quarantine_dir=quarantine_dir,
            reviewer_id=reviewer_id,
            output_path=output_dir / f"reviewed-{reviewer_id}.jsonl",
        )

    consensus_rows = []
    merged_rows = []
    disputed_rows = []
    input_by_id = {(row.get("id") or row.get("candidate_id")): row for row in input_rows}
    for candidate_id in [row.get("id") or row.get("candidate_id") for row in input_rows]:
        candidate_reviews = [
            row
            for rows in reviewer_outputs.values()
            for row in rows
            if row.get("candidate_id") == candidate_id
        ]
        consensus = reduce_review_council(candidate_reviews)
        consensus_rows.append(consensus)
        if consensus["route"] == "needs_consolidation":
            disputed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": input_by_id.get(candidate_id, {}),
                    "reviews": candidate_reviews,
                    "consensus": consensus,
                }
            )
            continue
        if consensus["route"] != "staged_training" or consensus["final_block"] is None:
            continue
        source = input_by_id.get(candidate_id, {})
        merged_rows.append(
            {
                **source,
                "candidate_id": candidate_id,
                "final_intent": consensus["final_intent"],
                "final_block": consensus["final_block"],
                "confidence": consensus["confidence"],
                "pattern_family": consensus["pattern_family"],
                "promotion_route": consensus["route"],
                "reviewer_ids": consensus["reviewer_ids"],
            }
        )

    write_jsonl(output_dir / "review-consensus.jsonl", consensus_rows)
    consolidated_path = output_dir / "review-consolidated.jsonl"
    consolidated_rows = consolidate_disputed_rows(
        disputed_rows,
        claude_cmd=claude_cmd,
        quarantine_dir=quarantine_dir,
        output_path=consolidated_path,
    )
    if not consolidated_path.exists():
        write_jsonl(consolidated_path, consolidated_rows)
    for row in consolidated_rows:
        if row["route"] == "quarantine":
            continue
        source = input_by_id.get(row["candidate_id"], {})
        merged_rows.append(
            {
                **source,
                "candidate_id": row["candidate_id"],
                "final_intent": row["final_intent"],
                "final_block": row["final_block"],
                "confidence": row["confidence"],
                "pattern_family": row["pattern_family"],
                "promotion_route": row["route"],
                "resolution_reason": row["resolution_reason"],
            }
        )
    return {
        "consensus_rows": consensus_rows,
        "merged_rows": merged_rows,
        "consolidated_rows": consolidated_rows,
    }


def promote_reviewed_rows(
    reviewed_rows: Sequence[dict],
    *,
    training_overlay: Path,
    regression_overlay: Path,
    replay_overlay: Path,
):
    return promote_reviewed_rows_to_staged(
        reviewed_rows,
        staged_training_overlay=training_overlay,
        staged_regression_overlay=regression_overlay,
        staged_replay_overlay=replay_overlay,
    )


def should_retrain(promote_summary: dict, reviewed_rows: Sequence[dict]):
    if promote_summary.get("regression_promoted", 0) > 0:
        return True
    if promote_summary.get("training_promoted", 0) >= 20:
        return True
    family_counts = Counter(row.get("pattern_family", "unknown") for row in reviewed_rows)
    return any(count >= 5 for count in family_counts.values())


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


def weak_label_for_clause(clause: str):
    hard_pass = hard_pass_intent(clause)
    if hard_pass:
        if hard_pass == "code_content":
            if not (
                INLINE_CODE.search(clause)
                and (should_consider_clause(clause) or META_REPORT_PATTERN.search(clause.lower()) or TYPE_WORDS.search(clause))
            ):
                return None, False
        return hard_pass, hard_pass not in BLOCK_LABELS
    hard_block = hard_block_decision(clause)
    if hard_block:
        return hard_block.intent, True
    if CAPABILITY_PROMISE_PATTERN.search(clause):
        return "capability_promise_unverified", True
    if should_consider_clause(clause):
        return regex_only_intent(clause), True
    return None, False


def source_hash_for(path: Path, text: str) -> str:
    return stable_sha1(f"{path}:{text}")[:16]


def mine_rows(root: Path, limit: int):
    seen = set()
    rows = []
    for transcript in root.rglob("*.jsonl"):
        for entry in read_jsonl(transcript):
            for block in extract_assistant_text(entry):
                for clause in split_text_to_clauses(block):
                    sanitized = sanitize_learning_text(clause)
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
                            "block": intent in BLOCK_LABELS,
                            "intent": intent,
                            "source_type": "transcript",
                            "source_hash": source_hash_for(transcript, sanitized),
                            "evidence_present": bool(EVIDENCE_PATTERN.search(sanitized)),
                            "quoted_or_code": bool(INLINE_CODE.search(sanitized) or sanitized.startswith("```")),
                            "review_status": "needs_review",
                            "notes": transcript.name,
                            "pattern_family": intent,
                        }
                    )
                    if len(rows) >= limit:
                        return rows
    return rows


def parse_mine_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description="Mine sanitized assistant clauses from Claude transcript JSONL files.")
    parser.add_argument("--root", default=TRANSCRIPT_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--include-safe", action="store_true")
    return parser.parse_args(list(argv))


def command_mine(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_mine_args(argv)
    rows = mine_rows(Path(args.root), args.limit)
    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    payload = {"rows": len(rows), "output": str(output_path)}
    if emit_output:
        print(json.dumps(payload))
    return payload


def parse_review_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description="Review sanitized learning candidates with a headless Claude CLI batch.")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-dir", required=True)
    parser.add_argument("--claude-cmd", default=REVIEW_CLAUDE_CMD)
    parser.add_argument("--max-candidates", type=int, default=100)
    return parser.parse_args(list(argv))


def command_review(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_review_args(argv)
    batch_path = Path(args.batch)
    rows = read_jsonl(batch_path)[: args.max_candidates]
    reviewed = review_rows_with_claude(
        rows,
        claude_cmd=args.claude_cmd,
        quarantine_dir=Path(args.quarantine_dir),
        output_path=Path(args.output),
    )
    payload = {"reviewed": len(reviewed), "output": args.output}
    if emit_output:
        print(json.dumps(payload))
    return payload


def parse_promote_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description="Promote Claude-reviewed rows into local overlay corpora.")
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--training-overlay", required=True)
    parser.add_argument("--regression-overlay", required=True)
    parser.add_argument("--replay-overlay", required=True)
    return parser.parse_args(list(argv))


def command_promote(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_promote_args(argv)
    payload = promote_reviewed_rows(
        read_jsonl(Path(args.reviewed)),
        training_overlay=Path(args.training_overlay),
        regression_overlay=Path(args.regression_overlay),
        replay_overlay=Path(args.replay_overlay),
    )
    if emit_output:
        print(json.dumps(payload))
    return payload


def parse_maybe_trigger_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description="Launch the learning cycle only when objective pending-queue thresholds are met.")
    parser.add_argument("--state-dir", default=str(STATE_PATH))
    parser.add_argument("--queue-path", default=QUEUE_PATH)
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--reviewed-path", default=str(STATE_PATH / "reviewed-claude.jsonl"))
    parser.add_argument("--review-state-path", default=str(STATE_PATH / "review-state.json"))
    parser.add_argument("--training-python", default=TRAINING_PYTHON)
    parser.add_argument("--claude-cmd", default=REVIEW_CLAUDE_CMD)
    parser.add_argument("--transcript-root", default=TRANSCRIPT_ROOT)
    parser.add_argument("--pending-threshold", type=int, default=25)
    parser.add_argument("--same-reason-threshold", type=int, default=5)
    parser.add_argument("--same-family-threshold", type=int, default=4)
    parser.add_argument("--oldest-age-seconds", type=int, default=12 * 60 * 60)
    parser.add_argument("--cooldown-seconds", type=int, default=2 * 60 * 60)
    parser.add_argument("--foreground", action="store_true")
    return parser.parse_args(list(argv))


def command_maybe_trigger(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_maybe_trigger_args(argv)
    state_dir = Path(args.state_dir)
    queue_path = Path(args.queue_path)
    reviewed_path = Path(args.reviewed_path)
    review_state_path = Path(args.review_state_path)
    learning_lock_path = state_dir / "learning-cycle.lock"
    trigger_lock_path = state_dir / "trigger-check.lock"
    now = datetime.now(timezone.utc).isoformat()

    try:
        with file_lock(trigger_lock_path):
            queue_rows = read_jsonl(queue_path)
            reviewed_rows = read_jsonl(reviewed_path)
            pending = pending_learning_rows(queue_rows, reviewed_rows)
            state = read_state_json(review_state_path)

            if learning_lock_path.exists():
                state.update(
                    {
                        "last_check_ts": now,
                        "pending_rows": len(pending),
                        "last_decision": "skipped",
                        "last_skip_reason": "learning_cycle_running",
                    }
                )
                write_state_json(review_state_path, state)
                payload = {"status": "skipped", "reason": "learning_cycle_running", "pending_rows": len(pending)}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            reason, metrics = trigger_reason_for_rows(
                pending,
                args.pending_threshold,
                args.same_reason_threshold,
                args.same_family_threshold,
                args.oldest_age_seconds,
            )
            remaining = cooldown_remaining_seconds(state, args.cooldown_seconds)
            if not reason:
                state.update(
                    {
                        "last_check_ts": now,
                        **metrics,
                        "last_decision": "skipped",
                        "last_skip_reason": "threshold_not_met",
                    }
                )
                write_state_json(review_state_path, state)
                payload = {"status": "skipped", "reason": "threshold_not_met", **metrics}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            if remaining > 0:
                state.update(
                    {
                        "last_check_ts": now,
                        **metrics,
                        "last_decision": "skipped",
                        "last_skip_reason": "cooldown_active",
                        "cooldown_remaining_seconds": remaining,
                    }
                )
                write_state_json(review_state_path, state)
                payload = {"status": "skipped", "reason": "cooldown_active", "cooldown_remaining_seconds": remaining, **metrics}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            command = self_command(
                "learning-cycle",
                "--state-dir",
                str(state_dir),
                "--queue-path",
                str(queue_path),
                "--log-path",
                str(Path(args.log_path)),
                "--transcript-root",
                args.transcript_root,
                "--claude-cmd",
                args.claude_cmd,
                "--training-python",
                args.training_python,
            )
            if args.foreground:
                result = subprocess.run(command, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT), check=False)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "learning cycle failed")
                launch_details = {"mode": "foreground", "stdout": result.stdout.strip()}
            else:
                logs_dir = state_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                launched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                output_log = logs_dir / f"learning-cycle-{launched_at}.log"
                with output_log.open("a", encoding="utf-8") as handle:
                    subprocess.Popen(
                        command,
                        cwd=str(WORKSPACE_ROOT),
                        stdout=handle,
                        stderr=handle,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        close_fds=True,
                    )
                launch_details = {"mode": "detached", "log_path": str(output_log)}

            state.update(
                {
                    "last_check_ts": now,
                    "last_cycle_ts": now,
                    "last_trigger_reason": reason,
                    "last_decision": "launched",
                    **metrics,
                }
            )
            write_state_json(review_state_path, state)
            payload = {"status": "launched", "trigger_reason": reason, **metrics, **launch_details}
            if emit_output:
                print(json.dumps(payload))
            return payload
    except FileExistsError:
        payload = {"status": "skipped", "reason": "trigger_check_running"}
        if emit_output:
            print(json.dumps(payload))
        return payload


def parse_learning_cycle_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description="Run one iterative Assumption Guard learning cycle.")
    parser.add_argument("--state-dir", default=str(STATE_PATH))
    parser.add_argument("--queue-path", default=QUEUE_PATH)
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--transcript-root", default=TRANSCRIPT_ROOT)
    parser.add_argument("--min-review-batch", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--training-python", default=TRAINING_PYTHON)
    parser.add_argument("--claude-cmd", default=REVIEW_CLAUDE_CMD)
    return parser.parse_args(list(argv))


def command_learning_cycle(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_learning_cycle_args(argv)
    state_dir = Path(args.state_dir)
    queue_path = Path(args.queue_path)
    log_path = Path(args.log_path)
    lock_path = state_dir / "learning-cycle.lock"
    mined_path = state_dir / "mined-review-input.jsonl"
    review_batch_path = state_dir / "review-batch.jsonl"
    quarantine_dir = state_dir / "quarantine"
    staged_training_overlay = state_dir / "staged-training-overlay.jsonl"
    staged_regression_overlay = state_dir / "staged-regression-overlay.jsonl"
    staged_replay_overlay = state_dir / "staged-replay-overlay.jsonl"
    accepted_training_overlay = state_dir / "accepted-training-overlay.jsonl"
    accepted_regression_overlay = state_dir / "accepted-regression-overlay.jsonl"
    accepted_replay_overlay = state_dir / "accepted-replay-overlay.jsonl"
    reviewed_merged_path = state_dir / "reviewed-claude.jsonl"
    promotion_manifest_path = state_dir / "promotion-manifest.jsonl"
    current_report_path = state_dir / "current-model-report.json"
    candidates_root = state_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    try:
        with file_lock(lock_path):
            mined_rows = mine_rows(Path(args.transcript_root), limit=max(args.max_candidates * 6, 600))
            write_jsonl(mined_path, mined_rows)

            review_rows = build_review_batch_rows(
                [queue_path],
                [mined_path],
                [log_path],
                limit=args.max_candidates,
            )
            write_jsonl(review_batch_path, review_rows)
            reason_counts = Counter(row.get("candidate_reason", "unknown") for row in review_rows)
            if len(review_rows) < args.min_review_batch and max(reason_counts.values(), default=0) < 3:
                payload = {"status": "skipped", "reason": "insufficient_review_batch", "rows": len(review_rows)}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            council_result = review_rows_with_council(
                review_rows,
                claude_cmd=args.claude_cmd,
                quarantine_dir=quarantine_dir,
                output_dir=state_dir,
            )
            merged_rows = council_result["merged_rows"]
            append_unique_jsonl(reviewed_merged_path, merged_rows, "candidate_id")

            promote_summary = promote_reviewed_rows_to_staged(
                merged_rows,
                staged_training_overlay=staged_training_overlay,
                staged_regression_overlay=staged_regression_overlay,
                staged_replay_overlay=staged_replay_overlay,
                accepted_replay_overlay=accepted_replay_overlay,
            )
            health = review_batch_health_metrics(council_result["consensus_rows"])
            if not review_batch_is_healthy(health):
                payload = {
                    "status": "captured",
                    "promoted": promote_summary,
                    "retrain": False,
                    "reason": "unhealthy_review_batch",
                    "health": health,
                }
                if emit_output:
                    print(json.dumps(payload))
                return payload
            if not should_retrain(promote_summary, merged_rows):
                payload = {"status": "captured", "promoted": promote_summary, "retrain": False}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            candidate_dir = candidates_root / timestamp
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model_path = candidate_dir / "assumption-guard-v2.onnx"
            tokenizer_path = candidate_dir / "assumption-guard-v2-tokenizer.json"
            meta_path = candidate_dir / "assumption-guard-v2-meta.json"
            report_path = candidate_dir / "assumption-guard-v2-report.json"

            run_self_json_command(
                "train",
                "--overlay-training-data",
                str(accepted_training_overlay),
                "--overlay-training-data",
                str(staged_training_overlay),
                "--overlay-regression-cases",
                str(accepted_regression_overlay),
                "--overlay-regression-cases",
                str(staged_regression_overlay),
                "--overlay-replay-cases",
                str(accepted_replay_overlay),
                "--overlay-replay-cases",
                str(staged_replay_overlay),
                "--output-model",
                str(model_path),
                "--output-tokenizer",
                str(tokenizer_path),
                "--output-meta",
                str(meta_path),
                "--output-report",
                str(report_path),
                env_overrides={"ASSUMPTION_GUARD_DISABLE": "1"},
            )

            candidate_report = json.loads(report_path.read_text())
            fallback_report_path = current_report_path if current_report_path.exists() else DEFAULT_REPORT
            current_report = json.loads(Path(fallback_report_path).read_text())
            family_metrics = targeted_family_metrics(candidate_report)
            accepted = (
                candidate_report["regression_results"]["matched"] == candidate_report["regression_results"]["total"]
                and candidate_report["replay_results"]["block_recall"] >= 0.95
                and candidate_report["replay_results"]["pass_recall"] >= 0.95
                and all(value >= 0.95 for value in family_metrics.values())
                and better_than_current(candidate_report, current_report)
            )
            if not accepted:
                payload = {"status": "trained", "promoted": False, "candidate_dir": str(candidate_dir)}
                if emit_output:
                    print(json.dumps(payload))
                return payload

            live_model = PACKAGE_DIR / "assumption-guard-v2.onnx"
            live_tokenizer = PACKAGE_DIR / "assumption-guard-v2-tokenizer.json"
            live_meta = PACKAGE_DIR / "assumption-guard-v2-meta.json"
            os.replace(model_path, live_model)
            os.replace(tokenizer_path, live_tokenizer)
            os.replace(meta_path, live_meta)
            advance_summary = advance_staged_rows_to_accepted(
                staged_training_overlay=staged_training_overlay,
                staged_regression_overlay=staged_regression_overlay,
                staged_replay_overlay=staged_replay_overlay,
                accepted_training_overlay=accepted_training_overlay,
                accepted_regression_overlay=accepted_regression_overlay,
                accepted_replay_overlay=accepted_replay_overlay,
            )
            append_unique_jsonl(
                promotion_manifest_path,
                [
                    {
                        "manifest_id": stable_sha1(f"{timestamp}:{candidate_dir}")[:16],
                        "candidate_dir": str(candidate_dir),
                        "selected_candidate": candidate_report.get("selected_candidate"),
                        "promoted_at": datetime.now(timezone.utc).isoformat(),
                        "reviewed_candidate_ids": [row.get("candidate_id") for row in merged_rows],
                        **advance_summary,
                    }
                ],
                "manifest_id",
            )
            current_report_path.write_text(json.dumps(candidate_report, indent=2))
            payload = {
                "status": "promoted",
                "candidate_dir": str(candidate_dir),
                "family_metrics": family_metrics,
                "advanced": advance_summary,
            }
            if emit_output:
                print(json.dumps(payload))
            return payload
    except FileExistsError:
        payload = {"status": "skipped", "reason": "lock_exists", "lock_path": str(lock_path)}
        if emit_output:
            print(json.dumps(payload))
        return payload


DEFAULT_MODEL = Path(MODEL_PATH)
DEFAULT_TOKENIZER = Path(TOKENIZER_PATH)
DEFAULT_META = Path(META_PATH)

ort = None
QuantType = None
quantize_dynamic = None
CalibratedClassifierCV = None
HashingVectorizer = None
TfidfTransformer = None
LogisticRegression = None
Pipeline = None
LinearSVC = None
torch = None
DataLoader = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
get_linear_schedule_with_warmup = None

class DatasetBase:
    pass


def ensure_training_dependencies():
    global ort, QuantType, quantize_dynamic
    global CalibratedClassifierCV, HashingVectorizer, TfidfTransformer, LogisticRegression, Pipeline, LinearSVC
    global torch, DataLoader, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    global DatasetBase

    if (
        ort is not None
        and torch is not None
        and CalibratedClassifierCV is not None
        and DataLoader is not None
        and AutoModelForSequenceClassification is not None
    ):
        return

    import onnxruntime as _ort
    from onnxruntime.quantization import QuantType as _QuantType, quantize_dynamic as _quantize_dynamic
    from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV
    from sklearn.feature_extraction.text import HashingVectorizer as _HashingVectorizer, TfidfTransformer as _TfidfTransformer
    from sklearn.linear_model import LogisticRegression as _LogisticRegression
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.svm import LinearSVC as _LinearSVC
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
    from transformers import (
        AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
        AutoTokenizer as _AutoTokenizer,
        get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,
    )

    ort = _ort
    QuantType = _QuantType
    quantize_dynamic = _quantize_dynamic
    CalibratedClassifierCV = _CalibratedClassifierCV
    HashingVectorizer = _HashingVectorizer
    TfidfTransformer = _TfidfTransformer
    LogisticRegression = _LogisticRegression
    Pipeline = _Pipeline
    LinearSVC = _LinearSVC
    torch = _torch
    DataLoader = _DataLoader
    DatasetBase = _Dataset
    AutoModelForSequenceClassification = _AutoModelForSequenceClassification
    AutoTokenizer = _AutoTokenizer
    get_linear_schedule_with_warmup = _get_linear_schedule_with_warmup


LABEL_TO_ID = {label: index for index, label in enumerate(v2.LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass
class SplitData:
    train: List[dict]
    dev: List[dict]
    test: List[dict]


class TextDataset(DatasetBase):
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


def parse_train_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", default=str(DEFAULT_DATASET))
    parser.add_argument("--overlay-training-data", action="append", default=[])
    parser.add_argument("--regression-cases", default=str(DEFAULT_REGRESSION))
    parser.add_argument("--overlay-regression-cases", action="append", default=[])
    parser.add_argument("--replay-cases", default=str(DEFAULT_REPLAY))
    parser.add_argument("--overlay-replay-cases", action="append", default=[])
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
    return parser.parse_args(list(argv))


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
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append(upgrade_row(json.loads(line), index))
    return rows


def load_cases(path: Path) -> List[dict]:
    if not path.exists():
        return []
    raw = path.read_text().strip()
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def merge_rows(*row_sets: Sequence[dict]) -> List[dict]:
    merged = {}
    for rows in row_sets:
        for row in rows:
            key = row["id"]
            merged[key] = row
    return list(merged.values())


def merge_cases(*case_sets: Sequence[dict]) -> List[dict]:
    merged = {}
    for rows in case_sets:
        for row in rows:
            key = stable_hash(f"{row.get('group', '')}:{row['text']}:{row['expected_block']}")
            merged[key] = row
    return list(merged.values())


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

        hard_pass = v2.hard_pass_intent(clause, previous_had_evidence=previous_had_evidence)
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


def baseline_pipeline(min_class_count: int):
    if min_class_count < 2:
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="auto",
            random_state=42,
        )
    else:
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42),
            cv=min(3, min_class_count),
            method="sigmoid",
        )
    return Pipeline(
        [
            ("hash", HashingVectorizer(ngram_range=(1, 3), n_features=2**18, alternate_sign=False)),
            ("tfidf", TfidfTransformer()),
            ("clf", classifier),
        ]
    )


def p_block_from_probabilities(probabilities: Sequence[float]) -> float:
    return sum(probabilities[LABEL_TO_ID[label]] for label in v2.BLOCK_LABELS)


def run_baseline_candidate(rows: SplitData, regression_cases: Sequence[dict], replay_cases: Sequence[dict]):
    train_labels = [row["intent"] for row in rows.train]
    min_class_count = min(
        (train_labels.count(label) for label in set(train_labels)),
        default=0,
    )
    pipe = baseline_pipeline(min_class_count)
    train_texts = [row["text"] for row in rows.train]
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
    targeted_family_accuracy = {}
    for family in ["verification_narration", "dependency_gap_grounded", "capability_promise_unverified"]:
        family_rows = [row for row in replay_rows if row["group"] == family]
        targeted_family_accuracy[family] = (
            sum(1 for row in family_rows if row["matched"]) / len(family_rows)
            if family_rows
            else 1.0
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
            "targeted_family_accuracy": targeted_family_accuracy,
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


def command_train(argv: Sequence[str], *, emit_output: bool = True):
    ensure_training_dependencies()
    args = parse_train_args(argv)
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
    overlay_rows = merge_rows(*(load_rows(Path(path)) for path in args.overlay_training_data))
    rows = merge_rows(rows, overlay_rows)
    splits = split_rows(rows)
    regression_cases = merge_cases(load_cases(regression_path), *(load_cases(Path(path)) for path in args.overlay_regression_cases))
    replay_cases = merge_cases(load_cases(replay_path), *(load_cases(Path(path)) for path in args.overlay_replay_cases))

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
            "overlay_rows": len(overlay_rows),
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
            "targeted_family_accuracy": selected["replay"]["targeted_family_accuracy"],
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
        "verification_narration_ok": report["replay_results"]["targeted_family_accuracy"]["verification_narration"] >= 0.95,
        "dependency_gap_grounded_ok": report["replay_results"]["targeted_family_accuracy"]["dependency_gap_grounded"] >= 0.95,
        "capability_promise_unverified_ok": report["replay_results"]["targeted_family_accuracy"]["capability_promise_unverified"] >= 0.95,
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
    payload = {
        "selected_candidate": selected["name"],
        "threshold": selected["threshold"],
        "acceptance": acceptance,
        "report": str(output_report),
    }
    if emit_output:
        print(json.dumps(payload))
    return report




def parse_replay_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(list(argv))


def command_replay(argv: Sequence[str], *, emit_output: bool = True):
    args = parse_replay_args(argv)
    cases = json.loads(Path(args.regression_cases).read_text())
    classifier = v2.OnnxIntentClassifier(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        meta_path=args.meta,
    ).load()

    rows = []
    matched = 0
    for case in cases:
        decisions = v2.evaluate_text(case["text"], classifier)
        blocked = bool(decisions)
        if blocked == case["expected_block"]:
            matched += 1
        rows.append(
            {
                "text": case["text"],
                "expected_block": case["expected_block"],
                "blocked": blocked,
                "matched": blocked == case["expected_block"],
                "intents": [decision.intent for decision in decisions],
            }
        )

    output = {
        "summary": {
            "matched": matched,
            "total": len(rows),
            "accuracy": matched / len(rows) if rows else 0.0,
        },
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    if emit_output:
        print(json.dumps(output["summary"]))
    return output




COMMANDS = {
    'maybe-trigger': command_maybe_trigger,
    'learning-cycle': command_learning_cycle,
    'review': command_review,
    'promote': command_promote,
    'mine': command_mine,
    'train': command_train,
    'replay': command_replay,
}


def main(argv: Optional[Sequence[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'hook':
        return hook_main()
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    if argv and argv[0] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description='Assumption Guard single-file orchestrator')
        parser.add_argument('mode', nargs='?', choices=['hook', *sorted(COMMANDS)])
        parser.print_help()
        return 0
    return hook_main()


if __name__ == '__main__':
    main()

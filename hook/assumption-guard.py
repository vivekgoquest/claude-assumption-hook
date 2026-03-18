#!/usr/bin/env python3
"""Assumption Guard v2 hook and runtime."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence


LABELS: List[str] = [
    "assert_unverified",
    "recommend_unverified",
    "unchecked_limitation",
    "reference_language",
    "verified_limitation",
    "describe_type",
    "reason_conditionally",
    "code_content",
    "idiomatic_compare",
    "recommend_supported",
]
BLOCK_LABELS = {"assert_unverified", "recommend_unverified", "unchecked_limitation"}

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
ARCHITECTURE_NOUNS = re.compile(
    r"\b(TypeScript|JavaScript|redis|staging|production|"
    r"state machine|queue worker|service|shared module)\b",
    re.IGNORECASE,
)
DIRECTIVE_START = re.compile(r"^\s*(use|prefer|return|reuse|deploy|consider|store)\b", re.IGNORECASE)

CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
MODEL_PATH = os.environ.get(
    "ASSUMPTION_GUARD_MODEL_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard-v2.onnx"),
)
TOKENIZER_PATH = os.environ.get(
    "ASSUMPTION_GUARD_TOKENIZER_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard-v2-tokenizer.json"),
)
META_PATH = os.environ.get(
    "ASSUMPTION_GUARD_META_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard-v2-meta.json"),
)
LOG_PATH = os.environ.get(
    "ASSUMPTION_GUARD_LOG_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard.log.jsonl"),
)

_CLASSIFIER = None
_CLASSIFIER_STATUS = None
_CLASSIFIER_LOAD_ATTEMPTED = False


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


def hard_pass_intent(clause: str) -> Optional[str]:
    stripped = clause.strip()
    lowered = stripped.lower()
    if not stripped:
        return "reference_language"
    if stripped.startswith("```") or COMMENT_LINE.match(stripped):
        return "code_content"
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
    if EVIDENCE_PATTERN.search(stripped) and LIMITATION_PATTERN.search(stripped):
        return "verified_limitation"
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
    return any(pattern.search(cleaned) for pattern in PREFILTER_PATTERNS)


def _contains_risky_marker(text: str) -> bool:
    return should_consider_clause(text) or bool(LIMITATION_PATTERN.search(text))


def regex_only_intent(clause: str, previous_clause: Optional[str] = None) -> str:
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


def evaluate_text(text: str, classifier: Optional[OnnxIntentClassifier] = None) -> List[ClauseDecision]:
    decisions: List[ClauseDecision] = []
    previous_clause = None
    previous_had_evidence = False
    for clause in split_text_to_clauses(text):
        if previous_had_evidence and RECOMMENDATION_HINT.search(clause):
            previous_clause = clause
            previous_had_evidence = False
            continue
        hard_pass = hard_pass_intent(clause)
        if hard_pass:
            previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue
        hard_block = hard_block_decision(clause)
        if hard_block:
            decisions.append(hard_block)
            previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue
        if not should_consider_clause(clause):
            previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
            previous_clause = clause
            continue
        if classifier is None:
            intent = regex_only_intent(
                clause,
                previous_clause=previous_clause if previous_had_evidence else None,
            )
            if intent in {"recommend_supported", "verified_limitation"}:
                previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
                previous_clause = clause
                continue
            decisions.append(
                ClauseDecision(
                    clause=clause,
                    blocked=True,
                    intent=intent,
                    source="regex_only",
                )
            )
            previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
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
        previous_had_evidence = bool(EVIDENCE_PATTERN.search(clause))
        previous_clause = clause
    return decisions


def read_jsonl(path):
    entries = []
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


def main():
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
        decisions = evaluate_text(full_text, classifier if status["ml_available"] else None)
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


if __name__ == "__main__":
    main()

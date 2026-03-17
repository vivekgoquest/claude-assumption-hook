#!/usr/bin/env python3
"""
Assumption Guard Hook for Claude Code.

A Stop hook that detects hedging/assumption language in Claude's responses
and blocks until Claude verifies each claim using its tools.

Two-stage detection: regex pre-filter (fast) → ML classifier (accurate).
The classifier filters out false positives: quotations, idioms, type analysis,
conditionals, code content, and acknowledged limitations.
"""

import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================================
# HEDGING PATTERNS — All 8 categories, \b-bounded, hardcoded
# ============================================================================

PATTERNS = [
    # Category 1: Epistemic modals
    r"\b(likely|unlikely|probably|possibly|perhaps|maybe|conceivably)\b",
    r"\b(might|may|could)\s+(be|have|cause|lead|result|work|mean|indicate)",

    # Category 2: Shields
    r"\b(I think|I believe|I suppose|I imagine|I suspect|I expect)\b",
    r"\b(I would say|I'd guess|I feel like|my understanding is)\b",
    r"\b(it seems|it appears|it looks like|it sounds like)\b",
    r"\b(seemingly|apparently|ostensibly)\b",

    # Category 3: Assumption markers
    r"\b(I assume|I presume|I'm guessing|presumably)\b",
    r"\b(my assumption is|assuming that|I would assume)\b",

    # Category 4: Disclaimers
    r"\b(I'm not sure|I'm not certain|I could be wrong)\b",
    r"\b(don't quote me|I haven't verified|off the top of my head)\b",
    r"\b(I'm fairly confident|I'm reasonably sure|I'm pretty sure)\b",
    r"\b(I'm not entirely sure|I'm somewhat uncertain)\b",

    # Category 5: Attribution hedges
    r"\b(based on my understanding|from what I can tell)\b",
    r"\b(from what I've seen|from what I recall)\b",
    r"\b(as far as I can tell|as far as I'm aware)\b",
    r"\b(if memory serves|if I recall correctly)\b",

    # Category 6: Conditional hedges
    r"\b(if I'm not mistaken|unless I'm wrong)\b",
    r"\b(correct me if I'm wrong)\b",

    # Category 7: Approximators
    r"\b(approximately|roughly|around|about)\s+\d",
    r"\b(somewhat|fairly|rather|relatively|sort of|kind of)\b",

    # Category 8: AI-specific patterns
    r"\b(if I recall correctly|from what I remember)\b",
    r"\b(as of my last update|based on my training)\b",
    r"\b(to the best of my knowledge|to my knowledge)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PATTERNS]

MODEL_PATH = os.path.join(os.path.expanduser("~"), ".claude", "assumption-guard-model.pkl")
LOG_PATH = os.path.join(os.path.expanduser("~"), ".claude", "assumption-guard.log.jsonl")


# ============================================================================
# ML MODEL CLASSES — must be defined here so pickle can deserialize them
# ============================================================================

class IntentFeatures(BaseEstimator, TransformerMixin):
    """Extract structural features that signal speaker intent."""

    HEDGING_WORDS = re.compile(
        r"\b(I think|I believe|I suspect|I assume|I presume|probably|likely|"
        r"unlikely|possibly|perhaps|maybe|conceivably|seemingly|apparently|"
        r"might be|could be|may be|it seems|it appears|it looks like|"
        r"it sounds like|fairly|rather|relatively|somewhat|kind of|sort of|"
        r"about \d|around \d|approximately|roughly)\b", re.IGNORECASE
    )
    META_WORDS = re.compile(
        r"\b(flagged|detected|pattern|regex|hook|blocked|caught|triggered|"
        r"matched|classified|filter|false.positive|category|example)\b", re.IGNORECASE
    )
    TYPE_WORDS = re.compile(
        r"\b(null|None|undefined|nil|string|number|int|float|bool|boolean|"
        r"array|list|dict|map|optional|type|return|throw|resolve|reject|"
        r"yield|emit)\b"
    )
    CONDITIONAL_START = re.compile(
        r"^\s*(if |unless |without |when |removing |with )", re.IGNORECASE
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            f = []
            stripped = text.strip()
            f.append(1.0 if stripped.startswith("|") and "|" in stripped[1:] else 0.0)
            f.append(1.0 if stripped.startswith(("#", "//", "/*", "```")) else 0.0)
            f.append(1.0 if '`' in text and self.HEDGING_WORDS.search(
                re.sub(r'`[^`]*`', '', text)) is None else 0.0)
            f.append(1.0 if self.META_WORDS.search(text) else 0.0)
            f.append(1.0 if self.TYPE_WORDS.search(text) else 0.0)
            f.append(1.0 if self.CONDITIONAL_START.match(stripped) else 0.0)
            in_quotes = 0
            for m in self.HEDGING_WORDS.finditer(text):
                before = text[:m.start()]
                if before.count('"') % 2 or before.count("'") % 2:
                    in_quotes += 1
            f.append(1.0 if in_quotes > 0 else 0.0)
            f.append(min(len(self.HEDGING_WORDS.findall(text)) / 3.0, 1.0))
            f.append(min(len(text) / 200.0, 1.0))
            f.append(1.0 if re.match(
                r"^\s*(I think|I believe|I suspect|I assume|I presume|I expect|"
                r"I feel|I imagine|I suppose|My understanding)",
                text, re.IGNORECASE) else 0.0)
            features.append(f)
        return csr_matrix(np.array(features))


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """TF-IDF + intent-signal features."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3), max_features=5000,
            sublinear_tf=True, strip_accents="unicode",
        )
        self.intent = IntentFeatures()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.intent.fit(X)
        return self

    def transform(self, X):
        return hstack([self.tfidf.transform(X), self.intent.transform(X)])


# Load ML model once at import time
_model = None


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        except Exception:
            pass
    return _model


def read_jsonl(path):
    """Read a JSONL file and return list of parsed JSON objects."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def find_last_user_text_index(entries):
    """Find the index of the last user text message (not tool_result)."""
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("type") != "user":
            continue
        msg = entry.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return i
        if isinstance(content, list):
            has_text = any(
                c.get("type") == "text" for c in content if isinstance(c, dict)
            )
            has_only_tool_results = all(
                c.get("type") == "tool_result" for c in content if isinstance(c, dict)
            )
            if has_text and not has_only_tool_results:
                return i
    return 0


def extract_assistant_texts(entries, from_index):
    """Extract all assistant text content from entries after from_index."""
    texts = []
    for entry in entries[from_index:]:
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        content = msg.get("content", [])
        # Fix #6: handle string-typed assistant content
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


def get_line_containing(text, pos):
    """Get the full line containing the character at position pos."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def scan_for_hedging(text):
    """Scan text for hedging patterns. Returns dict of line -> set of matched words."""
    flagged_lines = {}
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            line = get_line_containing(text, match.start())
            if line:
                flagged_lines.setdefault(line, set()).add(match.group())
    return flagged_lines


def filter_with_model(flagged_lines):
    """Use ML classifier to filter out false positives. Returns only genuine hedging."""
    model = get_model()
    if model is None:
        return flagged_lines

    # Fix #1: wrap predict in try/except — fall back to regex-only if model breaks
    genuine = {}
    for line, words in flagged_lines.items():
        try:
            pred = model.predict([line])[0]
        except Exception:
            return flagged_lines
        if pred == 1:  # 1 = BLOCK (genuine hedging)
            genuine[line] = words
    return genuine


def format_reason(flagged_lines):
    """Format the block reason with flagged lines and instructions."""
    parts = ["Your response contains unverified assumptions:\n"]
    for i, (line, words) in enumerate(flagged_lines.items(), 1):
        truncated = line[:200] + "..." if len(line) > 200 else line
        flagged_words = ", ".join(f'"{w}"' for w in sorted(words))
        parts.append(f'  {i}. "{truncated}"')
        parts.append(f"     Flagged: {flagged_words}\n")
    parts.append(
        "Verify EACH flagged claim using your tools (Read, Grep, Bash).\n"
        "Replace assumption language with verified facts, or state\n"
        "explicitly what you checked and why it cannot be confirmed.\n"
        "Do NOT simply rephrase — actually verify."
    )
    return "\n".join(parts)


def log_event(event):
    """Append a JSONL event to the log file."""
    event["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except OSError:
        pass


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "")
    stop_hook_active = input_data.get("stop_hook_active", False)  # logged but not used for decisions (intentional v1)
    cwd = input_data.get("cwd", "")

    if not transcript_path or not os.path.exists(transcript_path):
        sys.exit(0)

    # Fix #4: top-level exception handling — fail open with log entry
    try:
        entries = read_jsonl(transcript_path)
        if not entries:
            sys.exit(0)

        turn_start = find_last_user_text_index(entries)
        assistant_texts = extract_assistant_texts(entries, turn_start)

        if not assistant_texts:
            sys.exit(0)

        full_text = "\n".join(assistant_texts)

        # Stage 1: Regex pre-filter (fast, free)
        regex_flags = scan_for_hedging(full_text)

        if not regex_flags:
            log_event({
                "session": session_id,
                "project": os.path.basename(cwd) if cwd else "",
                "retry": stop_hook_active,
                "text_blocks": len(assistant_texts),
                "text_chars": len(full_text),
                "result": "pass",
                "stage": "regex",
                "flag_count": 0,
                "flags": [],
            })
            sys.exit(0)

        # Stage 2: ML classifier filters false positives
        confirmed_flags = filter_with_model(regex_flags)

        # Log with both stages visible
        log_event({
            "session": session_id,
            "project": os.path.basename(cwd) if cwd else "",
            "retry": stop_hook_active,
            "text_blocks": len(assistant_texts),
            "text_chars": len(full_text),
            "result": "block" if confirmed_flags else "pass",
            "stage": "model" if get_model() else "regex_only",
            "regex_flags": len(regex_flags),
            "model_flags": len(confirmed_flags),
            "filtered_out": len(regex_flags) - len(confirmed_flags),
            "flag_count": len(confirmed_flags),
            "flags": [
                {"line": line[:200], "words": sorted(words)}
                for line, words in confirmed_flags.items()
            ] if confirmed_flags else [],
            "filtered": [
                {"line": line[:200], "words": sorted(words)}
                for line, words in regex_flags.items()
                if line not in confirmed_flags
            ],
        })

        if not confirmed_flags:
            sys.exit(0)

        reason = format_reason(confirmed_flags)
        print(json.dumps({"decision": "block", "reason": reason}))
        sys.exit(0)

    except Exception as exc:
        # Fail open — log the error but don't block Claude
        log_event({
            "session": session_id,
            "project": os.path.basename(cwd) if cwd else "",
            "result": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })
        sys.exit(0)


if __name__ == "__main__":
    main()

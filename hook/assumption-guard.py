#!/usr/bin/env python3
"""
Assumption Guard Hook for Claude Code.

A Stop hook that detects unverified uncertainty in Claude's responses and
blocks until Claude verifies the statement with its tools.

Two-stage detection: regex pre-filter (fast) -> optional ML classifier
(more precise). The classifier filters out false positives such as
quotations, idioms, type analysis, conditionals, code content, and
verified limitations that already explain what was checked.
"""

import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone


# ============================================================================
# HEDGING / UNVERIFIED LANGUAGE PATTERNS — stdlib only, always available
# ============================================================================

PATTERNS = [
    # Category 1: Epistemic modals
    r"\b(likely|unlikely|probably|possibly|perhaps|maybe|conceivably)\b",
    r"\b(might|may|could)\s+(be|have|cause|lead|result|work|mean|indicate|"
    r"simplify|improve|clarify|help|reduce)\b",

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

    # Category 9: Explicitly unchecked / unverifiable-yet language
    r"\b(have not|haven't|did not|didn't)\s+(checked|verified|confirmed|"
    r"reviewed|read|looked at|inspected|tested|searched)\b",
    r"\b(cannot|can't|could not|couldn't)\s+(confirm|determine|verify)\b",
    r"\b(do not|don't)\s+know\b",
    r"\b(requires?|needs?)\s+(running|checking|verifying|testing)\b",
]

COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS]

CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
MODEL_PATH = os.environ.get(
    "ASSUMPTION_GUARD_MODEL_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard-model.pkl"),
)
LOG_PATH = os.environ.get(
    "ASSUMPTION_GUARD_LOG_PATH",
    os.path.join(CLAUDE_DIR, "assumption-guard.log.jsonl"),
)


# ============================================================================
# OPTIONAL ML RUNTIME — loaded lazily so regex-only mode always works
# ============================================================================

_ML_READY = None
_ML_IMPORT_ERROR = None
_MODEL = None
_MODEL_STATUS = None
_MODEL_LOAD_ATTEMPTED = False

CombinedFeatures = None
IntentFeatures = None
_MODEL_MODULE = None


def initialize_ml(raise_on_error=False):
    """Load optional ML dependencies and define feature classes lazily."""
    global _ML_READY, _ML_IMPORT_ERROR, CombinedFeatures, IntentFeatures, _MODEL_MODULE

    if _ML_READY is not None:
        if not _ML_READY and raise_on_error and _ML_IMPORT_ERROR is not None:
            raise _ML_IMPORT_ERROR
        return _ML_READY

    try:
        hook_dir = os.path.dirname(__file__)
        if hook_dir not in sys.path:
            sys.path.insert(0, hook_dir)
        import assumption_guard_model as model_module  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        _ML_READY = False
        _ML_IMPORT_ERROR = exc
        if raise_on_error:
            raise
        return False

    _MODEL_MODULE = model_module
    CombinedFeatures = model_module.CombinedFeatures
    IntentFeatures = model_module.IntentFeatures
    _ML_READY = True
    return True


def build_pipeline():
    """Create a fresh sklearn pipeline for training or evaluation."""
    initialize_ml(raise_on_error=True)
    return _MODEL_MODULE.build_pipeline()


def model_status(ml_available, fallback_reason=None, error_stage=None, error_detail=None):
    return {
        "ml_available": ml_available,
        "fallback_reason": fallback_reason,
        "error_stage": error_stage,
        "error_detail": error_detail,
    }


def get_model():
    """Load the ML model once and record why fallback happened if it fails."""
    global _MODEL, _MODEL_STATUS, _MODEL_LOAD_ATTEMPTED

    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL, dict(_MODEL_STATUS)

    _MODEL_LOAD_ATTEMPTED = True

    if not initialize_ml():
        _MODEL_STATUS = model_status(
            False,
            fallback_reason="ml_import_error",
            error_stage="ml_import",
            error_detail=f"{type(_ML_IMPORT_ERROR).__name__}: {_ML_IMPORT_ERROR}",
        )
        return None, dict(_MODEL_STATUS)

    if not os.path.exists(MODEL_PATH):
        _MODEL_STATUS = model_status(
            False,
            fallback_reason="model_missing",
            error_stage="model_load",
            error_detail=f"missing model at {MODEL_PATH}",
        )
        return None, dict(_MODEL_STATUS)

    try:
        with open(MODEL_PATH, "rb") as file_obj:
            _MODEL = pickle.load(file_obj)
    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        _MODEL_STATUS = model_status(
            False,
            fallback_reason="pickle_load_error",
            error_stage="model_load",
            error_detail=f"{type(exc).__name__}: {exc}",
        )
        return None, dict(_MODEL_STATUS)

    _MODEL_STATUS = model_status(True)
    return _MODEL, dict(_MODEL_STATUS)


# ============================================================================
# TRANSCRIPT AND DETECTION HELPERS
# ============================================================================

def read_jsonl(path):
    """Read a JSONL file and return parsed JSON objects."""
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
    """Find the last user-authored text message (not tool_result)."""
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
    """Extract all assistant text content from entries after from_index."""
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


def get_line_containing(text, position):
    """Get the full line containing the character at position."""
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def scan_for_hedging(text):
    """Scan text for unverified language. Returns line -> matched words."""
    flagged_lines = {}
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            line = get_line_containing(text, match.start())
            if line:
                flagged_lines.setdefault(line, set()).add(match.group())
    return flagged_lines


def filter_with_model(flagged_lines):
    """Filter false positives with the optional ML classifier."""
    model, status = get_model()
    if model is None:
        return flagged_lines, status

    genuine = {}
    for line, words in flagged_lines.items():
        try:
            prediction = model.predict([line])[0]
        except Exception as exc:  # pragma: no cover - exercised in subprocess tests
            fallback = model_status(
                False,
                fallback_reason="predict_error",
                error_stage="model_predict",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
            return flagged_lines, fallback
        if prediction == 1:
            genuine[line] = words
    return genuine, status


def format_reason(flagged_lines):
    """Format the block reason with flagged lines and next-step guidance."""
    parts = ["Your response contains unverified uncertainty:\n"]
    for index, (line, words) in enumerate(flagged_lines.items(), start=1):
        truncated = line[:200] + "..." if len(line) > 200 else line
        flagged_words = ", ".join(f'"{word}"' for word in sorted(words))
        parts.append(f'  {index}. "{truncated}"')
        parts.append(f"     Flagged: {flagged_words}\n")
    parts.append(
        "Verify EACH flagged statement using your tools (Read, Grep, Bash).\n"
        "Only keep recommendations or conclusions you can support from evidence.\n"
        "If it still cannot be confirmed, state explicitly what you checked and\n"
        "why the available evidence is insufficient.\n"
        "Do NOT simply rephrase — verify or justify."
    )
    return "\n".join(parts)


def log_event(event):
    """Append one JSONL event to the runtime log."""
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
        regex_flags = scan_for_hedging(full_text)

        if not regex_flags:
            log_event(
                {
                    "session": session_id,
                    "project": os.path.basename(cwd) if cwd else "",
                    "retry": stop_hook_active,
                    "text_blocks": len(assistant_texts),
                    "text_chars": len(full_text),
                    "result": "pass",
                    "stage": "regex",
                    "ml_available": None,
                    "fallback_reason": None,
                    "error_stage": None,
                    "flag_count": 0,
                    "flags": [],
                }
            )
            sys.exit(0)

        confirmed_flags, status = filter_with_model(regex_flags)
        stage = "model" if status["ml_available"] else "regex_only"

        log_event(
            {
                "session": session_id,
                "project": os.path.basename(cwd) if cwd else "",
                "retry": stop_hook_active,
                "text_blocks": len(assistant_texts),
                "text_chars": len(full_text),
                "result": "block" if confirmed_flags else "pass",
                "stage": stage,
                "ml_available": status["ml_available"],
                "fallback_reason": status["fallback_reason"],
                "error_stage": status["error_stage"],
                "error_detail": status["error_detail"],
                "regex_flags": len(regex_flags),
                "model_flags": len(confirmed_flags),
                "filtered_out": len(regex_flags) - len(confirmed_flags),
                "flag_count": len(confirmed_flags),
                "flags": [
                    {"line": line[:200], "words": sorted(words)}
                    for line, words in confirmed_flags.items()
                ],
                "filtered": [
                    {"line": line[:200], "words": sorted(words)}
                    for line, words in regex_flags.items()
                    if line not in confirmed_flags
                ],
            }
        )

        if not confirmed_flags:
            sys.exit(0)

        print(json.dumps({"decision": "block", "reason": format_reason(confirmed_flags)}))
        sys.exit(0)

    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        log_event(
            {
                "session": session_id,
                "project": os.path.basename(cwd) if cwd else "",
                "result": "error",
                "stage": "hook_error",
                "ml_available": None,
                "fallback_reason": None,
                "error_stage": "hook_runtime",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

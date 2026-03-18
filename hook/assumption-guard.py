#!/usr/bin/env python3
"""
Assumption Guard Hook for Claude Code.

A Stop hook that detects unverified uncertainty in Claude's responses and
blocks until Claude verifies the statement with its tools.

Default runtime: v2 clause-level detection with stdlib regex and hard rules
in front of an optional ONNX multiclass classifier. Legacy sklearn support is
still available behind ASSUMPTION_GUARD_BACKEND=v1_legacy.
"""

import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone

HOOK_DIR = os.path.dirname(__file__)
if HOOK_DIR not in sys.path:
    sys.path.insert(0, HOOK_DIR)

import assumption_guard_v2 as v2  # pylint: disable=wrong-import-position

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
    r"\b(need|needs)\s+to\s+verify\b",
    r"\b(requires?|needs?)\s+(running|checking|verifying|testing|benchmarking)\b",
    r"\bto\s+know\s+for\s+sure\b",
]

COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS]

CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
BACKEND = os.environ.get("ASSUMPTION_GUARD_BACKEND", "v2")
ONNX_MODEL_PATH = os.environ.get(
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
LEGACY_MODEL_PATH = os.environ.get(
    "ASSUMPTION_GUARD_LEGACY_MODEL_PATH",
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


def build_candidate_pipelines(model_names=None):
    """Create fresh candidate pipelines for offline evaluation."""
    initialize_ml(raise_on_error=True)
    return _MODEL_MODULE.build_candidate_pipelines(model_names=model_names)


def model_descriptions():
    """Return human-readable descriptions for supported offline model variants."""
    initialize_ml(raise_on_error=True)
    return _MODEL_MODULE.model_descriptions()


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

    if not os.path.exists(LEGACY_MODEL_PATH):
        _MODEL_STATUS = model_status(
            False,
            fallback_reason="model_missing",
            error_stage="model_load",
            error_detail=f"missing model at {LEGACY_MODEL_PATH}",
        )
        return None, dict(_MODEL_STATUS)

    try:
        with open(LEGACY_MODEL_PATH, "rb") as file_obj:
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
# OPTIONAL ONNX RUNTIME — loaded lazily so regex-only mode always works
# ============================================================================

_V2_CLASSIFIER = None
_V2_STATUS = None
_V2_LOAD_ATTEMPTED = False


def get_v2_classifier():
    """Load the ONNX classifier once and record explicit fallback reasons."""
    global _V2_CLASSIFIER, _V2_STATUS, _V2_LOAD_ATTEMPTED

    if _V2_LOAD_ATTEMPTED:
        return _V2_CLASSIFIER, dict(_V2_STATUS)

    _V2_LOAD_ATTEMPTED = True

    try:
        classifier = v2.OnnxIntentClassifier(
            model_path=ONNX_MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            meta_path=META_PATH,
        )
        classifier.load()
    except FileNotFoundError as exc:
        detail = str(exc)
        if detail.startswith("meta_missing:"):
            _V2_STATUS = model_status(
                False,
                fallback_reason="meta_missing",
                error_stage="meta_load",
                error_detail=detail,
            )
        elif detail.startswith("tokenizer_missing:"):
            _V2_STATUS = model_status(
                False,
                fallback_reason="tokenizer_missing",
                error_stage="tokenizer_load",
                error_detail=detail,
            )
        else:
            _V2_STATUS = model_status(
                False,
                fallback_reason="model_missing",
                error_stage="model_load",
                error_detail=detail,
            )
        return None, dict(_V2_STATUS)
    except RuntimeError as exc:
        detail = str(exc)
        if detail.startswith("ml_import_error:"):
            _V2_STATUS = model_status(
                False,
                fallback_reason="ml_import_error",
                error_stage="ml_import",
                error_detail=detail,
            )
        else:
            _V2_STATUS = model_status(
                False,
                fallback_reason="asset_load_error",
                error_stage="model_load",
                error_detail=detail,
            )
        return None, dict(_V2_STATUS)
    except json.JSONDecodeError as exc:
        _V2_STATUS = model_status(
            False,
            fallback_reason="asset_load_error",
            error_stage="meta_load",
            error_detail=f"{type(exc).__name__}: {exc}",
        )
        return None, dict(_V2_STATUS)
    except Exception as exc:  # pragma: no cover - exercised in subprocess tests
        error_stage = "model_load"
        detail = f"{type(exc).__name__}: {exc}"
        if "Tokenizer" in detail or "tokenizer" in detail:
            error_stage = "tokenizer_load"
        _V2_STATUS = model_status(
            False,
            fallback_reason="model_load_error",
            error_stage=error_stage,
            error_detail=detail,
        )
        return None, dict(_V2_STATUS)

    _V2_CLASSIFIER = classifier
    _V2_STATUS = model_status(True)
    return _V2_CLASSIFIER, dict(_V2_STATUS)


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
    """Filter false positives with the legacy sklearn classifier."""
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


def format_reason_legacy(flagged_lines):
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


def format_reason_v2(decisions):
    """Format the v2 block reason with clauses, intents, and model scores."""
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


def evaluate_v2_text(full_text):
    classifier, status = get_v2_classifier()
    decisions = v2.evaluate_text(full_text, classifier if status["ml_available"] else None)
    return decisions, status


def evaluate_v1_text(full_text):
    regex_flags = scan_for_hedging(full_text)
    if not regex_flags:
        return {}, model_status(True)
    return filter_with_model(regex_flags)


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
        backend = BACKEND
        if backend == "v1_legacy":
            confirmed_flags, status = evaluate_v1_text(full_text)
            stage = "model" if status["ml_available"] else "regex_only"
            flag_count = len(confirmed_flags)
            log_event(
                {
                    "session": session_id,
                    "project": os.path.basename(cwd) if cwd else "",
                    "retry": stop_hook_active,
                    "text_blocks": len(assistant_texts),
                    "text_chars": len(full_text),
                    "result": "block" if confirmed_flags else "pass",
                    "stage": stage,
                    "backend": "legacy" if status["ml_available"] else "regex",
                    "ml_available": status["ml_available"],
                    "fallback_reason": status["fallback_reason"],
                    "error_stage": status["error_stage"],
                    "error_detail": status["error_detail"],
                    "flag_count": flag_count,
                    "flags": [
                        {"line": line[:200], "words": sorted(words)}
                        for line, words in confirmed_flags.items()
                    ],
                    "predicted_intent": None,
                    "p_block": None,
                    "threshold": None,
                }
            )
            if not confirmed_flags:
                sys.exit(0)
            print(json.dumps({"decision": "block", "reason": format_reason_legacy(confirmed_flags)}))
            sys.exit(0)

        decisions, status = evaluate_v2_text(full_text)
        backend_name = "onnx" if status["ml_available"] else "regex"
        stage = "onnx" if status["ml_available"] else "regex_only"
        threshold = None
        classifier = _V2_CLASSIFIER
        if classifier is not None and classifier._meta is not None:  # pylint: disable=protected-access
            threshold = classifier.meta.get("threshold")

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

        print(json.dumps({"decision": "block", "reason": format_reason_v2(decisions)}))
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

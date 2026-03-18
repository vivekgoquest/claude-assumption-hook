"""Shared sklearn feature definitions for Assumption Guard."""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def trigger_in_quotes(pattern, text):
    count = 0
    for match in pattern.finditer(text):
        before = text[:match.start()]
        if before.count('"') % 2 or before.count("'") % 2:
            count += 1
    return 1.0 if count > 0 else 0.0


class IntentFeatures(BaseEstimator, TransformerMixin):
    """Extract structural features that signal speaker intent."""

    HEDGING_WORDS = re.compile(
        r"\b(I think|I believe|I suspect|I assume|I presume|probably|likely|"
        r"unlikely|possibly|perhaps|maybe|conceivably|seemingly|apparently|"
        r"might be|could be|may be|it seems|it appears|it looks like|"
        r"it sounds like|fairly|rather|relatively|somewhat|kind of|sort of|"
        r"about \d|around \d|approximately|roughly|not sure|not certain|"
        r"cannot confirm|cannot determine|can't confirm|can't determine|"
        r"have not checked|haven't checked|did not check|didn't check|"
        r"need to verify|requires running|if I recall correctly|if memory serves|"
        r"from what I remember|to my knowledge|as far as I can tell|"
        r"as far as I'm aware|off the top of my head|fairly confident|"
        r"reasonably sure|pretty sure)\b",
        re.IGNORECASE,
    )
    EPISTEMIC_START = re.compile(
        r"^\s*(I think|I believe|I suspect|I assume|I presume|I expect|"
        r"I feel|I imagine|I suppose|my understanding|perhaps|maybe|"
        r"probably|likely|from what I can tell|to my knowledge|"
        r"as far as I can tell)",
        re.IGNORECASE,
    )
    META_WORDS = re.compile(
        r"\b(flagged|detected|pattern|regex|hook|blocked|caught|triggered|"
        r"matched|classified|filter|false positive|category|example|"
        r"accuracy|metrics|dataset|taxonomy|regression case|"
        r"acceptance report|evaluation report|metrics report|"
        r"per-intent)\b",
        re.IGNORECASE,
    )
    TYPE_WORDS = re.compile(
        r"\b(null|None|undefined|nil|string|number|int|float|bool|boolean|"
        r"array|list|dict|map|optional|type|return|throw|resolve|reject|"
        r"yield|emit|jsonb|symbol|field|property|callback|payload|handler|"
        r"argument|variant|prototype|frame|bytes|memoryview|PathLike|"
        r"CancelledError|response|token|enum|variable)\b",
        re.IGNORECASE,
    )
    CONDITIONAL_START = re.compile(
        r"^\s*(if |unless |without |when |removing |with )",
        re.IGNORECASE,
    )
    RECOMMENDATION_WORDS = re.compile(
        r"\b(rename|renamed|refactor|extract|inline|split|helper|clearer|readable|"
        r"more readable|simplif(?:y|ied)|improve|unit test|integration test|"
        r"comment|document|cleanup|clean up|remove|removed|early return|guard clause|"
        r"memoiz(?:e|ation)|batch|stream|cache|move this|separate module|"
        r"dataclass|interface|error type|service|queue worker|prefer|use|"
        r"return early|normalize|validate)\b",
        re.IGNORECASE,
    )
    VERIFICATION_EVIDENCE = re.compile(
        r"\b(I (checked|searched|read|reviewed|inspected|looked at|"
        r"looked through|tested|verified|grep(?:ped)?|ran|traced|followed|"
        r"confirmed)\b|after (checking|reviewing|searching|reading|running|"
        r"testing)\b|I grepped\b|I found\b|based on (the|this) (repo|"
        r"codebase|tests|call sites|config|docs|search|trace)\b)",
        re.IGNORECASE,
    )
    UNCHECKED_LIMITATION = re.compile(
        r"\b(have not checked|haven't checked|did not check|didn't check|"
        r"cannot confirm|can't confirm|cannot determine|can't determine|"
        r"need to verify|requires running|requires checking|not sure|"
        r"not certain|without checking|off the top of my head|do not know|"
        r"don't know|without access)\b",
        re.IGNORECASE,
    )
    UNCERTAINTY_START = re.compile(
        r"^\s*(I'm not sure|I am not sure|I'm not certain|I am not certain|"
        r"I'm not entirely sure|I'm somewhat uncertain|off the top of my head)",
        re.IGNORECASE,
    )
    IDIOMATIC_COMPARE = re.compile(r"\b(rather than|instead of| over )\b", re.IGNORECASE)
    DIRECTIVE_START = re.compile(
        r"^\s*(use|prefer|return|call|keep|store|pass|batch|stream|"
        r"normalize|validate|reuse|sort|handle)\b",
        re.IGNORECASE,
    )
    TYPE_ALTERNATIVE = re.compile(r"\b(could|may|might)\b.*\b(or)\b", re.IGNORECASE)
    QUOTED_TEXT = re.compile(r"`[^`]*`|\"[^\"]*\"|'[^']*'")
    HEDGED_RECOMMENDATION = re.compile(
        r"\b(maybe|perhaps|probably|likely|might|could|may|seems like|"
        r"should probably)\b",
        re.IGNORECASE,
    )
    UNCHECKED_WITHOUT = re.compile(r"^\s*without\b", re.IGNORECASE)
    TYPE_SHAPE = re.compile(
        r"\b(field|property|return value|callback|payload|handler|"
        r"argument|response|token|enum|variable|prototype|frame|variant)\b",
        re.IGNORECASE,
    )
    ARCHITECTURE_RECOMMENDATION = re.compile(
        r"\b(TypeScript|database|staging|state machine|queue worker|"
        r"error type|shared module|service)\b",
        re.IGNORECASE,
    )
    NUMERIC_APPROXIMATION = re.compile(
        r"\b(roughly|around|about|approximately)\s+\d",
        re.IGNORECASE,
    )
    DIAGNOSTIC_GUESS = re.compile(
        r"\b(issue|bug|crash|problem|timeout|deadlock|failure|hook)\b",
        re.IGNORECASE,
    )
    QUOTED_EXAMPLE = re.compile(r"^\s*(\d+\.\s*)?[\"'`|]", re.IGNORECASE)
    REFERENCE_META = re.compile(
        r"\b(block|trigger|matched|belongs|taxonomy|regression|"
        r"sample response|detector)\b",
        re.IGNORECASE,
    )
    CODE_REFERENCE = re.compile(
        r"\b(comment|snippet|code|example|contains|literally says)\b",
        re.IGNORECASE,
    )
    PERFORMANCE_GUESS = re.compile(
        r"\b(response time|response takes|latency|throughput|ms)\b",
        re.IGNORECASE,
    )
    STATE_STORAGE_RECOMMENDATION = re.compile(
        r"\b(store|state)\b.*\b(database|memory|redis)\b",
        re.IGNORECASE,
    )
    EVENT_CONDITIONAL = re.compile(
        r"\b(event|callback|handler)\b.*\b(could|may|might)\b.*"
        r"\b(during|when|if|more than once|multiple times)\b",
        re.IGNORECASE,
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            stripped = text.strip()
            features.append(
                [
                    1.0 if stripped.startswith("|") and "|" in stripped[1:] else 0.0,
                    1.0 if stripped.startswith(("#", "//", "/*", "```")) else 0.0,
                    1.0
                    if "`" in text
                    and self.HEDGING_WORDS.search(re.sub(r"`[^`]*`", "", text)) is None
                    else 0.0,
                    1.0 if self.META_WORDS.search(text) else 0.0,
                    1.0 if self.TYPE_WORDS.search(text) else 0.0,
                    1.0 if self.CONDITIONAL_START.match(stripped) else 0.0,
                    trigger_in_quotes(self.HEDGING_WORDS, text),
                    min(len(self.HEDGING_WORDS.findall(text)) / 3.0, 1.0),
                    min(len(text) / 200.0, 1.0),
                    1.0 if self.EPISTEMIC_START.match(stripped) else 0.0,
                    1.0 if self.RECOMMENDATION_WORDS.search(text) else 0.0,
                    1.0 if self.VERIFICATION_EVIDENCE.search(text) else 0.0,
                    1.0 if self.UNCHECKED_LIMITATION.search(text) else 0.0,
                    1.0 if self.UNCERTAINTY_START.match(stripped) else 0.0,
                    1.0 if self.IDIOMATIC_COMPARE.search(text) else 0.0,
                    1.0
                    if self.VERIFICATION_EVIDENCE.search(text)
                    and self.RECOMMENDATION_WORDS.search(text)
                    else 0.0,
                    1.0
                    if self.VERIFICATION_EVIDENCE.search(text)
                    and self.UNCHECKED_LIMITATION.search(text)
                    else 0.0,
                    1.0 if self.TYPE_ALTERNATIVE.search(text) else 0.0,
                    1.0
                    if self.DIRECTIVE_START.match(stripped)
                    and self.IDIOMATIC_COMPARE.search(text)
                    else 0.0,
                    1.0
                    if self.META_WORDS.search(self.QUOTED_TEXT.sub("", text))
                    else 0.0,
                    1.0
                    if self.HEDGED_RECOMMENDATION.search(text)
                    and self.RECOMMENDATION_WORDS.search(text)
                    else 0.0,
                    1.0
                    if self.UNCHECKED_WITHOUT.match(stripped)
                    and self.UNCHECKED_LIMITATION.search(text)
                    and not self.VERIFICATION_EVIDENCE.search(text)
                    else 0.0,
                    1.0
                    if self.TYPE_SHAPE.search(text) and self.TYPE_WORDS.search(text)
                    else 0.0,
                    1.0
                    if self.ARCHITECTURE_RECOMMENDATION.search(text)
                    and self.RECOMMENDATION_WORDS.search(text)
                    and not self.VERIFICATION_EVIDENCE.search(text)
                    else 0.0,
                    1.0 if self.NUMERIC_APPROXIMATION.search(text) else 0.0,
                    1.0
                    if self.DIAGNOSTIC_GUESS.search(text)
                    and self.HEDGING_WORDS.search(text)
                    and not self.META_WORDS.search(text)
                    else 0.0,
                    1.0
                    if self.QUOTED_EXAMPLE.match(stripped)
                    and (trigger_in_quotes(self.HEDGING_WORDS, text) or self.META_WORDS.search(text))
                    else 0.0,
                    1.0
                    if self.QUOTED_EXAMPLE.match(stripped)
                    and self.REFERENCE_META.search(text)
                    else 0.0,
                    1.0
                    if "`" in text and self.CODE_REFERENCE.search(text)
                    else 0.0,
                    1.0
                    if self.PERFORMANCE_GUESS.search(text)
                    and self.HEDGING_WORDS.search(text)
                    and not self.META_WORDS.search(text)
                    else 0.0,
                    1.0
                    if self.STATE_STORAGE_RECOMMENDATION.search(text)
                    and self.IDIOMATIC_COMPARE.search(text)
                    and not self.VERIFICATION_EVIDENCE.search(text)
                    else 0.0,
                    1.0 if self.EVENT_CONDITIONAL.search(text) else 0.0,
                ]
            )
        return csr_matrix(np.array(features))


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """TF-IDF + intent-signal features."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.intent = IntentFeatures()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.intent.fit(X)
        return self

    def transform(self, X):
        return hstack([self.tfidf.transform(X), self.intent.transform(X)])


def build_pipeline():
    return build_pipeline_for("logreg")


MODEL_SPECS = {
    "logreg": {
        "description": (
            "TF-IDF (1-3 grams, 5000 features) + 32 intent features -> "
            "LogisticRegression"
        ),
        "factory": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.25,
            solver="liblinear",
            random_state=42,
        ),
    },
    "linear_svm": {
        "description": (
            "TF-IDF (1-3 grams, 5000 features) + 32 intent features -> "
            "Calibrated LinearSVC"
        ),
        "factory": lambda: CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                random_state=42,
            ),
            cv=3,
            method="sigmoid",
        ),
    },
}


def model_descriptions():
    return {name: spec["description"] for name, spec in MODEL_SPECS.items()}


def build_pipeline_for(model_name):
    if model_name not in MODEL_SPECS:
        raise KeyError(f"unknown model '{model_name}'")
    return Pipeline(
        [
            ("features", CombinedFeatures()),
            ("clf", MODEL_SPECS[model_name]["factory"]()),
        ]
    )


def build_candidate_pipelines(model_names=None):
    names = list(model_names or MODEL_SPECS)
    return {name: build_pipeline_for(name) for name in names}

"""Shared sklearn feature definitions for Assumption Guard."""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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
        r"need to verify|requires running)\b",
        re.IGNORECASE,
    )
    META_WORDS = re.compile(
        r"\b(flagged|detected|pattern|regex|hook|blocked|caught|triggered|"
        r"matched|classified|filter|false positive|category|example)\b",
        re.IGNORECASE,
    )
    TYPE_WORDS = re.compile(
        r"\b(null|None|undefined|nil|string|number|int|float|bool|boolean|"
        r"array|list|dict|map|optional|type|return|throw|resolve|reject|"
        r"yield|emit|jsonb|symbol)\b",
        re.IGNORECASE,
    )
    CONDITIONAL_START = re.compile(
        r"^\s*(if |unless |without |when |removing |with )",
        re.IGNORECASE,
    )
    IMPROVEMENT_WORDS = re.compile(
        r"\b(rename|refactor|extract|helper|clearer|readable|more readable|"
        r"simplif(?:y|ied)|improve|unit test|comment|cleanup|clean up|"
        r"split this function|early return)\b",
        re.IGNORECASE,
    )
    VERIFICATION_EVIDENCE = re.compile(
        r"\b(I (checked|searched|read|reviewed|inspected|looked at|"
        r"looked through|tested|verified|grep(?:ped)?|ran)\b|"
        r"after (checking|reviewing|searching|reading|running)\b|"
        r"I grepped\b)",
        re.IGNORECASE,
    )
    UNCHECKED_LIMITATION = re.compile(
        r"\b(have not checked|haven't checked|did not check|didn't check|"
        r"cannot confirm|can't confirm|cannot determine|can't determine|"
        r"need to verify|requires running|not sure|not certain)\b",
        re.IGNORECASE,
    )
    UNCERTAINTY_START = re.compile(
        r"^\s*(I'm not sure|I am not sure|I'm not certain|I am not certain|"
        r"I'm not entirely sure|I'm somewhat uncertain)",
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
                    1.0
                    if re.match(
                        r"^\s*(I think|I believe|I suspect|I assume|I presume|"
                        r"I expect|I feel|I imagine|I suppose|My understanding)",
                        text,
                        re.IGNORECASE,
                    )
                    else 0.0,
                    1.0 if self.IMPROVEMENT_WORDS.search(text) else 0.0,
                    1.0 if self.VERIFICATION_EVIDENCE.search(text) else 0.0,
                    1.0 if self.UNCHECKED_LIMITATION.search(text) else 0.0,
                    1.0 if self.UNCERTAINTY_START.match(stripped) else 0.0,
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
    return Pipeline(
        [
            ("features", CombinedFeatures()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
        ]
    )

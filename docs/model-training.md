# Assumption Guard — Model Training Documentation

## Overview

The assumption guard ML model is a binary text classifier that determines whether a line of text containing hedging language is **genuine hedging** (BLOCK) or a **false positive** (PASS). It was trained on 2026-03-17 and replaces the original regex-only approach that had a 60% false positive rate.

## Problem Statement

Regex detects hedging words ("I think", "probably", "could be", "rather", "fairly") but cannot distinguish between:

- **"I think the timeout is 30 seconds"** → genuine unverified claim (should BLOCK)
- **"The hook flagged the phrase 'I think'"** → quoting the words (should PASS)
- **"The return value could be None"** → type analysis (should PASS)
- **"Use map rather than forEach"** → English idiom (should PASS)

A model trained on **intent** can make this distinction.

---

## Training Data Construction

### Phase 1: Mining Real Transcripts

We mined all Claude Code session transcripts in `~/.claude/projects/` to find real-world hedging instances:

```
Total transcript files scanned: 60+
Total unique hedging lines found: 1,116
```

**Distribution of real hedging by trigger word:**

| Word | Occurrences |
|------|-------------|
| likely | 298 |
| might be | 128 |
| rather | 95 |
| could be | 90 |
| may have | 75 |
| it seems | 61 |
| may be | 52 |
| probably | 51 |
| maybe | 48 |
| I think | 27 |
| it looks like | 26 |
| might have | 23 |
| kind of | 20 |

This showed the real frequency distribution of hedging in Claude's output — "likely" and "might be" dominate, not "I think" as you might expect.

### Phase 2: Off-the-shelf Model Evaluation

Before training a custom model, we tested two small Ollama models (smollm2:360m and qwen2.5:0.5b) on the classification task. Both failed:

- **smollm2:360m** — output a single line for 8 input sentences. Could not parse the task.
- **qwen2.5:0.5b** — classified all 8 sentences as `UNVERIFIED_CLAIM` and entered an infinite generation loop. Zero discrimination between categories.

Conclusion: off-the-shelf small models cannot handle this classification without fine-tuning.

### Phase 3: TF-IDF + Logistic Regression (v1)

**Initial approach:** 436 examples, TF-IDF features only.

Training data was constructed in two stages:
1. **Synthetic examples** (146) — hand-written sentences for each category
2. **Auto-labeled real examples** (193) — mined from transcripts and labeled with heuristics (trigger word type, presence of meta-words, code comment prefix, etc.)

Categories were surface-level:
- `unverified_claim`, `speculation`, `assumption` (BLOCK)
- `quotation`, `idiom`, `type_analysis`, `conditional`, `acknowledged_limit`, `code_content` (PASS)

**Results:**
- 5-fold CV F1: 0.822 (binary), 0.620 (multi-class)
- 12/12 on hand-picked hard cases
- Model size: 403KB

**Problems discovered:**
1. After adding more quotation examples to fix table-row false positives, the model became too lenient on genuine speculation ("This is probably a race condition" → PASS).
2. Confidence was low (0.52-0.67) on critical cases.
3. The model learned surface features (presence of "flagged" = quotation) rather than understanding intent.

### Phase 4: Intent-Contrastive Training (v2 — final)

**Key insight:** Same hedging words, different intents. The model must learn *why* the word appears, not *that* it appears.

**Training data redesigned from scratch** with 7 intent categories:

#### BLOCK intent: `assert_unverified` (70 examples)

Speaker makes a factual claim without having checked with tools.

```
"I think the timeout is 30 seconds"
"The database probably uses PostgreSQL"
"It seems like the cache is returning stale data"
"I assume the API follows REST conventions"
"The file has about 500 lines"
```

All 70 examples follow the pattern: first-person assertion or speculative statement about a verifiable fact. The speaker has tools available (Read, Grep, Bash) but chose to guess instead.

#### PASS intent: `reference_language` (43 examples)

Speaker discusses, quotes, or mentions hedging words themselves.

```
'The hook flagged the phrase "I think" in the response'
'Flagged: "I think", "probably", "likely"'
'| Probably a race condition | BLOCK | BLOCK | Yes |'
'Genuine hedging ("I think the timeout is probably...") was blocked'
'Test 1: sentence with "probably" triggers BLOCK'
```

This is the category that caused the recursion problem with regex-only detection. The model must understand that quoting a hedging phrase is not hedging.

These examples include:
- Sentences with hedging words in double quotes
- Markdown table rows containing hedging examples
- Hook error messages listing flagged words
- Discussion of model classification results
- Numbered flag lists from hook feedback

#### PASS intent: `compare` (43 examples)

Idiomatic comparisons and degree words used non-hedgingly.

```
"Use a hashmap rather than a list for O(1) lookups"
"The implementation is fairly straightforward"
"This is a relatively small change"
"Check what kind of error the function returns"
"The code is somewhat verbose but readable"
```

"Rather than" is the most common false positive from regex. "Fairly", "relatively", "somewhat" are degree modifiers, not hedging. "Kind of" and "sort of" as type indicators ("what kind of error") are not hedging.

#### PASS intent: `describe_type` (20 examples)

Describing actual code types or state possibilities.

```
"The return value could be None or a string"
"The callback could be synchronous or asynchronous"
"The promise might resolve with an empty array"
"The column could store either TEXT or JSONB"
```

"Could be" and "might" in these sentences describe factual type constraints from the language's type system, not uncertainty about unverified claims.

#### PASS intent: `reason_conditionally` (15 examples)

Logical if-then reasoning.

```
"If the flag is set, validation would be skipped entirely"
"Without proper indexing, queries could become very slow"
"If the migration fails, the transaction would roll back"
```

These use hedging words ("would", "could") in conditional constructions where the speaker is reasoning about hypothetical scenarios, not making uncertain claims.

#### PASS intent: `acknowledge_limit` (12 examples)

Honest admission that verification is impossible.

```
"I cannot determine the exact version without checking package.json"
"Without access to the production logs, I cannot confirm this"
"This requires running the test suite to confirm"
```

These explicitly state what the speaker cannot verify and why — the opposite of unverified claims.

#### PASS intent: `code_content` (12 examples)

Hedging words inside code comments or blocks.

```
"# This might need refactoring in the future"
"// TODO: could be optimized with memoization"
"The comment says `# might be null here`"
```

---

## Feature Engineering

### TF-IDF Features

Standard text vectorization:
- N-gram range: (1, 3) — unigrams, bigrams, and trigrams
- Max features: 5,000
- Sublinear TF scaling: enabled
- Accent stripping: unicode

This captures surface-level word patterns. For example, the trigram "I think the" is a strong BLOCK signal, while "the hook flagged" is a strong PASS signal.

### Intent Features (10 structural signals)

These are the features that make the model work. TF-IDF alone cannot reliably distinguish quotation from assertion because the same words appear in both.

**Feature 1: `is_table_row`** (bool)
- Line starts with `|` and contains another `|`
- Signal: markdown table rows are documentation/examples, not assertions
- Impact: eliminates the main source of recursion false positives

**Feature 2: `is_code_comment`** (bool)
- Line starts with `#`, `//`, `/*`, or triple backtick
- Signal: code content, not prose assertion

**Feature 3: `trigger_only_in_backticks`** (bool)
- After removing all backtick-enclosed spans, no hedging words remain
- Signal: the hedging word is being *mentioned* (in code formatting), not *used*
- Implementation: `re.sub(r'`[^`]*`', '', text)` strips backtick spans, then re-checks for hedging

**Feature 4: `has_meta_words`** (bool)
- Contains: flagged, detected, pattern, regex, hook, blocked, caught, triggered, matched, classified, filter, false positive, category, example
- Signal: the sentence is discussing the hedging detection system itself

**Feature 5: `has_type_keywords`** (bool)
- Contains: null, None, undefined, nil, string, number, int, float, bool, boolean, array, list, dict, map, optional, type, return, throw, resolve, reject, yield, emit
- Signal: the sentence is about code types, not making a factual claim

**Feature 6: `starts_with_conditional`** (bool)
- Starts with: If, Unless, Without, When, Removing, With
- Signal: conditional/hypothetical reasoning, not assertion

**Feature 7: `trigger_in_quotes`** (bool)
- The hedging word appears between unbalanced quote marks (inside a quoted string)
- Implementation: counts `"` characters before the match position; odd count means inside quotes
- Signal: the word is being quoted/referenced

**Feature 8: `hedge_density`** (float, 0.0-1.0)
- Count of hedging words in the line, divided by 3, capped at 1.0
- Signal: lines with multiple hedging words are more likely to be genuine hedging ("I think this is probably a race condition") vs. single-word matches in idiomatic usage ("fairly straightforward")

**Feature 9: `line_length`** (float, 0.0-1.0)
- `len(text) / 200.0`, capped at 1.0
- Signal: longer prose sentences are more likely genuine assertions; short fragments are more likely labels, table cells, or code comments

**Feature 10: `first_person_start`** (bool)
- Line starts with: I think, I believe, I suspect, I assume, I presume, I expect, I feel, I imagine, I suppose, My understanding
- Signal: first-person epistemic assertions are the strongest hedging pattern — this is the speaker directly expressing their uncertain belief

### Feature Combination

TF-IDF produces a sparse matrix of ~5000 features. IntentFeatures produces a dense matrix of 10 features. These are horizontally stacked (scipy `hstack`) into a single feature matrix of ~5010 features.

The LogisticRegression classifier operates on this combined space, learning weights that balance surface-level word patterns (TF-IDF) with structural context (IntentFeatures).

---

## Model Architecture

```python
Pipeline([
    ("features", CombinedFeatures()),    # TF-IDF + IntentFeatures → ~5010 features
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",         # upweights minority class (BLOCK: 70 vs PASS: 145)
        C=1.0,                           # regularization strength
    )),
])
```

**Why LogisticRegression?**
- Fast training (<100ms on 215 examples)
- Fast inference (<1ms per sample)
- Produces calibrated probabilities
- Works well with sparse TF-IDF features
- `class_weight="balanced"` handles the 70/145 class imbalance

**Why not a neural model?**
- 215 training examples is far too small for a neural network
- A transformer-based model (even bert-tiny at 4.4M params) would overfit massively
- LogisticRegression with good features matches or exceeds neural performance at this data scale
- Inference must complete within the 10s hook timeout; LogisticRegression runs in <1ms

---

## Evaluation

### Cross-Validation

5-fold stratified cross-validation:
- **F1 (macro): 0.864 (+/- 0.051)**
- This means the model generalizes well — it's not just memorizing training examples

### Critical Test Cases (16/16 correct)

| Sentence | Expected | Predicted | Confidence |
|----------|----------|-----------|------------|
| I think the timeout is set to 30 seconds | BLOCK | BLOCK | 0.94 |
| This is probably a race condition | BLOCK | BLOCK | 0.59 |
| The database likely uses PostgreSQL | BLOCK | BLOCK | 0.67 |
| I assume the API returns JSON | BLOCK | BLOCK | 0.92 |
| It seems like the cache is stale | BLOCK | BLOCK | 0.68 |
| Maybe the connection pool is exhausted | BLOCK | BLOCK | 0.64 |
| The file has about 500 lines of code | BLOCK | BLOCK | 0.66 |
| I believe the retry count is set to 3 | BLOCK | BLOCK | 0.93 |
| It looks like the TLS certificate expired | BLOCK | BLOCK | 0.73 |
| The socket might be timing out | BLOCK | BLOCK | 0.69 |
| \| Probably a race condition \| BLOCK \| | PASS | PASS | 0.77 |
| \| I think the timeout is 30s \| BLOCK \| | PASS | PASS | 0.76 |
| Flagged: "I think", "probably", "likely" | PASS | PASS | 0.90 |
| Use hashmap rather than list | PASS | PASS | 0.85 |
| The return value could be None or string | PASS | PASS | 0.83 |
| The implementation is fairly straightforward | PASS | PASS | 0.64 |

**Confidence patterns:**
- First-person assertions ("I think", "I believe", "I assume") get the highest BLOCK confidence (0.92-0.94) — Feature 10 (`first_person_start`) drives this
- Meta-discussion gets the highest PASS confidence (0.90) — Feature 4 (`has_meta_words`) drives this
- Degree-word idioms ("fairly straightforward") get lower PASS confidence (0.64) — these are the hardest cases

---

## Iteration History

### Attempt 1: TF-IDF Only (436 examples)
- CV F1: 0.822
- Critical cases: 12/12
- Problem: low confidence (0.52-0.67), 417KB model size

### Attempt 2: Added Quotation Examples
- Added 50+ quotation examples (table rows, hook output, etc.)
- Problem: "This is probably a race condition" flipped to PASS — model became too lenient
- Root cause: quotation examples contain the same words as genuine hedging, so TF-IDF features pulled the decision boundary toward PASS

### Attempt 3: Balanced with More BLOCK Examples
- Added 38 more genuine speculation/hedging examples
- CV F1: 0.807
- Critical cases: 16/16
- Problem: model still classified markdown table rows as BLOCK at low confidence (0.50-0.60) because TF-IDF can't see structural context

### Attempt 4: Intent-Contrastive + Features (Final)
- Rebuilt dataset from scratch: 215 intent-contrastive examples
- Added 10 IntentFeatures for structural context
- CV F1: 0.864
- Critical cases: 16/16 with higher confidence
- Model size: 156KB (down from 417KB)
- Key improvement: Feature 10 (`first_person_start`) gives 0.94 confidence on "I think..." assertions; Feature 4 (`has_meta_words`) gives 0.90 confidence on discussion text

### Critical Bug Fix: Pickle Deserialization
The model was trained with custom classes (`CombinedFeatures`, `IntentFeatures`) defined in a temporary training script. When the hook tried to `pickle.load()` the model, Python couldn't find these classes in the hook's namespace — the load failed silently and the hook fell back to regex-only mode.

**Fix:** Define `CombinedFeatures` and `IntentFeatures` in the hook script itself (`assumption-guard.py`). The pickle deserializer resolves class references against the importing module, so the classes must exist where the model is loaded, not where it was trained.

---

## How to Retrain

### Prerequisites
- Python 3.8+
- scikit-learn, numpy, scipy
- The `CombinedFeatures` and `IntentFeatures` classes (defined in `hook/assumption-guard.py`)

### Steps

1. **Edit the training data** at `training/assumption-guard-training-labeled.jsonl`:
   ```json
   {"sentence": "the new example", "intent": "assert_unverified", "block": true}
   {"sentence": "another example", "intent": "reference_language", "block": false}
   ```

2. **Run the training script:**
   ```python
   import json, pickle, sys
   sys.path.insert(0, "hook")  # or wherever assumption-guard.py lives

   # Import the classes from the hook script
   exec(open("hook/assumption-guard.py").read())

   # Or define CombinedFeatures/IntentFeatures here — they must match
   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import cross_val_score, StratifiedKFold

   data = [json.loads(l) for l in open("training/assumption-guard-training-labeled.jsonl")]
   sentences = [d["sentence"] for d in data]
   labels = [1 if d["block"] else 0 for d in data]

   pipe = Pipeline([
       ("features", CombinedFeatures()),
       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
   ])

   # Evaluate
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   scores = cross_val_score(pipe, sentences, labels, cv=cv, scoring="f1_macro")
   print(f"CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

   # Train on full dataset and save
   pipe.fit(sentences, labels)
   with open("model/assumption-guard-model.pkl", "wb") as f:
       pickle.dump(pipe, f)
   ```

3. **Deploy:**
   ```bash
   cp model/assumption-guard-model.pkl ~/.claude/assumption-guard-model.pkl
   ```
   The hook picks up the new model on next invocation — no restart needed.

### Adding New Training Examples

**From the log** (`~/.claude/assumption-guard.log.jsonl`):
- Check `"filtered"` entries — lines the model let pass. If any are genuine hedging, add them as `block: true`.
- Check `"flags"` entries — lines that triggered a block. If any are false positives, add them as `block: false`.

**Intent labeling guide:**
| If the sentence... | Intent | Block? |
|---------------------|--------|--------|
| Makes a factual claim without verification | `assert_unverified` | true |
| Discusses or quotes hedging words | `reference_language` | false |
| Uses "rather than", "fairly", "relatively" idiomatically | `compare` | false |
| Describes code types ("could be None") | `describe_type` | false |
| Uses if-then reasoning | `reason_conditionally` | false |
| States what cannot be verified and why | `acknowledge_limit` | false |
| Contains hedging in code comments | `code_content` | false |

---

## Files

| File | Location | Purpose |
|------|----------|---------|
| `hook/assumption-guard.py` | Deploy to `~/.claude/hooks/` | Hook script with model classes |
| `model/assumption-guard-model.pkl` | Deploy to `~/.claude/` | Trained sklearn Pipeline |
| `training/assumption-guard-training-labeled.jsonl` | Deploy to `~/.claude/` | 215 labeled examples |
| `hook/settings-snippet.json` | Merge into `~/.claude/settings.json` | Hook registration config |

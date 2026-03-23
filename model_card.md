# Model Card: Mood Machine

This model card covers both versions of the Mood Machine classifier built during this lab:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

Both models were evaluated on the same labeled dataset in `dataset.py`.

---

## 1. Model Overview

**Model type:** Both models were built and compared.

**Intended purpose:**
Classify short, informal text messages (social media posts, chat messages) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`. The input is a single sentence. The output is one label.

**How it works (brief):**

*Rule-based:* Text is preprocessed into tokens. Each token is compared against hand-curated lists of positive and negative words, plus a table of emoji and slang signals. A numeric score is computed — positive words add points, negative words subtract them. Negation (e.g. "not happy") flips the polarity of the following word. If both positive and negative words fired, the label is `mixed`; otherwise the score sign determines the label.

*ML model:* The same labeled posts are converted into bag-of-words vectors using `CountVectorizer` (each unique word becomes a feature column; each post becomes a row of counts). A `LogisticRegression` classifier is trained to map those vectors to labels. No explicit rules are written — the model finds word-label correlations in the data automatically.

---

## 2. Data

**Dataset description:**
The final dataset contains **22 posts** in `SAMPLE_POSTS` with a matching label in `TRUE_LABELS` for each. The dataset was built in three rounds:

- *Round 1 (starter, 6 posts):* Simple, unambiguous examples provided by the lab template.
- *Round 2 (8 posts added):* Posts with slang, emojis, sarcasm, and mixed emotions — designed to stress-test the rule-based model.
- *Round 3 (8 posts added):* Targeted "breaker" sentences designed to expose specific failure modes: sarcasm with a clearly positive word ("I love Mondays"), double negation ("nothing went wrong"), internet slang with no sentiment words ("this slaps no cap"), and understated negativity ("could be worse I guess").

**Labeling process:**
Labels were assigned by human judgment during writing. The four allowed labels are `positive`, `negative`, `neutral`, and `mixed`. Several posts required deliberate decisions:

- `"This is fine"` — labeled `neutral` even though the phrase is common internet sarcasm for "everything is falling apart." A group might reasonably label this `negative`.
- `"not sad just... tired of everything idk"` — labeled `negative` (the overall sentiment is low energy and resigned) even though the negation of "sad" technically introduces a weak positive signal. The rule-based model labeled it `mixed`, which is arguably defensible.
- `"I absolutely love sitting in a 3-hour meeting that could've been an email 💀"` — labeled `negative` (clear sarcasm), but the model predicted `mixed` because `love` and `💀` both fired. The true label reflects human reading of intent; the model's reading reflects the literal token signals present.

**Important characteristics of the dataset:**

- Contains internet slang: "lowkey," "ngl," "no cap," "slaps," "immaculate" (used as hype slang)
- Contains emoji sentiment signals: 😭, 🔥, 💀, 😌, ✨
- Includes sarcasm: "I love Mondays," "cool cool cool," "I absolutely love sitting in a 3-hour meeting"
- Includes mixed emotions: exhaustion + pride, stress + accomplishment, bad food + good atmosphere
- Includes negation: "not happy," "not sad," "nothing went wrong," "not gonna lie I actually had fun"
- All posts are short (1–2 sentences), casual, and written in informal English

**Possible issues with the dataset:**

- **Very small (22 examples):** Not nearly enough to train a generalizable ML model. The ML results are training accuracy only — there is no held-out test set.
- **Label imbalance:** Roughly 9 negative, 7 positive, 3 mixed, 3 neutral. The model has seen very few `neutral` examples.
- **Single annotator:** All labels were assigned by one person. No inter-annotator agreement was measured. Several posts are genuinely ambiguous and different people would label them differently.
- **Narrow dialect:** All posts are in informal American internet English. The slang, emojis, and cultural references reflect a specific online community and age group.

---

## 3. How the Rule-Based Model Works

**Preprocessing (`preprocess` method):**
Text is lowercased and whitespace-trimmed. Known emoji and slang tokens are isolated into their own tokens using word-boundary-aware regex (so "surprisingly" is not split by the embedded string "ngl"). Punctuation is stripped from the edges of word tokens so that "sad..." becomes "sad" and matches the word list. Apostrophes are preserved so "don't" stays intact for negation lookup.

**Scoring (`score_text` method, via `_analyze` helper):**
Tokens are matched against:

1. `POSITIVE_WORDS` — a set of 15 words; each match adds +1 to the score
2. `NEGATIVE_WORDS` — a set of 15 words; each match adds -1 to the score
3. `EMOJI_SLANG_SIGNALS` — a dictionary of ~25 tokens mapped to score deltas (e.g. 😭 → -2, 🔥 → +2, 💀 → -1, ":)" → +1)

Negation: if the token immediately before a sentiment word is in the `NEGATORS` set (`not`, `never`, `no`, `don't`, `doesn't`, `didn't`, `won't`, `can't`, `isn't`, `aren't`, `nothing`, `nobody`, `nowhere`), the polarity of that word is flipped. Example: "not happy" → `happy` is in `POSITIVE_WORDS`, but `tokens[i-1]` is `not` → score -1 instead of +1.

**Label prediction (`predict_label` method):**
If both positive hits and negative hits were found → `"mixed"`.
Otherwise: score > 0 → `"positive"`, score < 0 → `"negative"`, score == 0 → `"neutral"`.

**Strengths:**
- Fully transparent: you can trace exactly which tokens changed the score and why
- Works on any input immediately without training data
- Negation handling correctly flips polarity for simple cases ("not happy," "not bad")
- Emoji signals give coverage for non-word sentiment that word lists miss
- Mixed detection distinguishes "nothing matched" (neutral) from "things cancelled out" (mixed)

**Weaknesses:**
- Cannot detect sarcasm — "I love Mondays" scores +1 because `love` fires with no negative context
- One-token-back negation misses cases where the negator and the target are separated: "nobody here is happy" — `nobody` precedes `here`, not `happy`
- Word list coverage is sparse — words like "proud," "hopeful," "worse," "garbage" had to be added manually after observing failures; any novel word is invisible
- Slang is hard to maintain — "this slaps no cap" is entirely invisible because neither "slaps" nor "cap" carry sentiment in the word lists, and "no" here means emphasis, not negation (but "no" is in `NEGATORS`, so any word following "no" gets its polarity flipped)

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words representation using `CountVectorizer`. Each unique word (token) in the training set becomes a column. Each post becomes a row of integer word counts. Punctuation and case handling is done by scikit-learn's default tokenizer.

**Training data:**
Trained on the same 22 posts and labels in `SAMPLE_POSTS` / `TRUE_LABELS`. No train/test split — the model trains and evaluates on identical data.

**Training behavior:**
After expanding from 14 to 22 posts, the ML model maintained 100% accuracy on training data. This is expected: with a dataset this small and a model this flexible, logistic regression memorizes the training examples rather than learning generalizable patterns.

When 8 new posts were added (Round 3), the ML model correctly predicted all of them — including posts it had never seen during the earlier 14-post training. This is not evidence of generalization; it means the new posts contained words that were already strongly associated with their labels in training. For example, "I love Mondays" was correctly labeled `negative` because the word "Mondays" only appeared in training data with a negative label.

**Strengths:**
- Learns from data without explicit rules — no need to manually maintain word lists
- Handles co-occurrence patterns that rules miss (e.g., "cool cool cool" after a string of bad events correlates with negative)
- Scales more gracefully as data grows

**Weaknesses:**
- 100% training accuracy is not real accuracy — there is no test set, so we have no measure of how the model performs on posts it has never seen
- Highly sensitive to individual label decisions: changing one label in a 22-example dataset can flip the model's weights significantly
- Cannot explain its decisions — there is no equivalent to `explain()` that shows which tokens drove the prediction
- Memorizes rather than generalizes at this scale — "I love Tuesdays" would likely be predicted `positive` even though "I love Mondays" was labeled `negative`, because "Tuesdays" has never appeared in training

---

## 5. Evaluation

**How the models were evaluated:**
Both models were evaluated on the full `SAMPLE_POSTS` dataset (22 posts) against `TRUE_LABELS`. This is training accuracy for the ML model and in-distribution accuracy for the rule-based model. No held-out test set exists.

| Model | Accuracy |
|---|---|
| Rule-based (final) | 73% (16/22) |
| ML (LogisticRegression + CountVectorizer) | 100% (22/22) |

The rule-based model improved from 36% (initial implementation with `pass` bodies) to 64% (after emoji signals and mixed detection) to 73% (after punctuation fix, word list expansion, and `NEGATORS` expansion).

**Examples of correct predictions (rule-based):**

- `"I am not happy about this"` → `negative` ✓
  Tokens: `['i', 'am', 'not', 'happy', 'about', 'this']`. `not` is a negator; `happy` fires as positive but is flipped → score -1 → negative. Negation logic working as intended.

- `"the food was bad but the vibes were immaculate ngl"` → `mixed` ✓
  Tokens include `bad` (negative word, -1) and `immaculate` (slang signal, +2). Both positive and negative hits present → `mixed`. Mixed detection correctly identified the conflicting signals.

- `"😭😭😭 why does everything happen at once"` → `negative` ✓
  Each `😭` scores -2. Total score = -6. Clear negative signal from emoji alone — no word list coverage needed.

**Examples of incorrect predictions (rule-based):**

- `"I love Mondays"` → predicted `positive`, true `negative`
  Tokens: `['i', 'love', 'mondays']`. `love` is in `POSITIVE_WORDS` → +1. No context signals sarcasm. The model has no mechanism to detect that "I love Mondays" is a conventional complaint. This is a fundamental limitation of keyword matching — the word is literally positive.

- `"nothing went wrong today surprisingly"` → predicted `negative`, true `positive`
  Tokens: `['nothing', 'went', 'wrong', 'today', 'surprisingly']`. `nothing` is a negator (index 0). `wrong` is a negative word (index 2). The negation check looks only one token back (`tokens[i-1]`), so `nothing` at index 0 never reaches `wrong` at index 2. `wrong` fires as -1 unchecked → score -1 → negative. The sentence means "everything went fine today" but the model gets -1.

- `"woke up late, missed the bus, spilled my coffee. cool cool cool"` → predicted `neutral`, true `negative`
  Tokens: `['woke', 'up', 'late', 'missed', 'the', 'bus', 'spilled', 'my', 'coffee', 'cool', 'cool', 'cool']`. None of these words are in any sentiment list. The sarcasm ("cool cool cool" = "this is terrible") is completely invisible — the model scores 0 and returns neutral.

**How rule-based and ML failures differed:**
The ML model got all three of the above correct. It learned that "Mondays" co-occurs with negative labels, that "surprisingly" in context is positive, and that the specific phrase "cool cool cool" after a list of mishaps correlates with negative. However, these are memorized associations, not understood patterns. The rule-based model's failures are predictable and explainable; the ML model's (future) failures will be harder to diagnose.

---

## 6. Limitations

**1. Sarcasm is undetectable with keyword matching.**
"I love Mondays" contains `love`, which the model scores +1. There is no signal that the phrase is sarcastic. Fixing this would require either a large external dataset of labeled sarcastic posts, or a model that understands broader context (e.g., a transformer). This was documented as a known limitation and not fixed.

**2. One-token negation window misses non-adjacent negation.**
The negation rule checks only `tokens[i-1]`. "Nobody here is happy" — `nobody` is at position 0, `happy` is at position 3. The negation never lands. Similarly "nothing went wrong today" — `nothing` at 0, `wrong` at 2 — fails for the same reason. Extending the window to two tokens would help ("nobody here") but would also create new false positives.

**3. Word list coverage is manually maintained and always incomplete.**
The model cannot score words it has never seen. Failures on "hopeful," "proud," "worse," "garbage" were only fixed after observing them in evaluation. Any novel word, name, or phrase scores 0. The model is not learning; it is looking up.

**4. Internet slang changes faster than a fixed word list.**
"This slaps no cap" scored `neutral` because "slaps" is not in any list. Slang evolves rapidly. A word list maintained today will have gaps within months.

**5. No real test set exists.**
All evaluation is on training data. The ML model's 100% accuracy is not meaningful without held-out examples. The rule-based model's 73% is more honest — it uses explicit rules rather than fitting to data — but still tells us nothing about performance on new posts.

**6. The dataset is 22 examples written by one person.**
Real-world text classifiers train on tens of thousands of examples from many sources. At 22 examples, both models are highly sensitive to individual labeling decisions. Changing one label would visibly shift the ML model's behavior.

---

## 7. Ethical Considerations

**Scope and dialect bias:**
The dataset was written entirely in informal American internet English by a single author. The slang ("lowkey," "ngl," "no cap," "slaps"), emoji usage, and cultural references reflect a specific online community. The model is optimized for this register and will systematically misclassify posts written in other dialects, languages, or communication styles. A user who writes "I'm gutted" (British English for devastated) or "this is peak" (UK slang for terrible) would get `neutral` predictions because neither phrase appears in any word list.

**Misclassifying distress:**
"Not sad just... tired of everything idk" — the model predicted `mixed` at one point, then `negative` after fixes. A real system used to monitor wellbeing would need to consistently flag this kind of understated negative post. Miscategorizing `negative` as `neutral` or `mixed` could cause harm if the output is used to triage support conversations or mental health check-ins.

**Sarcasm cuts both ways:**
"I absolutely love sitting in a 3-hour meeting" was labeled `negative` (sarcasm) but scored `mixed` or `positive` depending on the version. A moderation or support system using this model would surface this as positive when the person is actually venting frustration. The error is invisible to the system.

**Privacy:**
If this model were deployed on real social media posts or messages, those texts would need to be processed and potentially stored. Even though the model itself is simple, any system that reads personal messages to infer emotional state raises consent and privacy questions that go beyond model accuracy.

**Feedback loops:**
If mood predictions were used to rank, filter, or respond to content (e.g., "show more positive content"), systematic errors — especially around sarcasm or marginalized language — would compound over time. People whose communication style is systematically misread would receive increasingly miscalibrated responses.

---

## 8. Ideas for Improvement

- **Add a real held-out test set:** Split the dataset 80/20 before training, or collect a separate set of labeled posts that the model never sees during development. Current accuracy numbers are not reliable.

- **Expand negation window to two tokens:** Catching "nothing [word] [sentiment]" patterns would fix the "nothing went wrong" class of error with low risk of new false positives.

- **Use TF-IDF instead of CountVectorizer:** Rare words that are strong signals (e.g., "immaculate" used as hype slang) would be weighted higher relative to common filler words ("the," "is," "a"). This would likely improve ML predictions on short posts where one word carries most of the meaning.

- **Collect a second annotator's labels:** Have someone else label the same 22 posts independently. Measure agreement. Posts where annotators disagree are the hardest cases — they should be explicitly documented as ambiguous rather than assigned a single label.

- **Add a confidence score:** Instead of a hard label, the rule-based model could return its score alongside the label (e.g., `score=1` positive vs `score=5` positive). Low-confidence predictions (score near 0, or only one word matched) could be flagged for human review rather than returned as authoritative.

- **Use a small pre-trained model for comparison:** Running the same 22 posts through a publicly available sentiment API (e.g., a HuggingFace pipeline) would show how a model trained on millions of examples handles the same edge cases — sarcasm, slang, negation — and would provide a practical upper bound for what this kind of task looks like at scale.

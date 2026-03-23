# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


# Emoji and slang signals: token → score delta
# These are things the word lists will never cover.
EMOJI_SLANG_SIGNALS: Dict[str, int] = {
    # Positive emojis / slang
    "🔥": 2,
    "😊": 2,
    "😍": 2,
    "🥰": 2,
    "😂": 1,   # laughing — generally positive context
    "✨": 1,
    "🎉": 2,
    "💪": 1,
    "😌": 1,
    ":)": 1,
    ":-)": 1,
    "lowkey": 0,   # intensifier, neutral on its own
    "highkey": 0,
    "ngl": 0,      # "not gonna lie" — no inherent polarity
    "immaculate": 2,
    "insane": 1,   # in hype context usually positive
    # Negative emojis / slang
    "😭": -2,
    "😤": -2,
    "😠": -2,
    "🤮": -2,
    "💀": -1,   # "dead" — often used sarcastically/negatively
    "😢": -2,
    ":(": -1,
    ":-(": -1,
    "ugh": -1,
    "oof": -1,
}

NEGATORS = {
    "not", "never", "no",
    "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't",
    "nothing", "nobody", "nowhere",  # existential negators: "nothing good" = "not good"
}


class MoodAnalyzer:
    """
    A rule based mood classifier with negation handling, emoji signals, and mixed detection.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          1. Lowercase and strip whitespace.
          2. Split off known emoji/slang tokens before stripping punctuation,
             so "😭😭😭" becomes ["😭", "😭", "😭"] rather than being lost.
          3. Strip punctuation from the edges of each word token
             (e.g. "sad..." -> "sad", "coffee." -> "coffee").
             Apostrophes are kept so "don't" stays intact for negation.
        """
        cleaned = text.strip().lower()

        # First pass: isolate emoji and punctuation-only tokens by padding them
        # with spaces so they survive the split as their own tokens.
        #
        # IMPORTANT: use word-boundary matching (\b) for ASCII slang so that
        # "ngl" doesn't split "surprisingly" → ['surprisi', 'ngl', 'y'].
        # Emoji tokens are non-ASCII and don't have \b boundaries, so they
        # use a simple replace (they can't appear inside a word anyway).
        for token in EMOJI_SLANG_SIGNALS:
            if not token.isascii():
                # Emoji: safe to replace literally — can't be embedded in a word
                cleaned = cleaned.replace(token, f" {token} ")
            elif len(token) > 1:
                # ASCII slang ("ngl", "ugh", etc.): only match whole words
                cleaned = re.sub(r"\b" + re.escape(token) + r"\b", f" {token} ", cleaned)

        # Split on whitespace.
        raw_tokens = cleaned.split()

        # Second pass: strip leading/trailing punctuation from word tokens
        # but leave known emoji/slang tokens untouched.
        tokens = []
        for tok in raw_tokens:
            if tok in EMOJI_SLANG_SIGNALS:
                tokens.append(tok)
            else:
                # Strip punctuation from edges; keep internal apostrophes ("don't")
                stripped = re.sub(r"^[^\w']+|[^\w']+$", "", tok)
                if stripped:
                    tokens.append(stripped)

        return tokens

    # ---------------------------------------------------------------------
    # Internal analysis helper
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Tokenize text and return (score, positive_hits, negative_hits).

        This is the single place where the scoring logic lives.
        Both score_text and predict_label call this so they stay in sync.

        Enhancement — negation handling:
          If a negator like "not" or "never" immediately precedes a sentiment
          word, the polarity flips ("not happy" scores -1 instead of +1).
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in NEGATORS

            # Check word lists first
            if token in self.positive_words:
                if negated:
                    score -= 1
                    negative_hits.append(f"not-{token}")
                else:
                    score += 1
                    positive_hits.append(token)

            elif token in self.negative_words:
                if negated:
                    score += 1
                    positive_hits.append(f"not-{token}")
                else:
                    score -= 1
                    negative_hits.append(token)

            # Check emoji/slang signals
            elif token in EMOJI_SLANG_SIGNALS:
                delta = EMOJI_SLANG_SIGNALS[token]
                if negated:
                    delta = -delta
                score += delta
                if delta > 0:
                    positive_hits.append(token)
                elif delta < 0:
                    negative_hits.append(token)

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score for the given text.

        Delegates to _analyze; returns just the integer score.
        Positive words (and positive emoji/slang) increase it.
        Negative words (and negative emoji/slang) decrease it.
        Negation flips the polarity of the following word.
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Map the analysis of text to a mood label.

        Label rules:
          - Both positive and negative hits found  -> "mixed"
          - score > 0, no negative hits            -> "positive"
          - score < 0, no positive hits            -> "negative"
          - score == 0, no hits at all             -> "neutral"
          - score == 0, but hits cancel out        -> "mixed"

        Using both pos/neg hit counts (not just the score) allows us to
        distinguish "neutral" (nothing matched) from "mixed" (things matched
        but cancelled out).
        """
        score, positive_hits, negative_hits = self._analyze(text)

        has_positive = len(positive_hits) > 0
        has_negative = len(negative_hits) > 0

        if has_positive and has_negative:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining why the model chose its label.
        Shows score, which tokens matched, and the final label.
        """
        score, positive_hits, negative_hits = self._analyze(text)
        label = self.predict_label(text)

        return (
            f"Label={label!r}, Score={score} | "
            f"positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or []}"
        )

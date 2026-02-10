"""
Lexical and discourse-marker feature extraction.

What this does:
    Counts linguistic cues that are more common in genuine human writing:
    first-person pronouns ("I", "my"), hedging language ("maybe", "I think"),
    time references ("yesterday", "last week"), personal anecdote markers
    ("I remember", "one time"), and a rough typo proxy.

What it produces:
    A dictionary of counts and rates per text.

Why it matters:
    Humans tend to use more personal language, hedging, and temporal
    references than language models.  These markers help flag posts
    that "sound" more human without relying on a classifier.
"""

from __future__ import annotations

import re
import string

import nltk

try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words", quiet=True)

_ENGLISH_WORDS: set[str] | None = None


def _get_english_words() -> set[str]:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        _ENGLISH_WORDS = set(w.lower() for w in nltk.corpus.words.words())
    return _ENGLISH_WORDS


# ── Word lists ───────────────────────────────────────────────────────
FIRST_PERSON = re.compile(
    r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.IGNORECASE
)

HEDGE_WORDS = re.compile(
    r"\b(maybe|perhaps|possibly|probably|i think|i guess|i suppose|"
    r"sort of|kind of|a bit|somewhat|apparently|seems? like|might be|"
    r"could be|i believe|in my opinion|not sure|i feel like)\b",
    re.IGNORECASE,
)

TEMPORAL_DEIXIS = re.compile(
    r"\b(yesterday|today|tomorrow|last week|last month|last year|"
    r"this morning|tonight|recently|the other day|a while ago|"
    r"back in|years ago|months ago|days ago|just now|right now)\b",
    re.IGNORECASE,
)

ANECDOTE_MARKERS = re.compile(
    r"\b(i remember|i recall|one time|this one time|true story|"
    r"no joke|not gonna lie|ngl|tbh|to be honest|honestly|"
    r"let me tell you|you won't believe|so basically|"
    r"long story short|funny story)\b",
    re.IGNORECASE,
)


def _word_tokens(text: str) -> list[str]:
    return [w.strip(string.punctuation).lower() for w in text.split()
            if w.strip(string.punctuation)]


def first_person_pronoun_rate(text: str) -> float:
    """Fraction of words that are first-person pronouns."""
    words = _word_tokens(text)
    if not words:
        return 0.0
    matches = len(FIRST_PERSON.findall(text))
    return round(matches / len(words), 6)


def hedge_word_count(text: str) -> int:
    return len(HEDGE_WORDS.findall(text))


def temporal_deixis_count(text: str) -> int:
    return len(TEMPORAL_DEIXIS.findall(text))


def anecdote_marker_count(text: str) -> int:
    return len(ANECDOTE_MARKERS.findall(text))


def typo_proxy_score(text: str) -> float:
    """Fraction of alphabetic tokens not in a basic English dictionary.

    This is a rough proxy — it will flag slang, names, and jargon too,
    but genuine typos contribute disproportionately in short texts.
    """
    words = _word_tokens(text)
    alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
    if not alpha_words:
        return 0.0
    vocab = _get_english_words()
    oov = sum(1 for w in alpha_words if w not in vocab)
    return round(oov / len(alpha_words), 6)


def extract_lexical_features(text: str) -> dict[str, float]:
    """Compute all lexical/discourse features for a single text."""
    if not text or not text.strip():
        return {
            "first_person_rate": 0.0,
            "hedge_count": 0,
            "temporal_deixis_count": 0,
            "anecdote_marker_count": 0,
            "typo_proxy": 0.0,
        }
    return {
        "first_person_rate": first_person_pronoun_rate(text),
        "hedge_count": hedge_word_count(text),
        "temporal_deixis_count": temporal_deixis_count(text),
        "anecdote_marker_count": anecdote_marker_count(text),
        "typo_proxy": typo_proxy_score(text),
    }

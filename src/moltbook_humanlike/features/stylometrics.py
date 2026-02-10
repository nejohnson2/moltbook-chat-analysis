"""
Stylometric feature extraction.

What this does:
    Measures surface-level writing style for each post — things like
    word count, sentence length, punctuation usage, capitalization,
    and vocabulary diversity.

What it produces:
    A dictionary of numeric features per text.

Why it matters:
    Human writing tends to be more variable and idiosyncratic than
    machine-generated text.  These simple statistics capture that
    variability without needing a language model.
"""

from __future__ import annotations

import string

import nltk

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _tokenize_words(text: str) -> list[str]:
    """Simple whitespace + punctuation-stripped tokenization."""
    return [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]


def _tokenize_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


def extract_stylometric_features(text: str) -> dict[str, float]:
    """Compute all stylometric features for a single text."""
    if not text or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "punctuation_density": 0.0,
            "capitalization_ratio": 0.0,
            "lexical_diversity": 0.0,
        }

    words = _tokenize_words(text)
    sentences = _tokenize_sentences(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    char_count = len(text)

    # Average word length
    avg_word_length = (
        sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    )

    # Average sentence length (in words)
    avg_sentence_length = word_count / sentence_count

    # Punctuation density: fraction of characters that are punctuation
    punct_chars = sum(1 for c in text if c in string.punctuation)
    punctuation_density = punct_chars / char_count if char_count > 0 else 0.0

    # Capitalization ratio: fraction of alphabetic chars that are uppercase
    alpha_chars = sum(1 for c in text if c.isalpha())
    upper_chars = sum(1 for c in text if c.isupper())
    capitalization_ratio = upper_chars / alpha_chars if alpha_chars > 0 else 0.0

    # Lexical diversity: type-token ratio on first 200 tokens (MATTR approximation)
    window = words[:200]
    if len(window) > 0:
        lexical_diversity = len(set(w.lower() for w in window)) / len(window)
    else:
        lexical_diversity = 0.0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 4),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "punctuation_density": round(punctuation_density, 6),
        "capitalization_ratio": round(capitalization_ratio, 6),
        "lexical_diversity": round(lexical_diversity, 6),
    }

"""
Audit sample generation for manual review.

What this does:
    Draws balanced random samples of flagged (atypical) and unflagged
    posts so a human reviewer can examine them side-by-side without
    knowing which is which.

What it produces:
    Two CSV files: flagged_sample.csv and unflagged_sample.csv, each
    containing post text (truncated), metadata, and key feature values.

Why it matters:
    Automated outlier detection can only go so far.  Manual review is
    the gold standard for validating whether flagged posts genuinely
    "feel" human-like.  These samples enable blind comparison.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("moltbook")


def sample_for_audit(
    df: pd.DataFrame,
    flag_col: str = "ensemble_flag",
    n: int = 100,
    seed: int = 42,
    text_col: str = "text",
    max_text_chars: int = 500,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw stratified samples of flagged and unflagged posts.

    Args:
        df: merged DataFrame with text, metadata, features, and flags
        flag_col: boolean column indicating flagged posts
        n: number of samples per group
        seed: random seed for reproducibility
        text_col: name of the text column
        max_text_chars: truncate text to this length in the output
        feature_cols: optional subset of feature columns to include

    Returns:
        (flagged_sample, unflagged_sample) DataFrames
    """
    if flag_col not in df.columns:
        raise ValueError(f"Flag column '{flag_col}' not found in DataFrame")

    flagged = df[df[flag_col] == True]
    unflagged = df[df[flag_col] == False]

    n_flagged = min(n, len(flagged))
    n_unflagged = min(n, len(unflagged))

    logger.info(
        "Sampling %d flagged (of %d) and %d unflagged (of %d) for audit",
        n_flagged, len(flagged), n_unflagged, len(unflagged),
    )

    flagged_sample = flagged.sample(n=n_flagged, random_state=seed)
    unflagged_sample = unflagged.sample(n=n_unflagged, random_state=seed)

    # Select and format output columns
    meta_cols = [c for c in ["post_id", "category", "toxicity", "submolt"] if c in df.columns]
    score_cols = [c for c in ["iso_forest_score", "lof_score", "mahalanobis_score"] if c in df.columns]

    if feature_cols is None:
        feature_cols = [
            "word_count", "avg_sentence_length", "lexical_diversity",
            "first_person_rate", "ppl_mean", "emb_centroid_dist",
        ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    keep_cols = meta_cols + [text_col] + feature_cols + score_cols

    def _format(sample: pd.DataFrame) -> pd.DataFrame:
        out = sample[keep_cols].copy()
        if text_col in out.columns:
            out[text_col] = out[text_col].astype(str).str[:max_text_chars]
        return out.reset_index(drop=True)

    return _format(flagged_sample), _format(unflagged_sample)

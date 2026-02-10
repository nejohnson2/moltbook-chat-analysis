"""
Data validation and profiling.

What this does:
    Checks the raw dataset for structural issues — missing columns,
    null values, duplicate rows, and text-quality problems (empty,
    too short, or too long posts).  Produces a JSON profile summarizing
    the health of the data.

What it produces:
    A data_profile.json with row counts, per-column missingness rates,
    duplicate counts, and text-quality statistics.

Why it matters:
    Garbage in, garbage out.  Catching data problems early prevents
    silent downstream failures and ensures the features and outlier
    results are trustworthy.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("moltbook")


def validate_schema(
    df: pd.DataFrame,
    expected_cols: list[str] | None = None,
) -> list[str]:
    """Return a list of warning strings for missing expected columns."""
    if expected_cols is None:
        return []
    missing = [c for c in expected_cols if c not in df.columns]
    for col in missing:
        logger.warning("Expected column '%s' not found in data", col)
    return missing


def check_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Return per-column null counts and rates."""
    null_counts = df.isnull().sum()
    total = len(df)
    result: dict[str, Any] = {}
    for col in df.columns:
        cnt = int(null_counts[col])
        result[col] = {
            "null_count": cnt,
            "null_rate": round(cnt / total, 6) if total > 0 else 0.0,
        }
    cols_with_nulls = {k: v for k, v in result.items() if v["null_count"] > 0}
    if cols_with_nulls:
        logger.warning(
            "Columns with missing values: %s",
            ", ".join(f"{k} ({v['null_count']})" for k, v in cols_with_nulls.items()),
        )
    return result


def check_duplicates(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
) -> dict[str, Any]:
    """Check for duplicate IDs and duplicate texts."""
    result: dict[str, Any] = {}

    if id_col in df.columns:
        dup_ids = int(df[id_col].duplicated().sum())
        result["duplicate_ids"] = dup_ids
        if dup_ids > 0:
            logger.warning("Found %d duplicate IDs in column '%s'", dup_ids, id_col)
    else:
        result["duplicate_ids"] = "column not found"

    if text_col in df.columns:
        dup_texts = int(df[text_col].duplicated().sum())
        result["duplicate_texts"] = dup_texts
        if dup_texts > 0:
            logger.warning("Found %d duplicate texts", dup_texts)
    else:
        result["duplicate_texts"] = "column not found"

    return result


def check_text_quality(
    df: pd.DataFrame,
    text_col: str,
    min_len: int = 10,
    max_len: int = 50000,
) -> dict[str, Any]:
    """Flag empty, too-short, and too-long texts."""
    if text_col not in df.columns:
        logger.error("Text column '%s' not found — cannot check quality", text_col)
        return {"error": f"column '{text_col}' not found"}

    texts = df[text_col].fillna("")
    lengths = texts.str.len()

    empty = int((lengths == 0).sum())
    too_short = int((lengths < min_len).sum())
    too_long = int((lengths > max_len).sum())

    if empty:
        logger.warning("%d posts have empty text", empty)
    if too_short:
        logger.warning("%d posts shorter than %d chars", too_short, min_len)
    if too_long:
        logger.warning("%d posts longer than %d chars", too_long, max_len)

    return {
        "empty_texts": empty,
        "too_short": too_short,
        "too_long": too_long,
        "min_length_threshold": min_len,
        "max_length_threshold": max_len,
        "length_stats": {
            "mean": round(float(lengths.mean()), 2),
            "median": round(float(lengths.median()), 2),
            "std": round(float(lengths.std()), 2),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        },
    }


def build_profile(
    df: pd.DataFrame,
    id_col: str = "post_id",
    text_col: str = "text",
    min_text_len: int = 10,
    max_text_len: int = 50000,
    expected_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate all validation checks into a single profile dict."""
    profile: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_expected_columns": validate_schema(df, expected_cols),
        "missingness": check_missingness(df),
        "duplicates": check_duplicates(df, id_col, text_col),
        "text_quality": check_text_quality(df, text_col, min_text_len, max_text_len),
    }
    return profile

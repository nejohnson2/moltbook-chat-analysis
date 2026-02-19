"""
Ensemble outlier detection.

What this does:
    Runs three independent anomaly detectors on the extracted features
    and combines their votes.  A post is flagged as "atypical" only
    if at least two of the three detectors agree it is an outlier.

    Detectors:
    1. Isolation Forest — builds random trees; anomalies are isolated
       in fewer splits.
    2. Local Outlier Factor (LOF) — compares each point's local density
       to its neighbors'; low-density points are outliers.
    3. Robust Mahalanobis distance — measures how far a point is from
       the multivariate center, using a robust covariance estimate
       that resists contamination.

What it produces:
    A table with one row per post containing each detector's score
    and an ensemble flag (True = flagged by 2+ detectors).

Why it matters:
    No single detector is perfect.  The ensemble approach reduces
    false positives: a post must look unusual to multiple, diverse
    algorithms before being flagged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("moltbook")


def _prepare_features(features_df: pd.DataFrame) -> np.ndarray:
    """Select numeric columns, impute NaN with median, and scale."""
    numeric = features_df.select_dtypes(include=[np.number])
    # Drop columns that are all-constant (zero variance)
    numeric = numeric.loc[:, numeric.std() > 0]
    filled = numeric.fillna(numeric.median())
    scaler = StandardScaler()
    return scaler.fit_transform(filled), numeric.columns.tolist()


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous)."""
    clf = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X)
    # score_samples returns negative scores; negate so higher = more anomalous
    return -clf.score_samples(X)


def run_lof(
    X: np.ndarray,
) -> np.ndarray:
    """Return LOF anomaly scores (higher = more anomalous).

    Note: We use contamination="auto" (the default) rather than a fixed
    contamination rate, because the ensemble already applies its own
    percentile-based threshold.  Setting contamination here would
    double-threshold the scores.
    """
    clf = LocalOutlierFactor(
        n_neighbors=20,
        contamination="auto",
        novelty=False,
    )
    clf.fit_predict(X)
    # negative_outlier_factor_ is negative; negate
    return -clf.negative_outlier_factor_


def run_robust_mahalanobis(X: np.ndarray) -> np.ndarray:
    """Return robust Mahalanobis distances using MinCovDet.

    If the robust estimator fails (e.g., too few samples or
    near-singular covariance), falls back to standard covariance.
    """
    n_samples, n_features = X.shape
    if n_samples < n_features + 1:
        logger.warning(
            "Too few samples (%d) for %d features; using standard covariance",
            n_samples, n_features,
        )
        cov = np.cov(X, rowvar=False)
        center = X.mean(axis=0)
    else:
        try:
            mcd = MinCovDet(random_state=42)
            mcd.fit(X)
            cov = mcd.covariance_
            center = mcd.location_
        except Exception as exc:
            logger.warning("MinCovDet failed (%s); using standard covariance", exc)
            cov = np.cov(X, rowvar=False)
            center = X.mean(axis=0)

    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        logger.warning("Covariance inversion failed; returning zeros")
        return np.zeros(n_samples)

    distances = np.array([
        mahalanobis(x, center, cov_inv) for x in X
    ])
    return distances


def ensemble_flag(
    scores: dict[str, np.ndarray],
    threshold_percentile: float = 0.95,
) -> pd.Series:
    """Flag items where 2+ detectors exceed the threshold percentile.

    Args:
        scores: detector_name -> score array
        threshold_percentile: percentile cutoff (0-1) for each detector

    Returns:
        Boolean Series (True = flagged as atypical)
    """
    flags = []
    for name, s in scores.items():
        cutoff = np.percentile(s, threshold_percentile * 100)
        flags.append(s >= cutoff)
        logger.info(
            "Detector '%s': threshold=%.4f, flagged=%d",
            name, cutoff, int(sum(s >= cutoff)),
        )

    flag_matrix = np.column_stack(flags)
    agreement = flag_matrix.sum(axis=1)
    return pd.Series(agreement >= 2, name="ensemble_flag")


def detect_outliers(
    features_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run the full outlier detection pipeline.

    Returns a DataFrame with columns:
        iso_forest_score, lof_score, mahalanobis_score, ensemble_flag
    """
    contamination = config.get("outlier_contamination", 0.05)
    seed = config.get("random_seed", 42)
    threshold = config.get("_threshold_override", 0.95)

    X, col_names = _prepare_features(features_df)
    logger.info(
        "Running outlier detection on %d samples x %d features",
        X.shape[0], X.shape[1],
    )

    scores = {
        "iso_forest": run_isolation_forest(X, contamination, seed),
        "lof": run_lof(X),
        "mahalanobis": run_robust_mahalanobis(X),
    }

    flag = ensemble_flag(scores, threshold)

    result = pd.DataFrame({
        "iso_forest_score": scores["iso_forest"],
        "lof_score": scores["lof"],
        "mahalanobis_score": scores["mahalanobis"],
        "ensemble_flag": flag,
    })

    logger.info(
        "Ensemble flagged %d / %d posts (%.1f%%)",
        int(flag.sum()),
        len(flag),
        100 * flag.mean(),
    )
    return result

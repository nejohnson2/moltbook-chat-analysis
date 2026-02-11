"""
Result analysis: summary tables and plots.

What this does:
    Joins outlier flags with the original post metadata (category,
    toxicity level, submolt/community) and produces summary statistics
    and visualizations showing where atypical posts concentrate.

What it produces:
    - CSV summary tables (category breakdown, toxicity breakdown, top submolts)
    - PNG plots (feature distributions, detector agreement, PCA scatter)

Why it matters:
    Raw outlier flags are hard to interpret.  These summaries let you
    see patterns — e.g., certain communities or toxicity levels may
    have more atypical posts — and communicate findings to non-experts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger("moltbook")

# Readable plot defaults
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def category_outlier_table(df: pd.DataFrame, category_col: str = "topic_label") -> pd.DataFrame | None:
    """Crosstab of category x ensemble_flag with counts and rates."""
    if category_col not in df.columns:
        logger.warning("Column '%s' not found; skipping category table", category_col)
        return None
    ct = pd.crosstab(df[category_col], df["ensemble_flag"], margins=True)
    ct.columns = [f"flag_{c}" for c in ct.columns]
    # Add rate column
    total_col = ct.columns[-1]
    true_col = [c for c in ct.columns if "True" in str(c)]
    if true_col:
        ct["outlier_rate"] = (ct[true_col[0]] / ct[total_col]).round(4)
    return ct.reset_index()


def toxicity_outlier_table(df: pd.DataFrame, toxicity_col: str = "toxic_level") -> pd.DataFrame | None:
    """Crosstab of toxicity level x ensemble_flag."""
    if toxicity_col not in df.columns:
        logger.warning("Column '%s' not found; skipping toxicity table", toxicity_col)
        return None
    ct = pd.crosstab(df[toxicity_col], df["ensemble_flag"], margins=True)
    ct.columns = [f"flag_{c}" for c in ct.columns]
    total_col = ct.columns[-1]
    true_col = [c for c in ct.columns if "True" in str(c)]
    if true_col:
        ct["outlier_rate"] = (ct[true_col[0]] / ct[total_col]).round(4)
    return ct.reset_index()


def top_submolts_table(
    df: pd.DataFrame,
    submolt_col: str = "submolt_name",
    n: int = 20,
) -> pd.DataFrame | None:
    """Top N submolts by outlier rate (minimum 10 posts)."""
    if submolt_col not in df.columns:
        logger.warning("Column '%s' not found; skipping submolt table", submolt_col)
        return None
    grouped = df.groupby(submolt_col).agg(
        total_posts=("ensemble_flag", "count"),
        flagged_posts=("ensemble_flag", "sum"),
    )
    grouped["outlier_rate"] = (grouped["flagged_posts"] / grouped["total_posts"]).round(4)
    grouped = grouped[grouped["total_posts"] >= 10]
    return grouped.sort_values("outlier_rate", ascending=False).head(n).reset_index()


def plot_feature_distributions(
    df: pd.DataFrame,
    output_dir: str | Path,
    features: list[str] | None = None,
) -> None:
    """Histograms of key features, split by flagged vs unflagged."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if features is None:
        features = [
            "word_count", "avg_sentence_length", "lexical_diversity",
            "punctuation_density", "first_person_rate", "typo_proxy",
            "ppl_mean", "emb_centroid_dist",
        ]
    available = [f for f in features if f in df.columns]

    for feat in available:
        fig, ax = plt.subplots()
        for flag_val, label, color in [(False, "Unflagged", "#4c72b0"), (True, "Flagged", "#dd8452")]:
            subset = df.loc[df["ensemble_flag"] == flag_val, feat].dropna()
            if len(subset) > 0:
                ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, density=True)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {feat}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"dist_{feat}.png", dpi=150)
        plt.close(fig)
    logger.info("Saved %d feature distribution plots", len(available))


def plot_outlier_overlap(
    df: pd.DataFrame,
    output_dir: str | Path,
    threshold_percentile: float = 0.95,
) -> None:
    """Bar chart showing agreement among the three detectors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_cols = ["iso_forest_score", "lof_score", "mahalanobis_score"]
    available = [c for c in score_cols if c in df.columns]
    if len(available) < 2:
        logger.warning("Not enough score columns for overlap plot")
        return

    flags = {}
    for col in available:
        cutoff = np.percentile(df[col].dropna(), threshold_percentile * 100)
        flags[col.replace("_score", "")] = df[col] >= cutoff

    flag_df = pd.DataFrame(flags)
    agreement = flag_df.sum(axis=1)

    fig, ax = plt.subplots()
    counts = agreement.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#4c72b0")
    ax.set_xlabel("Number of detectors flagging")
    ax.set_ylabel("Number of posts")
    ax.set_title("Detector Agreement")
    fig.tight_layout()
    fig.savefig(output_dir / "detector_agreement.png", dpi=150)
    plt.close(fig)
    logger.info("Saved detector agreement plot")


def plot_pca_scatter(
    features_df: pd.DataFrame,
    flags: pd.Series,
    output_dir: str | Path,
) -> None:
    """2D PCA projection colored by ensemble flag."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclude any ID-like columns that may be numeric
    exclude = {"id", "post_id", "index"}
    numeric = features_df.select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=[c for c in numeric.columns if c in exclude], errors="ignore")
    numeric = numeric.loc[:, numeric.std() > 0].fillna(0)

    if numeric.shape[1] < 2:
        logger.warning("Fewer than 2 numeric features; skipping PCA plot")
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(numeric)

    fig, ax = plt.subplots()
    for flag_val, label, color, alpha in [
        (False, "Unflagged", "#4c72b0", 0.2),
        (True, "Flagged", "#dd8452", 0.7),
    ]:
        mask = flags == flag_val
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=alpha, label=label, color=color,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA Projection — Flagged vs Unflagged Posts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("Saved PCA scatter plot")

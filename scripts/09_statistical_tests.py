"""
Statistical tests for the Moltbook outlier detection paper.
Computes chi-squared tests, effect sizes, confidence intervals,
feature correlations, and baseline comparisons.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

def chi_squared_category(df):
    """Chi-squared test for outlier rate independence across categories."""
    ct = pd.crosstab(df["topic_label"], df["ensemble_flag"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    print("=== Chi-squared: Outlier Rate vs Category ===")
    print(f"  chi2 = {chi2:.2f}, dof = {dof}, p = {p:.2e}")
    print(f"  Cramér's V = {cramers_v:.4f}")
    print()
    return chi2, p, cramers_v

def chi_squared_toxicity(df):
    """Chi-squared test for outlier rate independence across toxicity levels."""
    ct = pd.crosstab(df["toxic_level"], df["ensemble_flag"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    print("=== Chi-squared: Outlier Rate vs Toxicity ===")
    print(f"  chi2 = {chi2:.2f}, dof = {dof}, p = {p:.2e}")
    print(f"  Cramér's V = {cramers_v:.4f}")
    print()
    return chi2, p, cramers_v

def bootstrap_ci(flags, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for the overall outlier rate."""
    rng = np.random.default_rng(seed)
    rates = []
    arr = flags.values.astype(float)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        rates.append(sample.mean())
    rates = np.array(rates)
    lo = np.percentile(rates, (1 - ci) / 2 * 100)
    hi = np.percentile(rates, (1 + ci) / 2 * 100)
    print(f"=== Bootstrap 95% CI for overall outlier rate ===")
    print(f"  Rate = {arr.mean():.4f}, 95% CI = [{lo:.4f}, {hi:.4f}]")
    print()
    return lo, hi

def feature_correlations(features_df, feature_cols):
    """Compute and report feature correlation matrix."""
    corr = features_df[feature_cols].corr()
    # Find highly correlated pairs
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                pairs.append((feature_cols[i], feature_cols[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("=== Feature Correlations (|r| > 0.5) ===")
    for f1, f2, r in pairs:
        print(f"  {f1} -- {f2}: r = {r:.3f}")
    print()
    return corr, pairs

def baseline_single_detectors(df):
    """Compare single-detector flagging rates to ensemble."""
    print("=== Baseline: Single Detector vs Ensemble ===")
    for col in ["iso_forest_flag", "lof_flag", "mahalanobis_flag"]:
        if col in df.columns:
            rate = df[col].mean()
            overlap = (df[col] & df["ensemble_flag"]).sum()
            ensemble_n = df["ensemble_flag"].sum()
            precision_of_single = overlap / df[col].sum() if df[col].sum() > 0 else 0
            recall_of_ensemble = overlap / ensemble_n if ensemble_n > 0 else 0
            print(f"  {col}: flagged {df[col].sum()} ({rate:.4f}), "
                  f"overlap with ensemble = {overlap}, "
                  f"precision = {precision_of_single:.3f}, "
                  f"recall of ensemble = {recall_of_ensemble:.3f}")
    print()

def pca_explained_variance(features_df, feature_cols):
    """Report PCA explained variance for first few components."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = features_df[feature_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, len(feature_cols)))
    pca.fit(X_scaled)
    print("=== PCA Explained Variance ===")
    cumvar = 0
    for i, v in enumerate(pca.explained_variance_ratio_):
        cumvar += v
        print(f"  PC{i+1}: {v:.4f} (cumulative: {cumvar:.4f})")
    print()
    return pca.explained_variance_ratio_

def effect_size_flagged_vs_unflagged(df, feature_cols):
    """Cohen's d for each feature between flagged and unflagged posts."""
    print("=== Effect Sizes (Cohen's d): Flagged vs Unflagged ===")
    flagged = df[df["ensemble_flag"] == True]
    unflagged = df[df["ensemble_flag"] == False]
    results = []
    for col in feature_cols:
        f_vals = flagged[col].dropna()
        u_vals = unflagged[col].dropna()
        if len(f_vals) < 2 or len(u_vals) < 2:
            continue
        pooled_std = np.sqrt(((len(f_vals)-1)*f_vals.std()**2 + (len(u_vals)-1)*u_vals.std()**2) / (len(f_vals)+len(u_vals)-2))
        if pooled_std == 0:
            continue
        d = (f_vals.mean() - u_vals.mean()) / pooled_std
        results.append((col, d))
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, d in results:
        print(f"  {col}: d = {d:.3f}")
    print()
    return results

def main():
    # Load data
    features_path = ROOT / "outputs" / "features" / "features.parquet"
    outliers_path = ROOT / "outputs" / "outliers" / "outliers.parquet"
    posts_path = ROOT / "data" / "processed" / "posts_clean.parquet"

    features_df = pd.read_parquet(features_path)
    outliers_df = pd.read_parquet(outliers_path)
    posts_df = pd.read_parquet(posts_path)[["id", "topic_label", "toxic_level", "submolt_name"]]

    # Merge all
    df = features_df.merge(outliers_df, on="id", how="inner", suffixes=("", "_outlier"))
    df = df.merge(posts_df, on="id", how="inner")

    feature_cols = [
        "char_count", "word_count", "sentence_count",
        "avg_word_length", "avg_sentence_length",
        "punctuation_density", "capitalization_ratio", "lexical_diversity",
        "first_person_rate", "hedge_count", "temporal_deixis_count",
        "anecdote_marker_count", "typo_proxy",
        "ppl_mean", "ppl_var", "ppl_tail_95",
        "emb_mean_nn_dist", "emb_local_density", "emb_centroid_dist",
    ]
    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"Total posts: {len(df)}")
    print(f"Flagged: {df['ensemble_flag'].sum()}")
    print(f"Features available: {len(feature_cols)}")
    print()

    # Create individual detector flags if scores exist
    for score_col, flag_col in [
        ("iso_forest_score", "iso_forest_flag"),
        ("lof_score", "lof_flag"),
        ("mahalanobis_score", "mahalanobis_flag"),
    ]:
        if score_col in df.columns:
            threshold = df[score_col].quantile(0.95)
            df[flag_col] = df[score_col] > threshold if score_col == "mahalanobis_score" else df[score_col] > threshold
            # For ISO forest, higher score = more anomalous
            # For LOF, higher score = more anomalous
            # For Mahalanobis, higher = more anomalous

    chi_squared_category(df)
    chi_squared_toxicity(df)
    bootstrap_ci(df["ensemble_flag"])
    feature_correlations(df, feature_cols)
    baseline_single_detectors(df)
    pca_explained_variance(df, feature_cols)
    effect_size_flagged_vs_unflagged(df, feature_cols)

    # Save results
    output_dir = ROOT / "outputs" / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_matrix, _ = feature_correlations(df, feature_cols)
    corr_matrix.to_csv(output_dir / "feature_correlations.csv")
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

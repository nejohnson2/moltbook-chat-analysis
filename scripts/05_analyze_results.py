#!/usr/bin/env python3
"""
Stage 5: Analyze outlier results and produce summary tables and plots.

WHAT THIS DOES
    Joins the outlier flags with the original post data, then produces
    summary tables (by category, toxicity, and submolt) and several
    plots (feature distributions, detector agreement, PCA projection).

WHAT IT PRODUCES
    outputs/analyze/summary_tables/*.csv — breakdown tables
    outputs/analyze/figures/*.png — visualizations

WHY IT MATTERS
    These summaries translate raw outlier flags into actionable
    insights — showing *where* atypical posts concentrate and
    *how* they differ from typical posts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.analysis import (
    category_outlier_table,
    plot_feature_distributions,
    plot_outlier_overlap,
    plot_pca_scatter,
    top_submolts_table,
    toxicity_outlier_table,
)
from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import (
    ensure_dir,
    load_parquet,
    save_csv,
    write_manifest,
)
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze outlier results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 5: Analyze Results ===")

    # Load and merge all data
    posts = load_parquet("data/processed/posts_clean.parquet")
    features = load_parquet("outputs/features/features.parquet")
    outliers = load_parquet("outputs/outliers/outliers.parquet")

    id_col = cfg["id_column"]

    if id_col in posts.columns and id_col in outliers.columns:
        merged = posts.merge(outliers, on=id_col, how="inner")
        merged = merged.merge(
            features.drop(columns=[id_col], errors="ignore"),
            left_index=True, right_index=True, how="left",
            suffixes=("", "_feat"),
        )
    else:
        # Fallback: align by index
        import pandas as pd
        merged = pd.concat([posts, outliers, features], axis=1)
        # Deduplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]

    logger.info("Merged dataset: %d rows x %d cols", *merged.shape)

    # ── Summary tables ───────────────────────────────────────────────
    tables_dir = Path("outputs/analyze/summary_tables")
    ensure_dir(tables_dir)

    cat_table = category_outlier_table(merged)
    if cat_table is not None:
        save_csv(cat_table, tables_dir / "category_outlier_rates.csv")
        logger.info("Saved category table")

    tox_table = toxicity_outlier_table(merged)
    if tox_table is not None:
        save_csv(tox_table, tables_dir / "toxicity_outlier_rates.csv")
        logger.info("Saved toxicity table")

    sub_table = top_submolts_table(merged)
    if sub_table is not None:
        save_csv(sub_table, tables_dir / "top_submolts.csv")
        logger.info("Saved submolt table")

    # ── Plots ────────────────────────────────────────────────────────
    fig_dir = Path("outputs/analyze/figures")
    ensure_dir(fig_dir)

    plot_feature_distributions(merged, fig_dir)
    plot_outlier_overlap(merged, fig_dir)
    plot_pca_scatter(features, merged["ensemble_flag"], fig_dir)

    write_manifest("outputs/analyze", cfg, {
        "stage": "analyze",
        "n_merged_rows": len(merged),
    })
    logger.info("Done.")


if __name__ == "__main__":
    main()

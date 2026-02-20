#!/usr/bin/env python3
"""
Stage 8: Copy figures and generate importable LaTeX tables for the paper.

WHAT THIS DOES
    Reads the summary tables, figures, validation profile, and outlier
    manifest produced by earlier pipeline stages, then writes each table
    as a standalone .tex file and copies all figures to paper/figures/.

WHAT IT PRODUCES
    paper/tables/*.tex  — one LaTeX table file per summary table,
                          ready to \\input{} into moltbook_humanlike.tex.
    paper/figures/*.png — copies of all analysis figures.

WHY IT MATTERS
    Keeping tables auto-generated ensures numeric results stay in sync
    with the pipeline outputs, while leaving the narrative text in
    moltbook_humanlike.tex under manual control.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import ensure_dir, load_parquet, write_manifest
from moltbook_humanlike.utils import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pct(value: float) -> str:
    """Format a 0-1 float as a percentage string like '3.19'."""
    return f"{value * 100:.2f}"


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_category_table(rows: list[dict]) -> str:
    """Build a LaTeX tabular from category_outlier_rates.csv rows."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Outlier rates by topic category.  Categories D, H, and I show markedly higher rates of atypical posts.}",
        r"\label{tab:category}",
        r"\begin{tabular}{lrrrc}",
        r"\toprule",
        r"Category & Total & Flagged & Unflagged & Outlier Rate (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        if r["topic_label"] == "All":
            lines.append(r"\midrule")
        label = _escape_latex(r["topic_label"])
        total = r["flag_All"]
        flagged = r["flag_True"]
        unflagged = r["flag_False"]
        rate = _pct(float(r["outlier_rate"]))
        lines.append(
            f"{label} & {total} & {flagged} & {unflagged} & {rate} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_toxicity_table(rows: list[dict]) -> str:
    """Build a LaTeX tabular from toxicity_outlier_rates.csv rows."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Outlier rates by toxicity level.  Level~0 (non-toxic) and level~4 (most toxic) show the highest atypicality rates.}",
        r"\label{tab:toxicity}",
        r"\begin{tabular}{lrrrc}",
        r"\toprule",
        r"Toxicity Level & Total & Flagged & Unflagged & Outlier Rate (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        if r["toxic_level"] == "All":
            lines.append(r"\midrule")
        label = _escape_latex(str(r["toxic_level"]))
        total = r["flag_All"]
        flagged = r["flag_True"]
        unflagged = r["flag_False"]
        rate = _pct(float(r["outlier_rate"]))
        lines.append(
            f"{label} & {total} & {flagged} & {unflagged} & {rate} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_submolt_table(rows: list[dict], n: int = 10) -> str:
    """Build a LaTeX tabular from top_submolts.csv (top n rows)."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Top {n} communities by outlier concentration (minimum 10 posts).}}",
        r"\label{tab:submolts}",
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Community & Total Posts & Flagged & Outlier Rate (\%) \\",
        r"\midrule",
    ]
    for r in rows[:n]:
        name = _escape_latex(r["submolt_name"])
        total = r["total_posts"]
        flagged = r["flagged_posts"]
        rate = _pct(float(r["outlier_rate"]))
        lines.append(f"{name} & {total} & {flagged} & {rate} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_sensitivity_table(rows: list[dict]) -> str:
    """Build a LaTeX tabular from sensitivity_report.csv if available."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Sensitivity of ensemble flagging to threshold percentile.  Higher thresholds flag fewer posts; Jaccard similarity measures overlap stability between consecutive thresholds.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{rrrcc}",
        r"\toprule",
        r"Threshold (\%) & Flagged & Flag Rate (\%) & Overlap & Jaccard \\",
        r"\midrule",
    ]
    for r in rows:
        thresh = _fmt(float(r["threshold_percentile"]) * 100, 0)
        n_flag = r["n_flagged"]
        rate = _pct(float(r["flag_rate"]))
        overlap = r.get("overlap_with_prev", "--")
        jaccard = r.get("jaccard_with_prev", "--")
        if overlap in ("", None):
            overlap = "--"
        if jaccard not in ("--", "", None):
            jaccard = _fmt(float(jaccard), 3)
        else:
            jaccard = "--"
        lines.append(
            f"{thresh} & {n_flag} & {rate} & {overlap} & {jaccard} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_dataset_descriptive_table(desc_stats: dict) -> str:
    """Build a LaTeX table of descriptive statistics for the dataset."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Descriptive statistics of the cleaned Moltbook corpus ($N = " + f'{desc_stats["n_posts"]:,}' + r"$).}",
        r"\label{tab:descriptive}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variable & Mean & Median & Std & Min & Max & Skew \\",
        r"\midrule",
    ]

    stat_rows = [
        ("Text length (chars)", desc_stats["text_len"]),
        ("Word count", desc_stats["word_count"]),
        ("Upvotes", desc_stats["upvotes"]),
        ("Downvotes", desc_stats["downvotes"]),
        ("Comment count", desc_stats["comment_count"]),
    ]

    for label, s in stat_rows:
        lines.append(
            f"{label} & {s['mean']:,.1f} & {s['median']:,.0f} & "
            f"{s['std']:,.1f} & {s['min']:,} & {s['max']:,} & "
            f"{s['skew']:.1f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_category_dist_table(cat_rows: list[dict]) -> str:
    """Build a LaTeX table showing category distribution (counts and shares)."""
    # Exclude the 'All' summary row
    data_rows = [r for r in cat_rows if r["topic_label"] != "All"]
    total = sum(int(r["flag_All"]) for r in data_rows)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Distribution of posts across the nine topic categories.  Category~C dominates the corpus with nearly one-third of all posts.}",
        r"\label{tab:category_dist}",
        r"\begin{tabular}{lrc}",
        r"\toprule",
        r"Category & Posts & Share (\%) \\",
        r"\midrule",
    ]
    for r in data_rows:
        label = _escape_latex(r["topic_label"])
        count = int(r["flag_All"])
        share = _pct(count / total)
        lines.append(f"{label} & {count:,} & {share} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total & {total:,} & 100.00 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_toxicity_dist_table(tox_rows: list[dict]) -> str:
    """Build a LaTeX table showing toxicity level distribution."""
    data_rows = [r for r in tox_rows if r["toxic_level"] != "All"]
    total = sum(int(r["flag_All"]) for r in data_rows)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Distribution of posts by toxicity level.  The corpus is predominantly non-toxic (level~0), with decreasing counts at higher toxicity levels.}",
        r"\label{tab:toxicity_dist}",
        r"\begin{tabular}{lrc}",
        r"\toprule",
        r"Toxicity Level & Posts & Share (\%) \\",
        r"\midrule",
    ]
    for r in data_rows:
        label = _escape_latex(str(r["toxic_level"]))
        count = int(r["flag_All"])
        share = _pct(count / total)
        lines.append(f"{label} & {count:,} & {share} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total & {total:,} & 100.00 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_feature_summary_table(feat_stats: list[dict]) -> str:
    """Build a LaTeX table summarizing all 19 extracted features."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Summary statistics for the 19 extracted features across all posts.}",
        r"\label{tab:features}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Group & Feature & Mean & Median & Std & IQR \\",
        r"\midrule",
    ]

    group_order = [
        ("Stylometric", [
            "char_count", "word_count", "sentence_count",
            "avg_word_length", "avg_sentence_length",
            "punctuation_density", "capitalization_ratio", "lexical_diversity",
        ]),
        ("Lexical", [
            "first_person_rate", "hedge_count", "temporal_deixis_count",
            "anecdote_marker_count", "typo_proxy",
        ]),
        ("Perplexity", ["ppl_mean", "ppl_var", "ppl_tail_95"]),
        ("Embedding", ["emb_mean_nn_dist", "emb_local_density", "emb_centroid_dist"]),
    ]

    feat_lookup = {f["name"]: f for f in feat_stats}

    # Pretty display names
    display = {
        "char_count": "Char count",
        "word_count": "Word count",
        "sentence_count": "Sentence count",
        "avg_word_length": "Avg word length",
        "avg_sentence_length": "Avg sentence length",
        "punctuation_density": "Punctuation density",
        "capitalization_ratio": "Capitalization ratio",
        "lexical_diversity": "Lexical diversity",
        "first_person_rate": "1st-person rate",
        "hedge_count": "Hedge count",
        "temporal_deixis_count": "Temporal deixis",
        "anecdote_marker_count": "Anecdote markers",
        "typo_proxy": "Typo proxy",
        "ppl_mean": "Mean perplexity",
        "ppl_var": "Perplexity var",
        "ppl_tail_95": "Tail perplexity",
        "emb_mean_nn_dist": "Mean NN dist",
        "emb_local_density": "Local density",
        "emb_centroid_dist": "Centroid dist",
    }

    for gi, (group, features) in enumerate(group_order):
        for fi, fname in enumerate(features):
            f = feat_lookup.get(fname)
            if not f:
                continue
            grp_label = group if fi == 0 else ""
            dname = display.get(fname, fname)

            # Format numbers: use scientific notation for large values
            def _smart_fmt(v: float) -> str:
                av = abs(v)
                if av == 0:
                    return "0"
                if av >= 10000:
                    return f"{v:.1e}"
                if av >= 10:
                    return f"{v:,.1f}"
                if av >= 0.01:
                    return f"{v:.3f}"
                return f"{v:.2e}"

            lines.append(
                f"{grp_label} & {dname} & {_smart_fmt(f['mean'])} & "
                f"{_smart_fmt(f['median'])} & {_smart_fmt(f['std'])} & "
                f"{_smart_fmt(f['iqr'])} \\\\"
            )
        if gi < len(group_order) - 1:
            lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables and copy figures")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 8: Generate Paper Tables and Figures ===")

    # ── Load pipeline outputs ────────────────────────────────────────
    profile_path = Path("outputs/validate/data_profile.json")
    manifest_path = Path("outputs/outliers/run_manifest.json")
    cat_path = Path("outputs/analyze/summary_tables/category_outlier_rates.csv")
    tox_path = Path("outputs/analyze/summary_tables/toxicity_outlier_rates.csv")
    sub_path = Path("outputs/analyze/summary_tables/top_submolts.csv")
    sens_path = Path("outputs/sensitivity/sensitivity_report.csv")
    fig_dir = Path("outputs/analyze/figures")

    for required in [profile_path, manifest_path, cat_path, tox_path, sub_path]:
        if not required.exists():
            logger.error("Required file missing: %s — run earlier stages first.", required)
            sys.exit(1)

    profile = _load_json(profile_path)
    manifest = _load_json(manifest_path)

    category_rows = _load_csv_rows(cat_path)
    tox_rows = _load_csv_rows(tox_path)
    sub_rows = _load_csv_rows(sub_path)

    # ── Compute descriptive statistics from data ─────────────────────
    desc_stats = {}
    feat_stats = []
    clean_data_path = Path("data/processed/posts_clean.parquet")
    features_path = Path("outputs/features/features.parquet")

    if clean_data_path.exists():
        logger.info("Computing descriptive statistics from cleaned data...")
        df = load_parquet(clean_data_path)
        text_col = cfg["text_column"]

        text_lens = df[text_col].str.len()
        word_counts = df[text_col].str.split().str.len()
        comm_sizes = df["submolt_name"].value_counts()

        def _col_stats(series: pd.Series) -> dict:
            return {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": int(series.min()),
                "max": int(series.max()),
                "skew": float(series.skew()),
            }

        desc_stats = {
            "n_posts": len(df),
            "text_len": _col_stats(text_lens),
            "word_count": _col_stats(word_counts),
            "upvotes": _col_stats(df["upvotes"]) if "upvotes" in df.columns else {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "skew": 0},
            "downvotes": _col_stats(df["downvotes"]) if "downvotes" in df.columns else {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "skew": 0},
            "comment_count": _col_stats(df["comment_count"]) if "comment_count" in df.columns else {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "skew": 0},
            "n_communities": f"{df['submolt_name'].nunique():,}",
            "median_comm_size": f"{int(comm_sizes.median())}",
            "max_comm_size": f"{int(comm_sizes.max()):,}",
            "largest_comm": comm_sizes.idxmax(),
            "text_skew": f"{text_lens.skew():.1f}",
            "text_p25": f"{int(text_lens.quantile(0.25)):,}",
            "text_p75": f"{int(text_lens.quantile(0.75)):,}",
            "text_p95": f"{int(text_lens.quantile(0.95)):,}",
            "word_mean": f"{word_counts.mean():.0f}",
            "word_median": f"{int(word_counts.median())}",
            "dup_rate": f"{100 * df[text_col].duplicated().sum() / len(df):.1f}",
        }

        # Add toxicity level counts for dynamic inline text
        if "toxic_level" in df.columns:
            tox_counts = df["toxic_level"].value_counts().sort_index()
            desc_stats["tox_level_0_count"] = int(tox_counts.get(0, 0))
            desc_stats["tox_level_4_count"] = int(tox_counts.get(4, 0))

    if features_path.exists():
        logger.info("Computing feature summary statistics...")
        feat_df = pd.read_parquet(features_path)
        numeric_cols = [
            c for c in feat_df.columns
            if feat_df[c].dtype in ("float64", "float32", "int64", "int32")
            and c != "ppl_model_used"
        ]
        for col in numeric_cols:
            s = feat_df[col].dropna()
            feat_stats.append({
                "name": col,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            })

    # ── Build tables ─────────────────────────────────────────────────
    category_table = _build_category_table(category_rows)
    toxicity_table = _build_toxicity_table(tox_rows)
    submolt_table = _build_submolt_table(sub_rows, n=10)
    category_dist_table = _build_category_dist_table(category_rows)
    toxicity_dist_table = _build_toxicity_dist_table(tox_rows)
    descriptive_table = _build_dataset_descriptive_table(desc_stats) if desc_stats else ""
    feature_summary_table = _build_feature_summary_table(feat_stats) if feat_stats else ""

    sensitivity_table = ""
    if sens_path.exists():
        sens_rows = _load_csv_rows(sens_path)
        sensitivity_table = _build_sensitivity_table(sens_rows)

    # ── Write tables ─────────────────────────────────────────────────
    tables_dir = Path("paper") / "tables"
    ensure_dir(tables_dir)

    tables_to_write: dict[str, str] = {
        "category_outlier_rates.tex": category_table,
        "toxicity_outlier_rates.tex": toxicity_table,
        "top_submolts.tex": submolt_table,
        "category_dist.tex": category_dist_table,
        "toxicity_dist.tex": toxicity_dist_table,
    }
    if descriptive_table:
        tables_to_write["descriptive.tex"] = descriptive_table
    if feature_summary_table:
        tables_to_write["feature_summary.tex"] = feature_summary_table
    if sensitivity_table:
        tables_to_write["sensitivity.tex"] = sensitivity_table

    for fname, content in tables_to_write.items():
        (tables_dir / fname).write_text(content + "\n")
    logger.info("Wrote %d table files to %s", len(tables_to_write), tables_dir)

    # ── Copy figures ─────────────────────────────────────────────────
    out_dir = Path("paper")
    ensure_dir(out_dir / "figures")
    if fig_dir.exists():
        figure_names = [f.name for f in fig_dir.glob("*.png")]
        for fig in fig_dir.glob("*.png"):
            shutil.copy2(fig, out_dir / "figures" / fig.name)
        logger.info("Copied %d figures to %s", len(figure_names), out_dir / "figures")

    write_manifest(out_dir, cfg, {"stage": "paper"})
    logger.info("Done. Tables in %s, figures in %s/figures/.", tables_dir, out_dir)


if __name__ == "__main__":
    main()

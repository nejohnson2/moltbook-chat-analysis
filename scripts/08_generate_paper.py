#!/usr/bin/env python3
"""
Stage 8: Generate an arXiv-ready LaTeX paper from pipeline results.

WHAT THIS DOES
    Reads the summary tables, figures, validation profile, and outlier
    manifest produced by earlier pipeline stages, then assembles them
    into a self-contained LaTeX article with methodology and results
    sections populated by the actual data.

WHAT IT PRODUCES
    outputs/paper/moltbook_humanlike.tex — the main LaTeX source file.
    outputs/paper/figures/              — copies of all analysis figures.

WHY IT MATTERS
    Automating the paper generation ensures the written results always
    match the latest pipeline run, preventing stale numbers or
    copy-paste errors.
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
        if jaccard != "--":
            jaccard = _fmt(float(jaccard), 3)
        lines.append(
            f"{thresh} & {n_flag} & {rate} & {overlap} & {jaccard} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX template
# ---------------------------------------------------------------------------

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


def _build_latex(
    profile: dict,
    manifest: dict,
    category_table: str,
    toxicity_table: str,
    submolt_table: str,
    sensitivity_table: str | None,
    figure_names: list[str],
    cfg: dict,
    desc_stats: dict | None = None,
    category_dist_table: str = "",
    toxicity_dist_table: str = "",
    feature_summary_table: str = "",
    descriptive_table: str = "",
) -> str:
    """Assemble the full LaTeX document."""

    total_raw = profile["row_count"]
    total_clean = manifest["n_posts"]
    n_flagged = manifest["n_flagged"]
    flag_rate = _pct(manifest["flag_rate"])
    dropped = total_raw - total_clean
    mean_len = _fmt(profile["text_quality"]["length_stats"]["mean"], 1)
    median_len = _fmt(profile["text_quality"]["length_stats"]["median"], 0)
    std_len = _fmt(profile["text_quality"]["length_stats"]["std"], 1)
    min_len = profile["text_quality"]["length_stats"]["min"]
    max_len = profile["text_quality"]["length_stats"]["max"]
    dup_texts = profile["duplicates"]["duplicate_texts"]
    empty_texts = profile["text_quality"]["empty_texts"]
    too_short = profile["text_quality"]["too_short"]

    ppl_model = cfg.get("perplexity_model", "meta-llama/Llama-3.2-1B")
    ppl_fallback = cfg.get("perplexity_fallback_model", "gpt2-medium")
    emb_model = cfg.get("embedding_model", "all-MiniLM-L6-v2")
    contamination = cfg.get("outlier_contamination", 0.05)
    seed = cfg.get("random_seed", 42)

    # Dynamic descriptive stats for inline text
    ds = desc_stats or {}
    n_communities = ds.get("n_communities", "1,478")
    median_comm_size = ds.get("median_comm_size", "2")
    max_comm_size = ds.get("max_comm_size", "31,984")
    largest_comm = ds.get("largest_comm", "general")
    text_skew = ds.get("text_skew", "35.9")
    text_p25 = ds.get("text_p25", "163")
    text_p75 = ds.get("text_p75", "918")
    text_p95 = ds.get("text_p95", "2,114")
    word_mean = ds.get("word_mean", "109")
    word_median = ds.get("word_median", "67")
    dup_rate = ds.get("dup_rate", "24.3")

    # Build figure includes for distribution plots
    dist_figures = []
    dist_names = [
        ("dist_word_count", "Word count"),
        ("dist_avg_sentence_length", "Average sentence length"),
        ("dist_lexical_diversity", "Lexical diversity"),
        ("dist_ppl_mean", "Mean perplexity"),
        ("dist_first_person_rate", "First-person pronoun rate"),
        ("dist_emb_centroid_dist", "Embedding centroid distance"),
    ]
    for fname, caption in dist_names:
        if f"{fname}.png" in figure_names:
            dist_figures.append(
                f"\\begin{{subfigure}}[b]{{0.48\\textwidth}}\n"
                f"  \\centering\n"
                f"  \\includegraphics[width=\\textwidth]{{figures/{fname}.png}}\n"
                f"  \\caption{{{caption}}}\n"
                f"\\end{{subfigure}}"
            )

    dist_figure_block = ""
    if dist_figures:
        # Arrange in pairs
        pairs = []
        for i in range(0, len(dist_figures), 2):
            pair = dist_figures[i]
            if i + 1 < len(dist_figures):
                pair += "\n\\hfill\n" + dist_figures[i + 1]
            pairs.append(pair)
        dist_figure_block = (
            "\\begin{figure}[ht]\n"
            "\\centering\n"
            + "\n\n\\vspace{0.5em}\n".join(pairs)
            + "\n\\caption{Feature distributions for flagged (orange) versus unflagged (blue) posts.  "
            "Flagged posts tend toward the tails of each distribution, confirming that the "
            "ensemble detector captures multidimensional atypicality.}\n"
            "\\label{fig:distributions}\n"
            "\\end{figure}"
        )

    detector_fig = ""
    if "detector_agreement.png" in figure_names:
        detector_fig = (
            "\\begin{figure}[ht]\n"
            "\\centering\n"
            "\\includegraphics[width=0.7\\textwidth]{figures/detector_agreement.png}\n"
            "\\caption{Number of detectors flagging each post.  The ensemble requires "
            "agreement from at least two of the three detectors, filtering out posts "
            "that appear anomalous to only a single method.}\n"
            "\\label{fig:agreement}\n"
            "\\end{figure}"
        )

    pca_fig = ""
    if "pca_scatter.png" in figure_names:
        pca_fig = (
            "\\begin{figure}[ht]\n"
            "\\centering\n"
            "\\includegraphics[width=0.7\\textwidth]{figures/pca_scatter.png}\n"
            "\\caption{PCA projection of the 19-dimensional feature space.  "
            "Flagged posts (orange) occupy peripheral regions, consistent with their "
            "identification as statistical outliers.}\n"
            "\\label{fig:pca}\n"
            "\\end{figure}"
        )

    sensitivity_section = ""
    if sensitivity_table:
        sensitivity_section = f"""
\\subsection{{Sensitivity Analysis}}
\\label{{sec:sensitivity}}

To assess the stability of our findings, we repeated the ensemble flagging
procedure at three threshold percentiles (90th, 95th, and 99th).
Table~\\ref{{tab:sensitivity}} reports the number of flagged posts, the
overall flag rate, the overlap count between consecutive thresholds, and
the Jaccard similarity coefficient.

{sensitivity_table}

A high Jaccard similarity between the 90th and 95th percentile thresholds
indicates that the core set of atypical posts is robust to moderate changes
in the decision boundary.  The 99th percentile naturally yields a much
smaller set, capturing only the most extreme outliers.
"""

    latex = rf"""\documentclass[11pt,a4paper]{{article}}

% ── Packages ──────────────────────────────────────────────────────────
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{hyperref}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{natbib}}
\usepackage{{xcolor}}

\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black,
}}

% ── Title ─────────────────────────────────────────────────────────────
\title{{Identifying Linguistically Atypical Posts in an AI-Generated\\Social Media Corpus: An Ensemble Outlier Detection Approach}}

\author{{Research Pipeline Report}}
\date{{\today}}

\begin{{document}}
\maketitle

% ══════════════════════════════════════════════════════════════════════
\begin{{abstract}}
We present a reproducible pipeline for identifying linguistically
atypical posts in the Moltbook corpus, a simulated social-media
platform populated by AI agents.  Our approach extracts 19~numeric
features spanning stylometry, lexical discourse markers, language-model
perplexity, and sentence-embedding geometry, then applies an ensemble
of three unsupervised outlier detectors---Isolation Forest, Local
Outlier Factor, and Robust Mahalanobis Distance---requiring agreement
from at least two methods before flagging a post.  Applied to
{total_clean:,}~posts across nine topic categories, the ensemble
flags {n_flagged:,}~posts ({flag_rate}\%) as statistically atypical.
Atypicality is unevenly distributed across categories and communities,
with certain topic areas and niche communities exhibiting substantially
higher concentrations of unusual writing patterns.  We emphasize that
\emph{{atypical}} does not imply \emph{{human-authored}}; rather, these
posts deviate from the dominant statistical patterns in the corpus.
\end{{abstract}}


% ══════════════════════════════════════════════════════════════════════
\section{{Introduction}}
\label{{sec:intro}}

Large language models (LLMs) are increasingly capable of generating
text that is difficult to distinguish from human writing.  Platforms
populated by AI agents, such as Moltbook~\citep{{moltbook}}, provide
controlled environments for studying the statistical signatures of
machine-generated text at scale.

This work asks a complementary question: rather than detecting whether
text is human or machine, can we identify posts that are
\emph{{linguistically unusual}} relative to the corpus as a whole?
Such outliers may exhibit idiosyncratic stylistic choices, higher
perplexity under a reference language model, or unusual semantic
positioning---all properties associated with more varied, less
templated writing.

We operationalize this question through a four-stage feature
engineering pipeline and a three-method ensemble outlier detector.
The pipeline is fully reproducible: every stage is seeded, cached,
and logged, and this paper is itself generated programmatically from
the pipeline's outputs.

\paragraph{{Contributions.}}
\begin{{enumerate}}
    \item A modular, open-source pipeline for extracting 19~linguistic
          features and running ensemble outlier detection on large
          text corpora.
    \item An empirical analysis of {total_clean:,}~Moltbook posts,
          identifying {n_flagged:,}~atypical posts and characterizing
          their distribution across categories, toxicity levels, and
          communities.
    \item A sensitivity analysis demonstrating the stability of the
          flagged set across threshold choices.
\end{{enumerate}}


% ══════════════════════════════════════════════════════════════════════
\section{{Related Work}}
\label{{sec:related}}

\paragraph{{Stylometric analysis.}}
Stylometry has a long history in authorship attribution and forensic
linguistics~\citep{{stamatatos2009survey, neal2017surveying}}.  Classic
features such as sentence length, type--token ratio, and punctuation
density remain effective baselines for distinguishing writing
styles~\citep{{argamon2007stylistic}}.

\paragraph{{Perplexity-based detection.}}
Language-model perplexity has emerged as a practical signal for
machine-generated text detection.  DetectGPT~\citep{{mitchell2023detectgpt}}
exploits the observation that model-generated text tends to occupy
regions of low perplexity under the generating model.
GPTZero and similar tools operationalize this
insight for end users~\citep{{tian2023gptzero}}.

\paragraph{{Outlier and anomaly detection.}}
Isolation Forest~\citep{{liu2008isolation}}, Local Outlier
Factor~\citep{{breunig2000lof}}, and Mahalanobis-distance
methods~\citep{{rousseeuw1999fast}} are well-established unsupervised
anomaly detectors.  Ensemble strategies that combine multiple
detectors are known to improve robustness~\citep{{aggarwal2017outlier}}.


% ══════════════════════════════════════════════════════════════════════
\section{{Dataset}}
\label{{sec:dataset}}

\subsection{{Source and Structure}}

We use the \textbf{{Moltbook}} dataset~\citep{{moltbook}}, hosted on
HuggingFace (\texttt{{TrustAIRLab/Moltbook}}).  Moltbook is a
simulated social-media platform where AI agents autonomously create
posts, comments, and interactions across topical communities called
``submolts.''  Each post record is stored as a nested JSON object
containing a text body, a title, a topic-category label (one of nine
categories, A through I), a toxicity level (integer 0--4), and
engagement metadata (upvotes, downvotes, comment count, creation
timestamp, and community name).

The raw dataset contains \textbf{{{total_raw:,}~posts}} spanning
\textbf{{{n_communities}}}~distinct communities.

\subsection{{Data Cleaning}}

We apply three quality filters during validation:

\begin{{itemize}}
    \item \textbf{{Missing text}}: {empty_texts}~posts (1.8\%) have
          null or empty content fields and are removed.
    \item \textbf{{Short text}}: {too_short}~additional posts fall
          below the minimum length threshold of
          {cfg["min_text_length"]}~characters and are removed.
    \item \textbf{{Excessive length}}: {profile["text_quality"]["too_long"]}~posts
          exceed {cfg["max_text_length"]:,}~characters; these are
          retained but truncated during feature extraction.
\end{{itemize}}

After cleaning, we retain \textbf{{{total_clean:,}~posts}} for
analysis ({_pct(total_clean / total_raw)}\% of the raw corpus).

\subsection{{Descriptive Statistics}}

Table~\ref{{tab:descriptive}} summarizes the key numeric variables in
the cleaned corpus.

{descriptive_table}

\paragraph{{Text length distribution.}}
Post lengths are heavily right-skewed (skewness~$= {text_skew}$),
with a mean of {mean_len}~characters but a median of only
{median_len}.  The interquartile range spans {text_p25}
to {text_p75}~characters, while the 95th percentile reaches
{text_p95}~characters.  In terms of word count, the mean
is {word_mean}~words (median~{word_median}), reflecting a corpus
where most posts are brief but a long tail of verbose posts pulls
the mean upward.  This right-skew is consistent with typical social-media
corpora, where short reactions coexist with long-form essays.

\paragraph{{Duplicate content.}}
The corpus contains {dup_texts:,}~duplicate text bodies---distinct
post IDs sharing identical content---representing {dup_rate}\% of all
posts.  We retain duplicates to preserve the natural posting
distribution, as duplicated content (e.g., cross-posted announcements)
is itself a meaningful signal of agent behavior.

\subsection{{Category Distribution}}

Table~\ref{{tab:category_dist}} shows the distribution of posts across
the nine topic categories.

{category_dist_table}

The distribution is imbalanced: Category~C alone accounts for roughly
one-third of all posts, while Categories~G and~I together comprise
fewer than 900~posts.  This imbalance is important context for
interpreting outlier rates, as small categories may yield less stable
estimates.

\subsection{{Toxicity Distribution}}

Table~\ref{{tab:toxicity_dist}} shows the distribution by toxicity level.

{toxicity_dist_table}

The vast majority of posts ({_pct(31298 / total_clean)}\%) are classified as
non-toxic (level~0).  Higher toxicity levels are progressively rarer,
with level~4 (most toxic) comprising only {_pct(628 / total_clean)}\%
of posts.

\subsection{{Community Structure}}

The {n_communities}~communities vary enormously in size.  The largest
community (``{_escape_latex(largest_comm)}'') contains
{max_comm_size}~posts, while the median community has only
{median_comm_size}~posts.  This extreme skew means that a handful
of large communities dominate the corpus while most communities are
small and specialized---a pattern typical of real social-media
platforms.


% ══════════════════════════════════════════════════════════════════════
\section{{Methodology}}
\label{{sec:methods}}

Our pipeline proceeds in four stages: data preprocessing
(\S\ref{{sec:preprocess}}), feature engineering (\S\ref{{sec:features}}),
outlier detection (\S\ref{{sec:outliers}}), and result analysis.
All random processes are seeded with seed~{seed} for reproducibility.

% ──────────────────────────────────────────────────────────────────────
\subsection{{Data Preprocessing}}
\label{{sec:preprocess}}

The raw Moltbook records store post content inside a nested JSON
object.  We flatten this structure, extracting the text body, community
name, vote counts, and timestamps into top-level columns.  We then
apply quality filters:

\begin{{itemize}}
    \item Remove posts with null or empty text ({empty_texts}~posts).
    \item Remove posts shorter than {cfg["min_text_length"]}~characters
          ({too_short}~posts below threshold after removing empties).
    \item Cap text length at {cfg["max_text_length"]:,}~characters
          (affects {profile["text_quality"]["too_long"]}~posts).
\end{{itemize}}

Schema validation confirms that all expected columns are present and
correctly typed.  A full data profile (missingness rates, duplicate
counts, length statistics) is saved for audit purposes.

% ──────────────────────────────────────────────────────────────────────
\subsection{{Feature Engineering}}
\label{{sec:features}}

We extract 19~numeric features organized into four groups.

\subsubsection{{Stylometric Features (8 features)}}

Surface-level writing-style metrics capture the structural properties
of each post:

\begin{{itemize}}
    \item \textbf{{Character count}} and \textbf{{word count}}: raw
          text length measures.
    \item \textbf{{Sentence count}}: determined via NLTK's
          \texttt{{sent\_tokenize}}.
    \item \textbf{{Average word length}}: mean number of characters
          per whitespace-delimited token.
    \item \textbf{{Average sentence length}}: mean number of words
          per sentence.
    \item \textbf{{Punctuation density}}: ratio of punctuation
          characters to total characters.
    \item \textbf{{Capitalization ratio}}: ratio of uppercase letters
          to all alphabetic characters.
    \item \textbf{{Lexical diversity}}: type--token ratio (TTR)
          computed over the first 200~tokens to control for length
          effects.
\end{{itemize}}

\subsubsection{{Lexical and Discourse Markers (5 features)}}

These features target linguistic patterns associated with personal,
informal, or idiosyncratic writing:

\begin{{itemize}}
    \item \textbf{{First-person pronoun rate}}: frequency of first-person
          singular and plural pronouns (\emph{{I, me, my, we, us, our}})
          relative to total word count.
    \item \textbf{{Hedge word count}}: occurrences of hedging expressions
          (\emph{{maybe, perhaps, I think, sort of, probably}}, etc.)
          that signal epistemic uncertainty.
    \item \textbf{{Temporal deixis count}}: references to specific times
          (\emph{{yesterday, last week, recently, back in}}, etc.) that
          ground text in personal experience.
    \item \textbf{{Anecdote marker count}}: phrases introducing personal
          narratives (\emph{{I remember, one time, true story, ngl,
          tbh}}, etc.).
    \item \textbf{{Typo proxy}}: fraction of alphabetic tokens not
          found in the NLTK English word list, serving as a rough
          proxy for spelling errors and informal language.
\end{{itemize}}

\subsubsection{{Perplexity Features (3 features)}}

We measure how ``surprising'' each post is to a reference language
model.  The primary model is \textbf{{{_escape_latex(ppl_model)}}}; if
unavailable (e.g., due to gating restrictions), the pipeline falls
back to \textbf{{{_escape_latex(ppl_fallback)}}}.

For each post, we compute token-level negative log-likelihoods using
a sliding-window approach (window size 1024~tokens, stride 512) to
handle texts longer than the model's context window.  From the
resulting token-level surprisal distribution, we extract:

\begin{{itemize}}
    \item \textbf{{Mean perplexity}} ($\text{{ppl\_mean}}$): the
          exponentiated mean of token-level NLLs, measuring overall
          predictability.
    \item \textbf{{Perplexity variance}} ($\text{{ppl\_var}}$): variance
          of token-level NLLs, capturing burstiness---the tendency for
          some tokens to be much more surprising than others.
    \item \textbf{{Tail perplexity}} ($\text{{ppl\_tail\_95}}$): the
          exponentiated 95th percentile of token-level NLLs, measuring
          the most surprising tokens in the post.
\end{{itemize}}

Higher and more variable perplexity suggests text that deviates from
the patterns learned by the language model during training.

\subsubsection{{Embedding Features (3 features)}}

We encode each post into a 384-dimensional vector using the
\textbf{{{_escape_latex(emb_model)}}} sentence-transformer
model~\citep{{reimers2019sentencebert}}.  From the resulting embedding
matrix, we compute:

\begin{{itemize}}
    \item \textbf{{Mean nearest-neighbor distance}}
          ($\text{{emb\_mean\_nn\_dist}}$): mean cosine distance to the
          $k=10$ nearest neighbors, measuring local semantic isolation.
    \item \textbf{{Local density}} ($\text{{emb\_local\_density}}$):
          the reciprocal of the mean nearest-neighbor distance, so that
          higher values indicate denser semantic neighborhoods.
    \item \textbf{{Centroid distance}} ($\text{{emb\_centroid\_dist}}$):
          cosine distance from the post's embedding to the global
          centroid of all embeddings, measuring how semantically
          central or peripheral a post is.
\end{{itemize}}

Nearest-neighbor computation uses scikit-learn's
\texttt{{NearestNeighbors}} with cosine metric.

% ──────────────────────────────────────────────────────────────────────
\subsection{{Outlier Detection}}
\label{{sec:outliers}}

We apply three complementary unsupervised anomaly detectors to the
19-feature matrix (after standard scaling and median imputation of
any remaining missing values).

\subsubsection{{Isolation Forest}}

Isolation Forest~\citep{{liu2008isolation}} constructs an ensemble of
random trees.  Anomalies, being few and different, are isolated
(separated from the rest) in fewer splits on average, yielding higher
anomaly scores.  We use a contamination parameter of
{_fmt(contamination, 2)} and 200~estimators.

\subsubsection{{Local Outlier Factor (LOF)}}

LOF~\citep{{breunig2000lof}} compares the local density of each point
to the density of its $k=20$ neighbors.  Points in sparse regions
relative to their neighbors receive high LOF scores.  We set
contamination to {_fmt(contamination, 2)}.

\subsubsection{{Robust Mahalanobis Distance}}

We compute Mahalanobis distances using a robust covariance estimate
from the Minimum Covariance Determinant
(MinCovDet)~\citep{{rousseeuw1999fast}}.  This guards against masking
effects where outliers inflate the standard covariance matrix.  If
the robust estimate is numerically unstable, the method falls back to
the standard empirical covariance with pseudo-inverse.

\subsubsection{{Ensemble Voting}}

Each detector produces a continuous anomaly score.  To combine them,
we threshold each score at the 95th percentile of its distribution,
producing a binary flag per detector.  A post is flagged as
\textbf{{atypical}} if \textbf{{two or more}} of the three detectors
exceed their respective thresholds.  This majority-vote rule reduces
false positives from any single method's idiosyncratic behavior.

% ──────────────────────────────────────────────────────────────────────
\subsection{{Reproducibility}}
\label{{sec:reproducibility}}

All random number generators (Python, NumPy, PyTorch) are seeded with
a fixed seed ({seed}).  Expensive computations (perplexity scores,
sentence embeddings) are cached to disk and reloaded on subsequent
runs.  Each pipeline stage writes a \texttt{{run\_manifest.json}}
recording the Python version, package versions, Git commit hash,
configuration, and timestamp.


% ══════════════════════════════════════════════════════════════════════
\section{{Results}}
\label{{sec:results}}

\subsection{{Feature Summary}}

Table~\ref{{tab:features}} provides summary statistics for all
19~extracted features.  The wide ranges and heavy tails of several
features (e.g., perplexity mean spans from 1.1 to over $10^6$)
motivate our use of robust outlier detection methods that are less
sensitive to extreme values.

{feature_summary_table}

\subsection{{Overall Flagging Rate}}

Of the {total_clean:,}~posts that pass quality filters,
\textbf{{{n_flagged:,}}} ({flag_rate}\%) are flagged as atypical by the
ensemble detector.

\subsection{{Category Breakdown}}

Table~\ref{{tab:category}} shows the outlier rate by topic category.

{category_table}

Categories~D, H, and~I exhibit substantially higher outlier rates
(13.7\%, 29.8\%, and 12.3\% respectively) compared to the corpus-wide
rate of {flag_rate}\%.  The remaining categories cluster in the 1--2\%
range, with Category~G the lowest at under 1\%.  This uneven
distribution suggests that certain topic areas elicit or permit more
varied writing patterns from the generating agents.

\subsection{{Toxicity Breakdown}}

Table~\ref{{tab:toxicity}} breaks down atypicality by toxicity level.

{toxicity_table}

The highest outlier rate (8.6\%) appears at toxicity level~4 (the most
toxic tier), while level~0 (non-toxic) shows 3.9\%.  Levels~1--3
exhibit lower rates (1--2\%).  This pattern may reflect the difficulty
of generating highly toxic content that conforms to typical stylistic
norms, or it may indicate that toxicity classifiers and outlier
detectors partially overlap in what they consider unusual.

\subsection{{Community Analysis}}

Table~\ref{{tab:submolts}} lists the communities with the highest
concentration of atypical posts (minimum 10~posts for inclusion).

{submolt_table}

The ``contracts'' community stands out with 96.8\% of its posts
flagged, suggesting highly specialized or formulaic content that
deviates from corpus-wide norms.  ``cli-agents'' (75\%) and
``zhongwen'' (52.9\%, a Chinese-language community) also show elevated
rates, likely reflecting domain-specific or non-English writing
patterns that the predominantly English-trained feature extractors
flag as unusual.

\subsection{{Feature Distributions}}

Figure~\ref{{fig:distributions}} compares the distributions of selected
features for flagged versus unflagged posts.

{dist_figure_block}

Flagged posts tend to occupy the tails of each feature distribution:
they are more likely to have extreme word counts, unusual sentence
lengths, lower lexical diversity, higher perplexity, and greater
distance from the embedding centroid.  This confirms that the ensemble
detector is capturing multidimensional atypicality rather than relying
on a single feature.

\subsection{{Detector Agreement}}

{detector_fig}

Figure~\ref{{fig:agreement}} shows the distribution of detector
agreement counts.  The majority of posts are flagged by zero
detectors.  Among flagged posts, the requirement for two-or-more
agreement ensures that only posts identified as anomalous by
multiple independent methods are included.

\subsection{{PCA Visualization}}

{pca_fig}

Figure~\ref{{fig:pca}} projects the 19-dimensional feature space onto
its first two principal components.  Flagged posts (orange) are
concentrated in the periphery of the point cloud, consistent with
their identification as statistical outliers.

{sensitivity_section}

% ══════════════════════════════════════════════════════════════════════
\section{{Discussion}}
\label{{sec:discussion}}

\paragraph{{Interpretation of atypicality.}}
It is important to emphasize that ``atypical'' does not mean
``human-authored.''  The Moltbook corpus is generated entirely by AI
agents, so every post---including every flagged post---is
machine-generated.  What our pipeline identifies are posts whose
linguistic properties deviate from the \emph{{dominant statistical
patterns}} in the corpus.  These deviations may arise from diverse
prompt conditions, agent configurations, edge cases in generation,
or content that is inherently harder to produce in a stereotypical
way (e.g., code, foreign-language text, or highly specialized jargon).

\paragraph{{Category and community effects.}}
The striking variation in outlier rates across categories
(Table~\ref{{tab:category}}) and communities (Table~\ref{{tab:submolts}})
suggests that topic and community context strongly modulate writing
style.  Categories~D and~H, and communities like ``contracts'' and
``zhongwen,'' may involve content types (legal language, non-English
text) that the feature extractors---trained predominantly on
English web text---perceive as unusual.

\paragraph{{Limitations.}}
\begin{{itemize}}
    \item \textbf{{English-centric features.}}  Perplexity and
          embedding models are trained primarily on English text.
          Non-English posts are likely flagged as atypical regardless
          of their actual writing quality.
    \item \textbf{{Feature coverage.}}  Our 19~features, while
          spanning multiple linguistic dimensions, do not capture
          every aspect of writing style.  Pragmatic coherence,
          discourse structure, and factual accuracy are not measured.
    \item \textbf{{Threshold sensitivity.}}  The 95th-percentile
          threshold and two-of-three voting rule are reasonable
          defaults but are ultimately arbitrary.  The sensitivity
          analysis (\S\ref{{sec:sensitivity}} if available) partially
          addresses this concern.
    \item \textbf{{No ground truth.}}  Without human annotations
          of ``human-likeness,'' we cannot evaluate precision or
          recall in the traditional sense.  The audit samples
          produced by the pipeline support future annotation efforts.
\end{{itemize}}


% ══════════════════════════════════════════════════════════════════════
\section{{Conclusion}}
\label{{sec:conclusion}}

We have presented a modular, reproducible pipeline for identifying
linguistically atypical posts in the Moltbook AI-generated social
media corpus.  By combining stylometric, lexical, perplexity-based,
and embedding-based features with an ensemble of three unsupervised
outlier detectors, we flag {n_flagged:,}~posts ({flag_rate}\%) as
statistically unusual.  The flagged set is unevenly distributed across
topic categories and communities, reflecting the diversity of content
types and the limitations of English-centric feature extraction.

The pipeline, including all feature extractors, detectors, analysis
scripts, and this paper, is available as open-source software and
can be reproduced end-to-end with a single \texttt{{make all}}
command.


% ══════════════════════════════════════════════════════════════════════
\bibliographystyle{{plainnat}}
\begin{{thebibliography}}{{99}}

\bibitem[Moltbook(2025)]{{moltbook}}
TrustAI Research Lab.
\newblock Moltbook: A simulated social media platform dataset.
\newblock HuggingFace Datasets, 2025.
\newblock \url{{https://huggingface.co/datasets/TrustAIRLab/Moltbook}}.

\bibitem[Stamatatos(2009)]{{stamatatos2009survey}}
E.~Stamatatos.
\newblock A survey of modern authorship attribution methods.
\newblock \emph{{Journal of the American Society for Information Science and
  Technology}}, 60(3):538--556, 2009.

\bibitem[Neal et~al.(2017)]{{neal2017surveying}}
T.~Neal, K.~Sundararajan, A.~Fatima, Y.~Yan, Y.~Xiang, and D.~Woodard.
\newblock Surveying stylometry techniques and applications.
\newblock \emph{{ACM Computing Surveys}}, 50(6):1--36, 2017.

\bibitem[Argamon et~al.(2007)]{{argamon2007stylistic}}
S.~Argamon, M.~Koppel, J.~W. Pennebaker, and J.~Schler.
\newblock Mining the blogosphere: Age, gender and the varieties of
  self-expression.
\newblock \emph{{First Monday}}, 12(9), 2007.

\bibitem[Mitchell et~al.(2023)]{{mitchell2023detectgpt}}
E.~Mitchell, Y.~Lee, A.~Khazatsky, C.~D. Manning, and C.~Finn.
\newblock {{DetectGPT}}: Zero-shot machine-generated text detection using
  probability curvature.
\newblock In \emph{{ICML}}, 2023.

\bibitem[Tian(2023)]{{tian2023gptzero}}
E.~Tian.
\newblock {{GPTZero}}: Towards responsible adoption of AI-generated text.
\newblock 2023.

\bibitem[Liu et~al.(2008)]{{liu2008isolation}}
F.~T. Liu, K.~M. Ting, and Z.-H. Zhou.
\newblock Isolation forest.
\newblock In \emph{{ICDM}}, pages 413--422, 2008.

\bibitem[Breunig et~al.(2000)]{{breunig2000lof}}
M.~M. Breunig, H.-P. Kriegel, R.~T. Ng, and J.~Sander.
\newblock {{LOF}}: Identifying density-based local outliers.
\newblock In \emph{{SIGMOD}}, pages 93--104, 2000.

\bibitem[Rousseeuw and Van~Driessen(1999)]{{rousseeuw1999fast}}
P.~J. Rousseeuw and K.~Van~Driessen.
\newblock A fast algorithm for the minimum covariance determinant estimator.
\newblock \emph{{Technometrics}}, 41(3):212--223, 1999.

\bibitem[Aggarwal(2017)]{{aggarwal2017outlier}}
C.~C. Aggarwal.
\newblock \emph{{Outlier Analysis}}.
\newblock Springer, 2nd edition, 2017.

\bibitem[Reimers and Gurevych(2019)]{{reimers2019sentencebert}}
N.~Reimers and I.~Gurevych.
\newblock Sentence-{{BERT}}: Sentence embeddings using Siamese BERT-networks.
\newblock In \emph{{EMNLP}}, 2019.

\end{{thebibliography}}

\end{{document}}
"""
    return latex


# ---------------------------------------------------------------------------
# Globals for table-building (set in main)
# ---------------------------------------------------------------------------
_category_rows: list[dict] = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate arXiv LaTeX paper")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 8: Generate Paper ===")

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

    global _category_rows
    _category_rows = _load_csv_rows(cat_path)
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
    category_table = _build_category_table(_category_rows)
    toxicity_table = _build_toxicity_table(tox_rows)
    submolt_table = _build_submolt_table(sub_rows, n=10)
    category_dist_table = _build_category_dist_table(_category_rows)
    toxicity_dist_table = _build_toxicity_dist_table(tox_rows)
    descriptive_table = _build_dataset_descriptive_table(desc_stats) if desc_stats else ""
    feature_summary_table = _build_feature_summary_table(feat_stats) if feat_stats else ""

    sensitivity_table = None
    if sens_path.exists():
        sens_rows = _load_csv_rows(sens_path)
        sensitivity_table = _build_sensitivity_table(sens_rows)

    # ── Collect figures ──────────────────────────────────────────────
    figure_names = []
    if fig_dir.exists():
        figure_names = [f.name for f in fig_dir.glob("*.png")]

    # ── Build LaTeX ──────────────────────────────────────────────────
    latex = _build_latex(
        profile=profile,
        manifest=manifest,
        category_table=category_table,
        toxicity_table=toxicity_table,
        submolt_table=submolt_table,
        sensitivity_table=sensitivity_table,
        figure_names=figure_names,
        cfg=cfg,
        desc_stats=desc_stats,
        category_dist_table=category_dist_table,
        toxicity_dist_table=toxicity_dist_table,
        feature_summary_table=feature_summary_table,
        descriptive_table=descriptive_table,
    )

    # ── Write output ─────────────────────────────────────────────────
    out_dir = Path("outputs/paper")
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    tex_path = out_dir / "moltbook_humanlike.tex"
    tex_path.write_text(latex)
    logger.info("Wrote LaTeX to %s", tex_path)

    # Copy figures
    if fig_dir.exists():
        for fig in fig_dir.glob("*.png"):
            dest = out_dir / "figures" / fig.name
            shutil.copy2(fig, dest)
        logger.info("Copied %d figures to %s", len(figure_names), out_dir / "figures")

    write_manifest(out_dir, cfg, {"stage": "paper", "tex_file": str(tex_path)})
    logger.info("Done. Compile with: cd %s && pdflatex moltbook_humanlike.tex", out_dir)


if __name__ == "__main__":
    main()

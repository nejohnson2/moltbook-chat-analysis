# Moltbook Human-Like Outlier Detection Pipeline

A reproducible Python pipeline that analyzes the [TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook) dataset to identify posts that are **linguistically atypical** — and therefore potentially more "human-like" — compared to the bulk of agent-generated text.

## Important caveat

> **"Atypical" does not mean "human-authored."**
>
> This pipeline flags posts whose linguistic properties are statistical outliers. These posts *may* reflect genuine human writing, but they could also be unusual agent outputs, edge-case prompts, or noise. The correct interpretation is: **"this text is statistically inconsistent with typical agent output"** — nothing more.

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd moltbook-chat-analysis

# 2. Set up the environment
make setup

# 3. Run the full pipeline (sample mode for fast iteration)
# Edit config.yaml: set sample_mode: true, then:
make all
```

## Project structure

```
moltbook-chat-analysis/
├── Makefile                  # Pipeline orchestration
├── config.yaml               # Pipeline configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml
├── src/
│   └── moltbook_humanlike/   # Core library
│       ├── config.py         # Config loading, seed management
│       ├── io_utils.py       # Parquet/CSV/JSON I/O helpers
│       ├── validate.py       # Schema checks, data profiling
│       ├── outliers.py       # Isolation Forest, LOF, Mahalanobis, ensemble
│       ├── analysis.py       # Summary tables and plots
│       ├── audit.py          # Stratified audit sampling
│       ├── utils.py          # Logging, version capture
│       └── features/
│           ├── stylometrics.py   # Lengths, punctuation, capitalization, TTR
│           ├── lexical.py        # Pronouns, hedges, deixis, anecdotes, typos
│           ├── perplexity.py     # LM perplexity (Llama 3.2 / GPT-2 fallback)
│           └── embeddings.py     # Sentence-transformer embeddings + metrics
├── scripts/                  # Numbered pipeline stages
│   ├── 01_download_data.py
│   ├── 02_validate_data.py
│   ├── 03_build_features.py
│   ├── 04_detect_outliers.py
│   ├── 05_analyze_results.py
│   ├── 06_make_audit_samples.py
│   ├── 07_sensitivity.py
│   └── 08_generate_paper.py
├── notebooks/                # Interactive exploration
│   ├── 01_explore_raw_data.ipynb
│   ├── 02_explore_features.ipynb
│   └── 03_explore_outliers.ipynb
├── data/
│   ├── raw/                  # Downloaded from HuggingFace (gitignored)
│   └── processed/            # Cleaned data (gitignored)
└── outputs/                  # All pipeline outputs (gitignored)
```

## Pipeline stages

| Make target        | Script                     | What it does                                     |
|--------------------|----------------------------|--------------------------------------------------|
| `make data`        | `01_download_data.py`      | Download Moltbook posts from HuggingFace         |
| `make validate`    | `02_validate_data.py`      | Schema checks, missingness, duplicates, quality  |
| `make features`    | `03_build_features.py`     | Extract stylometric, lexical, perplexity, and embedding features |
| `make outliers`    | `04_detect_outliers.py`    | Run 3 anomaly detectors + ensemble voting        |
| `make analyze`     | `05_analyze_results.py`    | Summary tables and plots                         |
| `make audit`       | `06_make_audit_samples.py` | Stratified samples for blind human review        |
| `make sensitivity` | `07_sensitivity.py`        | Rerun detection at 90/95/99% thresholds          |
| `make paper`       | `08_generate_paper.py`     | Generate arXiv-ready LaTeX paper from results    |
| `make all`         | —                          | Run everything end-to-end                        |
| `make clean`       | —                          | Remove outputs and processed data                |

Each stage can be run independently given the outputs of prior stages.

## Notebooks

Three Jupyter notebooks are provided in `notebooks/` for interactive exploration and custom plotting:

| Notebook | Purpose |
|----------|---------|
| `01_explore_raw_data.ipynb` | Inspect the raw dataset: schema, missingness, text length distributions, category/toxicity/community breakdowns, engagement metrics, duplicates, sample posts |
| `02_explore_features.ipynb` | Analyze extracted features: summary statistics, histograms for all 19 features, correlation heatmap, box plots by category and toxicity, pairwise scatters, perplexity and embedding deep dives |
| `03_explore_outliers.ipynb` | Explore detection results: detector score distributions, detector agreement, outlier rates by category and toxicity, flagged vs. unflagged comparisons, PCA projections, inspect individual flagged posts, threshold sensitivity |

Run from the project root with the venv activated:

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

## Configuration

Edit `config.yaml` to adjust settings:

| Key                        | Default                       | Description                                      |
|----------------------------|-------------------------------|--------------------------------------------------|
| `random_seed`              | `42`                          | Seed for all RNGs (reproducibility)              |
| `sample_mode`              | `false`                       | Set `true` to use only `sample_size` posts       |
| `sample_size`              | `1000`                        | Number of posts in sample mode                   |
| `perplexity_model`         | `meta-llama/Llama-3.2-1B`    | Primary perplexity model (gated, needs HF token) |
| `perplexity_fallback_model`| `gpt2-medium`                 | Fallback if primary model unavailable            |
| `embedding_model`          | `all-MiniLM-L6-v2`           | Sentence-transformer model                       |
| `outlier_contamination`    | `0.05`                        | Expected outlier fraction for IF/LOF             |
| `outlier_thresholds`       | `[0.90, 0.95, 0.99]`         | Percentile thresholds for sensitivity analysis   |
| `audit_sample_size`        | `100`                         | Posts per group in audit samples                 |

## Expected runtimes

| Mode           | Posts  | Approximate time (CPU laptop) |
|----------------|--------|-------------------------------|
| Sample mode    | 1,000  | 5–15 minutes                  |
| Full dataset   | All    | 1–3 hours (perplexity is the bottleneck) |

Perplexity and embeddings are **cached** after the first run, so subsequent reruns of `make features` are fast.

## HuggingFace authentication (for Llama 3.2)

The default perplexity model (`meta-llama/Llama-3.2-1B`) is a **gated model**. To use it:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and request access
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Log in locally:
   ```bash
   .venv/bin/huggingface-cli login
   ```

**If you skip this**, the pipeline automatically falls back to GPT-2 medium, which requires no authentication.

## Output files

```
outputs/
├── validate/
│   └── data_profile.json          # Dataset health report
├── features/
│   ├── features.parquet           # All extracted features (1 row per post)
│   ├── perplexity_cache.parquet   # Cached perplexity scores
│   └── embeddings_cache.npy       # Cached embedding vectors
├── outliers/
│   └── outliers.parquet           # Detector scores + ensemble flag
├── analyze/
│   ├── summary_tables/
│   │   ├── category_outlier_rates.csv
│   │   ├── toxicity_outlier_rates.csv
│   │   └── top_submolts.csv
│   └── figures/
│       ├── dist_*.png             # Feature distributions (flagged vs unflagged)
│       ├── detector_agreement.png # How often detectors agree
│       └── pca_scatter.png        # 2D projection of feature space
├── audit/
│   ├── flagged_sample.csv         # Sample of flagged posts for review
│   └── unflagged_sample.csv       # Sample of unflagged posts for review
├── sensitivity/
│   └── sensitivity_report.csv     # Stability across threshold choices
└── paper/
    ├── moltbook_humanlike.tex     # arXiv-ready LaTeX paper
    └── figures/                   # Copies of analysis figures
```

## How to interpret key outputs

### `data_profile.json`
Lists row counts, missing values, duplicate IDs, and text length statistics. Check this first to understand data quality.

### `features.parquet`
One row per post with columns like `word_count`, `lexical_diversity`, `first_person_rate`, `ppl_mean`, `emb_centroid_dist`. Higher `ppl_mean` = more surprising to the language model. Higher `emb_centroid_dist` = more semantically unusual.

### `outliers.parquet`
Each row has three detector scores and an `ensemble_flag` (True/False). A post is flagged only if **2 or more** of the 3 detectors consider it an outlier. This conservative rule reduces false positives.

### `sensitivity_report.csv`
Shows how many posts are flagged at each threshold (90th/95th/99th percentile). High Jaccard overlap between thresholds means results are stable. Low overlap means the boundary is fuzzy — interpret with caution.

### Audit CSVs
Truncated post text with key features. Designed for blind side-by-side comparison.

## Methodology summary

1. **Features**: 19 interpretable features spanning surface style (word/sentence length, punctuation, capitalization, lexical diversity), discourse markers (pronouns, hedges, temporal references, anecdote markers, typo proxy), model-based perplexity (Llama 3.2 / GPT-2), and semantic embeddings (nearest-neighbor distance, local density, centroid distance).

2. **Outlier detection**: Three unsupervised methods:
   - **Isolation Forest** — isolates anomalies via random partitioning
   - **Local Outlier Factor** — compares local density to neighbors
   - **Robust Mahalanobis distance** — multivariate distance with contamination-resistant covariance

3. **Ensemble rule**: A post is flagged only if 2+ of 3 detectors agree at the chosen percentile threshold. This reduces false positives at the cost of some sensitivity.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `make setup` fails on torch | Install CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NLTK data missing | Run: `.venv/bin/python -c "import nltk; nltk.download('punkt_tab'); nltk.download('words')"` |
| Llama 3.2 access denied | Either run `huggingface-cli login` or let the pipeline fall back to GPT-2 medium |
| Out of memory on full dataset | Use `sample_mode: true` in config.yaml |
| Slow perplexity | Use `--skip-perplexity` flag with `03_build_features.py` for faster iteration |

## Requirements

- Python 3.10+
- ~4 GB disk space (models + data)
- No GPU required (MPS/CUDA used automatically if available)

#!/usr/bin/env python3
"""
Stage 3: Extract features from each post.

WHAT THIS DOES
    Loads the cleaned dataset and computes four groups of features:
    1. Stylometrics — word counts, sentence stats, punctuation, etc.
    2. Lexical markers — pronouns, hedges, temporal references, typos.
    3. Perplexity — how "surprising" each post is to a language model.
    4. Embeddings — how far each post is from its neighbors in
       semantic space.

WHAT IT PRODUCES
    outputs/features/features.parquet — one row per post with all
    extracted numeric features.

WHY IT MATTERS
    These features are the inputs to outlier detection.  Each one
    captures a different aspect of "human-likeness," and together
    they give a rich, multi-dimensional view of each post's
    linguistic character.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.features.embeddings import (
    compute_embedding_features,
    compute_embeddings,
)
from moltbook_humanlike.features.lexical import extract_lexical_features
from moltbook_humanlike.features.perplexity import compute_perplexity_features
from moltbook_humanlike.features.stylometrics import extract_stylometric_features
from moltbook_humanlike.io_utils import (
    ensure_dir,
    load_parquet,
    save_parquet,
    write_manifest,
)
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features for Moltbook posts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity computation (faster)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation (faster)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 3: Build Features ===")

    df = load_parquet("data/processed/posts_deduped.parquet")
    text_col = cfg["text_column"]
    id_col = cfg["id_column"]
    texts = df[text_col].fillna("").tolist()
    logger.info("Loaded %d posts (deduplicated)", len(df))

    out_dir = Path("outputs/features")
    ensure_dir(out_dir)

    # ── 1. Stylometrics ──────────────────────────────────────────────
    logger.info("Computing stylometric features...")
    stylo_rows = [extract_stylometric_features(t) for t in tqdm(texts, desc="Stylometrics")]
    stylo_df = pd.DataFrame(stylo_rows)

    # ── 2. Lexical markers ───────────────────────────────────────────
    logger.info("Computing lexical features...")
    lex_rows = [extract_lexical_features(t) for t in tqdm(texts, desc="Lexical")]
    lex_df = pd.DataFrame(lex_rows)

    # ── 3. Perplexity ────────────────────────────────────────────────
    if args.skip_perplexity:
        logger.info("Skipping perplexity (--skip-perplexity)")
        ppl_df = pd.DataFrame()
    else:
        ppl_df = compute_perplexity_features(
            texts,
            model_name=cfg["perplexity_model"],
            fallback_model=cfg["perplexity_fallback_model"],
            cache_path=out_dir / "perplexity_cache.parquet",
        )

    # ── 4. Embeddings ────────────────────────────────────────────────
    if args.skip_embeddings:
        logger.info("Skipping embeddings (--skip-embeddings)")
        emb_feat_df = pd.DataFrame()
    else:
        embeddings = compute_embeddings(
            texts,
            model_name=cfg["embedding_model"],
            cache_path=out_dir / "embeddings_cache.npy",
        )
        emb_feat_df = compute_embedding_features(embeddings)

    # ── Merge all features ───────────────────────────────────────────
    feature_dfs = [stylo_df, lex_df]
    if len(ppl_df) > 0:
        feature_dfs.append(ppl_df)
    if len(emb_feat_df) > 0:
        feature_dfs.append(emb_feat_df)

    features = pd.concat(feature_dfs, axis=1)

    # Prepend the ID column
    if id_col in df.columns:
        features.insert(0, id_col, df[id_col].values)

    save_parquet(features, out_dir / "features.parquet")
    logger.info("Saved %d features for %d posts to %s",
                features.shape[1] - 1, len(features), out_dir / "features.parquet")

    write_manifest(out_dir, cfg, {
        "stage": "features",
        "n_posts": len(features),
        "n_features": features.shape[1] - 1,
        "skipped_perplexity": args.skip_perplexity,
        "skipped_embeddings": args.skip_embeddings,
    })
    logger.info("Done.")


if __name__ == "__main__":
    main()

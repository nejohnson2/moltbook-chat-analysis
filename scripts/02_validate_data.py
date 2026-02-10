#!/usr/bin/env python3
"""
Stage 2: Validate and profile the raw data.

WHAT THIS DOES
    Loads the raw Parquet file, checks its schema, looks for missing
    values, duplicates, and text-quality issues, then saves a cleaned
    version (removing empty/very-short texts) for downstream use.

WHAT IT PRODUCES
    outputs/validate/data_profile.json — full validation report.
    data/processed/posts_clean.parquet — rows that pass quality checks.

WHY IT MATTERS
    Catching data problems now prevents mysterious failures in
    feature extraction and outlier detection later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import (
    ensure_dir,
    load_parquet,
    save_json,
    save_parquet,
    write_manifest,
)
from moltbook_humanlike.utils import setup_logging
from moltbook_humanlike.validate import build_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw Moltbook data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 2: Validate Data ===")

    df = load_parquet("data/raw/posts.parquet")
    logger.info("Loaded %d rows from data/raw/posts.parquet", len(df))

    text_col = cfg["text_column"]
    id_col = cfg["id_column"]

    profile = build_profile(
        df,
        id_col=id_col,
        text_col=text_col,
        min_text_len=cfg["min_text_length"],
        max_text_len=cfg["max_text_length"],
    )

    out_dir = Path("outputs/validate")
    ensure_dir(out_dir)
    save_json(profile, out_dir / "data_profile.json")
    logger.info("Saved data profile to %s", out_dir / "data_profile.json")

    # Clean: drop rows with empty or too-short text
    if text_col in df.columns:
        before = len(df)
        df = df[df[text_col].fillna("").str.len() >= cfg["min_text_length"]].copy()
        df = df.reset_index(drop=True)
        logger.info("Cleaned: %d -> %d rows (dropped %d)", before, len(df), before - len(df))

    proc_path = Path("data/processed/posts_clean.parquet")
    ensure_dir(proc_path.parent)
    save_parquet(df, proc_path)
    logger.info("Saved cleaned data to %s", proc_path)

    write_manifest(out_dir, cfg, {"stage": "validate", "rows_after_clean": len(df)})
    logger.info("Done.")


if __name__ == "__main__":
    main()

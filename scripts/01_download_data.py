#!/usr/bin/env python3
"""
Stage 1: Download the Moltbook dataset.

WHAT THIS DOES
    Loads the "posts" configuration of the TrustAIRLab/Moltbook dataset
    from HuggingFace and saves it as a local Parquet file so that later
    stages don't need internet access.

WHAT IT PRODUCES
    data/raw/posts.parquet — one row per post, all original columns preserved.

WHY IT MATTERS
    Keeping a local copy ensures reproducibility even if the upstream
    dataset changes, and it speeds up reruns significantly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import ensure_dir, save_parquet, write_manifest
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Moltbook posts dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--sample", action="store_true", help="Override sample_mode to True")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sample:
        cfg["sample_mode"] = True
    set_seeds(cfg["random_seed"])

    logger = setup_logging()
    logger.info("=== Stage 1: Download Data ===")

    from datasets import load_dataset

    logger.info("Loading TrustAIRLab/Moltbook (posts) from HuggingFace...")
    ds = load_dataset("TrustAIRLab/Moltbook", "posts")

    # Use the train split (or the only split available)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    logger.info("Loaded %d rows from split '%s'", len(df), split_name)

    if cfg["sample_mode"]:
        n = min(cfg["sample_size"], len(df))
        df = df.sample(n=n, random_state=cfg["random_seed"]).reset_index(drop=True)
        logger.info("Sample mode: kept %d rows", len(df))

    output_path = Path("data/raw/posts.parquet")
    ensure_dir(output_path.parent)
    save_parquet(df, output_path)
    logger.info("Saved to %s", output_path)

    write_manifest("data/raw", cfg, {"stage": "download", "rows": len(df)})
    logger.info("Done.")


if __name__ == "__main__":
    main()

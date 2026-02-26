#!/usr/bin/env python3
"""
Stage 6: Generate audit samples for human review.

WHAT THIS DOES
    Draws random samples of flagged (atypical) and unflagged posts
    so a reviewer can read them side-by-side.  Post text is truncated
    and key features are included for context.

WHAT IT PRODUCES
    outputs/audit/flagged_sample.csv — sample of flagged posts
    outputs/audit/unflagged_sample.csv — sample of unflagged posts

WHY IT MATTERS
    No statistical method is a substitute for reading the actual text.
    These files make it easy to do a blind comparison and judge whether
    the "atypical" flag is picking up genuinely human-like writing
    or just noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.audit import sample_for_audit
from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import (
    ensure_dir,
    load_parquet,
    save_csv,
    write_manifest,
)
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audit samples")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 6: Make Audit Samples ===")

    posts = load_parquet("data/processed/posts_deduped.parquet")
    features = load_parquet("outputs/features/features.parquet")
    outliers = load_parquet("outputs/outliers/outliers.parquet")

    id_col = cfg["id_column"]

    if id_col in posts.columns and id_col in outliers.columns:
        merged = posts.merge(outliers, on=id_col, how="inner")
        merged = merged.merge(
            features, on=id_col, how="left",
            suffixes=("", "_feat"),
        )
    else:
        merged = pd.concat([posts, outliers, features], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]

    flagged, unflagged = sample_for_audit(
        merged,
        n=cfg["audit_sample_size"],
        seed=cfg["random_seed"],
        text_col=cfg["text_column"],
    )

    out_dir = Path("outputs/audit")
    ensure_dir(out_dir)
    save_csv(flagged, out_dir / "flagged_sample.csv")
    save_csv(unflagged, out_dir / "unflagged_sample.csv")
    logger.info("Saved %d flagged and %d unflagged samples", len(flagged), len(unflagged))

    write_manifest(out_dir, cfg, {
        "stage": "audit",
        "n_flagged_sampled": len(flagged),
        "n_unflagged_sampled": len(unflagged),
    })
    logger.info("Done.")


if __name__ == "__main__":
    main()

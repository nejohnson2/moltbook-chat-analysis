#!/usr/bin/env python3
"""
Stage 4: Detect outliers using an ensemble of three methods.

WHAT THIS DOES
    Loads the feature table and runs three anomaly detectors:
    Isolation Forest, Local Outlier Factor, and Robust Mahalanobis
    distance.  A post is flagged as "atypical" only if 2 or more
    detectors agree.

WHAT IT PRODUCES
    outputs/outliers/outliers.parquet — post_id, three detector scores,
    and the ensemble flag (True = atypical).

WHY IT MATTERS
    Using multiple detectors and requiring agreement reduces false
    positives.  The result is a conservative set of atypical posts
    that are worth closer inspection.
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
    save_parquet,
    write_manifest,
)
from moltbook_humanlike.outliers import detect_outliers
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect outlier posts")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 4: Detect Outliers ===")

    features = load_parquet("outputs/features/features.parquet")
    id_col = cfg["id_column"]
    logger.info("Loaded features: %d rows x %d cols", *features.shape)

    # Separate ID from numeric features
    ids = features[[id_col]] if id_col in features.columns else None
    feature_cols = features.select_dtypes(include=["number"])

    result = detect_outliers(feature_cols, cfg)

    # Re-attach IDs
    if ids is not None:
        result.insert(0, id_col, ids[id_col].values)

    out_dir = Path("outputs/outliers")
    ensure_dir(out_dir)
    save_parquet(result, out_dir / "outliers.parquet")
    logger.info("Saved outlier results to %s", out_dir / "outliers.parquet")

    write_manifest(out_dir, cfg, {
        "stage": "outliers",
        "n_posts": len(result),
        "n_flagged": int(result["ensemble_flag"].sum()),
        "flag_rate": round(float(result["ensemble_flag"].mean()), 4),
    })
    logger.info("Done.")


if __name__ == "__main__":
    main()

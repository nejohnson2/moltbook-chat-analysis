#!/usr/bin/env python3
"""
Stage 7: Sensitivity analysis across outlier thresholds.

WHAT THIS DOES
    Reruns the ensemble outlier detection at multiple percentile
    thresholds (90th, 95th, 99th) and measures how stable the
    flagged set is.  Reports the number of flagged posts and the
    overlap between consecutive thresholds.

WHAT IT PRODUCES
    outputs/sensitivity/sensitivity_report.csv — threshold, count,
    rate, and overlap with the next-stricter threshold.

WHY IT MATTERS
    If the set of flagged posts changes dramatically with small
    threshold changes, the result is fragile and should be
    interpreted cautiously.  Stability across thresholds increases
    confidence in the findings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moltbook_humanlike.config import load_config, set_seeds
from moltbook_humanlike.io_utils import (
    ensure_dir,
    load_parquet,
    save_csv,
    write_manifest,
)
from moltbook_humanlike.outliers import detect_outliers
from moltbook_humanlike.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg["random_seed"])
    logger = setup_logging()
    logger.info("=== Stage 7: Sensitivity Analysis ===")

    features = load_parquet("outputs/features/features.parquet")
    id_col = cfg["id_column"]
    feature_cols = features.drop(columns=[id_col], errors="ignore").select_dtypes(include=["number"])

    thresholds = cfg.get("outlier_thresholds", [0.90, 0.95, 0.99])
    logger.info("Testing thresholds: %s", thresholds)

    results: list[dict] = []
    prev_flagged: set | None = None

    for thresh in sorted(thresholds):
        run_cfg = dict(cfg)
        run_cfg["_threshold_override"] = thresh

        outlier_df = detect_outliers(feature_cols, run_cfg)
        flagged_idx = set(np.where(outlier_df["ensemble_flag"])[0])
        n_flagged = len(flagged_idx)
        rate = n_flagged / len(outlier_df)

        row = {
            "threshold_percentile": thresh,
            "n_flagged": n_flagged,
            "flag_rate": round(rate, 6),
        }

        if prev_flagged is not None:
            overlap = len(flagged_idx & prev_flagged)
            jaccard = (
                overlap / len(flagged_idx | prev_flagged)
                if len(flagged_idx | prev_flagged) > 0
                else 0.0
            )
            row["overlap_with_prev"] = overlap
            row["jaccard_with_prev"] = round(jaccard, 4)
        else:
            row["overlap_with_prev"] = None
            row["jaccard_with_prev"] = None

        results.append(row)
        prev_flagged = flagged_idx
        logger.info(
            "Threshold %.2f: flagged %d (%.2f%%)",
            thresh, n_flagged, rate * 100,
        )

    report = pd.DataFrame(results)
    out_dir = Path("outputs/sensitivity")
    ensure_dir(out_dir)
    save_csv(report, out_dir / "sensitivity_report.csv")
    logger.info("Saved sensitivity report")

    write_manifest(out_dir, cfg, {"stage": "sensitivity", "thresholds": thresholds})
    logger.info("Done.")


if __name__ == "__main__":
    main()

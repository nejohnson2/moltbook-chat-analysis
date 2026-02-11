"""
Configuration loading and seed management.

What this does:
    Loads pipeline settings from a YAML config file and merges any
    command-line overrides.  Also provides a single function to lock
    all random-number generators so results are reproducible.

Why it matters:
    Reproducibility is essential in research.  By centralizing config
    and seed management, every pipeline stage uses the same settings
    and produces the same results on every run.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Defaults (used when keys are absent from the YAML file) ─────────
DEFAULTS: dict[str, Any] = {
    "random_seed": 42,
    "sample_mode": False,
    "sample_size": 1000,
    "text_column": "content",
    "id_column": "id",
    "category_column": "topic_label",
    "toxicity_column": "toxic_level",
    "submolt_column": "submolt_name",
    "perplexity_model": "meta-llama/Llama-3.2-1B",
    "perplexity_fallback_model": "gpt2-medium",
    "embedding_model": "all-MiniLM-L6-v2",
    "outlier_contamination": 0.05,
    "outlier_thresholds": [0.90, 0.95, 0.99],
    "audit_sample_size": 100,
    "min_text_length": 10,
    "max_text_length": 50000,
}


def load_config(
    path: str | Path = "config.yaml",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config, fill in defaults, and apply CLI overrides."""
    cfg = dict(DEFAULTS)
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update(file_cfg)
    if overrides:
        cfg.update(overrides)
    return cfg


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and (optionally) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except (ImportError, AttributeError):
        pass


def get_device() -> str:
    """Return the best available torch device: mps > cuda > cpu."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"

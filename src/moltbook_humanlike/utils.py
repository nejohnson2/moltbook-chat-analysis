"""
Shared utilities: logging, version capture, git hash.

What this does:
    Provides a consistent logger for every pipeline stage and
    functions to record software versions and the current git
    commit, so each run is traceable.

Why it matters:
    When results change, version info lets you pinpoint whether
    a code change, library update, or data change was responsible.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path


TRACKED_PACKAGES = [
    "datasets",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "transformers",
    "torch",
    "sentence-transformers",
    "matplotlib",
    "seaborn",
    "nltk",
]


def setup_logging(log_dir: str | Path = "logs", name: str = "moltbook") -> logging.Logger:
    """Configure and return a logger that writes to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def capture_versions() -> dict[str, str]:
    """Return installed versions of key packages."""
    versions: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def get_git_hash() -> str | None:
    """Return the current short git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

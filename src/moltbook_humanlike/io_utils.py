"""
I/O helpers for reading and writing pipeline artifacts.

What this does:
    Provides simple, consistent functions for saving and loading
    Parquet files, CSVs, and JSON.  Also writes a run_manifest.json
    that records the exact software versions, config, and timestamp
    used for each pipeline stage.

Why it matters:
    Uniform I/O reduces bugs from inconsistent file handling and
    the manifest makes every run fully auditable.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import capture_versions, get_git_hash


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, **kwargs)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_manifest(
    output_dir: str | Path,
    config: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> None:
    """Write run_manifest.json capturing reproducibility metadata."""
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_hash": get_git_hash(),
        "package_versions": capture_versions(),
        "config": config,
    }
    if extra_info:
        manifest.update(extra_info)
    save_json(manifest, Path(output_dir) / "run_manifest.json")

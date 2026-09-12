"""Portable data-location configuration shared by the project scripts."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    """Return the configured data directory, with a repo-local fallback."""
    configured = os.environ.get("CRISPR_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "data"


DATA_ROOT = _data_root()

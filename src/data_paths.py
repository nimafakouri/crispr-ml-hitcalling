"""Portable data-location configuration shared by the project scripts."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Candidate data folders, checked in order when CRISPR_DATA_ROOT is not set:
#   1. <repo>/data          (repo-local layout)
#   2. <repo>/../data       (current layout: crispr_ai_ml/data next to crispr-ml-hitcalling/)
_DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "data",
    PROJECT_ROOT.parent / "data",
)


def _data_root() -> Path:             # "-> Path" is a type hint saying that this function is expected to return a Path object.
    """Return the data directory (the folder that contains raw/ and processed/).

    Resolution order:
    1. The CRISPR_DATA_ROOT environment variable, if set.
    2. The first existing folder in _DEFAULT_CANDIDATES.
    """
    configured = os.environ.get("CRISPR_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in _DEFAULT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(c) for c in _DEFAULT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the data folder. Set CRISPR_DATA_ROOT or create one of: "
        + searched
    )


DATA_ROOT = _data_root()

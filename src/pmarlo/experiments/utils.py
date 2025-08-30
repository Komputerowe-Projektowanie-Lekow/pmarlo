from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np


def timestamp_dir(base_dir: Union[str, Path]) -> Path:
    """Create and return a unique timestamped directory under base_dir.

    The directory name uses YYYYMMDD-HHMMSS to preserve lexicographic sort order.
    The directory is created if it does not already exist.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def tests_data_dir() -> Path:
    """Return the repository's ``tests/data`` directory robustly.

    - Honors environment override via ``PMARLO_TESTS_DIR`` when set.
    - Searches upward from this file for a parent that contains ``tests/data``.
    - Falls back to current working directory's ``tests/data`` if found.
    """
    # 1) Explicit override for CI or bespoke layouts
    env = os.getenv("PMARLO_TESTS_DIR")
    if env:
        p = Path(env)
        if (p / "3gd8-fixed.pdb").exists() or p.exists():
            return p

    # 2) Walk up parents to locate a repo root that has tests/data
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "tests" / "data"
        if candidate.exists():
            return candidate

    # 3) As a last resort, try CWD/tests/data
    cwd_candidate = Path.cwd() / "tests" / "data"
    if cwd_candidate.exists():
        return cwd_candidate

    # 4) If not found, construct a best-effort path relative to package root
    # (does not assert existence to avoid import-time failures)
    return here.parent.parent.parent / "tests" / "data"


def set_seed(seed: int | None) -> None:
    """Seed Python and NumPy RNGs for experiment reproducibility."""

    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))

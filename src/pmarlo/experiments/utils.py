from __future__ import annotations

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
    """Return the repository's ``tests/data`` directory.

    Resolves the path relative to this source file so it works whether the
    project is run from a source checkout or an installed package.
    """

    return Path(__file__).resolve().parents[3] / "tests" / "data"


def set_seed(seed: int | None) -> None:
    """Seed Python and NumPy RNGs for experiment reproducibility."""

    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from pmarlo.utils.path_utils import ensure_directory


def timestamp_dir(base_dir: Union[str, Path]) -> Path:
    """Create and return a unique timestamped directory under base_dir.

    The directory name uses YYYYMMDD-HHMMSS to preserve lexicographic sort order.
    The directory is created if it does not already exist. If the timestamped
    directory already exists (for example, when two runs start within the same
    second), numeric suffixes are appended until a free path is found.
    """
    base_path = ensure_directory(base_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    candidate = base_path / ts
    if not candidate.exists():
        ensure_directory(candidate)
        return candidate

    suffix = 1
    while True:
        candidate = base_path / f"{ts}-{suffix:02d}"
        if not candidate.exists():
            ensure_directory(candidate)
            return candidate
        suffix += 1


def _validate_assets_dir(path: Path, required_files: Iterable[str]) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected tests asset directory at {path} but it does not exist."
        )
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required test assets: " + ", ".join(missing))
    return path


def tests_data_dir() -> Path:
    """Locate the canonical ``tests/_assets`` directory or raise immediately."""

    required = ("3gd8-fixed.pdb", "traj.dcd")

    env = os.getenv("PMARLO_TESTS_DIR")
    if env:
        return _validate_assets_dir(Path(env), required)

    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "tests" / "_assets"
        if candidate.exists():
            return _validate_assets_dir(candidate, required)

    raise FileNotFoundError(
        "Unable to locate tests/_assets directory. Set PMARLO_TESTS_DIR to the "
        "correct path before importing pmarlo.experiments utilities."
    )


def default_output_root() -> str:
    """Return the default output root, honoring env override.

    If ``PMARLO_OUTPUT_ROOT`` is set, use it; otherwise default to
    ``experiments_output`` for backward compatibility.
    """
    env = os.getenv("PMARLO_OUTPUT_ROOT")
    return env if env and len(env) > 0 else "experiments_output"


def set_seed(seed: int | None) -> None:
    """Seed Python and NumPy RNGs for experiment reproducibility."""

    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))

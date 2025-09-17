from __future__ import annotations

"""
Seeding utilities for deterministic behavior across Python, NumPy, and Torch.

Expose a single entry point `set_global_seed(seed)` used by high‑level run
entrypoints to standardize determinism across runs and processes.
"""

import logging
import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Set global RNG seeds for reproducibility.

    Applies to Python's `random`, NumPy, and PyTorch (if available). Also sets
    `PYTHONHASHSEED` to stabilize hash‑based ordering in the current process.
    Silently ignores libraries that are not installed.
    """
    if seed is None:
        return
    s = int(seed) & 0xFFFFFFFF
    try:
        os.environ["PYTHONHASHSEED"] = str(s)
    except Exception:
        pass
    try:
        random.seed(s)
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore

        _np.random.seed(s)
    except Exception:
        pass
    try:  # optional
        import torch as _torch  # type: ignore

        _torch.manual_seed(s)
        try:
            _torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def quiet_external_loggers(level: int = logging.WARNING) -> None:
    """Lower verbosity from noisy third‑party libraries.

    Intended for import‑time use to keep console output readable. This does not
    alter PMARLO's own loggers.
    """
    noisy = [
        "openmm",
        "mdtraj",
        "mlcolvar",
        "torch",
    ]
    for name in noisy:
        try:
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = False
        except Exception:
            continue

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

import numpy as np
import torch


def set_global_seed(seed: Optional[int]) -> None:
    """Set global RNG seeds for reproducibility.

    Applies to Python's `random`, NumPy, and PyTorch. Also sets
    `PYTHONHASHSEED` to stabilize hash‑based ordering in the current process.
    """
    if seed is None:
        return

    s = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    deterministic_fn = getattr(torch, "use_deterministic_algorithms", None)
    if callable(deterministic_fn):
        deterministic_fn(True)
    else:
        # BUGFIX: Older PyTorch releases (<1.8) lack use_deterministic_algorithms.
        # Fall back to the legacy deterministic toggle and cudnn guards when available
        # so reproducibility is still enforced instead of raising AttributeError.
        legacy_fn = getattr(torch, "set_deterministic", None)
        if callable(legacy_fn):
            legacy_fn(True)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            if hasattr(cudnn, "deterministic"):
                cudnn.deterministic = True
            if hasattr(cudnn, "benchmark"):
                cudnn.benchmark = False


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
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = False

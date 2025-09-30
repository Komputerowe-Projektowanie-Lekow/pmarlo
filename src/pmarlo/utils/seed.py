from __future__ import annotations

"""
Seeding utilities for deterministic behavior across Python, NumPy, and Torch.

Expose a single entry point `set_global_seed(seed)` used by high‑level run
entrypoints to standardize determinism across runs and processes.
"""

import logging
import os
import random
from typing import Callable, Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Set global RNG seeds for reproducibility.

    Applies to Python's `random`, NumPy, and PyTorch (if available). Also sets
    `PYTHONHASHSEED` to stabilize hash‑based ordering in the current process.
    Silently ignores libraries that are not installed.
    """
    if seed is None:
        return
    s = int(seed) & 0xFFFFFFFF
    _safe(lambda: os.environ.__setitem__("PYTHONHASHSEED", str(s)))
    _safe(lambda: random.seed(s))
    _safe(_seed_numpy(s))
    _safe(_seed_torch(s))


def _safe(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        pass


def _seed_numpy(seed: int) -> Callable[[], None]:
    def _inner() -> None:
        import numpy as _np  # type: ignore

        _np.random.seed(seed)

    return _inner


def _seed_torch(seed: int) -> Callable[[], None]:
    def _inner() -> None:
        import torch as _torch  # type: ignore

        _torch.manual_seed(seed)
        try:
            _torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    return _inner


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

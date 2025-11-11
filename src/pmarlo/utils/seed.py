from __future__ import annotations

"""
Seeding utilities for deterministic behavior across Python, NumPy, and Torch.

Expose a single entry point `set_global_seed(seed)` used by high‑level run
entrypoints to standardize determinism across runs and processes.
"""

import logging
import os
import random
from typing import Any, Mapping, Optional

import numpy as np
import torch


def choose_sim_seed(mode: str, *, fixed: Optional[int] = None) -> Optional[int]:
    """Select a simulation seed for deterministic or stochastic modes.

    Parameters
    ----------
    mode:
        One of ``\"none\"``, ``\"fixed\"``, or ``\"auto\"``. Matching is case-insensitive.
    fixed:
        Seed value required when ``mode == \"fixed\"``.
    """
    normalized = mode.strip().lower()
    if normalized == "none":
        return None
    if normalized == "fixed":
        if fixed is None:
            raise ValueError("fixed seed mode requires a `fixed` seed value")
        return int(fixed)
    if normalized == "auto":
        return random.randint(1, 1_000_000)
    raise ValueError(f"Unknown seed mode: {mode}")


def extract_seed(transform_cfg: Mapping[str, Any]) -> int:
    """Extract seed value from nested configuration structures.

    Searches for seed values in a hierarchical configuration dictionary,
    looking in common locations used throughout the PMARLO workflow.

    Parameters
    ----------
    transform_cfg:
        Configuration mapping that may contain a ``seeds`` key with nested
        seed values. Supported keys include: ``analysis``, ``global``,
        ``shuffle``, and ``deeptica``.

    Returns
    -------
    int
        The first valid seed found in the hierarchy, or 2025 as a default.

    Examples
    --------
    >>> extract_seed({"seeds": {"analysis": 42}})
    42
    >>> extract_seed({"seeds": {"global": 100, "shuffle": 200}})
    100
    >>> extract_seed({})
    2025
    """
    seeds = transform_cfg.get("seeds")
    if isinstance(seeds, Mapping):
        for key in ("analysis", "global", "shuffle", "deeptica"):
            if key in seeds:
                try:
                    return int(seeds[key])
                except (TypeError, ValueError):
                    continue
    return 2025


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

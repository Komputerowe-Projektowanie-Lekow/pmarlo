from __future__ import annotations

import random
from typing import Any

import numpy as np

try:  # pragma: no cover - optional ML stack
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional dependency
    torch = None  # type: ignore[assignment]

__all__ = ["set_all_seeds", "safe_float"]


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs deterministically."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    if torch is not None:
        torch.manual_seed(value)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(value)
            except Exception:
                # CUDA may be unavailable or misconfigured; ignore quietly.
                pass


def safe_float(value: Any, default: float = 0.0) -> float:
    """Attempt to convert ``value`` to float, returning ``default`` on failure."""

    try:
        return float(value)
    except Exception:
        return float(default)

from __future__ import annotations

import logging
import random

import numpy as np

try:
    from sklearn.utils import check_random_state
except Exception:  # pragma: no cover - sklearn optional
    check_random_state = None  # type: ignore

logger = logging.getLogger("pmarlo")


def _quiet_external_loggers() -> None:
    """Suppress noisy third-party loggers to INFO level."""
    for name in ("openmm", "mdtraj", "dcdplugin"):
        logging.getLogger(name).setLevel(logging.INFO)


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed common RNGs for reproducible results.

    This seeds :mod:`random` and :mod:`numpy.random` with ``seed`` and, when
    available, initializes scikit-learn's global RNG state via
    :func:`sklearn.utils.check_random_state`.  The returned
    :class:`numpy.random.Generator` can be used for applications requiring a
    dedicated generator.

    Notes
    -----
    Molecular dynamics performed with OpenMM remain stochastic.  Where
    supported, integrators should be seeded explicitly via ``setRandomSeed``.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if check_random_state is not None:
        check_random_state(seed)
    else:  # pragma: no cover - optional dependency
        logger.info("scikit-learn not available; skipping estimator seeding")
    _quiet_external_loggers()
    return np.random.default_rng(seed)

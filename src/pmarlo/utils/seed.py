"""Randomness and logging utilities for reproducible PMARLO runs.

This module provides helpers to seed global random number generators and
silence verbose third-party libraries. Deterministic behaviour is
critical for scientific workflows, hence these utilities centralise the
configuration in one location.
"""

from __future__ import annotations

import contextlib
import io
import logging
import random
from typing import Iterator

import numpy as np
from sklearn.utils import check_random_state

logger = logging.getLogger("pmarlo")


def set_global_seed(seed: int) -> None:
    """Seed common RNG sources for deterministic behaviour.

    Parameters
    ----------
    seed:
        The integer seed to apply globally.

    Notes
    -----
    This function seeds Python's :mod:`random` module, NumPy's legacy
    ``numpy.random`` module and initialises scikit-learn's RNG via
    :func:`sklearn.utils.check_random_state`.

    Molecular dynamics in OpenMM remain stochastic as integrators often
    have their own random streams. If the integrator exposes a
    ``setRandomNumberSeed`` method it should be seeded separately.
    """

    random.seed(seed)
    np.random.seed(seed)
    # Initialise scikit-learn's global RNG helper
    check_random_state(seed)
    logger.info("Global seed set to %d", seed)


def quiet_external_loggers(level: int = logging.INFO) -> None:
    """Reduce stdout noise from verbose third-party libraries.

    Parameters
    ----------
    level:
        Log level to apply to known noisy loggers.
    """

    for name in ("openmm", "mdtraj", "dcdplugin"):
        logging.getLogger(name).setLevel(level)

    # Import mdtraj quietly to silence its plugin banner (e.g. dcdplugin)
    with _suppress_stdout():
        try:  # pragma: no cover - import side effect
            import mdtraj  # type: ignore  # noqa: F401
        except Exception:
            pass


@contextlib.contextmanager
def _suppress_stdout() -> Iterator[None]:
    """Context manager that redirects ``stdout`` to silence plugin banners."""

    old_stdout = io.StringIO()
    with contextlib.redirect_stdout(old_stdout):
        yield

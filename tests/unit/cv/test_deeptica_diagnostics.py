from __future__ import annotations

import numpy as np

from pmarlo.features.diagnostics import diagnose_deeptica_pairs


def _dataset(length: int, k: int = 3):  # noqa: ANN202
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(length, k)).astype(np.float64),
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
    }


def test_diagnostics_reports_no_pairs_when_lag_too_large():
    rep = diagnose_deeptica_pairs(_dataset(4), lag=5)
    assert rep.pairs_total == 0
    assert rep.n_trajectories == 1
    assert "too short for requested lag" in rep.message


def test_diagnostics_reports_pairs_for_reasonable_lag():
    rep = diagnose_deeptica_pairs(_dataset(20), lag=3)
    assert rep.pairs_total == 17
    assert rep.n_trajectories == 1
    assert "Uniform pairs:" in rep.message

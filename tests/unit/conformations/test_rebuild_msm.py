"""Conceptual tests for the `_rebuild_msm` helper."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("deeptime")

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


def _make_dummy_kis(n_states: int) -> KineticImportanceScore:
    """Instantiate KIS with a trivial MSM, regardless of the rebuild inputs."""
    T0 = np.eye(n_states, dtype=float)
    pi0 = np.ones(n_states, dtype=float) / float(n_states)
    return KineticImportanceScore(T0, pi0)


def test_rebuild_msm_two_state_symmetric_chain():
    """Alternating 2-state chain should yield the obvious MSM."""
    kis = _make_dummy_kis(n_states=2)
    dtrajs = [np.array([0, 1, 0, 1, 0, 1, 0], dtype=int)]

    T, pi = kis._rebuild_msm(dtrajs, lag=1)

    assert T.shape == (2, 2)
    assert pi.shape == (2,)

    assert np.all(T >= 0.0)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pi >= 0.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-12)

    T_expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    pi_expected = np.array([0.5, 0.5], dtype=float)

    assert np.allclose(T, T_expected, atol=1e-3)
    assert np.allclose(pi, pi_expected, atol=1e-6)
    assert np.allclose(pi @ T, pi, atol=1e-6)


def test_rebuild_msm_inactive_states_get_zero_pi_and_identity_row():
    """States outside the main connected set should stay neutral."""
    kis = _make_dummy_kis(n_states=3)

    traj01 = np.array([0, 1, 0, 1, 0, 1, 0], dtype=int)
    traj2 = np.array([2, 2, 2, 2, 2], dtype=int)

    T, pi = kis._rebuild_msm([traj01, traj2], lag=1)

    assert T.shape == (3, 3)
    assert pi.shape == (3,)

    assert np.all(T >= 0.0)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pi >= 0.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-12)

    assert np.isclose(pi[2], 0.0, atol=1e-10)
    assert np.allclose(T[2], np.array([0.0, 0.0, 1.0]), atol=1e-10)

    T_sub = T[:2, :2]
    T_sub_expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    assert np.allclose(T_sub, T_sub_expected, atol=1e-3)


def test_rebuild_msm_no_transitions_returns_identity_and_zero_pi():
    """Trajectories with no usable transitions should yield the neutral MSM."""
    kis = _make_dummy_kis(n_states=2)

    dtrajs = [np.array([0], dtype=int), np.array([1], dtype=int)]

    T, pi = kis._rebuild_msm(dtrajs, lag=1)

    assert T.shape == (2, 2)
    assert pi.shape == (2,)

    assert np.allclose(T, np.eye(2, dtype=float))
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(pi, np.zeros(2, dtype=float))


def test_rebuild_msm_lag_consistency_on_deterministic_cycle():
    """Lagged MSMs on a deterministic cycle must be powers of each other."""
    kis = _make_dummy_kis(n_states=3)

    base_cycle = np.array([0, 1, 2], dtype=int)
    traj = np.tile(base_cycle, 100)
    dtrajs = [traj]

    T1, pi1 = kis._rebuild_msm(dtrajs, lag=1)
    T2, pi2 = kis._rebuild_msm(dtrajs, lag=2)

    assert T1.shape == (3, 3)
    assert T2.shape == (3, 3)

    assert np.allclose(T1.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(T2.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pi1 >= 0.0)
    assert np.all(pi2 >= 0.0)
    assert np.isclose(pi1.sum(), 1.0, atol=1e-12)
    assert np.isclose(pi2.sum(), 1.0, atol=1e-12)
    assert np.allclose(pi1 @ T1, pi1, atol=1e-6)
    assert np.allclose(pi2 @ T2, pi2, atol=1e-6)

    assert np.allclose(T2, T1 @ T1, atol=1e-6)


def test_rebuild_msm_raises_on_nonpositive_lag():
    """Lag must be validated before counting transitions."""
    kis = _make_dummy_kis(n_states=2)
    dtrajs = [np.array([0, 1, 0, 1], dtype=int)]

    with pytest.raises(ValueError):
        kis._rebuild_msm(dtrajs, lag=0)

    with pytest.raises(ValueError):
        kis._rebuild_msm(dtrajs, lag=-5)

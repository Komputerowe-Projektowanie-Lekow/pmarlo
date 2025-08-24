import numpy as np

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM
from pmarlo.states.msm_bridge import build_simple_msm


def _build_simple_msm(dtraj, lag_time, mode="sliding"):
    msm = EnhancedMSM(output_dir=".")
    arr = np.asarray(dtraj, dtype=int)
    msm.dtrajs = [arr]
    msm.n_states = int(np.max(arr[arr >= 0])) + 1
    msm.estimator_backend = "pmarlo"
    msm.count_mode = mode
    msm.build_msm(lag_time=lag_time)
    return msm


def _assert_basic_properties(msm):
    T = msm.transition_matrix
    pi = msm.stationary_distribution
    assert T is not None and pi is not None
    assert np.all(T >= 0)
    np.testing.assert_allclose(T.sum(axis=1), 1.0)
    np.testing.assert_allclose(T.T @ pi, pi)
    np.testing.assert_allclose(pi.sum(), 1.0)


def test_sliding_counts_and_stationary():
    dtraj = [0, 1, 0, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    _assert_basic_properties(msm)


def test_strided_counts():
    dtraj = [0, 1, 0, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=2, mode="strided")
    _assert_basic_properties(msm)


def test_negative_states_ignored():
    dtraj = [0, -1, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    _assert_basic_properties(msm)


def test_unvisited_state_yields_stochastic_T():
    """MSM building handles states with zero counts."""
    dtrajs = [np.array([0, 0, 1, 0, 1], dtype=int)]
    T, pi = build_simple_msm(dtrajs, n_states=3, lag=1)
    np.testing.assert_allclose(T.sum(axis=1), 1.0)
    assert pi[2] == 0.0
    assert T[2, 2] == 1.0


def test_implied_timescales_lag_cap():
    dtraj = [0, 1, 0, 1, 0, 1]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    msm.compute_implied_timescales(lag_times=[1, 300], n_timescales=2)
    res = msm.implied_timescales
    assert res is not None
    lags = list(res.lag_times)
    assert all(lag_val <= 5 for lag_val in lags)
    assert res.timescales.shape[0] == len(lags)


def test_implied_timescales_no_valid_lag():
    dtraj = [0, 1]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    msm.compute_implied_timescales(lag_times=[5, 10], n_timescales=2)
    res = msm.implied_timescales
    assert res is not None
    assert res.lag_times.size == 0
    assert res.timescales.size == 0

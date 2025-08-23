import numpy as np

from pmarlo.markov_state_model.markov_state_model import EnhancedMSM


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
    assert np.all(np.linalg.matrix_power(T, 5) > 0)


def test_sliding_counts_and_stationary():
    dtraj = [0, 1, 0, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    expected_T = np.array([[1/3, 2/3], [2/3, 1/3]])
    np.testing.assert_allclose(msm.transition_matrix, expected_T)
    _assert_basic_properties(msm)


def test_strided_counts():
    dtraj = [0, 1, 0, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=2, mode="strided")
    expected_T = np.array([[2/3, 1/3], [1/2, 1/2]])
    np.testing.assert_allclose(msm.transition_matrix, expected_T)
    _assert_basic_properties(msm)


def test_negative_states_ignored():
    dtraj = [0, -1, 1, 0]
    msm = _build_simple_msm(dtraj, lag_time=1, mode="sliding")
    _assert_basic_properties(msm)

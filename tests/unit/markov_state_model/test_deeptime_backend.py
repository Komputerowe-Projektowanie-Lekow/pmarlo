import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

from pmarlo.markov_state_model import MarkovStateModel
from pmarlo.states.msm_bridge import build_simple_msm


def _simulate_chain(
    T: np.ndarray, n_steps: int, rng: np.random.Generator
) -> np.ndarray:
    n_states = T.shape[0]
    traj = np.empty(n_steps + n_states, dtype=int)
    traj[:n_states] = np.arange(n_states)
    for i in range(n_states, n_steps + n_states):
        traj[i] = rng.choice(n_states, p=T[traj[i - 1]])
    return traj


def test_deeptime_backend_matches_reference(tmp_path):
    rng = np.random.default_rng(42)
    T_true = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    dtraj = _simulate_chain(T_true, 5000, rng)

    msm = MarkovStateModel(output_dir=tmp_path)
    msm.dtrajs = [dtraj]
    msm.n_states = T_true.shape[0]
    msm.build_msm(lag_time=1)
    T = msm.transition_matrix
    pi = msm.stationary_distribution
    assert T is not None and pi is not None

    tce = TransitionCountEstimator(lagtime=1, count_mode="sliding", sparse=False)
    count_model = tce.fit([dtraj]).fetch_model()
    ref_model = MaximumLikelihoodMSM(reversible=False).fit(count_model).fetch_model()
    T_ref = np.asarray(ref_model.transition_matrix)
    pi_ref = np.asarray(ref_model.stationary_distribution)

    np.testing.assert_allclose(T, T_ref, atol=1e-6)
    np.testing.assert_allclose(pi, pi_ref, atol=1e-6)


def test_build_simple_msm_agrees_with_class(tmp_path):
    rng = np.random.default_rng(7)
    T_true = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    dtraj = _simulate_chain(T_true, 4000, rng)

    msm = MarkovStateModel(output_dir=tmp_path)
    msm.dtrajs = [dtraj]
    msm.n_states = T_true.shape[0]
    msm.build_msm(lag_time=1)

    T_func, pi_func = build_simple_msm([dtraj], n_states=T_true.shape[0], lag=1)

    np.testing.assert_allclose(msm.transition_matrix, T_func, atol=1e-12)
    np.testing.assert_allclose(msm.stationary_distribution, pi_func, atol=1e-12)


def test_multiple_trajectories_equal_concatenated():
    rng = np.random.default_rng(11)
    T_true = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )
    full = _simulate_chain(T_true, 20000, rng)
    split = len(full) // 2
    d1, d2 = full[:split], full[split:]

    T_multi, pi_multi = build_simple_msm([d1, d2], n_states=T_true.shape[0], lag=1)
    T_single, pi_single = build_simple_msm([full], n_states=T_true.shape[0], lag=1)

    np.testing.assert_allclose(T_multi, T_single, atol=5e-4)
    np.testing.assert_allclose(pi_multi, pi_single, atol=5e-4)


def test_lag_two_matches_matrix_square():
    rng = np.random.default_rng(23)
    T_true = np.array(
        [
            [0.75, 0.2, 0.05],
            [0.15, 0.7, 0.15],
            [0.1, 0.25, 0.65],
        ]
    )
    dtraj = _simulate_chain(T_true, 10000, rng)

    T_lag1, _ = build_simple_msm([dtraj], n_states=3, lag=1)
    T_lag2, _ = build_simple_msm([dtraj], n_states=3, lag=2)

    np.testing.assert_allclose(T_lag2, np.linalg.matrix_power(T_lag1, 2), atol=2e-2)

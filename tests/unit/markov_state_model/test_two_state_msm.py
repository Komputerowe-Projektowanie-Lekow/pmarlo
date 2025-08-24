import numpy as np

from pmarlo.markov_state_model import MarkovStateModel


def test_two_state_msm_recovers_timescale(tmp_path):
    rng = np.random.default_rng(0)
    T = np.array([[0.9, 0.1], [0.1, 0.9]])
    n_steps = 100000
    states = np.zeros(n_steps + 1, dtype=int)
    for i in range(n_steps):
        states[i + 1] = rng.choice(2, p=T[states[i]])
    msm = MarkovStateModel(output_dir=tmp_path, random_state=0)
    msm.dtrajs = [states]
    msm.count_mode = "sliding"
    msm.compute_implied_timescales(lag_times=[1], n_timescales=1, n_samples=200)
    res = msm.implied_timescales
    assert res is not None
    estimated = res.timescales[0, 0]
    expected = -1.0 / np.log(0.8)
    assert np.isfinite(estimated)
    assert abs(estimated - expected) / expected < 0.1

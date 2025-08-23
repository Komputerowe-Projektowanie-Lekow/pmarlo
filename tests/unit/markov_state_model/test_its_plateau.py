import numpy as np

from pmarlo.markov_state_model.markov_state_model import EnhancedMSM


def test_plateau_detection_recovers_known_timescale():
    rng = np.random.default_rng(0)
    # Symmetric two-state MSM with p=0.05 transition probability
    T = np.array([[0.95, 0.05], [0.05, 0.95]])
    n_steps = 10000
    states = np.zeros(n_steps, dtype=int)
    for i in range(1, n_steps):
        states[i] = rng.choice(2, p=T[states[i - 1]])

    msm = EnhancedMSM(output_dir=".")
    msm.dtrajs = [states]
    msm.n_states = 2
    msm.count_mode = "sliding"
    msm.build_msm(lag_time=1)
    msm.compute_implied_timescales(
        lag_times=[1, 2, 5, 10, 20, 30],
        n_timescales=1,
        n_samples=50,
        plateau_m=1,
        plateau_epsilon=0.2,
    )
    res = msm.implied_timescales
    assert res is not None
    assert res.recommended_lag_window is not None
    start, end = res.recommended_lag_window
    lags = res.lag_times
    mask = (lags >= start) & (lags <= end)
    ts = res.timescales[mask, 0]
    assert np.nanmax(ts) - np.nanmin(ts) <= 0.1 * np.nanmean(ts)
    # theoretical timescale for p=0.05
    t_true = -1.0 / np.log(0.9)
    assert abs(np.nanmean(ts) - t_true) / t_true < 0.2
    # rates should not collapse to zero
    assert res.rates[mask][-1] > 0.01

import numpy as np

from pmarlo.markov_state_model.utils import format_lag_window_ps


def test_format_lag_window_ps():
    dt = 0.002
    window_steps = (3, 5)
    window_ps = tuple(np.array(window_steps) * dt)
    assert format_lag_window_ps(window_ps) == "0.006–0.010 ps"

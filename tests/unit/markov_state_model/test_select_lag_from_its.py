import numpy as np

from pmarlo.utils.msm_utils import select_lag_from_its


def test_select_lag_detects_first_stable_plateau_region():
    lag_times = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    timescales = np.array(
        [
            [10.0],
            [8.0],
            [6.0],
            [6.1],
            [6.05],
            [6.0],
        ],
        dtype=float,
    )

    lag = select_lag_from_its(
        lag_times,
        timescales,
        min_lag_idx=2,
        plateau_threshold=0.05,
    )

    assert lag == 4


def test_select_lag_ignores_invalid_entries_when_searching_plateau():
    lag_times = np.array([1, 2, 3, 4], dtype=int)
    timescales = np.array(
        [
            [np.nan],
            [np.inf],
            [5.0],
            [7.0],
        ],
        dtype=float,
    )

    lag = select_lag_from_its(
        lag_times,
        timescales,
        min_lag_idx=1,
        plateau_threshold=0.01,
    )

    assert lag == 4

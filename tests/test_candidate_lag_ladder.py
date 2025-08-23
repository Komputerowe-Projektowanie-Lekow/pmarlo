from __future__ import annotations

import pytest

from pmarlo.utils.msm_utils import candidate_lag_ladder


def test_default_ladder_includes_range():
    lags = candidate_lag_ladder()
    assert lags[0] == 1
    assert lags[-1] == 200


def test_filtered_range():
    lags = candidate_lag_ladder(10, 50)
    assert lags[0] == 10
    assert lags[-1] == 50
    assert lags == sorted(lags)


def test_downsample_keeps_endpoints():
    lags = candidate_lag_ladder(1, 100, n_candidates=3)
    assert len(lags) == 3
    assert lags[0] == 1
    assert lags[-1] == 100


@pytest.mark.parametrize("min_lag,max_lag", [(0, 10), (10, 5)])
def test_invalid_range_raises(min_lag, max_lag):
    with pytest.raises(ValueError):
        candidate_lag_ladder(min_lag, max_lag)


def test_invalid_n_candidates():
    with pytest.raises(ValueError):
        candidate_lag_ladder(1, 10, n_candidates=0)


def test_outside_predefined_range_returns_bounds():
    lags = candidate_lag_ladder(5000, 6000)
    assert lags == [5000, 6000]

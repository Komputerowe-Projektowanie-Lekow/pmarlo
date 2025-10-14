import pytest

from src.pmarlo.analysis.diagnostics import derive_taus


def test_derive_taus_geometric_basic():
    taus = derive_taus([200, 150])
    assert taus, "Expected non-empty taus"
    assert all(isinstance(t, int) and t > 0 for t in taus)
    assert taus == sorted(set(taus)), "Taus must be strictly increasing and unique"
    assert max(taus) < 150


def test_derive_taus_geometric_fraction_and_bounds():
    taus = derive_taus([60, 80], fraction_max=0.3, max_lags=5)
    assert max(taus) <= int(60 * 0.3)


def test_derive_taus_non_geometric_base():
    taus = derive_taus([120], geometric=False, base=[2, 5, 10, 20, 40, 80, 160])
    assert taus == [2, 5, 10, 20, 40, 80]


def test_derive_taus_errors_on_small_min_length():
    with pytest.raises(ValueError):
        derive_taus([2], min_lag=2)  # min_len <= min_lag


def test_derive_taus_errors_on_invalid_fraction():
    with pytest.raises(ValueError):
        derive_taus([50], fraction_max=0.0)


def test_derive_taus_errors_on_empty_lengths():
    with pytest.raises(ValueError):
        derive_taus([], geometric=False, base=[2, 5])


def test_derive_taus_errors_on_non_geometric_without_base():
    with pytest.raises(ValueError):
        derive_taus([100], geometric=False, base=None)

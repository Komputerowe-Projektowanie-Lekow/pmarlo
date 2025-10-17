import sys

import numpy as np
import pytest

from pmarlo.markov_state_model import reduction as reduction_module
from pmarlo.markov_state_model.reduction import (
    pca_reduce,
    tica_reduce,
    vamp_reduce,
)


def test_pca_reduce_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ref = PCA(n_components=2).fit_transform(Xs)
    ours = pca_reduce(X, n_components=2, scale=True)
    signs = np.sign((ours * ref).sum(axis=0))
    ours *= signs
    assert np.allclose(ours, ref, atol=1e-6)


def test_pca_reduce_large_batch_equals_small():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 10))
    small = pca_reduce(X, n_components=3, scale=True)
    large = pca_reduce(X, n_components=3, scale=True, batch_size=500)
    cov_small = np.cov(small, rowvar=False)
    cov_large = np.cov(large, rowvar=False)
    assert np.allclose(cov_large, cov_small, atol=1e-1)


def test_nan_safe_pca():
    X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    Y = pca_reduce(X, n_components=2, scale=True)
    assert Y.shape == (2, 2)
    assert np.isfinite(Y).all()


def test_tica_reduce_matches_deeptime():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6))
    ours = tica_reduce(X, lag=2, n_components=2, scale=True)
    # Skip if deeptime's TICA is unavailable
    try:
        from deeptime.decomposition import TICA  # type: ignore
    except Exception:
        pytest.skip("deeptime TICA not available")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = TICA(lagtime=2, dim=2).fit([Xs]).fetch_model()
    ref = model.transform([Xs])[0]
    signs = np.sign((ours * ref).sum(axis=0))
    ours *= signs
    assert np.allclose(ours, ref, atol=1e-6)


def test_tica_reduce_nan_handling(monkeypatch):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 6))
    X[10, 0] = np.nan
    # force fallback implementation
    monkeypatch.setitem(sys.modules, "deeptime", None)
    monkeypatch.setitem(sys.modules, "pyemma", None)
    result = tica_reduce(X, lag=3, n_components=2, scale=True)
    assert np.isfinite(result).all()


def test_manual_tica_uses_numpy_cov(monkeypatch):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 5))
    monkeypatch.setitem(sys.modules, "deeptime", None)
    monkeypatch.setitem(sys.modules, "pyemma", None)
    monkeypatch.setattr(reduction_module, "scipy_eigh", None)

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_cov = reduction_module.np.cov

    def spy_cov(*args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return original_cov(*args, **kwargs)

    monkeypatch.setattr(reduction_module.np, "cov", spy_cov)
    _ = tica_reduce(X, lag=2, n_components=2, scale=False)
    assert calls, "expected numpy.cov to be invoked during manual TICA fallback"


def test_manual_vamp_uses_numpy_cov(monkeypatch):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(220, 4))
    monkeypatch.setitem(sys.modules, "deeptime", None)
    monkeypatch.setitem(sys.modules, "pyemma", None)

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_cov = reduction_module.np.cov

    def spy_cov(*args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return original_cov(*args, **kwargs)

    monkeypatch.setattr(reduction_module.np, "cov", spy_cov)
    _ = vamp_reduce(X, lag=2, n_components=2, scale=False)
    assert calls, "expected numpy.cov to be invoked during manual VAMP fallback"


def test_vamp_reduce_nan_handling(monkeypatch):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(400, 5))
    X[5, 2] = np.nan
    monkeypatch.setitem(sys.modules, "deeptime", None)
    monkeypatch.setitem(sys.modules, "pyemma", None)
    result = vamp_reduce(X, lag=4, n_components=2, scale=True)
    assert np.isfinite(result).all()

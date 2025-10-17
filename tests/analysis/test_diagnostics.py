import importlib
import pathlib
import sys
import types

import numpy as np
import pytest


def _load_compute_diagnostics():
    for name in list(sys.modules):
        if name.startswith("pmarlo.analysis") or name.startswith("pmarlo.ml"):
            sys.modules.pop(name)
    base = pathlib.Path("src/pmarlo")
    pkg = types.ModuleType("pmarlo")
    pkg.__path__ = [str(base)]
    sys.modules["pmarlo"] = pkg
    ml_pkg = types.ModuleType("pmarlo.ml")
    ml_pkg.__path__ = []
    sys.modules["pmarlo.ml"] = ml_pkg
    deeptica_pkg = types.ModuleType("pmarlo.ml.deeptica")
    deeptica_pkg.__path__ = []
    sys.modules["pmarlo.ml.deeptica"] = deeptica_pkg
    whitening_mod = types.ModuleType("pmarlo.ml.deeptica.whitening")

    def _identity(values, mean=None, transform=None, already_applied=None):
        return np.asarray(values, dtype=np.float64)

    whitening_mod.apply_output_transform = _identity
    sys.modules["pmarlo.ml.deeptica.whitening"] = whitening_mod
    importlib.invalidate_caches()
    diagnostics_mod = importlib.import_module("pmarlo.analysis.diagnostics")
    compute = diagnostics_mod.compute_diagnostics
    for name in (
        "pmarlo.ml.deeptica.whitening",
        "pmarlo.ml.deeptica",
        "pmarlo.ml",
        "pmarlo.analysis.diagnostics",
        "pmarlo.analysis",
        "pmarlo",
    ):
        sys.modules.pop(name, None)
    return compute


def _load_compute_and_exc():
    """Return (compute_diagnostics, InsufficientSamplesError) with same isolation logic."""
    for name in list(sys.modules):
        if name.startswith("pmarlo.analysis") or name.startswith("pmarlo.ml"):
            sys.modules.pop(name)
    base = pathlib.Path("src/pmarlo")
    pkg = types.ModuleType("pmarlo")
    pkg.__path__ = [str(base)]
    sys.modules["pmarlo"] = pkg
    ml_pkg = types.ModuleType("pmarlo.ml")
    ml_pkg.__path__ = []
    sys.modules["pmarlo.ml"] = ml_pkg
    deeptica_pkg = types.ModuleType("pmarlo.ml.deeptica")
    deeptica_pkg.__path__ = []
    sys.modules["pmarlo.ml.deeptica"] = deeptica_pkg
    whitening_mod = types.ModuleType("pmarlo.ml.deeptica.whitening")

    def _identity(values, mean=None, transform=None, already_applied=None):
        return np.asarray(values, dtype=np.float64)

    whitening_mod.apply_output_transform = _identity
    sys.modules["pmarlo.ml.deeptica.whitening"] = whitening_mod
    importlib.invalidate_caches()
    diagnostics_mod = importlib.import_module("pmarlo.analysis.diagnostics")
    compute = diagnostics_mod.compute_diagnostics
    exc_cls = diagnostics_mod.InsufficientSamplesError
    for name in (
        "pmarlo.ml.deeptica.whitening",
        "pmarlo.ml.deeptica",
        "pmarlo.ml",
        "pmarlo.analysis.diagnostics",
        "pmarlo.analysis",
        "pmarlo",
    ):
        sys.modules.pop(name, None)
    return compute, exc_cls


def test_compute_diagnostics_triviality_and_mass_warning():
    X = np.random.default_rng(0).normal(size=(200, 2))
    dataset = {"splits": {"train": {"X": X, "inputs": X.copy()}}}
    compute_diagnostics = _load_compute_diagnostics()
    result = compute_diagnostics(dataset, diag_mass=0.99)

    canon = result.get("canonical_correlation", {})
    assert (
        "train" in canon and canon["train"]
    ), "expected canonical correlations for train split"
    warnings = result.get("warnings", [])
    assert any("reparametrize" in w for w in warnings)
    assert any("MSM diagonal mass" in w for w in warnings)


def test_compute_diagnostics_handles_missing_inputs():
    rng = np.random.default_rng(2)
    dataset = {"splits": {"train": {"X": rng.normal(size=(40, 2))}}}
    compute_diagnostics = _load_compute_diagnostics()
    result = compute_diagnostics(dataset, diag_mass=0.1)

    assert result.get("canonical_correlation", {}) == {}
    assert "train" in result.get("autocorrelation", {})


def test_canonical_correlation_raises_on_insufficient_samples():
    # Only one paired sample -> should raise InsufficientSamplesError now.
    X = np.random.default_rng(123).normal(size=(1, 3))
    dataset = {"splits": {"tiny": {"X": X, "inputs": X.copy()}}}
    compute_diagnostics, InsufficientSamplesError = _load_compute_and_exc()
    with pytest.raises(InsufficientSamplesError):
        compute_diagnostics(dataset)


def test_covariance_uses_numpy_cov(monkeypatch):
    import pmarlo.analysis.diagnostics as diagnostics_module

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_cov = diagnostics_module.np.cov

    def spy_cov(*args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return original_cov(*args, **kwargs)

    monkeypatch.setattr(diagnostics_module.np, "cov", spy_cov)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    centered = X - np.mean(X, axis=0, keepdims=True)
    result = diagnostics_module._covariance(centered, centered.shape[0])
    assert calls, "expected numpy.cov to be used inside _covariance"
    expected = original_cov(centered, rowvar=False, ddof=1)
    assert np.allclose(result, expected)


def test_canonical_correlation_uses_sklearn(monkeypatch):
    import pmarlo.analysis.diagnostics as diagnostics_module

    class DummyCCA:
        def __init__(self, *, n_components: int, scale: bool, max_iter: int):
            self.params = {
                "n_components": n_components,
                "scale": scale,
                "max_iter": max_iter,
            }
            self._called = False

        def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self._called = True
            return X, Y

    created: list[DummyCCA] = []

    def factory(*, n_components: int, scale: bool, max_iter: int) -> DummyCCA:
        inst = DummyCCA(n_components=n_components, scale=scale, max_iter=max_iter)
        created.append(inst)
        return inst

    monkeypatch.setattr(diagnostics_module, "CCA", factory)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 2))
    correlations = diagnostics_module._canonical_correlations(X, Y)
    assert correlations, "expected non-empty correlations from patched CCA"
    assert created, "expected custom CCA factory to be invoked"
    assert all(inst._called for inst in created), "expected fit_transform to run"

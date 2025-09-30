import importlib
import pathlib
import sys
import types

import numpy as np


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

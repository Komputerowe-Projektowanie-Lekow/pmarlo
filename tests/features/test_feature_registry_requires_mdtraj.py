import importlib
import sys

import pytest


def test_get_feature_requires_mdtraj(monkeypatch):
    # Ensure we import a fresh copy so the lazy import logic runs inside the test.
    if "pmarlo.features" in sys.modules:
        del sys.modules["pmarlo.features"]
    module = importlib.import_module("pmarlo.features")

    # Sanity check: mdtraj is intentionally absent in the test environment.
    assert "mdtraj" not in sys.modules

    with pytest.raises(ModuleNotFoundError) as excinfo:
        module.get_feature("phi_psi")

    message = str(excinfo.value)
    assert "mdtraj" in message
    assert "Built-in molecular features" in message

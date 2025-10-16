import math

import pytest

from pmarlo import constants as const
from pmarlo.utils import thermodynamics


def test_kT_kJ_per_mol_matches_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy_constants = pytest.importorskip("scipy.constants")
    monkeypatch.setattr(thermodynamics, "_scipy_constants", scipy_constants, raising=False)
    temperature = 310.0
    expected = float(
        scipy_constants.k * temperature * scipy_constants.Avogadro / 1000.0
    )
    result = thermodynamics.kT_kJ_per_mol(temperature)
    assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=0.0)


def test_kT_kJ_per_mol_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thermodynamics, "_scipy_constants", None, raising=False)
    temperature = 310.0
    expected = float(
        const.BOLTZMANN_CONSTANT_J_PER_K
        * temperature
        * const.AVOGADRO_NUMBER
        / 1000.0
    )
    result = thermodynamics.kT_kJ_per_mol(temperature)
    assert math.isclose(result, expected, rel_tol=0.0, abs_tol=1e-12)

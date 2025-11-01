import math

from scipy import constants as scipy_constants

from pmarlo.utils import thermodynamics


def test_kT_kJ_per_mol_uses_scipy_constants() -> None:
    """Verify kT calculation uses scipy.constants for accurate physical constants."""
    temperature = 310.0
    expected = float(
        scipy_constants.k * temperature * scipy_constants.Avogadro / 1000.0
    )
    result = thermodynamics.kT_kJ_per_mol(temperature)
    assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=0.0)


def test_kT_kJ_per_mol_at_standard_temperature() -> None:
    """Verify kT at standard temperature (298.15 K) is ~2.48 kJ/mol."""
    result = thermodynamics.kT_kJ_per_mol(298.15)
    # kT at 298.15 K â‰ˆ 2.479 kJ/mol
    assert 2.47 < result < 2.49

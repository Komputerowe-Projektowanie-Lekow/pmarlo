"""Thermodynamic helper functions used across the codebase."""

from __future__ import annotations

from pmarlo import constants as const

try:  # SciPy may be unavailable in lightweight environments
    from scipy import constants as _scipy_constants  # type: ignore
except Exception:  # pragma: no cover - fallback is exercised in tests
    _scipy_constants = None


def kT_kJ_per_mol(temperature_kelvin: float) -> float:
    """Return the thermal energy ``kT`` in kJ/mol at the given temperature.

    Parameters
    ----------
    temperature_kelvin:
        Absolute temperature in Kelvin.
    """
    temperature = float(temperature_kelvin)
    if _scipy_constants is not None:
        return float(
            _scipy_constants.k * temperature * _scipy_constants.Avogadro / 1000.0
        )
    return float(
        const.BOLTZMANN_CONSTANT_J_PER_K * temperature * const.AVOGADRO_NUMBER / 1000.0
    )


__all__ = ["kT_kJ_per_mol"]

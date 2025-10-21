"""Thermodynamic helper functions used across the codebase."""

from __future__ import annotations

from scipy import constants as scipy_constants


def kT_kJ_per_mol(temperature_kelvin: float) -> float:
    """Return the thermal energy ``kT`` in kJ/mol at the given temperature.

    Uses scipy.constants for accurate physical constants.

    Parameters
    ----------
    temperature_kelvin:
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Thermal energy in kJ/mol.
    """
    temperature = float(temperature_kelvin)
    return float(scipy_constants.k * temperature * scipy_constants.Avogadro / 1000.0)


__all__ = ["kT_kJ_per_mol"]

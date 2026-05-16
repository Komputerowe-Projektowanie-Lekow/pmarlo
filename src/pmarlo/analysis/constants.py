"""Physical and numerical constants for the analysis submodule."""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# FES quality thresholds
# ---------------------------------------------------------------------------

FES_MAX_ENERGY_KT: Final[float] = 100.0
"""Bins above this free-energy value in kT are treated as empty."""

FES_EMPTY_BIN_RATIO_HIGH: Final[float] = 0.5
"""Empty-bin fraction that triggers a high-severity FES quality warning."""

FES_EMPTY_BIN_RATIO_LOW: Final[float] = 0.1
"""Empty-bin fraction that triggers a low-severity FES quality warning."""

FES_MIN_ENERGY_RANGE_KT: Final[float] = 1.0
"""Minimum useful FES energy range in kT before sampling is suspicious."""

# ---------------------------------------------------------------------------
# REMD sampling thresholds
# ---------------------------------------------------------------------------

REMD_MIN_TEMPERATURE_RANGE_K: Final[float] = 100.0
"""Minimum temperature span in K expected for useful REMD sampling."""

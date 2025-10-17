"""Centralised repository of project-wide constants.

Keeping these values in one place improves numerical stability and makes it
easier to evolve scientific defaults without hunting for magic numbers spread
throughout the codebase.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BOLTZMANN_CONSTANT_KJ_PER_MOL: Final[float] = 0.00831446261815324
"""Boltzmann constant expressed in kJ mol^-1 K^-1."""

BOLTZMANN_CONSTANT_J_PER_K: Final[float] = 1.380649e-23
"""Boltzmann constant expressed in J K⁻¹."""

AVOGADRO_NUMBER: Final[float] = 6.02214076e23
"""Avogadro's number (mol^-1)."""

# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------

NUMERIC_MIN_POSITIVE: Final[float] = 1e-12
"""Generic tiny positive epsilon used to avoid division by zero and log(0)."""

NUMERIC_MIN_POSITIVE_STRICT: Final[float] = 1e-15
"""Stronger epsilon for clipping operations that require extra guard rails."""

NUMERIC_ABSOLUTE_TOLERANCE: Final[float] = 1e-6
"""Default absolute tolerance for floating point comparisons."""

NUMERIC_RELATIVE_TOLERANCE: Final[float] = 1e-8
"""Default relative tolerance used in iterative algorithms."""

NUMERIC_PROGRESS_MIN_FRACTION: Final[float] = 1e-9
"""Lower bound for progress tracking ratios to avoid division by zero."""

NUMERIC_DIRICHLET_ALPHA: Final[float] = 1e-3
"""Default Dirichlet prior strength used in MSM estimators."""

NUMERIC_JITTER: Final[float] = 1e-8
"""Default jitter added to matrices to improve conditioning."""

NUMERIC_EXP_CLIP_MIN: Final[float] = -700.0
"""Lower bound for exponent arguments when exponentiating log weights."""

NUMERIC_EXP_CLIP_MAX: Final[float] = 700.0
"""Upper bound for exponent arguments when exponentiating log weights."""

NUMERIC_MIN_EIGEN_CLIP: Final[float] = 1e-8
"""Minimum eigenvalue used when clipping spectra to remain positive."""

NUMERIC_RIDGE_SCALE: Final[float] = 1e-5
"""Scaling factor for ridge regularisation in DeeptiCA models."""

NUMERIC_RARE_EVENT_EPSILON: Final[float] = 1e-6
"""Floor used when guarding rare events or denominators."""

NUMERIC_SOFT_ENERGY_LIMIT: Final[float] = 1e5
"""Soft guard threshold for suspicious energy magnitudes."""

NUMERIC_HARD_ENERGY_LIMIT: Final[float] = 1e6
"""Hard guard threshold for energy magnitudes that indicate corrupted data."""

NUMERIC_MIN_RATE: Final[float] = NUMERIC_ABSOLUTE_TOLERANCE
"""Lower bound when clamping rates or probabilities."""

NUMERIC_MAX_RATE: Final[float] = 1.0 - NUMERIC_MIN_RATE
"""Upper bound when clamping rates or probabilities."""

# ---------------------------------------------------------------------------
# Domain-specific defaults
# ---------------------------------------------------------------------------

DEEPTICA_DEFAULT_LEARNING_RATE: Final[float] = 3e-4
"""Default learning rate for DeeptiCA training loops."""

DEEPTICA_DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
"""Default weight decay coefficient for DeeptiCA optimisers."""

DEEPTICA_DEFAULT_VAMP_EPS: Final[float] = 1e-3
"""Default VAMP epsilon used across DeeptiCA losses."""

DEEPTICA_DEFAULT_VAMP_EPS_ABS: Final[float] = NUMERIC_ABSOLUTE_TOLERANCE
"""Absolute epsilon companion to VAMP epsilon."""

DEEPTICA_DEFAULT_VAMP_COND_REG: Final[float] = 1e-4
"""Default conditioning regulariser for VAMP objectives."""

DEEPTICA_DEFAULT_EIGEN_FLOOR: Final[float] = 1e-4
"""Nominal eigenvalue floor for DeeptiCA covariance estimates."""

DEEPTICA_DEFAULT_VARIANCE_WARN_THRESHOLD: Final[float] = NUMERIC_ABSOLUTE_TOLERANCE
"""Variance threshold beyond which DeeptiCA raises warnings."""

DEEPTICA_DEFAULT_JITTER: Final[float] = NUMERIC_JITTER
"""Jitter injected into DeeptiCA matrices to avoid singularities."""

DEEPTICA_DEFAULT_TRACE_FLOOR: Final[float] = NUMERIC_MIN_POSITIVE
"""Trace floor applied during DeeptiCA whitening."""

DEEPTICA_MIN_BATCH_FRACTION: Final[float] = 1e-3
"""Minimum fraction for DeeptiCA batch updates."""

DEEPTICA_MAX_BATCH_FRACTION: Final[float] = 0.9
"""Maximum fraction for DeeptiCA batch updates."""

DEEPTICA_MIN_VAMP_EPS: Final[float] = 1e-9
"""Lower bound for VAMP epsilon in DeeptiCA configurations."""

DEEPTICA_MIN_EIGEN_CLIP: Final[float] = NUMERIC_MIN_EIGEN_CLIP
"""Alias for DeeptiCA eigenvalue clipping floor."""

DEEPTICA_CONDITION_NUMBER_WARN: Final[float] = 1e6
"""Condition number threshold beyond which DeeptiCA raises warnings."""

# ---------------------------------------------------------------------------
# Replica-exchange configuration defaults
# ---------------------------------------------------------------------------

REPLICA_EXCHANGE_EWALD_TOLERANCE: Final[float] = 1e-4
"""Default OpenMM Ewald error tolerance for replica-exchange simulations."""

# ---------------------------------------------------------------------------
# MSM visualisation defaults
# ---------------------------------------------------------------------------

MSM_RATE_DISPLAY_SCALE: Final[float] = 1e-3
"""Scale factor used when annotating MSM rate plots (kHz representation)."""

MSM_RATE_INVERSE_SCALE: Final[float] = 1e3
"""Inverse of MSM rate scale to convert seconds^-1 to milliseconds^-1."""

# ---------------------------------------------------------------------------
# Metadata / schema versions
# ---------------------------------------------------------------------------

SHARD_INDEX_VERSION: Final[int] = 2
"""Current version of the shard index schema."""

SHARD_SCHEMA_VERSION: Final[str] = "2.0"
"""Version of the on-disk shard schema definition."""

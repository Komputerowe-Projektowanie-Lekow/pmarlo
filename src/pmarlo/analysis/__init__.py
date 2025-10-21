"""Utilities for downstream analysis of learned collective variables."""

from .counting import expected_pairs
from .debug_export import (
    AnalysisDebugData,
    compute_analysis_debug,
    export_analysis_debug,
)
from .diagnostics import compute_diagnostics
from .errors import CountingLogicError
from .fes import compute_weighted_fes, ensure_fes_inputs_whitened
from .msm import ensure_msm_inputs_whitened, prepare_msm_discretization
from .project_cv import apply_whitening_from_metadata
from .validation import ValidationError, validate_features

__all__ = [
    "AnalysisDebugData",
    "compute_analysis_debug",
    "export_analysis_debug",
    "apply_whitening_from_metadata",
    "ensure_msm_inputs_whitened",
    "prepare_msm_discretization",
    "ensure_fes_inputs_whitened",
    "compute_weighted_fes",
    "compute_diagnostics",
    "expected_pairs",
    "validate_features",
    "ValidationError",
    "CountingLogicError",
]

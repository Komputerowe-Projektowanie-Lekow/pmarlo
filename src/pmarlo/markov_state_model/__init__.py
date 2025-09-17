# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Markov State Model module for PMARLO.

Provides enhanced MSM analysis with TRAM/dTRAM and comprehensive reporting.
"""

from .ck_runner import CKRunResult, run_ck
from .enhanced_msm import EnhancedMSM as MarkovStateModel
from .enhanced_msm import run_complete_msm_analysis
from .free_energy import (
    FESResult,
    PMFResult,
    generate_1d_pmf,
    generate_2d_fes,
    periodic_kde_2d,
)
from .reduction import (
    get_available_methods,
    pca_reduce,
    reduce_features,
    tica_reduce,
    vamp_reduce,
)
from .results import (
    BaseResult,
    CKResult,
    ClusteringResult,
    DemuxResult,
    ITSResult,
    MSMResult,
    REMDResult,
)
from .reweighter import Reweighter
from .msm_builder import MSMBuilder, MSMResult as BuilderMSMResult

__all__ = [
    "MarkovStateModel",
    "run_complete_msm_analysis",
    "run_ck",
    "CKRunResult",
    "FESResult",
    "PMFResult",
    "generate_1d_pmf",
    "generate_2d_fes",
    "periodic_kde_2d",
    "pca_reduce",
    "tica_reduce",
    "vamp_reduce",
    "reduce_features",
    "get_available_methods",
    # Result classes
    "BaseResult",
    "REMDResult",
    "DemuxResult",
    "ClusteringResult",
    "MSMResult",
    "CKResult",
    "ITSResult",
    # Facades
    "Reweighter",
    "MSMBuilder",
    "BuilderMSMResult",
]

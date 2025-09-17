"""Feature (CV) layer: registry and built-in features.

Phase A: minimal registry with phi/psi built-in to keep backward compatibility.
"""

# Import built-ins to trigger registration
from . import builtins as _builtins  # noqa: F401
from .base import FEATURE_REGISTRY, get_feature, register_feature  # noqa: F401

# Collective variables and DeepTICA functionality
from .collective_variables import CVModel  # noqa: F401
from .data_loaders import LaggedPairs, make_loaders  # noqa: F401
from .deeptica import DeepTICAConfig, DeepTICAModel, train_deeptica  # noqa: F401
from .diagnostics import (  # noqa: F401
    PairDiagItem,
    PairDiagReport,
    diagnose_deeptica_pairs,
)
from .pairs import make_training_pairs_from_shards, scaled_time_pairs  # noqa: F401
from .ramachandran import (  # noqa: F401
    RamachandranResult,
    compute_ramachandran,
    compute_ramachandran_fes,
    periodic_hist2d,
)

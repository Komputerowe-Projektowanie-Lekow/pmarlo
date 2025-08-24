"""Feature (CV) layer: registry and built-in features.

Phase A: minimal registry with phi/psi built-in to keep backward compatibility.
"""

# Import built-ins to trigger registration
from . import builtins as _builtins  # noqa: F401
from .base import FEATURE_REGISTRY, get_feature, register_feature  # noqa: F401

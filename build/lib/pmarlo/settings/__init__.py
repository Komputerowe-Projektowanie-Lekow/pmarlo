from __future__ import annotations

"""
Runtime configuration helpers for PMARLO.

This module exposes accessors for bias configuration defaults and the canonical
feature specification used by the TorchScript CV bias module.
"""

from .loader import (
    REQUIRED_CONFIG_KEYS,
    ConfigurationError,
    ensure_scaler_finite,
    load_defaults,
    load_feature_spec,
    resolve_feature_spec_path,
)

__all__ = [
    "ConfigurationError",
    "REQUIRED_CONFIG_KEYS",
    "load_defaults",
    "load_feature_spec",
    "resolve_feature_spec_path",
    "ensure_scaler_finite",
]

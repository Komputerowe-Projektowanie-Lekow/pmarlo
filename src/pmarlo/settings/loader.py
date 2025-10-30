from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]

REQUIRED_CONFIG_KEYS = {"enable_cv_bias", "bias_mode", "torch_threads", "precision"}
ALLOWED_BIAS_MODES = {"harmonic"}
ALLOWED_PRECISIONS = {"single", "double"}
DEFAULT_CONFIG_FILENAME = "defaults.yaml"
DEFAULT_SPEC_FILENAME = "feature_spec.yaml"
CONFIG_ENV_VAR = "PMARLO_CONFIG_FILE"


class ConfigurationError(RuntimeError):
    """Raised when configuration values are missing or invalid."""


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME


def _resolve_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def load_defaults() -> Dict[str, Any]:
    """
    Load the default configuration, validating required keys and value domains.
    """

    env_override = os.environ.get(CONFIG_ENV_VAR)
    if env_override:
        config_path = Path(env_override).expanduser().resolve()
    else:
        config_path = _default_config_path()

    config_dir = config_path.parent
    payload = _load_yaml(config_path)

    missing = REQUIRED_CONFIG_KEYS - payload.keys()
    if missing:
        raise ConfigurationError(
            f"Configuration {config_path} missing required keys: "
            + ", ".join(sorted(missing))
        )

    truthy = {"true", "yes", "1"}
    falsy = {"false", "no", "0"}

    def _coerce_boolean(value: Any, key: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
        raise ConfigurationError(
            f"{key} must be a boolean or one of: {', '.join(sorted(truthy | falsy))}."
        )

    enable_bias = _coerce_boolean(payload["enable_cv_bias"], "enable_cv_bias")
    bias_mode = str(payload["bias_mode"]).strip().lower()
    if bias_mode not in ALLOWED_BIAS_MODES:
        raise ConfigurationError(
            f"Unsupported bias_mode '{payload['bias_mode']}'. "
            f"Supported: {', '.join(sorted(ALLOWED_BIAS_MODES))}"
        )
    try:
        torch_threads = int(payload["torch_threads"])
    except Exception as exc:
        raise ConfigurationError("torch_threads must be an integer") from exc
    if torch_threads <= 0:
        raise ConfigurationError("torch_threads must be a positive integer")

    precision = str(payload["precision"]).strip().lower()
    if precision not in ALLOWED_PRECISIONS:
        raise ConfigurationError(
            f"Unsupported precision '{payload['precision']}'. "
            f"Supported: {', '.join(sorted(ALLOWED_PRECISIONS))}"
        )

    feature_spec_path = payload.get("feature_spec_path")
    if feature_spec_path is None:
        feature_spec_path = config_dir / DEFAULT_SPEC_FILENAME
    else:
        feature_spec_path = _resolve_path(config_dir, feature_spec_path)

    payload = dict(payload)
    payload.update(
        {
            "enable_cv_bias": enable_bias,
            "bias_mode": bias_mode,
            "torch_threads": torch_threads,
            "precision": precision,
            "config_path": str(config_path),
            "feature_spec_path": str(feature_spec_path),
        }
    )
    return payload


def resolve_feature_spec_path() -> Path:
    """Return the resolved path to the canonical feature spec."""

    cfg = load_defaults()
    path = Path(cfg["feature_spec_path"])
    if not path.exists():
        raise ConfigurationError(f"Feature specification not found: {path}")
    return path


def load_feature_spec() -> Tuple[Dict[str, Any], str]:
    """
    Load the canonical feature specification and return the spec with its SHA-256 hash.
    """

    from pmarlo.features.deeptica.ts_feature_extractor import canonicalize_feature_spec

    spec_path = resolve_feature_spec_path()
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in feature spec: {spec_path}") from exc
    if not isinstance(spec, dict):
        raise ConfigurationError(
            f"Feature specification must be a mapping at root: {spec_path}"
        )
    canonicalize_feature_spec(spec)  # validates structure
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return spec, spec_hash


def ensure_scaler_finite(mean: np.ndarray, scale: np.ndarray) -> None:
    if mean.size == 0 or scale.size == 0:
        raise ConfigurationError("Scaler parameters must not be empty.")
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
        raise ConfigurationError("Scaler parameters must be finite.")

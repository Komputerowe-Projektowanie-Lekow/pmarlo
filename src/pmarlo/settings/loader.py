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
PROTEIN_METRICS_REQUIRED_KEYS = {
    "hydrophobic_residues",
    "aromatic_residues",
    "basic_residues",
    "acidic_residues",
    "pka_side",
    "pka_n_terminus",
    "pka_c_terminus",
    "ph_lower",
    "ph_upper",
    "pi_max_iterations",
    "pi_tolerance",
}


class ConfigurationError(RuntimeError):
    """Raised when configuration values are missing or invalid."""


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME


def _resolve_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
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


def load_defaults() -> Dict[str, Any]:
    """
    Load the default configuration, validating required keys and value domains.
    """

    env_override = os.environ.get(CONFIG_ENV_VAR)
    if env_override:
        config_path = Path(env_override).expanduser().resolve()
    else:
        config_path = _default_config_path()

    return _load_defaults_from_path(str(config_path))


@lru_cache(maxsize=None)
def _load_defaults_from_path(config_path_str: str) -> Dict[str, Any]:
    """Load defaults from a specific configuration path."""

    # BUGFIX: Cache entries per resolved path so environment overrides are respected.
    config_path = Path(config_path_str)
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


load_defaults.cache_clear = _load_defaults_from_path.cache_clear  # type: ignore[attr-defined]
load_defaults.cache_info = _load_defaults_from_path.cache_info  # type: ignore[attr-defined]


def load_protein_metrics_config() -> Dict[str, Any]:
    """Load and validate protein metric configuration values."""

    cfg = load_defaults()
    metrics = cfg.get("protein_metrics")
    if not isinstance(metrics, dict):
        raise ConfigurationError("Configuration missing 'protein_metrics' mapping.")

    missing = PROTEIN_METRICS_REQUIRED_KEYS - metrics.keys()
    if missing:
        raise ConfigurationError(
            "protein_metrics missing required keys: "
            + ", ".join(sorted(missing))
        )

    def _coerce_residue_set(value: Any, key: str) -> set[str]:
        if isinstance(value, str):
            entries = [char for char in value.replace(" ", "") if char]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            entries = [item.strip() for item in value if item.strip()]
        else:
            raise ConfigurationError(
                f"{key} must be a string or list of residue codes."
            )

        residues: set[str] = set()
        for entry in entries:
            if len(entry) != 1:
                raise ConfigurationError(
                    f"{key} entries must be single-letter residue codes."
                )
            residues.add(entry.upper())

        if not residues:
            raise ConfigurationError(f"{key} must not be empty.")
        return residues

    def _coerce_float(value: Any, key: str) -> float:
        if isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a float.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{key} must be a float.") from exc

    def _coerce_int(value: Any, key: str) -> int:
        if isinstance(value, bool):
            raise ConfigurationError(f"{key} must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{key} must be an integer.") from exc

    hydrophobic_residues = _coerce_residue_set(
        metrics["hydrophobic_residues"], "protein_metrics.hydrophobic_residues"
    )
    aromatic_residues = _coerce_residue_set(
        metrics["aromatic_residues"], "protein_metrics.aromatic_residues"
    )
    basic_residues = _coerce_residue_set(
        metrics["basic_residues"], "protein_metrics.basic_residues"
    )
    acidic_residues = _coerce_residue_set(
        metrics["acidic_residues"], "protein_metrics.acidic_residues"
    )

    overlap = basic_residues & acidic_residues
    if overlap:
        raise ConfigurationError(
            "protein_metrics basic_residues and acidic_residues must be disjoint: "
            + ", ".join(sorted(overlap))
        )

    pka_side_raw = metrics["pka_side"]
    if not isinstance(pka_side_raw, dict):
        raise ConfigurationError("protein_metrics.pka_side must be a mapping.")

    pka_side: Dict[str, float] = {}
    for residue, value in pka_side_raw.items():
        if not isinstance(residue, str):
            raise ConfigurationError(
                "protein_metrics.pka_side keys must be residue codes."
            )
        key = residue.strip().upper()
        if len(key) != 1:
            raise ConfigurationError(
                "protein_metrics.pka_side keys must be single-letter residue codes."
            )
        pka_side[key] = _coerce_float(value, f"protein_metrics.pka_side.{key}")

    if not pka_side:
        raise ConfigurationError("protein_metrics.pka_side must not be empty.")

    missing_pka = (basic_residues | acidic_residues) - set(pka_side)
    if missing_pka:
        raise ConfigurationError(
            "protein_metrics.pka_side missing entries for: "
            + ", ".join(sorted(missing_pka))
        )

    pka_n_terminus = _coerce_float(
        metrics["pka_n_terminus"], "protein_metrics.pka_n_terminus"
    )
    pka_c_terminus = _coerce_float(
        metrics["pka_c_terminus"], "protein_metrics.pka_c_terminus"
    )
    ph_lower = _coerce_float(metrics["ph_lower"], "protein_metrics.ph_lower")
    ph_upper = _coerce_float(metrics["ph_upper"], "protein_metrics.ph_upper")
    if ph_lower >= ph_upper:
        raise ConfigurationError(
            "protein_metrics ph bounds must satisfy ph_lower < ph_upper."
        )

    pi_max_iterations = _coerce_int(
        metrics["pi_max_iterations"], "protein_metrics.pi_max_iterations"
    )
    if pi_max_iterations <= 0:
        raise ConfigurationError(
            "protein_metrics.pi_max_iterations must be a positive integer."
        )

    pi_tolerance = _coerce_float(
        metrics["pi_tolerance"], "protein_metrics.pi_tolerance"
    )
    if pi_tolerance <= 0.0:
        raise ConfigurationError("protein_metrics.pi_tolerance must be positive.")

    return {
        "hydrophobic_residues": hydrophobic_residues,
        "aromatic_residues": aromatic_residues,
        "basic_residues": basic_residues,
        "acidic_residues": acidic_residues,
        "pka_side": pka_side,
        "pka_n_terminus": pka_n_terminus,
        "pka_c_terminus": pka_c_terminus,
        "ph_lower": ph_lower,
        "ph_upper": ph_upper,
        "pi_max_iterations": pi_max_iterations,
        "pi_tolerance": pi_tolerance,
    }


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

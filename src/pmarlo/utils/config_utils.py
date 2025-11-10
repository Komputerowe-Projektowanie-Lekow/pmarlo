"""Utilities for configuration management and manipulation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

__all__ = ["deep_merge", "resolve_deeptica"]


def deep_merge(
    base: Mapping[str, Any] | None, override: Mapping[str, Any] | None
) -> Dict[str, Any]:
    """Recursively merge two mapping objects.

    This function performs a deep merge where:
    - Keys from both mappings are combined
    - For conflicting keys with mapping values, the merge is recursive
    - For conflicting keys with non-mapping values, the override takes precedence
    - None values for either parameter are treated as empty mappings

    Parameters
    ----------
    base : Mapping[str, Any] | None
        The base mapping to merge from. Can be None (treated as empty).
    override : Mapping[str, Any] | None
        The override mapping to merge from. Can be None (treated as empty).
        Values from this mapping take precedence over the base.

    Returns
    -------
    Dict[str, Any]
        A new dictionary containing the merged result.

    Examples
    --------
    >>> base = {"a": 1, "b": {"x": 2, "y": 3}}
    >>> override = {"b": {"y": 4, "z": 5}, "c": 6}
    >>> deep_merge(base, override)
    {'a': 1, 'b': {'x': 2, 'y': 4, 'z': 5}, 'c': 6}
    """
    result: Dict[str, Any] = {}

    if base:
        for key, value in base.items():
            if isinstance(value, Mapping):
                result[key] = deep_merge(value, None)
            else:
                result[key] = value

    if override:
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

    return result


def resolve_deeptica(
    transform_cfg: Mapping[str, Any],
) -> tuple[bool, Dict[str, Any] | None]:
    """Parse and validate DeepTICA configuration sections.

    Extracts the "deeptica" section from a transform configuration, validates
    and coerces its parameters, and returns whether it's enabled along with
    the sanitized parameters.

    Parameters
    ----------
    transform_cfg : Mapping[str, Any]
        The transform configuration containing an optional "deeptica" section.

    Returns
    -------
    tuple[bool, Dict[str, Any] | None]
        A tuple of (enabled, params) where:
        - enabled (bool): Whether DeepTICA is enabled (from "enabled" key, default True)
        - params (Dict[str, Any] | None): Sanitized DeepTICA parameters dict, or None if
          no valid deeptica section exists or no parameters remain after removing "enabled"

    Notes
    -----
    The function performs the following transformations:
    - Extracts and removes the "enabled" key (defaults to True if missing)
    - Coerces "skip_on_failure" to bool if present
    - Coerces "min_pairs" to int if present and valid, removes it otherwise
    - Returns None for params if the deeptica section is not a mapping or becomes empty

    Examples
    --------
    >>> cfg = {"deeptica": {"enabled": True, "min_pairs": "100", "skip_on_failure": 1}}
    >>> enabled, params = resolve_deeptica(cfg)
    >>> enabled
    True
    >>> params
    {'min_pairs': 100, 'skip_on_failure': True}

    >>> cfg = {"deeptica": {"enabled": False}}
    >>> enabled, params = resolve_deeptica(cfg)
    >>> enabled
    False
    >>> params is None
    True
    """
    deeptica_cfg = transform_cfg.get("deeptica")
    if not isinstance(deeptica_cfg, MutableMapping):
        return False, None
    cfg = dict(deeptica_cfg)
    enabled = bool(cfg.pop("enabled", True))

    if "skip_on_failure" in cfg:
        cfg["skip_on_failure"] = bool(cfg["skip_on_failure"])

    if "min_pairs" in cfg:
        try:
            cfg["min_pairs"] = int(cfg["min_pairs"])
        except (TypeError, ValueError):
            cfg.pop("min_pairs", None)

    return enabled, cfg if cfg else None

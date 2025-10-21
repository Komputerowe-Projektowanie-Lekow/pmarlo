from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import mdtraj as md  # type: ignore
import numpy as np


class FeatureComputer(Protocol):
    name: str

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray: ...

    def is_periodic(self) -> np.ndarray:
        """Boolean flags per returned dimension indicating periodicity."""
        ...


FEATURE_REGISTRY: Dict[str, FeatureComputer] = {}


def register_feature(fc: FeatureComputer) -> None:
    """Register a feature implementation.

    The registry is case-insensitive to make user-provided specifications
    robust to capitalization.  Only a single instance is stored per feature
    name to avoid unnecessary copies.
    """

    FEATURE_REGISTRY[fc.name.lower()] = fc


def get_feature(name: str) -> FeatureComputer:
    """Retrieve a registered feature implementation by name.

    Parameters
    ----------
    name:
        Name of the feature.  Lookup is case-insensitive.
    """

    key = name.lower()
    if key not in FEATURE_REGISTRY:
        raise KeyError(f"Feature '{name}' is not registered")
    return FEATURE_REGISTRY[key]


def _normalize_and_split_feature_spec(s: str) -> Tuple[str, str | None]:
    """Normalize a feature specification string.

    This helper normalizes the prefix of the specification to lower case so
    that both namespaced (e.g. ``"dist:AtomPair"``) and simple names
    (e.g. ``"Rg"``) can be resolved in a case-insensitive manner.
    """

    s = s.strip()
    if ":" not in s:
        # Simple feature name
        return s.lower(), None
    prefix_local, rest_local = s.split(":", 1)
    return prefix_local.strip().lower(), rest_local.strip()


def _parse_distance_feature(rest: str | None) -> Tuple[str, Dict[str, Any]]:
    if rest is None:
        return "distance_pair", {}
    name, args = _parse_name_args(rest)
    if name.lower() in {"atompair", "pair", "atoms"} and len(args) == 2:
        try:
            i_idx = int(args[0])
            j_idx = int(args[1])
            return "distance_pair", {"i": i_idx, "j": j_idx}
        except ValueError:
            return "distance_pair", {}
    return "distance_pair", {}


def _parse_contacts_feature(rest: str | None) -> Tuple[str, Dict[str, Any]]:
    if rest is None:
        return "contacts_pair", {}
    name, args = _parse_name_args(rest)
    if name.lower() in {"atompair", "pair", "atoms"} and len(args) >= 2:
        try:
            i_idx = int(args[0])
            j_idx = int(args[1])
            rcut_val = float(args[2]) if len(args) >= 3 else 0.5
            return "contacts_pair", {"i": i_idx, "j": j_idx, "rcut": rcut_val}
        except ValueError:
            return "contacts_pair", {}
    return "contacts_pair", {}


def parse_feature_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a lightweight feature spec string to (feature_name, kwargs).

    Supported (Phase B minimal):
    - "phi_psi"
    - "Rg"
    - "chi1"
    - "dist:atompair(i,j)" where i,j are integer atom indices (0-based)
    """

    prefix, rest = _normalize_and_split_feature_spec(spec)

    # Simple names without namespace
    if rest is None:
        return prefix, {}

    # Namespaced forms
    if prefix in {"dist", "distance", "distances"}:
        return _parse_distance_feature(rest)
    if prefix in {"contacts", "contact"}:
        return _parse_contacts_feature(rest)
    if prefix in {"sasa"}:
        return "sasa", {}
    if prefix in {"hbonds", "hbond", "hbondscount"}:
        return "hbonds_count", {}
    if prefix in {"ssfrac", "secondary"}:
        return "ssfrac", {}

    # Default: return normalized original spec (preserve unknown namespaces)
    return spec.strip().lower(), {}


def _parse_name_args(s: str) -> Tuple[str, Tuple[str, ...]]:
    """Parse 'name(arg1,arg2,...)' -> (name, (arg1, arg2, ...))."""
    s = s.strip()
    if "(" in s and s.endswith(")"):
        name = s[: s.index("(")].strip()
        inside = s[s.index("(") + 1 : -1].strip()
        if not inside:
            return name, tuple()
        parts = tuple(a.strip() for a in inside.split(","))
        return name, parts
    return s, tuple()

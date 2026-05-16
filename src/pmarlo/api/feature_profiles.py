"""Named feature profiles for API and notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    """A reusable feature extraction profile."""

    name: str
    description: str
    feature_type: str
    features: list[str]
    cv_biasing_compatible: bool
    display_features: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type,
            "features": list(self.features),
            "cv_biasing_compatible": self.cv_biasing_compatible,
            "display_features": list(self.display_features or self.features),
        }


FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "cv_analysis": FeatureProfile(
        name="cv_analysis",
        description="Collective variables for analysis and MSM building.",
        feature_type="cv",
        features=["Rg", "RMSD_ref"],
        cv_biasing_compatible=False,
        display_features=[
            "Rg - radius of gyration after CA alignment",
            "RMSD_ref - CA-aligned RMSD relative to the reference structure",
        ],
    ),
    "molecular_cv_biasing": FeatureProfile(
        name="molecular_cv_biasing",
        description="Molecular features suitable for CV-biased simulations.",
        feature_type="molecular",
        features=[
            "distance([0, 1])",
            "distance([1, 2])",
            "angle([0, 1, 2])",
            "dihedral([0, 1, 2, 3])",
            "dihedral([1, 2, 4, 7])",
        ],
        cv_biasing_compatible=True,
        display_features=[
            "distance([0, 1]) - N-CA distance",
            "distance([1, 2]) - CA-C distance",
            "angle([0, 1, 2]) - N-CA-C bond angle",
            "dihedral([0, 1, 2, 3]) - N-CA-C-O dihedral",
            "dihedral([1, 2, 4, 7]) - CA-C-CB-HB1 dihedral",
        ],
    ),
    "molecular_custom": FeatureProfile(
        name="molecular_custom",
        description="Molecular features loaded from a feature_spec.yaml file.",
        feature_type="molecular",
        features=[],
        cv_biasing_compatible=True,
        display_features=[],
    ),
}


def load_feature_profile(
    profile_name: str,
    spec_path: str | Path | None = None,
) -> FeatureProfile:
    """Load a named feature profile.

    ``molecular_custom`` requires a YAML feature specification with entries shaped
    like the canonical ``pmarlo.settings`` feature spec.
    """

    if profile_name not in FEATURE_PROFILES:
        available = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(
            f"Unknown feature profile: {profile_name}. Available profiles: {available}"
        )

    profile = FEATURE_PROFILES[profile_name]
    if profile_name != "molecular_custom":
        return profile

    if spec_path is None:
        raise ValueError("spec_path is required for molecular_custom profile")

    resolved_spec_path = Path(spec_path)
    if not resolved_spec_path.exists():
        raise FileNotFoundError(
            f"Feature specification not found: {resolved_spec_path}"
        )

    spec = yaml.safe_load(resolved_spec_path.read_text(encoding="utf-8")) or {}
    if not isinstance(spec, dict):
        raise ValueError(
            f"Feature specification root must be a mapping: {resolved_spec_path}"
        )

    features: list[str] = []
    display_features: list[str] = []
    for feature in spec.get("features", []):
        if not isinstance(feature, dict):
            raise ValueError("Feature specification entries must be mappings")
        spec_text = _feature_entry_to_spec(feature)
        features.append(spec_text)
        label = str(feature.get("name") or "").strip()
        display_features.append(f"{label} ({spec_text})" if label else spec_text)

    return FeatureProfile(
        name=profile.name,
        description=profile.description,
        feature_type=profile.feature_type,
        features=features,
        cv_biasing_compatible=profile.cv_biasing_compatible,
        display_features=display_features,
    )


def get_feature_profile_info(
    profile_name: str,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return metadata for a named feature profile."""

    if profile_name not in FEATURE_PROFILES:
        return {"exists": False, "name": profile_name}

    profile = FEATURE_PROFILES[profile_name]
    info = profile.to_dict()
    info["exists"] = True
    info["feature_count"] = len(profile.features) if profile.features else "variable"

    if profile_name == "molecular_custom":
        info["spec_path"] = str(spec_path) if spec_path is not None else None
        info["spec_status"] = "spec_path_not_provided"
        if spec_path is not None:
            try:
                custom_profile = load_feature_profile(profile_name, spec_path)
            except FileNotFoundError:
                info["spec_status"] = f"missing:{spec_path}"
            except Exception as exc:
                info["spec_status"] = f"error:{exc}"
            else:
                info.update(custom_profile.to_dict())
                info["exists"] = True
                info["feature_count"] = len(custom_profile.features)
                info["spec_status"] = "loaded"

    return info


def validate_profile_for_cv_biasing(profile_name: str) -> tuple[bool, str]:
    """Return whether a feature profile is compatible with CV biasing."""

    info = get_feature_profile_info(profile_name)
    if not info["exists"]:
        return False, f"Unknown profile: {profile_name}"
    if not info["cv_biasing_compatible"]:
        return False, (
            f"Profile '{profile_name}' uses {info['feature_type']} features that are "
            "not compatible with OpenMM CV biasing."
        )
    return True, "Profile is compatible with CV biasing"


def _feature_entry_to_spec(feature: dict[str, Any]) -> str:
    feature_type = str(feature.get("type") or "").strip()
    atom_indices = feature.get("atom_indices", [])
    if feature_type == "distance" and len(atom_indices) == 2:
        return f"distance({atom_indices})"
    if feature_type == "angle" and len(atom_indices) == 3:
        return f"angle({atom_indices})"
    if feature_type == "dihedral" and len(atom_indices) == 4:
        return f"dihedral({atom_indices})"
    raise ValueError(f"Unsupported feature specification entry: {feature!r}")

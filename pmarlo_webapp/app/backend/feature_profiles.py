"""Feature profile definitions for shard creation.

This module defines different feature extraction profiles that can be used
when creating shards from simulation trajectories.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml


@dataclass
class FeatureProfile:
    """A feature extraction profile for shard creation.

    Attributes
    ----------
    name : str
        Profile name (e.g., "cv_analysis", "molecular_cv_biasing")
    description : str
        Human-readable description
    feature_type : str
        Type of features: "cv" (collective variables) or "molecular" (atomic features)
    features : List[str]
        List of feature specifications
    cv_biasing_compatible : bool
        Whether this profile can be used for CV-biased simulations
    display_features : Optional[List[str]]
        Human-readable feature labels corresponding to ``features``.
    """
    name: str
    description: str
    feature_type: str
    features: List[str]
    cv_biasing_compatible: bool
    display_features: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type,
            "features": self.features,
            "cv_biasing_compatible": self.cv_biasing_compatible,
            "display_features": self.display_features or self.features,
        }


# Predefined feature profiles
FEATURE_PROFILES = {
    "cv_analysis": FeatureProfile(
        name="cv_analysis",
        description="Collective Variables (Rg, RMSD) - For analysis and MSM building only",
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
        description="Molecular Features (distances, angles, dihedrals) - For CV-biased simulations",
        feature_type="molecular",
        features=[
            "distance([0, 1])",      # N-CA distance
            "distance([1, 2])",      # CA-C distance
            "angle([0, 1, 2])",      # N-CA-C angle
            "dihedral([0, 1, 2, 3])", # N-CA-C-O dihedral
            "dihedral([1, 2, 4, 7])", # CA-C-CB-HB1 dihedral
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
        description="Custom Molecular Features - Load from feature_spec.yaml",
        feature_type="molecular",
        features=[],  # Will be loaded from feature_spec.yaml
        cv_biasing_compatible=True,
        display_features=[],
    ),
}


def load_feature_profile(profile_name: str, spec_path: Path = None) -> FeatureProfile:
    """Load a feature profile by name.

    Parameters
    ----------
    profile_name : str
        Name of the profile to load
    spec_path : Path, optional
        Path to feature_spec.yaml for custom molecular profiles

    Returns
    -------
    FeatureProfile
        The loaded feature profile

    Raises
    ------
    ValueError
        If profile name is unknown or spec file is missing for custom profiles
    """
    if profile_name not in FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile: {profile_name}. "
            f"Available profiles: {list(FEATURE_PROFILES.keys())}"
        )

    profile = FEATURE_PROFILES[profile_name]

    # Load custom molecular features from spec file
    if profile_name == "molecular_custom":
        if spec_path is None:
            raise ValueError(
                "feature_spec.yaml path required for molecular_custom profile"
            )

        if not spec_path.exists():
            raise FileNotFoundError(f"Feature specification not found: {spec_path}")

        with spec_path.open("r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        features: List[str] = []
        display_features: List[str] = []
        for feat in spec.get("features", []):
            feat_type = feat.get("type")
            atom_indices = feat.get("atom_indices", [])
            label = str(feat.get("name") or "").strip()

            spec_text: str | None = None
            if feat_type == "distance" and len(atom_indices) == 2:
                spec_text = f"distance({atom_indices})"
            elif feat_type == "angle" and len(atom_indices) == 3:
                spec_text = f"angle({atom_indices})"
            elif feat_type == "dihedral" and len(atom_indices) == 4:
                spec_text = f"dihedral({atom_indices})"

            if spec_text is None:
                continue

            features.append(spec_text)
            if label:
                display_features.append(f"{label} ({spec_text})")
            else:
                display_features.append(spec_text)

        # Create a modified profile with loaded features
        profile = FeatureProfile(
            name=profile.name,
            description=profile.description,
            feature_type=profile.feature_type,
            features=features,
            cv_biasing_compatible=profile.cv_biasing_compatible,
            display_features=display_features or None,
        )

    return profile


def get_feature_profile_info(
    profile_name: str,
    spec_path: Path | None = None,
) -> Dict[str, Any]:
    """Get information about a feature profile without loading features.

    Parameters
    ----------
    profile_name : str
        Name of the profile
    spec_path : Path, optional
        Optional path to feature_spec.yaml for custom profiles

    Returns
    -------
    dict
        Profile information
    """
    if profile_name not in FEATURE_PROFILES:
        return {
            "exists": False,
            "name": profile_name,
        }

    profile = FEATURE_PROFILES[profile_name]
    features = list(profile.features)
    display_features = list(profile.display_features or profile.features)
    feature_count: int | str = len(features) if features else "variable"
    info: Dict[str, Any] = {
        "exists": True,
        "name": profile.name,
        "description": profile.description,
        "feature_type": profile.feature_type,
        "cv_biasing_compatible": profile.cv_biasing_compatible,
        "feature_count": feature_count,
        "features": features,
        "display_features": display_features,
    }

    if profile_name == "molecular_custom":
        info["spec_path"] = str(spec_path) if spec_path is not None else None
        info["spec_status"] = "unresolved"
        if spec_path is None:
            info["spec_status"] = "spec_path_not_provided"
        else:
            try:
                custom_profile = load_feature_profile(profile_name, spec_path=spec_path)
            except FileNotFoundError:
                info["spec_status"] = f"missing:{spec_path}"
            except Exception as exc:
                info["spec_status"] = f"error:{exc}"
            else:
                info["features"] = list(custom_profile.features)
                resolved_display = custom_profile.display_features or custom_profile.features
                info["display_features"] = list(resolved_display)
                info["feature_count"] = len(custom_profile.features)
                info["spec_status"] = "loaded"

        if not info["features"]:
            info["feature_count"] = "variable"

    return info


def validate_profile_for_cv_biasing(profile_name: str) -> tuple[bool, str]:
    """Validate if a profile can be used for CV biasing.

    Parameters
    ----------
    profile_name : str
        Name of the profile to validate

    Returns
    -------
    bool
        True if compatible with CV biasing
    str
        Validation message
    """
    info = get_feature_profile_info(profile_name)

    if not info["exists"]:
        return False, f"Unknown profile: {profile_name}"

    if not info["cv_biasing_compatible"]:
        return False, (
            f"Profile '{profile_name}' uses {info['feature_type']} features "
            "which cannot be computed on-the-fly in OpenMM. "
            "Use 'molecular_cv_biasing' or 'molecular_custom' for CV biasing."
        )

    return True, "Profile is compatible with CV biasing"

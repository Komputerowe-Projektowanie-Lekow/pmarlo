"""Quick smoke test for molecular features."""

import pytest


def test_feature_import():
    """Test that features module imports correctly."""
    from pmarlo.features import get_feature
    assert get_feature is not None


def test_feature_registration():
    """Test that basic features are registered."""
    from pmarlo.features import get_feature

    for name in ['distance', 'angle', 'dihedral']:
        fc = get_feature(name)
        assert fc.name == name


def test_feature_parsing():
    """Test basic feature spec parsing."""
    from pmarlo.features.base import parse_feature_spec

    test_specs = [
        ("distance([0, 1])", "distance", {"atoms": [0, 1]}),
        ("angle([0, 1, 2])", "angle", {"atoms": [0, 1, 2]}),
        ("dihedral([0, 1, 2, 3])", "dihedral", {"atoms": [0, 1, 2, 3]}),
    ]

    for spec, expected_name, expected_kwargs in test_specs:
        name, kwargs = parse_feature_spec(spec)
        assert name == expected_name
        assert kwargs == expected_kwargs


def test_profile_loading():
    """Test loading feature profile."""
    from pmarlo_webapp.app.backend.feature_profiles import load_feature_profile

    profile = load_feature_profile("molecular_cv_biasing")
    assert profile.name == "molecular_cv_biasing"
    assert len(profile.features) > 0

from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo.api import (
    FEATURE_PROFILES,
    extract_shards_with_features,
    get_feature_profile_info,
    load_feature_profile,
    validate_profile_for_cv_biasing,
)
from pmarlo.data.shard import read_shard


def test_molecular_feature_profile_is_available() -> None:
    profile = load_feature_profile("molecular_cv_biasing")

    assert "molecular_cv_biasing" in FEATURE_PROFILES
    assert profile.name == "molecular_cv_biasing"
    assert profile.cv_biasing_compatible is True
    assert profile.features

    ok, message = validate_profile_for_cv_biasing(profile.name)
    assert ok is True
    assert "compatible" in message


def test_custom_feature_profile_loads_canonical_spec() -> None:
    spec_path = Path("src/pmarlo/settings/feature_spec.yaml")

    profile = load_feature_profile("molecular_custom", spec_path=spec_path)
    info = get_feature_profile_info("molecular_custom", spec_path=spec_path)

    assert profile.features == [
        "distance([0, 1])",
        "distance([1, 2])",
        "angle([0, 1, 2])",
        "dihedral([0, 1, 2, 3])",
        "dihedral([1, 2, 4, 7])",
    ]
    assert info["spec_status"] == "loaded"
    assert info["feature_count"] == len(profile.features)


def test_extract_shards_with_features_uses_real_trajectory(
    tmp_path: Path,
    test_fixed_pdb_file: Path,
    test_trajectory_file: Path,
) -> None:
    profile = load_feature_profile("molecular_cv_biasing")
    out_dir = tmp_path / "shards"

    shard_paths = extract_shards_with_features(
        pdb_file=test_fixed_pdb_file,
        traj_files=[test_trajectory_file],
        out_dir=out_dir,
        feature_specs=profile.features,
        stride=2,
        temperature=300.0,
        seed_start=10,
        frames_per_shard=8,
        provenance={"run_id": "api-feature-profile-test"},
    )

    assert shard_paths
    meta, data, dtraj = read_shard(shard_paths[0])

    assert dtraj is None
    assert data.shape[1] == len(profile.features)
    assert meta.temperature_K == 300.0
    assert meta.source["run_id"] == "api-feature-profile-test"
    assert meta.source["stride"] == 2
    assert meta.source["frame_stride"] == 2
    assert meta.source["n_frames"] == data.shape[0] * 2

    periodic = meta.source["periodic"]
    columns = meta.source["columns"]
    periodic_by_column = dict(zip(columns, periodic))
    for column, is_periodic in periodic_by_column.items():
        if column.startswith("distance("):
            assert is_periodic is False
        else:
            assert is_periodic is True
    assert np.isfinite(data).all()

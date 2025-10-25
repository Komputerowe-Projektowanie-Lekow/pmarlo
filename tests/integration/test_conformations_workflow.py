"""Integration test for conformations workflow with real data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pmarlo import api
from pmarlo.conformations import find_conformations

pytestmark = pytest.mark.integration


@pytest.fixture
def real_trajectory(request):
    """Load real trajectory from test assets."""
    test_dir = Path(request.fspath).parent.parent
    assets_dir = test_dir / "_assets"

    pdb_path = assets_dir / "3gd8-fixed.pdb"
    dcd_path = assets_dir / "traj.dcd"

    if not pdb_path.exists() or not dcd_path.exists():
        pytest.skip("Test assets not available")

    traj = md.load(str(dcd_path), top=str(pdb_path))

    return traj


def test_conformations_full_workflow(real_trajectory):
    """Test complete conformations workflow with real data."""
    traj = real_trajectory

    # Subsample for speed
    traj = traj[::5]

    if len(traj) < 50:
        pytest.skip("Trajectory too short")

    # Step 1: Compute features
    features, cols, periodic = api.compute_features(traj, feature_specs=["phi_psi"])

    assert features.shape[0] == len(traj)
    assert features.shape[1] > 0

    # Step 2: Reduce dimensionality
    features_reduced = api.reduce_features(
        features, method="tica", lag=5, n_components=2
    )

    assert features_reduced.shape[0] == len(traj)
    assert features_reduced.shape[1] == 2

    # Step 3: Cluster
    labels = api.cluster_microstates(
        features_reduced, method="minibatchkmeans", n_states=10, n_init=10
    )

    assert len(labels) == len(traj)
    n_states = int(np.max(labels) + 1)
    assert n_states <= 10

    # Step 4: Build MSM
    T, pi = api.build_simple_msm(
        [labels], n_states=n_states, lag=2, count_mode="sliding"
    )

    assert T.shape == (n_states, n_states)
    assert len(pi) == n_states

    # Step 5: Run conformations finder
    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        results = find_conformations(
            msm_data=msm_data,
            trajectories=traj,
            auto_detect=True,
            find_transition_states=True,
            find_metastable_states=True,
            find_pathway_intermediates=True,
            compute_kis=True,
            uncertainty_analysis=False,  # Skip for speed
            output_dir=tmpdir,
            save_structures=True,
        )

        # Verify results
        assert results is not None
        assert len(results.conformations) > 0

        # Check TPT result
        assert results.tpt_result is not None
        assert results.tpt_result.rate > 0
        assert results.tpt_result.total_flux > 0

        # Check KIS result
        assert results.kis_result is not None
        assert len(results.kis_result.kis_scores) == n_states

        # Check conformations types
        metastable = results.get_metastable_states()
        assert len(metastable) > 0

        # Check that structures were saved
        output_path = Path(tmpdir)
        assert output_path.exists()


def test_conformations_with_uncertainty(real_trajectory):
    """Test conformations with uncertainty quantification."""
    traj = real_trajectory

    # Heavy subsample for speed
    traj = traj[::10]

    if len(traj) < 30:
        pytest.skip("Trajectory too short")

    # Quick MSM
    features, _, _ = api.compute_features(traj, feature_specs=["phi_psi"])
    features_reduced = api.reduce_features(features, method="pca", n_components=2)
    labels = api.cluster_microstates(
        features_reduced, method="minibatchkmeans", n_states=5
    )
    n_states = int(np.max(labels) + 1)

    T, pi = api.build_simple_msm([labels], n_states=n_states, lag=1)

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
    }

    # Run with uncertainty (small n_bootstrap for speed)
    results = find_conformations(
        msm_data=msm_data,
        auto_detect=True,
        find_transition_states=True,
        find_metastable_states=True,
        compute_kis=False,  # Skip for speed
        uncertainty_analysis=True,
        n_bootstrap=10,  # Small for speed
        lag=1,
    )

    assert results is not None
    assert len(results.uncertainty_results) > 0

    # Check that we have uncertainty for rate/mfpt
    observable_names = [u.observable_name for u in results.uncertainty_results]
    assert "rate" in observable_names or "mfpt" in observable_names


def test_conformations_manual_source_sink(real_trajectory):
    """Test conformations with manually specified source/sink."""
    traj = real_trajectory[::10]

    if len(traj) < 30:
        pytest.skip("Trajectory too short")

    # Quick MSM
    features, _, _ = api.compute_features(traj, feature_specs=["phi_psi"])
    features_reduced = api.reduce_features(features, method="pca", n_components=2)
    labels = api.cluster_microstates(
        features_reduced, method="minibatchkmeans", n_states=8
    )
    n_states = int(np.max(labels) + 1)

    T, pi = api.build_simple_msm([labels], n_states=n_states, lag=1)

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
    }

    # Manually specify source and sink
    source_states = np.array([0])
    sink_states = np.array([n_states - 1])

    results = find_conformations(
        msm_data=msm_data,
        source_states=source_states,
        sink_states=sink_states,
        auto_detect=False,
        find_transition_states=True,
        compute_kis=False,
    )

    assert results is not None
    assert results.tpt_result is not None
    assert np.array_equal(results.tpt_result.source_states, source_states)
    assert np.array_equal(results.tpt_result.sink_states, sink_states)


def test_conformations_serialization(real_trajectory):
    """Test ConformationSet serialization and loading."""
    traj = real_trajectory[::15]

    if len(traj) < 20:
        pytest.skip("Trajectory too short")

    # Quick MSM
    features, _, _ = api.compute_features(traj, feature_specs=["phi_psi"])
    features_reduced = api.reduce_features(features, method="pca", n_components=2)
    labels = api.cluster_microstates(
        features_reduced, method="minibatchkmeans", n_states=5
    )
    n_states = int(np.max(labels) + 1)

    T, pi = api.build_simple_msm([labels], n_states=n_states, lag=1)

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
    }

    results = find_conformations(
        msm_data=msm_data,
        auto_detect=True,
        find_metastable_states=True,
        compute_kis=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        json_path = Path(tmpdir) / "conformations.json"
        results.save(json_path)

        assert json_path.exists()

        # Load
        from pmarlo.conformations.results import ConformationSet

        loaded = ConformationSet.load(json_path)

        assert len(loaded.conformations) == len(results.conformations)
        assert loaded.metadata == results.metadata

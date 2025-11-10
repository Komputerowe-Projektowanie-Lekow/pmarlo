"""Integration tests for TPT functionality in EnhancedMSM.

Tests the complete TPT workflow from MSM building to analysis and visualization.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

deeptime = pytest.importorskip("deeptime")

from pmarlo.markov_state_model import MarkovStateModel


def create_test_msm_with_tpt():
    """Create a test MSM and run TPT analysis."""
    # Create a 10-state MSM with clear metastable states
    T = np.array(
        [
            [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15, 0.7, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.7, 0.15, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
        ]
    )

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    return T, pi


def test_tpt_full_workflow():
    """Test complete TPT workflow on MSM."""
    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "test_tpt_workflow",
        )

        # Set MSM parameters
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 10

        # Define source and sink
        source = [0, 1]
        sink = [8, 9]

        # 1. Compute reactive flux
        flux = msm.reactive_flux(source, sink)
        assert flux.total_flux > 0
        assert flux.rate > 0
        assert flux.mfpt > 0

        # 2. Compute committors
        q_forward = msm.compute_committor(source, sink, forward=True)
        assert q_forward.shape == (10,)
        assert q_forward[0] == pytest.approx(0.0, abs=1e-10)
        assert q_forward[8] == pytest.approx(1.0, abs=1e-10)

        # 3. Get flux matrices
        gross_flux = msm.get_gross_flux(source, sink)
        net_flux = msm.get_net_flux(source, sink)
        assert gross_flux.shape == (10, 10)
        assert net_flux.shape == (10, 10)
        assert np.all(net_flux >= 0)

        # 4. Get rate and MFPT
        rate = msm.get_transition_rate(source, sink)
        mfpt = msm.get_mfpt(source, sink)
        assert rate > 0
        assert mfpt > 0
        assert mfpt == pytest.approx(1.0 / rate, rel=1e-10)

        # 5. Pathway decomposition
        pathways, pathway_fluxes = msm.pathway_decomposition(source, sink, fraction=0.95)
        assert len(pathways) > 0
        assert len(pathway_fluxes) == len(pathways)
        assert all(p[0] in source for p in pathways)
        assert all(p[-1] in sink for p in pathways)

        # 6. Identify TSE
        ts_states = msm.identify_transition_state_ensemble(source, sink, tolerance=0.15)
        assert len(ts_states) >= 0  # May be empty

        # 7. Find bottlenecks
        bottlenecks = msm.find_bottleneck_states(source, sink, top_n=5)
        assert len(bottlenecks) == 5

        # 8. Coarse-grain flux
        intermediate = [i for i in range(10) if i not in source + sink]
        sets = [source, intermediate, sink]
        cg_sets, cg_flux = msm.coarse_grain_flux(source, sink, sets)
        assert len(cg_sets) == 3
        assert cg_flux.gross_flux.shape == (3, 3)


def test_tpt_vs_deeptime_comprehensive():
    """Comprehensive comparison with deeptime's implementation."""
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        # PMARLO MSM
        msm_pmarlo = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "pmarlo",
        )
        msm_pmarlo.transition_matrix = T
        msm_pmarlo.stationary_distribution = pi
        msm_pmarlo.n_states = 10

        # Deeptime MSM
        msm_deeptime = DeeptimeMSM(T, stationary_distribution=pi)

        source = [0, 1]
        sink = [8, 9]

        # Compare reactive flux
        flux_pmarlo = msm_pmarlo.reactive_flux(source, sink)
        flux_deeptime = msm_deeptime.reactive_flux(source, sink)

        # All quantities should match exactly
        np.testing.assert_allclose(
            flux_pmarlo.forward_committor,
            flux_deeptime.forward_committor,
            rtol=1e-10,
            err_msg="Forward committors don't match",
        )

        np.testing.assert_allclose(
            flux_pmarlo.backward_committor,
            flux_deeptime.backward_committor,
            rtol=1e-10,
            err_msg="Backward committors don't match",
        )

        np.testing.assert_allclose(
            flux_pmarlo.gross_flux,
            flux_deeptime.gross_flux,
            rtol=1e-10,
            err_msg="Gross flux doesn't match",
        )

        np.testing.assert_allclose(
            flux_pmarlo.net_flux,
            flux_deeptime.net_flux,
            rtol=1e-10,
            err_msg="Net flux doesn't match",
        )

        assert flux_pmarlo.total_flux == pytest.approx(
            flux_deeptime.total_flux, rel=1e-10
        )
        assert flux_pmarlo.rate == pytest.approx(flux_deeptime.rate, rel=1e-10)
        assert flux_pmarlo.mfpt == pytest.approx(flux_deeptime.mfpt, rel=1e-10)

        # Compare pathways
        paths_pmarlo, fluxes_pmarlo = msm_pmarlo.pathway_decomposition(
            source, sink, fraction=0.95, maxiter=1000
        )
        paths_deeptime, fluxes_deeptime = flux_deeptime.pathways(
            fraction=0.95, maxiter=1000
        )

        assert len(paths_pmarlo) == len(paths_deeptime)
        np.testing.assert_allclose(fluxes_pmarlo, fluxes_deeptime, rtol=1e-10)

        # Compare coarse-grained flux
        intermediate = [i for i in range(10) if i not in source + sink]
        sets = [source, intermediate, sink]

        cg_sets_pmarlo, cg_flux_pmarlo = msm_pmarlo.coarse_grain_flux(source, sink, sets)
        cg_sets_deeptime, cg_flux_deeptime = flux_deeptime.coarse_grain(sets)

        np.testing.assert_allclose(
            cg_flux_pmarlo.gross_flux, cg_flux_deeptime.gross_flux, rtol=1e-10
        )
        np.testing.assert_allclose(
            cg_flux_pmarlo.net_flux, cg_flux_deeptime.net_flux, rtol=1e-10
        )


def test_tpt_drunkards_walk_integration():
    """Integration test using drunkard's walk from deeptime."""
    from deeptime.data import drunkards_walk

    # Create simulator
    sim = drunkards_walk(
        grid_size=(10, 10),
        bar_location=[(0, 0), (0, 1), (1, 0), (1, 1)],
        home_location=[(8, 8), (8, 9), (9, 8), (9, 9)],
    )

    # Add barriers
    sim.add_barrier((5, 1), (5, 5))
    sim.add_barrier((0, 9), (5, 8))

    with TemporaryDirectory() as tmpdir:
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "drunkards_walk",
        )

        msm.transition_matrix = sim.msm.transition_matrix
        msm.stationary_distribution = sim.msm.stationary_distribution
        msm.n_states = sim.msm.n_states

        source = sim.home_state
        sink = sim.bar_state

        # Test all methods work
        flux = msm.reactive_flux(source, sink)
        assert flux.total_flux > 0

        q = msm.compute_committor(source, sink)
        assert q.shape == (sim.n_states,)

        paths, path_fluxes = msm.pathway_decomposition(source, sink, fraction=0.3)
        assert len(paths) > 0

        # Compare with deeptime
        flux_deeptime = sim.msm.reactive_flux(source, sink)

        assert flux.total_flux == pytest.approx(flux_deeptime.total_flux, rel=1e-10)
        assert flux.rate == pytest.approx(flux_deeptime.rate, rel=1e-10)
        assert flux.mfpt == pytest.approx(flux_deeptime.mfpt, rel=1e-10)


def test_tpt_error_handling():
    """Test error handling in TPT methods."""
    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        # Test with no transition matrix
        msm_empty = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "empty",
        )

        with pytest.raises(ValueError, match="Must call build_msm"):
            msm_empty.reactive_flux([0], [5])

        # Test with overlapping source/sink
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "test",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 10

        with pytest.raises(ValueError, match="must not overlap"):
            msm.reactive_flux([0, 1], [1, 2])

        with pytest.raises(ValueError, match="must not overlap"):
            msm.compute_committor([0, 1], [1, 2])


def test_flux_conservation():
    """Test that flux is conserved at intermediate states."""
    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "flux_conservation",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 10

        source = [0, 1]
        sink = [8, 9]
        intermediate = [i for i in range(10) if i not in source + sink]

        net_flux = msm.get_net_flux(source, sink)

        # Check conservation at each intermediate state
        for state in intermediate:
            flux_in = np.sum(net_flux[:, state])
            flux_out = np.sum(net_flux[state, :])

            assert flux_in == pytest.approx(flux_out, rel=1e-8, abs=1e-10), (
                f"Flux not conserved at state {state}: "
                f"in={flux_in:.6e}, out={flux_out:.6e}"
            )


def test_net_flux_formula():
    """Test that net flux is computed correctly."""
    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "net_flux",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 10

        source = [0, 1]
        sink = [8, 9]

        gross_flux = msm.get_gross_flux(source, sink)
        net_flux = msm.get_net_flux(source, sink)

        # Net flux should be: f_ij^+ = max(0, f_ij - f_ji)
        expected_net_flux = np.maximum(0, gross_flux - gross_flux.T)

        np.testing.assert_allclose(
            net_flux, expected_net_flux, rtol=1e-10, err_msg="Net flux formula incorrect"
        )


def test_committor_boundary_conditions():
    """Test that committors satisfy boundary conditions."""
    T, pi = create_test_msm_with_tpt()

    with TemporaryDirectory() as tmpdir:
        msm = MarkovStateModel(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "committor_bc",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 10

        source = [0, 1]
        sink = [8, 9]

        q_forward = msm.compute_committor(source, sink, forward=True)

        # Source states should have q+ = 0
        for s in source:
            assert q_forward[s] == pytest.approx(0.0, abs=1e-10)

        # Sink states should have q+ = 1
        for s in sink:
            assert q_forward[s] == pytest.approx(1.0, abs=1e-10)

        # All states should have 0 <= q+ <= 1
        assert np.all(q_forward >= -1e-10)
        assert np.all(q_forward <= 1.0 + 1e-10)


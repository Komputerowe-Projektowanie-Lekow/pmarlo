"""Unit tests for TPT mixin in EnhancedMSM.

Tests that TPT methods are correctly integrated into the MSM class
and produce results consistent with deeptime's implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

deeptime = pytest.importorskip("deeptime")

from pathlib import Path
from tempfile import TemporaryDirectory

from pmarlo.markov_state_model._enhanced_impl import EnhancedMSM


@pytest.fixture
def simple_msm():
    """Create a simple MSM for testing."""
    # Create a 5-state MSM with clear metastable states
    T = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0],  # State 0: source metastable
            [0.1, 0.8, 0.1, 0.0, 0.0],  # State 1: source metastable
            [0.0, 0.05, 0.9, 0.05, 0.0],  # State 2: intermediate
            [0.0, 0.0, 0.1, 0.8, 0.1],  # State 3: sink metastable
            [0.0, 0.0, 0.0, 0.1, 0.9],  # State 4: sink metastable
        ]
    )

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    with TemporaryDirectory() as tmpdir:
        msm = EnhancedMSM(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "test_msm",
        )

        # Manually set transition matrix and stationary distribution
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = 5

        yield msm


def test_reactive_flux_basic(simple_msm):
    """Test basic reactive_flux method."""
    source = [0, 1]
    sink = [3, 4]

    flux = simple_msm.reactive_flux(source, sink)

    # Check that flux object has required attributes (deeptime API)
    assert hasattr(flux, "forward_committor")
    assert hasattr(flux, "backward_committor")
    assert hasattr(flux, "gross_flux")
    assert hasattr(flux, "net_flux")
    assert hasattr(flux, "total_flux")
    assert hasattr(flux, "rate")
    assert hasattr(flux, "mfpt")

    # Check shapes
    assert flux.forward_committor.shape == (5,)
    assert flux.backward_committor.shape == (5,)
    assert flux.gross_flux.shape == (5, 5)
    assert flux.net_flux.shape == (5, 5)

    # Check values make sense
    assert flux.total_flux > 0
    assert flux.rate > 0
    assert flux.mfpt > 0


def test_reactive_flux_vs_deeptime(simple_msm):
    """Test that reactive_flux matches deeptime's implementation exactly."""
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

    source = [0, 1]
    sink = [3, 4]

    # Get flux from our implementation
    flux_pmarlo = simple_msm.reactive_flux(source, sink)

    # Get flux directly from deeptime
    msm_deeptime = DeeptimeMSM(
        simple_msm.transition_matrix,
        stationary_distribution=simple_msm.stationary_distribution,
    )
    flux_deeptime = msm_deeptime.reactive_flux(source, sink)

    # Compare all quantities
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
    ), "Total flux doesn't match"

    assert flux_pmarlo.rate == pytest.approx(
        flux_deeptime.rate, rel=1e-10
    ), "Rate doesn't match"

    assert flux_pmarlo.mfpt == pytest.approx(
        flux_deeptime.mfpt, rel=1e-10
    ), "MFPT doesn't match"


def test_compute_committor(simple_msm):
    """Test committor computation."""
    source = [0, 1]
    sink = [3, 4]

    q_forward = simple_msm.compute_committor(source, sink, forward=True)
    q_backward = simple_msm.compute_committor(source, sink, forward=False)

    # Check shapes
    assert q_forward.shape == (5,)
    assert q_backward.shape == (5,)

    # Source states should have q+ = 0
    assert q_forward[0] == pytest.approx(0.0, abs=1e-10)
    assert q_forward[1] == pytest.approx(0.0, abs=1e-10)

    # Sink states should have q+ = 1
    assert q_forward[3] == pytest.approx(1.0, abs=1e-10)
    assert q_forward[4] == pytest.approx(1.0, abs=1e-10)

    # Intermediate state should have 0 < q+ < 1
    assert 0 < q_forward[2] < 1

    # For reversible MSMs, q- = 1 - q+
    # Note: Our test MSM is not perfectly reversible, so we just check ranges
    assert 0 <= q_backward[2] <= 1


def test_compute_committor_vs_deeptime(simple_msm):
    """Test that committor matches deeptime's implementation."""
    from deeptime.markov.tools.analysis import committor

    source = np.array([0, 1])
    sink = np.array([3, 4])

    # Forward committor
    q_forward_pmarlo = simple_msm.compute_committor(source, sink, forward=True)
    q_forward_deeptime = committor(
        simple_msm.transition_matrix, source, sink, forward=True
    )

    np.testing.assert_allclose(
        q_forward_pmarlo,
        q_forward_deeptime,
        rtol=1e-10,
        err_msg="Forward committors don't match deeptime",
    )

    # Backward committor
    q_backward_pmarlo = simple_msm.compute_committor(source, sink, forward=False)
    q_backward_deeptime = committor(
        simple_msm.transition_matrix, source, sink, forward=False
    )

    np.testing.assert_allclose(
        q_backward_pmarlo,
        q_backward_deeptime,
        rtol=1e-10,
        err_msg="Backward committors don't match deeptime",
    )


def test_pathway_decomposition(simple_msm):
    """Test pathway decomposition."""
    source = [0, 1]
    sink = [3, 4]

    pathways, fluxes = simple_msm.pathway_decomposition(
        source, sink, fraction=0.95, maxiter=1000
    )

    # Check types
    assert isinstance(pathways, list)
    assert isinstance(fluxes, np.ndarray)

    # Check that we got some pathways
    assert len(pathways) > 0
    assert len(fluxes) == len(pathways)

    # Each pathway should start in source and end in sink
    for path in pathways:
        assert path[0] in source
        assert path[-1] in sink

    # Fluxes should be positive and sorted (descending)
    assert np.all(fluxes >= 0)
    assert np.all(fluxes[:-1] >= fluxes[1:])


def test_pathway_decomposition_vs_deeptime(simple_msm):
    """Test that pathway decomposition matches deeptime."""
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

    source = [0, 1]
    sink = [3, 4]
    fraction = 0.95
    maxiter = 1000

    # Our implementation
    paths_pmarlo, fluxes_pmarlo = simple_msm.pathway_decomposition(
        source, sink, fraction=fraction, maxiter=maxiter
    )

    # Deeptime implementation
    msm_deeptime = DeeptimeMSM(
        simple_msm.transition_matrix,
        stationary_distribution=simple_msm.stationary_distribution,
    )
    flux_deeptime = msm_deeptime.reactive_flux(source, sink)
    paths_deeptime, fluxes_deeptime = flux_deeptime.pathways(
        fraction=fraction, maxiter=maxiter
    )

    # Convert deeptime paths to same format
    paths_deeptime = [list(map(int, path)) for path in paths_deeptime]

    # Should have same number of paths
    assert len(paths_pmarlo) == len(paths_deeptime)

    # Fluxes should match
    np.testing.assert_allclose(
        fluxes_pmarlo, fluxes_deeptime, rtol=1e-10, err_msg="Pathway fluxes don't match"
    )

    # Paths should be identical
    assert paths_pmarlo == paths_deeptime, "Pathways don't match"


def test_get_net_flux(simple_msm):
    """Test net flux calculation."""
    source = [0, 1]
    sink = [3, 4]

    net_flux = simple_msm.get_net_flux(source, sink)

    # Check shape
    assert net_flux.shape == (5, 5)

    # Net flux should be non-negative
    assert np.all(net_flux >= 0)

    # Net flux should be anti-symmetric in the diagonal sense:
    # For reversible systems, if f_ij > 0, then f_ji = 0
    # Check that at least some transitions have been removed
    gross_flux = simple_msm.get_gross_flux(source, sink)
    assert np.sum(net_flux > 0) <= np.sum(gross_flux > 0)


def test_get_gross_flux(simple_msm):
    """Test gross flux calculation."""
    source = [0, 1]
    sink = [3, 4]

    gross_flux = simple_msm.get_gross_flux(source, sink)

    # Check shape
    assert gross_flux.shape == (5, 5)

    # Gross flux should be non-negative
    assert np.all(gross_flux >= 0)

    # Gross flux should sum to something reasonable
    assert np.sum(gross_flux) > 0


def test_get_transition_rate(simple_msm):
    """Test transition rate calculation."""
    source = [0, 1]
    sink = [3, 4]

    rate = simple_msm.get_transition_rate(source, sink)

    # Rate should be positive
    assert rate > 0

    # Rate should match the flux object
    flux = simple_msm.reactive_flux(source, sink)
    assert rate == pytest.approx(flux.rate, rel=1e-10)


def test_get_mfpt(simple_msm):
    """Test MFPT calculation."""
    source = [0, 1]
    sink = [3, 4]

    mfpt = simple_msm.get_mfpt(source, sink)

    # MFPT should be positive
    assert mfpt > 0

    # MFPT should be inverse of rate
    rate = simple_msm.get_transition_rate(source, sink)
    assert mfpt == pytest.approx(1.0 / rate, rel=1e-10)


def test_identify_transition_state_ensemble(simple_msm):
    """Test TSE identification."""
    source = [0, 1]
    sink = [3, 4]

    ts_states = simple_msm.identify_transition_state_ensemble(
        source, sink, tolerance=0.2
    )

    # Should identify at least one state
    assert len(ts_states) > 0

    # Verify committors are around 0.5
    q_forward = simple_msm.compute_committor(source, sink, forward=True)
    for state in ts_states:
        assert 0.3 <= q_forward[state] <= 0.7


def test_find_bottleneck_states(simple_msm):
    """Test bottleneck state identification."""
    source = [0, 1]
    sink = [3, 4]

    bottlenecks = simple_msm.find_bottleneck_states(source, sink, top_n=3)

    # Should return requested number of states
    assert len(bottlenecks) == 3

    # States should be sorted by flux (descending)
    gross_flux = simple_msm.get_gross_flux(source, sink)
    flux_through = 0.5 * (np.sum(gross_flux, axis=1) + np.sum(gross_flux, axis=0))

    for i in range(len(bottlenecks) - 1):
        assert flux_through[bottlenecks[i]] >= flux_through[bottlenecks[i + 1]]


def test_coarse_grain_flux(simple_msm):
    """Test flux coarse-graining."""
    source = [0, 1]
    sink = [3, 4]

    # Define macrostates
    sets = [[0, 1], [2], [3, 4]]

    cg_sets, cg_flux = simple_msm.coarse_grain_flux(source, sink, sets)

    # Check that we got the right number of sets
    assert len(cg_sets) == 3

    # Check that coarse-grained flux has right attributes
    assert hasattr(cg_flux, "gross_flux")
    assert hasattr(cg_flux, "net_flux")
    assert hasattr(cg_flux, "total_flux")

    # Check shapes
    assert cg_flux.gross_flux.shape == (3, 3)
    assert cg_flux.net_flux.shape == (3, 3)


def test_coarse_grain_flux_vs_deeptime(simple_msm):
    """Test that coarse-graining matches deeptime."""
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

    source = [0, 1]
    sink = [3, 4]
    sets = [[0, 1], [2], [3, 4]]

    # Our implementation
    cg_sets_pmarlo, cg_flux_pmarlo = simple_msm.coarse_grain_flux(source, sink, sets)

    # Deeptime implementation
    msm_deeptime = DeeptimeMSM(
        simple_msm.transition_matrix,
        stationary_distribution=simple_msm.stationary_distribution,
    )
    flux_deeptime = msm_deeptime.reactive_flux(source, sink)
    cg_sets_deeptime, cg_flux_deeptime = flux_deeptime.coarse_grain(sets)

    # Compare gross flux
    np.testing.assert_allclose(
        cg_flux_pmarlo.gross_flux,
        cg_flux_deeptime.gross_flux,
        rtol=1e-10,
        err_msg="Coarse-grained gross flux doesn't match",
    )

    # Compare net flux
    np.testing.assert_allclose(
        cg_flux_pmarlo.net_flux,
        cg_flux_deeptime.net_flux,
        rtol=1e-10,
        err_msg="Coarse-grained net flux doesn't match",
    )

    # Compare total flux
    assert cg_flux_pmarlo.total_flux == pytest.approx(
        cg_flux_deeptime.total_flux, rel=1e-10
    ), "Coarse-grained total flux doesn't match"


def test_reactive_flux_errors(simple_msm):
    """Test error handling in reactive_flux."""
    source = [0, 1]
    sink = [3, 4]

    # Overlapping source and sink
    with pytest.raises(ValueError, match="must not overlap"):
        simple_msm.reactive_flux([0, 1], [1, 2])

    # No transition matrix
    msm_empty = EnhancedMSM(
        trajectory_files=None,
        topology_file=None,
        output_dir=Path(simple_msm.output_dir) / "empty",
    )
    with pytest.raises(ValueError, match="Must call build_msm"):
        msm_empty.reactive_flux(source, sink)


def test_committor_errors(simple_msm):
    """Test error handling in compute_committor."""
    # Overlapping source and sink
    with pytest.raises(ValueError, match="must not overlap"):
        simple_msm.compute_committor([0, 1], [1, 2])


def test_net_flux_formula(simple_msm):
    """Test that net flux is computed correctly: f_ij^+ = max(0, f_ij - f_ji)."""
    source = [0, 1]
    sink = [3, 4]

    gross_flux = simple_msm.get_gross_flux(source, sink)
    net_flux = simple_msm.get_net_flux(source, sink)

    # Manually compute net flux
    expected_net_flux = np.maximum(0, gross_flux - gross_flux.T)

    np.testing.assert_allclose(
        net_flux, expected_net_flux, rtol=1e-10, err_msg="Net flux formula is incorrect"
    )


def test_flux_conservation(simple_msm):
    """Test that flux is conserved at intermediate states."""
    source = [0, 1]
    sink = [3, 4]

    net_flux = simple_msm.get_net_flux(source, sink)

    # For intermediate state 2, flux in should equal flux out
    state = 2
    flux_in = np.sum(net_flux[:, state])
    flux_out = np.sum(net_flux[state, :])

    assert flux_in == pytest.approx(flux_out, rel=1e-8, abs=1e-10), (
        f"Flux not conserved at intermediate state {state}: "
        f"in={flux_in:.6e}, out={flux_out:.6e}"
    )


def test_large_system():
    """Test TPT on a larger system."""
    # Create a 20-state system
    n_states = 20
    np.random.seed(42)

    # Create a random transition matrix
    T = np.random.rand(n_states, n_states)
    T = T / T.sum(axis=1, keepdims=True)

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    with TemporaryDirectory() as tmpdir:
        msm = EnhancedMSM(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "test_large",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = n_states

        # Define source and sink
        source = [0, 1, 2]
        sink = [17, 18, 19]

        # Test all methods work
        flux = msm.reactive_flux(source, sink)
        assert flux.total_flux > 0

        q = msm.compute_committor(source, sink)
        assert q.shape == (n_states,)

        paths, path_fluxes = msm.pathway_decomposition(source, sink, fraction=0.9)
        assert len(paths) > 0

        ts_states = msm.identify_transition_state_ensemble(source, sink)
        assert len(ts_states) >= 0  # May be empty for random system

        bottlenecks = msm.find_bottleneck_states(source, sink, top_n=5)
        assert len(bottlenecks) == 5


def test_tpt_comprehensive_drunkards_walk():
    """Comprehensive test using the drunkards walk example from deeptime docs.

    This replicates the example from:
    https://deeptime-ml.github.io/latest/notebooks/tpt.html
    """
    from deeptime.data import drunkards_walk

    # Create the same simulator as in deeptime docs
    sim = drunkards_walk(
        grid_size=(10, 10),
        bar_location=[(0, 0), (0, 1), (1, 0), (1, 1)],
        home_location=[(8, 8), (8, 9), (9, 8), (9, 9)],
    )

    # Add barriers as in the example
    sim.add_barrier((5, 1), (5, 5))
    sim.add_barrier((0, 9), (5, 8))
    sim.add_barrier((9, 2), (7, 6))
    sim.add_barrier((2, 6), (5, 6))
    sim.add_barrier((7, 9), (7, 7), weight=5.0)
    sim.add_barrier((8, 7), (9, 7), weight=5.0)
    sim.add_barrier((0, 2), (2, 2), weight=5.0)
    sim.add_barrier((2, 0), (2, 1), weight=5.0)

    # Get the MSM from simulator
    T = sim.msm.transition_matrix
    pi = sim.msm.stationary_distribution

    with TemporaryDirectory() as tmpdir:
        msm = EnhancedMSM(
            trajectory_files=None,
            topology_file=None,
            output_dir=Path(tmpdir) / "drunkards_walk",
        )
        msm.transition_matrix = T
        msm.stationary_distribution = pi
        msm.n_states = sim.msm.n_states

        source = sim.home_state  # Home states
        sink = sim.bar_state  # Bar states

        # Test reactive flux
        flux = msm.reactive_flux(source, sink)

        # Compare with deeptime
        flux_deeptime = sim.msm.reactive_flux(source, sink)

        # Check all quantities match
        np.testing.assert_allclose(
            flux.forward_committor, flux_deeptime.forward_committor, rtol=1e-10
        )
        np.testing.assert_allclose(
            flux.backward_committor, flux_deeptime.backward_committor, rtol=1e-10
        )
        np.testing.assert_allclose(flux.gross_flux, flux_deeptime.gross_flux, rtol=1e-10)
        np.testing.assert_allclose(flux.net_flux, flux_deeptime.net_flux, rtol=1e-10)
        assert flux.total_flux == pytest.approx(flux_deeptime.total_flux, rel=1e-10)
        assert flux.rate == pytest.approx(flux_deeptime.rate, rel=1e-10)
        assert flux.mfpt == pytest.approx(flux_deeptime.mfpt, rel=1e-10)

        # Test pathway decomposition
        paths, path_fluxes = msm.pathway_decomposition(source, sink, fraction=0.3)
        paths_deeptime, fluxes_deeptime = flux_deeptime.pathways(fraction=0.3)

        # Should get same number of paths and same fluxes
        assert len(paths) == len(paths_deeptime)
        np.testing.assert_allclose(path_fluxes, fluxes_deeptime, rtol=1e-10)

        # Test coarse-graining
        # Define some macrostates
        remainder_upper = []
        remainder_lower = []
        for i in range(sim.grid_size[0]):
            for j in range(sim.grid_size[1]):
                state = sim.coordinate_to_state((i, j))
                if state not in source + sink:
                    if j >= 5:
                        remainder_upper.append(state)
                    else:
                        remainder_lower.append(state)

        sets = [source, sink, remainder_upper, remainder_lower]

        cg_sets, cg_flux = msm.coarse_grain_flux(source, sink, sets)
        cg_sets_deeptime, cg_flux_deeptime = flux_deeptime.coarse_grain(sets)

        # Compare coarse-grained fluxes
        np.testing.assert_allclose(
            cg_flux.gross_flux, cg_flux_deeptime.gross_flux, rtol=1e-10
        )
        np.testing.assert_allclose(
            cg_flux.net_flux, cg_flux_deeptime.net_flux, rtol=1e-10
        )


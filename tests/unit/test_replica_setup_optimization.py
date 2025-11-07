"""
Test suite for the Compute-once, Broadcast Pattern optimization in replica setup.

This module validates the thread-safe, immutable minimized state cache that
eliminates redundant minimization compute across replicas.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from pmarlo.replica_exchange.replica_setup import (
    MinimizedState,
    MinimizedStateCache,
    create_minimized_state_from_simulation,
)


class TestMinimizedState:
    """Test immutable MinimizedState dataclass."""

    def test_basic_creation(self):
        """Verify basic state creation with valid data."""
        positions = np.random.randn(100, 3)
        state = MinimizedState(
            positions=positions,
            potential_energy=-1234.5,
            replica_index=0,
            minimization_iterations=350,
        )

        assert state.potential_energy == -1234.5
        assert state.replica_index == 0
        assert state.minimization_iterations == 350
        assert state.positions.shape == (100, 3)

    def test_positions_immutability(self):
        """Ensure positions array is read-only after creation."""
        positions = np.random.randn(50, 3)
        state = MinimizedState(
            positions=positions,
            potential_energy=-500.0,
            replica_index=0,
            minimization_iterations=100,
        )

        # Positions should be read-only
        assert not state.positions.flags.writeable

        # Attempting to modify should raise
        with pytest.raises(ValueError, match="read-only"):
            state.positions[0, 0] = 999.0

    def test_invalid_positions_shape(self):
        """Reject positions with incorrect dimensions."""
        # 1D array should fail
        with pytest.raises(ValueError, match="must have shape"):
            MinimizedState(
                positions=np.random.randn(100),
                potential_energy=-100.0,
                replica_index=0,
                minimization_iterations=50,
            )

        # Wrong second dimension should fail
        with pytest.raises(ValueError, match="must have shape"):
            MinimizedState(
                positions=np.random.randn(100, 2),
                potential_energy=-100.0,
                replica_index=0,
                minimization_iterations=50,
            )

    def test_invalid_energy(self):
        """Reject non-finite energy values."""
        positions = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="must be finite"):
            MinimizedState(
                positions=positions,
                potential_energy=np.inf,
                replica_index=0,
                minimization_iterations=50,
            )

        with pytest.raises(ValueError, match="must be finite"):
            MinimizedState(
                positions=positions,
                potential_energy=np.nan,
                replica_index=0,
                minimization_iterations=50,
            )

    def test_memory_footprint(self):
        """Validate memory footprint estimation."""
        n_atoms = 1000
        positions = np.random.randn(n_atoms, 3)
        state = MinimizedState(
            positions=positions,
            potential_energy=-1000.0,
            replica_index=0,
            minimization_iterations=200,
        )

        # Expected: 1000 atoms * 3 coords * 8 bytes = 24000 bytes = ~0.023 MB
        footprint = state.memory_footprint_mb()
        assert 0.02 < footprint < 0.03

    def test_to_openmm_positions(self):
        """Verify conversion to OpenMM Quantity."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = MinimizedState(
            positions=positions,
            potential_energy=-100.0,
            replica_index=0,
            minimization_iterations=50,
        )

        openmm_pos = state.to_openmm_positions()
        # Should have unit attached
        assert hasattr(openmm_pos, 'unit')
        assert hasattr(openmm_pos, 'value_in_unit')


class TestMinimizedStateCache:
    """Test thread-safe cache implementation."""

    def test_cache_miss_computes(self):
        """First access should trigger computation."""
        cache = MinimizedStateCache()

        # Mock compute function
        mock_state = MinimizedState(
            positions=np.random.randn(10, 3),
            potential_energy=-500.0,
            replica_index=0,
            minimization_iterations=100,
        )

        compute_fn = Mock(return_value=mock_state)

        # First call should compute
        result = cache.get_or_compute(
            replica_index=0,
            compute_fn=compute_fn,
            some_arg="test",
        )

        assert result is mock_state
        compute_fn.assert_called_once_with(some_arg="test")

        stats = cache.get_statistics()
        assert stats['misses'] == 1
        assert stats['hits'] == 0

    def test_cache_hit_reuses(self):
        """Subsequent access should reuse cached state."""
        cache = MinimizedStateCache()

        mock_state = MinimizedState(
            positions=np.random.randn(10, 3),
            potential_energy=-500.0,
            replica_index=0,
            minimization_iterations=100,
        )

        compute_fn = Mock(return_value=mock_state)

        # First call - cache miss
        result1 = cache.get_or_compute(replica_index=0, compute_fn=compute_fn)

        # Second call - cache hit (should NOT call compute_fn again)
        result2 = cache.get_or_compute(replica_index=1, compute_fn=compute_fn)

        assert result1 is result2
        assert result1 is mock_state
        compute_fn.assert_called_once()  # Only called once

        stats = cache.get_statistics()
        assert stats['misses'] == 1
        assert stats['hits'] == 1

    def test_invalidate_clears_cache(self):
        """Invalidation should clear cached state."""
        cache = MinimizedStateCache()

        mock_state = MinimizedState(
            positions=np.random.randn(10, 3),
            potential_energy=-500.0,
            replica_index=0,
            minimization_iterations=100,
        )

        compute_fn = Mock(return_value=mock_state)

        # Populate cache
        cache.get_or_compute(replica_index=0, compute_fn=compute_fn)
        assert cache.is_populated

        # Invalidate
        cache.invalidate()
        assert not cache.is_populated

        # Next call should compute again
        cache.get_or_compute(replica_index=1, compute_fn=compute_fn)
        assert compute_fn.call_count == 2

    def test_thread_safety_mock(self):
        """Verify lock is used (basic check)."""
        cache = MinimizedStateCache()

        # Cache should have a lock
        assert hasattr(cache, '_lock')
        assert cache._lock is not None

    def test_invalid_compute_fn_return(self):
        """Reject compute_fn that doesn't return MinimizedState."""
        cache = MinimizedStateCache()

        # Compute function returns wrong type
        bad_compute = Mock(return_value="not a MinimizedState")

        with pytest.raises(TypeError, match="must return MinimizedState"):
            cache.get_or_compute(replica_index=0, compute_fn=bad_compute)


class TestCreateMinimizedStateFromSimulation:
    """Test extraction of state from OpenMM simulation."""

    def test_extraction_with_mock_simulation(self):
        """Verify state extraction from mocked simulation."""
        # Create mock simulation with context
        mock_simulation = MagicMock()

        # Mock positions
        mock_pos_list = [
            MagicMock(x=1.0, y=2.0, z=3.0),
            MagicMock(x=4.0, y=5.0, z=6.0),
        ]

        # Mock state
        mock_state = MagicMock()
        mock_state.getPositions.return_value = MagicMock(
            value_in_unit=Mock(return_value=mock_pos_list)
        )
        mock_state.getPotentialEnergy.return_value = MagicMock(
            value_in_unit=Mock(return_value=-1234.5)
        )

        mock_simulation.context.getState.return_value = mock_state
        mock_simulation.context.getPlatform.return_value.getName.return_value = "CPU"

        # Extract state
        state = create_minimized_state_from_simulation(
            simulation=mock_simulation,
            replica_index=2,
            minimization_iterations=300,
            additional_metadata={'test_key': 'test_value'},
        )

        assert state.replica_index == 2
        assert state.minimization_iterations == 300
        assert state.potential_energy == -1234.5
        assert state.positions.shape == (2, 3)
        assert state.metadata['test_key'] == 'test_value'
        assert state.metadata['platform'] == 'CPU'


class TestIntegrationScenario:
    """Integration tests simulating real replica setup workflow."""

    def test_multi_replica_cache_efficiency(self):
        """Simulate setup of 8 replicas with cache."""
        cache = MinimizedStateCache()
        n_replicas = 8

        # Expensive compute function (tracks call count)
        call_count = [0]

        def expensive_minimization():
            call_count[0] += 1
            return MinimizedState(
                positions=np.random.randn(100, 3),
                potential_energy=-5000.0,
                replica_index=0,
                minimization_iterations=350,
            )

        # Simulate replica setup
        states = []
        for i in range(n_replicas):
            state = cache.get_or_compute(
                replica_index=i,
                compute_fn=expensive_minimization,
            )
            states.append(state)

        # Should only compute once
        assert call_count[0] == 1

        # All states should be identical (same reference)
        for state in states:
            assert state is states[0]

        # Cache efficiency should be high
        stats = cache.get_statistics()
        assert stats['misses'] == 1
        assert stats['hits'] == n_replicas - 1

        efficiency = 100 * stats['hits'] / (stats['hits'] + stats['misses'])
        assert efficiency > 85.0  # 7/8 = 87.5%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


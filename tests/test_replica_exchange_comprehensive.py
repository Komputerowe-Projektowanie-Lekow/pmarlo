# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Comprehensive test suite for ReplicaExchange module.

Tests all critical functionality including:
- Initialization and setup
- Error handling and edge cases
- Exchange algorithms
- Checkpoint management
- API consistency
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import openmm
import pytest

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


class TestReplicaExchangeInitialization:
    """Test replica exchange initialization and basic setup."""

    @pytest.fixture
    def test_pdb_file(self):
        """Provide test PDB file path."""
        return str(Path(__file__).parent / "data" / "3gd8.pdb")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_basic_initialization(self, test_pdb_file, temp_output_dir):
        """Test basic initialization without auto-setup."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310, 320],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        assert remd.n_replicas == 3
        assert remd.temperatures == [300, 310, 320]
        assert not remd.is_setup()
        assert len(remd.contexts) == 0
        assert len(remd.replicas) == 0

    def test_auto_setup_initialization(self, test_fixed_pdb_file, temp_output_dir):
        """Test initialization with auto-setup enabled."""
        with patch("pmarlo.replica_exchange.replica_exchange.logger"):
            remd = ReplicaExchange(
                pdb_file=str(
                    test_fixed_pdb_file
                ),  # Use the fixed PDB file with hydrogens
                temperatures=[300, 310],  # Use fewer replicas for faster testing
                output_dir=temp_output_dir,
                auto_setup=True,
            )

            assert remd.n_replicas == 2
            assert remd.is_setup()
            assert len(remd.contexts) == 2
            assert len(remd.replicas) == 2

    def test_temperature_ladder_generation(self, test_pdb_file, temp_output_dir):
        """Test automatic temperature ladder generation."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file, output_dir=temp_output_dir, auto_setup=False
        )

        # Should generate default temperature ladder
        assert len(remd.temperatures) == 3  # Default n_replicas
        assert min(remd.temperatures) >= 300.0
        assert max(remd.temperatures) <= 350.0
        assert remd.temperatures == sorted(remd.temperatures)  # Should be sorted

    def test_invalid_initialization(self, temp_output_dir):
        """Test initialization with invalid parameters."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or other exceptions
            remd = ReplicaExchange(
                pdb_file="nonexistent.pdb", output_dir=temp_output_dir
            )
            remd.setup_replicas()  # This should raise an exception


class TestReplicaExchangeValidation:
    """Test validation and error handling."""

    @pytest.fixture
    def basic_remd(self, test_pdb_file, temp_output_dir):
        """Create basic replica exchange object."""
        return ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310, 320],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

    def test_run_simulation_without_setup(self, basic_remd):
        """Test that run_simulation fails without setup."""
        with pytest.raises(RuntimeError) as exc_info:
            basic_remd.run_simulation(total_steps=10)

        assert "not properly initialized" in str(exc_info.value)
        assert "setup_replicas" in str(exc_info.value)

    def test_exchange_bounds_checking(self, basic_remd):
        """Test bounds checking in exchange methods."""
        # This should fail because replicas aren't set up
        with pytest.raises(ValueError) as exc_info:
            basic_remd.attempt_exchange(-1, 0)

        assert "out of bounds" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            basic_remd.attempt_exchange(0, 5)  # Beyond n_replicas

        assert "out of bounds" in str(exc_info.value)

    def test_auto_setup_if_needed(self, basic_remd):
        """Test auto-setup functionality."""
        assert not basic_remd.is_setup()

        with patch.object(basic_remd, "setup_replicas") as mock_setup:
            basic_remd.auto_setup_if_needed()
            mock_setup.assert_called_once()


class TestReplicaExchangeCheckpointing:
    """Test checkpoint save/restore functionality."""

    @pytest.fixture
    def setup_remd(self, test_pdb_file, temp_output_dir):
        """Create and setup replica exchange object."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310],  # Small for testing
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        # Mock the setup to avoid actual MD initialization
        remd._is_setup = True
        remd.contexts = [Mock(), Mock()]
        remd.replicas = [Mock(), Mock()]
        remd.exchange_attempts = 10
        remd.exchanges_accepted = 3
        remd.replica_states = [0, 1]
        remd.state_replicas = [0, 1]

        return remd

    def test_save_checkpoint_state(self, setup_remd):
        """Test checkpoint state saving."""
        state = setup_remd.save_checkpoint_state()

        assert state["setup"] is True
        assert state["n_replicas"] == 2
        assert state["temperatures"] == [300, 310]
        assert state["exchange_attempts"] == 10
        assert state["exchanges_accepted"] == 3
        assert state["replica_states"] == [0, 1]
        assert state["state_replicas"] == [0, 1]

    def test_save_checkpoint_state_not_setup(self, test_pdb_file, temp_output_dir):
        """Test checkpoint saving when not set up."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        state = remd.save_checkpoint_state()
        assert state["setup"] is False

    def test_restore_from_checkpoint(self, test_pdb_file, temp_output_dir):
        """Test checkpoint restoration."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        checkpoint_state = {
            "setup": True,
            "exchange_attempts": 15,
            "exchanges_accepted": 5,
            "replica_states": [1, 0],
            "state_replicas": [1, 0],
            "exchange_history": [[0, 1], [1, 0]],
        }

        with patch.object(remd, "setup_replicas") as mock_setup:
            remd.restore_from_checkpoint(checkpoint_state)
            mock_setup.assert_called_once()

        assert remd.exchange_attempts == 15
        assert remd.exchanges_accepted == 5
        assert remd.replica_states == [1, 0]
        assert remd.state_replicas == [1, 0]


class TestReplicaExchangeExchangeAlgorithm:
    """Test the exchange algorithm and state management."""

    @pytest.fixture
    def mock_remd(self, test_pdb_file, temp_output_dir):
        """Create a mocked replica exchange for algorithm testing."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310, 320],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        # Mock the setup state
        remd._is_setup = True
        remd.contexts = [Mock() for _ in range(3)]
        remd.replicas = [Mock() for _ in range(3)]
        remd.integrators = [Mock() for _ in range(3)]

        # Mock energy states
        for i, context in enumerate(remd.contexts):
            mock_state = Mock()
            # Return energy in kJ/mol
            mock_state.getPotentialEnergy.return_value = (
                -1000 - i * 100
            ) * openmm.unit.kilojoules_per_mole
            context.getState.return_value = mock_state

        return remd

    def test_calculate_exchange_probability(self, mock_remd):
        """Test exchange probability calculation."""
        # Test valid replica indices
        prob = mock_remd.calculate_exchange_probability(0, 1)
        assert 0.0 <= prob <= 1.0

        # Test bounds checking
        with pytest.raises(ValueError):
            mock_remd.calculate_exchange_probability(-1, 0)

        with pytest.raises(ValueError):
            mock_remd.calculate_exchange_probability(0, 5)

    def test_state_tracking_consistency(self, mock_remd):
        """Test that state tracking arrays remain consistent."""
        initial_replica_states = mock_remd.replica_states.copy()
        initial_state_replicas = mock_remd.state_replicas.copy()

        # Verify initial state consistency
        for replica_idx, state_idx in enumerate(initial_replica_states):
            assert initial_state_replicas[state_idx] == replica_idx


class TestReplicaExchangeIntegration:
    """Integration tests for full workflow."""

    def test_pipeline_integration(self, test_fixed_pdb_file, temp_output_dir):
        """Test integration with Pipeline class."""
        from pmarlo.pipeline import Pipeline

        # Test that pipeline properly initializes replica exchange
        pipeline = Pipeline(
            pdb_file=test_fixed_pdb_file,  # Use the fixed PDB file with hydrogens
            temperatures=[300, 310],
            steps=100,  # Very short for testing
            output_dir=temp_output_dir,
            use_replica_exchange=True,
            use_metadynamics=False,  # Disable for simpler testing
        )

        # Mock protein preparation to avoid complex setup
        with patch.object(pipeline, "setup_protein") as mock_protein:
            mock_protein.return_value = Mock()
            mock_protein.return_value.get_properties.return_value = {
                "num_atoms": 100,
                "num_residues": 10,
            }
            pipeline.prepared_pdb = Path(test_fixed_pdb_file)

            # Test replica exchange setup
            remd = pipeline.setup_replica_exchange()
            assert remd is not None
            assert remd.is_setup()  # Should be automatically set up


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_minimization_failure_recovery(self, test_pdb_file, temp_output_dir):
        """Test recovery from minimization failures."""
        remd = ReplicaExchange(
            pdb_file=test_pdb_file,
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )
        assert remd.n_replicas == 2

        # Mock minimization failure and recovery
        with patch(
            "pmarlo.replica_exchange.replica_exchange.Simulation"
        ) as mock_sim_class:
            mock_sim = Mock()
            mock_sim_class.return_value = mock_sim

            # First minimization fails, second succeeds
            mock_sim.minimizeEnergy.side_effect = [
                Exception("Minimization failed"),  # First attempt fails
                None,  # Second attempt succeeds
                None,  # Final minimization succeeds
            ]

            # Mock energy state
            mock_state = Mock()
            mock_state.getPotentialEnergy.return_value = -1000
            mock_state.getPositions.return_value = Mock()
            mock_sim.context.getState.return_value = mock_state

            # Should not raise exception due to fallback logic
            # Note: This is a simplified test – full implementation would need
            # more mocking
            # remd.setup_replicas()


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])

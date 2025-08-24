from unittest.mock import patch

import pytest

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


class TestReplicaExchangeInitialization:
    """Test replica exchange initialization and basic setup."""

    def test_basic_initialization(self, test_pdb_file, temp_output_dir):
        """Test basic initialization without auto-setup."""
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
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
                pdb_file=str(test_fixed_pdb_file),
                temperatures=[300, 310],
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
            pdb_file=str(test_pdb_file),
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        assert len(remd.temperatures) == 3
        assert min(remd.temperatures) >= 300.0
        assert max(remd.temperatures) <= 350.0
        assert remd.temperatures == sorted(remd.temperatures)

    def test_invalid_initialization(self, temp_output_dir):
        """Test initialization with invalid parameters."""
        with pytest.raises(Exception):
            remd = ReplicaExchange(
                pdb_file="nonexistent.pdb",
                output_dir=temp_output_dir,
            )
            remd.setup_replicas()


class TestReplicaExchangeValidation:
    """Test validation and error handling."""

    @pytest.fixture
    def basic_remd(self, test_pdb_file, temp_output_dir):
        return ReplicaExchange(
            pdb_file=str(test_pdb_file),
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
        with pytest.raises(ValueError) as exc_info:
            basic_remd.attempt_exchange(-1, 0)
        assert "out of bounds" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            basic_remd.attempt_exchange(0, 5)
        assert "out of bounds" in str(exc_info.value)

    def test_auto_setup_if_needed(self, basic_remd):
        """Test auto-setup functionality."""
        assert not basic_remd.is_setup()
        with patch.object(basic_remd, "setup_replicas") as mock_setup:
            basic_remd.auto_setup_if_needed()
            mock_setup.assert_called_once()

from unittest.mock import Mock

import openmm
import pytest

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


class TestReplicaExchangeExchangeAlgorithm:
    """Test the exchange algorithm and state management."""

    @pytest.fixture
    def mock_remd(self, test_pdb_file, temp_output_dir):
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310, 320],
            output_dir=temp_output_dir,
            auto_setup=False,
        )
        remd._is_setup = True
        remd.contexts = [Mock() for _ in range(3)]
        remd.replicas = [Mock() for _ in range(3)]
        remd.integrators = [Mock() for _ in range(3)]
        for i, context in enumerate(remd.contexts):
            mock_state = Mock()
            mock_state.getPotentialEnergy.return_value = (
                -1000 - i * 100
            ) * openmm.unit.kilojoules_per_mole
            context.getState.return_value = mock_state
        return remd

    def test_calculate_exchange_probability(self, mock_remd):
        prob = mock_remd.calculate_exchange_probability(0, 1)
        assert 0.0 <= prob <= 1.0
        with pytest.raises(ValueError):
            mock_remd.calculate_exchange_probability(-1, 0)
        with pytest.raises(ValueError):
            mock_remd.calculate_exchange_probability(0, 5)

    def test_state_tracking_consistency(self, mock_remd):
        initial_replica_states = mock_remd.replica_states.copy()
        initial_state_replicas = mock_remd.state_replicas.copy()
        for replica_idx, state_idx in enumerate(initial_replica_states):
            assert initial_state_replicas[state_idx] == replica_idx
